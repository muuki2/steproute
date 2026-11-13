"""Privacy: text capture policy, sensitive detection and export scrubbing. Standard library only.

Everything here is best effort pattern matching; see ``docs/PRIVACY.md`` for the guarantees and
the limits. Rule names are the only thing ever reported about a match, never the matched text.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from steproute.core.schema import Action, Decision, Outcome, StepState

__all__ = [
    "DEFAULT_SECRET_PATTERNS",
    "DEFAULT_SENSITIVE_GLOBS",
    "FEATURE_TEXT_HASH",
    "FEATURE_TEXT_LEN",
    "REDACTED",
    "Detection",
    "PrivacyConfig",
    "Rule",
    "ScrubOptions",
    "detect_sensitive",
    "keyed_hash",
    "mark_sensitive",
    "redact_secrets",
    "redact_state",
    "scrub_decision",
    "scrub_outcome",
]

#: ``StepState.features`` key holding ``len(task_text)`` when the text was not captured.
FEATURE_TEXT_LEN = "task_text_len"
#: ``StepState.features`` key holding the salted hash of the text (only if ``hash_text``).
FEATURE_TEXT_HASH = "task_text_hash"
#: Replacement for any secret-pattern match in a scrubbed export.
REDACTED = "[REDACTED]"

#: Token prefixes match unless glued to a preceding letter or digit (``_`` and ``-`` are fine).
_B = r"(?<![A-Za-z0-9])"


@dataclass(frozen=True)
class Rule:
    """A named glob or regular expression."""

    name: str
    pattern: str


DEFAULT_SENSITIVE_GLOBS: tuple[Rule, ...] = (
    Rule("dotenv_files", "**/.env"),
    Rule("dotenv_variants", "**/.env.*"),
    Rule("dotenv_suffix", "**/*.env"),
    Rule("pem_files", "**/*.pem"),
    Rule("key_files", "**/*.key"),
    Rule("keystore_p12", "**/*.p12"),
    Rule("keystore_pfx", "**/*.pfx"),
    Rule("ssh_dir", "**/.ssh/**"),
    Rule("ssh_key_rsa", "**/id_rsa*"),
    Rule("ssh_key_ed25519", "**/id_ed25519*"),
    Rule("ssh_key_ecdsa", "**/id_ecdsa*"),
    Rule("ssh_key_dsa", "**/id_dsa*"),
    Rule("aws_dir", "**/.aws/**"),
    Rule("gcloud_dir", "**/.config/gcloud/**"),
    Rule("azure_dir", "**/.azure/**"),
    Rule("kube_config", "**/.kube/config"),
    Rule("netrc", "**/.netrc"),
    Rule("secrets_dir", "**/secrets/**"),
)

DEFAULT_SECRET_PATTERNS: tuple[Rule, ...] = (
    Rule("aws_access_key_id", _B + r"(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    Rule("github_token", _B + r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    Rule("api_key_sk", _B + r"sk-[A-Za-z0-9_-]{20,}"),
    Rule("slack_token", _B + r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    Rule("google_api_key", _B + r"AIza[0-9A-Za-z_-]{35}"),
    Rule("private_key_block", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"),
    Rule("jwt", _B + r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    Rule("bearer_token", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    Rule(
        "secret_assignment",
        r"(?i)(?<![a-z])(?:password|passwd|pwd|secret(?:[_-]?access)?(?:[_-]?key)?|api[_-]?key"
        r"|access[_-]?token|auth[_-]?token)"
        r"""\b["']?\s*[:=]\s*["']?[^\s"',;]+""",
    ),
)

#: ``callback(task_text, paths) -> bool``; True means the step is sensitive.
SensitiveCallback = Callable[[str | None, Sequence[str]], bool]


@dataclass(frozen=True)
class PrivacyConfig:
    """What may be stored and what counts as sensitive.

    ``capture_text`` is off by default: ``task_text`` is then never stored. ``hash_text`` stores
    an HMAC-SHA256 of the text (keyed with ``salt``) next to its length; it needs a salt.
    """

    capture_text: bool = False
    hash_text: bool = False
    salt: str = field(default="", repr=False)
    sensitive_globs: tuple[Rule, ...] = DEFAULT_SENSITIVE_GLOBS
    secret_patterns: tuple[Rule, ...] = DEFAULT_SECRET_PATTERNS
    callback: SensitiveCallback | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.hash_text and not self.salt:
            raise ValueError("hash_text requires a non-empty salt")
        for r in self.secret_patterns:
            re.compile(r.pattern)


# --- matching -------------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@lru_cache(maxsize=256)
def _glob_regex(glob: str) -> re.Pattern[str]:
    out: list[str] = []
    i = 0
    g = glob.replace("\\", "/").casefold()
    while i < len(g):
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        elif g[i] == "?":
            out.append("[^/]")
            i += 1
        elif g[i] == "[":
            j = g.find("]", i + 2)
            if j == -1:
                out.append(re.escape(g[i]))
                i += 1
            else:
                out.append(g[i : j + 1])
                i = j + 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _norm_path(path: str) -> str:
    p = path.replace("\\", "/").casefold()
    while p.startswith("./"):
        p = p[2:]
    return p.lstrip("/")


@dataclass(frozen=True)
class Detection:
    """Result of sensitive detection: the verdict and the names of the rules that fired."""

    sensitive: bool
    rules: tuple[str, ...] = ()


def detect_sensitive(
    text: str | None, paths: Sequence[str], config: PrivacyConfig | None = None
) -> Detection:
    """Whether ``text`` or any of ``paths`` looks sensitive, and which rules said so.

    Only rule names are returned. A raising user callback fails closed (sensitive, rule
    ``callback_error``).
    """
    cfg = config or PrivacyConfig()
    fired: list[str] = []
    normed = [_norm_path(p) for p in paths]
    for rule in cfg.sensitive_globs:
        rx = _glob_regex(rule.pattern)
        if any(rx.match(p) for p in normed):
            fired.append(rule.name)
    if text:
        for rule in cfg.secret_patterns:
            if _regex(rule.pattern).search(text):
                fired.append(rule.name)
    if cfg.callback is not None:
        try:
            if cfg.callback(text, tuple(paths)):
                fired.append("user_callback")
        except Exception:
            fired.append("callback_error")
    return Detection(bool(fired), tuple(fired))


def mark_sensitive(
    state: StepState,
    paths: Sequence[str] = (),
    config: PrivacyConfig | None = None,
    *,
    text: str | None = None,
) -> StepState:
    """A copy of ``state`` with ``sensitive=True`` if detection fires. Never clears the flag.

    ``text`` defaults to ``state.task_text``.
    """
    det = detect_sensitive(state.task_text if text is None else text, paths, config)
    return dataclasses.replace(state, sensitive=state.sensitive or det.sensitive)


# --- write path -----------------------------------------------------------------------------


def keyed_hash(salt: str, value: str, *, length: int = 16) -> str:
    """HMAC-SHA256 of ``value`` keyed with ``salt``, as the first ``length`` hex characters."""
    mac = hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return mac.hexdigest()[:length]


def redact_state(state: StepState, config: PrivacyConfig) -> StepState:
    """The copy of ``state`` that may be written to disk. The argument is never mutated.

    Sensitive detection runs on the text first (it can only set the flag). With
    ``capture_text`` off the text is dropped and only its length (and, if configured, its
    salted hash) is recorded in ``features``.
    """
    out = mark_sensitive(state, (), config)
    features = dict(out.features)
    text = out.task_text
    if config.capture_text or text is None:
        return dataclasses.replace(out, features=features)
    features[FEATURE_TEXT_LEN] = len(text)
    if config.hash_text:
        features[FEATURE_TEXT_HASH] = keyed_hash(config.salt, text, length=32)
    return dataclasses.replace(out, task_text=None, features=features)


# --- scrubbing ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScrubOptions:
    """Parameters of a scrubbed export."""

    salt: str = field(repr=False)
    bucket_seconds: float = 3600.0
    secret_patterns: tuple[Rule, ...] = DEFAULT_SECRET_PATTERNS

    def __post_init__(self) -> None:
        if not self.salt:
            raise ValueError("scrubbing requires a non-empty salt")
        if not self.bucket_seconds > 0:
            raise ValueError("bucket_seconds must be positive")


def redact_secrets(value: str, patterns: Sequence[Rule] = DEFAULT_SECRET_PATTERNS) -> str:
    """Replace every secret-pattern match in ``value`` with ``REDACTED``."""
    for rule in patterns:
        value = _regex(rule.pattern).sub(REDACTED, value)
    return value


def _bucket(ts: float, size: float) -> float:
    if not math.isfinite(ts):
        return 0.0
    return math.floor(ts / size) * size


def _scrub_free(d: dict[str, Any], opts: ScrubOptions) -> dict[str, Any]:
    """Keep only booleans, numbers and null; strings, lists and objects can carry text."""
    keep: dict[str, Any] = {}
    for k, v in d.items():
        if v is None or isinstance(v, bool | int | float):
            keep[redact_secrets(k, opts.secret_patterns)] = v
    return keep


def _scrub_action(a: Action, opts: ScrubOptions) -> Action:
    return Action(
        model=redact_secrets(a.model, opts.secret_patterns),
        effort=a.effort,
        escalate_to=None if a.escalate_to is None else _scrub_action(a.escalate_to, opts),
    )


def _scrub_decision_id(decision_id: str, opts: ScrubOptions) -> str:
    # Hashed too: a collector may have built the id from the session id.
    return keyed_hash(opts.salt, "decision:" + decision_id, length=32)


def scrub_decision(d: Decision, opts: ScrubOptions) -> Decision:
    """A scrubbed copy of ``d``; see ``docs/PRIVACY.md`` for exactly what changes."""

    def red(s: str) -> str:
        return redact_secrets(s, opts.secret_patterns)

    s = d.state
    state = dataclasses.replace(
        s,
        harness=red(s.harness),
        session_id=keyed_hash(opts.salt, "session:" + s.session_id),
        task_id=None if s.task_id is None else keyed_hash(opts.salt, "task:" + s.task_id),
        kind=red(s.kind),
        current_model=None if s.current_model is None else red(s.current_model),
        failed_attempts=[_scrub_action(a, opts) for a in s.failed_attempts],
        tools=[red(t) for t in s.tools],
        task_text=None,
        features=_scrub_free(s.features, opts),
    )
    return dataclasses.replace(
        d,
        decision_id=_scrub_decision_id(d.decision_id, opts),
        ts=_bucket(d.ts, opts.bucket_seconds),
        state=state,
        candidates=[_scrub_action(a, opts) for a in d.candidates],
        action=_scrub_action(d.action, opts),
        policy=red(d.policy),
    )


def scrub_outcome(o: Outcome, opts: ScrubOptions) -> Outcome:
    """A scrubbed copy of ``o``."""
    return dataclasses.replace(
        o,
        decision_id=_scrub_decision_id(o.decision_id, opts),
        ts=_bucket(o.ts, opts.bucket_seconds),
        signals=_scrub_free(o.signals, opts),
        final_model=None
        if o.final_model is None
        else redact_secrets(o.final_model, opts.secret_patterns),
    )
