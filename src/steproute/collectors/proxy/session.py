"""Per-session state of the proxy: step index, cache owner, cached prefix, sensitivity.

Standard library only. See ``docs/PROXY.md`` ("Session state") for the rules. The request is
split into **units**: unit 0 is the tool definitions, then one unit per message. A unit is
identified by a hash of its canonical JSON. Comparing the units of a new request with those
of the last executed prompt tells the proxy how much of the new prompt the provider can have
cached, and comparing with the units already scanned tells it which content still needs the
sensitive-data scan.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from steproute.core.privacy import PrivacyConfig, detect_sensitive, keyed_hash

__all__ = [
    "PER_UNIT_TOKENS",
    "Prompt",
    "Session",
    "SessionTable",
    "scan_units",
    "session_id_from_messages",
    "split_prompt",
]

#: Token overhead per message (role markers, separators). An estimate.
PER_UNIT_TOKENS = 4
#: Characters of a tokenised image part are not text; count each image as this many tokens.
IMAGE_TOKENS = 1000
_MAX_PATHS = 256
_PATH_KEYS = frozenset(
    {
        "path",
        "paths",
        "file",
        "files",
        "file_path",
        "filepath",
        "filename",
        "dir",
        "directory",
        "cwd",
        "target",
        "source",
        "destination",
        "notebook_path",
    }
)
# Path-looking tokens in free text: something with a slash, or a dotfile name. Each segment
# after the first must follow a "/": the earlier form ``(?:/?[\w.@+-]+)*/...`` let a long
# identifier split into segments in exponentially many ways (a 24-character word took
# seconds, a 37-character one hung the proxy), and matched exactly the same tokens.
_PATH_TOKEN = re.compile(r"(?:~|\.{0,2})?(?:/?[\w.@+-]+)?(?:/[\w.@+-]+)+|(?<![\w/])\.[\w.-]+")


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:32]


@dataclass(frozen=True)
class Prompt:
    """A request split into units. ``chars[i]`` is the estimation weight of unit ``i``."""

    units: tuple[str, ...]
    chars: tuple[int, ...]
    images: tuple[int, ...]

    def est_tokens(self, start: int = 0, chars_per_token: float = 4.0) -> int:
        """Estimated tokens of units ``start:``."""
        if start >= len(self.units):
            return 0
        text = sum(self.chars[start:])
        return (
            math.ceil(text / chars_per_token)
            + PER_UNIT_TOKENS * (len(self.units) - start)
            + IMAGE_TOKENS * sum(self.images[start:])
        )

    @property
    def total_chars(self) -> int:
        return sum(self.chars)


def _content_weight(message: Mapping[str, Any]) -> tuple[int, int]:
    """(characters, image parts) of a message, for the token estimate."""
    images = 0
    stripped = dict(message)
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and part.get("type") in (
                "image_url",
                "image",
                "input_image",
            ):
                images += 1
                continue
            parts.append(part)
        stripped["content"] = parts
    return len(_canon(stripped)), images


def split_prompt(messages: Sequence[Any], tools: Sequence[Any] | None) -> Prompt:
    units = [_hash("tools:" + _canon(list(tools or [])))]
    chars = [len(_canon(list(tools or []))) if tools else 0]
    images = [0]
    for m in messages:
        units.append(_hash("msg:" + _canon(m)))
        if isinstance(m, Mapping):
            c, i = _content_weight(m)
        else:
            c, i = len(_canon(m)), 0
        chars.append(c)
        images.append(i)
    return Prompt(tuple(units), tuple(chars), tuple(images))


def common_prefix(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def session_id_from_messages(messages: Sequence[Any], salt: str) -> str:
    """Fallback session id: a keyed hash of the messages up to and including the first user
    message (system prompt plus the opening request), which stay fixed through a session."""
    head: list[Any] = []
    for m in messages:
        head.append(m)
        if isinstance(m, Mapping) and m.get("role") == "user":
            break
    return "h:" + keyed_hash(salt, _canon(head), length=24)


# --- sensitive scan -------------------------------------------------------------------------


def _texts_and_paths(value: Any, texts: list[str], paths: list[str], key: str | None) -> None:
    if isinstance(value, str):
        texts.append(value)
        if key is not None and key.lower() in _PATH_KEYS:
            paths.append(value)
        return
    if isinstance(value, Mapping):
        for k, v in value.items():
            _texts_and_paths(v, texts, paths, str(k))
        return
    if isinstance(value, list | tuple):
        for v in value:
            _texts_and_paths(v, texts, paths, key)


def _message_parts(message: Any) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    paths: list[str] = []
    if not isinstance(message, Mapping):
        _texts_and_paths(message, texts, paths, None)
        return texts, paths
    for k, v in message.items():
        if k == "tool_calls" and isinstance(v, list):
            for call in v:
                fn = call.get("function") if isinstance(call, Mapping) else None
                args = fn.get("arguments") if isinstance(fn, Mapping) else None
                if isinstance(args, str):
                    texts.append(args)
                    try:
                        parsed = json.loads(args)
                    except ValueError:
                        parsed = None
                    if parsed is not None:
                        _texts_and_paths(parsed, [], paths, None)
                if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
                    texts.append(fn["name"])
        elif k in ("content", "name", "refusal", "reasoning_content"):
            _texts_and_paths(v, texts, paths, None)
    return texts, paths


def scan_units(messages: Sequence[Any], start: int, config: PrivacyConfig) -> tuple[str, ...]:
    """Run sensitive detection over ``messages[start:]``; return the rule names that fired.

    Text: every string in content, names, tool-call arguments and refusals. Paths: values of
    path-like keys in tool-call arguments, plus path-looking tokens in the text (capped at 256
    per scan). Only rule names are returned, never matched text.
    """
    texts: list[str] = []
    paths: list[str] = []
    for m in messages[start:]:
        t, p = _message_parts(m)
        texts.extend(t)
        paths.extend(p)
    text = "\n".join(texts)
    for match in _PATH_TOKEN.finditer(text):
        if len(paths) >= _MAX_PATHS:
            break
        paths.append(match.group(0))
    if not text and not paths:
        return ()
    return detect_sensitive(text or None, paths[:_MAX_PATHS], config).rules


# --- sessions -------------------------------------------------------------------------------


@dataclass
class Session:
    """What the proxy knows about one session. Mutated only under the session lock."""

    session_id: str
    next_step: int = 0
    #: Cache owner: the (catalog model, effort) of the last attempt that got a response.
    owner_model: str | None = None
    owner_effort: str | None = None
    #: Tokens the provider holds for the owner (the last executed prompt, 0 below its minimum).
    cached_tokens: int = 0
    #: Actual prompt tokens of the last executed attempt (from the provider's usage).
    last_prompt_tokens: int = 0
    #: Units and per-unit weights of the last executed prompt.
    cache_units: tuple[str, ...] = ()
    cache_chars: tuple[int, ...] = ()
    cache_ts: float = 0.0
    cache_ttl_s: float = 300.0
    #: Units already scanned for sensitive content, and the sticky verdict.
    scanned_units: tuple[str, ...] = ()
    sensitive: bool = False
    sensitive_rules: tuple[str, ...] = ()
    chars_per_token: float = 4.0
    last_used: float = 0.0
    #: Decision id and units of the previous request, for retry detection.
    last_decision_id: str | None = None
    last_units: tuple[str, ...] = ()
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)


@dataclass(frozen=True)
class Estimate:
    """What the proxy believes before the call."""

    prompt_tokens: int
    cached_prefix_tokens: int
    current_model: str | None
    current_effort: str | None
    retry_of: str | None


class SessionTable:
    """Bounded, expiring map of sessions (LRU, at most ``max_sessions``; idle ones expire)."""

    def __init__(
        self,
        *,
        max_sessions: int = 1024,
        idle_s: float = 3600.0,
        chars_per_token: float = 4.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.max_sessions = max_sessions
        self.idle_s = idle_s
        self.chars_per_token = chars_per_token
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self.evicted = 0
        self.expired = 0

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _get_or_create(self, session_id: str) -> Session:
        now = self._clock()
        with self._lock:
            while self._sessions:  # expire from the least recently used end
                oldest = next(iter(self._sessions.values()))
                if now - oldest.last_used <= self.idle_s:
                    break
                self._sessions.popitem(last=False)
                self.expired += 1
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = Session(session_id, chars_per_token=self.chars_per_token, last_used=now)
                self._sessions[session_id] = sess
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
                    self.evicted += 1
            else:
                self._sessions.move_to_end(session_id)
            sess.last_used = now
            return sess

    @contextmanager
    def locked(self, session_id: str) -> Iterator[Session]:
        sess = self._get_or_create(session_id)
        with sess.lock:
            yield sess

    # --- the two transitions ---

    def begin(
        self,
        session_id: str,
        prompt: Prompt,
        messages: Sequence[Any],
        privacy: PrivacyConfig,
    ) -> tuple[int, Estimate, bool, tuple[str, ...]]:
        """Start a step: assign its index, estimate the prompt and cache, scan new content.

        Returns ``(step_idx, estimate, sensitive, rules)``. The scan covers only the messages
        after the longest prefix already scanned; a positive verdict is sticky for the session.
        """
        now = self._clock()
        with self.locked(session_id) as s:
            step = s.next_step
            s.next_step += 1
            # Sensitive scan of new content only. Unit i >= 1 is messages[i - 1]; unit 0 (the
            # tool definitions) is not content and is not scanned, but a change there resets
            # the prefix, so everything is rescanned.
            k = common_prefix(prompt.units, s.scanned_units)
            if k < len(prompt.units):
                rules = scan_units(messages, max(0, k - 1), privacy)
                if rules and not s.sensitive:
                    s.sensitive = True
                    s.sensitive_rules = rules
                s.scanned_units = prompt.units
            # Prompt and cache estimate.
            cpt = s.chars_per_token
            k = common_prefix(prompt.units, s.cache_units)
            if s.cache_units and k == len(s.cache_units):
                prefix = s.last_prompt_tokens
                cached = s.cached_tokens
            elif k > 0 and s.cache_units:
                share = sum(s.cache_chars[:k]) / max(1, sum(s.cache_chars))
                prefix = round(s.last_prompt_tokens * share)
                cached = min(s.cached_tokens, prefix)
            else:
                k, prefix, cached = 0, 0, 0
            est = prefix + prompt.est_tokens(k, cpt)
            if s.cache_units and now - s.cache_ts > s.cache_ttl_s:
                cached = 0
            retry_of = s.last_decision_id if s.last_units and s.last_units == prompt.units else None
            s.last_units = prompt.units
            estimate = Estimate(
                prompt_tokens=est,
                cached_prefix_tokens=min(cached, est),
                current_model=s.owner_model,
                current_effort=s.owner_effort,
                retry_of=retry_of,
            )
            return step, estimate, s.sensitive, s.sensitive_rules

    def set_decision(self, session_id: str, decision_id: str | None) -> None:
        with self.locked(session_id) as s:
            s.last_decision_id = decision_id

    def record_attempt(
        self,
        session_id: str,
        prompt: Prompt,
        *,
        model: str,
        effort: str | None,
        prompt_tokens: int,
        cache_min_tokens: int,
        cache_ttl_s: float,
        cacheable: bool = True,
    ) -> None:
        """An attempt on ``(model, effort)`` got a response with ``prompt_tokens``.

        The provider now holds the whole prompt for that owner (none of it if the prompt is
        below the provider's minimum, or if the request asked for no caching: an Anthropic
        request without a breakpoint, ``cacheable = False``), whether or not the response
        passed its check. The chars-per-token ratio is recalibrated from the real count.
        """
        with self.locked(session_id) as s:
            s.owner_model = model
            s.owner_effort = effort
            s.last_prompt_tokens = prompt_tokens
            s.cached_tokens = (
                prompt_tokens if cacheable and prompt_tokens >= cache_min_tokens else 0
            )
            s.cache_units = prompt.units
            s.cache_chars = prompt.chars
            s.cache_ts = self._clock()
            s.cache_ttl_s = cache_ttl_s
            text_tokens = (
                prompt_tokens
                - PER_UNIT_TOKENS * len(prompt.units)
                - IMAGE_TOKENS * sum(prompt.images)
            )
            if prompt.total_chars > 0 and text_tokens > 0:
                s.chars_per_token = min(20.0, max(0.5, prompt.total_chars / text_tokens))
