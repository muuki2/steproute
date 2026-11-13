"""The replay runner's configuration: one strict TOML file. Standard library only.

Every default is the safe one: the D3 caps ($80 Anthropic, $20 DeepSeek per UTC calendar
month), a $3 per-run cap, sequential local runs, and no key in the file (only the NAME of the
environment variable that holds it). A cap above its default needs ``allow_raise = true`` in
the same table, so raising one is always a visible, deliberate edit. See ``docs/REPLAY.md``.
"""

from __future__ import annotations

import dataclasses
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from steproute.collectors.proxy.config import is_env_name, is_loopback_host

__all__ = [
    "DEFAULT_CAPS",
    "LOCAL_PROVIDERS",
    "Limits",
    "ProviderConfig",
    "ReplayConfig",
    "ReplayConfigError",
    "load_replay_config",
]

#: Monthly caps in USD per provider (decision D3). Raising one needs ``allow_raise = true``.
DEFAULT_CAPS: Mapping[str, float] = {"anthropic": 80.0, "deepseek": 20.0}
#: Providers that run on this machine and cost nothing.
LOCAL_PROVIDERS = ("ollama",)

ENVIRONMENTS = ("docker", "local")


class ReplayConfigError(ValueError):
    """The replay configuration is invalid; nothing may run."""


@dataclass(frozen=True)
class ProviderConfig:
    """How the runner reaches one provider. ``api_key_env`` is a variable NAME, never a key."""

    name: str
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_s: float = 600.0


@dataclass(frozen=True)
class Limits:
    """Per-run limits. ``run_cap_usd`` stops a run that is burning money; it is then recorded
    as a labelled failure (``run_cap``) and not retried."""

    run_cap_usd: float = 3.0
    step_limit: int = 250
    #: Largest prompt a paid call may carry, in tokens. A run whose next prompt would pass it
    #: stops (``context_limit``). This is what bounds the worst case of a single call.
    max_prompt_tokens: int = 200_000
    #: ``max_tokens`` sent on every call.
    max_output_tokens: int = 16_384
    #: Decision D2: the context cap for local models.
    local_context_cap: int = 32_768
    command_timeout_s: float = 60.0
    run_timeout_s: float = 7200.0
    #: Retries of a whole run after a provider or infrastructure error or an interruption.
    max_retries: int = 2
    #: Retries of one call after a transient provider error (429, 5xx), inside a run.
    call_retries: int = 2
    call_backoff_s: float = 5.0
    #: Observation text longer than this is cut to its head and tail (mini-SWE-agent: 10,000).
    max_observation_chars: int = 10_000
    #: After this many finished Anthropic runs, stop Anthropic if the observed cache-read
    #: share of input tokens is below ``min_cache_read_share`` (prompt caching is not reaching
    #: the provider: every later run would cost about 3 times the estimate). 0 disables.
    cache_canary_runs: int = 3
    min_cache_read_share: float = 0.3


@dataclass(frozen=True)
class ReplayConfig:
    caps: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CAPS))
    limits: Limits = field(default_factory=Limits)
    providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    #: Paid runs in parallel. Local runs are always sequential, one model loaded at a time.
    paid_concurrency: int = 2
    environment: str = "docker"
    docker: str = "docker"
    #: ``[check]``: the SWE-bench harness command prefix and the dataset rows file.
    harness: tuple[str, ...] = ("python", "-m", "swebench.harness.run_evaluation")
    rows: Path | None = None
    catalog_path: Path | None = None
    #: Unload the previous Ollama model before loading the next (``keep_alive = 0``).
    manage_ollama: bool = True
    path: Path | None = None

    def provider(self, name: str) -> ProviderConfig:
        return self.providers.get(name) or ProviderConfig(name)

    def cap(self, provider: str) -> float | None:
        """The monthly cap, ``None`` for a local provider. A paid provider without a cap is a
        configuration error caught at load and at plan time."""
        if provider in LOCAL_PROVIDERS:
            return None
        return self.caps.get(provider)


_DEFAULT_PROVIDERS = {
    "anthropic": ProviderConfig("anthropic", api_key_env="ANTHROPIC_API_KEY"),
    "deepseek": ProviderConfig("deepseek", api_key_env="DEEPSEEK_API_KEY"),
    "ollama": ProviderConfig("ollama", base_url="http://127.0.0.1:11434/v1"),
}

_LIMIT_KEYS: dict[str, str] = {
    "run_cap_usd": "num",
    "step_limit": "int",
    "max_prompt_tokens": "int",
    "max_output_tokens": "int",
    "local_context_cap": "int",
    "command_timeout_s": "num",
    "run_timeout_s": "num",
    "max_retries": "int",
    "call_retries": "int",
    "call_backoff_s": "num",
    "max_observation_chars": "int",
    "cache_canary_runs": "int",
    "min_cache_read_share": "num",
}


def _value(where: str, v: Any, kind: str) -> Any:
    if kind == "int" and isinstance(v, int) and not isinstance(v, bool) and v >= 0:
        return v
    if kind == "num" and isinstance(v, int | float) and not isinstance(v, bool) and v >= 0:
        return float(v)
    if kind == "bool" and isinstance(v, bool):
        return v
    if kind == "str" and isinstance(v, str) and v:
        return v
    if kind == "strs" and isinstance(v, list) and v and all(isinstance(x, str) and x for x in v):
        return tuple(v)
    raise ReplayConfigError(f"{where}: expected a non-negative {kind}")


def _table(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    t = data.get(key, {})
    if not isinstance(t, Mapping):
        raise ReplayConfigError(f"[{key}] must be a table")
    return t


def parse_replay_config(data: Mapping[str, Any], *, base: Path | None = None) -> ReplayConfig:
    """Validate a parsed config. Unknown sections and keys are errors."""
    base = base or Path.cwd()
    known = {"caps", "limits", "concurrency", "providers", "environment", "check", "catalog"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ReplayConfigError(f"unknown section(s) {unknown}; expected {sorted(known)}")

    caps_t = dict(_table(data, "caps"))
    allow = caps_t.pop("allow_raise", False)
    if not isinstance(allow, bool):
        raise ReplayConfigError("[caps] allow_raise must be true or false")
    caps = dict(DEFAULT_CAPS)
    for prov, v in caps_t.items():
        if prov in LOCAL_PROVIDERS:
            raise ReplayConfigError(f"[caps] {prov} is local and free; it takes no cap")
        usd = _value(f"[caps] {prov}", v, "num")
        default = DEFAULT_CAPS.get(prov)
        if default is not None and usd > default and not allow:
            raise ReplayConfigError(
                f"[caps] {prov} = {usd:g} is above the default ${default:g} a month; raising a "
                "cap needs `allow_raise = true` in [caps] (decision D3: only after Gate A)"
            )
        caps[prov] = usd

    lim_t = _table(data, "limits")
    bad = sorted(set(lim_t) - set(_LIMIT_KEYS))
    if bad:
        raise ReplayConfigError(f"[limits] unknown key(s) {bad}")
    limits = Limits(**{k: _value(f"[limits] {k}", v, _LIMIT_KEYS[k]) for k, v in lim_t.items()})
    if limits.run_cap_usd <= 0:
        raise ReplayConfigError("[limits] run_cap_usd must be positive")
    if limits.step_limit < 1 or limits.max_prompt_tokens < 1 or limits.max_output_tokens < 1:
        raise ReplayConfigError("[limits] step_limit and the token limits must be at least 1")
    if not 0.0 <= limits.min_cache_read_share <= 1.0:
        raise ReplayConfigError("[limits] min_cache_read_share must be in [0, 1]")

    conc = _table(data, "concurrency")
    if sorted(set(conc) - {"paid"}):
        raise ReplayConfigError("[concurrency] takes only `paid`")
    paid = _value("[concurrency] paid", conc.get("paid", 2), "int")
    if paid < 1:
        raise ReplayConfigError("[concurrency] paid must be at least 1")

    providers = dict(_DEFAULT_PROVIDERS)
    for name, t in _table(data, "providers").items():
        if not isinstance(t, Mapping):
            raise ReplayConfigError(f"[providers.{name}] must be a table")
        extra = sorted(set(t) - {"api_key_env", "base_url", "timeout_s"})
        if extra:
            raise ReplayConfigError(f"[providers.{name}] unknown key(s) {extra}")
        prev = providers.get(name, ProviderConfig(name))
        env = t.get("api_key_env", prev.api_key_env)
        if env is not None and (not isinstance(env, str) or not is_env_name(env)):
            # Not echoed: someone may have pasted the key itself.
            raise ReplayConfigError(
                f"[providers.{name}] api_key_env must be the NAME of an environment variable"
            )
        url = t.get("base_url", prev.base_url)
        if url is not None:
            parts = urlsplit(str(url))
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ReplayConfigError(f"[providers.{name}] base_url must be an http(s) URL")
            if parts.username or parts.password:
                raise ReplayConfigError(f"[providers.{name}] base_url must not carry credentials")
            if name in LOCAL_PROVIDERS and not is_loopback_host(parts.hostname):
                raise ReplayConfigError(f"[providers.{name}] a local provider must be on loopback")
        timeout = _value(f"[providers.{name}] timeout_s", t.get("timeout_s", prev.timeout_s), "num")
        providers[name] = ProviderConfig(name, env, url, timeout)

    env_t = _table(data, "environment")
    if sorted(set(env_t) - {"kind", "docker", "manage_ollama"}):
        raise ReplayConfigError("[environment] takes kind, docker and manage_ollama")
    kind = env_t.get("kind", "docker")
    if kind not in ENVIRONMENTS:
        raise ReplayConfigError(f"[environment] kind must be one of {list(ENVIRONMENTS)}")
    docker = _value("[environment] docker", env_t.get("docker", "docker"), "str")
    manage = _value("[environment] manage_ollama", env_t.get("manage_ollama", True), "bool")

    check = _table(data, "check")
    if sorted(set(check) - {"harness", "rows"}):
        raise ReplayConfigError("[check] takes harness and rows")
    harness = _value("[check] harness", check.get("harness", list(ReplayConfig.harness)), "strs")
    rows = check.get("rows")
    rows_p = (base / _value("[check] rows", rows, "str")).resolve() if rows is not None else None

    cat = _table(data, "catalog")
    if sorted(set(cat) - {"path"}):
        raise ReplayConfigError("[catalog] takes only path")
    cat_p = (base / _value("[catalog] path", cat["path"], "str")).resolve() if cat else None

    return ReplayConfig(
        caps=caps,
        limits=limits,
        providers=providers,
        paid_concurrency=paid,
        environment=kind,
        docker=docker,
        harness=harness,
        rows=rows_p,
        catalog_path=cat_p,
        manage_ollama=manage,
    )


def load_replay_config(path: str | Path | None) -> ReplayConfig:
    """Load a config file, or the defaults when ``path`` is ``None``."""
    if path is None:
        return parse_replay_config({})
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ReplayConfigError(f"{p}: invalid TOML: {exc}") from None
    except OSError as exc:
        raise ReplayConfigError(f"cannot read {p}: {exc}") from None
    cfg = parse_replay_config(data, base=p.resolve().parent)
    return dataclasses.replace(cfg, path=p.resolve())
