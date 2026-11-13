"""Proxy configuration: one strict TOML file. Standard library only.

See ``docs/PROXY.md`` for the reference. Unknown sections and keys are an error, secrets are
never in the file (only the NAME of the environment variable that holds each one), and every
check that can be made without the network is made at load time.
"""

from __future__ import annotations

import ipaddress
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from steproute.core.catalog import Catalog

__all__ = [
    "CheckConfig",
    "MODES",
    "PROVIDERS",
    "ProxyConfig",
    "ProxyConfigError",
    "UpstreamConfig",
    "is_env_name",
    "is_loopback_host",
    "load_proxy_config",
    "validate_against_catalog",
]

#: ``route`` applies decisions; ``shadow`` decides and logs but runs the client's own model.
MODES = ("route", "shadow")
#: LiteLLM provider prefix per upstream. ``openai`` is any OpenAI-compatible server (Ollama
#: ``/v1``, vLLM, ...); ``anthropic`` and ``deepseek`` use LiteLLM's native providers.
PROVIDERS = ("openai", "anthropic", "deepseek")
SOURCES = ("inprocess", "http")
TASK_TEXT = ("none", "last_user")
FORCED_TOOL_CHOICE = ("keep", "auto")
ON_UPSTREAM_ERROR = ("client_model", "error")

DEFAULT_PORT = 8788
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

#: OpenAI chat-completions request fields forwarded upstream. Anything else in a client's body
#: is dropped (and counted): LiteLLM treats unknown keyword arguments as its own options, and
#: some of those (``api_base``, ``api_key``, ``extra_headers``, ...) would let a request body
#: redirect the call or change credentials.
FORWARDED_PARAMS = frozenset(
    {
        "messages",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "seed",
        "response_format",
        "user",
        "reasoning_effort",
        "thinking",
        "modalities",
        "audio",
        "prediction",
        "service_tier",
        "stream_options",
    }
)
#: Never forwardable, even through ``[proxy] extra_params``.
FORBIDDEN_PARAMS = frozenset(
    {
        "api_base",
        "base_url",
        "api_key",
        "api_version",
        "custom_llm_provider",
        "extra_headers",
        "headers",
        "metadata",
        "mock_response",
        "callbacks",
        "num_retries",
        "max_retries",
        "fallbacks",
        "timeout",
        "model_list",
        "litellm_logging_obj",
    }
)


class ProxyConfigError(ValueError):
    """The proxy configuration is invalid; the proxy must not start."""


def is_env_name(value: str) -> bool:
    return (
        bool(value)
        and not value[0].isdigit()
        and all(c.isascii() and (c.isalnum() or c == "_") for c in value)
    )


def is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class UpstreamConfig:
    """One upstream server and the catalog models it serves."""

    name: str
    provider: str
    models: tuple[str, ...]
    base_url: str | None = None
    api_key_env: str | None = None
    local: bool = False
    #: catalog id -> the model name the upstream expects (default: the catalog id).
    model_names: Mapping[str, str] = field(default_factory=dict)
    #: The catalog's ``effort_map`` value -> the value sent as ``reasoning_effort``. For
    #: Ollama ``/v1`` the catalog's ``"true"``/``"false"`` (native ``think``) must become
    #: ``"medium"``/``"none"``.
    reasoning_values: Mapping[str, str] = field(default_factory=dict)
    #: Request fields removed before the call (for example sampling parameters that current
    #: Claude models reject with a 400).
    drop_params: tuple[str, ...] = ()
    #: ``auto`` turns a forced ``tool_choice`` (``required`` or a named tool) into ``auto``.
    forced_tool_choice: str = "keep"
    cache_ttl_s: float = 300.0
    timeout_s: float = 600.0

    def upstream_model(self, catalog_id: str) -> str:
        return self.model_names.get(catalog_id, catalog_id)


@dataclass(frozen=True)
class CheckConfig:
    """Response checks that trigger the next cascade step (non-streaming only)."""

    empty_output: bool = True
    tool_call_parse: bool = True
    schema: bool = True
    refusal: bool = False


@dataclass(frozen=True)
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    token_env: str | None = None
    mode: str = "route"
    harness: str = "proxy"
    on_upstream_error: str = "client_model"
    extra_params: tuple[str, ...] = ()
    session_header: str = "X-Steproute-Session"
    max_sessions: int = 1024
    session_idle_s: float = 3600.0
    chars_per_token: float = 4.0
    session_salt_env: str | None = None
    source: str = "inprocess"
    service_config: Path | None = None
    service_url: str = "http://127.0.0.1:8787"
    service_token_env: str | None = None
    decide_timeout_ms: float = 250.0
    report_timeout_ms: float = 2000.0
    catalog_path: Path | None = None
    checks: CheckConfig = field(default_factory=CheckConfig)
    task_text: str = "none"
    local_fallback: str | None = None
    aliases: Mapping[str, str] = field(default_factory=dict)
    upstreams: tuple[UpstreamConfig, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not 0 <= self.port <= 65535:
            raise ProxyConfigError(f"server.port must be 0..65535, got {self.port!r}")
        if self.max_body_bytes < 1:
            raise ProxyConfigError("server.max_body_bytes must be at least 1")
        for key, value, allowed in (
            ("proxy.mode", self.mode, MODES),
            ("proxy.on_upstream_error", self.on_upstream_error, ON_UPSTREAM_ERROR),
            ("decisions.source", self.source, SOURCES),
            ("privacy.task_text", self.task_text, TASK_TEXT),
        ):
            if value not in allowed:
                raise ProxyConfigError(f"{key} must be one of {list(allowed)}, got {value!r}")
        for key, env in (
            ("server.token_env", self.token_env),
            ("session.salt_env", self.session_salt_env),
            ("decisions.token_env", self.service_token_env),
        ):
            if env is not None and not is_env_name(env):
                raise ProxyConfigError(
                    f"{key} must be the NAME of an environment variable, not a secret value"
                )
        if self.max_sessions < 1 or not self.session_idle_s > 0:
            raise ProxyConfigError("session.max_sessions and session.idle_s must be positive")
        if not 0.5 <= self.chars_per_token <= 20:
            raise ProxyConfigError("session.chars_per_token must be in [0.5, 20]")
        if not self.decide_timeout_ms > 0 or not self.report_timeout_ms > 0:
            raise ProxyConfigError("decisions timeouts must be positive")
        bad = sorted(set(self.extra_params) & FORBIDDEN_PARAMS)
        if bad:
            raise ProxyConfigError(f"proxy.extra_params may not include {bad}")
        if self.source == "inprocess" and self.catalog_path is not None:
            raise ProxyConfigError(
                "[catalog] is only for decisions.source = 'http'; in-process decisions use "
                "the catalog of decisions.service_config"
            )
        if self.source == "http" and self.service_config is not None:
            raise ProxyConfigError("decisions.service_config is only for source = 'inprocess'")
        if not self.upstreams:
            raise ProxyConfigError("at least one [[upstream]] is required")
        names: set[str] = set()
        served: dict[str, str] = {}
        for up in self.upstreams:
            if up.name in names:
                raise ProxyConfigError(f"duplicate upstream name {up.name!r}")
            names.add(up.name)
            for model in up.models:
                if model in served:
                    raise ProxyConfigError(
                        f"model {model!r} is served by both {served[model]!r} and {up.name!r}"
                    )
                served[model] = up.name
        for alias, target in self.aliases.items():
            if target not in served:
                raise ProxyConfigError(
                    f"alias {alias!r} points to {target!r}, which no upstream serves"
                )
        if self.local_fallback is not None and self.local_fallback not in served:
            raise ProxyConfigError(f"privacy.local_fallback {self.local_fallback!r} is not served")

    def upstream_for(self, catalog_id: str) -> UpstreamConfig | None:
        for up in self.upstreams:
            if catalog_id in up.models:
                return up
        return None


def validate_against_catalog(config: ProxyConfig, catalog: Catalog) -> list[str]:
    """Check the upstreams against ``catalog``; return warnings, raise on errors.

    Errors: a served model missing from the catalog; a local catalog model on an upstream not
    marked ``local``; a ``local_fallback`` that is not a local model. Warnings: non-holdout
    catalog models no upstream serves (the policy may choose them; the proxy then skips that
    step).
    """
    for up in config.upstreams:
        for model in up.models:
            if model not in catalog:
                raise ProxyConfigError(f"upstream {up.name!r}: {model!r} is not in the catalog")
            if catalog.get(model).local and not up.local:
                raise ProxyConfigError(
                    f"upstream {up.name!r} serves local model {model!r} but is not local = true"
                )
    if config.local_fallback is not None and not catalog.get(config.local_fallback).local:
        raise ProxyConfigError(f"privacy.local_fallback {config.local_fallback!r} is not local")
    unserved = [m.id for m in catalog.pool() if config.upstream_for(m.id) is None]
    if unserved:
        return [f"no upstream serves {', '.join(unserved)}; decisions for them fall back"]
    return []


# --- loading --------------------------------------------------------------------------------

_SCALARS: dict[str, dict[str, tuple[str, str]]] = {
    "server": {
        "host": ("host", "str"),
        "port": ("port", "int"),
        "max_body_bytes": ("max_body_bytes", "int"),
        "token_env": ("token_env", "str"),
    },
    "proxy": {
        "mode": ("mode", "str"),
        "harness": ("harness", "str"),
        "on_upstream_error": ("on_upstream_error", "str"),
        "extra_params": ("extra_params", "strs"),
    },
    "session": {
        "header": ("session_header", "str"),
        "max_sessions": ("max_sessions", "int"),
        "idle_s": ("session_idle_s", "num"),
        "chars_per_token": ("chars_per_token", "num"),
        "salt_env": ("session_salt_env", "str"),
    },
    "decisions": {
        "source": ("source", "str"),
        "service_config": ("service_config", "path"),
        "url": ("service_url", "str"),
        "token_env": ("service_token_env", "str"),
        "timeout_ms": ("decide_timeout_ms", "num"),
        "report_timeout_ms": ("report_timeout_ms", "num"),
    },
    "catalog": {"path": ("catalog_path", "path")},
    "privacy": {
        "task_text": ("task_text", "str"),
        "local_fallback": ("local_fallback", "str"),
    },
}
_CHECK_KEYS = ("empty_output", "tool_call_parse", "schema", "refusal")
_UPSTREAM_KEYS: dict[str, str] = {
    "name": "str",
    "provider": "str",
    "models": "strs",
    "base_url": "str",
    "api_key_env": "str",
    "local": "bool",
    "model_names": "strmap",
    "reasoning_values": "strmap",
    "drop_params": "strs",
    "forced_tool_choice": "str",
    "cache_ttl_s": "num",
    "timeout_s": "num",
}


def _coerce(where: str, value: Any, kind: str, base: Path) -> Any:
    if kind == "int":
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif kind == "num":
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    elif kind == "bool":
        if isinstance(value, bool):
            return value
    elif kind == "str":
        if isinstance(value, str) and value:
            return value
    elif kind == "path":
        if isinstance(value, str) and value:
            return (base / Path(value).expanduser()).resolve()
    elif kind == "strs":
        if isinstance(value, list) and all(isinstance(v, str) and v for v in value):
            return tuple(value)
    elif kind == "strmap":
        if isinstance(value, Mapping) and all(
            isinstance(k, str) and isinstance(v, str) and v for k, v in value.items()
        ):
            return dict(value)
    raise ProxyConfigError(f"{where}: wrong type or empty value (expected {kind})")


def _upstream(index: int, table: Any, base: Path) -> UpstreamConfig:
    where = f"[[upstream]] #{index + 1}"
    if not isinstance(table, Mapping):
        raise ProxyConfigError(f"{where} must be a table")
    extra = sorted(set(table) - set(_UPSTREAM_KEYS))
    if extra:
        raise ProxyConfigError(
            f"{where}: unknown key(s) {extra}; expected {sorted(_UPSTREAM_KEYS)}"
        )
    values = {k: _coerce(f"{where} {k}", v, _UPSTREAM_KEYS[k], base) for k, v in table.items()}
    for required in ("name", "provider", "models"):
        if required not in values:
            raise ProxyConfigError(f"{where}: {required!r} is required")
    where = f"upstream {values['name']!r}"
    if values["provider"] not in PROVIDERS:
        raise ProxyConfigError(f"{where}: provider must be one of {list(PROVIDERS)}")
    if not values["models"]:
        raise ProxyConfigError(f"{where}: models must not be empty")
    if values.get("forced_tool_choice", "keep") not in FORCED_TOOL_CHOICE:
        raise ProxyConfigError(f"{where}: forced_tool_choice must be one of {FORCED_TOOL_CHOICE}")
    env = values.get("api_key_env")
    if env is not None and not is_env_name(env):
        # Not echoed: someone may have pasted the key itself.
        raise ProxyConfigError(f"{where}: api_key_env must be the NAME of an environment variable")
    url = values.get("base_url")
    if url is None and values["provider"] == "openai":
        raise ProxyConfigError(f"{where}: base_url is required for provider 'openai'")
    if url is not None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ProxyConfigError(f"{where}: base_url must be an http(s) URL")
        if parts.username or parts.password:
            raise ProxyConfigError(f"{where}: base_url must not carry credentials")
    if values.get("local", False):
        host = urlsplit(url).hostname if url else None
        if host is None or not is_loopback_host(host):
            raise ProxyConfigError(
                f"{where}: local = true needs a loopback base_url (127.0.0.1, ::1 or localhost); "
                "a model on another machine is not local for the privacy rule"
            )
    for key in ("cache_ttl_s", "timeout_s"):
        if key in values and not values[key] > 0:
            raise ProxyConfigError(f"{where}: {key} must be positive")
    unknown = sorted(set(values.get("model_names", {})) - set(values["models"]))
    if unknown:
        raise ProxyConfigError(f"{where}: model_names keys {unknown} are not in models")
    return UpstreamConfig(**values)


def load_proxy_config(source: str | Path | Mapping[str, Any]) -> ProxyConfig:
    """Load a ``ProxyConfig`` from a TOML file or a parsed mapping.

    Relative paths in a file resolve against the file's directory (against the current
    directory for a mapping). Raises ``ProxyConfigError``.
    """
    base = Path.cwd()
    if isinstance(source, Mapping):
        data: Mapping[str, Any] = source
    else:
        path = Path(source)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ProxyConfigError(f"{path}: invalid TOML: {exc}") from None
        except OSError as exc:
            raise ProxyConfigError(f"cannot read config: {exc}") from None
        base = path.resolve().parent
    known = sorted({*_SCALARS, "checks", "aliases", "upstream"})
    values: dict[str, Any] = {}
    for section, body in data.items():
        if section not in known:
            raise ProxyConfigError(f"unknown section [{section}]; expected {known}")
        if section == "upstream":
            if not isinstance(body, list):
                raise ProxyConfigError("upstreams are an array of tables: use [[upstream]]")
            values["upstreams"] = tuple(_upstream(i, t, base) for i, t in enumerate(body))
            continue
        if not isinstance(body, Mapping):
            raise ProxyConfigError(f"[{section}] must be a table")
        if section == "checks":
            extra = sorted(set(body) - set(_CHECK_KEYS))
            if extra:
                raise ProxyConfigError(
                    f"[checks] unknown key(s) {extra}; expected {list(_CHECK_KEYS)}"
                )
            flags = {k: _coerce(f"[checks] {k}", v, "bool", base) for k, v in body.items()}
            values["checks"] = CheckConfig(**flags)
            continue
        if section == "aliases":
            values["aliases"] = _coerce("[aliases]", dict(body), "strmap", base)
            continue
        for key, value in body.items():
            if key not in _SCALARS[section]:
                raise ProxyConfigError(
                    f"[{section}] unknown key {key!r}; expected {sorted(_SCALARS[section])}"
                )
            name, kind = _SCALARS[section][key]
            values[name] = _coerce(f"[{section}] {key}", value, kind, base)
    return ProxyConfig(**values)
