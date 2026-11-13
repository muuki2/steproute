"""The upstream call through the LiteLLM SDK, and the request rewrite rules.

``litellm`` is imported lazily, by ``load_litellm``, which first switches off everything the
pinned version (1.109.1) does on the network besides the upstream call (``docs/PROXY.md``,
"LiteLLM's own network behaviour"):

* ``LITELLM_OFFLINE_MODE=true``: no remote model-cost-map fetch at import (and its retry
  thread), no remote Anthropic beta-header config on the first Anthropic call, no
  Hugging Face tokenizer download.
* ``LITELLM_LOCAL_MODEL_COST_MAP=true`` and ``LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS=true``:
  the older per-feature switches for the same fetches.
* No callbacks are registered, message logging is off, debug banners are suppressed, and
  every call passes ``max_retries=0`` (the proxy's cascade, not LiteLLM, decides retries).
* LiteLLM's own Anthropic cache-marker injection (``enable_anthropic_prompt_caching``, also
  switched on by ``LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING``) is forced off: the proxy places
  the breakpoints itself (``with_cache_breakpoints``), so what it believes about the cache is
  what it asked for.
"""

from __future__ import annotations

import importlib
import logging
import os
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from steproute.collectors.proxy.config import UpstreamConfig

__all__ = [
    "EPHEMERAL",
    "OFFLINE_ENV",
    "Prepared",
    "UpstreamError",
    "UpstreamPool",
    "has_cache_markers",
    "load_litellm",
    "with_cache_breakpoints",
]

log = logging.getLogger("steproute.proxy")

#: Environment set before ``import litellm``. Forced, not defaulted: the proxy must not fetch.
OFFLINE_ENV = {
    "LITELLM_OFFLINE_MODE": "true",
    "LITELLM_LOCAL_MODEL_COST_MAP": "true",
    "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "true",
}

_lock = threading.Lock()
_litellm: Any = None


def load_litellm() -> Any:
    """Import and configure ``litellm`` once. Raises ``ImportError`` without the extra."""
    global _litellm
    with _lock:
        if _litellm is not None:
            return _litellm
        os.environ.update(OFFLINE_ENV)
        mod: Any = importlib.import_module("litellm")
        mod.suppress_debug_info = True
        mod.turn_off_message_logging = True
        mod.disable_hf_tokenizer_download = True
        mod.drop_params = False  # parameter changes are the proxy's, explicit and logged
        mod.enable_anthropic_prompt_caching = False  # cache breakpoints are the proxy's too
        for name in (
            "callbacks",
            "success_callback",
            "failure_callback",
            "input_callback",
            "service_callback",
            "_async_success_callback",
            "_async_failure_callback",
        ):
            if isinstance(getattr(mod, name, None), list):
                getattr(mod, name).clear()
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        _litellm = mod
        return mod


class UpstreamError(Exception):
    """An upstream call failed before producing a response."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind


def _as_error(exc: Exception) -> UpstreamError:
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or not 400 <= status <= 599:
        status = 504 if "timeout" in type(exc).__name__.lower() else 502
    return UpstreamError(status, type(exc).__name__, str(exc)[:500])


def _dump(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    dumped = obj.model_dump(exclude_none=True)
    return dumped if isinstance(dumped, dict) else {}


#: The breakpoint the proxy writes: Anthropic's default 5-minute entry (no ``ttl``), which is
#: the write the catalog prices (``cache_write_mult``) and ``ANTHROPIC_CACHE_TTL_S`` assumes.
EPHEMERAL: Mapping[str, str] = {"type": "ephemeral"}


def _marked(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "cache_control" in value or any(_marked(v) for v in value.values())
    if isinstance(value, list):
        return any(_marked(v) for v in value)
    return False


def has_cache_markers(kwargs: Mapping[str, Any]) -> bool:
    """Whether the request already carries a ``cache_control`` marker anywhere: on a message,
    a content block, or a tool (top level or inside ``function``)."""
    return _marked(kwargs.get("messages")) or _marked(kwargs.get("tools"))


def _mark_message(message: Mapping[str, Any]) -> dict[str, Any] | None:
    """A copy of ``message`` with a breakpoint on its last text block, or ``None``."""
    content = message.get("content")
    if isinstance(content, str):
        # LiteLLM copies a message-level marker onto the block it builds from a string
        # (``AnthropicConfig.translate_system_message``), so the text is not reshaped.
        return {**message, "cache_control": dict(EPHEMERAL)} if content else None
    if isinstance(content, list):
        for i in range(len(content) - 1, -1, -1):
            part = content[i]
            if isinstance(part, Mapping) and part.get("type") == "text" and part.get("text"):
                parts = list(content)
                parts[i] = {**part, "cache_control": dict(EPHEMERAL)}
                return {**message, "content": parts}
    return None


def with_cache_breakpoints(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Add Anthropic prompt-cache breakpoints to a chat-completions request.

    Returns the new keyword arguments and who placed the markers: ``"proxy"``, or
    ``"client"`` when the request already had some (it is then left exactly as it is). The
    client's objects are never mutated. The proxy uses two of the four allowed breakpoints
    (``docs/PROXY.md``, "Prompt-cache breakpoints"):

    * the static prefix: the last text block of the leading system messages or, without a
      system prompt, the last tool definition. It caches tools and system together and
      survives any edit later in the conversation;
    * the moving end: Anthropic's request-level automatic breakpoint (top-level
      ``cache_control``), which lands on the last cacheable block and moves forward as the
      conversation grows, so step ``n + 1`` reads what step ``n`` wrote.
    """
    if has_cache_markers(kwargs):
        return dict(kwargs), "client"
    out = dict(kwargs)
    messages = kwargs.get("messages")
    marked = None
    lead = -1
    if isinstance(messages, list):
        while (
            lead + 1 < len(messages)
            and isinstance(messages[lead + 1], Mapping)
            and messages[lead + 1].get("role") == "system"
        ):
            lead += 1
        if lead >= 0:
            marked = _mark_message(messages[lead])
    if isinstance(messages, list) and marked is not None:
        out["messages"] = [*messages[:lead], marked, *messages[lead + 1 :]]
    else:
        tools = kwargs.get("tools")
        if isinstance(tools, list) and tools and isinstance(tools[-1], Mapping):
            out["tools"] = [*tools[:-1], {**tools[-1], "cache_control": dict(EPHEMERAL)}]
    out["cache_control"] = dict(EPHEMERAL)
    return out, "proxy"


@dataclass(frozen=True)
class Prepared:
    """A request ready for one upstream: the LiteLLM model string and keyword arguments.

    ``cache`` is who placed Anthropic prompt-cache markers: ``"proxy"``, ``"client"`` or
    ``"none"``.
    """

    upstream: UpstreamConfig
    model: str
    kwargs: dict[str, Any]
    dropped: tuple[str, ...]
    cache: str = "none"

    @property
    def cacheable(self) -> bool:
        """Whether the provider holds this prompt in its cache after the call. Anthropic
        caches only up to a breakpoint, so a request without one leaves nothing cached;
        DeepSeek and Ollama cache automatically."""
        return self.upstream.provider != "anthropic" or self.cache != "none"


class UpstreamPool:
    """Makes the calls. ``environ`` supplies the API keys named by ``api_key_env``."""

    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self.environ = os.environ if environ is None else environ

    def prepare(
        self,
        upstream: UpstreamConfig,
        catalog_id: str,
        params: Mapping[str, Any],
        *,
        rewrite: bool,
        reasoning: str | None,
    ) -> Prepared:
        """Build the call for ``catalog_id`` on ``upstream`` from the client's ``params``.

        With ``rewrite`` (route mode): the client's reasoning fields are replaced by
        ``reasoning`` (``None`` sends none), and the upstream's parameter policy is applied
        (``drop_params``, forced ``tool_choice``). Without it (shadow mode, fail-open to the
        client's model) the parameters go out as the client sent them.
        """
        kwargs = dict(params)
        dropped: list[str] = []
        if rewrite:
            for key in ("reasoning_effort", "thinking", "reasoning"):
                if kwargs.pop(key, None) is not None and reasoning is None:
                    dropped.append(key)
            if reasoning is not None:
                kwargs["reasoning_effort"] = reasoning
            for key in upstream.drop_params:
                if key in kwargs:
                    kwargs.pop(key)
                    dropped.append(key)
            choice = kwargs.get("tool_choice")
            forced = choice == "required" or (
                isinstance(choice, Mapping) and choice.get("type") in ("function", "tool")
            )
            if upstream.forced_tool_choice == "auto" and forced:
                kwargs["tool_choice"] = "auto"
                dropped.append("tool_choice")
        cache = "none"
        if upstream.provider == "anthropic":
            if upstream.marks_cache(rewrite):
                kwargs, cache = with_cache_breakpoints(kwargs)
            elif has_cache_markers(kwargs):
                cache = "client"
        return Prepared(
            upstream,
            f"{upstream.provider}/{upstream.upstream_model(catalog_id)}",
            kwargs,
            tuple(dropped),
            cache,
        )

    def _base(self, p: Prepared) -> dict[str, Any]:
        up = p.upstream
        key = self.environ.get(up.api_key_env, "") if up.api_key_env else ""
        extra: dict[str, Any] = {
            "model": p.model,
            "api_key": key or "steproute-no-key",
            "timeout": up.timeout_s,
            "max_retries": 0,
        }
        if up.base_url:
            extra["api_base"] = up.base_url
        return {**p.kwargs, **extra}

    def complete(self, p: Prepared) -> dict[str, Any]:
        litellm = load_litellm()
        try:
            response = litellm.completion(**self._base(p), stream=False)
        except Exception as exc:
            raise _as_error(exc) from None
        return _dump(response)

    def stream(self, p: Prepared) -> Iterator[dict[str, Any]]:
        """Chunks as dicts. The first chunk is fetched before returning, so a failure to
        connect or an HTTP error surfaces here as ``UpstreamError``, before any byte has gone
        to the client; later failures surface from the iterator."""
        litellm = load_litellm()
        kwargs = self._base(p)
        options = dict(kwargs.pop("stream_options", None) or {})
        options["include_usage"] = True
        try:
            it = iter(litellm.completion(**kwargs, stream=True, stream_options=options))
            first = next(it, None)
        except Exception as exc:
            raise _as_error(exc) from None
        return _chain(first, it)


def _chain(first: Any, it: Iterator[Any]) -> Iterator[dict[str, Any]]:
    if first is not None:
        yield _dump(first)
    try:
        for chunk in it:
            yield _dump(chunk)
    except Exception as exc:
        raise _as_error(exc) from None
