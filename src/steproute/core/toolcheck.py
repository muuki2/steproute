"""Tool-call smoke test for OpenAI-compatible endpoints (local Ollama by default).

Sends one chat-completions request with one simple tool defined and passes only if the reply
carries a structured tool call. A call written as text in ``content`` is a failure. Stdlib only;
the HTTP transport is injectable so tests open no sockets. Cloud providers are deliberately not
reachable from here: they are checked through the proxy (issue 12).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_BASE_URL",
    "CheckResult",
    "Transport",
    "TransportError",
    "check_model",
    "urllib_transport",
]

DEFAULT_BASE_URL = "http://localhost:11434/v1"

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


class TransportError(Exception):
    """The request did not produce a JSON response. ``unreachable`` means no HTTP answer."""

    def __init__(self, message: str, *, unreachable: bool = False):
        super().__init__(message)
        self.unreachable = unreachable


#: ``(url, json_body, timeout_seconds) -> parsed_json_response``; raises ``TransportError``.
Transport = Callable[[str, dict[str, Any], float], dict[str, Any]]


@dataclass(frozen=True)
class CheckResult:
    model: str
    ok: bool
    detail: str
    unreachable: bool = False


def urllib_transport(url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST ``body`` as JSON with ``urllib`` and parse the JSON reply."""
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        raise TransportError(f"HTTP {exc.code} {exc.reason}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransportError(str(reason), unreachable=True) from None
    try:
        parsed = json.loads(payload)
    except ValueError:
        raise TransportError("response is not JSON") from None
    if not isinstance(parsed, dict):
        raise TransportError("response is not a JSON object")
    return parsed


def _inspect(response: dict[str, Any]) -> tuple[bool, str]:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return False, "response has no choices[0].message"
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    if not calls:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and content.strip():
            return False, "no structured tool_calls; the reply was plain text"
        return False, "no structured tool_calls in the reply"
    try:
        fn = calls[0]["function"]
        name = fn["name"]
        args = fn["arguments"]
    except (KeyError, IndexError, TypeError):
        return False, "tool_calls[0] has no function name/arguments"
    if not isinstance(name, str) or not name:
        return False, "tool call has no function name"
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except ValueError:
            return False, "tool call arguments are not valid JSON"
    else:
        parsed = args
    if not isinstance(parsed, dict):
        return False, "tool call arguments are not a JSON object"
    return True, f"structured tool call {name}({json.dumps(parsed)})"


def check_model(
    model: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    transport: Transport = urllib_transport,
    timeout: float = 120.0,
) -> CheckResult:
    """Run the smoke test for one model id against ``base_url``."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Paris right now? Use the get_weather tool.",
            }
        ],
        "tools": [_TOOL],
        "stream": False,
    }
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        response = transport(url, body, timeout)
    except TransportError as exc:
        if exc.unreachable:
            return CheckResult(model, False, f"server unreachable: {exc}", unreachable=True)
        return CheckResult(model, False, f"request failed: {exc}")
    ok, detail = _inspect(response)
    return CheckResult(model, ok, detail)
