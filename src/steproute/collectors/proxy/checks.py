"""Response checks and usage normalisation. Standard library only (``jsonschema`` is imported
lazily for the schema check; it ships with LiteLLM).

Checks run on the first choice of a chat-completions response, as a plain dict. Each returns
``None`` when the response passes or the name of the failed check. The mapping of each check
to the signal vocabulary of ``docs/LABELS.md`` is ``signals_for``.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from steproute.collectors.proxy.config import CheckConfig
from steproute.core.cost import AttemptUsage

__all__ = [
    "CHECK_EMPTY",
    "CHECK_REFUSAL",
    "CHECK_SCHEMA",
    "CHECK_TOOL_CALL",
    "CheckResult",
    "StreamAccumulator",
    "normalise_usage",
    "run_checks",
    "signals_for",
]

CHECK_EMPTY = "empty_output"
CHECK_TOOL_CALL = "tool_call_parse"
CHECK_SCHEMA = "schema"
CHECK_REFUSAL = "refusal"


@dataclass(frozen=True)
class CheckResult:
    """Outcome of the checks on one response."""

    failed: str | None
    #: Whether a schema check actually ran (and so ``schema_valid`` is evidence either way).
    schema_checked: bool = False
    refusal: bool = False


def _first_message(response: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return {}, None
    msg = choices[0].get("message")
    return (msg if isinstance(msg, Mapping) else {}), choices[0].get("finish_reason")


def _tool_names(tools: Sequence[Any] | None) -> set[str] | None:
    if not tools:
        return None
    names = set()
    for t in tools:
        fn = t.get("function") if isinstance(t, Mapping) else None
        if isinstance(fn, Mapping) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names or None


def _tool_calls_ok(calls: Sequence[Any], allowed: set[str] | None) -> bool:
    for call in calls:
        fn = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(fn, Mapping):
            return False
        name, args = fn.get("name"), fn.get("arguments")
        if not isinstance(name, str) or not name or (allowed is not None and name not in allowed):
            return False
        if args in (None, ""):
            continue  # a call without arguments is valid for a tool without parameters
        if not isinstance(args, str):
            return False
        try:
            parsed = json.loads(args)
        except ValueError:
            return False
        if not isinstance(parsed, dict):
            return False
    return True


def _schema_of(response_format: Any) -> tuple[bool, Any]:
    """(check JSON at all, schema or None) for a request's ``response_format``."""
    if not isinstance(response_format, Mapping):
        return False, None
    kind = response_format.get("type")
    if kind == "json_object":
        return True, None
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        if isinstance(spec, Mapping) and isinstance(spec.get("schema"), Mapping):
            return True, spec["schema"]
        return True, None
    return False, None


def _schema_ok(content: str, schema: Any) -> bool:
    try:
        value = json.loads(content)
    except ValueError:
        return False
    if schema is None:
        return isinstance(value, dict)
    # Lazy: ships with litellm, not with the core.
    jsonschema: Any = importlib.import_module("jsonschema")

    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError:
        return False
    except jsonschema.SchemaError:
        return True  # a broken schema is the client's problem, not the model's failure
    return True


def run_checks(
    response: Mapping[str, Any], request: Mapping[str, Any], config: CheckConfig
) -> CheckResult:
    """Run the configured checks on ``response`` (a chat-completions dict).

    Definitions (first choice only):

    * ``refusal``: ``finish_reason == "content_filter"`` (LiteLLM's mapping of Anthropic's
      ``stop_reason: "refusal"``) or a non-empty ``message.refusal``. A refusal is reported
      as such and never also as empty output.
    * ``tool_call_parse``: some tool call has no function name, a name not among the
      request's tools, or ``arguments`` that are not a JSON object.
    * ``empty_output``: no tool calls and ``content`` missing or only whitespace.
    * ``schema``: the request set ``response_format`` (``json_object`` or ``json_schema``),
      the reply is text, and it is not a JSON object or fails the JSON Schema.
    """
    msg, finish = _first_message(response)
    content = msg.get("content")
    calls = msg.get("tool_calls") or []
    refused = finish == "content_filter" or bool(msg.get("refusal"))
    if refused:
        return CheckResult(CHECK_REFUSAL if config.refusal else None, refusal=True)
    if calls:
        if config.tool_call_parse and not _tool_calls_ok(calls, _tool_names(request.get("tools"))):
            return CheckResult(CHECK_TOOL_CALL)
        return CheckResult(None)
    text = content if isinstance(content, str) else ""
    if isinstance(content, list):  # content parts
        text = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, Mapping) and isinstance(p.get("text"), str)
        )
    if config.empty_output and not text.strip():
        return CheckResult(CHECK_EMPTY)
    want_json, schema = _schema_of(request.get("response_format"))
    if config.schema and want_json and text.strip():
        ok = _schema_ok(text, schema)
        return CheckResult(None if ok else CHECK_SCHEMA, schema_checked=True)
    return CheckResult(None)


def signals_for(result: CheckResult) -> dict[str, Any]:
    """The ``docs/LABELS.md`` signals a check result is evidence for.

    * ``tool_call_parse`` failed -> ``tool_error = True`` (error tier, failure only).
    * ``empty_output`` failed -> ``schema_valid = False``: the reply breaks the minimal
      response contract (a message has content or a tool call).
    * ``schema`` failed -> ``schema_valid = False``; passed -> ``schema_valid = True``.
    * ``refusal`` -> the raw ``refusal = True`` only, which the labeller ignores: a refusal
      can be the right answer, so it is neither success nor failure evidence.

    A passed empty-output or tool-call check is not evidence of success and adds nothing.
    """
    out: dict[str, Any] = {}
    if result.failed == CHECK_TOOL_CALL:
        out["tool_error"] = True
    elif result.failed in (CHECK_EMPTY, CHECK_SCHEMA):
        out["schema_valid"] = False
    elif result.schema_checked:
        out["schema_valid"] = True
    if result.refusal:
        out["refusal"] = True
    return out


class StreamAccumulator:
    """Collects what the checks need from a stream, without holding the stream back.

    Keeps the finish reason, whether any text arrived, tool-call names and argument strings
    (assembled by index), and the text only when a schema check will need it.
    """

    def __init__(self, keep_text: bool) -> None:
        self.keep_text = keep_text
        self.text_parts: list[str] = []
        self.any_text = False
        self.finish_reason: Any = None
        self.refusal = False
        self.calls: dict[int, dict[str, str]] = {}

    def add(self, chunk: Mapping[str, Any]) -> None:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            text = delta.get("content")
            if isinstance(text, str) and text:
                if text.strip():
                    self.any_text = True
                if self.keep_text:
                    self.text_parts.append(text)
            if delta.get("refusal"):
                self.refusal = True
            for tc in delta.get("tool_calls") or []:
                if not isinstance(tc, Mapping):
                    continue
                idx = tc.get("index", 0)
                slot = self.calls.setdefault(
                    int(idx) if isinstance(idx, int) else 0, {"name": "", "arguments": ""}
                )
                fn = tc.get("function")
                if isinstance(fn, Mapping):
                    if isinstance(fn.get("name"), str):
                        slot["name"] += fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["arguments"] += fn["arguments"]

    def as_response(self) -> dict[str, Any]:
        """The assembled first choice, in the non-streaming shape, for ``run_checks``."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.text_parts)
            if self.keep_text
            else ("x" if self.any_text else ""),
        }
        if self.refusal:
            message["refusal"] = True
        if self.calls:
            message["tool_calls"] = [
                {"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for _, c in sorted(self.calls.items())
            ]
        return {"choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}]}


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))


def normalise_usage(usage: Mapping[str, Any] | None, provider: str) -> AttemptUsage | None:
    """A LiteLLM usage dict in the ``docs/COST-MODEL.md`` token convention.

    LiteLLM reports ``prompt_tokens`` including cached tokens for every provider (it adds
    Anthropic's cache reads and writes back in). So ``cache_read = cached_tokens``; for
    Anthropic ``cache_write = cache_creation_input_tokens`` (or
    ``prompt_tokens_details.cache_creation_tokens``); otherwise 0; and ``input`` is the
    rest. ``None`` when the response carried no usage.
    """
    if not isinstance(usage, Mapping) or "prompt_tokens" not in usage:
        return None
    prompt = _int(usage.get("prompt_tokens"))
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, Mapping) else {}
    read = _int(details.get("cached_tokens"))
    write = 0
    if provider == "anthropic":
        write = _int(usage.get("cache_creation_input_tokens")) or _int(
            details.get("cache_creation_tokens")
        )
    read = min(read, prompt)
    write = min(write, prompt - read)
    return AttemptUsage(
        input_tokens=prompt - read - write,
        cache_read_tokens=read,
        cache_write_tokens=write,
        output_tokens=_int(usage.get("completion_tokens")),
    )
