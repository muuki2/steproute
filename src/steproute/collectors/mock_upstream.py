"""A scriptable mock of an OpenAI-compatible chat-completions server, on loopback only.

Standard library only. Used by the proxy tests (and usable by the replay runner's CI) so no
test ever reaches a real provider. Each request pops the next scripted ``Reply`` (or uses the
default one), and every request is recorded so a test can assert what the proxy sent: the
rewritten model, the reasoning parameter, the dropped sampling parameters.

Supported: ``POST /v1/chat/completions`` (non-streaming JSON and SSE streaming, with a final
usage chunk when ``stream_options.include_usage`` is set) and ``GET /v1/models``.

The same server also speaks Anthropic's Messages API on ``POST /v1/messages``, well enough
for LiteLLM's ``anthropic`` provider (point it at ``anthropic_base_url``): non-streaming JSON
and SSE streaming (``message_start`` ... ``message_stop``), Anthropic error bodies, and usage
with ``cache_read_input_tokens`` (``Reply.cached_tokens``) and ``cache_creation_input_tokens``
(``Reply.cache_write_tokens``) reported apart from ``input_tokens``, as Anthropic does. The
mock does not cache anything itself: the counts are what the test scripts.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

__all__ = ["MockUpstream", "Received", "Reply"]


@dataclass(frozen=True)
class Reply:
    """One scripted answer.

    ``kind``:

    * ``"text"``: an assistant message with ``content``.
    * ``"empty"``: an assistant message with empty content and no tool calls.
    * ``"tool"``: one tool call named ``tool_name`` with ``tool_args`` as the raw arguments
      string (so a test can send malformed JSON).
    * ``"error"``: HTTP ``status`` with an OpenAI-style error body.
    * ``"refusal"``: empty content with ``finish_reason = "content_filter"``.
    """

    kind: str = "text"
    content: str = "ok"
    tool_name: str = "read_file"
    tool_args: str = '{"path": "README.md"}'
    status: int = 500
    prompt_tokens: int = 100
    completion_tokens: int = 10
    cached_tokens: int | None = None
    #: Anthropic only: tokens written to the cache. ``prompt_tokens`` is the whole prompt, so
    #: the Messages response reports ``input_tokens = prompt_tokens - cached - written``.
    cache_write_tokens: int = 0
    #: Seconds to sleep between streamed chunks (to prove chunks arrive one by one).
    chunk_delay: float = 0.0
    #: Number of content pieces a streamed text reply is split into.
    pieces: int = 3
    #: ``False`` leaves ``usage`` out of the response (a provider that reports none).
    usage: bool = True
    #: ``"text"`` replies with a tool call too: the assistant text goes with the call.
    tool_text: str | None = None
    #: The tool call's id; ``None`` keeps the fixed ``toolu_mock_1`` / ``call_mock_1``. A
    #: scripted agent loop that keeps every call in its history needs distinct ids.
    tool_id: str | None = None


@dataclass
class Received:
    """One request the mock received."""

    path: str
    headers: dict[str, str]
    body: dict[str, Any]


#: Picks the reply from the request itself (for a scripted agent whose answer depends on the
#: conversation so far). Returning ``None`` falls back to the script, then the default.
Responder = Callable[[Received], "Reply | None"]


@dataclass
class _State:
    script: deque[Reply] = field(default_factory=deque)
    default: Reply = field(default_factory=Reply)
    received: list[Received] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    responder: Responder | None = None


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: _State) -> None:
        self.state = state
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        return None

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": []})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        st = self.server.state
        rec = Received(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
        with st.lock:
            st.received.append(rec)
            responder = st.responder
        reply = responder(rec) if responder is not None else None
        if reply is None:
            with st.lock:
                reply = st.script.popleft() if st.script else st.default
        if self.path.rstrip("/").endswith("/v1/messages"):
            return self._anthropic(reply, body)
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        if reply.kind == "error":
            return self._json(
                reply.status,
                {"error": {"message": "scripted failure", "type": "mock", "code": reply.status}},
            )
        model = str(body.get("model", "mock"))
        if body.get("stream"):
            return self._stream(reply, model, body)
        message, finish = _message(reply)
        out: dict[str, Any] = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if reply.usage:
            out["usage"] = _usage(reply)
        self._json(200, out)

    def _stream(self, reply: Reply, model: str, body: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        base = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
        }

        def send(obj: dict[str, Any] | str) -> None:
            text = obj if isinstance(obj, str) else json.dumps(obj)
            self.wfile.write(f"data: {text}\n\n".encode())
            self.wfile.flush()

        def chunk(delta: dict[str, Any], finish: str | None = None) -> None:
            send({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
            if reply.chunk_delay:
                time.sleep(reply.chunk_delay)

        message, finish = _message(reply)
        chunk({"role": "assistant", "content": ""})
        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["function"]["name"], "arguments": ""},
                        }
                    ]
                }
            )
            chunk(
                {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": call["function"]["arguments"]}}
                    ]
                }
            )
        else:
            text = message.get("content") or ""
            n = max(1, reply.pieces)
            step = max(1, -(-len(text) // n)) if text else 1
            for i in range(0, len(text), step):
                chunk({"content": text[i : i + step]})
        chunk({}, finish)
        options = body.get("stream_options") or {}
        if isinstance(options, dict) and options.get("include_usage") and reply.usage:
            send({**base, "choices": [], "usage": _usage(reply)})
        send("[DONE]")

    # --- Anthropic Messages API ---

    def _anthropic(self, reply: Reply, body: dict[str, Any]) -> None:
        if reply.kind == "error":
            kind = {400: "invalid_request_error", 429: "rate_limit_error"}.get(
                reply.status, "api_error"
            )
            return self._json(
                reply.status,
                {"type": "error", "error": {"type": kind, "message": "scripted failure"}},
            )
        model = str(body.get("model", "mock"))
        content, stop = _anthropic_content(reply)
        usage = _anthropic_usage(reply)
        if body.get("stream"):
            return self._anthropic_stream(reply, model, content, stop, usage)
        self._json(
            200,
            {
                "id": "msg_mock",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": content,
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": usage,
            },
        )

    def _anthropic_stream(
        self,
        reply: Reply,
        model: str,
        content: list[dict[str, Any]],
        stop: str,
        usage: dict[str, Any],
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(event: str, data: dict[str, Any]) -> None:
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
            if reply.chunk_delay:
                time.sleep(reply.chunk_delay)

        start_usage = {**usage, "output_tokens": 0}
        send(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": start_usage,
                },
            },
        )
        for index, block in enumerate(content):
            if block["type"] == "tool_use":
                send(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {**block, "input": {}},
                    },
                )
                send(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": reply.tool_args},
                    },
                )
            else:
                send(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text = block["text"]
                n = max(1, reply.pieces)
                step = max(1, -(-len(text) // n)) if text else 1
                for i in range(0, len(text), step):
                    send(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text[i : i + step]},
                        },
                    )
            send("content_block_stop", {"type": "content_block_stop", "index": index})
        send(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
        send("message_stop", {"type": "message_stop"})


def _anthropic_content(reply: Reply) -> tuple[list[dict[str, Any]], str]:
    if reply.kind == "tool":
        try:
            args = json.loads(reply.tool_args)
        except ValueError:
            args = {}
        blocks: list[dict[str, Any]] = []
        if reply.tool_text:
            blocks.append({"type": "text", "text": reply.tool_text})
        blocks.append(
            {
                "type": "tool_use",
                "id": reply.tool_id or "toolu_mock_1",
                "name": reply.tool_name,
                "input": args,
            }
        )
        return blocks, "tool_use"
    if reply.kind == "empty":
        return [], "end_turn"
    if reply.kind == "refusal":
        return [], "refusal"
    return [{"type": "text", "text": reply.content}], "end_turn"


def _anthropic_usage(reply: Reply) -> dict[str, Any]:
    read = reply.cached_tokens or 0
    write = reply.cache_write_tokens
    return {
        "input_tokens": max(0, reply.prompt_tokens - read - write),
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
        "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0},
        "output_tokens": reply.completion_tokens,
    }


def _message(reply: Reply) -> tuple[dict[str, Any], str]:
    if reply.kind == "tool":
        return (
            {
                "role": "assistant",
                "content": reply.tool_text,
                "tool_calls": [
                    {
                        "id": reply.tool_id or "call_mock_1",
                        "type": "function",
                        "function": {"name": reply.tool_name, "arguments": reply.tool_args},
                    }
                ],
            },
            "tool_calls",
        )
    if reply.kind == "empty":
        return {"role": "assistant", "content": ""}, "stop"
    if reply.kind == "refusal":
        return {"role": "assistant", "content": ""}, "content_filter"
    return {"role": "assistant", "content": reply.content}, "stop"


def _usage(reply: Reply) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.completion_tokens,
        "total_tokens": reply.prompt_tokens + reply.completion_tokens,
    }
    if reply.cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": reply.cached_tokens}
    return usage


class MockUpstream:
    """A running mock server. Use as a context manager, or call ``start`` and ``stop``."""

    def __init__(self, default: Reply | None = None) -> None:
        self._state = _State(default=default or Reply())
        self._server = _Server(self._state)
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    @property
    def anthropic_base_url(self) -> str:
        """The ``api_base`` for LiteLLM's ``anthropic`` provider, which appends ``/v1/messages``."""
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> MockUpstream:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> MockUpstream:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def script(self, *replies: Reply) -> None:
        """Queue replies for the next requests, in order."""
        with self._state.lock:
            self._state.script.extend(replies)

    def set_responder(self, responder: Responder | None) -> None:
        """Choose each reply from the request (``None`` switches back to the script)."""
        with self._state.lock:
            self._state.responder = responder

    def set_default(self, reply: Reply) -> None:
        with self._state.lock:
            self._state.default = reply

    @property
    def received(self) -> list[Received]:
        with self._state.lock:
            return list(self._state.received)

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [r.body for r in self.received if r.path.rstrip("/").endswith("/chat/completions")]

    def messages_bodies(self) -> list[dict[str, Any]]:
        """The raw Anthropic Messages requests received, in order."""
        return [r.body for r in self.received if r.path.rstrip("/").endswith("/v1/messages")]

    def clear(self) -> None:
        with self._state.lock:
            self._state.received.clear()
            self._state.script.clear()
