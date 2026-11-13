"""A scriptable mock of an OpenAI-compatible chat-completions server, on loopback only.

Standard library only. Used by the proxy tests (and usable by the replay runner's CI) so no
test ever reaches a real provider. Each request pops the next scripted ``Reply`` (or uses the
default one), and every request is recorded so a test can assert what the proxy sent: the
rewritten model, the reasoning parameter, the dropped sampling parameters.

Supported: ``POST /v1/chat/completions`` (non-streaming JSON and SSE streaming, with a final
usage chunk when ``stream_options.include_usage`` is set) and ``GET /v1/models``.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
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
    #: Seconds to sleep between streamed chunks (to prove chunks arrive one by one).
    chunk_delay: float = 0.0
    #: Number of content pieces a streamed text reply is split into.
    pieces: int = 3


@dataclass
class Received:
    """One request the mock received."""

    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class _State:
    script: deque[Reply] = field(default_factory=deque)
    default: Reply = field(default_factory=Reply)
    received: list[Received] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


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
        with st.lock:
            st.received.append(
                Received(self.path, {k.lower(): v for k, v in self.headers.items()}, body)
            )
            reply = st.script.popleft() if st.script else st.default
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
        self._json(
            200,
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": _usage(reply),
            },
        )

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
        if isinstance(options, dict) and options.get("include_usage"):
            send({**base, "choices": [], "usage": _usage(reply)})
        send("[DONE]")


def _message(reply: Reply) -> tuple[dict[str, Any], str]:
    if reply.kind == "tool":
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock_1",
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

    def set_default(self, reply: Reply) -> None:
        with self._state.lock:
            self._state.default = reply

    @property
    def received(self) -> list[Received]:
        with self._state.lock:
            return list(self._state.received)

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [r.body for r in self.received if r.path.rstrip("/").endswith("/chat/completions")]

    def clear(self) -> None:
        with self._state.lock:
            self._state.received.clear()
            self._state.script.clear()
