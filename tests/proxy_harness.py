"""Test harness for the proxy: mock upstreams on loopback, an in-process service, a client.

Never reaches a real provider: every upstream is a ``MockUpstream`` on 127.0.0.1, and the
``no_network`` guard refuses any socket connection that is not loopback.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.collectors.mock_upstream import MockUpstream
from steproute.collectors.proxy.config import CheckConfig, ProxyConfig, UpstreamConfig
from steproute.collectors.proxy.engine import Proxy
from steproute.collectors.proxy.server import make_proxy_server
from steproute.collectors.proxy.source import DecisionSource, InProcessSource
from steproute.collectors.proxy.upstream import UpstreamPool
from steproute.core.catalog import Catalog, load_default_catalog
from steproute.core.policy import Policy
from steproute.core.privacy import PrivacyConfig
from steproute.core.rules import RulePolicy
from steproute.core.schema import Action, Decision, Outcome, StepState, new_decision_id
from steproute.core.service import Service
from steproute.core.store import TraceStore

LOOPBACK = {"127.0.0.1", "::1", "localhost"}
CLOUD_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "deepseek-flash")
LOCAL_MODELS = ("qwen3.5:9b", "gpt-oss:20b", "qwen3-coder:30b")


class NetworkGuard:
    """Refuses (and records) every non-loopback connect and name lookup."""

    def __init__(self) -> None:
        self.blocked: list[str] = []

    def install(self, monkeypatch: Any) -> None:
        orig_connect = socket.socket.connect
        orig_gai = socket.getaddrinfo
        guard = self

        def connect(sock: socket.socket, address: Any) -> Any:
            host = address[0] if isinstance(address, tuple) else str(address)
            if isinstance(address, tuple) and host not in LOOPBACK:
                guard.blocked.append(host)
                raise OSError(f"test guard: non-loopback connect to {host} refused")
            return orig_connect(sock, address)

        def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
            if host is not None and str(host) not in LOOPBACK:
                guard.blocked.append(str(host))
                raise OSError(f"test guard: lookup of {host} refused")
            return orig_gai(host, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


class ScriptedPolicy:
    """Returns the given single-step actions in order (the last one repeats)."""

    name = "scripted"

    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions
        self.seen: list[StepState] = []

    def decide(self, state: StepState) -> Decision:
        self.seen.append(state)
        a = self.actions[min(len(self.seen) - 1, len(self.actions) - 1)]
        return Decision(new_decision_id(), time.time(), state, [a], a, self.name, 1.0)

    def report(self, outcome: Outcome) -> None:
        return None


class BrokenSource:
    """A decision source that always fails (the service is down)."""

    name = "broken"

    def decide(self, state: StepState) -> Decision:
        from steproute.collectors.proxy.source import SourceError

        raise SourceError("unreachable:test")

    def report(self, outcome: Outcome) -> Outcome:
        from steproute.collectors.proxy.source import SourceError

        raise SourceError("unreachable:test")


@dataclass
class Reply:
    status: int
    headers: dict[str, str]
    body: Any
    events: list[tuple[float, Any]] = field(default_factory=list)


@dataclass
class Harness:
    proxy: Proxy
    cloud: MockUpstream
    local: MockUpstream
    store: TraceStore
    port: int
    service: Service | None
    #: Anthropic Messages-format mock; serves the models passed as ``anthropic_models``.
    anthropic: MockUpstream | None = None

    def post(
        self,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
        path: str = "/v1/chat/completions",
    ) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        data = json.dumps(body).encode()
        h = {"Content-Type": "application/json", "Content-Length": str(len(data))}
        h.update(headers or {})
        conn.request("POST", path, body=data, headers=h)
        resp = conn.getresponse()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        if hdrs.get("content-type", "").startswith("text/event-stream"):
            events: list[tuple[float, Any]] = []
            buf = b""
            while True:
                piece = resp.read1(65536) if hasattr(resp, "read1") else resp.read(1)
                if not piece:
                    break
                buf += piece
                while b"\n\n" in buf:
                    raw, buf = buf.split(b"\n\n", 1)
                    text = raw.decode().removeprefix("data: ")
                    events.append(
                        (time.monotonic(), text if text == "[DONE]" else json.loads(text))
                    )
            conn.close()
            return Reply(resp.status, hdrs, None, events)
        raw = resp.read()
        conn.close()
        return Reply(resp.status, hdrs, json.loads(raw) if raw else None)

    def get(self, path: str) -> Reply:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        conn.request("GET", path)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        return Reply(resp.status, {k.lower(): v for k, v in resp.getheaders()}, json.loads(raw))

    def finished(self, n: int = 1, timeout: float = 5.0) -> None:
        """Wait until ``n`` outcomes are in the store (reporting runs after the response)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sum(1 for r in self.store.records() if isinstance(r, Outcome)) >= n:
                return
            time.sleep(0.01)
        raise AssertionError(f"expected {n} outcomes in the store")

    def decisions(self) -> list[Decision]:
        return [r for r in self.store.records() if isinstance(r, Decision)]

    def outcomes(self) -> list[Outcome]:
        return [r for r in self.store.records() if isinstance(r, Outcome)]


def upstreams(
    cloud: MockUpstream,
    local: MockUpstream,
    anthropic: MockUpstream | None = None,
    anthropic_models: tuple[str, ...] = (),
    anthropic_extra: dict[str, Any] | None = None,
    **cloud_extra: Any,
) -> tuple[UpstreamConfig, ...]:
    """The cloud models on an OpenAI-format mock, except ``anthropic_models``, which go to an
    Anthropic-format mock through LiteLLM's ``anthropic`` provider. The local upstream has no
    ``reasoning_values``: the catalog's values go out as they are."""
    out = [
        UpstreamConfig(
            name="cloud",
            provider="openai",
            base_url=cloud.base_url,
            models=tuple(m for m in CLOUD_MODELS if m not in anthropic_models),
            local=False,
            drop_params=("temperature", "top_p", "top_k"),
            **cloud_extra,
        ),
        UpstreamConfig(
            name="local",
            provider="openai",
            base_url=local.base_url,
            models=LOCAL_MODELS,
            local=True,
        ),
    ]
    if anthropic is not None and anthropic_models:
        out.append(
            UpstreamConfig(
                name="anthropic",
                provider="anthropic",
                base_url=anthropic.anthropic_base_url,
                models=anthropic_models,
                drop_params=("temperature", "top_p", "top_k"),
                forced_tool_choice="auto",
                **(anthropic_extra or {}),
            )
        )
    return tuple(out)


@dataclass
class Running:
    harness: Harness
    stop: Any


def start(
    tmp_path: Path,
    *,
    policy: Policy | None = None,
    mode: str = "route",
    source: DecisionSource | None = None,
    checks: CheckConfig | None = None,
    local_fallback: str | None = "qwen3-coder:30b",
    catalog: Catalog | None = None,
    cloud_extra: dict[str, Any] | None = None,
    anthropic_models: tuple[str, ...] = (),
    anthropic_extra: dict[str, Any] | None = None,
    **config_extra: Any,
) -> Iterator[Harness]:
    """Generator fixture body: yields a running harness, then shuts everything down."""
    catalog = catalog or load_default_catalog()
    privacy = PrivacyConfig()
    store = TraceStore(tmp_path / "traces", privacy=privacy)
    service: Service | None = None
    if source is None:
        service = Service(
            policy=policy or RulePolicy(catalog, privacy=privacy),
            catalog=catalog,
            store=store,
            privacy=privacy,
            default_action=None,
            local_fallback=Action("qwen3-coder:30b", "medium"),
            budget_ms=2000,
        )
        source = InProcessSource(service)
    with MockUpstream() as cloud, MockUpstream() as local, MockUpstream() as anthropic:
        config = ProxyConfig(
            mode=mode,
            upstreams=upstreams(
                cloud, local, anthropic, anthropic_models, anthropic_extra, **(cloud_extra or {})
            ),
            checks=checks or CheckConfig(),
            local_fallback=local_fallback,
            **config_extra,
        )
        proxy = Proxy(
            config, source=source, catalog=catalog, privacy=privacy, pool=UpstreamPool({})
        )
        server = make_proxy_server(proxy, None, host="127.0.0.1", port=0)
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        try:
            yield Harness(proxy, cloud, local, store, server.server_address[1], service, anthropic)
        finally:
            server.shutdown()
            server.server_close()


def chat(
    model: str = "qwen3-coder:30b", content: str = "Fix the failing test.", **extra: Any
) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}
