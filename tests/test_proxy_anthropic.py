"""The proxy's Anthropic path against an Anthropic Messages-format mock (issue 46).

Covers the prompt-cache breakpoints the proxy sends (and does not send), how Anthropic's
cache-read and cache-write counts land in the ``Outcome`` and its cost, the shadow-mode and
client-marker rules, and the catalog's Ollama reasoning values going out without an override.
Every upstream is a mock on 127.0.0.1; ``guard`` fails the test on any other connection.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning:pydantic._internal._generate_schema"),
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince211"),
]

from proxy_harness import Harness, NetworkGuard, ScriptedPolicy, start  # noqa: E402
from steproute.collectors.mock_upstream import Reply  # noqa: E402
from steproute.collectors.proxy.config import (  # noqa: E402
    ProxyConfigError,
    UpstreamConfig,
    load_proxy_config,
)
from steproute.collectors.proxy.upstream import (  # noqa: E402
    UpstreamPool,
    load_litellm,
    with_cache_breakpoints,
)
from steproute.core.catalog import load_default_catalog  # noqa: E402
from steproute.core.cost import AttemptUsage, price_usage  # noqa: E402
from steproute.core.schema import Action, Outcome  # noqa: E402

CATALOG = load_default_catalog()
SONNET = "claude-sonnet-5-5"
EPHEMERAL = {"type": "ephemeral"}
SYSTEM = "You are a coding agent. " * 40
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }
]


@pytest.fixture(autouse=True)
def guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkGuard]:
    g = NetworkGuard()
    g.install(monkeypatch)
    yield g
    assert not g.blocked, f"a test tried to reach the network: {g.blocked}"


def _anthropic(tmp_path: Path, **kw: Any) -> Iterator[Harness]:
    kw.setdefault("policy", ScriptedPolicy([Action(SONNET, "high")]))
    return start(tmp_path, anthropic_models=(SONNET,), **kw)


def _request(messages: list[dict[str, Any]], model: str = SONNET, **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": messages, "tools": TOOLS, **extra}


def _opening() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Fix the failing test in tests/test_app.py."},
    ]


def _tokens(o: Outcome) -> tuple[int, int, int, int]:
    """(input, cache read, cache write, output) of an outcome."""
    return (o.input_tokens, o.cache_read_tokens, o.cache_write_tokens, o.output_tokens)


def _markers(value: Any) -> int:
    """Number of ``cache_control`` keys anywhere in a JSON value."""
    if isinstance(value, dict):
        return ("cache_control" in value) + sum(_markers(v) for v in value.values())
    if isinstance(value, list):
        return sum(_markers(v) for v in value)
    return 0


# --- criterion 1: breakpoints present, and absent when the option is off --------------------


def test_anthropic_request_carries_cache_breakpoints(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        r = h.post(_request(_opening(), temperature=0.3), {"X-Steproute-Session": "a"})
        assert r.status == 200, r.body
        (sent,) = h.anthropic.messages_bodies()
        assert not h.cloud.chat_bodies()
        assert sent["model"] == SONNET
        assert sent["output_config"] == {"effort": "high"}  # the decision's effort
        assert sent["thinking"]["type"] == "adaptive"
        assert "temperature" not in sent
        # Breakpoint 1: the end of the static prefix (tools render before system).
        assert sent["system"] == [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}]
        # Breakpoint 2: request-level automatic caching, the moving end of the conversation.
        assert sent["cache_control"] == EPHEMERAL
        # Nothing else is marked: two of the four allowed breakpoints.
        assert _markers(sent) == 2
        assert _markers(sent["messages"]) == 0 and _markers(sent["tools"]) == 0
        h.finished()
        assert h.outcomes()[0].signals["cache_markers"] == "proxy"


def test_anthropic_request_carries_no_breakpoint_when_the_option_is_off(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path, anthropic_extra={"cache_breakpoints": "off"}):
        assert h.anthropic is not None
        h.anthropic.set_default(Reply(prompt_tokens=800, completion_tokens=5))
        assert h.post(_request(_opening()), {"X-Steproute-Session": "a"}).status == 200
        (sent,) = h.anthropic.messages_bodies()
        assert _markers(sent) == 0 and "cache_control" not in json.dumps(sent)
        assert sent["output_config"] == {"effort": "high"}
        h.finished()
        assert "cache_markers" not in h.outcomes()[0].signals
        # Nothing was asked to be cached, so the session believes nothing is.
        s = h.proxy.sessions.get("a")
        assert s is not None and s.owner_model == SONNET and s.cached_tokens == 0


def test_without_a_system_prompt_the_last_tool_is_the_static_breakpoint(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        h.post(_request([{"role": "user", "content": "hi"}]))
        (sent,) = h.anthropic.messages_bodies()
        assert sent["tools"][-1]["cache_control"] == EPHEMERAL
        assert sent["cache_control"] == EPHEMERAL and _markers(sent) == 2


# --- criterion 2: a session writes on step 1 and reads on step 2 ----------------------------


def test_session_writes_the_cache_then_reads_it_and_costs_match(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        # Step 1 writes the whole prompt; step 2 reads it back and writes the new remainder.
        h.anthropic.script(
            Reply(
                prompt_tokens=3000, completion_tokens=40, cached_tokens=0, cache_write_tokens=3000
            ),
            Reply(
                prompt_tokens=3600, completion_tokens=30, cached_tokens=3000, cache_write_tokens=600
            ),
        )
        messages = _opening()
        r1 = h.post(_request(messages), {"X-Steproute-Session": "warm"})
        assert r1.status == 200
        messages = [
            *messages,
            r1.body["choices"][0]["message"],
            {"role": "user", "content": "Now run it."},
        ]
        r2 = h.post(_request(messages), {"X-Steproute-Session": "warm"})
        assert r2.status == 200
        h.finished(2)
        b1, b2 = h.anthropic.messages_bodies()
        # The breakpoints are in the same places on both steps, and step 1's prompt is a
        # byte-identical prefix of step 2's, so step 2's automatic breakpoint finds step 1's.
        for b in (b1, b2):
            assert b["cache_control"] == EPHEMERAL and b["system"][-1]["cache_control"] == EPHEMERAL
            assert _markers(b) == 2
        assert b1["system"] == b2["system"] and b1["tools"] == b2["tools"]
        assert b2["messages"][: len(b1["messages"])] == b1["messages"]

        d1, d2 = h.decisions()
        by_id = {o.decision_id: o for o in h.outcomes()}
        o1, o2 = by_id[d1.decision_id], by_id[d2.decision_id]
        assert _tokens(o1) == (0, 0, 3000, 40)  # step 1: cold, everything written
        assert _tokens(o2) == (0, 3000, 600, 30)  # step 2: prefix read, remainder written
        sonnet = CATALOG.get(SONNET)
        assert o1.cost_usd == pytest.approx(
            price_usage(sonnet, AttemptUsage(0, 0, 3000, 40)).total_usd
        )
        assert o2.cost_usd == pytest.approx(
            price_usage(sonnet, AttemptUsage(0, 3000, 600, 30)).total_usd
        )
        # 3000 * 2.50 + 40 * 10 per MTok; 3000 * 0.10 + 600 * 2.50 + 30 * 10 per MTok.
        assert o1.cost_usd == pytest.approx(0.0079)
        assert o2.cost_usd == pytest.approx(0.0021)
        # What the proxy believed before step 2 matches what the provider then reported.
        assert d2.state.current_model == SONNET and d2.state.current_effort == "high"
        assert d2.state.cached_prefix_tokens == 3000 == o2.cache_read_tokens


def test_streaming_against_the_anthropic_mock_reports_cache_counts(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        reply = Reply(content="streamed", prompt_tokens=2000, completion_tokens=12)
        h.anthropic.set_default(replace(reply, cached_tokens=1500, cache_write_tokens=400))
        r = h.post(_request(_opening(), stream=True))
        assert r.status == 200
        text = "".join(
            e["choices"][0]["delta"].get("content") or ""
            for _, e in r.events
            if isinstance(e, dict) and e.get("choices")
        )
        assert text == "streamed"
        (sent,) = h.anthropic.messages_bodies()
        assert sent["stream"] is True and sent["cache_control"] == EPHEMERAL
        h.finished()
        (o,) = h.outcomes()
        assert _tokens(o) == (100, 1500, 400, 12)


def test_anthropic_tool_call_refusal_and_error_map_back(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        h.anthropic.script(
            Reply(kind="tool", tool_name="read_file", tool_args='{"path": "a.py"}'),
            Reply(kind="refusal"),
            Reply(kind="error", status=529),
        )
        r = h.post(_request(_opening()))
        call = r.body["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "read_file"
        assert json.loads(call["function"]["arguments"]) == {"path": "a.py"}
        r = h.post(_request(_opening()))
        assert r.body["choices"][0]["finish_reason"] == "content_filter"
        r = h.post(_request(_opening()))
        # The routed attempt failed; the client's own model is the same upstream, so the
        # fallback also hits the mock's default reply.
        assert r.status == 200
        h.finished(3)


# --- shadow mode and the client's own markers ----------------------------------------------


def test_shadow_mode_sends_no_breakpoint_by_default(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path, mode="shadow"):
        assert h.anthropic is not None
        h.anthropic.set_default(Reply(prompt_tokens=900, cache_write_tokens=0))
        body = _request(_opening())
        assert h.post(body, {"X-Steproute-Session": "sh"}).status == 200
        (sent,) = h.anthropic.messages_bodies()
        assert _markers(sent) == 0
        h.finished()
        s = h.proxy.sessions.get("sh")
        assert s is not None and s.cached_tokens == 0  # nothing was asked to be cached


def test_shadow_mode_with_always_marks_the_unchanged_request(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path, mode="shadow", anthropic_extra={"cache_breakpoints": "always"}):
        assert h.anthropic is not None
        h.anthropic.set_default(Reply(prompt_tokens=900, cache_write_tokens=900))
        assert h.post(_request(_opening()), {"X-Steproute-Session": "sh"}).status == 200
        (sent,) = h.anthropic.messages_bodies()
        assert sent["cache_control"] == EPHEMERAL and _markers(sent) == 2
        h.finished()
        s = h.proxy.sessions.get("sh")
        assert s is not None and s.cached_tokens == 900


def test_client_markers_are_left_alone(tmp_path: Path) -> None:
    for h in _anthropic(tmp_path):
        assert h.anthropic is not None
        messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Context.", "cache_control": EPHEMERAL},
                    {"type": "text", "text": "Fix the test."},
                ],
            },
        ]
        assert h.post(_request(messages)).status == 200
        (sent,) = h.anthropic.messages_bodies()
        assert "cache_control" not in sent  # no request-level breakpoint added
        assert "cache_control" not in sent["system"][-1]  # no system breakpoint added
        assert sent["messages"][0]["content"][0]["cache_control"] == EPHEMERAL
        assert _markers(sent) == 1
        h.finished()
        assert h.outcomes()[0].signals["cache_markers"] == "client"


def test_with_cache_breakpoints_does_not_mutate_the_request() -> None:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        },
        {"role": "system", "content": "c"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "mid-conversation, not the static prefix"},
    ]
    before = json.dumps(messages)
    out, who = with_cache_breakpoints({"messages": messages, "tools": TOOLS})
    assert who == "proxy" and json.dumps(messages) == before
    assert out["messages"][1] == {"role": "system", "content": "c", "cache_control": EPHEMERAL}
    assert _markers(out["messages"]) == 1 and _markers(out["tools"]) == 0
    # A list-content system message gets the marker on its last text block.
    out, _ = with_cache_breakpoints({"messages": messages[:1] + messages[2:3]})
    assert out["messages"][0]["content"][1]["cache_control"] == EPHEMERAL
    assert "cache_control" not in out["messages"][0]["content"][0]


def test_litellm_never_injects_markers_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    litellm = load_litellm()
    assert litellm.enable_anthropic_prompt_caching is False
    up = UpstreamConfig(name="a", provider="anthropic", models=(SONNET,), cache_breakpoints="off")
    p = UpstreamPool({}).prepare(up, SONNET, {"messages": _opening()}, rewrite=True, reasoning=None)
    assert p.cache == "none" and not p.cacheable and "cache_control" not in p.kwargs


# --- configuration --------------------------------------------------------------------------


def test_cache_breakpoint_config_rules() -> None:
    def up(provider: str, **kw: Any) -> UpstreamConfig:
        base = "http://127.0.0.1:1/v1" if provider == "openai" else None
        return UpstreamConfig(name="u", provider=provider, models=("m",), base_url=base, **kw)

    assert up("anthropic").cache_breakpoints == "route"
    assert up("deepseek").cache_breakpoints == "off"
    assert up("openai").cache_breakpoints == "off"
    with pytest.raises(ProxyConfigError, match="only for provider 'anthropic'"):
        up("deepseek", cache_breakpoints="route")
    with pytest.raises(ProxyConfigError, match="cache_breakpoints must be one of"):
        up("anthropic", cache_breakpoints="yes")
    # The proxy requests the 5-minute cache, so it may not assume a longer one.
    with pytest.raises(ProxyConfigError, match="5-minute cache"):
        up("anthropic", cache_ttl_s=3600)
    assert up("anthropic", cache_ttl_s=3600, cache_breakpoints="off").cache_ttl_s == 3600
    cfg = load_proxy_config(
        {
            "upstream": [
                {
                    "name": "a",
                    "provider": "anthropic",
                    "models": [SONNET],
                    "cache_breakpoints": "always",
                },
            ]
        }
    )
    assert cfg.upstreams[0].cache_breakpoints == "always"


# --- criterion 3: the catalog's Ollama values go out without an override --------------------


@pytest.mark.parametrize(("effort", "sent"), [("none", "none"), ("medium", "medium")])
def test_qwen35_catalog_values_reach_ollama_v1_unchanged(
    tmp_path: Path, effort: str, sent: str
) -> None:
    profile = CATALOG.get("qwen3.5:9b")
    assert profile.effort_map == {"none": "none", "medium": "medium"}
    policy = ScriptedPolicy([Action("qwen3.5:9b", effort)])
    for h in start(tmp_path, policy=policy):
        local = next(u for u in h.proxy.config.upstreams if u.local)
        assert not local.reasoning_values  # no per-upstream override
        assert h.post({"model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}]})
        (body,) = h.local.chat_bodies()
        assert body["model"] == "qwen3.5:9b" and body["reasoning_effort"] == sent
