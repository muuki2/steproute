"""Integration tests for the proxy against mock upstreams on loopback (issue 12).

Covers route, cascade on each check, shadow mode, streaming, the 5-step session cache state,
and the privacy rule end to end (issue 16, criterion 3). No test reaches a real provider:
every upstream is a mock on 127.0.0.1 and ``guard`` refuses any other connection.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

# LiteLLM's pydantic models warn about ReadOnly TypedDict items when first imported.
pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning:pydantic._internal._generate_schema"),
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince211"),
]

from proxy_harness import (  # noqa: E402
    BrokenSource,
    Harness,
    NetworkGuard,
    ScriptedPolicy,
    chat,
    start,
)
from steproute.collectors.mock_upstream import Reply  # noqa: E402
from steproute.collectors.proxy.checks import normalise_usage  # noqa: E402
from steproute.collectors.proxy.engine import SHADOW_PREFIX  # noqa: E402
from steproute.collectors.proxy.session import split_prompt  # noqa: E402
from steproute.collectors.proxy.source import HttpSource  # noqa: E402
from steproute.core.catalog import load_default_catalog  # noqa: E402
from steproute.core.cost import AttemptUsage, price_usage  # noqa: E402
from steproute.core.schema import Action  # noqa: E402

CATALOG = load_default_catalog()


@pytest.fixture(autouse=True)
def guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[NetworkGuard]:
    g = NetworkGuard()
    g.install(monkeypatch)
    yield g
    assert not g.blocked, f"a test tried to reach the network: {g.blocked}"


@pytest.fixture
def h(tmp_path: Path) -> Iterator[Harness]:
    yield from start(tmp_path)


def _cost(model: str, usage: AttemptUsage) -> float:
    return price_usage(CATALOG.get(model), usage).total_usd


# --- route ----------------------------------------------------------------------------------


def test_route_rewrites_model_and_effort_and_logs_a_joined_trace(h: Harness) -> None:
    h.cloud.set_default(
        Reply(content="done", prompt_tokens=1000, completion_tokens=50, cached_tokens=400)
    )
    r = h.post(
        chat(temperature=0.2, top_p=0.5, reasoning_effort="low"), {"X-Steproute-Session": "s1"}
    )
    assert r.status == 200
    assert r.body["choices"][0]["message"]["content"] == "done"
    assert r.body["model"] == "qwen3-coder:30b"  # the client's name is echoed
    assert r.headers["x-steproute-model"] == "claude-sonnet-5-5"
    assert r.headers["x-steproute-effort"] == "medium"
    assert r.headers["x-steproute-mode"] == "route"
    sent = h.cloud.chat_bodies()
    assert len(sent) == 1 and not h.local.chat_bodies()
    assert sent[0]["model"] == "claude-sonnet-5-5"
    assert sent[0]["reasoning_effort"] == "medium"  # catalog effort_map, client's "low" replaced
    assert "temperature" not in sent[0] and "top_p" not in sent[0]  # the upstream's drop_params
    h.finished()
    (d,) = h.decisions()
    (o,) = h.outcomes()
    assert o.decision_id == d.decision_id == r.headers["x-steproute-decision-id"]
    assert d.state.harness == "proxy" and d.state.session_id == "s1" and d.state.step_idx == 0
    assert d.state.features["proxy"]["mode"] == "route"
    assert d.state.task_text is None and "task_text_len" not in d.state.features
    # Token convention: prompt_tokens includes the cached ones.
    assert (o.input_tokens, o.cache_read_tokens, o.cache_write_tokens, o.output_tokens) == (
        600,
        400,
        0,
        50,
    )
    assert o.cost_usd == pytest.approx(_cost("claude-sonnet-5-5", AttemptUsage(600, 400, 0, 50)))
    assert o.final_model == "claude-sonnet-5-5" and not o.escalated
    assert o.signals["attempts"] == 1
    assert o.signals["params_dropped"] == ["temperature", "top_p"]


def test_unknown_client_params_are_not_forwarded(h: Harness) -> None:
    r = h.post(chat(api_base="http://evil.example", extra_headers={"x": "y"}, metadata={"a": 1}))
    assert r.status == 200
    sent = h.cloud.chat_bodies()[0]
    for key in ("api_base", "extra_headers", "metadata"):
        assert key not in sent
    h.finished()
    assert h.decisions()[0].state.features["proxy"]["dropped_client_params"] == 3


def test_forced_tool_choice_is_downgraded_when_the_upstream_says_so(tmp_path: Path) -> None:
    for harness in start(tmp_path, cloud_extra={"forced_tool_choice": "auto"}):
        tools = [
            {
                "type": "function",
                "function": {"name": "read_file", "parameters": {"type": "object"}},
            }
        ]
        r = harness.post(chat(tools=tools, tool_choice="required"))
        assert r.status == 200
        assert harness.cloud.chat_bodies()[0]["tool_choice"] == "auto"


# --- cascade --------------------------------------------------------------------------------


def test_cascade_on_empty_output(h: Harness) -> None:
    h.cloud.script(
        Reply(kind="empty", prompt_tokens=800),
        Reply(content="fixed", prompt_tokens=800, cached_tokens=0),
    )
    r = h.post(chat(), {"X-Steproute-Session": "c1"})
    assert r.status == 200 and r.body["choices"][0]["message"]["content"] == "fixed"
    sent = h.cloud.chat_bodies()
    # Rule baseline: sonnet medium -> sonnet high -> opus low.
    assert [(b["model"], b["reasoning_effort"]) for b in sent] == [
        ("claude-sonnet-5-5", "medium"),
        ("claude-sonnet-5-5", "high"),
    ]
    assert r.headers["x-steproute-attempts"] == "2"
    h.finished()
    (o,) = h.outcomes()
    assert o.escalated and o.final_model == "claude-sonnet-5-5"
    assert o.signals["failed_checks"] == ["empty_output"]
    one = AttemptUsage(800, 0, 0, 10)
    assert o.cost_usd == pytest.approx(2 * _cost("claude-sonnet-5-5", one))
    assert o.input_tokens == 1600 and o.success is None  # a passing final answer is no evidence
    s = h.proxy.sessions.get("c1")
    assert s is not None and (s.owner_model, s.owner_effort, s.cached_tokens) == (
        "claude-sonnet-5-5",
        "high",
        800,
    )


def test_cascade_on_malformed_tool_call_then_labels_failure(h: Harness) -> None:
    bad = Reply(kind="tool", tool_args="{not json")
    h.cloud.script(bad, bad, bad)
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    ]
    r = h.post(chat(tools=tools))
    assert r.status == 200  # the last answer is returned: the client decides what to do
    assert [b["model"] for b in h.cloud.chat_bodies()] == [
        "claude-sonnet-5-5",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
    ]
    h.finished()
    (o,) = h.outcomes()
    assert o.signals["failed_checks"] == ["tool_call_parse"] * 3
    assert o.signals["tool_error"] is True and o.success is False
    assert o.final_model == "claude-opus-5-5"


def test_cascade_on_schema_validation(h: Harness) -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    h.cloud.script(
        Reply(content="not json"), Reply(content='{"n": "x"}'), Reply(content='{"n": 3}')
    )
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
    r = h.post(chat(response_format=fmt))
    assert r.body["choices"][0]["message"]["content"] == '{"n": 3}'
    h.finished()
    (o,) = h.outcomes()
    assert o.signals["failed_checks"] == ["schema", "schema"]
    assert o.signals["schema_valid"] is True and o.success is True


def test_cascade_on_upstream_error_then_falls_back_to_the_client_model(h: Harness) -> None:
    h.cloud.set_default(Reply(kind="error", status=400))
    h.local.set_default(Reply(content="local answer"))
    r = h.post(chat("qwen3-coder:30b", temperature=0.3))
    assert r.status == 200 and r.body["choices"][0]["message"]["content"] == "local answer"
    assert len(h.cloud.chat_bodies()) == 3  # the whole chain was tried
    (passthrough,) = h.local.chat_bodies()
    assert passthrough["model"] == "qwen3-coder:30b" and passthrough["temperature"] == 0.3
    h.finished()
    (o,) = h.outcomes()
    assert o.signals["fallback_to_client"] is True and o.signals["upstream_errors"] == 3
    assert o.success is None  # the client's model standing in is not evidence for the chain


def test_upstream_errors_without_fallback_return_the_error(tmp_path: Path) -> None:
    for harness in start(tmp_path, on_upstream_error="error"):
        harness.cloud.set_default(Reply(kind="error", status=429))
        r = harness.post(chat())
        assert r.status == 429 and r.body["error"]["type"] == "upstream_error"
        assert not harness.local.chat_bodies()


def test_refusal_is_recorded_but_does_not_cascade_by_default(h: Harness) -> None:
    h.cloud.set_default(Reply(kind="refusal"))
    r = h.post(chat())
    assert r.status == 200 and len(h.cloud.chat_bodies()) == 1
    h.finished()
    (o,) = h.outcomes()
    assert o.signals["refusal"] is True and o.success is None


# --- shadow ---------------------------------------------------------------------------------


def test_shadow_mode_runs_the_client_request_unchanged_and_never_joins(tmp_path: Path) -> None:
    for h in start(tmp_path, mode="shadow"):
        body = chat("qwen3-coder:30b", temperature=0.7, reasoning_effort="high")
        r = h.post(body, {"X-Steproute-Session": "sh"})
        assert r.status == 200 and r.headers["x-steproute-mode"] == "shadow"
        assert not h.cloud.chat_bodies()
        (sent,) = h.local.chat_bodies()
        assert sent["model"] == "qwen3-coder:30b"
        assert sent["temperature"] == 0.7 and sent["reasoning_effort"] == "high"
        h.finished()
        (d,) = h.decisions()
        (o,) = h.outcomes()
        # The recommendation is logged and marked; the executed call's outcome is under a
        # different id, so no join attaches the client model's result to the router's action.
        assert d.state.features["proxy"]["mode"] == "shadow"
        assert d.action.model == "claude-sonnet-5-5"
        assert (
            o.decision_id == SHADOW_PREFIX + d.decision_id == r.headers["x-steproute-decision-id"]
        )
        assert o.final_model == "qwen3-coder:30b"
        assert all(out is None for _, out in h.store.joined())


# --- streaming ------------------------------------------------------------------------------


def test_streaming_passes_chunks_through_as_they_arrive(h: Harness) -> None:
    h.cloud.set_default(
        Reply(content="abcdef", pieces=3, chunk_delay=0.25, prompt_tokens=500, cached_tokens=100)
    )
    r = h.post(chat(stream=True))
    assert r.status == 200 and r.headers["transfer-encoding"] == "chunked"
    texts = [e for _, e in r.events if e != "[DONE]"]
    content = "".join(
        (c["choices"][0]["delta"].get("content") or "") for c in texts if c["choices"]
    )
    assert content == "abcdef"
    assert r.events[-1][1] == "[DONE]"
    assert all(c["choices"] for c in texts)  # usage-only chunk not forwarded (not requested)
    assert all(c["model"] == "qwen3-coder:30b" for c in texts)
    # Unbuffered: the first content arrived long before the last one was produced.
    times = [
        t
        for t, e in r.events
        if e != "[DONE]" and e["choices"] and e["choices"][0]["delta"].get("content")
    ]
    assert times[-1] - times[0] >= 0.4
    sent = h.cloud.chat_bodies()[0]
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}
    h.finished()
    (o,) = h.outcomes()
    assert (o.input_tokens, o.cache_read_tokens, o.output_tokens) == (400, 100, 10)
    assert o.signals["stream"] is True and o.signals["attempts"] == 1


def test_streaming_forwards_usage_when_the_client_asks(h: Harness) -> None:
    r = h.post(chat(stream=True, stream_options={"include_usage": True}))
    usage = [e for _, e in r.events if e != "[DONE]" and e.get("usage")]
    assert usage and usage[-1]["usage"]["prompt_tokens"] == 100


def test_streaming_has_no_check_cascade(h: Harness) -> None:
    h.cloud.set_default(Reply(kind="empty"))
    r = h.post(chat(stream=True))
    assert r.status == 200 and len(h.cloud.chat_bodies()) == 1
    h.finished()
    (o,) = h.outcomes()
    assert o.signals["failed_checks"] == ["empty_output"] and o.signals["schema_valid"] is False
    assert not o.escalated and o.success is False


def test_streaming_error_before_first_byte_tries_the_next_step(h: Harness) -> None:
    h.cloud.script(Reply(kind="error", status=503), Reply(content="second"))
    r = h.post(chat(stream=True))
    assert r.status == 200
    assert [b["reasoning_effort"] for b in h.cloud.chat_bodies()] == ["medium", "high"]
    h.finished()
    assert h.outcomes()[0].signals["upstream_errors"] == 1


def test_streaming_tool_call_is_checked_after_the_stream(h: Harness) -> None:
    h.cloud.set_default(Reply(kind="tool", tool_args='{"path": "a.py"}'))
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}
    ]
    r = h.post(chat(stream=True, tools=tools))
    deltas = [c["choices"][0]["delta"] for _, c in r.events if c != "[DONE]" and c["choices"]]
    assert any(d.get("tool_calls") for d in deltas)
    h.finished()
    assert "failed_checks" not in h.outcomes()[0].signals


# --- session state --------------------------------------------------------------------------


def test_five_step_session_switching_model_once(tmp_path: Path) -> None:
    """Acceptance criterion 3: exact state at every step of a 5-step session."""
    plan = [
        Action("qwen3-coder:30b", "medium"),
        Action("qwen3-coder:30b", "medium"),
        Action("gpt-oss:20b", "high"),  # the one switch
        Action("gpt-oss:20b", "high"),
        Action("gpt-oss:20b", "high"),
    ]
    policy = ScriptedPolicy(plan)
    actual = [1200, 1500, 1800, 2100, 2400]  # prompt tokens the provider reports per step
    cached = [0, 1200, 0, 1800, 2100]  # cached tokens it reports (cold after the switch)
    for h in start(tmp_path, policy=policy):
        h.local.script(
            *[
                Reply(content=f"answer {i}", prompt_tokens=p, cached_tokens=c)
                for i, (p, c) in enumerate(zip(actual, cached, strict=True))
            ]
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a coding agent. " * 200},
            {"role": "user", "content": "Step 0: read the repository."},
        ]
        expected = []
        cpt = 4.0
        prev = split_prompt(messages, None)
        for i in range(5):
            prompt = split_prompt(messages, None)
            if i == 0:
                est = prompt.est_tokens(0, cpt)
            else:
                # The previous prompt is a prefix: its real size plus the estimate of the rest.
                est = actual[i - 1] + prompt.est_tokens(len(prev.units), cpt)
            r = h.post(
                {"model": "qwen3-coder:30b", "messages": messages}, {"X-Steproute-Session": "five"}
            )
            assert r.status == 200
            text_tokens = actual[i] - 4 * len(prompt.units)
            cpt = min(20.0, max(0.5, prompt.total_chars / text_tokens))
            prev = prompt
            expected.append(est)
            messages = [
                *messages,
                {"role": "assistant", "content": f"answer {i}"},
                {"role": "user", "content": f"Step {i + 1}: continue with the next file."},
            ]
        h.finished(5)
        states = [d.state for d in h.decisions()]
        table = [
            (s.step_idx, s.current_model, s.current_effort, s.cached_prefix_tokens, s.prompt_tokens)
            for s in states
        ]
        assert table == [
            (0, None, None, 0, expected[0]),
            (1, "qwen3-coder:30b", "medium", 1200, expected[1]),
            (2, "qwen3-coder:30b", "medium", 1500, expected[2]),
            (3, "gpt-oss:20b", "high", 1800, expected[3]),
            (4, "gpt-oss:20b", "high", 2100, expected[4]),
        ]
        # What was sent: the switch rewrote model and reasoning effort at step 2.
        assert [(b["model"], b.get("reasoning_effort")) for b in h.local.chat_bodies()] == [
            ("qwen3-coder:30b", None),  # no reasoning control: effort_map is empty
            ("qwen3-coder:30b", None),
            ("gpt-oss:20b", "high"),
            ("gpt-oss:20b", "high"),
            ("gpt-oss:20b", "high"),
        ]
        reads = [o.cache_read_tokens for o in h.outcomes()]
        assert reads == cached
        s = h.proxy.sessions.get("five")
        assert s is not None and (s.owner_model, s.owner_effort, s.cached_tokens, s.next_step) == (
            "gpt-oss:20b",
            "high",
            2400,
            5,
        )


def test_session_id_falls_back_to_a_hash_of_the_opening_messages(h: Harness) -> None:
    first = chat(content="Open the project.")
    h.post(first)
    second = {
        **first,
        "messages": [
            *first["messages"],
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "Next."},
        ],
    }
    h.post(second)
    h.finished(2)
    a, b = h.decisions()
    assert a.state.session_id == b.state.session_id and a.state.session_id.startswith("h:")
    assert (a.state.step_idx, b.state.step_idx) == (0, 1)


def test_identical_request_marks_the_previous_step_retried(h: Harness) -> None:
    body = chat()
    h.post(body, {"X-Steproute-Session": "rt"})
    h.finished(1)
    h.post(body, {"X-Steproute-Session": "rt"})
    h.finished(3)
    first_id = h.decisions()[0].decision_id
    latest = [o for o in h.outcomes() if o.decision_id == first_id][-1]
    assert latest.signals["retried"] is True and latest.success is False


# --- late signals ---------------------------------------------------------------------------


def test_late_report_merges_signals_into_the_emitted_outcome(h: Harness) -> None:
    h.cloud.set_default(Reply(prompt_tokens=300, completion_tokens=20))
    r = h.post(chat())
    did = r.headers["x-steproute-decision-id"]
    h.finished()
    rep = h.post({"decision_id": did, "signals": {"tests_passed": True}}, path="/report")
    assert rep.status == 200 and rep.body["success"] is True
    joined = dict((d.decision_id, o) for d, o in h.store.joined())
    o = joined[did]
    assert o is not None and o.success is True and o.input_tokens == 300 and o.output_tokens == 20
    assert o.signals["tests_passed"] is True and o.signals["attempts"] == 1
    assert h.post({"decision_id": "nope", "signals": {}}, path="/report").status == 404
    assert h.post({"decision_id": did, "signals": {}, "extra": 1}, path="/report").status == 400


# --- privacy end to end (issue 16, acceptance criterion 3) ----------------------------------


def _fake_secret() -> str:
    # Built at runtime from pieces so no credential-shaped literal is in the repository.
    return "AK" + "IA" + "".join(chr(ord("A") + i) for i in range(16))


def _sensitive_requests() -> list[tuple[str, dict[str, Any]]]:
    secret_text = chat("claude-sonnet-5-5", content=f"deploy with key {_fake_secret()} now")
    path_call = {
        "model": "claude-sonnet-5-5",
        "messages": [
            {"role": "user", "content": "Load the configuration."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "app/config/.env"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "DEBUG=1"},
        ],
    }
    return [("secret", secret_text), ("path", path_call)]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["route", "shadow", "source_down"])
def test_sensitive_state_never_reaches_a_cloud_model(
    tmp_path: Path, mode: str, stream: bool
) -> None:
    kwargs: dict[str, Any] = {"mode": "shadow" if mode == "shadow" else "route"}
    if mode == "source_down":
        kwargs["source"] = BrokenSource()
    for h in start(tmp_path, **kwargs):
        h.cloud.set_default(Reply(kind="error", status=500))  # even errors must not happen
        for name, body in _sensitive_requests():
            r = h.post({**body, "stream": stream}, {"X-Steproute-Session": f"{mode}-{name}"})
            assert r.status == 200, (name, r.body)
            # Sticky: a later step of the same session without the secret stays local too.
            r2 = h.post(
                chat("claude-sonnet-5-5", content="unrelated follow-up", stream=stream),
                {"X-Steproute-Session": f"{mode}-{name}"},
            )
            assert r2.status == 200
        assert h.cloud.received == [], "a sensitive step reached the cloud upstream"
        assert {b["model"] for b in h.local.chat_bodies()} <= {"qwen3-coder:30b"}
        assert len(h.local.chat_bodies()) == 4
        if mode != "source_down":
            h.finished(4)
            assert all(d.state.sensitive for d in h.decisions())
            assert all(d.action.model == "qwen3-coder:30b" for d in h.decisions())


def test_sensitive_without_any_local_option_is_refused(tmp_path: Path) -> None:
    for h in start(tmp_path, source=BrokenSource(), local_fallback=None):
        r = h.post(_sensitive_requests()[0][1])
        assert r.status == 503 and r.body["error"]["type"] == "steproute_sensitive"
        assert h.cloud.received == [] and h.local.received == []


def test_non_sensitive_source_failure_fails_open_to_the_client_model(tmp_path: Path) -> None:
    for h in start(tmp_path, source=BrokenSource()):
        r = h.post(chat("claude-haiku-5-5", temperature=0.5))
        assert r.status == 200 and r.headers["x-steproute-mode"] == "fail_open"
        (sent,) = h.cloud.chat_bodies()
        assert sent["model"] == "claude-haiku-5-5" and sent["temperature"] == 0.5
        assert h.proxy.counters["source_failures"] == 1


def test_on_no_decision_error_refuses_instead_of_failing_open(tmp_path: Path) -> None:
    for h in start(tmp_path, source=BrokenSource(), on_no_decision="error"):
        r = h.post(chat("claude-haiku-5-5"))
        assert r.status == 503 and r.body["error"]["type"] == "steproute_no_decision"
        assert h.cloud.received == [] and h.local.received == []


def test_a_response_without_usage_is_unknown_cost_not_zero(h: Harness) -> None:
    """LiteLLM fills in all-zero usage when the provider sends none; the proxy must record the
    cost as unknown (and say so), and the service must not price the zeros as $0."""
    h.cloud.set_default(Reply(usage=False))
    r = h.post(chat("claude-haiku-5-5"))
    assert r.status == 200
    h.finished()
    (outcome,) = h.outcomes()
    assert outcome.cost_usd is None
    assert outcome.signals.get("usage_missing") is True


# --- decisions over HTTP --------------------------------------------------------------------


def test_http_decision_source_and_its_failure(tmp_path: Path) -> None:
    from steproute.core.privacy import PrivacyConfig
    from steproute.core.rules import RulePolicy
    from steproute.core.schema import Action as A
    from steproute.core.service import Service, make_server
    from steproute.core.store import TraceStore

    store = TraceStore(tmp_path / "svc")
    service = Service(
        policy=RulePolicy(CATALOG),
        catalog=CATALOG,
        store=store,
        privacy=PrivacyConfig(),
        default_action=None,
        local_fallback=A("qwen3-coder:30b", "medium"),
        budget_ms=2000,
    )
    server = make_server(service, None, port=0)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    src = HttpSource(f"http://127.0.0.1:{server.server_address[1]}", decide_timeout_s=2.0)
    for h in start(tmp_path / "p", source=src):
        r = h.post(chat(), {"X-Steproute-Session": "http"})
        assert r.status == 200 and h.cloud.chat_bodies()[0]["model"] == "claude-sonnet-5-5"
        did = r.headers["x-steproute-decision-id"]
        for _ in range(200):
            if any(o.decision_id == did for _, o in store.joined() if o):
                break
            threading.Event().wait(0.01)
        assert [d.decision_id for d, o in store.joined() if o] == [did]
        server.shutdown()
        server.server_close()
        r = h.post(chat("claude-haiku-5-5"))
        assert r.status == 200 and r.headers["x-steproute-mode"] == "fail_open"
        assert h.cloud.chat_bodies()[-1]["model"] == "claude-haiku-5-5"


# --- endpoints ------------------------------------------------------------------------------


def test_models_and_health(h: Harness) -> None:
    models = h.get("/v1/models").body
    ids = {m["id"] for m in models["data"]}
    assert "claude-sonnet-5-5" in ids and "qwen3-coder:30b" in ids
    health = h.get("/healthz").body
    assert health["status"] == "ok" and health["mode"] == "route"


def test_bad_requests(h: Harness) -> None:
    assert h.post({"model": "x"}).status == 400
    assert h.post(chat(), {"X-Steproute-Source": "other"}).status == 400
    # An unknown model can still be routed in route mode...
    assert h.post(chat("some-unknown-model")).status == 200


def test_unknown_model_in_shadow_mode_is_404(tmp_path: Path) -> None:
    for h in start(tmp_path, mode="shadow"):
        assert h.post(chat("some-unknown-model")).status == 404


# --- usage normalisation --------------------------------------------------------------------


def test_usage_normalisation_per_provider() -> None:
    # Anthropic through LiteLLM: prompt_tokens = input + cache creation + cache read.
    anth = {
        "prompt_tokens": 1000,
        "completion_tokens": 5,
        "cache_creation_input_tokens": 300,
        "prompt_tokens_details": {"cached_tokens": 600, "cache_creation_tokens": 300},
    }
    assert normalise_usage(anth, "anthropic") == AttemptUsage(100, 600, 300, 5)
    # DeepSeek, Ollama /v1, any OpenAI-compatible server: no write charge.
    ds = {
        "prompt_tokens": 1000,
        "completion_tokens": 5,
        "prompt_cache_hit_tokens": 600,
        "prompt_tokens_details": {"cached_tokens": 600},
    }
    assert normalise_usage(ds, "deepseek") == AttemptUsage(400, 600, 0, 5)
    assert normalise_usage({**anth}, "openai") == AttemptUsage(400, 600, 0, 5)
    assert normalise_usage(None, "openai") is None


# --- LiteLLM's own network behaviour --------------------------------------------------------


_IMPORT_PROBE = r"""
import socket, sys
seen = []
real = socket.getaddrinfo
def gai(host, *a, **k):
    if str(host) not in ("127.0.0.1", "::1", "localhost"):
        seen.append(str(host)); raise OSError("blocked")
    return real(host, *a, **k)
socket.getaddrinfo = gai
from steproute.collectors.proxy.upstream import load_litellm
litellm = load_litellm()
from litellm.anthropic_beta_headers_manager import _load_beta_headers_config
_load_beta_headers_config()
print("SEEN=" + ",".join(seen))
"""


def test_litellm_import_and_setup_make_no_network_call() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LITELLM_")}
    out = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=120,
    ).stdout
    assert "SEEN=\n" in out or out.strip().endswith("SEEN="), out


def test_cli_import_does_not_import_litellm() -> None:
    code = "import sys, steproute.cli, steproute.collectors.proxy; print('litellm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- real local Ollama (opt-in; docs/PROXY.md "Ollama smoke test") --------------------------


@pytest.mark.skipif(
    os.environ.get("STEPROUTE_TEST_OLLAMA") != "1", reason="set STEPROUTE_TEST_OLLAMA=1"
)
def test_ollama_smoke_produces_joined_traces(tmp_path: Path) -> None:
    """Two steps through the proxy to a real local Ollama; skipped (not failed) without it."""
    import socket

    from steproute.collectors.proxy.config import ProxyConfig, UpstreamConfig
    from steproute.collectors.proxy.engine import Proxy
    from steproute.collectors.proxy.server import make_proxy_server
    from steproute.collectors.proxy.source import InProcessSource
    from steproute.collectors.proxy.upstream import UpstreamPool
    from steproute.core.policy import FixedPolicy
    from steproute.core.privacy import PrivacyConfig
    from steproute.core.service import Service
    from steproute.core.store import TraceStore

    model = os.environ.get("STEPROUTE_TEST_OLLAMA_MODEL", "qwen3-coder:30b")
    try:
        socket.create_connection(("127.0.0.1", 11434), timeout=1).close()
    except OSError:
        pytest.skip("Ollama is not running on 127.0.0.1:11434")
    profile = CATALOG.get(model)
    effort = "medium" if "medium" in profile.efforts else profile.efforts[0]
    store = TraceStore(tmp_path / "traces")
    service = Service(
        policy=FixedPolicy(Action(model, effort), CATALOG),
        catalog=CATALOG,
        store=store,
        privacy=PrivacyConfig(),
        default_action=None,
        local_fallback=None,
    )
    up = UpstreamConfig(
        name="ollama",
        provider="openai",
        base_url="http://127.0.0.1:11434/v1",
        models=(model,),
        local=True,
    )
    proxy = Proxy(
        ProxyConfig(upstreams=(up,)),
        source=InProcessSource(service),
        catalog=CATALOG,
        privacy=PrivacyConfig(),
        pool=UpstreamPool({}),
    )
    server = make_proxy_server(proxy, None, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        h = Harness(proxy, None, None, store, server.server_address[1], service)  # type: ignore[arg-type]
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Reply with: ready."}]
        for _ in range(2):
            r = h.post(
                {"model": model, "messages": messages, "max_tokens": 64},
                {"X-Steproute-Session": "ollama-smoke"},
            )
            assert r.status == 200, r.body
            messages += [r.body["choices"][0]["message"], {"role": "user", "content": "Again."}]
        h.finished(2, timeout=30)
        pairs = list(store.joined())
        assert len(pairs) == 2 and all(o is not None for _, o in pairs)
        for _, o in pairs:
            assert o is not None and o.input_tokens + o.cache_read_tokens > 0
        assert pairs[1][0].state.current_model == model
        assert pairs[1][0].state.cached_prefix_tokens > 0
    finally:
        server.shutdown()
        server.server_close()
