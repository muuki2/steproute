"""Issue 71: a tool call the provider cannot parse is the model's step, not a provider error.

Ollama (0.40.16, ``/v1/chat/completions``) fails the whole request with HTTP 500 when the
model writes tool-call arguments that are not valid JSON. The proxy gives that answer the
error kind ``tool_call_parse``; the replay client and the scaffold handle it as a reply with
invalid JSON arguments, as for every other provider. Also ``resume --retry-errors``, which
runs the cells given up as ``provider_error`` or ``infra_error`` again. Loopback mocks only.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

from proxy_harness import NetworkGuard  # noqa: E402
from replay_harness import ScriptedAgent, build, config_for, runner  # noqa: E402
from steproute.cli import main  # noqa: E402
from steproute.collectors.mock_upstream import MockUpstream, Received, Reply  # noqa: E402
from steproute.collectors.proxy.upstream import TOOL_CALL_PARSE, _as_error  # noqa: E402
from steproute.core.store import TraceStore  # noqa: E402
from steproute.eval.data import load_grid  # noqa: E402
from steproute.replay.agent import (  # noqa: E402
    FORMAT_ERROR,
    INVALID_JSON,
    BashAgent,
    ChatReply,
)
from steproute.replay.journal import Journal  # noqa: E402
from steproute.replay.proxyrun import ProxyChatClient  # noqa: E402

LOCAL = ("qwen3-coder:30b", "medium")
#: Stands for the model's own text inside Ollama's error message: it must never be kept.
RAW = "RAW-MODEL-OUTPUT-71"
OLLAMA_PARSE_ERROR = {
    "message": (
        "error parsing tool call: raw='{\"command\":\"apply_patch <<'PATCH'\\n*** Begin Patch "
        f"{RAW} PATCH\"]}}', err=invalid character ']' after object key:value pair"
    )
}


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> NetworkGuard:
    g = NetworkGuard()
    g.install(monkeypatch)
    return g


@pytest.fixture
def mock() -> Any:
    with MockUpstream() as m:
        yield m


def _grid(out: Path) -> dict[tuple[str, str, str], Any]:
    return {(r.task_id, r.model, r.effort): r for r in load_grid(out / "grid.jsonl").runs}


def _bodies_of(mock: Any, key: str) -> list[dict[str, Any]]:
    return [b for b in mock.chat_bodies() if f"TASK-KEY: {key}" in json.dumps(b)]


# --- the proxy's error kind ----------------------------------------------------------------


class _FakeError(Exception):
    def __init__(self, text: str, status: int, body: Any = None) -> None:
        super().__init__(text)
        self.status_code = status
        self.body = body


def test_the_upstream_error_kind_needs_status_500_and_the_message_prefix() -> None:
    msg = OLLAMA_PARSE_ERROR["message"]
    # LiteLLM keeps the provider's error object as ``body``; its own text has a prefix.
    e = _as_error(
        _FakeError(
            f"litellm.InternalServerError: OpenAIException - {msg}", 500, {**OLLAMA_PARSE_ERROR}
        )
    )
    assert e.kind == TOOL_CALL_PARSE and e.status == 500 and RAW not in str(e)
    # without a body, the provider message after LiteLLM's prefix
    e = _as_error(_FakeError(f"litellm.InternalServerError: OpenAIException - {msg}", 500))
    assert e.kind == TOOL_CALL_PARSE and RAW not in str(e)
    # another status, or the words anywhere but at the start: not this kind
    assert _as_error(_FakeError(msg, 400, {**OLLAMA_PARSE_ERROR})).kind == "_FakeError"
    other = {"message": "llama runner: error parsing tool call"}
    assert _as_error(_FakeError(other["message"], 500, other)).kind == "_FakeError"


# --- the scaffold --------------------------------------------------------------------------


class _Replies:
    def __init__(self, reply: ChatReply) -> None:
        self.reply = reply
        self.sent: list[list[dict[str, Any]]] = []

    def complete(self, messages: Any, tools: Any) -> ChatReply:
        self.sent.append([dict(m) for m in messages])
        return self.reply


def test_the_scaffold_handles_an_unreadable_reply_as_invalid_json() -> None:
    bad_json = {
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": '{"command":"x"]}'}}],
    }
    runs = {}
    for name, reply in (
        ("unreadable", ChatReply({}, None, unreadable=True)),
        ("invalid json", ChatReply(bad_json, "tool_calls")),
    ):
        client = _Replies(reply)
        result = BashAgent().run("task", None, client, nonce="n")  # type: ignore[arg-type]
        runs[name] = (result, client)
    for result, _ in runs.values():
        # the same counted steps and the same end: three replies in a row without a usable call
        assert result.exit == "no_submission" and result.steps == 3
        assert result.messages[-1]["content"] == FORMAT_ERROR.format(error=INVALID_JSON)
    history = runs["unreadable"][1].sent[1]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "user"]
    assert history[2] == {"role": "assistant", "content": ""}


# --- the run continues ---------------------------------------------------------------------


def test_an_unparseable_tool_call_from_ollama_is_a_format_error_step(
    tmp_path: Path,
    mock: Any,
    guard: NetworkGuard,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    failed: list[int] = []

    def parse_error_once(step: int, rec: Received) -> Reply | None:
        if step == 0 and not failed:
            failed.append(step)
            return Reply(kind="error", status=500, error_body=dict(OLLAMA_PARSE_ERROR))
        return None

    # steps: the call Ollama fails, then write the fix, diff, submit (extra_steps=1 shifts the
    # script by one, so the failed step stands where the harmless `ls` would be)
    agent = ScriptedAgent(overrides={"a": parse_error_once}, extra_steps=1)
    mock.set_responder(agent)
    answers: list[tuple[int, dict[str, Any]]] = []
    orig = ProxyChatClient._post

    def post(self: ProxyChatClient, body: Any) -> Any:
        status, headers, obj = orig(self, body)
        answers.append((status, obj))
        return status, headers, obj

    monkeypatch.setattr(ProxyChatClient, "_post", post)
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    buf = io.StringIO()
    code = runner(plan, manifest, config_for(mock), out, environ={}, stdout=buf).run(execute=True)
    assert code == 0 and failed == [0]

    # the run went on to the end: one attempt, the failed call counted as a step and a call
    run = _grid(out)[("add-a", *LOCAL)]
    assert run.success is True and run.meta["status"] == "passed"
    assert run.meta["attempts"] == 1
    assert run.meta["steps"] == 4
    assert run.meta["calls_missing_usage"] == 1
    (att,) = Journal.open(out / "runs.jsonl").attempts.values()
    assert att.end is not None and att.end["calls"] == 4
    # no call retry, no restart: four requests in all
    assert len(_bodies_of(mock, "a")) == 4

    # what the proxy answered: the kind only, never the model's text
    (bad,) = [(s, o) for s, o in answers if s != 200]
    assert bad[0] == 500 and bad[1]["error"]["upstream_kind"] == TOOL_CALL_PARSE
    assert RAW not in json.dumps(bad[1])

    # the model got the same format error as for invalid JSON, after an empty assistant turn
    after = _bodies_of(mock, "a")[1]["messages"]
    assert [m["role"] for m in after] == ["system", "user", "assistant", "user"]
    assert after[2] == {"role": "assistant", "content": ""}
    assert after[3]["content"] == FORMAT_ERROR.format(error=INVALID_JSON)

    # the step's trace: a model step labelled with the run's result, its usage unknown
    pairs = [(d, o) for d, o in TraceStore(out / "traces").joined() if o is not None]
    assert len(pairs) == 4
    (parse_step,) = [o for _, o in pairs if o.final_model is None]
    # recorded as the tool-call check records the same mistake in a reply
    assert parse_step.signals["failed_checks"] == ["tool_call_parse"]
    assert parse_step.signals["tool_error"] is True
    assert parse_step.cost_usd is None
    assert parse_step.success is True and "replay_call_failed" not in parse_step.signals

    # the raw upstream text is nowhere: not in the log, the output, or any file of the run
    assert "tool_call_parse" in caplog.text  # the proxy logged the failure, by its kind only
    assert RAW not in caplog.text and RAW not in buf.getvalue()
    for p in out.rglob("*"):
        if p.is_file():
            assert RAW not in p.read_text("utf-8", errors="replace"), p
    assert guard.blocked == []


@pytest.mark.parametrize(
    "error",
    [
        {"message": "model runner has unexpectedly stopped"},
        {"message": "llama runner: error parsing tool call: raw='x'"},
        {"message": "scripted failure", "type": "mock", "code": 500},
    ],
)
def test_any_other_500_is_still_a_provider_error(
    tmp_path: Path, mock: Any, error: dict[str, Any]
) -> None:
    agent = ScriptedAgent(
        overrides={"a": lambda step, rec: Reply(kind="error", status=500, error_body=error)}
    )
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    run = _grid(out)[("add-a", *LOCAL)]
    assert run.success is None and run.meta["status"] == "provider_error"
    assert run.meta["attempts"] == 3
    assert run.meta["reason"] == "provider answered HTTP 500 (upstream_error)"


# --- resume --retry-errors -----------------------------------------------------------------


def test_retry_errors_runs_given_up_cells_again_and_only_with_the_option(
    tmp_path: Path, mock: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = {"on": True}

    def maybe_fail(step: int, rec: Received) -> Reply | None:
        return Reply(kind="error", status=500) if broken["on"] else None

    mock.set_responder(ScriptedAgent(overrides={"a": maybe_fail}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a", "add-b"])
    out = tmp_path / "out"
    cfg = config_for(mock)

    def attempts() -> list[Any]:
        cells = Journal.open(out / "runs.jsonl").by_cell()
        return cells[("add-a", *LOCAL, 0)]

    assert runner(plan, manifest, cfg, out, environ={}).run(execute=True) == 0
    assert [a.status for a in attempts()] == ["provider_error"] * 3
    assert _grid(out)[("add-b", *LOCAL)].success is True

    # status tells how many cells are in that state
    assert main(["replay", "status", "--out", str(out), "--ledger", str(tmp_path / "l")]) == 0
    assert "cells given up after 2 retries (provider_error or infra_error): 1" in (
        capsys.readouterr().out
    )

    # without the option: left alone, even with the provider healthy again
    broken["on"] = False
    buf = io.StringIO()
    assert runner(plan, manifest, cfg, out, environ={}, stdout=buf).run(execute=True) == 0
    assert len(attempts()) == 3
    assert "given up: 1 cell(s)" in buf.getvalue() and "--retry-errors" in buf.getvalue()

    # with the option: one more attempt in the same journal, marked as a new round
    buf = io.StringIO()
    r = runner(plan, manifest, cfg, out, environ={}, stdout=buf, retry_errors=True)
    assert r.run(execute=True) == 0
    att = attempts()
    assert [a.status for a in att] == ["provider_error"] * 3 + ["passed"]
    assert att[-1].attempt == 4 and att[-1].start.get("retry_errors") is True
    assert "runs them again" in buf.getvalue()
    grid = _grid(out)
    assert grid[("add-a", *LOCAL)].success is True and grid[("add-a", *LOCAL)].meta["attempts"] == 4
    # the finished cell is not run again
    assert len(Journal.open(out / "runs.jsonl").by_cell()[("add-b", *LOCAL, 0)]) == 1


def test_retry_errors_applies_the_retry_limit_afresh_and_once_per_invocation(
    tmp_path: Path, mock: Any
) -> None:
    mock.set_responder(ScriptedAgent(overrides={"a": lambda s, r: Reply(kind="error", status=500)}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    cfg = config_for(mock)
    runner(plan, manifest, cfg, out, environ={}).run(execute=True)
    code = runner(plan, manifest, cfg, out, environ={}, retry_errors=True).run(execute=True)
    assert code == 0  # given up again: finished, not pending
    att = Journal.open(out / "runs.jsonl").by_cell()[("add-a", *LOCAL, 0)]
    # 1 + max_retries in the first round, and as many again in the new one
    assert [a.status for a in att] == ["provider_error"] * 6
    assert [bool(a.start.get("retry_errors")) for a in att] == [False] * 3 + [True] + [False] * 2
    run = _grid(out)[("add-a", *LOCAL)]
    assert run.meta["status"] == "provider_error" and run.meta["attempts"] == 6
