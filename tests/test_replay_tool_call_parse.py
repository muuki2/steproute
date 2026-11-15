"""Issue 71: a tool call the provider cannot parse is the model's step, not a provider error.

Ollama (0.40.16, ``/v1/chat/completions``) fails the whole request with HTTP 500 when the
model writes tool-call arguments that are not valid JSON. The proxy gives that answer the
error kind ``tool_call_parse``; the replay client and the scaffold handle it as a reply with
invalid JSON arguments, as for every other provider. Also ``resume --retry-errors``, which
runs the cells given up as ``provider_error`` or ``infra_error`` again. Loopback mocks only.

Issue 75: the proxy returns the rejected arguments and the parser's error to the client (in its
answer only), and the scaffold quotes them back to the model; the same message for invalid JSON
that a provider returns inside a normal reply.
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
from steproute.collectors.proxy.upstream import (  # noqa: E402
    REJECTED_MAX_CHARS,
    TOOL_CALL_PARSE,
    _as_error,
    _rejected_call,
)
from steproute.core.store import TraceStore  # noqa: E402
from steproute.eval.data import load_grid  # noqa: E402
from steproute.replay.agent import (  # noqa: E402
    ARGUMENTS_SHAPE,
    FORMAT_ERROR,
    INVALID_JSON,
    QUOTE_MAX_CHARS,
    BashAgent,
    ChatReply,
    RejectedCall,
    invalid_json_error,
)
from steproute.replay.journal import Journal  # noqa: E402
from steproute.replay.proxyrun import ProxyChatClient  # noqa: E402

LOCAL = ("qwen3-coder:30b", "medium")
#: Stands for the model's own text inside Ollama's error message: it goes back to the model
#: (the proxy's answer, the next prompt, ``trajectory.json``) and nowhere else.
RAW = "RAW-MODEL-OUTPUT-71"
PARSER_ERROR = "invalid character ']' after object key:value pair"
RAW_ARGS = '{"command":"apply_patch <<\'PATCH\'\\n*** Begin Patch ' + RAW + ' PATCH"]}'
OLLAMA_PARSE_ERROR = {"message": f"error parsing tool call: raw='{RAW_ARGS}', err={PARSER_ERROR}"}


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
    # the rejected arguments ride on the error, apart from its text
    assert e.rejected == {
        "arguments": RAW_ARGS,
        "parser_error": PARSER_ERROR,
        "length": len(RAW_ARGS),
        "truncated": False,
    }
    # without a body, the provider message after LiteLLM's prefix
    e = _as_error(_FakeError(f"litellm.InternalServerError: OpenAIException - {msg}", 500))
    assert e.kind == TOOL_CALL_PARSE and RAW not in str(e)
    assert e.rejected is not None and e.rejected["arguments"] == RAW_ARGS
    # another status, or the words anywhere but at the start: not this kind
    assert _as_error(_FakeError(msg, 400, {**OLLAMA_PARSE_ERROR})).kind == "_FakeError"
    other = {"message": "llama runner: error parsing tool call"}
    assert _as_error(_FakeError(other["message"], 500, other)).kind == "_FakeError"


def test_the_rejected_arguments_are_split_on_the_last_err_marker_and_capped() -> None:
    # the arguments themselves contain "'" and "', err=": the split is on the last marker
    tricky = """{"command":"echo 'a', err=b' && x"]}"""
    r = _rejected_call(f"error parsing tool call: raw='{tricky}', err=bad thing")
    assert r is not None and r["arguments"] == tricky and r["parser_error"] == "bad thing"
    # long arguments are cut, and the answer says so and gives the full length
    long = '{"command":"' + "x" * (REJECTED_MAX_CHARS + 50) + '"]}'
    r = _rejected_call(f"error parsing tool call: raw='{long}', err=e")
    assert r is not None and r["truncated"] is True and r["length"] == len(long)
    assert r["arguments"] == long[:REJECTED_MAX_CHARS]
    # another shape: the kind stays, without arguments
    assert _rejected_call("error parsing tool call: something else") is None
    assert _rejected_call("other: raw='x', err=y") is None
    e = _as_error(_FakeError("x", 500, {"message": "error parsing tool call: no quote"}))
    assert e.kind == TOOL_CALL_PARSE and e.rejected is None


def test_only_the_error_body_carries_the_rejected_call() -> None:
    from steproute.collectors.proxy.engine import _upstream_error_body

    e = _as_error(_FakeError("x", 500, {**OLLAMA_PARSE_ERROR}))
    body = _upstream_error_body(500, f"upstream failed ({e.kind}): {e}", e)["error"]
    assert body["upstream_kind"] == TOOL_CALL_PARSE
    assert body["rejected_tool_call"]["arguments"] == RAW_ARGS
    assert RAW not in body["message"]
    # any other upstream error: no such field
    other = _as_error(_FakeError("boom", 500, {"message": "model runner stopped"}))
    assert "rejected_tool_call" not in _upstream_error_body(500, "m", other)["error"]
    assert "rejected_tool_call" not in _upstream_error_body(502, "m", None)["error"]


# --- the scaffold --------------------------------------------------------------------------


class _Replies:
    def __init__(self, reply: ChatReply) -> None:
        self.reply = reply
        self.sent: list[list[dict[str, Any]]] = []

    def complete(self, messages: Any, tools: Any) -> ChatReply:
        self.sent.append([dict(m) for m in messages])
        return self.reply


BAD_ARGS = '{"command":"x"]}'
#: What Python's json says about BAD_ARGS.
PY_ERROR = "Expecting ',' delimiter: line 1 column 15 (char 14)"


def test_the_scaffold_handles_an_unreadable_reply_as_invalid_json() -> None:
    bad_json = {
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": BAD_ARGS}}],
    }
    runs = {}
    for name, reply in (
        ("unreadable", ChatReply({}, None, unreadable=True)),
        (
            "unreadable, quoted",
            ChatReply({}, None, unreadable=True, rejected=RejectedCall(BAD_ARGS, PARSER_ERROR)),
        ),
        ("invalid json", ChatReply(bad_json, "tool_calls")),
    ):
        client = _Replies(reply)
        result = BashAgent().run("task", None, client, nonce="n")  # type: ignore[arg-type]
        runs[name] = (result, client)
    for result, client in runs.values():
        # the same counted steps and the same end: three replies in a row without a usable call
        assert result.exit == "no_submission" and result.steps == 3
        # and the same history: the invalid call is never kept, so a strict provider accepts
        # the next request
        history = client.sent[1]
        assert [m["role"] for m in history] == ["system", "user", "assistant", "user"]
        assert history[2] == {"role": "assistant", "content": ""}
    # the format error quotes the arguments and the parser's error when there are any
    expected = {
        "unreadable": invalid_json_error(None, None),
        "unreadable, quoted": invalid_json_error(BAD_ARGS, PARSER_ERROR),
        "invalid json": invalid_json_error(BAD_ARGS, PY_ERROR),
    }
    for name, (result, _) in runs.items():
        assert result.messages[-1]["content"] == FORMAT_ERROR.format(error=expected[name])
    quoted = runs["invalid json"][0].messages[-1]["content"]
    assert f"<arguments>\n{BAD_ARGS}\n</arguments>" in quoted and PY_ERROR in quoted
    assert ARGUMENTS_SHAPE in quoted and INVALID_JSON in quoted
    assert ARGUMENTS_SHAPE in runs["unreadable"][0].messages[-1]["content"]


def test_the_quoted_arguments_are_cut_and_the_cut_is_said() -> None:
    args = "x" * (QUOTE_MAX_CHARS + 10)
    text = invalid_json_error(args, "e")
    assert "x" * QUOTE_MAX_CHARS + "\n</arguments>" in text
    assert "x" * (QUOTE_MAX_CHARS + 1) not in text
    assert f"the first {QUOTE_MAX_CHARS:,} of {QUOTE_MAX_CHARS + 10:,} characters" in text
    # cut by the proxy already: the full length is the one the provider sent
    assert "of 9,000 characters" in invalid_json_error("y" * 100, "e", 9000)
    assert "cut" not in invalid_json_error("short", "e")


def test_a_reply_with_text_and_invalid_json_keeps_its_text_but_not_the_call() -> None:
    reply = {
        "content": "THOUGHT: edit it.",
        "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": BAD_ARGS}}],
    }
    client = _Replies(ChatReply(reply, "tool_calls"))
    result = BashAgent(max_format_errors=1).run(
        "task",
        None,  # type: ignore[arg-type]
        client,
        nonce="n",
    )
    assert result.messages[2] == {"role": "assistant", "content": "THOUGHT: edit it."}
    assert result.messages[3]["role"] == "user" and BAD_ARGS in result.messages[3]["content"]
    assert len(result.messages) == 4


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

    # what the proxy answered: the kind, and the rejected call in its own field, never in the
    # error's text
    (bad,) = [(s, o) for s, o in answers if s != 200]
    assert bad[0] == 500 and bad[1]["error"]["upstream_kind"] == TOOL_CALL_PARSE
    assert RAW not in bad[1]["error"]["message"]
    assert bad[1]["error"]["rejected_tool_call"] == {
        "arguments": RAW_ARGS,
        "parser_error": PARSER_ERROR,
        "length": len(RAW_ARGS),
        "truncated": False,
    }

    # the model got the format error for invalid JSON, quoting what it sent and the parser's
    # error, after an empty assistant turn
    after = _bodies_of(mock, "a")[1]["messages"]
    assert [m["role"] for m in after] == ["system", "user", "assistant", "user"]
    assert after[2] == {"role": "assistant", "content": ""}
    assert after[3]["content"] == FORMAT_ERROR.format(
        error=invalid_json_error(RAW_ARGS, PARSER_ERROR, len(RAW_ARGS))
    )

    # the step's trace: a model step labelled with the run's result, its usage unknown
    pairs = [(d, o) for d, o in TraceStore(out / "traces").joined() if o is not None]
    assert len(pairs) == 4
    (parse_step,) = [o for _, o in pairs if o.final_model is None]
    # recorded as the tool-call check records the same mistake in a reply
    assert parse_step.signals["failed_checks"] == ["tool_call_parse"]
    assert parse_step.signals["tool_error"] is True
    assert parse_step.cost_usd is None
    assert parse_step.success is True and "replay_call_failed" not in parse_step.signals

    # the raw upstream text is only in the trajectory, which holds the model's output anyway:
    # not in the log, the output, the journal, the traces or any other file of the run
    assert "tool_call_parse" in caplog.text  # the proxy logged the failure, by its kind only
    _only_in_trajectory(RAW, out, caplog.text + buf.getvalue())
    assert guard.blocked == []


def _only_in_trajectory(marker: str, out: Path, logs: str) -> None:
    assert marker not in logs
    seen = []
    for p in out.rglob("*"):
        if p.is_file() and marker in p.read_text("utf-8", errors="replace"):
            seen.append(p.name)
    assert seen == ["trajectory.json"]


# --- issue 75: a rejected apply_patch call, corrected -----------------------------------

#: Only in the rejected call: where it shows up, the rejected text went there.
MARKER = "REJECTED-PATCH-75"
_PATCH = (
    "*** Begin Patch\n*** Update File: calc.py\n@@ def add(a, b):\n"
    "-    return a - b\n+{}\n*** End Patch\n"
)


def _apply_patch_command(added: str) -> str:
    return "apply_patch <<'PATCH'\n" + _PATCH.format(added) + "PATCH"


def _tool(command: str, step: int) -> Reply:
    return Reply(
        kind="tool",
        tool_name="bash",
        tool_args=json.dumps({"command": command}),
        tool_text="THOUGHT: next step.",
        tool_id=f"call_{step}",
    )


def test_a_rejected_apply_patch_call_is_quoted_then_corrected_and_applied(
    tmp_path: Path,
    mock: Any,
    guard: NetworkGuard,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    # the model's habit: one JSON string, then a stray "]" as if "command" were a list
    rejected = (
        '{"command":' + json.dumps(_apply_patch_command(f"    return a + b  # {MARKER}")) + "]}"
    )
    ollama_error = {"message": f"error parsing tool call: raw='{rejected}', err={PARSER_ERROR}"}

    def model(step: int, rec: Received) -> Reply | None:
        if step == 0:
            return Reply(kind="error", status=500, error_body=dict(ollama_error))
        if step == 1:
            return _tool(_apply_patch_command("    return a + b"), step)
        if step == 2:
            return _tool("git diff -- calc.py > patch.txt", step)
        return _tool("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt", step)

    mock.set_responder(ScriptedAgent(overrides={"a": model}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    buf = io.StringIO()
    code = runner(plan, manifest, config_for(mock), out, environ={}, stdout=buf).run(execute=True)
    assert code == 0

    bodies = _bodies_of(mock, "a")
    assert len(bodies) == 4
    # the prompt after the rejected call quotes what the model sent and the parser's error
    quoted = bodies[1]["messages"][-1]
    assert quoted["role"] == "user"
    assert f"<arguments>\n{rejected}\n</arguments>" in quoted["content"]
    assert f"Parser error: {PARSER_ERROR}" in quoted["content"]
    assert ARGUMENTS_SHAPE in quoted["content"]
    # the corrected call ran apply_patch in the workspace, which changed the file
    observation = bodies[2]["messages"][-1]
    assert observation["role"] == "tool" and "<returncode>0</returncode>" in observation["content"]
    assert "M calc.py" in observation["content"]

    # the run submitted that change and passed
    run = _grid(out)[("add-a", *LOCAL)]
    assert run.success is True and run.meta["status"] == "passed"
    assert run.meta["steps"] == 4 and run.meta["scaffold"] == "steproute-bash-2"
    patch = (out / "runs" / run.run_id / "patch.diff").read_text("utf-8")
    assert "-    return a - b" in patch and "+    return a + b" in patch
    assert MARKER not in patch

    # the rejected text: in the trajectory (model output), and in no log, trace, journal line
    # or result file
    _only_in_trajectory(MARKER, out, caplog.text + buf.getvalue())
    assert guard.blocked == []


def test_invalid_json_inside_a_normal_reply_gets_the_same_quoted_message(
    tmp_path: Path, mock: Any
) -> None:
    bad = '{"command":"ls -la"]}'
    try:
        json.loads(bad)
    except ValueError as exc:
        py_error = str(exc)

    def bad_once(step: int, rec: Received) -> Reply | None:
        if step == 0:
            return Reply(kind="tool", tool_name="bash", tool_args=bad, tool_text="THOUGHT: look.")
        return None

    # steps: the reply with invalid JSON, then write the fix, diff, submit
    mock.set_responder(ScriptedAgent(overrides={"a": bad_once}, extra_steps=1))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    assert runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True) == 0
    run = _grid(out)[("add-a", *LOCAL)]
    assert run.success is True and run.meta["steps"] == 4

    after = _bodies_of(mock, "a")[1]["messages"]
    # the invalid call is not sent back (a strict provider would reject the request); its
    # text is, as for a call the provider rejected
    assert [m["role"] for m in after] == ["system", "user", "assistant", "user"]
    assert after[2]["content"] == "THOUGHT: look." and "tool_calls" not in after[2]
    assert after[3]["content"] == FORMAT_ERROR.format(error=invalid_json_error(bad, py_error))


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

    # the plan text says what will run: the given-up cell is not "to run" without the option
    # (and not in the order or the estimates), and is with it
    without = "  runs: 2 in the plan, 1 finished, 1 given up, 0 to run"
    with_option = "  runs: 2 in the plan, 1 finished, 1 to run"
    order_line = f"    1. {LOCAL[0]}"
    for retry, counts, in_order in ((False, without, False), (True, with_option, True)):
        buf = io.StringIO()
        dry = runner(plan, manifest, cfg, out, environ={}, stdout=buf, retry_errors=retry)
        assert dry.run() == 0
        text = buf.getvalue()
        assert counts in text.splitlines()
        assert (order_line in text) is in_order
        assert ("   ollama        1 runs" in text) is in_order
        assert ("   ollama        0 runs" in text) is not in_order

    # without the option: left alone, even with the provider healthy again
    broken["on"] = False
    buf = io.StringIO()
    assert runner(plan, manifest, cfg, out, environ={}, stdout=buf).run(execute=True) == 0
    assert len(attempts()) == 3  # nothing ran, as the start of the run said
    assert without in buf.getvalue().splitlines()
    assert "given up: 1 cell(s)" in buf.getvalue() and "--retry-errors" in buf.getvalue()

    # with the option: one more attempt in the same journal, marked as a new round
    buf = io.StringIO()
    r = runner(plan, manifest, cfg, out, environ={}, stdout=buf, retry_errors=True)
    assert r.run(execute=True) == 0
    assert with_option in buf.getvalue().splitlines()
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
