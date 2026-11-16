"""Issue 80: a run that ends without a result from the scaffold keeps its conversation.

``trajectory.json`` and ``workspace.diff`` used to be written only from the scaffold's result,
so a run the client stopped (``context_limit``, ``provider_error``, ...) left neither. The
client now remembers the messages of its latest call, and the runner stores them, with the
workspace's diff, when a run ends early. A run the scaffold itself ended (passed, failed, step
limit) is stored as before, from the scaffold's own result. Loopback mocks only.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

from proxy_harness import NetworkGuard  # noqa: E402
from replay_harness import (  # noqa: E402
    GOOD,
    ScriptedAgent,
    build,
    config_for,
    environ_with,
    fake_key,
    runner,
    write_command,
)
from steproute.collectors.mock_upstream import MockUpstream, Received, Reply  # noqa: E402
from steproute.core.policy import FixedPolicy  # noqa: E402
from steproute.core.schema import Action  # noqa: E402
from steproute.eval.data import load_grid  # noqa: E402
from steproute.replay.agent import AgentResult, BashAgent  # noqa: E402
from steproute.replay.config import Limits  # noqa: E402
from steproute.replay.journal import Attempt, Journal  # noqa: E402
from steproute.replay.runner import EXIT_FATAL  # noqa: E402
from steproute.replay.workspace import ExecResult, LocalWorkspace  # noqa: E402

LOCAL = ("qwen3-coder:30b", "medium")
HAIKU = ("claude-haiku-5-5", "low")
#: The output of the last command before the stop: in the transcript, never sent to the model.
LAST_OUTPUT = "SEEN-BEFORE-THE-STOP-80"
#: The scripted fix, as the workspace's diff shows it.
FIX = "+    return a + b"
STEP = ["assistant", "tool"]
STOPPED_FILES = ["result.json", "trajectory.json", "workspace.diff"]


@pytest.fixture(autouse=True)
def guard(monkeypatch: pytest.MonkeyPatch) -> NetworkGuard:
    g = NetworkGuard()
    g.install(monkeypatch)
    return g


@pytest.fixture
def mock() -> Any:
    with MockUpstream() as m:
        yield m


def _attempts(out: Path) -> list[Attempt]:
    return sorted(Journal.open(out / "runs.jsonl").attempts.values(), key=lambda a: a.attempt)


def _end(att: Attempt) -> dict[str, Any]:
    assert att.end is not None
    return att.end


def _stored(out: Path, att: Attempt) -> tuple[list[dict[str, Any]] | None, str | None]:
    """The run's ``trajectory.json`` (parsed) and ``workspace.diff``; ``None``: no such file."""
    traj, diff = (out / "runs" / att.run_id / n for n in ("trajectory.json", "workspace.diff"))
    return (
        json.loads(traj.read_text("utf-8")) if traj.exists() else None,
        diff.read_text("utf-8") if diff.exists() else None,
    )


def _names(out: Path, att: Attempt) -> list[str]:
    return sorted(p.name for p in (out / "runs" / att.run_id).iterdir())


def _command(message: dict[str, Any]) -> str:
    (call,) = message["tool_calls"]
    return str(json.loads(call["function"]["arguments"])["command"])


def _bodies_of(mock: Any, key: str) -> list[dict[str, Any]]:
    return [b for b in mock.chat_bodies() if f"TASK-KEY: {key}" in json.dumps(b)]


def _huge_prompt_at_step_1(command: str) -> Callable[[int, Received], Reply | None]:
    """Step 0 is the script's (it writes the fix). Step 1 runs ``command``, and the provider
    reports a prompt above the local cap (32,768 - 2,048 tokens) for it, so the client stops
    the run before the next call."""

    def reply(step: int, rec: Received) -> Reply | None:
        if step != 1:
            return None
        return Reply(
            kind="tool",
            tool_name="bash",
            tool_args=json.dumps({"command": command}),
            tool_text="THOUGHT: look.",
            prompt_tokens=31_000,
            completion_tokens=20,
        )

    return reply


def _run_to_the_context_limit(
    tmp_path: Path, mock: Any, command: str = f"echo {LAST_OUTPUT}", **deps: Any
) -> tuple[Path, Attempt]:
    mock.set_responder(ScriptedAgent(overrides={"a": _huge_prompt_at_step_1(command)}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ={}, **deps).run(execute=True)
    (att,) = _attempts(out)
    # the run's record is what it always was: a labelled failure after two steps
    end = _end(att)
    assert end["status"] == "context_limit" and end["success"] is False
    assert "passes the local cap" in end["reason"]
    assert end["steps"] == 2 and end["calls"] == 2
    assert end["scaffold"] == "steproute-bash-2"
    return out, att


# --- runs the client stopped -----------------------------------------------------------------


def test_a_run_stopped_at_the_context_limit_keeps_its_conversation_and_diff(
    tmp_path: Path, mock: Any, guard: NetworkGuard
) -> None:
    out, att = _run_to_the_context_limit(tmp_path, mock)
    traj, diff = _stored(out, att)
    assert traj is not None and diff is not None
    # both steps made before the stop, each with its command and its output
    assert [m["role"] for m in traj] == ["system", "user", *STEP, *STEP]
    assert _command(traj[2]) == write_command(GOOD)
    assert _command(traj[4]) == f"echo {LAST_OUTPUT}"
    assert LAST_OUTPUT in traj[5]["content"]
    # the workspace as the stop left it
    assert FIX in diff
    # nothing was submitted, so there is still no patch and no check
    assert _names(out, att) == STOPPED_FILES
    # what the model saw is unchanged: two requests, and never the last command's output
    bodies = _bodies_of(mock, "a")
    assert len(bodies) == 2 and LAST_OUTPUT not in json.dumps(bodies)
    assert [m["role"] for m in bodies[-1]["messages"]] == ["system", "user", *STEP]
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.success is False and run.meta["status"] == "context_limit"
    assert guard.blocked == []


def test_a_run_ended_by_a_provider_error_keeps_its_conversation_and_diff(
    tmp_path: Path, mock: Any, guard: NetworkGuard
) -> None:
    def fails_after_the_fix(step: int, rec: Received) -> Reply | None:
        return Reply(kind="error", status=500) if step >= 1 else None

    agent = ScriptedAgent(overrides={"a": fails_after_the_fix})
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    atts = _attempts(out)
    # retried as before: 1 + max_retries attempts, each one step and then 1 + call_retries
    # failed calls
    assert [a.status for a in atts] == ["provider_error"] * 3
    assert sorted(agent.seen) == sorted([("a", 0)] * 3 + [("a", 1)] * 9)
    for att in atts:
        end = _end(att)
        assert end["success"] is None and end["steps"] == 1 and end["calls"] == 1
        assert end["reason"].startswith("provider answered HTTP 500")
        traj, diff = _stored(out, att)
        assert traj is not None and diff is not None
        # the step made before the provider failed, and what it did to the workspace
        assert [m["role"] for m in traj] == ["system", "user", *STEP]
        assert _command(traj[2]) == write_command(GOOD)
        assert FIX in diff
        assert _names(out, att) == STOPPED_FILES
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.success is None and run.meta["status"] == "provider_error"
    assert run.meta["attempts"] == 3
    assert guard.blocked == []


def test_a_paid_run_stopped_at_its_cap_keeps_its_conversation_and_diff(
    tmp_path: Path, mock: Any
) -> None:
    mock.set_responder(ScriptedAgent(extra_steps=50))  # harmless `ls` steps: it never submits
    plan, manifest = build(tmp_path, [HAIKU], tasks=["add-a"])
    limits = Limits(
        call_backoff_s=0.0, cache_canary_runs=0, run_cap_usd=0.002, max_prompt_tokens=50_000
    )
    out = tmp_path / "out"
    runner(
        plan, manifest, config_for(mock, limits=limits), out, environ=environ_with("anthropic")
    ).run(execute=True)
    (att,) = _attempts(out)
    end = _end(att)
    assert end["status"] == "run_cap" and end["success"] is False and end["cost_usd"] > 0
    assert end["steps"] == end["calls"] >= 1
    traj, diff = _stored(out, att)
    # every step made before the cap, and a workspace that `ls` left as it was
    assert traj is not None and diff == ""
    assert [m["role"] for m in traj] == ["system", "user", *STEP * end["steps"]]
    assert _command(traj[-2]) == "ls"
    assert _names(out, att) == STOPPED_FILES


def test_a_stopped_runs_files_pass_through_the_redactor(tmp_path: Path, mock: Any) -> None:
    secret = fake_key("stopped")  # stands for a value loaded from --env-file
    # the last command before the stop writes the value to its output and to a new file
    out, att = _run_to_the_context_limit(
        tmp_path, mock, f"echo {secret} | tee leak.txt", secrets=[secret]
    )
    traj, diff = _stored(out, att)
    assert traj is not None and diff is not None
    assert _command(traj[4]) == "echo [redacted] | tee leak.txt"
    assert "[redacted]" in traj[5]["content"]
    assert "leak.txt" in diff and "+[redacted]" in diff
    for p in out.rglob("*"):
        if p.is_file():
            assert secret not in p.read_text("utf-8", errors="replace"), p


def test_an_interrupted_run_keeps_its_conversation_and_diff(tmp_path: Path, mock: Any) -> None:
    holder: dict[str, Any] = {}

    def stop_during_step_1(step: int, rec: Received) -> Reply | None:
        if step == 1:
            holder["runner"].stop.set()  # what Ctrl-C does; this call's reply still arrives
        return None

    mock.set_responder(ScriptedAgent(overrides={"a": stop_during_step_1}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    holder["runner"] = runner(plan, manifest, config_for(mock), out, environ={})
    holder["runner"].run(execute=True)
    (att,) = _attempts(out)
    end = _end(att)
    assert end["status"] == "interrupted" and end["reason"] == "the runner is stopping"
    assert end["success"] is None and end["steps"] == 2
    traj, diff = _stored(out, att)
    assert traj is not None and diff is not None
    assert [m["role"] for m in traj] == ["system", "user", *STEP, *STEP]
    assert FIX in diff


# --- runs ended by an error ------------------------------------------------------------------


def test_a_configuration_mismatch_keeps_the_conversation_of_its_call(
    tmp_path: Path, mock: Any
) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [("qwen3.5:9b", "none")], tasks=["add-a"])

    def wrong(spec: Any, cat: Any) -> FixedPolicy:  # a bug that runs another effort
        return FixedPolicy(Action(spec.model, "medium"), cat, name="replay-fixed")

    out = tmp_path / "out"
    code = runner(plan, manifest, config_for(mock), out, environ={}, policy=wrong).run(execute=True)
    assert code == EXIT_FATAL
    (att,) = _attempts(out)
    end = _end(att)
    assert end["status"] == "config_mismatch"
    assert "planned qwen3.5:9b (none) but executed qwen3.5:9b (medium)" in end["reason"]
    # the mismatch is found on the first call: its two messages, and an untouched workspace
    traj, diff = _stored(out, att)
    assert traj is not None and [m["role"] for m in traj] == ["system", "user"]
    assert diff == ""


class _DiesOnTheSecondCommand(LocalWorkspace):
    """A workspace that fails under the scaffold: an error of the runner, not of the model."""

    commands = 0

    def execute(self, command: str, *, timeout_s: float) -> ExecResult:
        self.commands += 1
        if self.commands == 2:
            raise OSError("the container is gone")
        return super().execute(command, timeout_s=timeout_s)


def test_a_runner_error_under_the_scaffold_keeps_the_conversation_of_the_last_call(
    tmp_path: Path, mock: Any
) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    limits = Limits(call_backoff_s=0.0, cache_canary_runs=0, max_retries=0)
    cfg = config_for(mock, limits=limits)
    runner(plan, manifest, cfg, out, environ={}, workspace=_DiesOnTheSecondCommand).run(
        execute=True
    )
    (att,) = _attempts(out)
    end = _end(att)
    assert end["status"] == "infra_error" and end["success"] is None
    assert end["reason"] == "runner error: OSError: the container is gone"
    # the conversation as the second call carried it: the reply to that call, whose command
    # failed, never reached the client again and is not in it
    traj, diff = _stored(out, att)
    assert traj is not None and diff is not None
    assert [m["role"] for m in traj] == ["system", "user", *STEP]
    assert FIX in diff


# --- storing the files never changes how the run ended ---------------------------------------


@pytest.mark.parametrize("broken", ["trajectory.json", "workspace.diff", "diff()"])
def test_a_failure_to_store_one_file_leaves_the_run_and_the_other_file_alone(
    tmp_path: Path, mock: Any, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    class NoDiff(LocalWorkspace):
        def diff(self) -> str:
            raise OSError("the container is gone")  # not the WorkspaceError a diff may raise

    write = Path.write_text

    def write_text(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name == broken:
            raise OSError("no space left on device")
        return write(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    deps: dict[str, Any] = {"workspace": NoDiff} if broken == "diff()" else {}
    # the status, the reason and the step count are checked in the helper
    out, att = _run_to_the_context_limit(tmp_path, mock, **deps)
    traj, diff = _stored(out, att)
    if broken == "trajectory.json":
        assert traj is None and diff is not None and FIX in diff
    else:
        assert diff is None and traj is not None and len(traj) == 6
    assert "result.json" in _names(out, att)


# --- runs the scaffold ended are stored as before --------------------------------------------


@pytest.mark.parametrize(("status", "step_limit"), [("passed", 250), ("step_limit", 2)])
def test_a_run_the_scaffold_ended_still_stores_the_scaffolds_own_result(
    tmp_path: Path, mock: Any, status: str, step_limit: int
) -> None:
    results: list[AgentResult] = []

    class Recording(BashAgent):
        def run(self, task_text: str, workspace: Any, client: Any, *, nonce: str) -> AgentResult:
            results.append(super().run(task_text, workspace, client, nonce=nonce))
            return results[-1]

    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    runner(
        plan,
        manifest,
        config_for(mock),
        out,
        environ={},
        scaffold=lambda: Recording(step_limit=step_limit),
    ).run(execute=True)
    (att,) = _attempts(out)
    (result,) = results
    end = _end(att)
    assert end["status"] == status and end["steps"] == result.steps
    traj, diff = _stored(out, att)
    assert diff is not None and FIX in diff
    # the scaffold's history, which ends after the last request the client carried
    assert traj == result.messages
    last_request = _bodies_of(mock, "a")[-1]["messages"]
    assert len(traj) == len(last_request) + 2
    if status == "passed":
        assert traj[-1]["content"] == "submitted"
        assert _names(out, att) == ["check.json", "patch.diff", *STOPPED_FILES]
    else:
        assert _names(out, att) == STOPPED_FILES
