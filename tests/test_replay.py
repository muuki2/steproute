"""The replay runner (issue 18), end to end against the loopback mock upstream.

Acceptance criteria: resume without duplicates or losses (``test_crash_and_resume...``), the
spend cap (``test_spend_cap...``), a month rollover (``test_month_rollover...``) and a 3-task,
2-configuration smoke grid (``test_smoke_grid...``). No test opens a connection beyond
loopback (``NetworkGuard``), and no test reads a real key: every key is built at run time.
"""

from __future__ import annotations

import io
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

from proxy_harness import NetworkGuard  # noqa: E402
from replay_harness import (  # noqa: E402
    MANIFEST,
    ScriptedAgent,
    build,
    config_for,
    environ_with,
    fake_key,
    runner,
)
from steproute.collectors.mock_upstream import MockUpstream, Reply  # noqa: E402
from steproute.core.catalog import load_default_catalog  # noqa: E402
from steproute.core.policy import FixedPolicy  # noqa: E402
from steproute.core.schema import Action  # noqa: E402
from steproute.core.store import TraceStore  # noqa: E402
from steproute.eval.data import load_grid  # noqa: E402
from steproute.replay.config import Limits  # noqa: E402
from steproute.replay.journal import Journal, LockHeld, OutputLock  # noqa: E402
from steproute.replay.ledger import Ledger  # noqa: E402
from steproute.replay.runner import EXIT_CAP, EXIT_FATAL, MissingKey, run_bound  # noqa: E402

LOCAL = ("qwen3-coder:30b", "medium")
HAIKU = ("claude-haiku-5-5", "low")


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
    g = load_grid(out / "grid.jsonl")
    return {(r.task_id, r.model, r.effort): r for r in g.runs}


# --- criterion 4: the smoke grid -------------------------------------------------------------


def test_smoke_grid_three_tasks_two_configurations(
    tmp_path: Path, mock: Any, guard: NetworkGuard
) -> None:
    agent = ScriptedAgent()
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [LOCAL, HAIKU])
    out = tmp_path / "out"
    t0 = time.monotonic()
    code = runner(plan, manifest, config_for(mock), out, environ=environ_with("anthropic")).run(
        execute=True
    )
    assert code == 0
    assert time.monotonic() - t0 < 30
    grid = _grid(out)
    assert len(grid) == 6
    for model, effort in (LOCAL, HAIKU):
        assert grid[("add-a", model, effort)].success is True
        assert grid[("add-b", model, effort)].success is True
        assert grid[("add-c", model, effort)].success is False
    for (_tid, model, _), r in grid.items():
        if model == LOCAL[0]:
            assert r.cost_usd == 0.0
        else:
            assert r.cost_usd is not None and r.cost_usd > 0
        assert r.meta["status"] in ("passed", "failed")
        assert r.session_id and r.session_id.startswith("replay-")
    # Per-step traces through the proxy, every one with source = "replay" and the run's session.
    pairs = list(TraceStore(out / "traces").joined())
    assert len(pairs) == 6 * 3
    sessions = {r.session_id for r in grid.values()}
    for dec, outcome in pairs:
        assert dec.state.source == "replay"
        assert dec.state.session_id in sessions
        assert dec.policy == "replay-fixed"
        assert outcome is not None
        # the task-level label is copied onto every step
        assert outcome.signals.get("replay_status") in ("passed", "failed")
        assert outcome.success is (outcome.signals["replay_status"] == "passed")
    # what reached the "provider": the planned model, and the per-run nonce at the start
    bodies = mock.messages_bodies()
    assert {b["model"] for b in bodies} == {"claude-haiku-5-5"}
    nonces = {b["system"][0]["text"].split("\n")[0] for b in bodies}
    assert len(nonces) == 3  # one per run: no two runs share a cache prefix
    assert all(b.get("cache_control") == {"type": "ephemeral"} for b in bodies)
    assert guard.blocked == []
    # the ledger agrees with the grid for the paid runs
    led = Ledger(out / "ledger.jsonl")
    paid = sum(r.cost_usd or 0 for r in grid.values() if r.model == HAIKU[0])
    assert led.spent("anthropic") == pytest.approx(paid)
    assert not led.open_intents()


def test_local_only_run_leaves_an_empty_ledger(tmp_path: Path, mock: Any) -> None:
    # Found on the first real local run (issue 56): no ledger file at all, so the run's
    # directory could not show that nothing was spent.
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    assert runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True) == 0
    assert (out / "ledger.jsonl").read_text() == ""
    assert Ledger(out / "ledger.jsonl").by_month() == {}
    assert _grid(out)[("add-a", *LOCAL)].success is True


def test_dry_run_is_the_default_and_touches_nothing(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL, HAIKU])
    out = tmp_path / "out"
    buf = io.StringIO()
    r = runner(plan, manifest, config_for(mock), out, environ=environ_with("anthropic"), stdout=buf)
    assert r.run() == 0
    assert not out.exists()
    assert mock.received == []
    text = buf.getvalue()
    assert "DRY RUN" in text and "cap $80.00/month" in text and "month-to-date" in text
    assert fake_key("anthropic") not in text and "STEPROUTE_TEST_ANTHROPIC: set" in text


# --- money safety --------------------------------------------------------------------------


def test_missing_deepseek_key_exits_before_any_call(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL, ("deepseek-flash", "low"), HAIKU])
    out = tmp_path / "out"
    buf = io.StringIO()
    r = runner(plan, manifest, config_for(mock), out, environ=environ_with("anthropic"), stdout=buf)
    with pytest.raises(MissingKey):
        r.run(execute=True)
    assert mock.received == []
    assert not (out / "ledger.jsonl").exists() and not (out / "runs.jsonl").exists()
    assert "DeepSeek configurations are required" in buf.getvalue()


EFFORTS3 = [HAIKU, ("claude-haiku-5-5", "medium"), ("claude-haiku-5-5", "high")]


def _expensive(mock: Any) -> Any:
    """Runs of about $0.15 each (prompts of 80K to 128K tokens, Haiku's upper tier), a per-run
    cap of $1, so the worst-case bound of a run is about $1.17."""
    mock.set_responder(ScriptedAgent(scale=40))
    return Limits(call_backoff_s=0.0, cache_canary_runs=0, run_cap_usd=1.0)


def test_spend_cap_stops_within_one_run_of_the_limit(tmp_path: Path, mock: Any) -> None:
    limits = _expensive(mock)
    plan, manifest = build(tmp_path, EFFORTS3)
    cap = 1.4
    cfg = config_for(mock, limits=limits, caps={"anthropic": cap, "deepseek": 20.0})
    bound = run_bound(plan.specs[0], load_default_catalog(), cfg)
    assert 1.0 < bound < cap
    out = tmp_path / "out"
    buf = io.StringIO()
    code = runner(plan, manifest, cfg, out, environ=environ_with("anthropic"), stdout=buf).run(
        execute=True
    )
    assert code == EXIT_CAP
    spent = Ledger(out / "ledger.jsonl").spent("anthropic")
    ran = len(Journal.open(out / "runs.jsonl").attempts)
    assert 2 <= ran < len(plan.specs)
    assert spent <= cap  # never passed, counting realised cost
    assert spent + bound > cap  # and stopped only when the next run could pass it
    per_run = spent / ran
    assert cap - spent < bound + per_run  # within one run of the limit
    assert "monthly cap" in buf.getvalue()


def test_parallel_runs_cannot_jointly_pass_the_cap(
    tmp_path: Path, mock: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = ScriptedAgent(delay_s=0.15)
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [HAIKU, ("claude-haiku-5-5", "medium")])
    limits = Limits(
        call_backoff_s=0.0, cache_canary_runs=0, max_prompt_tokens=20_000, max_output_tokens=1_000
    )
    cfg = config_for(mock, limits=limits, paid=3)
    bound = run_bound(plan.specs[0], load_default_catalog(), cfg)
    cap = 2.2 * bound  # two runs may be in flight together, never three
    cfg = config_for(mock, limits=limits, caps={"anthropic": cap, "deepseek": 20.0}, paid=3)
    out = tmp_path / "out"
    from steproute.replay import ledger as ledger_mod

    held: list[int] = []
    orig = ledger_mod.CapGuard.admit

    def admit(self: Any, provider: str, run_id: str, bound: float) -> None:
        orig(self, provider, run_id, bound)
        held.append(len(self._held))

    monkeypatch.setattr(ledger_mod.CapGuard, "admit", admit)
    runner(plan, manifest, cfg, out, environ=environ_with("anthropic")).run(execute=True)
    assert max(held) == 2  # three workers, but never three reservations at once
    j = Journal.open(out / "runs.jsonl")
    assert len(j.attempts) == len(plan.specs)  # every run still ran, two at a time
    spans = [(a.started, float((a.end or {})["ts"])) for a in j.attempts.values()]
    overlap = max(sum(1 for s, e in spans if s <= t < e) for t, _ in spans)
    assert overlap <= 2
    assert Ledger(out / "ledger.jsonl").spent("anthropic") <= cap


def test_missing_usage_is_charged_at_worst_case_never_zero(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent(usage=False))
    plan, manifest = build(tmp_path, [("deepseek-flash", "low")], tasks=["add-a"])
    limits = Limits(
        call_backoff_s=0.0, cache_canary_runs=0, max_prompt_tokens=20_000, max_output_tokens=1_000
    )
    cfg = config_for(mock, limits=limits)
    out = tmp_path / "out"
    runner(plan, manifest, cfg, out, environ=environ_with("deepseek")).run(execute=True)
    led = Ledger(out / "ledger.jsonl")
    flagged = led.flagged()
    assert flagged and all("missing_usage" in i.flags for i in flagged)
    worst = run_bound(plan.specs[0], load_default_catalog(), cfg) - limits.run_cap_usd
    assert all(i.settled_usd == pytest.approx(worst) for i in flagged)
    run = _grid(out)[("add-a", "deepseek-flash", "low")]
    assert run.cost_usd == pytest.approx(worst * len(flagged))
    assert run.meta["calls_missing_usage"] == len(flagged)
    # the trace keeps the cost unknown, not $0
    outcomes = [o for _, o in TraceStore(out / "traces").joined() if o is not None]
    assert outcomes and all(o.cost_usd is None and o.signals.get("usage_missing") for o in outcomes)


def test_per_run_cap_ends_the_run_as_a_labelled_failure_not_retried(
    tmp_path: Path, mock: Any
) -> None:
    mock.set_responder(ScriptedAgent(extra_steps=50))
    plan, manifest = build(tmp_path, [HAIKU], tasks=["add-a"])
    limits = Limits(
        call_backoff_s=0.0, cache_canary_runs=0, run_cap_usd=0.002, max_prompt_tokens=50_000
    )
    out = tmp_path / "out"
    runner(
        plan, manifest, config_for(mock, limits=limits), out, environ=environ_with("anthropic")
    ).run(execute=True)
    run = _grid(out)[("add-a", *HAIKU)]
    assert run.meta["status"] == "run_cap" and run.success is False and run.meta["attempts"] == 1


def test_executed_configuration_must_match_the_plan(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [("qwen3.5:9b", "none")])
    catalog = load_default_catalog()

    def wrong(spec: Any, cat: Any) -> FixedPolicy:  # a bug that runs another effort
        return FixedPolicy(Action(spec.model, "medium"), cat, name="replay-fixed")

    out = tmp_path / "out"
    buf = io.StringIO()
    code = runner(plan, manifest, config_for(mock), out, environ={}, policy=wrong, stdout=buf).run(
        execute=True
    )
    assert code == EXIT_FATAL
    j = Journal.open(out / "runs.jsonl")
    statuses = [a.status for a in j.attempts.values()]
    assert statuses == ["config_mismatch"]  # and the runner stopped: no further run started
    assert "planned qwen3.5:9b (none) but executed qwen3.5:9b (medium)" in buf.getvalue()
    assert catalog.get("qwen3.5:9b").local


def test_provider_errors_are_retried_with_a_bound_and_never_labelled(
    tmp_path: Path, mock: Any
) -> None:
    agent = ScriptedAgent(overrides={"a": lambda step, rec: Reply(kind="error", status=500)})
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a", "add-b"])
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    grid = _grid(out)
    a = grid[("add-a", *LOCAL)]
    assert a.success is None and a.meta["status"] == "provider_error"
    assert a.meta["attempts"] == 3  # 1 + max_retries
    assert sum(1 for k, s in agent.seen if k == "a") == 3 * 3  # each attempt: 1 + call_retries
    assert grid[("add-b", *LOCAL)].success is True


def test_a_call_retried_after_an_upstream_error_is_not_labelled_a_failure(
    tmp_path: Path, mock: Any
) -> None:
    # Seen on the first real Ollama run (issue 56): one HTTP 500 from Ollama, retried by the
    # runner, left that step's decision labelled success = false (the proxy's "retried"
    # signal), a model failure in the training data for what was an infrastructure error.
    failed_once: list[int] = []

    def flaky(step: int, rec: Any) -> Reply | None:
        if step == 1 and not failed_once:
            failed_once.append(1)
            return Reply(kind="error", status=500)
        return None

    mock.set_responder(ScriptedAgent(overrides={"a": flaky}))
    plan, manifest = build(tmp_path, [LOCAL], tasks=["add-a"])
    out = tmp_path / "out"
    assert runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True) == 0
    assert _grid(out)[("add-a", *LOCAL)].success is True
    pairs = list(TraceStore(out / "traces").joined())
    assert len(pairs) == 4  # three steps that answered, one that got the 500
    errored = [(d, o) for d, o in pairs if o is not None and o.final_model is None]
    assert len(errored) == 1
    _, o = errored[0]
    assert o.success is None and o.signals["replay_call_failed"] is True
    assert o.signals["replay_status"] == "passed" and "tests_passed" not in o.signals
    assert all(o.success is True for _, o in pairs if o is not None and o.final_model)


def test_keys_never_reach_any_output(tmp_path: Path, mock: Any) -> None:
    key = fake_key("anthropic")

    def leaky(step: int, rec: Any) -> Reply | None:
        if step == 0:  # the model tries to read the environment
            return Reply(kind="tool", tool_name="bash", tool_args=json.dumps({"command": "env"}))
        return None

    agent = ScriptedAgent(overrides={"a": leaky}, extra_steps=0)
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [HAIKU], tasks=["add-a"])
    out = tmp_path / "out"
    buf = io.StringIO()
    runner(
        plan, manifest, config_for(mock), out, environ=environ_with("anthropic"), stdout=buf
    ).run(execute=True)
    for p in out.rglob("*"):
        if p.is_file():
            assert key not in p.read_text("utf-8", errors="replace"), p
    assert key not in buf.getvalue()
    # the command's output went back to the model: check it had no key in it
    for b in mock.messages_bodies():
        assert key not in json.dumps(b)


# --- resume, rollover, locking -------------------------------------------------------------


class _Crash(BaseException):
    """The process dying: not an Exception, so nothing in the runner catches it."""


def test_crash_and_resume_neither_duplicates_nor_loses_runs(
    tmp_path: Path, mock: Any, guard: NetworkGuard, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL, HAIKU])
    out = tmp_path / "out"
    cfg = config_for(mock)
    env = environ_with("anthropic")
    crash = {"on": True}
    from steproute.replay.agent import BashAgent

    class Dying(BashAgent):
        """Makes one paid call on task b, then the "process dies": the paid lane's thread
        ends with no end line written, exactly what a kill leaves on disk."""

        def run(self, task_text: str, workspace: Any, client: Any, *, nonce: str) -> Any:
            if crash["on"] and "TASK-KEY: b" in task_text and client.spec.provider == "anthropic":
                client.complete(self.initial_messages(task_text, nonce), [])
                raise _Crash()
            return super().run(task_text, workspace, client, nonce=nonce)

    monkeypatch.setattr(threading, "excepthook", lambda args: None)
    runner(plan, manifest, cfg, out, environ=env, scaffold=Dying).run(execute=True)
    j = Journal.open(out / "runs.jsonl")
    in_flight = j.in_flight()
    assert in_flight, "a run must have been left in flight"
    finished_before = {a.cell: a.run_id for a in j.finished()}
    spent_before = Ledger(out / "ledger.jsonl").spent("anthropic")
    assert spent_before > 0

    crash["on"] = False
    buf = io.StringIO()
    assert runner(plan, manifest, cfg, out, environ=env, stdout=buf).run(execute=True) == 0
    assert "was in flight when the runner stopped" in buf.getvalue()
    grid = load_grid(out / "grid.jsonl")  # raises on a duplicated run
    assert len(grid.runs) == 6
    j2 = Journal.open(out / "runs.jsonl")
    for cell, run_id in finished_before.items():
        assert next(a for a in j2.finished() if a.cell == cell).run_id == run_id  # not re-run
    statuses = sorted(a.status or "" for a in j2.attempts.values())
    assert statuses.count("interrupted") == len(in_flight)
    # the interrupted attempt's spend stays in the ledger
    assert Ledger(out / "ledger.jsonl").spent("anthropic") >= spent_before
    assert guard.blocked == []


def _ts(y: int, m: int, d: int, h: int = 12) -> float:
    return datetime(y, m, d, h, tzinfo=UTC).timestamp()


def test_month_rollover_resets_spend_and_resume_repeats_nothing(tmp_path: Path, mock: Any) -> None:
    limits = _expensive(mock)
    plan, manifest = build(tmp_path, EFFORTS3)
    cfg = config_for(mock, limits=limits, caps={"anthropic": 1.4, "deepseek": 20.0})
    out = tmp_path / "out"
    now = {"t": _ts(2026, 11, 30, 23)}
    env = environ_with("anthropic")
    code = runner(plan, manifest, cfg, out, environ=env, clock=lambda: now["t"]).run(execute=True)
    assert code == EXIT_CAP
    done_nov = {a.cell: a.run_id for a in Journal.open(out / "runs.jsonl").finished()}
    assert 0 < len(done_nov) < len(plan.specs)
    nov = Ledger(out / "ledger.jsonl", clock=lambda: now["t"]).spent("anthropic")
    assert 0 < nov <= 1.4

    now["t"] = _ts(2026, 12, 1, 0)  # midnight UTC: a new month
    assert Ledger(out / "ledger.jsonl", clock=lambda: now["t"]).spent("anthropic") == 0.0
    code = runner(plan, manifest, cfg, out, environ=env, clock=lambda: now["t"]).run(execute=True)
    assert code == 0
    j = Journal.open(out / "runs.jsonl")
    done = {a.cell: a.run_id for a in j.finished()}
    assert len(done) == len(plan.specs)
    for cell, rid in done_nov.items():
        assert done[cell] == rid  # not repeated
    assert len(j.attempts) == len(plan.specs)  # nothing ran twice
    led = Ledger(out / "ledger.jsonl", clock=lambda: now["t"])
    assert led.spent("anthropic", "2026-11") == pytest.approx(nov)
    assert 0 < led.spent("anthropic", "2026-12") <= 1.4
    assert len(load_grid(out / "grid.jsonl").runs) == len(plan.specs)


def test_two_runners_on_one_directory(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL])
    out = tmp_path / "out"
    with OutputLock(out):
        with pytest.raises(LockHeld):
            runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    assert mock.received == []
    # a lock held by another process
    import subprocess
    import sys

    from steproute.tasks.check import workspace_env

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; from steproute.replay.journal import OutputLock; "
            "from pathlib import Path; l = OutputLock(Path(sys.argv[1])).acquire(); "
            "print('held', flush=True); time.sleep(30)",
            str(out),
        ],
        stdout=subprocess.PIPE,
        text=True,
        env=workspace_env(),
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
        with pytest.raises(LockHeld, match="another replay runner"):
            runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    finally:
        holder.kill()
        holder.wait()
    assert mock.received == []


def test_holdout_models_go_to_a_separate_split(tmp_path: Path, mock: Any) -> None:
    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [("devstral:24b", "medium"), LOCAL], tasks=["add-a"])
    assert [s.model for s in plan.specs] == ["qwen3-coder:30b", "devstral:24b"]  # holdout pass last
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ={}).run(execute=True)
    pool = load_grid(out / "grid.jsonl")
    held = load_grid(out / "grid-holdout.jsonl")
    assert {r.model for r in pool.runs} == {"qwen3-coder:30b"}
    assert {r.model for r in held.runs} == {"devstral:24b"}
    assert held.header["split"] == "holdout"
    assert all(d.action.model != "devstral:24b" for d, _ in TraceStore(out / "traces").joined())
    assert {d.action.model for d, _ in TraceStore(out / "traces-holdout").joined()} == {
        "devstral:24b"
    }


# --- the report ------------------------------------------------------------------------------


def test_pilot_report_on_mock_data_is_marked_mock(tmp_path: Path, mock: Any) -> None:
    from steproute.replay.plan import build_named_plan
    from steproute.replay.report import pilot_report
    from steproute.tasks.estimate import load_assumptions
    from steproute.tasks.manifest import load_manifest

    mock.set_responder(ScriptedAgent())
    plan, manifest = build(tmp_path, [LOCAL, HAIKU])
    out = tmp_path / "out"
    runner(plan, manifest, config_for(mock), out, environ=environ_with("anthropic")).run(
        execute=True
    )
    cat = load_default_catalog()
    m = load_manifest(MANIFEST)
    a = load_assumptions()
    md = pilot_report(
        out,
        catalog=cat,
        manifest=m,
        pilot=build_named_plan("pilot", m, cat, a),
        sparse=build_named_plan("sparse", m, cat, a),
    )
    assert md.startswith("> **MOCK DATA. NOT A PILOT.**")
    assert "| claude-haiku-5-5 | low | 3 | 67% (2/3) |" in md
    assert "| qwen3-coder:30b | medium | 3 | 67% (2/3) |" in md
    assert "re-estimated from the pilot" in md
    assert "| anthropic | 2026-" in md or "| anthropic | " in md


def test_threads_left_behind_are_daemons() -> None:
    assert all(
        t.daemon or t is threading.main_thread() for t in threading.enumerate() if t.is_alive()
    )
