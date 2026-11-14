"""DeepSeek off-peak scheduling, time-of-day pricing and the $10 budget (issue 57).

Everything runs against the loopback mock upstream with a fake wall clock: no provider is
called, no real key exists, and no test depends on the hour it runs at.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")

from proxy_harness import NetworkGuard  # noqa: E402
from replay_harness import ScriptedAgent, build, config_for, environ_with, fake_key  # noqa: E402
from steproute.collectors.mock_upstream import MockUpstream, Received, Reply  # noqa: E402
from steproute.core.catalog import load_default_catalog  # noqa: E402
from steproute.core.offpeak import peak_hours  # noqa: E402
from steproute.eval.data import load_grid  # noqa: E402
from steproute.replay.agent import BASH_TOOL, AgentResult  # noqa: E402
from steproute.replay.config import DEFAULT_SCHEDULES, Limits, Schedule  # noqa: E402
from steproute.replay.journal import Journal  # noqa: E402
from steproute.replay.ledger import Ledger, read_lines  # noqa: E402
from steproute.replay.plan import ReplayPlan, build_named_plan  # noqa: E402
from steproute.replay.runner import EXIT_CAP, Runner, RunnerDeps, worst_call  # noqa: E402
from steproute.tasks.check import CheckResult  # noqa: E402
from steproute.tasks.estimate import load_assumptions  # noqa: E402
from steproute.tasks.manifest import load_default_manifest  # noqa: E402

CAT = load_default_catalog()
HOURS = peak_hours(CAT.get("deepseek-flash"))
assert HOURS is not None
FLASH = ("deepseek-flash", "low")
SONNET = ("claude-sonnet-5-5", "low")


def ts(d: int, h: int, m: int = 0) -> float:
    """An instant in the week of Monday 16 November 2026 (d=16 Monday ... 22 Sunday)."""
    return datetime(2026, 11, d, h, m, tzinfo=UTC).timestamp()


class FakeClock:
    """A shared wall clock. ``sleep`` advances it; nothing waits for real."""

    def __init__(self, start: float) -> None:
        self.t = start
        self.lock = threading.Lock()
        self.slept = 0.0

    def __call__(self) -> float:
        with self.lock:
            return self.t

    def advance(self, s: float) -> None:
        with self.lock:
            self.t += s

    def sleep(self, s: float) -> None:
        self.slept += s
        self.advance(s)


@pytest.fixture
def guard(monkeypatch: pytest.MonkeyPatch) -> NetworkGuard:
    g = NetworkGuard()
    g.install(monkeypatch)
    return g


@pytest.fixture
def mock() -> Any:
    with MockUpstream() as m:
        yield m


def _deepseek_lines(out: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, _ = read_lines(out / "ledger.jsonl")
    intents = [r for r in rows if r["kind"] == "intent" and r["provider"] == "deepseek"]
    ids = {r["id"] for r in intents}
    settles = [r for r in rows if r["kind"] == "settle" and r["id"] in ids]
    return intents, settles


def _timed(agent: ScriptedAgent, clock: FakeClock, seconds: float) -> Callable[[Received], Any]:
    """The scripted agent, with every call taking ``seconds`` of fake wall time."""

    def respond(rec: Received) -> Any:
        clock.advance(seconds)
        return agent(rec)

    return respond


def _runner(plan: Any, manifest: Any, cfg: Any, out: Path, clock: FakeClock, **kw: Any) -> Runner:
    buf = kw.pop("stdout", None) or io.StringIO()
    return Runner(
        plan,
        config=cfg,
        catalog=CAT,
        manifest=manifest,
        out=out,
        deps=RunnerDeps(
            environ=kw.pop("environ", environ_with("deepseek", "anthropic")),
            out=buf,
            clock=clock,
            sleep=clock.sleep,
            **kw,
        ),
    )


# --- the start gate, the wait message and the pause backstop --------------------------------


def test_runs_start_only_off_peak_with_the_margin_and_say_until_when(
    tmp_path: Path, mock: Any, guard: NetworkGuard
) -> None:
    clock = FakeClock(ts(16, 0, 30))  # Monday 00:30 UTC: off-peak, but peak starts at 01:00
    mock.set_responder(_timed(ScriptedAgent(), clock, 30))
    plan, manifest = build(tmp_path, [FLASH], tasks=["add-a", "add-b"])
    cfg = config_for(mock, schedules=dict(DEFAULT_SCHEDULES))
    out = tmp_path / "out"
    buf = io.StringIO()
    assert _runner(plan, manifest, cfg, out, clock, stdout=buf).run(execute=True) == 0
    text = buf.getvalue()
    # 30 min left is less than the 2 h margin (the run timeout); 04:00-06:00 is exactly 2 h.
    assert "the next start is 2026-11-16 04:00 UTC (" in text and "local)" in text
    assert "nothing else to run, waiting" in text
    intents, settles = _deepseek_lines(out)
    assert intents and len(settles) == len(intents)
    assert min(r["ts"] for r in intents) >= ts(16, 4)
    assert all(not HOURS.is_peak(r["ts"]) for r in intents)  # no call sent in peak hours
    assert {r["period"] for r in settles} == {"offpeak"}
    for r in settles:  # billed at half the list price
        assert r["usd"] == pytest.approx(r["list_usd"] / 2)
    grid = {(g.task_id, g.model): g for g in load_grid(out / "grid.jsonl").runs}
    run = grid[("add-a", FLASH[0])]
    assert run.cost_usd == pytest.approx(run.meta["list_cost_usd"] / 2)
    assert run.meta["calls_by_period"] == {"offpeak": 3}
    assert guard.blocked == []


def test_other_providers_keep_running_while_deepseek_is_gated(tmp_path: Path, mock: Any) -> None:
    clock = FakeClock(ts(17, 7))  # Tuesday 07:00: peak
    mock.set_responder(_timed(ScriptedAgent(), clock, 5))
    plan, manifest = build(tmp_path, [FLASH, SONNET], tasks=["add-a", "add-b"])
    assert [s.model for s in plan.specs][:2] == [FLASH[0], FLASH[0]]  # DeepSeek is first
    cfg = config_for(mock, schedules=dict(DEFAULT_SCHEDULES))
    out = tmp_path / "out"
    buf = io.StringIO()
    assert _runner(plan, manifest, cfg, out, clock, stdout=buf).run(execute=True) == 0
    rows, _ = read_lines(out / "ledger.jsonl")
    order = [r["provider"] for r in rows if r["kind"] == "intent"]
    first_ds = order.index("deepseek")
    assert set(order[:first_ds]) == {"anthropic"} and "deepseek" not in order[:first_ds]
    assert "other providers' runs continue meanwhile" in buf.getvalue()
    intents, _ = _deepseek_lines(out)
    assert min(r["ts"] for r in intents) >= ts(17, 10)  # waited for the window, not dropped
    assert len(load_grid(out / "grid.jsonl").runs) == 4


def test_a_run_reaching_peak_hours_pauses_and_the_pause_is_not_run_time(
    tmp_path: Path, mock: Any
) -> None:
    clock = FakeClock(ts(16, 0, 45))  # 15 min of off-peak left
    real_pause = {"done": False}

    def sleep(s: float) -> None:
        if not real_pause["done"]:  # the pause takes real time too, longer than the timeout
            real_pause["done"] = True
            time.sleep(1.5)
        clock.sleep(s)

    mock.set_responder(_timed(ScriptedAgent(extra_steps=6), clock, 120))
    plan, manifest = build(tmp_path, [FLASH], tasks=["add-a"])
    limits = Limits(call_backoff_s=0.0, cache_canary_runs=0, run_timeout_s=1.0)
    # start margin 0: the run starts at 00:45 and must meet the 01:00 boundary mid-run
    sched = {"deepseek": Schedule(offpeak_only=True, start_margin_s=0.0, call_margin_s=300.0)}
    cfg = config_for(mock, limits=limits, schedules=sched)
    out = tmp_path / "out"
    buf = io.StringIO()
    r = _runner(plan, manifest, cfg, out, clock, stdout=buf)
    r.deps.sleep = sleep
    assert r.run(execute=True) == 0
    assert "paused for deepseek peak hours until 2026-11-16 04:00 UTC" in buf.getvalue()
    intents, settles = _deepseek_lines(out)
    stamps = sorted(i["ts"] for i in intents)
    assert any(t < ts(16, 1) for t in stamps) and any(t >= ts(16, 4) for t in stamps)
    assert not any(HOURS.is_peak(t) for t in stamps)
    assert {s["period"] for s in settles} == {"offpeak"}
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.meta["status"] == "passed"  # not wall_time: the pause did not count
    end = next(iter(Journal.open(out / "runs.jsonl").attempts.values())).end or {}
    assert end["paused_s"] >= 1.5


def test_a_call_crossing_into_peak_is_billed_at_peak(tmp_path: Path, mock: Any) -> None:
    clock = FakeClock(ts(16, 0, 59))
    mock.set_responder(_timed(ScriptedAgent(), clock, 90))  # the first call ends at 01:00:30
    plan, manifest = build(tmp_path, [FLASH], tasks=["add-a"])
    sched = {"deepseek": Schedule(offpeak_only=True, start_margin_s=0.0, call_margin_s=0.0)}
    cfg = config_for(mock, schedules=sched)
    out = tmp_path / "out"
    assert _runner(plan, manifest, cfg, out, clock).run(execute=True) == 0
    intents, settles = _deepseek_lines(out)
    assert settles[0]["period"] == "peak" and settles[0]["usd"] == settles[0]["list_usd"]
    # the next call waited for 04:00 and is off-peak again
    assert all(i["ts"] >= ts(16, 4) for i in intents[1:])
    assert {s["period"] for s in settles[1:]} == {"offpeak"}
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.meta["calls_by_period"] == {"peak": 1, "offpeak": 2}
    assert run.meta["list_cost_usd"] > run.cost_usd


def test_without_the_policy_deepseek_runs_at_any_hour_at_peak_price(
    tmp_path: Path, mock: Any
) -> None:
    clock = FakeClock(ts(16, 7))
    mock.set_responder(_timed(ScriptedAgent(), clock, 5))
    plan, manifest = build(tmp_path, [FLASH], tasks=["add-a"])
    cfg = config_for(mock, schedules={"deepseek": Schedule(offpeak_only=False)})
    out = tmp_path / "out"
    assert _runner(plan, manifest, cfg, out, clock).run(execute=True) == 0
    _, settles = _deepseek_lines(out)
    assert {s["period"] for s in settles} == {"peak"}
    assert all(s["usd"] == s["list_usd"] for s in settles)


def test_dry_run_shows_the_off_peak_estimate_and_the_next_window(tmp_path: Path) -> None:
    manifest = load_default_manifest()
    plan = build_named_plan("pilot", manifest, CAT, load_assumptions())
    buf = io.StringIO()
    clock = FakeClock(ts(16, 2))  # Monday 02:00: peak
    r = Runner(
        plan,
        config=__import__("steproute.replay.config", fromlist=["x"]).load_replay_config(None),
        catalog=CAT,
        manifest=manifest,
        out=tmp_path / "out",
        deps=RunnerDeps(environ={}, out=buf, clock=clock),
    )
    assert r.run() == 0
    text = buf.getvalue()
    line = next(x for x in text.splitlines() if x.strip().startswith("deepseek"))
    assert "~$17.57" in line and "at peak (list) prices" in line and "cap $10.00/month" in line
    assert "total budget $10.00 (all months)" in text
    assert "scheduled off-peak only: ~$8.79" in text and "against $10.00 left" in text
    assert "the central estimate fits" in text
    assert "now 2026-11-16 02:00 UTC (" in text and "is peak" in text
    assert "next start (off-peak with 2 h ahead): 2026-11-16 04:00 UTC (" in text


# --- the budget: the simulated pilot DeepSeek block ----------------------------------------


@dataclass
class CostScaffold:
    """Makes ``calls`` calls whose usage at peak prices adds up to ``factor`` times the run's
    central estimate, then submits. The cost per call travels in the prompt to the mock."""

    factor: float
    calls: int = 4
    name: str = "cost-sim"

    def run(self, task_text: str, workspace: Any, client: Any, *, nonce: str) -> AgentResult:
        per = client.spec.est_usd[0] * self.factor / self.calls
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": f"Run {nonce}."},
            {"role": "user", "content": f"COST={per:.9f} MODEL={client.spec.model}"},
        ]
        for i in range(self.calls):
            client.complete(msgs, [BASH_TOOL])
            msgs += [
                {"role": "assistant", "content": f"step {i}"},
                {"role": "user", "content": "go"},
            ]
        return AgentResult("submitted", "diff --git a/x b/x\n", self.calls, msgs)


@dataclass
class _NoWorkspace:
    def execute(self, command: str, *, timeout_s: float) -> Any:  # pragma: no cover
        raise AssertionError("not used")

    def diff(self) -> str:
        return ""

    def close(self) -> None:
        return None


@dataclass
class _Pass:
    name: str = "fake"

    def check(self, task: Any, patch: str, *, timeout_s: float | None = None) -> CheckResult:
        return CheckResult(task.id, "passed", "fake", "simulated")


@dataclass
class _CostResponder:
    clock: FakeClock
    seconds: float = 15.0
    seen: list[str] = field(default_factory=list)

    def __call__(self, rec: Received) -> Reply:
        self.clock.advance(self.seconds)
        text = json.dumps(rec.body)
        m = re.search(r"COST=([0-9.]+) MODEL=([\w.:-]+)", text)
        assert m is not None
        usd, model = float(m.group(1)), m.group(2)
        p = CAT.get(model)
        out = 10
        prompt = round((usd - out * p.price_out * 1e-6) / (p.price_in * 1e-6))
        self.seen.append(model)
        return Reply(kind="text", content="ok", prompt_tokens=prompt, completion_tokens=out)


def _pilot_deepseek_block() -> tuple[ReplayPlan, Any]:
    manifest = load_default_manifest()
    pilot = build_named_plan("pilot", manifest, CAT, load_assumptions())
    specs = tuple(s for s in pilot.specs if s.provider == "deepseek")
    plan = ReplayPlan("pilot-deepseek", "the pilot's DeepSeek block", specs, pilot.tasks)
    return plan, manifest


def _simulate(tmp_path: Path, mock: Any, factor: float) -> tuple[int, Path, str, FakeClock]:
    plan, manifest = _pilot_deepseek_block()
    assert len(plan.specs) == 100
    assert sum(s.est_usd[0] for s in plan.specs) == pytest.approx(17.57, abs=0.01)
    clock = FakeClock(ts(16, 0))  # Monday 00:00 UTC: the gate and both windows are crossed
    mock.set_responder(_CostResponder(clock))
    cfg = config_for(
        mock,
        caps={"anthropic": 80.0, "deepseek": 10.0},
        totals={"deepseek": 10.0},
        schedules=dict(DEFAULT_SCHEDULES),
        paid=2,
    )
    out = tmp_path / "out"
    buf = io.StringIO()
    r = _runner(
        plan,
        manifest,
        cfg,
        out,
        clock,
        stdout=buf,
        environ=environ_with("deepseek"),
        scaffold=lambda: CostScaffold(factor),
        workspace=lambda task: _NoWorkspace(),
        backend=lambda task: _Pass(),
        task_text=lambda task: "simulated task",
    )
    code = r.run(execute=True)
    return code, out, buf.getvalue(), clock


def _check_never_passed(out: Path, cap: float) -> float:
    """Replays the ledger line by line: spend (an open intent at its worst case) never passes
    ``cap`` at any point, and returns the final spend."""
    rows, bad = read_lines(out / "ledger.jsonl")
    assert bad == 0
    charged: dict[str, float] = {}
    peak = 0.0
    for r in rows:
        if r["kind"] == "intent":
            charged[r["id"]] = r["usd"]
        else:
            charged[r["id"]] = r["usd"]
        peak = max(peak, sum(charged.values()))
    assert peak <= cap + 1e-9
    return sum(charged.values())


def test_the_simulated_pilot_deepseek_block_completes_under_10_dollars(
    tmp_path: Path, mock: Any, guard: NetworkGuard
) -> None:
    code, out, text, clock = _simulate(tmp_path, mock, 1.0)
    assert code == 0, text[-2000:]
    grid = load_grid(out / "grid.jsonl")
    assert len(grid.runs) == 100 and all(r.meta["status"] == "passed" for r in grid.runs)
    led = Ledger(out / "ledger.jsonl", clock=clock)
    spent = led.spent_total("deepseek")
    assert spent == pytest.approx(8.79, abs=0.02)  # the central estimate at off-peak prices
    assert sum(r.meta["list_cost_usd"] for r in grid.runs) == pytest.approx(17.57, abs=0.03)
    assert _check_never_passed(out, 10.0) == pytest.approx(spent)
    intents, settles = _deepseek_lines(out)
    assert not any(HOURS.is_peak(i["ts"]) for i in intents)
    assert {s["period"] for s in settles} == {"offpeak"}
    assert guard.blocked == []
    # The report names both costs: billed and the same usage at list (peak) price.
    from steproute.replay.report import pilot_report

    manifest = load_default_manifest()
    a = load_assumptions()
    md = pilot_report(
        out,
        catalog=CAT,
        manifest=manifest,
        pilot=build_named_plan("pilot", manifest, CAT, a),
        sparse=build_named_plan("sparse", manifest, CAT, a),
    )
    assert "$ total billed | $ total at list price" in md
    assert "| deepseek | offpeak | $8.79 | $17.57 |" in md
    row = next(x for x in md.splitlines() if x.startswith("| deepseek-v4-pro | high |"))
    assert "| 1.00 |" in row  # realised / estimated, both at list price


def test_with_costs_one_and_a_half_times_higher_the_runner_stops_at_10_dollars(
    tmp_path: Path, mock: Any
) -> None:
    code, out, text, clock = _simulate(tmp_path, mock, 1.5)
    assert code == EXIT_CAP
    led = Ledger(out / "ledger.jsonl", clock=clock)
    spent = led.spent_total("deepseek")
    assert spent <= 10.0
    assert _check_never_passed(out, 10.0) <= 10.0
    # the budget is used, not abandoned at the first refusal: less than the floor plus one
    # worst-case v4-pro call is left over
    wc = worst_call(_pilot_deepseek_block()[0].specs[-1], CAT, config_for(mock))
    assert 10.0 - spent < 0.5 + wc
    j = Journal.open(out / "runs.jsonl")
    statuses = [a.status for a in j.attempts.values()]
    assert set(statuses) <= {"passed", "budget_stop"}  # nothing is a failure
    grid = load_grid(out / "grid.jsonl")
    assert all(r.meta["status"] == "passed" for r in grid.runs)
    pending = 100 - len(grid.runs)
    assert pending > 0
    assert f"{pending} cell(s) pending (deepseek {pending})" in text
    assert "Pending cells are not failures" in text


def test_a_reduced_budget_run_ends_as_budget_stop_and_is_resumed_later(
    tmp_path: Path, mock: Any
) -> None:
    clock = FakeClock(ts(21, 12))  # Saturday: off-peak all day
    mock.set_responder(_CostResponder(clock))
    plan, manifest = _pilot_deepseek_block()
    one = ReplayPlan("one", "one run", plan.specs[-1:], plan.tasks)  # v4-pro high
    spec = one.specs[0]
    wc = worst_call(spec, CAT, config_for(mock))
    out = tmp_path / "out"

    def go(total: float, factor: float) -> tuple[int, str]:
        buf = io.StringIO()
        cfg = config_for(mock, totals={"deepseek": total}, schedules=dict(DEFAULT_SCHEDULES))
        r = _runner(
            one,
            manifest,
            cfg,
            out,
            clock,
            stdout=buf,
            environ=environ_with("deepseek"),
            scaffold=lambda: CostScaffold(factor, calls=12),
            workspace=lambda task: _NoWorkspace(),
            backend=lambda task: _Pass(),
            task_text=lambda task: "simulated task",
        )
        return r.run(execute=True), buf.getvalue()

    # Headroom of floor + 0.1 + one worst call: the run gets a $0.60 budget and costs more.
    code, text = go(0.6 + wc, 4.5)
    assert code == EXIT_CAP
    assert "runs with a budget of $0.60 and stops as budget_stop" in text
    j = Journal.open(out / "runs.jsonl")
    (a,) = j.attempts.values()
    assert a.status == "budget_stop" and (a.end or {})["success"] is None
    assert a.start["budget_usd"] == pytest.approx(0.6) and a.start["bound_usd"] == pytest.approx(
        0.6 + wc
    )
    assert not (out / "grid.jsonl").exists()  # not in the grid
    assert Ledger(out / "ledger.jsonl").spent_total("deepseek") <= 0.6 + wc
    # Below the floor: not started at all, and pending.
    spent = Ledger(out / "ledger.jsonl").spent_total("deepseek")
    code, text = go(spent + 0.49 + wc, 1.0)
    assert code == EXIT_CAP and "total budget" in text
    assert len(Journal.open(out / "runs.jsonl").attempts) == 1
    # With budget again, a resume runs it; the stopped attempt did not use up a retry.
    code, text = go(10.0, 1.0)
    assert code == 0
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.meta["status"] == "passed" and run.meta["attempts"] == 2
    assert run.meta["spent_on_earlier_attempts_usd"] > 0.5


def test_run_cap_label_and_meaning_are_unchanged_under_a_full_budget(
    tmp_path: Path, mock: Any
) -> None:
    clock = FakeClock(ts(21, 12))
    mock.set_responder(_CostResponder(clock))
    plan, manifest = _pilot_deepseek_block()
    one = ReplayPlan("one", "one run", plan.specs[-1:], plan.tasks)
    out = tmp_path / "out"
    cfg = config_for(mock, schedules=dict(DEFAULT_SCHEDULES))
    # 30 x the central estimate at peak = about $6.80 off-peak: past the $3 per-run cap.
    # 80 calls keep each prompt under max_prompt_tokens.
    r = _runner(
        one,
        manifest,
        cfg,
        out,
        clock,
        environ=environ_with("deepseek"),
        scaffold=lambda: CostScaffold(30.0, calls=80),
        workspace=lambda task: _NoWorkspace(),
        backend=lambda task: _Pass(),
        task_text=lambda task: "simulated task",
    )
    r.run(execute=True)
    run = load_grid(out / "grid.jsonl").runs[0]
    assert run.meta["status"] == "run_cap" and run.success is False
    assert 3.0 <= run.cost_usd < 3.0 + worst_call(one.specs[0], CAT, cfg)


# --- balance -------------------------------------------------------------------------------


def test_insufficient_balance_halts_deepseek_and_leaves_its_cells_pending(
    tmp_path: Path, mock: Any
) -> None:
    clock = FakeClock(ts(21, 12))
    refused: list[int] = []

    def broke(step: int, rec: Received) -> Reply | None:
        if "deepseek" in str(rec.body.get("model")):
            refused.append(step)
            return Reply(kind="error", status=402, content="Insufficient Balance")
        return None

    agent = ScriptedAgent(overrides={"a": broke})
    mock.set_responder(agent)
    plan, manifest = build(tmp_path, [FLASH, SONNET], tasks=["add-a", "add-b"])
    cfg = config_for(mock, schedules=dict(DEFAULT_SCHEDULES))
    out = tmp_path / "out"
    buf = io.StringIO()
    code = _runner(plan, manifest, cfg, out, clock, stdout=buf).run(execute=True)
    assert code == EXIT_CAP
    text = buf.getvalue()
    assert "deepseek halted: the provider reported an insufficient balance" in text
    j = Journal.open(out / "runs.jsonl")
    ds = [a for a in j.attempts.values() if a.cell[1] == FLASH[0]]
    assert [a.status for a in ds] == ["balance_stop"]  # one attempt, no retry, then halted
    assert (ds[0].end or {})["success"] is None
    assert refused == [0]  # one DeepSeek call, not retried
    grid = load_grid(out / "grid.jsonl")
    assert {r.model for r in grid.runs} == {SONNET[0]}  # Anthropic went on; DeepSeek pending
    assert "2 cell(s) pending (deepseek 2)" in text


# --- keys ------------------------------------------------------------------------------------


def test_env_file_values_never_reach_any_output(tmp_path: Path, mock: Any) -> None:
    from steproute.replay.envfile import load_env_file

    other = "-".join(["fake", "other", "secret", "q" * 10])
    env_file = tmp_path / "keys.env"
    env_file.write_text(
        f"export STEPROUTE_TEST_DEEPSEEK='{fake_key('deepseek')}'\nOTHER_SECRET=\"{other}\"\n",
        "utf-8",
    )
    values = load_env_file(env_file)
    clock = FakeClock(ts(21, 12))

    def leaky(step: int, rec: Any) -> Reply | None:
        if step == 0:  # the model tries to read the environment
            return Reply(kind="tool", tool_name="bash", tool_args=json.dumps({"command": "env"}))
        return None

    mock.set_responder(ScriptedAgent(overrides={"a": leaky}))
    plan, manifest = build(tmp_path, [FLASH], tasks=["add-a"])
    cfg = config_for(mock, schedules=dict(DEFAULT_SCHEDULES))
    out = tmp_path / "out"
    buf = io.StringIO()
    environ = {**values}
    r = _runner(
        plan, manifest, cfg, out, clock, stdout=buf, environ=environ, secrets=list(values.values())
    )
    assert r.run(execute=True) == 0
    for p in out.rglob("*"):
        if p.is_file():
            body = p.read_text("utf-8", errors="replace")
            assert fake_key("deepseek") not in body and other not in body, p
    assert fake_key("deepseek") not in buf.getvalue() and other not in buf.getvalue()
