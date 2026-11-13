"""Replay-runner pieces that need no LiteLLM: plans, ledger, guard, journal, config, CLI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from steproute.cli import main
from steproute.core.catalog import load_default_catalog
from steproute.replay.config import (
    DEFAULT_CAPS,
    ReplayConfigError,
    load_replay_config,
    parse_replay_config,
)
from steproute.replay.journal import Journal, LockHeld, OutputLock
from steproute.replay.ledger import CapExceeded, CapGuard, Ledger, month_of
from steproute.replay.plan import build_named_plan
from steproute.replay.preflight import preflight
from steproute.tasks.estimate import build_plans, estimate_plan, load_assumptions
from steproute.tasks.manifest import load_default_manifest

CAT = load_default_catalog()
MAN = load_default_manifest()
ASSUME = load_assumptions()


# --- plans ---------------------------------------------------------------------------------


def _configs(plan: Any) -> list[tuple[str, str]]:
    return list(plan.configs())


def test_pilot_plan_is_cheapest_first_and_matches_the_task_set_document() -> None:
    plan = build_named_plan("pilot", MAN, CAT, ASSUME)
    assert _configs(plan) == [
        ("qwen3.5:9b", "none"),
        ("qwen3.5:9b", "medium"),
        ("gpt-oss:20b", "low"),
        ("gpt-oss:20b", "medium"),
        ("gpt-oss:20b", "high"),
        ("qwen3-coder:30b", "medium"),
        ("claude-haiku-5-5", "low"),
        ("claude-haiku-5-5", "medium"),
        ("claude-haiku-5-5", "high"),
        ("claude-haiku-5-5", "xhigh"),
        ("claude-haiku-5-5", "max"),
        ("claude-sonnet-5-5", "low"),  # stage 1 ends: the stage gate is checked here
        ("deepseek-flash", "none"),
        ("deepseek-flash", "low"),
        ("deepseek-flash", "high"),
        ("deepseek-flash", "max"),
        ("deepseek-v4-pro", "high"),
        ("claude-sonnet-5-5", "high"),
        ("claude-opus-5-5", "medium"),
        ("claude-fable-5-2", "low"),
        ("devstral:24b", "medium"),  # the holdout pass, separate split
    ]
    counts: dict[str, int] = {}
    for s in plan.specs:
        counts[s.provider] = counts.get(s.provider, 0) + 1
    assert counts == {"ollama": 140, "anthropic": 180, "deepseek": 100}  # TASKSET section 5
    assert {s.task_id for s in plan.specs} == {t.id for t in MAN.pilot()}
    # configuration-major: tasks in manifest order within each configuration
    first = [s.task_id for s in plan.specs if (s.model, s.effort) == ("claude-haiku-5-5", "low")]
    assert first == [t.id for t in MAN.pilot()]
    central = sum(s.est_usd[0] for s in plan.specs if s.provider == "anthropic")
    assert round(central) == 41  # the document's $41 central
    assert plan.gate is not None and len(plan.gate.reduced_tasks) == 10
    strata = [MAN.get(t).stratum for t in plan.gate.reduced_tasks]
    assert sorted(strata) == sorted(["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3 + ["unsolved"])


def test_sparse_plan_is_option_b() -> None:
    plan = build_named_plan("sparse", MAN, CAT, ASSUME)
    est = estimate_plan(build_plans(CAT)["sparse"], CAT, ASSUME)
    counts: dict[str, int] = {}
    for s in plan.specs:
        counts[s.provider] = counts.get(s.provider, 0) + 1
    assert counts == dict(est.runs)
    sweep = plan.meta["sweep_tasks"]
    assert len(sweep) == 30
    assert {t.id for t in MAN.pilot()} <= set(sweep)  # pilot cells are reused
    strata: dict[str, int] = {}
    for t in sweep:
        strata[MAN.get(t).stratum or "?"] = strata.get(MAN.get(t).stratum or "?", 0) + 1
    assert strata == {"easy": 8, "medium": 9, "hard": 10, "unsolved": 3}
    central = sum(s.est_usd[0] for s in plan.specs if s.provider == "anthropic")
    assert round(central) == 456  # the document's option B central
    # the default effort of every paid model runs on all 150 tasks
    opus = [s for s in plan.specs if s.model == "claude-opus-5-5" and s.effort == "medium"]
    assert len(opus) == 150
    order = [s.model for s in plan.specs if not s.local]
    assert order.index("claude-haiku-5-5") < order.index("claude-sonnet-5-5")
    assert max(i for i, m in enumerate(order) if m == "claude-opus-5-5") < order.index(
        "claude-fable-5-2"
    )


# --- ledger and guard ----------------------------------------------------------------------


def test_month_is_the_utc_calendar_month() -> None:
    assert month_of(1798761599.0) == "2026-12"  # 2026-12-31 23:59:59 UTC
    assert month_of(1798761600.0) == "2027-01"


def test_a_crash_in_the_middle_of_a_ledger_write(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    done = led.intend("anthropic", 2.0, "r1")
    led.settle(done, 0.25)
    open_ = led.intend("anthropic", 2.0, "r2")
    # The process dies while writing r2's settle line: a torn, unterminated line.
    line = json.dumps({"v": 1, "kind": "settle", "id": open_.id, "usd": 0.1})
    with open(path, "ab") as f:
        f.write(line.encode()[: len(line) // 2])
    led2 = Ledger(path)
    assert led2.skipped_lines == 1
    assert led2.spent("anthropic") == pytest.approx(0.25 + 2.0)  # unsettled: worst case
    assert [i.run_id for i in led2.open_intents()] == ["r2"]
    # The next write starts on a fresh line and is readable.
    nxt = led2.intend("anthropic", 1.0, "r3")
    led2.settle(nxt, 0.5)
    led3 = Ledger(path)
    assert led3.skipped_lines == 1
    assert led3.spent("anthropic") == pytest.approx(0.25 + 2.0 + 0.5)
    # A torn intent (killed while writing it, so the call was never made) charges nothing.
    with open(path, "ab") as f:
        f.write(b'{"v":1,"kind":"intent","id":"zz","usd":9')
    assert Ledger(path).spent("anthropic") == pytest.approx(2.75)


def test_cap_guard_with_parallel_admissions(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "ledger.jsonl")
    guard = CapGuard(led, {"anthropic": 10.0})
    admitted: list[str] = []
    lock = threading.Lock()

    def try_admit(i: int) -> None:
        try:
            guard.admit("anthropic", f"r{i}", 3.0)
            with lock:
                admitted.append(f"r{i}")
        except CapExceeded:
            pass

    threads = [threading.Thread(target=try_admit, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(admitted) == 3  # 3 x $3 <= $10 < 4 x $3
    # spend by an admitted run lowers its outstanding reservation, never below zero
    i = led.intend("anthropic", 1.0, admitted[0])
    led.settle(i, 1.0)
    assert guard.headroom("anthropic") == pytest.approx(10.0 - 1.0 - 2.0 - 3.0 - 3.0)
    guard.release(admitted[0])
    assert guard.headroom("anthropic") == pytest.approx(10.0 - 1.0 - 6.0)
    with pytest.raises(CapExceeded):
        guard.admit("deepseek", "x", 0.1)  # no cap configured: refused


# --- journal and lock ----------------------------------------------------------------------


def test_journal_resume_semantics(tmp_path: Path) -> None:
    j = Journal.open(tmp_path / "runs.jsonl")
    cell = {"task_id": "t", "model": "m", "effort": "low", "repeat": 0}
    j.record_start({"run_id": "a", "cell": cell, "attempt": 1, "session_id": "s-a"})
    j.record_end("a", {"status": "provider_error"})
    j.record_start({"run_id": "b", "cell": cell, "attempt": 2, "session_id": "s-b"})
    j2 = Journal.open(tmp_path / "runs.jsonl")
    assert [a.run_id for a in j2.in_flight()] == ["b"]
    assert list(j2.finished()) == []
    j2.close_interrupted({"b": 0.5})
    j3 = Journal.open(tmp_path / "runs.jsonl")
    assert j3.attempts["b"].status == "interrupted"
    assert j3.attempts["b"].end is not None and j3.attempts["b"].end["cost_usd"] == 0.5
    j3.record_start({"run_id": "c", "cell": cell, "attempt": 3, "session_id": "s-c"})
    j3.record_end("c", {"status": "passed", "success": True})
    assert [a.run_id for a in Journal.open(tmp_path / "runs.jsonl").finished()] == ["c"]


def test_output_lock_is_exclusive(tmp_path: Path) -> None:
    with OutputLock(tmp_path):
        with pytest.raises(LockHeld):
            OutputLock(tmp_path).acquire()
    OutputLock(tmp_path).acquire().release()  # free again after release


# --- config --------------------------------------------------------------------------------


def test_config_defaults_are_the_safe_ones() -> None:
    cfg = load_replay_config(None)
    assert dict(cfg.caps) == dict(DEFAULT_CAPS) == {"anthropic": 80.0, "deepseek": 20.0}
    assert cfg.limits.run_cap_usd == 3.0
    assert cfg.provider("anthropic").api_key_env == "ANTHROPIC_API_KEY"
    assert cfg.cap("ollama") is None


def test_raising_a_cap_needs_an_explicit_value() -> None:
    with pytest.raises(ReplayConfigError, match="allow_raise"):
        parse_replay_config({"caps": {"anthropic": 120.0}})
    cfg = parse_replay_config({"caps": {"anthropic": 120.0, "allow_raise": True}})
    assert cfg.caps["anthropic"] == 120.0
    assert parse_replay_config({"caps": {"deepseek": 5.0}}).caps["deepseek"] == 5.0  # lower: fine


def test_a_pasted_key_is_refused_without_echoing_it() -> None:
    pasted = "-".join(["fake", "pasted", "key", "z" * 10])
    with pytest.raises(ReplayConfigError) as exc:
        parse_replay_config({"providers": {"anthropic": {"api_key_env": pasted}}})
    assert pasted not in str(exc.value)


def test_unknown_keys_are_errors() -> None:
    with pytest.raises(ReplayConfigError):
        parse_replay_config({"limits": {"run_cap": 3}})
    with pytest.raises(ReplayConfigError):
        parse_replay_config({"spend": {}})


# --- CLI and preflight ---------------------------------------------------------------------


def test_cli_run_without_execute_is_a_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "out"
    assert main(["replay", "run", "--plan", "pilot", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "DRY RUN" in text and "420 to run" in text
    assert "anthropic   180 runs  ~$41.28 ($23.28-$178.60)" in text
    assert "a real run would stop here: deepseek" in text
    assert not out.exists()


def test_cli_execute_without_keys_stops_before_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "out"
    assert main(["replay", "run", "--plan", "pilot", "--out", str(out), "--execute"]) == 2
    assert "DeepSeek configurations are required" in capsys.readouterr().out
    assert not out.exists()


def test_preflight_reports_without_changing_anything(tmp_path: Path) -> None:
    plan = build_named_plan("pilot", MAN, CAT, ASSUME)
    calls: list[list[str]] = []

    def run(cmd: Any, timeout: float) -> tuple[int, str]:
        calls.append(list(cmd))
        return (1, "Cannot connect to the Docker daemon") if cmd[1] == "info" else (0, "")

    def fetch(url: str, timeout: float) -> str:
        return json.dumps({"models": [{"name": "qwen3-coder:30b"}]})

    env = {"ANTHROPIC_API_KEY": "-".join(["fake", "k" * 8])}
    checks = {
        c.name: c
        for c in preflight(
            plan,
            MAN,
            load_replay_config(None),
            environ=env,
            out=tmp_path / "out",
            run=run,
            fetch=fetch,
            disk_free_gb=lambda p: 50.0,
        )
    }
    assert checks["key anthropic"].ok and "set" in checks["key anthropic"].detail
    assert env["ANTHROPIC_API_KEY"] not in checks["key anthropic"].detail
    assert not checks["key deepseek"].ok
    assert not checks["docker daemon"].ok
    assert not checks["ollama models"].ok and "gpt-oss:20b" in checks["ollama models"].detail
    assert not checks["free disk"].ok
    assert all(c[:2] != ["docker", "pull"] for c in calls)  # read-only
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("ratio", "cut"), [(1.0, False), (2.0, True)])
def test_pilot_stage_gate(tmp_path: Path, ratio: float, cut: bool) -> None:
    import io

    from steproute.replay.runner import Runner, RunnerDeps

    plan = build_named_plan("pilot", MAN, CAT, ASSUME)
    r = Runner(
        plan,
        config=load_replay_config(None),
        catalog=CAT,
        manifest=MAN,
        out=tmp_path,
        deps=RunnerDeps(out=io.StringIO()),
    )
    r.journal = Journal.open(tmp_path / "runs.jsonl")
    r.ledger = Ledger(tmp_path / "ledger.jsonl")
    stage1 = [x for x in plan.specs if x.stage == 1 and x.provider == "anthropic"]
    for i, s in enumerate(stage1):
        cost = s.est_usd[0] * ratio
        cell = {"task_id": s.task_id, "model": s.model, "effort": s.effort, "repeat": 0}
        r.journal.record_start(
            {"run_id": f"r{i}", "cell": cell, "attempt": 1, "session_id": f"s{i}"}
        )
        r.journal.record_end(f"r{i}", {"status": "failed", "success": False, "cost_usd": cost})
        r.ledger.settle(r.ledger.intend("anthropic", 5.0, f"r{i}"), cost)
    r._evaluate_gate()
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["cut"] is cut
    assert gate["r"] == pytest.approx(ratio)
    stage2_tasks = {s.task_id for s in r._pending() if s.stage == 2 and not s.local}
    assert len(stage2_tasks) == (10 if cut else 20)


def test_the_config_example_in_the_docs_loads(tmp_path: Path) -> None:
    import re

    doc = (Path(__file__).parent.parent / "docs" / "REPLAY.md").read_text("utf-8")
    (block,) = re.findall(r"```toml example=replay\n(.*?)```", doc, re.S)
    path = tmp_path / "replay.toml"
    path.write_text(block, "utf-8")
    cfg = load_replay_config(path)
    assert dict(cfg.caps) == dict(DEFAULT_CAPS)
    assert cfg.limits == load_replay_config(None).limits  # the example shows the defaults
