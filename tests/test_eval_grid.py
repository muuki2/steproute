"""Grid contract and full-information metrics on a hand-computed grid.

Four tasks, four configurations (cost per run, which tasks it solves, latency):

    cheap  = m1 (low)     $1    t1                100 ms
    dud    = m4 (low)     $2    none              200 ms
    odd    = m3 (medium)  $5    t3                500 ms
    strong = m2 (high)    $10   t1 t2 t4         1000 ms
"""

import json
import math
import warnings
from pathlib import Path

import pytest

from steproute.core.schema import Action
from steproute.eval.data import (
    Cell,
    DataFormatError,
    GridData,
    GridRun,
    GridTask,
    detect_kind,
    load_grid,
    write_grid,
)
from steproute.eval.grid import evaluate_grid, simulate_action
from steproute.eval.results import BEATS, DOES_NOT_BEAT, NOT_SHOWN, EvalSettings
from steproute.eval.targets import DistributionTarget, FixedTarget, OracleTarget

CHEAP, DUD, ODD, STRONG = (
    Action("m1", "low"),
    Action("m4", "low"),
    Action("m3", "medium"),
    Action("m2", "high"),
)
SOLVES = {CHEAP: {"t1"}, DUD: set(), ODD: {"t3"}, STRONG: {"t1", "t2", "t4"}}
COST = {CHEAP: 1.0, DUD: 2.0, ODD: 5.0, STRONG: 10.0}
LAT = {CHEAP: 100.0, DUD: 200.0, ODD: 500.0, STRONG: 1000.0}
TASKS = ("t1", "t2", "t3", "t4")
SETTINGS = EvalSettings(bootstrap=200, min_clusters=2)


def _runs(unlabelled: tuple[str, Action] | None = None) -> list[GridRun]:
    out = []
    for t in TASKS:
        for a in (CHEAP, DUD, ODD, STRONG):
            ok: bool | None = t in SOLVES[a]
            if unlabelled == (t, a):
                ok = None
            out.append(GridRun(t, a.model, a.effort, ok, COST[a], LAT[a]))
    return out


def _grid(runs: list[GridRun] | None = None) -> GridData:
    rs = tuple(runs or _runs())
    return GridData("hand", rs, {t: GridTask(t) for t in TASKS})


def _policy(ev, name):  # type: ignore[no-untyped-def]
    return next(p for p in ev.policies if p.name == name)


def _m(p, key):  # type: ignore[no-untyped-def]
    return p.metrics[key].value


def test_single_configuration_metrics() -> None:
    ev = evaluate_grid(_grid(), [FixedTarget(CHEAP, "cheap")], SETTINGS)
    p = _policy(ev, "cheap")
    assert _m(p, "success_rate") == 0.25
    assert _m(p, "cost_per_task") == 1.0
    assert _m(p, "cost_per_solved") == 4.0  # total spend / tasks solved, failures paid
    assert _m(p, "switches_per_task") == 0.0
    assert _m(p, "added_latency_p95_ms") == 0.0
    assert _m(p, "total_latency_p50_ms") == 100.0
    assert _m(p, "quality_retention") == pytest.approx(0.25 / 0.75)
    assert p.diagnostics["strongest_configuration"] == "m2 (high)"
    # The policy is itself a frontier vertex: a tie, so it does not beat the frontier.
    assert _m(p, "frontier_margin") == pytest.approx(0.0)
    assert p.verdict.status == DOES_NOT_BEAT


def test_cascade_simulation_by_hand() -> None:
    ev = evaluate_grid(_grid(), [FixedTarget(Action("m1", "low", STRONG), "c")], SETTINGS)
    p = _policy(ev, "c")
    # t1 solved by the first step ($1); t2, t3, t4 pay both steps ($11); t3 still fails.
    assert _m(p, "success_rate") == 0.75
    assert _m(p, "cost_per_task") == pytest.approx(34 / 4)
    assert _m(p, "cost_per_solved") == pytest.approx(8.5 / 0.75)
    assert _m(p, "switches_per_task") == 0.75
    # Added latency per task: 0, 100, 100, 100 -> lower median 100; total 100 or 1100.
    assert _m(p, "added_latency_p50_ms") == 100.0
    assert _m(p, "added_latency_p95_ms") == 100.0
    assert _m(p, "total_latency_p50_ms") == 1100.0
    # Frontier: origin, cheap (1, .25), strong (10, .75). At $8.5 it reaches .25 + 7.5/9 * .5.
    assert _m(p, "frontier_margin") == pytest.approx(0.75 - (0.25 + 7.5 / 9 * 0.5))
    assert _m(p, "cost_saving_at_equal_success") == pytest.approx(10 - 8.5)


def test_retention_above_one_when_policy_solves_what_the_strongest_fails() -> None:
    ev = evaluate_grid(_grid(), [FixedTarget(Action("m2", "high", ODD), "so")], SETTINGS)
    p = _policy(ev, "so")
    assert _m(p, "success_rate") == 1.0
    assert _m(p, "quality_retention") == pytest.approx(1 / 0.75)  # not capped at 1
    assert _m(p, "strongest_solved_retained") == 1.0  # capped share, by definition
    saving = p.metrics["cost_saving_at_equal_success"]
    assert saving.value == math.inf and saving.to_json()["value"] is None
    assert "no single configuration reaches" in str(saving.note)


def test_cost_per_solved_when_nothing_is_solved() -> None:
    ev = evaluate_grid(_grid(), [FixedTarget(DUD, "dud")], SETTINGS)
    p = _policy(ev, "dud")
    assert _m(p, "success_rate") == 0.0
    cps = p.metrics["cost_per_solved"]
    assert cps.value == math.inf and cps.note == "no task solved"
    j = cps.to_json()
    assert j["value"] is None and j["unbounded"] is True
    json.dumps(ev.to_json(), allow_nan=False)  # infinities never reach the JSON


def test_stochastic_policy_is_scored_in_expectation() -> None:
    mix = DistributionTarget(lambda s: ((CHEAP, 0.5), (STRONG, 0.5)), "mix")
    p = _policy(evaluate_grid(_grid(), [mix], SETTINGS), "mix")
    assert _m(p, "success_rate") == pytest.approx(0.5)
    assert _m(p, "cost_per_task") == pytest.approx(5.5)
    # A random mix of two configurations lies on or below the frontier, never beyond it.
    assert _m(p, "frontier_margin") <= 1e-12


def test_unsupported_choices_are_reported_not_dropped() -> None:
    p = _policy(evaluate_grid(_grid(), [FixedTarget(Action("m9", "low"), "x")], SETTINGS), "x")
    assert p.verdict.status == NOT_SHOWN
    assert p.diagnostics["tasks_unsupported"] == 4
    assert "cannot score the policy's choice on 4 of 4 tasks" in p.verdict.reasons[0]
    assert p.metrics["success_rate"].value is None


def test_cascade_needs_only_the_cells_it_reaches() -> None:
    cells = {("m2", "high"): Cell(1, 1, 1, 1.0, 10.0, 1000.0)}
    out = simulate_action(Action("m2", "high", Action("m9", "low")), cells)
    assert not isinstance(out, str) and out.success == 1.0 and out.cost == 10.0
    cells[("m2", "high")] = Cell(1, 1, 1, 0.0, 10.0, 1000.0)
    assert simulate_action(Action("m2", "high", Action("m9", "low")), cells) == (
        "no run of m9 (low)"
    )


def test_unlabelled_runs_exclude_the_task_and_feed_the_sensitivity_check() -> None:
    g = _grid(_runs(unlabelled=("t2", CHEAP)))
    ev = evaluate_grid(g, [FixedTarget(Action("m1", "low", STRONG), "c")], SETTINGS)
    assert ev.provenance["tasks_evaluated"] == 3
    assert ev.provenance["unlabelled_runs"] == 1
    assert ev.provenance["tasks_excluded_by_reason"]["unlabelled_run"] == 1
    p = _policy(ev, "c")
    sens = p.diagnostics["unlabelled_sensitivity"]
    assert set(sens) == {"unlabelled_as_failure", "unlabelled_as_success"}
    assert any("unlabelled" in n for n in ev.notes)


def test_too_few_tasks_is_not_shown() -> None:
    ev = evaluate_grid(_grid(), [FixedTarget(STRONG, "s")], EvalSettings(bootstrap=200))
    assert _policy(ev, "s").verdict.status == NOT_SHOWN
    assert "only 4 tasks" in _policy(ev, "s").verdict.reasons[0]


def test_oracle_is_flagged_as_hindsight() -> None:
    p = _policy(evaluate_grid(_grid(), [OracleTarget()], SETTINGS), "oracle")
    assert p.hindsight
    assert _m(p, "success_rate") == 1.0  # every task is solved by some configuration...
    assert _m(p, "cost_per_task") == pytest.approx((1 + 10 + 5 + 10) / 4)  # ...cheapest one
    assert "hindsight oracle" in p.verdict.sentence
    assert p.verdict.status in (BEATS, NOT_SHOWN)


def test_split_and_exclusion() -> None:
    tasks = {t: GridTask(t, "test" if t != "t1" else "train") for t in TASKS}
    g = GridData("hand", tuple(_runs()), tasks)
    ev = evaluate_grid(g, [], SETTINGS, split="test", exclude_configs=[("m4", "low")])
    assert ev.provenance["tasks_in_split"] == 3
    assert "m4 (low)" not in ev.provenance["configurations"]


# --- the contract ------------------------------------------------------------------------


def test_grid_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    tasks = [GridTask(t, "test", None, {"repo": "x"}) for t in TASKS]
    write_grid(path, _runs(), tasks, {"note": "hand"})
    g = load_grid(path)
    assert detect_kind(path) == "grid"
    assert g.runs == tuple(_runs())
    assert g.tasks["t1"].features == {"repo": "x"}
    assert g.header["note"] == "hand" and g.sha256
    assert g.configs() == (("m1", "low"), ("m2", "high"), ("m3", "medium"), ("m4", "low"))


def test_grid_loader_skips_bad_lines_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    write_grid(path, _runs()[:2])
    with open(path, "a") as f:
        f.write("not json\n")
        f.write('{"type":"run","v":"0.1","data":{"task_id":"t9","model":"m","effort":"bad",')
        f.write('"success":true,"cost_usd":1}}\n')
        f.write('{"type":"run","v":"0.1","data":{"task_id":"t9","model":"m","effort":"low",')
        f.write('"success":true,"cost_usd":1,"future_field":3}}\n')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = load_grid(path)
    assert g.skipped_lines == 2
    assert len(g.runs) == 3
    assert any("future_field" in str(w.message) for w in caught)
    with open(path, "a") as f:
        f.write(json.dumps({"type": "run", "v": "0.1", "data": _runs()[0].to_dict()}) + "\n")
    with warnings.catch_warnings(), pytest.raises(DataFormatError, match="duplicate run"):
        warnings.simplefilter("ignore")
        load_grid(path)


def test_grid_without_runs_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    write_grid(path, [], [GridTask("t1")])
    with pytest.raises(DataFormatError, match="no run lines"):
        load_grid(path)
