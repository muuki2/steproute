"""One cost basis for the evaluator: list price (peak catalog price on the token counts).

An off-peak DeepSeek grid row and a peak one with the same usage cost the same; a subscription
trace with tokens and a stored cost of 0 gets its list cost; a row with no usable cost is an
error; a report shows the billed figure where it differs.
"""

import json
from typing import Any

import pytest

from steproute.core.catalog import load_default_catalog
from steproute.core.cost import AttemptUsage, price_usage
from steproute.core.schema import Action, Decision, Outcome, StepState
from steproute.eval.data import (
    COST_BASIS,
    GridData,
    GridRun,
    GridTask,
    logged_from_pairs,
)
from steproute.eval.grid import evaluate_grid
from steproute.eval.ope import evaluate_logged
from steproute.eval.report import build_report
from steproute.eval.results import EvalSettings, Evaluation
from steproute.eval.targets import FixedTarget

CAT = load_default_catalog(warn_unverified=False)
SETTINGS = EvalSettings(bootstrap=100, min_clusters=1, min_ess=0)
TASKS = ("t1", "t2", "t3")
CHEAP, DEAR = Action("deepseek-flash", "high"), Action("claude-sonnet-5-5", "high")


def _row(task: str, a: Action, ok: bool, billed: float | None, meta: dict[str, Any]) -> GridRun:
    return GridRun(task, a.model, a.effort, ok, billed, 10.0, meta=meta)


def _grid(offpeak: bool, extra: list[GridRun] | None = None) -> GridData:
    runs = []
    for t in TASKS:
        if offpeak:  # billed at half the list price, as the replay runner writes it
            meta: dict[str, Any] = {"list_cost_usd": 2.0, "calls_by_period": {"offpeak": 4}}
            runs.append(_row(t, CHEAP, t != "t3", 1.0, meta))
        else:
            meta = {"list_cost_usd": 2.0, "calls_by_period": {"peak": 4}}
            runs.append(_row(t, CHEAP, t != "t3", 2.0, meta))
        runs.append(_row(t, DEAR, True, 5.0, {"list_cost_usd": 5.0}))
    runs += extra or []
    return GridData("g", tuple(runs), {t: GridTask(t) for t in TASKS})


def _cost(ev: Evaluation) -> float:
    value = ev.policies[0].metrics["cost_per_task"].value
    assert value is not None
    return float(value)


def test_same_usage_same_cost_off_peak_and_peak() -> None:
    pol = [FixedTarget(CHEAP)]
    off = evaluate_grid(_grid(True), pol, SETTINGS)
    peak = evaluate_grid(_grid(False), pol, SETTINGS)
    assert _cost(off) == pytest.approx(2.0)
    assert _cost(peak) == pytest.approx(2.0)
    # The billed cost is reported next to it, for the off-peak data only.
    assert off.policies[0].diagnostics["billed_cost_per_task"] == pytest.approx(1.0)
    assert "billed_cost_per_task" not in peak.policies[0].diagnostics


def test_row_with_no_usable_cost_is_reported_as_an_error() -> None:
    bad = GridRun("t1", CHEAP.model, "low", True, 1.0, meta={"calls_by_period": {"offpeak": 2}})
    assert bad.list_cost == (None, "missing")
    nothing = GridRun("t1", CHEAP.model, "medium", True, None)
    assert nothing.list_cost_usd is None
    ev = evaluate_grid(_grid(True, [bad, nothing]), [FixedTarget(CHEAP)], SETTINGS)
    errs = ev.provenance["cost_basis"]["errors"]
    assert errs["runs_without_usable_cost"] == 2
    assert ev.provenance["uncosted_runs"] == 2


def test_billed_cost_alone_is_list_only_without_off_peak_calls() -> None:
    plain = GridRun("t1", CHEAP.model, "low", True, 3.0)
    assert plain.list_cost == (3.0, "billed_as_list")


def _sub_pair(i: int, cost: float | None, billing: str | None) -> tuple[Decision, Outcome]:
    state = StepState("claude-code", f"s{i}", 0, "main", 1000)
    a = Action("claude-sonnet-5-5", "high")
    d = Decision(f"d{i}", 1.0, state, [a], a, "log", 1.0)
    signals = {} if billing is None else {"billing": billing}
    o = Outcome(
        f"d{i}",
        2.0,
        True,
        signals=signals,
        final_model=a.model,
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=2000,
        cost_usd=cost,
    )
    return d, o


def test_subscription_trace_with_tokens_gets_list_cost() -> None:
    expect = price_usage(CAT.get("claude-sonnet-5-5"), AttemptUsage(1000, 2000, 0, 500)).total_usd
    pairs = [
        _sub_pair(1, 0.0, "subscription"),
        _sub_pair(2, None, "subscription"),
        _sub_pair(3, 99.0, "api"),
    ]
    data = logged_from_pairs(pairs, CAT)
    for row in data.rows:
        assert row.cost == pytest.approx(expect)
        assert row.cost_source == "tokens"
    assert data.rows[0].billed_cost == 0.0  # what the subscription recorded, kept apart
    assert data.rows[2].billed_cost == 99.0


def test_subscription_trace_without_tokens_has_no_cost() -> None:
    d, o = _sub_pair(1, 0.0, "subscription")
    bare = Outcome(
        o.decision_id, o.ts, True, signals=o.signals, final_model=o.final_model, cost_usd=0.0
    )
    row = logged_from_pairs([(d, bare)], CAT).rows[0]
    assert row.cost is None and row.cost_source == "missing"


def test_report_shows_both_figures_and_the_basis() -> None:
    ev = evaluate_grid(_grid(True), [FixedTarget(CHEAP)], SETTINGS)
    report = build_report([ev])
    assert report["cost_basis"] == COST_BASIS
    assert "billed_cost_per_task" in json.dumps(report)
    cb = report["evaluations"][0]["provenance"]["cost_basis"]
    assert cb["billed_cost_per_task_where_different"]["configurations"] == {
        "deepseek-flash (high)": pytest.approx(1.0)
    }

    logged = logged_from_pairs([_sub_pair(i, 0.0, "subscription") for i in range(4)], CAT)
    lev = evaluate_logged(logged, [FixedTarget(DEAR)], SETTINGS)
    lcb = build_report([lev])["evaluations"][0]["provenance"]["cost_basis"]
    assert lcb["steps_where_billed_differs_from_list"] == 4
    assert lcb["billed_total_usd_over_those_steps"] == 0.0
    assert lcb["list_total_usd_over_those_steps"] > 0
