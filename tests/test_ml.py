"""The tabular predictor: features, splits, calibration, reproducibility, the saved format, the
policy integration and the CLI, all on the evaluator's synthetic data.

Needs the ml extra; skipped without it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from steproute.cli import main  # noqa: E402
from steproute.core import Action, ModelProfile, ScoredPolicy, StepState  # noqa: E402
from steproute.core.catalog import Catalog, load_default_catalog  # noqa: E402
from steproute.core.cost import state_after_failure  # noqa: E402
from steproute.eval.data import GridData, GridRun, GridTask, load_grid, write_grid  # noqa: E402
from steproute.eval.synthetic import (  # noqa: E402
    RULES_TOML,
    catalog,
    default_world,
    make_grid,
    write_example,
)
from steproute.ml.calibration import (  # noqa: E402
    Calibrator,
    auc,
    brier,
    ece,
    fit_calibrator,
    log_loss,
    reliability,
    select_calibrator,
    sigmoid,
    window_gap,
)
from steproute.ml.data import build_dataset, probe_table_from_grid  # noqa: E402
from steproute.ml.evaluate import calibration_report, evaluator_report  # noqa: E402
from steproute.ml.features import (  # noqa: E402
    FeatureBuilder,
    FeatureSpec,
    ProbeResult,
    ProbeTable,
    task_feature_keys,
)
from steproute.ml.model import (  # noqa: E402
    FeatureSpecMismatch,
    ModelFormatError,
    TabularModel,
    TabularPredictor,
    TrainConfig,
    train,
)
from steproute.ml.splits import assign_roles  # noqa: E402
from steproute.ml.stamp import STAMP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WORLD = default_world()
CATALOG = catalog(WORLD)


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_example(tmp_path_factory.mktemp("ml"), n_tasks=500, seed=4)


@pytest.fixture(scope="module")
def grid(example: dict[str, Path]) -> GridData:
    return load_grid(example["grid"])


@pytest.fixture(scope="module")
def trained(grid: GridData) -> TabularModel:
    return train(grid, CATALOG, TrainConfig(seed=7)).model


def _start(grid: GridData, tid: str) -> StepState:
    state = grid.tasks[tid].state
    assert state is not None
    return state


def _fixed_inputs(cat: Catalog, grid: GridData) -> list[tuple[StepState, Action]]:
    """A fixed set of (state, step) pairs, including cascade states after a failure."""
    pairs: list[tuple[StepState, Action]] = []
    cfgs = [(m.id, e) for m in cat.models for e in m.efforts]
    for tid in sorted(grid.tasks)[:6]:
        base = _start(grid, tid)
        for model, effort in cfgs:
            pairs.append((base, Action(model, effort)))
        first = Action(cfgs[0][0], cfgs[0][1])
        after = state_after_failure(base, first, cat.get(first.model))
        for model, effort in cfgs[1:]:
            pairs.append((after, Action(model, effort)))
    return pairs


# --- features --------------------------------------------------------------------------------


def test_feature_names_hold_no_model_identity() -> None:
    names = FeatureSpec(task_keys=("hint",)).names
    assert len(names) == len(set(names))
    for name in names:
        assert "model_id" not in name and name not in ("model", "id")
    # The only per-model quantities are profile attributes and probe results.
    assert {"tier", "local", "log_price_in", "probe_rate", "probe_n"} <= set(names)


def test_renaming_every_model_gives_the_same_predictions(
    grid: GridData, trained: TabularModel
) -> None:
    """The model sees profiles and probe results, never ids: rename everything, same answers."""
    old = sorted({m.id for m in CATALOG.models})
    # Reverse the sort order too, so any accidental dependence on id order shows up.
    mapping = {m: f"x{len(old) - 1 - i}-{'q' * i}" for i, m in enumerate(old)}
    cat2 = Catalog(tuple(dataclasses.replace(m, id=mapping[m.id]) for m in CATALOG.models))
    runs2 = tuple(dataclasses.replace(r, model=mapping[r.model]) for r in grid.runs)
    grid2 = GridData(grid.source, runs2, grid.tasks, grid.header, 0, grid.sha256)

    # 1. The same trained model, with the catalog, probe table and states renamed.
    probes2 = ProbeTable(
        trained.probes.order,
        {(mapping[m], e): rows for (m, e), rows in trained.probes.results.items()},
    )
    p1 = TabularPredictor(trained, CATALOG)
    p2 = TabularPredictor(trained, cat2, probes2)
    pairs = _fixed_inputs(CATALOG, grid)
    pairs2 = _fixed_inputs(cat2, grid2)
    assert len(pairs) == len(pairs2)
    got1 = p1.predict_many(pairs)
    got2 = p2.predict_many(pairs2)
    assert got1 == got2
    assert len(set(got1)) > 10  # not a constant

    # 2. Retraining on the renamed data gives the same model (training order and splits do
    # not depend on ids either).
    retrained = train(grid2, cat2, TrainConfig(seed=7)).model
    assert TabularPredictor(retrained, cat2).predict_many(pairs2) == got1


def test_probe_stats_by_hand_and_nesting() -> None:
    cfg = ("m", "medium")
    table = ProbeTable(
        ["p1", "p2", "p3", "p4"],
        {
            cfg: {
                "p1": ProbeResult(True, 0.01),
                "p2": ProbeResult(False, 0.03),
                "p3": ProbeResult(None, 0.5),  # unlabelled: not a failure, not counted
                "p4": ProbeResult(True, None),
            }
        },
    )
    s0, s1, s2, s_all = (table.stats(cfg, k) for k in (0, 1, 2, 4))
    assert s0.n == 0 and math.isnan(s0.rate) and math.isnan(s0.log_cost)  # cold start
    assert (s1.n, s1.rate) == (1, 1.0)
    assert (s2.n, s2.rate) == (2, 0.5)
    assert s2.log_cost == pytest.approx((math.log1p(10.0) + math.log1p(30.0)) / 2)
    assert (s_all.n, s_all.rate) == (3, pytest.approx(2 / 3))
    assert table.stats(cfg, None) == s_all or table.stats(cfg, None).n == 3
    assert table.stats(cfg, 99).n == 3
    other = table.stats(("unknown", "low"), 4)
    assert other.n == 0 and math.isnan(other.rate)
    again = ProbeTable.from_dict(json.loads(json.dumps(table.to_dict())))
    assert again.fingerprint() == table.fingerprint()


def test_cold_start_features_are_missing_not_zero() -> None:
    spec = FeatureSpec()
    table = ProbeTable(["p1"], {("big", "medium"): {"p1": ProbeResult(True, 0.1)}})
    cat = Catalog((ModelProfile("big", "anthropic", 5.0, 25.0, 200_000, tier=2),))
    state = StepState("replay", "s", 0, "main", 1000)
    step = Action("big", "medium")
    names = spec.names
    warm = FeatureBuilder(spec, cat, table, probe_k=None).matrix([(state, step)])[0]
    cold = FeatureBuilder(spec, cat, table, probe_k=0).matrix([(state, step)])[0]
    assert warm[names.index("probe_n")] == 1 and warm[names.index("probe_rate")] == 1.0
    assert cold[names.index("probe_n")] == 0 and math.isnan(cold[names.index("probe_rate")])
    with pytest.raises(KeyError, match="not in the catalog"):
        FeatureBuilder(spec, cat).matrix([(state, Action("ghost", "medium"))])


def test_failed_attempts_and_cache_reach_the_features() -> None:
    cat = Catalog(
        (
            ModelProfile("small", "ollama", 0.0, 0.0, 32_000, tier=1, local=True),
            ModelProfile("big", "anthropic", 5.0, 25.0, 200_000, tier=3),
        )
    )
    spec = FeatureSpec()
    b = FeatureBuilder(spec, cat)
    n = spec.names
    state = StepState("replay", "s", 0, "tool_followup", 10_000, tools=["a", "b"])
    after = state_after_failure(state, Action("small", "medium"), cat.get("small"))
    fresh, cascade = b.matrix([(state, Action("big", "medium")), (after, Action("big", "medium"))])
    assert fresh[n.index("n_failed")] == 0 and cascade[n.index("n_failed")] == 1
    assert math.isnan(fresh[n.index("failed_max_tier")])
    assert cascade[n.index("failed_max_tier")] == 1
    assert cascade[n.index("tier_minus_failed_max")] == 2
    assert cascade[n.index("cached_frac")] == 1.0 and fresh[n.index("cached_frac")] == 0.0
    assert fresh[n.index("n_tools")] == 2 and fresh[n.index("kind_tool_followup")] == 1
    # The cache owner is a relation to the step, not an identity.
    small = b.matrix([(after, Action("small", "medium"))])[0]
    assert small[n.index("owns_cache")] == 1.0 and cascade[n.index("owns_cache")] == 0.0
    assert small[n.index("model_already_failed")] == 1.0


def test_task_features_pass_through_numeric_only() -> None:
    states = [
        StepState("r", "s", 0, "main", 1, features={"hint": 2, "repo": "django", "flag": True}),
        StepState("r", "s", 0, "main", 1, features={"hint": 3, "nested": {"a": 1}}),
    ]
    assert task_feature_keys(states) == ("flag", "hint")


# --- the training table ----------------------------------------------------------------------


def _tiny_grid() -> tuple[GridData, Catalog]:
    cat = Catalog(
        (
            ModelProfile("small", "p", 1.0, 5.0, 100_000, tier=1),
            ModelProfile("big", "p", 5.0, 25.0, 100_000, tier=2),
        )
    )

    def run(t: str, m: str, ok: bool) -> GridRun:
        return GridRun(t, m, "medium", ok, 0.1)

    runs = (
        run("t1", "small", False),
        run("t1", "big", True),
        run("t2", "small", True),
        run("t2", "big", False),
        run("t3", "small", False),
        run("t3", "big", False),
    )
    tasks = {t: GridTask(t, None, None, {"prompt_tokens": 100}) for t in ("t1", "t2", "t3")}
    return GridData("tiny", runs, tasks), cat


def test_conditional_rows_are_the_paired_outcomes_after_a_failure() -> None:
    grid, cat = _tiny_grid()
    builder = FeatureBuilder(FeatureSpec(), cat)
    ids = ["t1", "t2", "t3"]
    ds = build_dataset(grid, ids, cat, builder, seed=0, max_history=1)
    rows = [
        (t, c[0], h, int(y))
        for t, c, h, y in zip(ds.task_ids, ds.config, ds.history, ds.y, strict=True)
    ]
    assert len([r for r in rows if r[2] == 0]) == 6
    # "small" failed on t1 and t3, so "big" is asked there; the label is big's own outcome.
    assert sorted(r for r in rows if r[2] == 1) == [("t1", "big", 1, 1), ("t3", "big", 1, 0)]
    only_fresh = build_dataset(grid, ids, cat, builder, seed=0, max_history=0)
    assert set(only_fresh.history) == {0}
    held = build_dataset(grid, ids, cat, builder, seed=0, max_history=1, exclude_models=["small"])
    assert {c[0] for c in held.config} == {"big"} and set(held.history) == {0}


def test_dataset_rows_do_not_depend_on_task_order(grid: GridData) -> None:
    probes = probe_table_from_grid(grid, sorted(grid.tasks)[:10])
    b = FeatureBuilder(FeatureSpec(task_keys=("hint",)), CATALOG, probes)
    tasks = sorted(grid.tasks)[10:60]
    kw: dict[str, Any] = {"seed": 3, "history_cap": 5, "probe_sizes": (0, 5, None)}
    a = build_dataset(grid, tasks, CATALOG, b, **kw)
    c = build_dataset(grid, tasks[::-1], CATALOG, b, **kw)
    assert np.array_equal(a.X, c.X, equal_nan=True) and np.array_equal(a.y, c.y)


# --- splits ----------------------------------------------------------------------------------


def test_roles_are_by_task_and_probes_stay_out_of_every_other_role(
    grid: GridData, trained: TabularModel
) -> None:
    roles = trained.roles
    counts = roles.counts()
    assert counts["probe"] == 50 and sum(counts.values()) == len(grid.tasks)
    probe = set(roles.probe_order)
    assert len(probe) == 50
    builder = FeatureBuilder(trained.spec, CATALOG, trained.probes)
    seen: dict[str, set[str]] = {}
    for role in ("train", "tune", "calibration", "test"):
        ds = build_dataset(grid, roles.tasks(role), CATALOG, builder, seed=7, max_history=1)
        seen[role] = set(ds.task_ids)
        assert seen[role] <= set(roles.tasks(role))
        assert not (seen[role] & probe)  # no probe task is a row of any role
    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (seen[a] & seen[b])  # a task's rows never straddle two roles
    # The probe table describes configurations only through probe tasks.
    for rows in trained.probes.results.values():
        assert set(rows) <= probe
    # Cascade rows are built from the same task's runs and stay in its role.
    train_ds = build_dataset(grid, roles.tasks("train"), CATALOG, builder, seed=7, max_history=1)
    assert 1 in train_ds.history and set(train_ds.task_ids) <= set(roles.tasks("train"))


def test_groups_stay_whole_and_cut_probe_groups_are_excluded() -> None:
    ids = [f"t{i:03d}" for i in range(300)]
    group = {t: f"repo{int(t[1:]) // 7}" for t in ids}  # 43 groups of up to 7
    roles = assign_roles(ids, seed=1, n_probe=20, group_of=group.__getitem__)
    by_group: dict[str, set[str]] = {}
    for t, r in roles.role.items():
        by_group.setdefault(group[t], set()).add(r)
    for g, rs in by_group.items():
        # A group is whole in one role; only a group cut to size has probe and excluded tasks.
        assert len(rs) == 1 or rs == {"probe", "excluded"}, (g, rs)
    assert roles.counts()["probe"] == 20
    again = assign_roles(ids, seed=1, n_probe=20, group_of=group.__getitem__)
    assert again.role == roles.role and again.probe_order == roles.probe_order
    other = assign_roles(ids, seed=2, n_probe=20, group_of=group.__getitem__)
    assert other.role != roles.role


def test_grid_splits_are_respected_when_train_and_test_both_exist() -> None:
    ids = [f"t{i:03d}" for i in range(200)]
    splits: dict[str, str | None] = {
        t: ("test" if i % 5 == 0 else "validation" if i % 5 == 1 else "train")
        for i, t in enumerate(ids)
    }
    roles = assign_roles(ids, seed=0, n_probe=10, grid_splits=splits)
    assert roles.source == "grid splits"
    for t in ids:
        if splits[t] == "test":
            assert roles.role[t] == "test"
        elif splits[t] == "validation":
            assert roles.role[t] == "calibration"
        else:
            assert roles.role[t] in ("probe", "train", "tune")
    # With only one split name the file's splits are ignored, and the notes say so.
    flat = assign_roles(ids, seed=0, n_probe=10, grid_splits=dict.fromkeys(ids, "test"))
    assert flat.source == "seeded random" and flat.notes


# --- calibration -----------------------------------------------------------------------------

# Hand example. Five rows predicted 0.2 with two successes, five predicted 0.8 with five.
HAND_P = [0.2] * 5 + [0.8] * 5
HAND_Y = [1, 1, 0, 0, 0] + [1, 1, 1, 1, 1]


def test_metrics_on_a_hand_computed_example() -> None:
    # Bin [0.2, 0.3) holds the first five: observed 0.4 against 0.2, gap 0.2.
    # Bin [0.8, 0.9) holds the rest: observed 1.0 against 0.8, gap 0.2. ECE = 0.5*0.2 + 0.5*0.2.
    assert ece(HAND_P, HAND_Y) == pytest.approx(0.2)
    # Brier: first group 2*(0.8)^2 + 3*(0.2)^2 = 1.40; second 5*(0.2)^2 = 0.20; mean 1.6 / 10.
    assert brier(HAND_P, HAND_Y) == pytest.approx(0.16)
    # Log loss: 2*-ln(0.2) + 3*-ln(0.8) + 5*-ln(0.8), over 10 rows.
    assert log_loss(HAND_P, HAND_Y) == pytest.approx((-2 * math.log(0.2) - 8 * math.log(0.8)) / 10)
    # AUC: 7 positives x 3 negatives = 21 pairs; the 5 positives at 0.8 beat all 3 (15), the 2
    # positives at 0.2 tie with the 3 negatives at 0.2 (6 pairs, half each = 3). 18 / 21.
    assert auc(HAND_P, HAND_Y) == pytest.approx(18 / 21)
    n, gap = window_gap(HAND_P, HAND_Y, 0.2)
    assert n == 5 and gap == pytest.approx(0.2)
    assert window_gap(HAND_P, HAND_Y, 0.5) == (0, None)
    rel = reliability(HAND_P, HAND_Y)
    assert [r["count"] for r in rel] == [0, 0, 5, 0, 0, 0, 0, 0, 5, 0]
    assert rel[2]["observed"] == pytest.approx(0.4)
    assert rel[8]["mean_predicted"] == pytest.approx(0.8)
    assert ece([0.15, 0.85], [0, 1]) == pytest.approx(0.15)
    assert auc([0.1, 0.2], [1, 1]) is None
    assert ece([1.0], [1]) == 0.0  # p = 1.0 falls in the last bin


def test_platt_recovers_a_known_curve_and_isotonic_is_monotone() -> None:
    a, b = 0.5, 0.3
    margins: list[float] = []
    labels: list[int] = []
    for m in (-3.0, -1.0, 0.0, 1.0, 3.0):
        k = round(sigmoid(a * m + b) * 4000)
        margins += [m] * 4000
        labels += [1] * k + [0] * (4000 - k)
    cal = fit_calibrator("platt", margins, labels)
    assert cal.kind == "platt"
    assert cal.a == pytest.approx(a, abs=0.01) and cal.b == pytest.approx(b, abs=0.01)
    iso = fit_calibrator("isotonic", margins, labels)
    points = [-5 + 0.25 * i for i in range(41)]
    values = [iso(m) for m in points]
    assert values == sorted(values) and all(0.0 < v < 1.0 for v in values)  # never 0 or 1
    # A single class, too few rows, or "none": the identity, never a wrong fit.
    assert fit_calibrator("platt", [0.0] * 30, [1] * 30).kind == "none"
    assert fit_calibrator("platt", [0.0] * 5, [1, 0, 1, 0, 1]).kind == "none"
    assert fit_calibrator("none", margins, labels).kind == "none"
    again = Calibrator.from_dict(json.loads(json.dumps(iso.to_dict())))
    assert [again(m) for m in points] == values


def test_auto_keeps_the_identity_for_calibrated_scores_and_fixes_overconfident_ones() -> None:
    rng = random.Random(0)
    groups = [f"g{i % 40}" for i in range(4000)]
    margins = [rng.gauss(0, 2) for _ in groups]
    ok = [int(rng.random() < sigmoid(m)) for m in margins]  # already calibrated
    cal, losses = select_calibrator(margins, ok, groups, seed=0)
    assert cal.kind == "none" and set(losses) == {"none", "platt", "isotonic"}
    over = [3.0 * m for m in margins]  # the same scores, made overconfident
    cal2, _ = select_calibrator(over, ok, groups, seed=0)
    assert cal2.kind in ("platt", "isotonic")
    fixed = log_loss([cal2(m) for m in over], ok)
    assert fixed < log_loss([sigmoid(m) for m in over], ok) - 0.05


def test_calibrated_model_is_not_worse_than_uncalibrated_on_held_out_tasks(
    tmp_path: Path,
) -> None:
    """On the held-out test tasks, by the metrics the report gives (ECE, Brier, log loss).

    Two models: the default (``auto`` verifies calibration and keeps the identity when it does
    not help, so it cannot be worse), and an explicit Platt fit, which has to earn it.
    """
    runs, tasks = make_grid(WORLD, 900, 5, split=None)
    path = tmp_path / "g.jsonl"
    write_grid(path, runs, tasks, {"generator": "steproute.eval.synthetic"})
    g = load_grid(path)
    for method in ("auto", "platt"):
        model = train(g, CATALOG, TrainConfig(seed=2, calibration=method)).model
        rep = calibration_report(model, g, CATALOG, replicates=0)
        for key, grp in rep["groups"].items():
            u, c = grp["uncalibrated"], grp["calibrated"]
            for metric in ("ece", "brier", "log_loss"):
                assert c[metric] <= u[metric] + 0.002, (method, key, metric)
            assert c["auc"] == pytest.approx(u["auc"], abs=0.01)  # a monotone map keeps ranking
    assert model.calibrator.kind == "platt"


def test_calibration_report_has_intervals_thresholds_and_the_stamp(
    grid: GridData, trained: TabularModel
) -> None:
    rep = calibration_report(trained, grid, CATALOG, replicates=40, seed=1)
    assert rep["stamp"] == STAMP and rep["synthetic"]
    assert set(rep["groups"]) == {"first_step", "after_one_failure"}
    g = rep["groups"]["first_step"]
    cal = g["calibrated"]
    assert cal["ece_ci"][0] <= cal["ece_ci"][1]
    assert cal["brier_ci"][0] <= cal["brier"] <= cal["brier_ci"][1] + 0.05
    assert [t["tau"] for t in cal["thresholds"]] == [0.5, 0.7, 0.8, 0.9]
    assert sum(r["count"] for r in cal["reliability"]) == g["n_rows"]
    assert g["brier_of_constant_base_rate"] > cal["brier"]  # it has skill over the base rate
    assert calibration_report(trained, grid, CATALOG, replicates=40, seed=1) == rep


# --- reproducibility -------------------------------------------------------------------------

_TRAIN_CODE = "import sys\nfrom steproute.cli import main\nsys.exit(main(sys.argv[1:]))\n"


def _train_in_new_process(grid_path: Path, cat: Path, out: Path, seed: int) -> None:
    proc = subprocess.run(
        [
            sys.executable, "-c", _TRAIN_CODE, "ml", "train",
            "--data", str(grid_path), "--catalog", str(cat), "--out", str(out),
            "--seed", str(seed),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr


def test_same_seed_gives_the_same_model_in_separate_processes(
    example: dict[str, Path], grid: GridData, tmp_path: Path
) -> None:
    """Two trainings in two fresh processes: byte-identical model file, identical predictions."""
    outs = [tmp_path / "a", tmp_path / "b", tmp_path / "c"]
    for out, seed in zip(outs, (11, 11, 12), strict=True):
        _train_in_new_process(example["grid"], example["catalog"], out, seed)
    a, b, c = (TabularModel.load(o) for o in outs)
    assert (outs[0] / "model.txt").read_bytes() == (outs[1] / "model.txt").read_bytes()
    assert a.meta["data_fingerprint"] == b.meta["data_fingerprint"]
    assert a.meta["data_fingerprint"] != c.meta["data_fingerprint"]
    pairs = _fixed_inputs(CATALOG, grid)
    pa = TabularPredictor(a, CATALOG).predict_many(pairs)
    pb = TabularPredictor(b, CATALOG).predict_many(pairs)
    pc = TabularPredictor(c, CATALOG).predict_many(pairs)
    assert pa == pb  # identical, not merely close
    assert pa != pc  # a different seed is a different model
    assert a.meta["seed"] == 11 and a.meta["lightgbm_params"]["deterministic"] is True
    assert a.meta["lightgbm_params"]["num_threads"] == 1
    assert {"python", "lightgbm", "numpy"} <= set(a.meta["versions"])


def test_saved_and_reloaded_model_predicts_the_same(
    grid: GridData, trained: TabularModel, tmp_path: Path
) -> None:
    path = trained.save(tmp_path / "m")
    loaded = TabularModel.load(path)
    pairs = _fixed_inputs(CATALOG, grid)
    live = TabularPredictor(trained, CATALOG).predict_many(pairs)
    assert TabularPredictor(loaded, CATALOG).predict_many(pairs) == live
    # No pickle: four plain files.
    assert sorted(p.name for p in path.iterdir()) == [
        "meta.json", "model.txt", "probes.json", "roles.json",
    ]  # fmt: skip


# --- the saved format ------------------------------------------------------------------------


def _edit_meta(path: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    meta = json.loads((path / "meta.json").read_text())
    edit(meta)
    (path / "meta.json").write_text(json.dumps(meta))


def test_loader_refuses_a_feature_spec_mismatch(trained: TabularModel, tmp_path: Path) -> None:
    saved = trained.save(tmp_path / "model")
    TabularModel.load(saved)  # sanity: it loads as saved

    def reorder(meta: dict[str, Any]) -> None:
        names = meta["feature_spec"]["names"]
        names[0], names[1] = names[1], names[0]

    _edit_meta(saved, reorder)
    with pytest.raises(FeatureSpecMismatch, match="column list differs"):
        TabularModel.load(saved)


def test_loader_refuses_other_feature_versions_and_spec_edits(
    trained: TabularModel, tmp_path: Path
) -> None:
    def bump(meta: dict[str, Any]) -> None:
        meta["feature_spec"]["version"] += 1

    p = trained.save(tmp_path / "v")
    _edit_meta(p, bump)
    with pytest.raises(FeatureSpecMismatch, match="feature version"):
        TabularModel.load(p)

    # A consistent edit of spec and names still fails against the booster's own columns.
    def drop_task_key(meta: dict[str, Any]) -> None:
        fs = meta["feature_spec"]
        fs["task_keys"] = []
        fs["names"] = list(FeatureSpec(task_keys=()).names)

    q = trained.save(tmp_path / "t")
    _edit_meta(q, drop_task_key)
    with pytest.raises(FeatureSpecMismatch, match="booster's columns"):
        TabularModel.load(q)


def test_loader_refuses_tampering_and_other_formats(trained: TabularModel, tmp_path: Path) -> None:
    p = trained.save(tmp_path / "x")
    (p / "model.txt").write_text((p / "model.txt").read_text() + "\n# edited\n")
    with pytest.raises(ModelFormatError, match="hash"):
        TabularModel.load(p)
    q = trained.save(tmp_path / "y")
    _edit_meta(q, lambda m: m.update(format="something-else"))
    with pytest.raises(ModelFormatError, match="not a steproute-tabular"):
        TabularModel.load(q)
    r = trained.save(tmp_path / "z")
    _edit_meta(r, lambda m: m.update(format_version=99))
    with pytest.raises(ModelFormatError, match="format version"):
        TabularModel.load(r)
    with pytest.raises(ModelFormatError, match="cannot read"):
        TabularModel.load(tmp_path / "missing")


def test_saved_metadata_records_what_produced_the_model(
    trained: TabularModel, tmp_path: Path
) -> None:
    saved = trained.save(tmp_path / "model")
    meta = json.loads((saved / "meta.json").read_text())
    assert meta["format"] == "steproute-tabular" and meta["synthetic"] is True
    assert meta["stamp"] == STAMP
    assert meta["model_sha256"] == hashlib.sha256((saved / "model.txt").read_bytes()).hexdigest()
    assert meta["data"]["grid_sha256"] and len(meta["data_fingerprint"]) == 64
    assert meta["feature_spec"]["names"] == list(FeatureSpec(task_keys=("hint",)).names)
    assert meta["calibrator"]["kind"] in ("none", "platt", "isotonic")
    # No absolute paths or personal details in the metadata.
    assert str(Path.home()) not in (saved / "meta.json").read_text()


# --- the policy ------------------------------------------------------------------------------


def test_predictor_works_inside_scored_policy_and_batch_equals_per_call(
    grid: GridData, trained: TabularModel
) -> None:
    pred = TabularPredictor(trained, CATALOG)

    class PerCall:
        def predict(self, state: StepState, action_step: Action) -> float:
            return pred.predict(state, action_step)

    batch = ScoredPolicy(pred, CATALOG, 0.8, 0.0, seed=1)
    single = ScoredPolicy(PerCall(), CATALOG, 0.8, 0.0, seed=1)
    states = [_start(grid, t) for t in sorted(grid.tasks)[:5]]
    states.append(
        StepState("proxy", "s", 2, "main", 1000, current_model="cloud-mid",
                  current_effort="medium", cached_prefix_tokens=900, features={"hint": 2})
    )  # fmt: skip
    for s in states:
        assert batch.score(s) == single.score(s)  # exact, not approximate
        d = batch.decide(s)
        assert d.predicted_success is not None and 0.0 <= d.predicted_success <= 1.0
    # predict is predict_many of one pair, in [0, 1], and deterministic.
    a = Action("cloud-mid", "medium")
    assert pred.predict(states[0], a) == pred.predict_many([(states[0], a)])[0]
    assert pred.predict(states[0], a) == pred.predict(states[0], a)
    assert 0.0 <= pred.predict(states[0], a) <= 1.0
    assert pred.predict_many([]) == []


def test_a_failed_earlier_attempt_lowers_the_next_step_in_the_right_direction(
    grid: GridData, trained: TabularModel
) -> None:
    """The grid has paired runs, so the model learns that a task that beat a weak model is
    harder for the next one: P(success | weak model failed) is below the fresh P(success)."""
    pred = TabularPredictor(trained, CATALOG)
    fresh, cond = [], []
    for tid in sorted(grid.tasks)[:120]:
        st = _start(grid, tid)
        after = state_after_failure(st, Action("cloud-small", "low"), CATALOG.get("cloud-small"))
        fresh.append(pred.predict(st, Action("cloud-mid", "medium")))
        cond.append(pred.predict(after, Action("cloud-mid", "medium")))
    assert sum(cond) / len(cond) < sum(fresh) / len(fresh) - 0.02


def test_probes_move_predictions_for_a_model_with_no_training_rows(grid: GridData) -> None:
    """A held-out model: no training rows, only profile and probe results."""
    model = train(grid, CATALOG, TrainConfig(seed=3), holdout_models=["cloud-mid"]).model
    assert ("cloud-mid", "medium") in model.probes.results  # its probes still describe it
    st = _start(grid, sorted(grid.tasks)[0])
    a = Action("cloud-mid", "medium")
    cold = TabularPredictor(model, CATALOG, n_probes=0).predict(st, a)
    warm = TabularPredictor(model, CATALOG, n_probes=50).predict(st, a)
    assert 0.0 < cold < 1.0 and 0.0 < warm < 1.0 and cold != warm


def test_default_catalog_models_can_be_scored_without_probes(trained: TabularModel) -> None:
    default = load_default_catalog(warn_unverified=False)
    policy = ScoredPolicy(TabularPredictor(trained, default), default, 0.8, 0.0, seed=0)
    d = policy.decide(StepState("proxy", "s", 0, "main", 20_000, tools=["Bash"]))
    assert d.predicted_success is not None and 0.0 <= d.predicted_success <= 1.0


# --- the CLI and the report ------------------------------------------------------------------


def test_cli_train_and_evaluate_end_to_end(
    example: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rules = tmp_path / "rules.toml"
    rules.write_text(RULES_TOML)
    started = time.perf_counter()
    model_dir, out_dir = tmp_path / "model", tmp_path / "out"
    common = ["--data", str(example["grid"]), "--catalog", str(example["catalog"])]
    train_args = ["ml", "train", *common, "--out", str(model_dir), "--seed", "5"]
    eval_args = [
        "ml", "evaluate", "--model", str(model_dir), *common, "--rules", str(rules),
        "--out-dir", str(out_dir), "--bootstrap", "100", "--metric-bootstrap", "30",
    ]  # fmt: skip
    assert main(train_args) == 0
    assert main(eval_args) == 0
    assert time.perf_counter() - started < 30.0  # a few seconds in practice
    assert capsys.readouterr().out.count(STAMP) >= 4

    report = json.loads((out_dir / "report.json").read_text())
    assert report["format"] == "steproute-eval-report"
    assert report["stamp"] == STAMP and report["synthetic_data"] is True
    assert STAMP in report["notice"]
    names = [v["policy"] for v in report["verdicts"]]
    assert "tabular tau=0.8" in names and "tabular-uncalibrated tau=0.8" in names
    assert any(n.startswith("rules") for n in names) and "oracle" in names
    assert all(v["sentence"].startswith(STAMP) for v in report["verdicts"])
    assert all(s.startswith(STAMP) for s in report["summary"])
    grid_ev = report["evaluations"][0]
    assert grid_ev["stamp"] == STAMP and grid_ev["kind"] == "grid"
    assert all(p["verdict"]["sentence"].startswith(STAMP) for p in grid_ev["policies"])
    assert grid_ev["provenance"]["model"]["synthetic"] is True
    # Only the test tasks were scored: probe, train, tune and calibration tasks are not.
    n_test = TabularModel.load(model_dir).roles.counts()["test"]
    assert grid_ev["provenance"]["tasks_in_split"] == n_test
    assert grid_ev["provenance"]["tasks_evaluated"] == n_test

    svg = (out_dir / "report.svg").read_text()
    assert svg.count(STAMP) >= 3 and "</svg>" in svg
    md = (out_dir / "calibration.md").read_text()
    assert md.startswith(f"# {STAMP}") and md.rstrip().endswith(f"**{STAMP}**")
    cal = json.loads((out_dir / "calibration.json").read_text())
    assert cal["stamp"] == STAMP and "first_step" in cal["groups"]


def test_the_stamp_follows_the_data_not_a_flag(tmp_path: Path) -> None:
    """A grid whose header does not name the synthetic generator gets no stamp."""
    runs, tasks = make_grid(WORLD, 300, 9, split=None)
    path = tmp_path / "g.jsonl"
    write_grid(path, runs, tasks, {"runner": "replay 0.1"})
    g = load_grid(path)
    model = train(g, CATALOG, TrainConfig(seed=1)).model
    assert model.synthetic is False and model.meta["stamp"] is None
    report, svg, _ = evaluator_report(model, g, CATALOG, taus=(0.8,))
    assert "stamp" not in report and STAMP not in json.dumps(report) and STAMP not in svg
    assert calibration_report(model, g, CATALOG, replicates=0)["stamp"] is None


def test_models_missing_from_the_catalog_are_refused(grid: GridData) -> None:
    with pytest.raises(ValueError, match="catalog does not"):
        train(grid, load_default_catalog(warn_unverified=False))


def test_checked_in_synthetic_artifacts_carry_the_stamp() -> None:
    folder = ROOT / "docs" / "synthetic" / "tabular-predictor"
    assert folder.is_dir()
    assert not list(folder.rglob("model.txt")) and not list(folder.rglob("*.pkl"))
    for name in ("report.json", "report.svg", "calibration.md", "README.md"):
        assert STAMP in (folder / name).read_text(), name
    report = json.loads((folder / "report.json").read_text())
    assert report["stamp"] == STAMP
    assert all(v["sentence"].startswith(STAMP) for v in report["verdicts"])
    card = (ROOT / "docs" / "MODEL-CARD-TABULAR.md").read_text()
    assert card.startswith(f"# {STAMP}") and card.rstrip().endswith(f"**{STAMP}**")
