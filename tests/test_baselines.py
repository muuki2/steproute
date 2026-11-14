"""External baselines (issue 22): the shared data, inputs, policy and report, and each adapter's
own logic, on small fakes and the evaluator's synthetic world.

The shared code needs numpy (the ml extra). The trained adapters need the baselines extra
(paretobandit, torch, torch-geometric); those tests skip cleanly without it.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy")

import numpy as np  # noqa: E402

from steproute.baselines import graphrouter, paretobandit, routellm  # noqa: E402
from steproute.baselines.common import (  # noqa: E402
    ArmPolicy,
    BaselineData,
    ContextEncoder,
    arm_label,
    baseline_report,
)
from steproute.core.schema import StepState  # noqa: E402
from steproute.eval.data import GridData, load_grid, write_grid  # noqa: E402
from steproute.eval.results import EvalSettings  # noqa: E402
from steproute.eval.synthetic import (  # noqa: E402
    RULES_TOML,
    catalog,
    default_world,
    make_grid,
)
from steproute.ml.features import config_order  # noqa: E402
from steproute.ml.splits import assign_roles  # noqa: E402
from steproute.ml.stamp import STAMP  # noqa: E402

FAST = EvalSettings(bootstrap=100, seed=0)


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A 300-task synthetic grid on disk, its catalog and rules."""
    world = default_world()
    d = tmp_path_factory.mktemp("synth")
    runs, tasks = make_grid(world, 300, seed=3)
    write_grid(d / "grid.jsonl", runs, tasks, {"generator": "steproute.eval.synthetic", "seed": 3})
    from steproute.eval.synthetic import catalog_toml

    (d / "catalog.toml").write_text(catalog_toml(world), encoding="utf-8")
    (d / "rules.toml").write_text(RULES_TOML, encoding="utf-8")
    return {"dir": d, "grid": load_grid(d / "grid.jsonl"), "catalog": catalog(world)}


@pytest.fixture(scope="module")
def data(synth: dict[str, Any]) -> BaselineData:
    return BaselineData(synth["grid"], synth["catalog"], seed=0, n_probes=20)


@pytest.fixture(scope="module")
def encoder(data: BaselineData) -> ContextEncoder:
    return ContextEncoder(data.catalog).fit([data.state(t) for t in data.tasks("train")])


# --- shared: data ---------------------------------------------------------------------------


def test_roles_match_the_tabular_split(data: BaselineData, synth: dict[str, Any]) -> None:
    grid: GridData = synth["grid"]
    roles = assign_roles(
        sorted(grid.tasks),
        seed=0,
        n_probe=20,
        grid_splits={t: grid.tasks[t].split for t in grid.tasks},
    )
    assert data.roles.role == roles.role
    assert set(data.tasks("test")) <= set(roles.tasks("test"))
    assert not set(data.tasks("train")) & set(data.tasks("test"))
    assert data.arms == tuple(config_order(grid.configs(), synth["catalog"]))


def test_outcomes_are_the_grid_cells(data: BaselineData, synth: dict[str, Any]) -> None:
    tasks = data.tasks("train")[:5]
    s, c = data.outcomes(tasks)
    cells = synth["grid"].cells()
    for i, t in enumerate(tasks):
        for j, arm in enumerate(data.arms):
            assert s[i, j] == cells[(t, arm)].success
            assert c[i, j] == pytest.approx(cells[(t, arm)].cost)


def test_incomplete_tasks_are_left_out(synth: dict[str, Any]) -> None:
    grid: GridData = synth["grid"]
    broken = next(iter(sorted(grid.tasks)))
    runs = tuple(
        r if r.task_id != broken else dataclasses.replace(r, success=None) for r in grid.runs
    )
    g2 = GridData(grid.source, runs, grid.tasks, grid.header)
    d2 = BaselineData(g2, synth["catalog"], seed=0, n_probes=20)
    role = d2.roles.role[broken]
    assert broken not in d2.tasks(role)


def test_a_sparse_configuration_can_be_excluded(synth: dict[str, Any]) -> None:
    """A configuration run on a few tasks only would empty every role; excluding it keeps the
    others, and the report's frontier leaves it out too."""
    grid: GridData = synth["grid"]
    keep = set(sorted(grid.tasks)[:30])
    sparse = ("cloud-big", "high")
    runs = tuple(r for r in grid.runs if r.config != sparse or r.task_id in keep)
    g2 = GridData(grid.source, runs, grid.tasks, grid.header)
    dense = BaselineData(g2, synth["catalog"], seed=0, n_probes=20)
    assert len(dense.tasks("test")) < 15
    d2 = BaselineData(g2, synth["catalog"], seed=0, n_probes=20, exclude_configs=[sparse])
    assert sparse not in d2.arms and len(d2.tasks("test")) > 30
    enc = ContextEncoder(d2.catalog).fit([d2.state(t) for t in d2.tasks("train")])
    _, _, ev = baseline_report(d2, [_hint_policy(d2, enc)], {}, settings=FAST)
    labels = [c["label"] for c in ev.to_json()["frontier"]["configurations"]]
    assert "cloud-big (high)" not in labels and "cloud-big (medium)" in labels
    assert ev.provenance["baseline"]["excluded_configs"] == ["high:cloud-big"]


# --- shared: inputs -------------------------------------------------------------------------


def _state(hint: int, **kw: Any) -> StepState:
    return StepState(
        harness="t",
        session_id="s",
        step_idx=0,
        kind="main",
        prompt_tokens=1000,
        source="replay",
        features={"hint": hint, **kw},
    )


def test_encoder_keeps_varying_columns_and_one_hots_small_integers() -> None:
    cat = catalog(default_world())
    enc = ContextEncoder(cat).fit([_state(h) for h in (0, 1, 2, 2, 3)])
    # Everything but the hint is constant on these states, so it is dropped.
    assert enc.names == (
        "task_hint",
        "task_hint==0",
        "task_hint==1",
        "task_hint==2",
        "task_hint==3",
    )
    x = enc.transform_many([_state(h) for h in (0, 1, 2, 2, 3)])
    assert x[:, 0].mean() == pytest.approx(0.0)
    assert x[:, 0].std() == pytest.approx(1.0)
    assert x[:, 1:].sum(axis=1).tolist() == [1, 1, 1, 1, 1]
    unseen = enc.transform(_state(7))
    assert unseen[1:].tolist() == [0, 0, 0, 0]


def test_encoder_does_not_one_hot_continuous_or_wide_columns() -> None:
    cat = catalog(default_world())
    states = [_state(h, score=h * 0.37) for h in range(12)]
    enc = ContextEncoder(cat, max_levels=8).fit(states)
    assert "task_hint" in enc.names and "task_score" in enc.names
    assert not any("==" in n for n in enc.names)


def test_encoder_reads_no_text_or_ids() -> None:
    cat = catalog(default_world())
    a = _state(1)
    b = dataclasses.replace(a, task_text="secret prompt", task_id="x", session_id="y")
    enc = ContextEncoder(cat).fit([_state(0), _state(1), _state(2)])
    assert enc.transform(a).tolist() == enc.transform(b).tolist()


# --- shared: policy and report --------------------------------------------------------------


def test_arm_policy_is_deterministic_and_cached(
    data: BaselineData, encoder: ContextEncoder
) -> None:
    calls: list[int] = []
    rng = np.random.default_rng(0)

    def choose(x: Any) -> tuple[str, str]:
        calls.append(1)
        return data.arms[int(rng.integers(len(data.arms)))]

    pol = ArmPolicy("random-once", choose, encoder)
    s = data.state(data.tasks("test")[0])
    first = pol.distribution(s)
    assert all(pol.distribution(s) == first for _ in range(5))
    assert len(calls) == 1
    ((action, p),) = first
    assert p == 1.0 and (action.model, action.effort) in data.arms


def _hint_policy(data: BaselineData, encoder: ContextEncoder) -> ArmPolicy:
    hint_col = encoder.names.index("task_hint")
    order = sorted(data.arms, key=data.arms.index)

    def choose(x: Any) -> tuple[str, str]:
        z = float(x[hint_col])
        return order[min(len(order) - 1, max(0, int(round(z + 2))))]

    return ArmPolicy("fake hint router", choose, encoder)


def test_report_scores_test_tasks_with_rules_and_oracle_and_is_stamped(
    data: BaselineData, encoder: ContextEncoder, synth: dict[str, Any]
) -> None:
    pol = _hint_policy(data, encoder)
    report, svg, ev = baseline_report(
        data,
        [pol],
        {"upstream": {"name": "fake"}},
        rules=str(synth["dir"] / "rules.toml"),
        settings=FAST,
        command=["python", "-m", "steproute.baselines", "fake", "..."],
    )
    names = [p.name for p in ev.policies]
    assert names == ["fake hint router", "rules", "oracle"]
    assert report["stamp"] == STAMP and report["synthetic_data"] is True
    assert all(v["sentence"].startswith(STAMP) for v in report["verdicts"])
    assert STAMP in svg
    assert ev.provenance["baseline"]["tasks_by_role"]["test"] == len(data.tasks("test"))
    text = json.dumps(report)
    assert str(synth["dir"]) not in text  # no local paths in the report


def test_report_without_synthetic_header_is_not_stamped(
    data: BaselineData, encoder: ContextEncoder, synth: dict[str, Any]
) -> None:
    grid: GridData = synth["grid"]
    real = GridData(grid.source, grid.runs, grid.tasks, {"generator": "replay"})
    d2 = BaselineData(real, data.catalog, seed=0, n_probes=20)
    report, svg, _ = baseline_report(
        d2, [_hint_policy(d2, encoder)], {}, rules=str(synth["dir"] / "rules.toml"), settings=FAST
    )
    assert "stamp" not in report and STAMP not in svg


def test_the_logged_evaluator_accepts_a_baseline_policy(
    data: BaselineData, encoder: ContextEncoder
) -> None:
    """The adapters are ordinary TargetPolicy objects, so step-level traces can score them too."""
    from steproute.eval.data import logged_from_pairs
    from steproute.eval.ope import evaluate_logged
    from steproute.eval.synthetic import make_logged

    pairs = make_logged(default_world(), 40, seed=5)
    ev = evaluate_logged(logged_from_pairs(pairs), [_hint_policy(data, encoder)], FAST)
    assert ev.policies[0].name == "fake hint router"


# --- adapters: logic without the heavy libraries --------------------------------------------


def test_paretobandit_arm_prices_and_anchors(data: BaselineData) -> None:
    arms = paretobandit.BanditArms.from_data(data, data.tasks("train"))
    reg = arms.registry()
    assert len(reg) == len(data.arms)
    for arm, cost in zip(arms.arms, arms.mean_cost, strict=True):
        entry = reg[f"{arm[0]}|{arm[1]}"]
        # The router's per-1k price is (input + output) / 2 / 1000: the mean task cost.
        assert (entry["input_cost_per_m"] + entry["output_cost_per_m"]) / 2 / 1000 == (
            pytest.approx(cost)
        )
    lo, hi = arms.anchors()
    assert lo == min(c for c in arms.mean_cost if c > 0) and hi == max(arms.mean_cost)


def test_graphrouter_performance_and_scaling() -> None:
    s = np.array([[1.0, 0.0], [1.0, 1.0]])
    c = np.array([[0.0, 2.0], [1.0, 4.0]])
    p = graphrouter.performance(s, c, beta=0.4, alpha=0.6)
    assert p[0, 0] == pytest.approx(0.6) and p[0, 1] == pytest.approx(-0.2)
    x = np.array([[1.0, 5.0], [3.0, 5.0]])
    assert graphrouter.minmax(x).tolist() == [[0.0, 0.0], [1.0, 0.0]]
    assert graphrouter.minmax(np.array([[5.0, 5.0]]), x).tolist() == [[2.0, 0.0]]


def test_graphrouter_llm_features_one_row_per_arm(data: BaselineData) -> None:
    feats, names = graphrouter.llm_features(data)
    assert feats.shape == (len(data.arms), len(names))
    assert np.isfinite(feats).all() and feats.min() >= 0 and feats.max() <= 1
    assert len({tuple(r) for r in feats.tolist()}) == len(data.arms)  # arms are distinguishable


def test_routellm_battles_drop_ties() -> None:
    s = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert sorted(routellm.battles(s)) == [(0, 1, 0), (2, 1, 0)]


def test_routellm_default_pair_and_thresholds() -> None:
    arms = [("a", "low"), ("b", "low"), ("c", "high")]
    s = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    c = np.array([[0.0, 1.0, 5.0], [0.0, 1.0, 5.0]])
    assert routellm.default_pair(arms, s, c) == (0, 2)
    cuts = routellm.thresholds([i / 99 for i in range(100)], [0.1, 0.5, 0.9])
    assert cuts == pytest.approx([0.9, 0.5, 0.1])


def test_cli_names_the_missing_extra(synth: dict[str, Any], tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['pareto_bandit'] = None\n"
        "from steproute.baselines.__main__ import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, "routellm", "--data", str(synth["dir"] / "grid.jsonl"),
         "--out-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert out.returncode == 2
    assert "[baselines]" in out.stderr


def test_adapters_import_without_the_extra() -> None:
    code = (
        "import sys\n"
        "for m in ('torch', 'torch_geometric', 'pareto_bandit'):\n"
        "    sys.modules[m] = None\n"
        "import steproute.baselines.common, steproute.baselines.paretobandit\n"
        "import steproute.baselines.graphrouter, steproute.baselines.routellm\n"
        "import steproute.baselines.__main__\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


# --- adapters: trained (baselines extra) ----------------------------------------------------


def _two_arm_data(n: int = 120) -> tuple[paretobandit.BanditArms, Any, Any]:
    """Arm 0 (cheap) solves tasks with x < 0, arm 1 (dear) those with x > 0."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n, 1))
    success = np.stack([(x[:, 0] < 0), (x[:, 0] > 0)], axis=1).astype(float)
    arms = paretobandit.BanditArms((("cheap", "low"), ("dear", "high")), (0.01, 1.0))
    return arms, x, success


def test_paretobandit_full_learns_the_context() -> None:
    pytest.importorskip("pareto_bandit")
    arms, x, success = _two_arm_data()
    router = paretobandit.train_router(arms, x, success, mode="full", cost_penalty=0.0, seed=0)
    pick = paretobandit._chooser(router, arms.by_id())
    assert pick(np.array([-1.5])) == ("cheap", "low")
    assert pick(np.array([1.5])) == ("dear", "high")
    # A heavy cost penalty sends everything to the cheap arm.
    stingy = paretobandit.train_router(arms, x, success, mode="full", cost_penalty=5.0, seed=0)
    assert paretobandit._chooser(stingy, arms.by_id())(np.array([1.5])) == ("cheap", "low")


def test_paretobandit_online_sees_only_the_picked_arm() -> None:
    pytest.importorskip("pareto_bandit")
    arms, x, success = _two_arm_data(60)
    router = paretobandit.train_router(arms, x, success, mode="online", cost_penalty=0.0, seed=0)
    assert router.bandit.t <= len(x)  # one feedback per task, not one per arm
    with pytest.raises(ValueError):
        paretobandit.train_router(arms, x, success, mode="nope", cost_penalty=0.0, seed=0)


def test_paretobandit_policies_on_synthetic(data: BaselineData, encoder: ContextEncoder) -> None:
    pytest.importorskip("pareto_bandit")
    pols, prov = paretobandit.build_policies(
        data, encoder, cost_penalties=(0.0, 1.0), modes=("full",)
    )
    assert [p.name for p in pols] == [
        "paretobandit full lambda_c=0",
        "paretobandit full lambda_c=1",
    ]
    assert prov["upstream"]["commit"] == paretobandit.UPSTREAM.commit
    test = data.tasks("test")
    cost = {a: c for a, c in zip(data.arms, data.outcomes(test)[1].mean(axis=0), strict=True)}
    spend = [np.mean([cost[p.arm(data.state(t))] for t in test]) for p in pols]
    assert spend[1] <= spend[0]


def test_graphrouter_fast_scoring_matches_the_upstream_graph_predict() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from steproute.baselines._vendor.graphrouter_gnn import GNNPredictor, build_graph

    rng = np.random.default_rng(0)
    nq, nl, dq, dl = 30, 4, 5, 3
    q = rng.random((nq, dq))
    llm = rng.random((nl, dl))
    perf = rng.random((nq, nl))
    label = np.eye(nl)[perf.argmax(axis=1)]
    edges = nq * nl
    train = torch.zeros(edges)
    train[: 24 * nl] = 1
    val = torch.zeros(edges)
    val[24 * nl :] = 1
    cfg = {"train_epoch": 5, "llm_num": nl}
    pred = GNNPredictor(dq, dl, 8, 1, cfg)
    g = build_graph(q, llm, perf.reshape(-1), label.reshape(-1), train, val, train)
    gv = build_graph(q, llm, perf.reshape(-1), label.reshape(-1), train, val, val)
    pred.train_validate(g, gv, torch.Generator().manual_seed(0))
    states = pred.llm_states(g)
    new = rng.random((3, dq))
    fast = pred.score_queries(new, states).argmax(dim=1).tolist()
    # Upstream style: append the new queries to the graph with zero, invisible edges.
    q2 = np.vstack([q, new])
    perf2 = np.vstack([perf, np.zeros((3, nl))])
    e2 = len(q2) * nl
    tm, vm, pm = torch.zeros(e2), torch.zeros(e2), torch.zeros(e2)
    tm[: 24 * nl] = 1
    vm[24 * nl : edges] = 1
    pm[edges:] = 1
    lab2 = np.eye(nl)[perf2.argmax(axis=1)]
    g2 = build_graph(q2, llm, perf2.reshape(-1), lab2.reshape(-1), tm, vm, pm)
    assert pred.predict(g2).tolist() == fast


def test_graphrouter_policies_on_synthetic(data: BaselineData, encoder: ContextEncoder) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pols, prov = graphrouter.build_policies(data, encoder, betas=(0.0,))
    assert [p.name for p in pols] == ["graphrouter beta=0"]
    assert prov["upstream"]["licence"] == "MIT"
    assert {p.arm(data.state(t)) for p in pols for t in data.tasks("test")} <= set(data.arms)


def test_routellm_mf_learns_and_routes_by_share(
    data: BaselineData, encoder: ContextEncoder
) -> None:
    pytest.importorskip("torch")
    pols, prov = routellm.build_policies(data, encoder, shares=(0.1, 0.9))
    test = data.tasks("test")
    strong = prov["pair"]["strong"]
    share = [np.mean([arm_label(p.arm(data.state(t))) == strong for t in test]) for p in pols]
    assert share[0] < share[1]
    assert prov["tune_pairwise_accuracy"] is None or prov["tune_pairwise_accuracy"] > 0.5
    with pytest.raises(ValueError):
        routellm.build_policies(data, encoder, pair=("low:nope", "high:cloud-big"))


def test_cli_writes_stamped_reports(synth: dict[str, Any], tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("pareto_bandit")
    from steproute.baselines.__main__ import main

    d = synth["dir"]
    rc = main(
        ["routellm", "--data", str(d / "grid.jsonl"), "--catalog", str(d / "catalog.toml"),
         "--rules", str(d / "rules.toml"), "--out-dir", str(tmp_path), "--probes", "20",
         "--bootstrap", "100", "--pair", "medium:local-small,high:cloud-big"]
    )  # fmt: skip
    assert rc == 0
    report = json.loads((tmp_path / "routellm" / "report.json").read_text())
    assert report["stamp"] == STAMP
    base = report["evaluations"][0]["provenance"]["baseline"]
    assert base["pair"] == {"weak": "medium:local-small", "strong": "high:cloud-big"}
    assert str(tmp_path) not in json.dumps(report) and str(d) not in json.dumps(report)
