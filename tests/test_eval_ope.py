"""Off-policy estimators against hand-computed examples, and their failure modes."""

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from steproute.core import CandidateSpec, ScoredPolicy, TraceStore
from steproute.core.schema import Action, CandidateSet, Decision, Outcome, StepState
from steproute.eval.data import LoggedRow, load_logged, logged_from_pairs
from steproute.eval.ope import evaluate_logged
from steproute.eval.results import NOT_SHOWN, EvalSettings
from steproute.eval.synthetic import (
    SynthLogging,
    catalog,
    default_world,
    hint_policy,
    make_logged,
    true_logged_value,
)
from steproute.eval.targets import DistributionTarget, FixedTarget, as_target

X, Y = Action("x", "low"), Action("y", "low")
LOOSE = EvalSettings(bootstrap=100, min_ess=0, min_clusters=1)


def _state(i: int, **kw: object) -> StepState:
    return StepState("h", f"s{i}", 0, "main", 100, **kw)  # type: ignore[arg-type]


def _pair(
    i: int, a: Action, mu: float, r: bool | None, c: float | None, cands: Sequence[Action] = (X, Y)
) -> tuple[Decision, Outcome]:
    d = Decision(f"d{i}", 1.0, _state(i), list(cands), a, "log", mu)
    return d, Outcome(f"d{i}", 2.0, r, cost_usd=c)


#: Four steps in four sessions; the logging policy gave the taken action these propensities.
HAND = [
    _pair(1, X, 0.5, True, 1.0),
    _pair(2, X, 0.5, False, 1.0),
    _pair(3, Y, 0.5, True, 3.0),
    _pair(4, Y, 0.25, True, 3.0),
]


@dataclass
class ConstModel:
    """External reward model (predict only): x -> (0.5, $1), y -> (0.6, $2)."""

    name: str = "const"

    def predict(self, state: StepState, action: Action) -> tuple[float, float]:
        return (0.5, 1.0) if action == X else (0.6, 2.0)


def _est(ev, name, estimator, metric):  # type: ignore[no-untyped-def]
    p = next(p for p in ev.policies if p.name == name)
    return p.estimators[estimator][metric].value


def test_estimators_by_hand_deterministic_target() -> None:
    ev = evaluate_logged(
        logged_from_pairs(HAND), [FixedTarget(Y, "y")], LOOSE, reward_model=ConstModel(),
        frontier=False,
    )  # fmt: skip
    # w = 0, 0, 2, 4
    assert _est(ev, "y", "ips", "success_rate") == pytest.approx(6 / 4)  # IPS can exceed 1
    assert _est(ev, "y", "snips", "success_rate") == pytest.approx(6 / 6)
    assert _est(ev, "y", "dr", "success_rate") == pytest.approx((0.6 + 0.6 + 1.4 + 2.2) / 4)
    assert _est(ev, "y", "ips", "cost_per_step") == pytest.approx(18 / 4)
    assert _est(ev, "y", "snips", "cost_per_step") == pytest.approx(18 / 6)
    assert _est(ev, "y", "dr", "cost_per_step") == pytest.approx((2 + 2 + 4 + 6) / 4)
    assert _est(ev, "y", "dr", "cost_per_solved") == pytest.approx(3.5 / 1.2)
    assert ev.policies[0].diagnostics["ess_success"] == pytest.approx(36 / 20)


def test_estimators_by_hand_stochastic_target() -> None:
    half = DistributionTarget(lambda s: ((X, 0.5), (Y, 0.5)), "half")
    ev = evaluate_logged(
        logged_from_pairs(HAND), [half], LOOSE, reward_model=ConstModel(), frontier=False
    )
    # w = 1, 1, 1, 2; direct term 0.5 * 0.5 + 0.5 * 0.6 = 0.55
    assert _est(ev, "half", "ips", "success_rate") == pytest.approx(4 / 4)
    assert _est(ev, "half", "snips", "success_rate") == pytest.approx(4 / 5)
    assert _est(ev, "half", "dr", "success_rate") == pytest.approx((1.05 + 0.05 + 0.95 + 1.35) / 4)


def test_low_ess_withholds_the_headline_but_keeps_the_estimators() -> None:
    ev = evaluate_logged(
        logged_from_pairs(HAND), [FixedTarget(Y, "y")], EvalSettings(bootstrap=100),
        reward_model=ConstModel(),
    )  # fmt: skip
    p = ev.policies[0]
    assert p.metrics["success_rate"].value is None
    assert "withheld: effective sample size 1.8" in str(p.metrics["success_rate"].note)
    assert p.estimators["dr"]["success_rate"].value == pytest.approx(1.2)
    assert p.verdict.status == NOT_SHOWN
    assert any("effective sample size" in r for r in p.verdict.reasons)


def test_support_violations_are_detected_and_rows_kept() -> None:
    z = Action("z", "low")
    ev = evaluate_logged(logged_from_pairs(HAND), [FixedTarget(z, "z")], LOOSE)
    p = ev.policies[0]
    assert p.diagnostics["unsupported_mass"] == 1.0
    assert p.diagnostics["rows_with_unsupported_mass"] == 4
    assert ev.provenance["decisions"] == 4
    assert p.verdict.status == NOT_SHOWN
    assert any("never took" in r for r in p.verdict.reasons)


def test_deterministic_logger_supports_only_its_own_action() -> None:
    pairs = [_pair(i, X, 1.0, True, 1.0) for i in range(12)]
    p = evaluate_logged(logged_from_pairs(pairs), [FixedTarget(Y, "y")], LOOSE).policies[0]
    assert p.diagnostics["unsupported_mass"] == 1.0


def test_unknown_support_without_a_candidate_set() -> None:
    cs = CandidateSet("v1:" + "0" * 32, 2, 1)
    pairs = []
    for i in range(12):
        d, o = _pair(i, X, 0.5, True, 1.0, cands=())
        d.candidate_set = cs
        pairs.append((d, o))
    data = logged_from_pairs(pairs)  # no catalog: the compact set cannot be rebuilt
    assert data.candidate_issues == {"no_catalog_to_rebuild": 12}
    p = evaluate_logged(data, [FixedTarget(Y, "y")], LOOSE).policies[0]
    assert p.diagnostics["unknown_support_mass"] == 1.0
    assert any("--catalog" in r for r in p.verdict.reasons)


def test_invalid_propensities_are_counted_and_block_the_verdict() -> None:
    pairs = [*HAND, _pair(9, X, 0.0, True, 1.0)]
    ev = evaluate_logged(logged_from_pairs(pairs), [FixedTarget(X, "x")], LOOSE)
    assert ev.provenance["invalid_propensity"] == 1
    assert any("invalid propensity" in r for r in ev.policies[0].verdict.reasons)


def test_clipping_reports_and_widens_by_its_bias_bound() -> None:
    st = EvalSettings(bootstrap=100, min_ess=0, min_clusters=1, clip=2.0)
    ev = evaluate_logged(
        logged_from_pairs(HAND), [FixedTarget(Y, "y")], st, reward_model=ConstModel(),
        frontier=False,
    )  # fmt: skip
    p = ev.policies[0]
    # The weight 4 is capped at 2: clipped mass 2 over 4 labelled rows, rewards in [0, 1].
    assert p.diagnostics["clip_bias_bound"]["success_rate"] == pytest.approx(0.5)
    assert p.estimators["ips"]["success_rate"].value == pytest.approx(4 / 4)
    ci = p.estimators["ips"]["success_rate"].ci
    assert ci is not None and ci.hi - ci.lo >= 1.0
    assert p.diagnostics["clipped_rows"] == 1


def test_unlabelled_rows_are_excluded_and_bounded() -> None:
    pairs = [*HAND[:3], _pair(4, Y, 0.25, None, 3.0)]
    ev = evaluate_logged(logged_from_pairs(pairs), [FixedTarget(Y, "y")], LOOSE)
    assert ev.provenance["unlabelled_steps"] == 1
    p = ev.policies[0]
    assert p.estimators["ips"]["success_rate"].n == 3
    assert set(p.diagnostics["unlabelled_sensitivity"]) == {
        "unlabelled_as_failure",
        "unlabelled_as_success",
    }


@dataclass
class SpyModel:
    """A fittable model that records what it was fitted on and refuses to predict a row it
    has seen: proves cross-fitting never evaluates on training data."""

    name: str = "spy"

    def fit(self, rows: Sequence[LoggedRow]) -> "SpyFitted":
        return SpyFitted({r.cluster for r in rows})


@dataclass
class SpyFitted:
    seen: set[str]
    name: str = "spy-fitted"

    def predict(self, state: StepState, action: Action) -> tuple[float, float]:
        cluster = f"task:{state.task_id}" if state.task_id else f"session:{state.session_id}"
        assert cluster not in self.seen, "predicted a row the model was fitted on"
        return 0.5, 1.0


def test_reward_model_is_cross_fitted_out_of_fold() -> None:
    w = default_world()
    data = logged_from_pairs(make_logged(w, 30, 1))
    ev = evaluate_logged(data, [FixedTarget(w.configs[2].action)], LOOSE, reward_model=SpyModel())
    assert "cross-fitted" in str(ev.provenance["reward_model"])


def _cfgs():  # type: ignore[no-untyped-def]
    return sorted(default_world().configs, key=lambda c: c.ability)


def test_dr_stays_unbiased_with_a_misspecified_reward_model() -> None:
    """Right propensities, absurd reward model: DR is still centred on the truth, while the
    model alone (the direct method) is far off."""

    @dataclass
    class Absurd:
        name: str = "absurd"

        def predict(self, state: StepState, action: Action) -> tuple[float, float]:
            return 0.99, 5.0

    w = default_world()
    cfgs = _cfgs()
    target = hint_policy("by-hint", lambda h: cfgs[min(h + 1, 5)].action)
    truth = true_logged_value(w, target)
    ests = []
    for rep in range(30):
        data = logged_from_pairs(make_logged(w, 60, 500 + rep))
        ev = evaluate_logged(data, [target], LOOSE, reward_model=Absurd(), frontier=False)
        ests.append(ev.policies[0].estimators["dr"]["success_rate"].value)
    mean = sum(ests) / len(ests)
    sd = math.sqrt(sum((e - mean) ** 2 for e in ests) / (len(ests) - 1))
    assert abs(mean - truth.success) < 3 * sd / math.sqrt(len(ests))
    assert abs(0.99 - truth.success) > 0.15  # the model alone would be badly wrong


def test_wrong_propensities_bias_the_estimate_silently() -> None:
    """Documents the failure mode: a collector that logs the wrong propensity (here, 1/K
    whatever the policy did) makes IPS wrong, and nothing in the logs can reveal it. Only
    supplying the logging policy exposes the mismatch."""
    w = default_world()
    cfgs = _cfgs()
    target = hint_policy("by-hint", lambda h: cfgs[min(h, 5)].action)
    truth = true_logged_value(w, target)
    uniform = lambda dist, a: 1.0 / len(dist)  # noqa: E731
    misses = 0
    ests = []
    for rep in range(20):
        data = logged_from_pairs(make_logged(w, 80, 900 + rep, logged_propensity=uniform))
        ev = evaluate_logged(data, [target], LOOSE, frontier=False)
        e = ev.policies[0].estimators["ips"]["success_rate"]
        assert e.ci is not None
        ests.append(e.value)
        misses += not (e.ci.lo <= truth.success <= e.ci.hi)
        assert ev.provenance["propensity_mismatches"] is None  # cannot be checked
    assert abs(sum(ests) / len(ests) - truth.success) > 0.15
    assert misses >= 15  # the intervals confidently exclude the truth
    data = logged_from_pairs(make_logged(w, 80, 900, logged_propensity=uniform))
    ev = evaluate_logged(data, [target], LOOSE, logging_policy=SynthLogging(w), frontier=False)
    assert ev.provenance["propensity_mismatches"] > 0
    assert any("disagrees with the logging policy" in r for r in ev.policies[0].verdict.reasons)


def test_determinism_under_a_seed() -> None:
    w = default_world()
    data = logged_from_pairs(make_logged(w, 40, 3))
    target = [FixedTarget(_cfgs()[3].action, "mid")]
    a = evaluate_logged(data, target, EvalSettings(bootstrap=100, seed=7)).to_json()
    b = evaluate_logged(data, target, EvalSettings(bootstrap=100, seed=7)).to_json()
    c = evaluate_logged(data, target, EvalSettings(bootstrap=100, seed=8)).to_json()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


class HintPredictor:
    """A stub predictor for ScoredPolicy on the synthetic catalog: reads the hint."""

    def __init__(self) -> None:
        self.world = default_world()

    def predict(self, state: StepState, step: Action) -> float:
        return self.world.p(int(state.features["hint"]), self.world.config(step))


def test_scored_policy_logs_round_trip_through_the_store(tmp_path: Path) -> None:
    """A real ScoredPolicy logs through TraceStore with compact candidate sets; the
    evaluator rebuilds them from the catalog, checks every propensity against the logging
    policy, and evaluates another ScoredPolicy as the target."""
    w = default_world()
    cat = catalog(w)
    spec = CandidateSpec(max_depth=1)
    log_policy = ScoredPolicy(
        HintPredictor(), cat, tau=0.7, epsilon=0.5, spec=spec, seed=1, inline_candidates_max=0
    )
    store = TraceStore(tmp_path, clock=lambda: 1_767_225_600.0)
    rng = random.Random(0)
    for i in range(60):
        h = i % w.levels
        state = StepState(
            "synthetic", f"s{i}", 0, "main", 1000, task_id=f"t{i}", features={"hint": h}
        )
        d = log_policy.decide(state)
        store.log_decision(d)
        ok = rng.random() < w.p(h, w.config(d.action))
        store.log_outcome(Outcome(d.decision_id, 2.0, ok, cost_usd=0.01))
    data = load_logged(tmp_path, cat)
    assert data.candidate_issues == {}
    assert all(r.candidates is not None and len(r.candidates) == 6 for r in data.rows)
    target = as_target(
        ScoredPolicy(HintPredictor(), cat, tau=0.9, epsilon=0.0, spec=spec), name="tau-0.9"
    )
    ev = evaluate_logged(data, [target], LOOSE, logging_policy=as_target(log_policy))
    assert ev.provenance["propensity_mismatches"] == 0
    assert ev.provenance["rows_with_unknown_support"] == 0
    assert ev.policies[0].diagnostics["unsupported_mass"] == 0.0
