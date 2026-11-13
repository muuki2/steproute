"""Policy: hand-worked example, propensities, tau monotonicity, determinism, privacy, report."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
from collections import Counter
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steproute.core import (
    Action,
    CandidateSetMismatch,
    CandidateSpec,
    Decision,
    FixedPolicy,
    ModelProfile,
    NoFeasibleAction,
    Outcome,
    Policy,
    ScoredPolicy,
    StepState,
    expected_cost,
    filter_feasible,
    rebuild_candidates,
)
from steproute.core.catalog import Catalog, load_default_catalog
from steproute.core.cost import CostModelError
from steproute.core.policy import EST_OUTPUT_FEATURE, _Scorer

DEFAULT = load_default_catalog(warn_unverified=False)

# --- the hand-worked example of docs/POLICY.md ----------------------------------------------
#
# Two models, no cache discount or write premium (both multipliers 1.0), so the cache owner
# does not change any price:
#   small: tier 1, efforts low and high, $1 in / $5 out, high doubles the output
#   big:   tier 2, effort medium,        $5 in / $25 out
# State: 10,000 prompt tokens, est_output_tokens = 1,000.
#   c(small low)  = 10,000 * 1/1e6 + 1,000 * 5/1e6  = 0.010 + 0.005 = 0.015
#   c(small high) = 10,000 * 1/1e6 + 2,000 * 5/1e6  = 0.010 + 0.010 = 0.020
#   c(big medium) = 10,000 * 5/1e6 + 1,000 * 25/1e6 = 0.050 + 0.025 = 0.075
# Stub success probabilities, conditional on the failed attempts before the step:
#   fresh:              small low 0.60, small high 0.75, big medium 0.90
#   after small low:    small high 0.50, big medium 0.80
#   after small high:   big medium 0.70
# Candidates (enumeration order), cost = c1 + (1 - p1) c2, success = 1 - (1 - p1)(1 - p2):
#   0 small low                  0.015                     0.60
#   1 small high                 0.020                     0.75
#   2 big medium                 0.075                     0.90
#   3 small low  -> small high   0.015 + 0.40*0.020 = 0.023    1 - 0.40*0.50 = 0.80
#   4 small low  -> big medium   0.015 + 0.40*0.075 = 0.045    1 - 0.40*0.20 = 0.92
#   5 small high -> big medium   0.020 + 0.25*0.075 = 0.03875  1 - 0.25*0.30 = 0.925

SMALL = ModelProfile(
    "small",
    "p",
    1.0,
    5.0,
    100_000,
    tier=1,
    efforts=("low", "high"),
    effort_output_mult={"low": 1.0, "high": 2.0},
)
BIG = ModelProfile("big", "p", 5.0, 25.0, 100_000, tier=2, efforts=("medium",))
EXAMPLE = Catalog((SMALL, BIG))
S_LOW, S_HIGH, B_MED = Action("small", "low"), Action("small", "high"), Action("big", "medium")

STUB_P = {
    ((), S_LOW): 0.60,
    ((), S_HIGH): 0.75,
    ((), B_MED): 0.90,
    ((S_LOW,), S_HIGH): 0.50,
    ((S_LOW,), B_MED): 0.80,
    ((S_HIGH,), B_MED): 0.70,
}

EXPECTED = [  # (action, cost, success), enumeration order
    (S_LOW, 0.015, 0.60),
    (S_HIGH, 0.020, 0.75),
    (B_MED, 0.075, 0.90),
    (Action("small", "low", S_HIGH), 0.023, 0.80),
    (Action("small", "low", B_MED), 0.045, 0.92),
    (Action("small", "high", B_MED), 0.03875, 0.925),
]


class Stub:
    """Probability from a table keyed by (failed attempts, step)."""

    def __init__(self, table: dict[tuple[tuple[Action, ...], Action], float]) -> None:
        self.table = table
        self.calls: list[tuple[tuple[Action, ...], Action]] = []

    def predict(self, state: StepState, action_step: Action) -> float:
        key = (tuple(state.failed_attempts), action_step)
        self.calls.append(key)
        return self.table[key]


class Constant:
    def __init__(self, p: float = 0.7) -> None:
        self.p = p

    def predict(self, state: StepState, action_step: Action) -> float:
        return self.p


class Learner(Constant):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[Decision, Outcome]] = []

    def update(self, decision: Decision, outcome: Outcome) -> None:
        self.seen.append((decision, outcome))


def example_state(**kw: Any) -> StepState:
    return StepState("test", "s", 0, "main", 10_000, **kw)


def ids() -> Any:
    counter = iter(range(10**9))
    return lambda: f"d{next(counter)}"


def example_policy(tau: float, epsilon: float = 0.0, **kw: Any) -> ScoredPolicy:
    kw.setdefault("est_output_tokens", 1000)
    kw.setdefault("seed", 7)
    kw.setdefault("clock", lambda: 1000.0)
    kw.setdefault("new_id", ids())
    return ScoredPolicy(Stub(STUB_P), EXAMPLE, tau, epsilon, **kw)


def test_example_scores_match_hand_arithmetic() -> None:
    scores = example_policy(0.9).score(example_state())
    assert [s.action for s in scores] == [a for a, _, _ in EXPECTED]
    for s, (_, cost, p) in zip(scores, EXPECTED, strict=True):
        assert s.cost_usd == pytest.approx(cost, abs=1e-15)
        assert s.p_success == pytest.approx(p, abs=1e-15)


@pytest.mark.parametrize(
    ("tau", "chosen"),
    [
        (0.91, Action("small", "high", B_MED)),  # a cascade wins: 0.03875 < 0.045 < 0.075
        (0.85, Action("small", "high", B_MED)),  # big medium (0.90) qualifies but costs more
        (0.78, Action("small", "low", S_HIGH)),  # the cheap cascade reaches 0.80
        (0.70, S_HIGH),  # a single step suffices
        (0.50, S_LOW),
        (0.95, Action("small", "high", B_MED)),  # nothing reaches tau: highest success, 0.925
        (1.00, Action("small", "high", B_MED)),
    ],
)
def test_example_choice(tau: float, chosen: Action) -> None:
    d = example_policy(tau).decide(example_state())
    assert d.action == chosen
    assert d.propensity == 1.0
    want = next(e for e in EXPECTED if e[0] == chosen)
    assert d.predicted_cost == pytest.approx(want[1], abs=1e-15)
    assert d.predicted_success == pytest.approx(want[2], abs=1e-15)
    assert d.candidates == [a for a, _, _ in EXPECTED]  # 6 <= inline_candidates_max


def test_example_propensities() -> None:
    # epsilon 0.1 over 6 feasible actions: greedy 0.9 + 0.1/6, every other 0.1/6.
    probs = dict(example_policy(0.91, 0.1).action_probabilities(example_state()))
    greedy = Action("small", "high", B_MED)
    assert probs[greedy] == pytest.approx(0.9 + 0.1 / 6)
    for a, p in probs.items():
        if a != greedy:
            assert p == pytest.approx(0.1 / 6)
    assert math.fsum(probs.values()) == pytest.approx(1.0, abs=1e-12)


def test_predictor_sees_post_failure_state() -> None:
    stub = Stub(STUB_P)
    ScoredPolicy(stub, EXAMPLE, 0.9, 0.0, est_output_tokens=1000, seed=1).decide(example_state())
    # once per distinct chain prefix (6 nodes), each with the earlier failures filled in
    assert sorted(stub.calls, key=repr) == sorted(STUB_P, key=repr)


# --- tie-break -------------------------------------------------------------------------------


def test_tie_break_equal_cost_prefers_higher_success_then_order() -> None:
    a = ModelProfile("a", "p", 1.0, 1.0, 100_000, tier=1)
    b = ModelProfile("b", "p", 1.0, 1.0, 100_000, tier=1)
    cat = Catalog((a, b))
    s = example_state()
    table = {((), Action("a")): 0.8, ((), Action("b")): 0.9}
    pol = ScoredPolicy(Stub(table), cat, 0.5, 0.0, spec=CandidateSpec(max_depth=1), seed=1)
    assert pol.decide(s).action == Action("b")  # equal cost, higher success
    table[((), Action("a"))] = 0.9
    pol = ScoredPolicy(Stub(table), cat, 0.5, 0.0, spec=CandidateSpec(max_depth=1), seed=1)
    assert pol.decide(s).action == Action("a")  # all equal: enumeration order
    # nothing reaches tau, equal success: lower cost wins, then order
    pol = ScoredPolicy(Stub(table), cat, 0.95, 0.0, spec=CandidateSpec(max_depth=1), seed=1)
    assert pol.decide(s).action == Action("a")


# --- fast scoring path equals expected_cost --------------------------------------------------


def _states() -> list[StepState]:
    base = dict(tools=["Bash"])
    return [
        StepState("t", "s", 0, "main", 20_000, **base),  # type: ignore[arg-type]
        StepState(
            "t",
            "s",
            1,
            "main",
            20_000,
            cached_prefix_tokens=18_000,
            current_model="claude-sonnet-5-5",
            current_effort="high",
        ),
        StepState(
            "t",
            "s",
            2,
            "main",
            300,  # below Anthropic's cache minimum
            cached_prefix_tokens=200,
            current_model="claude-haiku-5-5",
            current_effort="low",
        ),
        StepState(
            "t",
            "s",
            3,
            "main",
            150_000,  # a higher price tier; past the local models' context
            cached_prefix_tokens=120_000,
            current_model="claude-haiku-5-5",
            current_effort="medium",
        ),
        StepState(
            "t", "s", 4, "main", 5_000, cached_prefix_tokens=4_000, current_model="gpt-oss:20b"
        ),
    ]


class Hashy:
    """A deterministic predictor that depends on the step and on the failure history."""

    def predict(self, state: StepState, action_step: Action) -> float:
        key = f"{state.failed_attempts}|{action_step}|{state.current_model}"
        return (sum(map(ord, key)) % 97) / 96.0


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_fast_path_equals_expected_cost_for_every_candidate(depth: int) -> None:
    pred = Hashy()
    pol = ScoredPolicy(pred, DEFAULT, 0.9, 0.0, spec=CandidateSpec(max_depth=depth), seed=1)
    for s in _states():
        scores = pol.score(s)
        assert len(scores) > 0
        for sc in scores:
            ref = expected_cost(s, sc.action, DEFAULT, 1000.0, pred.predict)
            assert sc.cost_usd == ref.total_usd, sc.action.label()
            assert sc.p_success == ref.p_success, sc.action.label()


def test_feasibility_cache_equals_filter_feasible() -> None:
    scorer = _Scorer(DEFAULT, CandidateSpec(max_depth=2))
    for prompt in (0, 1_000, 32_768, 32_769, 200_000, 1_000_000, 1_000_001):
        for sensitive in (False, True):
            for tools in ([], ["Bash"]):
                s = StepState("t", "s", 0, "main", prompt, sensitive=sensitive, tools=tools)
                try:
                    want = filter_feasible(s, scorer.actions, DEFAULT)
                except NoFeasibleAction:
                    with pytest.raises(NoFeasibleAction):
                        scorer.feasible(s)
                    continue
                for _ in range(2):  # miss, then hit
                    assert tuple(scorer.actions[i] for i in scorer.feasible(s)) == want


# --- criterion 2: empirical frequencies match logged propensities ----------------------------

N_DECISIONS = 20_000


@pytest.mark.parametrize("epsilon", [0.3, 0.0, 1.0])
def test_empirical_frequencies_match_propensities(epsilon: float) -> None:
    pol = example_policy(0.91, epsilon, seed=12345)
    s = example_state()
    probs = dict(pol.action_probabilities(s))
    assert math.fsum(probs.values()) == pytest.approx(1.0, abs=1e-12)
    counts: Counter[Action] = Counter()
    for _ in range(N_DECISIONS):
        d = pol.decide(s)
        counts[d.action] += 1
        assert d.propensity == probs[d.action]  # the logged value is the true probability
    assert set(counts) <= set(probs)
    for a, p in probs.items():
        freq = counts[a] / N_DECISIONS
        # Tolerance: 5 binomial standard deviations (about 0.017 at p = 0.5).
        tol = 5.0 * math.sqrt(p * (1.0 - p) / N_DECISIONS) + 1e-12
        assert abs(freq - p) <= tol, (a.label(), freq, p)
    if epsilon == 0.0:
        assert counts == Counter({Action("small", "high", B_MED): N_DECISIONS})
    if epsilon == 1.0:
        assert all(p == pytest.approx(1 / 6) for p in probs.values())
    assert pol.stats().decisions == N_DECISIONS


# --- criterion 3: raising tau never lowers the chosen action's predicted success ----------


@st.composite
def catalogs(draw: st.DrawFn) -> Catalog:
    n = draw(st.integers(1, 4))
    efforts_pool = ["none", "low", "medium", "high", "max"]
    models = []
    for i in range(n):
        efforts = draw(st.lists(st.sampled_from(efforts_pool), min_size=1, max_size=3, unique=True))
        models.append(
            ModelProfile(
                f"m{i}",
                "p",
                draw(st.sampled_from([0.0, 0.5, 1.0, 3.0])),
                draw(st.sampled_from([0.0, 1.0, 5.0, 15.0])),
                100_000,
                tier=draw(st.integers(0, 3)),
                efforts=tuple(efforts),
                cache_read_mult=draw(st.sampled_from([0.1, 1.0])),
                cache_write_mult=draw(st.sampled_from([1.0, 1.25])),
            )
        )
    return Catalog(tuple(models))


class Table:
    """Random but fixed probabilities, keyed by the step and the failure history."""

    def __init__(self, seed: int, coarse: bool) -> None:
        self.seed = seed
        self.coarse = coarse  # coarse values create many exact ties

    def predict(self, state: StepState, action_step: Action) -> float:
        key = f"{self.seed}|{state.failed_attempts}|{action_step}".encode()
        h = int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")
        return (h % 5) / 4.0 if self.coarse else h / 0xFFFFFFFF


@settings(max_examples=200, deadline=None)
@given(
    catalogs(),
    st.integers(0, 2**32),
    st.booleans(),
    st.floats(0.0, 1.0),
    st.floats(0.0, 1.0),
    st.integers(1, 3),
    st.sampled_from([0, 600, 5_000]),
)
def test_raising_tau_never_lowers_predicted_success(
    cat: Catalog,
    seed: int,
    coarse: bool,
    t1: float,
    t2: float,
    depth: int,
    cached: int,
) -> None:
    lo, hi = sorted((t1, t2))
    pred = Table(seed, coarse)
    spec = CandidateSpec(max_depth=depth)
    s = StepState("t", "s", 0, "main", 5_000, cached_prefix_tokens=cached, current_model="m0")
    a = ScoredPolicy(pred, cat, lo, 0.0, spec=spec, seed=0).greedy(s)
    b = ScoredPolicy(pred, cat, hi, 0.0, spec=spec, seed=0).greedy(s)
    assert b.p_success >= a.p_success
    if a.p_success >= hi:  # the low-tau choice still qualifies, so nothing changes
        assert b.action == a.action


# --- determinism and threads ------------------------------------------------------------------


def _sequence(seed: int, n: int = 300) -> list[tuple[Action, float]]:
    pol = example_policy(0.91, 0.5, seed=seed)
    return [(d.action, d.propensity) for d in (pol.decide(example_state()) for _ in range(n))]


def test_same_seed_same_decisions_different_seed_differs() -> None:
    assert _sequence(1) == _sequence(1)
    assert _sequence(1) != _sequence(2)


def test_decisions_are_byte_identical_with_injected_clock_and_ids() -> None:
    a = [example_policy(0.91, 0.5, seed=3).decide(example_state()).to_dict() for _ in range(3)]
    b = [example_policy(0.91, 0.5, seed=3).decide(example_state()).to_dict() for _ in range(3)]
    assert json.dumps(a) == json.dumps(b)
    assert a[0]["ts"] == 1000.0 and a[0]["decision_id"] == "d0"


def test_threads_draw_each_k_once() -> None:
    # Concurrent decide calls: the multiset of outcomes equals the sequential one, because
    # decision k's draw depends only on (seed, k).
    seq = Counter(_sequence(9, 400))
    pol = example_policy(0.91, 0.5, seed=9)
    out: list[tuple[Action, float]] = []
    lock = threading.Lock()

    def work() -> None:
        for _ in range(100):
            d = pol.decide(example_state())
            with lock:
                out.append((d.action, d.propensity))

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert Counter(out) == seq


# --- privacy and hard constraints -------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("tau", [0.0, 0.9, 1.0])
def test_sensitive_state_never_gets_a_cloud_model(epsilon: float, tau: float) -> None:
    local = {m.id for m in DEFAULT.models if m.local}
    s = StepState("t", "s", 0, "main", 2_000, sensitive=True)
    pol = ScoredPolicy(Constant(0.4), DEFAULT, tau, epsilon, seed=5)
    for _ in range(300):
        d = pol.decide(s)
        assert {step.model for step in d.action.chain()} <= local
        for c in (*d.candidates, *rebuild_candidates(d, DEFAULT)):
            assert {x.model for x in c.chain()} <= local
    explored = pol.stats().explored
    if epsilon > 0:
        assert explored > 0  # the exploration branch really ran
    for a, _ in pol.action_probabilities(s):
        assert {step.model for step in a.chain()} <= local


def test_no_feasible_action_propagates() -> None:
    s = StepState("t", "s", 0, "main", 50_000, sensitive=True)  # past every local context cap
    with pytest.raises(NoFeasibleAction):
        ScoredPolicy(Constant(), DEFAULT, 0.9, 0.5, seed=1).decide(s)


def test_allow_deny_apply() -> None:
    spec = CandidateSpec(max_depth=2, deny=["claude-fable-5-2", "claude-opus-5-5"])
    pol = ScoredPolicy(Constant(0.1), DEFAULT, 0.99, 1.0, spec=spec, seed=2)
    for _ in range(200):
        models = {a.model for a in pol.decide(example_state()).action.chain()}
        assert not models & {"claude-fable-5-2", "claude-opus-5-5"}
    with pytest.raises(ValueError, match="not in the catalog"):
        ScoredPolicy(Constant(), DEFAULT, 0.9, 0.0, spec=CandidateSpec(allow=["nope"]))


def test_fixed_policy() -> None:
    a = Action("qwen3.5:9b", "medium")
    pol: Policy = FixedPolicy(a, DEFAULT, clock=lambda: 5.0, new_id=lambda: "x")
    d = pol.decide(StepState("t", "s", 0, "main", 100, sensitive=True))
    assert (d.action, d.propensity, d.candidates, d.ts, d.decision_id) == (a, 1.0, [a], 5.0, "x")
    pol.report(Outcome("x", 6.0, True))
    cloud = FixedPolicy(Action("claude-haiku-5-5", "low"), DEFAULT)
    with pytest.raises(NoFeasibleAction):
        cloud.decide(StepState("t", "s", 0, "main", 100, sensitive=True))
    with pytest.raises(ValueError):
        FixedPolicy(Action("nope"), DEFAULT)
    with pytest.raises(ValueError, match="never allowed"):
        FixedPolicy(a, DEFAULT, deny=["qwen3.5:9b"])


# --- compact candidate sets ------------------------------------------------------------------


def test_compact_decision_rebuilds_and_verifies() -> None:
    s = _states()[1]
    pol = ScoredPolicy(Constant(), DEFAULT, 0.9, 0.2, seed=4)
    d = pol.decide(s)
    assert d.candidates == []  # 555 > inline_candidates_max
    assert d.candidate_set is not None and d.candidate_set.count == 555
    # JSON roundtrip, then rebuild
    back = Decision.from_dict(json.loads(json.dumps(d.to_dict())))
    rebuilt = rebuild_candidates(back, DEFAULT)
    assert rebuilt == tuple(sc.action for sc in pol.score(s))
    assert d.action in rebuilt
    assert len(json.dumps(d.to_dict())) < 2_000
    # always inline when asked
    full = ScoredPolicy(Constant(), DEFAULT, 0.9, 0.2, seed=4, inline_candidates_max=None)
    assert tuple(full.decide(s).candidates) == rebuilt
    assert rebuild_candidates(full.decide(s), DEFAULT) == rebuilt


def test_rebuild_detects_mismatch() -> None:
    s = _states()[1]
    d = ScoredPolicy(Constant(), DEFAULT, 0.9, 0.0, seed=4).decide(s)
    # a catalog whose set-shaping fields differ
    other = Catalog(tuple(m for m in DEFAULT.models if m.id != "claude-opus-5-5"))
    with pytest.raises(CandidateSetMismatch, match="fingerprint"):
        rebuild_candidates(d, other)
    # a price change does not matter
    cheaper = Catalog(
        tuple(dataclasses.replace(m, price_in=m.price_in / 2) for m in DEFAULT.models)
    )
    assert len(rebuild_candidates(d, cheaper)) == 555
    # the logged state was changed after the decision (marked sensitive)
    changed = dataclasses.replace(d, state=dataclasses.replace(s, sensitive=True))
    with pytest.raises(CandidateSetMismatch):
        rebuild_candidates(changed, DEFAULT)
    # an explicit list that disagrees
    bad = dataclasses.replace(d, candidates=[Action("qwen3.5:9b", "medium")])
    with pytest.raises(CandidateSetMismatch, match="explicit"):
        rebuild_candidates(bad, DEFAULT)
    # no candidate_set: the explicit list is returned as is
    legacy = dataclasses.replace(d, candidate_set=None, candidates=[d.action])
    assert rebuild_candidates(legacy, DEFAULT) == (d.action,)


def test_fingerprint_stable_and_sensitive_to_parameters() -> None:
    fp = CandidateSpec().fingerprint(DEFAULT)
    assert fp == CandidateSpec(max_depth=2, deny=()).fingerprint(load_default_catalog())
    assert fp.startswith("v1:") and len(fp) == 35
    assert CandidateSpec(max_depth=1).fingerprint(DEFAULT) != fp
    assert CandidateSpec(deny=["claude-opus-5-5"]).fingerprint(DEFAULT) != fp
    assert CandidateSpec(include_holdout=True).fingerprint(DEFAULT) != fp
    assert CandidateSpec(deny=["b", "a", "a"]).deny == ("a", "b")


# --- output-token estimate -------------------------------------------------------------------


def test_output_estimate_sources() -> None:
    base = example_policy(0.9).score(example_state())[0].cost_usd  # 1000 tokens: 0.015
    assert base == pytest.approx(0.015)
    via_feature = example_policy(0.9).score(example_state(features={EST_OUTPUT_FEATURE: 2000}))
    assert via_feature[0].cost_usd == pytest.approx(0.020)
    via_fn = example_policy(0.9, est_output_tokens=lambda s: 0.0).score(example_state())
    assert via_fn[0].cost_usd == pytest.approx(0.010)
    with pytest.raises(CostModelError):
        example_policy(0.9).score(example_state(features={EST_OUTPUT_FEATURE: "many"}))


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        example_policy(1.5)
    with pytest.raises(ValueError):
        example_policy(0.5, -0.1)
    with pytest.raises(TypeError):
        ScoredPolicy(object(), EXAMPLE, 0.5, 0.0)  # type: ignore[arg-type]
    with pytest.raises(CostModelError, match=r"\[0, 1\]"):
        ScoredPolicy(Constant(1.5), EXAMPLE, 0.5, 0.0).decide(example_state())
    with pytest.raises(CostModelError):
        ScoredPolicy(Constant(math.nan), EXAMPLE, 0.5, 0.0).decide(example_state())


# --- report and update -----------------------------------------------------------------------


def test_report_without_update_is_a_noop() -> None:
    pol = example_policy(0.9)
    d = pol.decide(example_state())
    pol.report(Outcome(d.decision_id, 1001.0, True))
    pol.report(Outcome("unknown", 1001.0, True))
    assert pol.stats().pending == 0 and pol.stats().updates == 0


def test_report_forwards_to_update_once() -> None:
    now = [1000.0]
    learner = Learner()
    pol = ScoredPolicy(learner, EXAMPLE, 0.9, 0.0, seed=1, clock=lambda: now[0], new_id=ids())
    d = pol.decide(example_state())
    assert pol.stats().pending == 1
    pol.report(Outcome(d.decision_id, 1001.0, None))  # unlabelled: ignored, still pending
    assert learner.seen == [] and pol.stats().pending == 1
    out = Outcome(d.decision_id, 1002.0, True)
    pol.report(out)
    assert learner.seen == [(d, out)]
    pol.report(out)  # repeated: unknown now
    pol.report(Outcome("never-seen", 1003.0, False))
    st_ = pol.stats()
    assert (st_.updates, st_.unknown_reports, st_.unlabelled_reports, st_.pending) == (1, 2, 1, 0)


def test_report_expired_and_bounded_memory() -> None:
    now = [0.0]
    learner = Learner()
    pol = ScoredPolicy(
        learner,
        EXAMPLE,
        0.9,
        0.0,
        seed=1,
        clock=lambda: now[0],
        new_id=ids(),
        max_pending=3,
        pending_ttl_s=100.0,
    )
    old = pol.decide(example_state())
    now[0] = 150.0
    pol.report(Outcome(old.decision_id, 150.0, True))
    assert pol.stats().expired_reports == 1 and learner.seen == []
    ds = [pol.decide(example_state()) for _ in range(5)]
    assert pol.stats().pending == 3 and pol.stats().evicted == 2
    pol.report(Outcome(ds[0].decision_id, 151.0, True))  # evicted: unknown
    pol.report(Outcome(ds[4].decision_id, 151.0, True))
    assert [d.decision_id for d, _ in learner.seen] == [ds[4].decision_id]
    now[0] = 400.0
    pol.decide(example_state())  # inserting prunes everything older than the ttl
    assert pol.stats().pending == 1


def test_scrubbed_compact_decision_still_rebuilds() -> None:
    from steproute.core.privacy import ScrubOptions, scrub_decision

    s = dataclasses.replace(_states()[1], task_text="secret prompt", tools=["Bash"])
    spec = CandidateSpec(deny=["claude-fable-5-2"])
    d = ScoredPolicy(Constant(), DEFAULT, 0.9, 0.1, spec=spec, seed=8).decide(s)
    scrubbed = scrub_decision(d, ScrubOptions(salt="test-salt"))
    assert scrubbed.candidate_set == d.candidate_set
    assert rebuild_candidates(scrubbed, DEFAULT) == rebuild_candidates(d, DEFAULT)


# --- sensitive detection before routing -------------------------------------------------------


def _fake_secret() -> str:
    # Built from pieces at runtime so no credential-shaped literal is in the source.
    return "".join(["g", "h", "p", "_"]) + "Zq7" * 12


def _secret_state(**kw: Any) -> StepState:
    text = "please fix the deploy script, the token is " + _fake_secret() + " thanks"
    return StepState("proxy", "s", 0, "main", 2_000, task_text=text, **kw)


LOCAL = {m.id for m in DEFAULT.models if m.local}


def _local_only(a: Action) -> bool:
    return {x.model for x in a.chain()} <= LOCAL


def test_decide_marks_secret_state_and_round_trips() -> None:
    pol = ScoredPolicy(Constant(0.4), DEFAULT, 0.9, 0.5, seed=21)
    # Same feasibility-cache shape, not sensitive: fills the cache with cloud candidates first.
    clean = pol.decide(StepState("proxy", "s", 0, "main", 2_000, task_text="fix the script"))
    assert not clean.state.sensitive
    assert not all(_local_only(a) for a, _ in pol.action_probabilities(clean.state))
    raw = _secret_state()
    assert not raw.sensitive
    for _ in range(50):  # both branches run at epsilon 0.5
        d = pol.decide(raw)
        assert d.state.sensitive and not raw.sensitive  # a marked copy; the input is untouched
        assert _local_only(d.action)
        rebuilt = rebuild_candidates(d, DEFAULT)
        assert all(_local_only(a) for a in rebuilt)
        probs = dict(pol.action_probabilities(d.state))
        assert probs[d.action] == d.propensity
        assert tuple(probs) == rebuilt
        # the unmarked input and the logged state give the same answers everywhere
        assert pol.action_probabilities(raw) == pol.action_probabilities(d.state)
        assert pol.score(raw) == pol.score(d.state)
        assert pol.greedy(raw) == pol.greedy(d.state)
    assert pol.stats().explored > 0


def test_paths_feature_drives_detection() -> None:
    pol = ScoredPolicy(Constant(0.4), DEFAULT, 0.9, 0.0, seed=1)
    s = StepState("proxy", "s", 0, "main", 2_000, features={"paths": ["app/config/.env"]})
    d = pol.decide(s)
    assert d.state.sensitive and _local_only(d.action)
    for bad in ("app/config/.env", ["app/config/.env", 3], {"p": 1}):  # wrong types: ignored
        s = StepState("proxy", "s", 0, "main", 2_000, features={"paths": bad})
        assert not pol.decide(s).state.sensitive


def test_privacy_config_is_used_and_flag_never_cleared() -> None:
    from steproute.core.privacy import PrivacyConfig

    def decide(privacy: PrivacyConfig, s: StepState) -> Decision:
        return ScoredPolicy(Constant(), DEFAULT, 0.9, 0.0, seed=1, privacy=privacy).decide(s)

    lax = PrivacyConfig(secret_patterns=(), sensitive_globs=())
    assert not decide(lax, _secret_state()).state.sensitive
    strict = PrivacyConfig(callback=lambda text, paths: "deploy" in (text or ""))
    s = StepState("proxy", "s", 0, "main", 2_000, task_text="deploy now")
    assert decide(strict, s).state.sensitive
    already = StepState("proxy", "s", 0, "main", 2_000, sensitive=True)
    assert decide(lax, already).state.sensitive


def test_fixed_policy_marks_before_the_filter() -> None:
    cloud = FixedPolicy(Action("claude-haiku-5-5", "low"), DEFAULT)
    with pytest.raises(NoFeasibleAction):
        cloud.decide(_secret_state())
    local = FixedPolicy(Action("qwen3.5:9b", "medium"), DEFAULT)
    d = local.decide(_secret_state())
    assert d.state.sensitive and d.propensity == 1.0


def test_store_second_line_uses_paths_too() -> None:
    from steproute.core.privacy import PrivacyConfig, redact_state

    s = StepState("proxy", "s", 0, "main", 10, features={"paths": ["home/u/.ssh/id_ed25519"]})
    assert redact_state(s, PrivacyConfig()).sensitive
