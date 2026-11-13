"""Policy interface and the constrained-argmin policy. Standard library only.

See ``docs/POLICY.md``. The decision rule is ``argmin expected_cost subject to
P(success) >= tau`` over the feasible actions, with a pluggable ``Predictor`` supplying the
per-step success probability and epsilon-uniform exploration on top. Every ``Decision`` carries
the true probability the policy gave the chosen action, because off-policy evaluation divides
by it.

The policy never writes traces: it returns a ``Decision`` and the caller logs it.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from steproute.core.actions import NoFeasibleAction, enumerate_actions, filter_feasible
from steproute.core.catalog import Catalog
from steproute.core.cost import (
    CostModelError,
    owns_cache,
    planned_usage,
    price_usage,
    state_after_failure,
)
from steproute.core.privacy import PrivacyConfig, mark_before_routing
from steproute.core.schema import (
    Action,
    CandidateSet,
    Decision,
    ModelProfile,
    Outcome,
    StepState,
    effort_rank,
    new_decision_id,
)

__all__ = [
    "DEFAULT_EST_OUTPUT_TOKENS",
    "ENUMERATION_VERSION",
    "EST_OUTPUT_FEATURE",
    "CandidateSetMismatch",
    "CandidateSpec",
    "FixedPolicy",
    "OutputEstimator",
    "Policy",
    "PolicyStats",
    "Predictor",
    "ScoredAction",
    "ScoredPolicy",
    "UpdatablePredictor",
    "rebuild_candidates",
]

#: Output-token estimate used when nothing better is supplied. A PLACEHOLDER, not a
#: measurement: replace it with an estimator fitted on the pilot replay (issue 18).
DEFAULT_EST_OUTPUT_TOKENS = 1000.0

#: ``StepState.features`` key that overrides the output-token estimate for one state.
EST_OUTPUT_FEATURE = "est_output_tokens"

#: Version of the candidate-set definition hashed into ``CandidateSet.fingerprint``. Bump it
#: whenever ``enumerate_actions`` or ``filter_feasible`` change which actions they return or in
#: what order, so old fingerprints stop verifying instead of silently matching a new list.
ENUMERATION_VERSION = 1

#: Maps a state to an output-token estimate (before the effort multiplier), for every step.
OutputEstimator = Callable[[StepState], float]


# --- protocols ------------------------------------------------------------------------------


class Predictor(Protocol):
    """Success probability of one ``(model, effort)`` step.

    ``predict(state, action_step)`` is called with the state the step actually runs in: for a
    cascade's later step that is the state the failed earlier attempts leave behind
    (``state.failed_attempts`` lists them), so the value is the probability of success *given*
    those failures. ``action_step`` has ``escalate_to=None``. Must return a number in [0, 1]
    and should be deterministic for a given state (see "Limitations" in ``docs/POLICY.md``).
    """

    def predict(self, state: StepState, action_step: Action) -> float: ...


@runtime_checkable
class UpdatablePredictor(Predictor, Protocol):
    """A predictor that learns from outcomes. ``update`` is optional for a ``Predictor``."""

    def update(self, decision: Decision, outcome: Outcome) -> None: ...


class Policy(Protocol):
    """What every collector and the service call.

    Implementations must route ``mark_before_routing(state, privacy)`` rather than the raw
    state, before the hard filter, and log that marked state in ``Decision.state``.
    """

    def decide(self, state: StepState) -> Decision: ...

    def report(self, outcome: Outcome) -> None: ...


# --- candidate sets -------------------------------------------------------------------------


class CandidateSetMismatch(ValueError):
    """A decision's candidate set cannot be rebuilt from the state and catalog given."""


def _names(ids: Iterable[str] | None) -> tuple[str, ...] | None:
    if ids is None:
        return None
    if isinstance(ids, str):
        raise TypeError("allow/deny must be an iterable of model ids, not a string")
    return tuple(sorted(set(ids)))


def _as_catalog(profiles: Catalog | Iterable[ModelProfile]) -> Catalog:
    return profiles if isinstance(profiles, Catalog) else Catalog(tuple(profiles))


@dataclass(frozen=True)
class CandidateSpec:
    """How a policy builds its candidate set: enumeration depth plus the hard-constraint
    parameters of ``filter_feasible``. ``allow`` and ``deny`` are stored sorted and unique."""

    max_depth: int = 2
    allow: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()
    include_holdout: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, int)
            or self.max_depth < 1
        ):
            raise ValueError(f"max_depth must be an integer >= 1, got {self.max_depth!r}")
        object.__setattr__(self, "allow", _names(self.allow))
        object.__setattr__(self, "deny", _names(self.deny) or ())

    def enumerate(self, catalog: Catalog) -> tuple[Action, ...]:
        return enumerate_actions(catalog, self.max_depth, include_holdout=self.include_holdout)

    def feasible(
        self, state: StepState, catalog: Catalog, actions: Sequence[Action] | None = None
    ) -> tuple[Action, ...]:
        """The candidate set for ``state``. Raises ``NoFeasibleAction`` if it is empty."""
        return filter_feasible(
            state,
            self.enumerate(catalog) if actions is None else actions,
            catalog,
            allow=self.allow,
            deny=self.deny,
            include_holdout=self.include_holdout,
        )

    def fingerprint(self, catalog: Catalog) -> str:
        """Stable hash of the candidate-set definition: ``v1:`` plus 32 hex characters.

        Covers exactly what decides which actions are candidates and in what order: per model
        (file order) its id, tier, efforts, ``local``, ``context_window``, ``tools`` and
        ``holdout``; the parameters of this spec; and ``ENUMERATION_VERSION``. Prices and
        ``meta`` are not covered, so a price update does not invalidate old decisions.
        """
        models = [
            [
                m.id,
                m.tier,
                sorted(m.efforts, key=effort_rank),
                m.local,
                m.context_window,
                m.tools,
                m.holdout,
            ]
            for m in catalog.models
        ]
        payload = {
            "enumeration": ENUMERATION_VERSION,
            "models": models,
            "max_depth": self.max_depth,
            "allow": None if self.allow is None else list(self.allow),
            "deny": list(self.deny),
            "include_holdout": self.include_holdout,
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return "v1:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def record(self, fingerprint: str, count: int) -> CandidateSet:
        return CandidateSet(
            fingerprint=fingerprint,
            count=count,
            max_depth=self.max_depth,
            include_holdout=self.include_holdout,
            allow=self.allow,
            deny=self.deny,
        )

    @classmethod
    def from_record(cls, cs: CandidateSet) -> CandidateSpec:
        return cls(
            max_depth=cs.max_depth, allow=cs.allow, deny=cs.deny, include_holdout=cs.include_holdout
        )


def rebuild_candidates(
    decision: Decision, catalog: Catalog | Iterable[ModelProfile]
) -> tuple[Action, ...]:
    """The candidate list a decision chose among, rebuilt and verified.

    Without ``decision.candidate_set`` the explicit ``decision.candidates`` are returned as
    they are. Otherwise the set is rebuilt from ``decision.state`` and ``catalog`` and
    checked against the fingerprint, the count and (when present) the explicit list. Raises
    ``CandidateSetMismatch`` if any check fails, for example because ``catalog`` is not the one
    the decision was made with or the logged state differs from the state the policy saw.

    The logged state is used as it is, without running sensitive detection again: policies
    log the state they routed, which is already marked (``mark_before_routing``), and
    detection could only set a flag that is already set.
    """
    cs = decision.candidate_set
    if cs is None:
        return tuple(decision.candidates)
    cat = _as_catalog(catalog)
    spec = CandidateSpec.from_record(cs)
    got = spec.fingerprint(cat)
    if got != cs.fingerprint:
        raise CandidateSetMismatch(
            f"candidate-set fingerprint {cs.fingerprint} does not match this catalog ({got})"
        )
    try:
        actions = spec.feasible(decision.state, cat)
    except (NoFeasibleAction, ValueError) as exc:
        raise CandidateSetMismatch(f"cannot rebuild the candidate set: {exc}") from exc
    if len(actions) != cs.count:
        raise CandidateSetMismatch(
            f"rebuilt {len(actions)} candidates, the decision recorded {cs.count}; "
            "was the logged state changed after the decision (e.g. marked sensitive)?"
        )
    if decision.candidates and tuple(decision.candidates) != actions:
        raise CandidateSetMismatch("the explicit candidate list differs from the rebuilt one")
    return actions


# --- scoring --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredAction:
    """One feasible action with its expected cost (USD) and predicted overall success."""

    action: Action
    cost_usd: float
    p_success: float


def _check_p(value: object, step: Action) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CostModelError(f"predictor returned {value!r} for {step.label()}; expected a number")
    p = float(value)
    if not 0.0 <= p <= 1.0:  # also false for NaN
        raise CostModelError(f"predictor returned {value!r} for {step.label()}; expected [0, 1]")
    return p


class _Scorer:
    """Scores a catalog's candidates, sharing work between cascades.

    Built once per (catalog, spec). The candidate list is enumerated once and turned into a
    prefix tree: every distinct chain prefix is one node, so per decision the predictor is
    called once per prefix, the post-failure state is built once per prefix, and each step is
    priced once per distinct cache situation. ``tests/test_policy.py`` checks that the result
    equals ``expected_cost`` exactly for every candidate.
    """

    def __init__(self, catalog: Catalog, spec: CandidateSpec) -> None:
        self.catalog = catalog
        self.spec = spec
        self.actions = spec.enumerate(catalog)
        self.index = {a: i for i, a in enumerate(self.actions)}
        by_id = {m.id: m for m in catalog.models}
        node_of: dict[tuple[Action, ...], int] = {}
        self.parent: list[int] = []
        self.step: list[Action] = []
        self.profile: list[ModelProfile] = []
        self.paths: list[tuple[int, ...]] = []
        for action in self.actions:
            chain = action.chain()
            path: list[int] = []
            for d in range(len(chain)):
                key = chain[: d + 1]
                node = node_of.get(key)
                if node is None:
                    node = len(self.step)
                    node_of[key] = node
                    self.parent.append(path[-1] if path else -1)
                    self.step.append(chain[d])
                    self.profile.append(by_id[chain[d].model])
                path.append(node)
            self.paths.append(tuple(path))
        # filter_feasible reads the state only through these three things, so its result can
        # be cached under them. The number of distinct keys is tiny (bounded by
        # 4 * (number of distinct context windows + 1)).
        self._windows = tuple(sorted({m.context_window for m in catalog.models}))
        self._feasible: dict[tuple[bool, bool, tuple[bool, ...]], tuple[int, ...]] = {}

    def feasible(self, state: StepState) -> tuple[int, ...]:
        """Indices into ``self.actions`` of the feasible candidates, via ``filter_feasible``."""
        key = (
            bool(state.sensitive),
            bool(state.tools),
            tuple(state.prompt_tokens > w for w in self._windows),
        )
        hit = self._feasible.get(key)
        if hit is None:
            kept = self.spec.feasible(state, self.catalog, self.actions)  # may raise
            hit = tuple(self.index[a] for a in kept)
            self._feasible[key] = hit
        return hit

    def score(
        self,
        state: StepState,
        est_output_tokens: float,
        predict: Callable[[StepState, Action], float],
    ) -> tuple[tuple[int, ...], list[float], list[float]]:
        """``(feasible indices, expected cost per feasible action, overall success per one)``."""
        feasible = self.feasible(state)
        n_nodes = len(self.step)
        p_node = [math.nan] * n_nodes
        c_node = [math.nan] * n_nodes
        state_in: list[StepState | None] = [None] * n_nodes
        state_after: dict[int, StepState] = {}
        # The price of a step depends on the state only through whether the step owns the
        # cache and, if so, how much of the prompt is cached (prompt length and the output
        # estimate are fixed within a decision).
        prices: dict[tuple[Action, bool, int], float] = {}
        costs: list[float] = []
        succ: list[float] = []
        for i in feasible:
            path = self.paths[i]
            for node in path:
                if state_in[node] is not None:
                    continue
                par = self.parent[node]
                if par < 0:
                    s = state
                else:
                    after = state_after.get(par)
                    if after is None:
                        prev = state_in[par]
                        assert prev is not None
                        after = state_after_failure(prev, self.step[par], self.profile[par])
                        state_after[par] = after
                    s = after
                state_in[node] = s
                step = self.step[node]
                profile = self.profile[node]
                p_node[node] = _check_p(predict(s, step), step)
                warm = owns_cache(s, step, profile)
                pkey = (step, warm, s.cached_prefix_tokens if warm else -1)
                price = prices.get(pkey)
                if price is None:
                    usage = planned_usage(s, step, profile, est_output_tokens)
                    price = price_usage(profile, usage).total_usd
                    prices[pkey] = price
                c_node[node] = price
            # Same fold, in the same order, as expected_cost, so the floats agree exactly.
            rest = 0.0
            for node in reversed(path[1:]):
                rest = c_node[node] + (1.0 - p_node[node]) * rest
            first = path[0]
            costs.append(c_node[first] + (1.0 - p_node[first]) * rest)
            survive = 1.0
            for node in path:
                survive *= 1.0 - p_node[node]
            succ.append(1.0 - survive)
        return feasible, costs, succ


def _greedy(costs: Sequence[float], succ: Sequence[float], tau: float) -> int:
    """Position of the greedy choice.

    Among positions with ``succ >= tau``, the lowest cost; ties go to the higher success, then
    the earlier position (enumeration order). If none reaches ``tau``: the highest success;
    ties go to the lower cost, then the earlier position. The order is the same whatever
    ``tau`` is, which is what makes raising ``tau`` monotone.
    """
    best = -1
    for j in range(len(costs)):
        if succ[j] < tau:
            continue
        if best < 0 or costs[j] < costs[best] or (costs[j] == costs[best] and succ[j] > succ[best]):
            best = j
    if best >= 0:
        return best
    best = 0
    for j in range(1, len(costs)):
        if succ[j] > succ[best] or (succ[j] == succ[best] and costs[j] < costs[best]):
            best = j
    return best


# --- randomness -----------------------------------------------------------------------------


def _draws(seed: int, k: int) -> tuple[float, int]:
    """Two independent uniform draws for decision ``k``: a float in [0, 1) and a 64-bit int."""
    digest = hashlib.blake2b(f"{seed}:{k}".encode("ascii"), digest_size=16).digest()
    a = int.from_bytes(digest[:8], "big")
    b = int.from_bytes(digest[8:], "big")
    return (a >> 11) * (1.0 / 9007199254740992.0), b


def _check_unit(value: float, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number in [0, 1], got {value!r}")
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{what} must be in [0, 1], got {value!r}")
    return v


def _output_tokens(state: StepState, estimator: OutputEstimator) -> float:
    override = state.features.get(EST_OUTPUT_FEATURE)
    value = estimator(state) if override is None else override
    where = f"features[{EST_OUTPUT_FEATURE!r}]" if override is not None else "output estimator"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CostModelError(f"{where} must be a number, got {value!r}")
    v = float(value)
    if not (v >= 0.0 and math.isfinite(v)):
        raise CostModelError(f"{where} must be finite and >= 0, got {value!r}")
    return v


# --- policies -------------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyStats:
    """Counters of a ``ScoredPolicy``, for monitoring. A snapshot."""

    decisions: int
    explored: int
    pending: int
    updates: int
    unknown_reports: int
    expired_reports: int
    unlabelled_reports: int
    evicted: int


class ScoredPolicy:
    """Constrained argmin over the feasible actions, with epsilon-uniform exploration.

    Greedy rule: among feasible actions whose predicted overall success is at least ``tau``,
    the one with the lowest expected cost; if none reaches ``tau``, the one with the highest
    predicted success (tie-breaks in ``docs/POLICY.md``). With probability ``epsilon`` the
    action is instead drawn uniformly from the feasible set. The logged propensity is
    ``(1 - epsilon) * [a is greedy] + epsilon / n_feasible``.

    Determinism: decision ``k`` (counted from 0 in the order ``decide`` is entered) draws its
    randomness from ``(seed, k)`` only. Single-threaded, the same seed and the same sequence of
    states give the same decisions. Concurrently, each decision is still a pure function of its
    state and its ``k``; only which state gets which ``k`` depends on thread scheduling.

    ``report(outcome)`` forwards the outcome to ``predictor.update(decision, outcome)`` when
    the predictor has an ``update`` method, using a bounded map of recent decisions; see
    ``docs/POLICY.md``.
    """

    def __init__(
        self,
        predictor: Predictor,
        catalog: Catalog | Iterable[ModelProfile],
        tau: float,
        epsilon: float,
        *,
        spec: CandidateSpec | None = None,
        seed: int | None = None,
        est_output_tokens: float | OutputEstimator = DEFAULT_EST_OUTPUT_TOKENS,
        inline_candidates_max: int | None = 16,
        max_pending: int = 10_000,
        pending_ttl_s: float = 3600.0,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = new_decision_id,
        name: str = "scored",
        privacy: PrivacyConfig | None = None,
    ) -> None:
        if not callable(getattr(predictor, "predict", None)):
            raise TypeError("predictor must have a predict(state, action_step) method")
        self.predictor = predictor
        self.catalog = _as_catalog(catalog)
        self.tau = _check_unit(tau, "tau")
        self.epsilon = _check_unit(epsilon, "epsilon")
        self.spec = spec or CandidateSpec()
        known = {m.id for m in self.catalog.models}
        for label, ids in (("allow", self.spec.allow or ()), ("deny", self.spec.deny)):
            unknown = sorted(set(ids) - known)
            if unknown:
                raise ValueError(f"{label} list names models not in the catalog: {unknown}")
        self.seed = int.from_bytes(os.urandom(8), "big") if seed is None else int(seed)
        if callable(est_output_tokens):
            self._estimator: OutputEstimator = est_output_tokens
        else:
            constant = float(est_output_tokens)
            self._estimator = lambda _state: constant
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self.inline_candidates_max = inline_candidates_max
        self.max_pending = max_pending
        self.pending_ttl_s = float(pending_ttl_s)
        self.name = name
        self.privacy = privacy or PrivacyConfig()
        self._clock = clock
        self._new_id = new_id
        self._scorer = _Scorer(self.catalog, self.spec)
        self.fingerprint = self.spec.fingerprint(self.catalog)
        self._predict = predictor.predict
        update = getattr(predictor, "update", None)
        self._update: Callable[[Decision, Outcome], None] | None = (
            update if callable(update) else None
        )
        self._lock = threading.Lock()
        self._counter = itertools.count()
        self._pending: OrderedDict[str, Decision] = OrderedDict()
        self._n = {
            "explored": 0,
            "updates": 0,
            "unknown": 0,
            "expired": 0,
            "unlabelled": 0,
            "evicted": 0,
        }
        self._decisions = 0

    @property
    def actions(self) -> tuple[Action, ...]:
        """Every enumerated candidate, before the hard filter, in enumeration order."""
        return self._scorer.actions

    # --- scoring ---

    def _score(
        self, state: StepState
    ) -> tuple[StepState, tuple[int, ...], list[float], list[float]]:
        # Sensitive detection runs first, so the hard filter, the feasibility cache key, the
        # predictor and the logged Decision all see the marked state.
        marked = mark_before_routing(state, self.privacy)
        est = _output_tokens(marked, self._estimator)
        return (marked, *self._scorer.score(marked, est, self._predict))

    def score(self, state: StepState) -> tuple[ScoredAction, ...]:
        """Every feasible action with its expected cost and success, in enumeration order."""
        _, feasible, costs, succ = self._score(state)
        acts = self._scorer.actions
        return tuple(ScoredAction(acts[i], costs[j], succ[j]) for j, i in enumerate(feasible))

    def greedy(self, state: StepState) -> ScoredAction:
        """The action ``decide`` takes when it does not explore."""
        _, feasible, costs, succ = self._score(state)
        g = _greedy(costs, succ, self.tau)
        return ScoredAction(self._scorer.actions[feasible[g]], costs[g], succ[g])

    def action_probabilities(self, state: StepState) -> tuple[tuple[Action, float], ...]:
        """The policy's full distribution over the feasible actions (sums to 1).

        This is ``pi(a | state)`` for off-policy evaluation of this policy as a target.
        """
        _, feasible, costs, succ = self._score(state)
        g = _greedy(costs, succ, self.tau)
        n = len(feasible)
        acts = self._scorer.actions
        return tuple((acts[i], self._propensity(j == g, n)) for j, i in enumerate(feasible))

    def _propensity(self, is_greedy: bool, n: int) -> float:
        explore = self.epsilon / n
        return (1.0 - self.epsilon) + explore if is_greedy else explore

    # --- Policy protocol ---

    def decide(self, state: StepState) -> Decision:
        """Choose an action for ``state``. Raises ``NoFeasibleAction`` if none is allowed.

        ``Decision.state`` is ``state`` after sensitive detection (``mark_before_routing``).
        """
        marked, feasible, costs, succ = self._score(state)
        n = len(feasible)
        g = _greedy(costs, succ, self.tau)
        with self._lock:
            k = next(self._counter)
        u, bits = _draws(self.seed, k)
        explored = u < self.epsilon
        j = (bits * n) >> 64 if explored else g
        acts = self._scorer.actions
        inline = self.inline_candidates_max is None or n <= self.inline_candidates_max
        decision = Decision(
            decision_id=self._new_id(),
            ts=self._clock(),
            state=marked,
            candidates=[acts[i] for i in feasible] if inline else [],
            action=acts[feasible[j]],
            policy=self.name,
            propensity=self._propensity(j == g, n),
            predicted_success=succ[j],
            predicted_cost=costs[j],
            candidate_set=self.spec.record(self.fingerprint, n),
        )
        with self._lock:
            self._decisions += 1
            if explored:
                self._n["explored"] += 1
            if self._update is not None:
                self._remember(decision)
        return decision

    def _remember(self, decision: Decision) -> None:
        # Called with the lock held. Oldest first, so expiry and eviction pop from the left.
        pending = self._pending
        cutoff = decision.ts - self.pending_ttl_s
        while pending:
            oldest = next(iter(pending.values()))
            if oldest.ts >= cutoff:
                break
            pending.popitem(last=False)
            self._n["evicted"] += 1
        pending[decision.decision_id] = decision
        while len(pending) > self.max_pending:
            pending.popitem(last=False)
            self._n["evicted"] += 1

    def report(self, outcome: Outcome) -> None:
        """Pass ``outcome`` to the predictor's ``update``, if it has one.

        Never raises for an unknown, expired or repeated decision id; those are counted in
        ``stats()`` and ignored. An outcome with ``success=None`` is counted and ignored, and
        its decision stays pending so a later labelled outcome can still be used. The first
        labelled outcome consumes the decision. Exceptions from ``update`` propagate.
        """
        if self._update is None:
            return
        with self._lock:
            decision = self._pending.get(outcome.decision_id)
            if decision is None:
                self._n["unknown"] += 1
                return
            if self._clock() - decision.ts > self.pending_ttl_s:
                del self._pending[outcome.decision_id]
                self._n["expired"] += 1
                return
            if outcome.success is None:
                self._n["unlabelled"] += 1
                return
            del self._pending[outcome.decision_id]
            self._n["updates"] += 1
        self._update(decision, outcome)

    def stats(self) -> PolicyStats:
        with self._lock:
            n = self._n
            return PolicyStats(
                decisions=self._decisions,
                explored=n["explored"],
                pending=len(self._pending),
                updates=n["updates"],
                unknown_reports=n["unknown"],
                expired_reports=n["expired"],
                unlabelled_reports=n["unlabelled"],
                evicted=n["evicted"],
            )


class FixedPolicy:
    """Always the same action, with propensity 1.0. Still subject to the hard constraints.

    For the replay runner and the service's fail-open default. ``decide`` raises
    ``NoFeasibleAction`` when the state forbids the action (for example a cloud model on a
    sensitive state); the caller must handle that. Sensitive detection runs first, as in
    ``ScoredPolicy``, and ``Decision.state`` is the marked state. ``report`` does nothing.
    """

    def __init__(
        self,
        action: Action,
        catalog: Catalog | Iterable[ModelProfile],
        *,
        allow: Iterable[str] | None = None,
        deny: Iterable[str] | None = None,
        include_holdout: bool = False,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = new_decision_id,
        name: str = "fixed",
        privacy: PrivacyConfig | None = None,
    ) -> None:
        self.action = action
        self.catalog = _as_catalog(catalog)
        self.allow = _names(allow)
        self.deny = _names(deny) or ()
        self.include_holdout = include_holdout
        self.name = name
        self.privacy = privacy or PrivacyConfig()
        self._clock = clock
        self._new_id = new_id
        # Programming errors (unknown model or effort) surface at construction.
        probe = StepState("probe", "probe", 0, "main", 0)
        try:
            filter_feasible(
                probe,
                (action,),
                self.catalog,
                allow=self.allow,
                deny=self.deny,
                include_holdout=include_holdout,
            )
        except NoFeasibleAction as exc:
            # The probe has no tools, no text and is not sensitive, so only allow, deny or
            # holdout can refuse it: the policy could never decide anything.
            raise ValueError(
                f"the fixed action {action.label()!r} is never allowed: {exc}"
            ) from None

    def decide(self, state: StepState) -> Decision:
        state = mark_before_routing(state, self.privacy)
        filter_feasible(
            state,
            (self.action,),
            self.catalog,
            allow=self.allow,
            deny=self.deny,
            include_holdout=self.include_holdout,
        )
        return Decision(
            decision_id=self._new_id(),
            ts=self._clock(),
            state=state,
            candidates=[self.action],
            action=self.action,
            policy=self.name,
            propensity=1.0,
        )

    def report(self, outcome: Outcome) -> None:
        return None
