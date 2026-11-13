"""Cache-aware cost model. Standard library only. See ``docs/COST-MODEL.md``.

Cost is computed, not learned: arithmetic over catalog prices, token counts and who owns the
prompt cache. The same pricing function, ``price_usage``, prices a planned attempt
(``expected_cost``) and a realized one (``realized_cost``), so the two agree whenever the
predicted token counts equal the realized ones.

Prices are USD per million tokens. All results are USD; latencies are milliseconds.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from steproute.core.catalog import Catalog
from steproute.core.schema import Action, ModelProfile, Outcome, StepState

__all__ = [
    "AttemptCost",
    "AttemptUsage",
    "CostBreakdown",
    "CostModelError",
    "PSuccess",
    "UsageCost",
    "attempt_latency",
    "expected_cost",
    "owns_cache",
    "planned_usage",
    "price_usage",
    "realized_cost",
    "state_after_failure",
]

_PER_TOKEN = 1e-6  # prices are per million tokens

#: Success probability input to ``expected_cost``: one number (single-step actions only), one
#: number per step of the chain, or a callable given the state each step actually runs in.
PSuccess = float | Sequence[float] | Callable[[StepState, Action], float]


class CostModelError(ValueError):
    """An input the cost model cannot price correctly. Never silently mispriced."""


# --- validation helpers ---------------------------------------------------------------------


def _check_number(value: float, what: str, *, low: float = 0.0, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CostModelError(f"{what} must be a number, got {value!r}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise CostModelError(f"{what} must be finite, got {value!r}")
    if v < low or (high is not None and v > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise CostModelError(f"{what} must be {bound}, got {value!r}")
    return v


def _check_profile(p: ModelProfile) -> None:
    for name in ("price_in", "price_out", "cache_read_mult"):
        _check_number(getattr(p, name), f"{p.id}: {name}")
    # The cost model splits a written token into base input plus a write premium. A write
    # discount (multiplier below 1) is not a real provider shape and would make the premium
    # negative, so it is refused rather than priced.
    _check_number(p.cache_write_mult, f"{p.id}: cache_write_mult", low=1.0)
    if isinstance(p.cache_min_tokens, bool) or not isinstance(p.cache_min_tokens, int):
        raise CostModelError(f"{p.id}: cache_min_tokens must be an integer")
    if p.cache_min_tokens < 0:
        raise CostModelError(f"{p.id}: cache_min_tokens must not be negative")
    for i, t in enumerate(p.price_tiers):
        _check_number(t.price_in, f"{p.id}: price_tiers[{i}].price_in")
        _check_number(t.price_out, f"{p.id}: price_tiers[{i}].price_out")


def _check_state(state: StepState) -> None:
    for name in ("prompt_tokens", "cached_prefix_tokens"):
        v = getattr(state, name)
        if isinstance(v, bool) or not isinstance(v, int):
            raise CostModelError(f"state.{name} must be an integer, got {v!r}")
        if v < 0:
            raise CostModelError(f"state.{name} must not be negative, got {v}")
    if state.cached_prefix_tokens > state.prompt_tokens:
        raise CostModelError(
            f"state.cached_prefix_tokens ({state.cached_prefix_tokens}) exceeds "
            f"state.prompt_tokens ({state.prompt_tokens})"
        )


def _output_mult(profile: ModelProfile, effort: str) -> float:
    if effort not in profile.efforts:
        raise CostModelError(
            f"{profile.id}: effort {effort!r} is not supported (efforts {list(profile.efforts)})"
        )
    if effort not in profile.effort_output_mult:
        raise CostModelError(
            f"{profile.id}: effort_output_mult has no entry for {effort!r}; add one to the "
            "catalog (1.0 if unmeasured)"
        )
    return _check_number(profile.effort_output_mult[effort], f"{profile.id}: effort_output_mult")


def _profile_map(profiles: Catalog | Iterable[ModelProfile]) -> dict[str, ModelProfile]:
    out: dict[str, ModelProfile] = {}
    for p in profiles.models if isinstance(profiles, Catalog) else profiles:
        if p.id in out:
            raise CostModelError(f"duplicate profile id {p.id!r}")
        out[p.id] = p
    return out


# --- token counts and prices ----------------------------------------------------------------


@dataclass(frozen=True)
class AttemptUsage:
    """Token counts of one attempt, in the ``Outcome`` convention (``docs/COST-MODEL.md``).

    ``input_tokens`` are prompt tokens neither read from nor billed as written to the cache;
    ``cache_read_tokens`` were read from it; ``cache_write_tokens`` were billed as cache writes.
    The three are disjoint and sum to the full prompt. Counts may be fractional when predicted.
    """

    input_tokens: float = 0.0
    cache_read_tokens: float = 0.0
    cache_write_tokens: float = 0.0
    output_tokens: float = 0.0

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            _check_number(getattr(self, f.name), f"usage.{f.name}")

    @property
    def prompt_tokens(self) -> float:
        """Full prompt length, cache reads and writes included (selects the price tier)."""
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @classmethod
    def from_outcome(cls, outcome: Outcome) -> AttemptUsage:
        """The token fields of an ``Outcome``, which must follow the same convention."""
        return cls(
            input_tokens=outcome.input_tokens,
            cache_read_tokens=outcome.cache_read_tokens,
            cache_write_tokens=outcome.cache_write_tokens,
            output_tokens=outcome.output_tokens,
        )


@dataclass(frozen=True)
class UsageCost:
    """Token counts of one request priced with one profile. All amounts USD.

    ``input_usd`` is every prompt token not read from the cache, at the base input price
    (written tokens included). ``cache_write_usd`` is only the write premium,
    ``(cache_write_mult - 1) * price_in`` per written token, so it is 0 on providers without one.
    ``cache_read_usd`` is the cache hits at ``cache_read_mult * price_in``. ``price_in`` and
    ``price_out`` are the rates in force after the prompt-size tier is applied.
    """

    price_in: float
    price_out: float
    input_usd: float
    cache_read_usd: float
    cache_write_usd: float
    output_usd: float

    @property
    def cache_usd(self) -> float:
        """Cache reads plus the write premium."""
        return self.cache_read_usd + self.cache_write_usd

    @property
    def total_usd(self) -> float:
        return self.input_usd + self.cache_read_usd + self.cache_write_usd + self.output_usd


def price_usage(profile: ModelProfile, usage: AttemptUsage) -> UsageCost:
    """Price one request's token counts with ``profile``.

    The tier is chosen by the full prompt length (``usage.prompt_tokens``) and applies to the
    whole request, output included. This is the one pricing function: ``expected_cost`` uses it
    on planned counts, ``realized_cost`` on an ``Outcome``'s counts, and a collector pricing a
    multi-attempt decision calls it once per attempt and sums the totals into ``cost_usd``.
    """
    _check_profile(profile)
    prompt = usage.prompt_tokens
    price_in, price_out = profile.prices_for(math.ceil(prompt))
    rate = price_in * _PER_TOKEN
    return UsageCost(
        price_in=price_in,
        price_out=price_out,
        input_usd=(usage.input_tokens + usage.cache_write_tokens) * rate,
        cache_read_usd=usage.cache_read_tokens * rate * profile.cache_read_mult,
        cache_write_usd=usage.cache_write_tokens * rate * (profile.cache_write_mult - 1.0),
        output_usd=usage.output_tokens * price_out * _PER_TOKEN,
    )


def realized_cost(outcome: Outcome, profile: ModelProfile) -> float:
    """Actual cost in USD of a single-attempt ``Outcome``, for when the provider reports none.

    Raises ``CostModelError`` for an escalated outcome: its token fields may sum attempts on
    different models and prompt lengths, and no single profile or tier prices that correctly.
    Price each attempt with ``price_usage`` instead and sum. Also raises if
    ``outcome.final_model`` is set and is not ``profile.id``.
    """
    if outcome.escalated:
        raise CostModelError(
            "realized_cost cannot price an escalated outcome (several attempts, possibly on "
            "several models); price each attempt with price_usage() and sum into cost_usd"
        )
    if outcome.final_model is not None and outcome.final_model != profile.id:
        raise CostModelError(
            f"outcome.final_model {outcome.final_model!r} does not match profile {profile.id!r}"
        )
    return price_usage(profile, AttemptUsage.from_outcome(outcome)).total_usd


# --- cache ownership and state transitions --------------------------------------------------


def owns_cache(state: StepState, step: Action, profile: ModelProfile) -> bool:
    """Whether ``step`` can read the cache described by ``state``.

    The cache owner is the pair ``(state.current_model, state.current_effort)``. The step owns
    it if it runs the same model and either the same effort or a profile whose
    ``effort_change_keeps_cache`` is true. An unknown owner effort (``None``) counts as a match
    only when the model supports a single effort (so the effort cannot have differed);
    otherwise the cache is treated as lost, which never understates a cost.
    """
    if step.model != state.current_model or step.model != profile.id:
        return False
    if profile.effort_change_keeps_cache:
        return True
    if state.current_effort is None:
        return len(profile.efforts) == 1
    return state.current_effort == step.effort


def planned_usage(
    state: StepState, step: Action, profile: ModelProfile, est_output_tokens: float
) -> AttemptUsage:
    """Predicted token counts of running ``step`` in ``state``.

    - Prompt below ``profile.cache_min_tokens``: nothing is cached; the whole prompt is input.
    - Step owns the cache: the cached prefix is read; the remainder is written (billed at the
      write rate), since after the attempt the whole prompt is cached.
    - Otherwise the whole prompt is written. On a provider with a write premium this is the
      switch tax; without one it is the lost read discount.
    - Output: ``est_output_tokens * effort_output_mult[step.effort]``.
    """
    _check_state(state)
    est = _check_number(est_output_tokens, "est_output_tokens")
    out = est * _output_mult(profile, step.effort)
    prompt = state.prompt_tokens
    if prompt < profile.cache_min_tokens:
        return AttemptUsage(input_tokens=prompt, output_tokens=out)
    if owns_cache(state, step, profile):
        cached = state.cached_prefix_tokens
        return AttemptUsage(
            cache_read_tokens=cached, cache_write_tokens=prompt - cached, output_tokens=out
        )
    return AttemptUsage(cache_write_tokens=prompt, output_tokens=out)


def state_after_failure(state: StepState, step: Action, profile: ModelProfile) -> StepState:
    """The state the next step of a cascade runs in, after ``step`` ran in ``state`` and failed.

    The next attempt resends the same prompt. The cache is now owned by ``(step.model,
    step.effort)`` and holds the whole prompt, unless the prompt was below
    ``profile.cache_min_tokens`` (then nothing is cached). ``step`` is appended to
    ``failed_attempts`` so a predictor can condition on the failure. ``state`` is not changed.
    """
    _check_state(state)
    cached = state.prompt_tokens if state.prompt_tokens >= profile.cache_min_tokens else 0
    return dataclasses.replace(
        state,
        current_model=step.model,
        current_effort=step.effort,
        cached_prefix_tokens=cached,
        failed_attempts=[*state.failed_attempts, Action(step.model, step.effort)],
        tools=list(state.tools),
        features=dict(state.features),
    )


def attempt_latency(
    state: StepState, step: Action, profile: ModelProfile, usage: AttemptUsage
) -> float | None:
    """Estimated latency in ms of one attempt, or ``None`` if the profile has no estimate.

    ``latency_base_ms + latency_ms_per_input_token * (prompt tokens not read from cache)
    + latency_ms_per_output_token * output tokens``, plus ``latency_switch_ms`` (for example a
    local model reload) when ``step.model`` is not ``state.current_model``.
    """
    base = profile.latency_base_ms
    per_in = profile.latency_ms_per_input_token
    per_out = profile.latency_ms_per_output_token
    if base is None or per_in is None or per_out is None:
        return None
    switch = profile.latency_switch_ms or 0.0
    for v in (base, per_in, per_out, switch):
        _check_number(v, f"{profile.id}: latency")
    uncached = usage.input_tokens + usage.cache_write_tokens
    total = base + per_in * uncached + per_out * usage.output_tokens
    if step.model != state.current_model:
        total += switch
    return total


# --- expected cost of an action -------------------------------------------------------------


@dataclass(frozen=True)
class AttemptCost:
    """One step of an action, priced in the state it would run in.

    ``cost`` and ``latency_ms`` are conditional on the attempt running. ``reach_prob`` is the
    probability that it runs (1 for the first step), and ``p_success`` the probability it
    succeeds given that it runs.
    """

    step: Action
    warm: bool
    usage: AttemptUsage
    cost: UsageCost
    latency_ms: float | None
    p_success: float
    reach_prob: float

    @property
    def total_usd(self) -> float:
        return self.cost.total_usd


@dataclass(frozen=True)
class CostBreakdown:
    """Expected cost of an action. All amounts USD, latencies ms.

    First attempt (always paid): ``input_usd``, ``cache_read_usd``, ``cache_write_usd`` (write
    premium only), ``output_usd``, summing to ``first_attempt_usd``.

    Failure branch (an expectation): ``escalation_usd`` is
    ``(1 - p1) * (c2 + (1 - p2) * (c3 + ...))``, each ``c_i`` priced in the state the failed
    attempts leave behind. ``total_usd = first_attempt_usd + escalation_usd``.

    ``max_usd`` is the cost if every step runs (all but the last fail), useful for spend caps.
    ``p_success`` is ``1 - prod(1 - p_i)``. ``latency_ms`` is the expected latency and
    ``first_attempt_latency_ms`` the first step's; both are ``None`` unless every profile
    involved carries a latency estimate. ``attempts`` has the per-step detail.
    """

    action: Action
    attempts: tuple[AttemptCost, ...]
    input_usd: float
    cache_read_usd: float
    cache_write_usd: float
    output_usd: float
    first_attempt_usd: float
    escalation_usd: float
    total_usd: float
    max_usd: float
    p_success: float
    first_attempt_latency_ms: float | None
    latency_ms: float | None

    @property
    def cache_usd(self) -> float:
        """First-attempt cache reads plus write premium."""
        return self.cache_read_usd + self.cache_write_usd


def _p_resolver(p_success: PSuccess, n: int) -> Callable[[int, StepState, Action], float]:
    if callable(p_success):
        fn = p_success
        return lambda i, s, a: _check_number(fn(s, a), f"p_success for step {i}", high=1.0)
    if isinstance(p_success, int | float) and not isinstance(p_success, bool):
        if n != 1:
            raise CostModelError(
                f"a single p_success is ambiguous for a {n}-step cascade; pass one probability "
                "per step (conditional on the earlier steps failing) or a callable"
            )
        p = _check_number(p_success, "p_success", high=1.0)
        return lambda i, s, a: p
    if isinstance(p_success, Sequence) and not isinstance(p_success, str):
        if len(p_success) != n:
            raise CostModelError(f"p_success has {len(p_success)} entries; the chain has {n}")
        ps = [_check_number(p, f"p_success[{i}]", high=1.0) for i, p in enumerate(p_success)]
        return lambda i, s, a: ps[i]
    raise CostModelError(f"unsupported p_success {p_success!r}")


def _per_step(est: float | Sequence[float], n: int) -> list[float]:
    if isinstance(est, Sequence) and not isinstance(est, str):
        if len(est) != n:
            raise CostModelError(f"est_output_tokens has {len(est)} entries; the chain has {n}")
        return [_check_number(e, f"est_output_tokens[{i}]") for i, e in enumerate(est)]
    return [_check_number(est, "est_output_tokens")] * n


def expected_cost(
    state: StepState,
    action: Action,
    profiles: Catalog | Iterable[ModelProfile],
    est_output_tokens: float | Sequence[float],
    p_success: PSuccess,
) -> CostBreakdown:
    """Expected cost of taking ``action`` in ``state``.

    ``est_output_tokens`` is the caller's estimate before the effort multiplier: one number for
    every step, or one per step. ``p_success`` is the probability each step succeeds *given it
    runs*, i.e. given every earlier step failed: one number for a single-step action, a sequence
    with one entry per step, or a callable ``(state, step) -> float`` called with the state each
    step actually runs in (``state.failed_attempts`` lists the steps that failed before it).
    Overall success ``1 - prod(1 - p_i)`` is exact when the ``p_i`` are conditional in this
    sense; independent per-step probabilities overstate it because failures are correlated.

    Raises ``CostModelError`` for a model absent from ``profiles``, an unsupported effort, a
    missing ``effort_output_mult`` entry, a probability outside [0, 1], negative or non-finite
    numbers, or ``cached_prefix_tokens > prompt_tokens``.
    """
    _check_state(state)
    by_id = _profile_map(profiles)
    chain = action.chain()
    ests = _per_step(est_output_tokens, len(chain))
    p_of = _p_resolver(p_success, len(chain))

    attempts: list[AttemptCost] = []
    current = state
    reach = 1.0
    for i, step in enumerate(chain):
        profile = by_id.get(step.model)
        if profile is None:
            raise CostModelError(f"model {step.model!r} is not in the catalog")
        usage = planned_usage(current, step, profile, ests[i])
        p_i = p_of(i, current, step)
        attempts.append(
            AttemptCost(
                step=step,
                warm=owns_cache(current, step, profile) and usage.cache_read_tokens > 0,
                usage=usage,
                cost=price_usage(profile, usage),
                latency_ms=attempt_latency(current, step, profile, usage),
                p_success=p_i,
                reach_prob=reach,
            )
        )
        reach *= 1.0 - p_i
        if i + 1 < len(chain):
            current = state_after_failure(current, step, profile)

    # Fold from the last step: rest = c_i + (1 - p_i) * rest. The worst case folds the same way
    # with every (1 - p_i) set to 1, so first <= total <= max holds exactly in floating point.
    rest = 0.0
    worst = 0.0
    lat_rest: float | None = 0.0
    for a in reversed(attempts[1:]):
        rest = a.total_usd + (1.0 - a.p_success) * rest
        worst = a.total_usd + worst
        if lat_rest is not None and a.latency_ms is not None:
            lat_rest = a.latency_ms + (1.0 - a.p_success) * lat_rest
        else:
            lat_rest = None
    first = attempts[0]
    escalation = (1.0 - first.p_success) * rest
    expected_latency: float | None = None
    if first.latency_ms is not None and lat_rest is not None:
        expected_latency = first.latency_ms + (1.0 - first.p_success) * lat_rest
    survive = 1.0
    for a in attempts:
        survive *= 1.0 - a.p_success
    return CostBreakdown(
        action=action,
        attempts=tuple(attempts),
        input_usd=first.cost.input_usd,
        cache_read_usd=first.cost.cache_read_usd,
        cache_write_usd=first.cost.cache_write_usd,
        output_usd=first.cost.output_usd,
        first_attempt_usd=first.total_usd,
        escalation_usd=escalation,
        total_usd=first.total_usd + escalation,
        max_usd=first.total_usd + worst,
        p_success=1.0 - survive,
        first_attempt_latency_ms=first.latency_ms,
        latency_ms=expected_latency,
    )
