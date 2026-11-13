"""Cost model: worked examples from docs/COST-MODEL.md, acceptance criteria, edges, properties."""

from __future__ import annotations

import dataclasses
import math
import re
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steproute.core import (
    Action,
    AttemptUsage,
    CostModelError,
    ModelProfile,
    Outcome,
    StepState,
    expected_cost,
    owns_cache,
    planned_usage,
    price_usage,
    realized_cost,
    state_after_failure,
)
from steproute.core.catalog import Catalog, CatalogError, load_default_catalog

ROOT = Path(__file__).resolve().parent.parent
COST_MD = ROOT / "docs" / "COST-MODEL.md"
CAT = load_default_catalog()
TOL = 1e-12

OPUS, SONNET, HAIKU = "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"
FLASH, PRO = "deepseek-flash", "deepseek-v4-pro"


def state(
    prompt: int,
    cached: int = 0,
    model: str | None = None,
    effort: str | None = None,
    kind: str = "main",
) -> StepState:
    return StepState(
        "proxy",
        "s1",
        1,
        kind,
        prompt,
        cached_prefix_tokens=cached,
        current_model=model,
        current_effort=effort,
    )


# --- worked examples ------------------------------------------------------------------------


def _parts(prefix: str, b: object) -> dict[str, float]:
    names = ("input_usd", "cache_read_usd", "cache_write_usd", "output_usd", "total_usd")
    return {f"{prefix}.{n}": getattr(b, n) for n in names}


def ex_anthropic() -> dict[str, float]:
    s = state(50_000, 45_000, OPUS, "medium")
    stay = expected_cost(s, Action(OPUS, "medium"), CAT, 2_000, 0.9)
    switch = expected_cost(s, Action(SONNET, "medium"), CAT, 2_000, 0.9)
    return {**_parts("stay", stay), **_parts("switch", switch)}


def ex_deepseek() -> dict[str, float]:
    s = state(50_000, 45_000, FLASH, "high")
    stay = expected_cost(s, Action(FLASH, "high"), CAT, 2_000, 0.9)
    switch = expected_cost(s, Action(PRO, "high"), CAT, 2_000, 0.9)
    twin = dataclasses.replace(CAT.get(FLASH), id="flash-twin")
    to_twin = expected_cost(s, Action("flash-twin", "high"), [*CAT.models, twin], 2_000, 0.9)
    out = {**_parts("stay", stay), **_parts("switch", switch)}
    del out["switch.cache_read_usd"]
    out["flash_switch_tax_usd"] = to_twin.total_usd - stay.total_usd
    return out


def ex_haiku() -> dict[str, float]:
    out: dict[str, float] = {}
    for p in (100_000, 100_001):
        b = expected_cost(state(p, 90_000, HAIKU, "medium"), Action(HAIKU, "medium"), CAT, 1_000, 1)
        key = f"at_{p}"
        out.update(_parts(key, b))
        out[f"{key}.price_in"] = b.attempts[0].cost.price_in
    for k in ("input_usd", "cache_read_usd", "cache_write_usd", "output_usd"):
        del out[f"at_100000.{k}"]
    return out


def ex_effort() -> dict[str, float]:
    s = state(40_000, 36_000, SONNET, "low")
    a = Action(SONNET, "low", Action(SONNET, "high"))
    keep = dataclasses.replace(CAT.get(SONNET), effort_change_keeps_cache=True)
    out: dict[str, float] = {}
    for key, profiles in (("no_keep", CAT), ("keep", [keep])):
        b = expected_cost(s, a, profiles, 1_500, [0.7, 0.6])
        out[f"{key}.first_attempt_usd"] = b.first_attempt_usd
        out[f"{key}.c2_usd"] = b.attempts[1].total_usd
        out[f"{key}.escalation_usd"] = b.escalation_usd
        out[f"{key}.total_usd"] = b.total_usd
        out[f"{key}.max_usd"] = b.max_usd
        out["p_success"] = b.p_success
    del out["keep.first_attempt_usd"]
    return out


def ex_cross() -> dict[str, float]:
    s = state(20_000, kind="subagent")
    b = expected_cost(s, Action(HAIKU, "medium", Action(OPUS, "medium")), CAT, 1_000, [0.6, 0.85])
    alone = expected_cost(s, Action(OPUS, "medium"), CAT, 1_000, 0.85)
    return {
        "first_attempt_usd": b.first_attempt_usd,
        "c2_usd": b.attempts[1].total_usd,
        "escalation_usd": b.escalation_usd,
        "total_usd": b.total_usd,
        "max_usd": b.max_usd,
        "p_success": b.p_success,
        "opus_alone_total_usd": alone.total_usd,
    }


def local_profile() -> ModelProfile:
    return dataclasses.replace(
        CAT.get("qwen3-coder:30b"),
        price_in=0.02,
        price_out=0.08,
        cache_read_mult=0.0,
        latency_base_ms=200.0,
        latency_ms_per_input_token=0.5,
        latency_ms_per_output_token=25.0,
        latency_switch_ms=8000.0,
    )


def ex_local() -> dict[str, float]:
    p = local_profile()
    a = Action(p.id, "medium")
    switch = expected_cost(state(12_000, 10_000, "gpt-oss:20b", "medium"), a, [p], 800, 0.5)
    stay = expected_cost(state(12_000, 10_000, p.id, None), a, [p], 800, 0.5)
    out: dict[str, float] = {}
    for key, b in (("switch", switch), ("stay", stay)):
        assert b.latency_ms is not None
        out[f"{key}.total_usd"] = b.total_usd
        out[f"{key}.latency_ms"] = b.latency_ms
    return out


EXAMPLES: dict[str, Callable[[], dict[str, float]]] = {
    "anthropic-stay-vs-switch": ex_anthropic,
    "deepseek-stay-vs-switch": ex_deepseek,
    "haiku-tier-boundary": ex_haiku,
    "effort-cascade": ex_effort,
    "cross-model-cascade": ex_cross,
    "local-compute-price": ex_local,
}


def _doc_examples() -> dict[str, dict[str, float]]:
    md = COST_MD.read_text(encoding="utf-8")
    blocks = re.findall(r"```text cost-example=([\w-]+)\n(.*?)\n```", md, flags=re.S)
    out: dict[str, dict[str, float]] = {}
    for name, body in blocks:
        values: dict[str, float] = {}
        for line in body.splitlines():
            key, _, value = line.partition(" = ")
            values[key.strip()] = float(value)
        out[name] = values
    return out


def test_doc_has_every_example() -> None:
    assert set(_doc_examples()) == set(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_worked_example_reproduced(name: str) -> None:
    doc = _doc_examples()[name]
    got = EXAMPLES[name]()
    assert set(doc) == set(got), f"{name}: doc and test disagree on which numbers are stated"
    for key, value in doc.items():
        assert abs(got[key] - value) <= TOL, f"{name}: {key} = {got[key]!r}, doc says {value!r}"


# --- acceptance criteria --------------------------------------------------------------------


@pytest.mark.parametrize("model", [OPUS, SONNET, HAIKU, FLASH, PRO])
def test_warm_stay_cheaper_than_switch_to_identical_price(model: str) -> None:
    profile = CAT.get(model)
    effort = profile.efforts[-1]
    twin = dataclasses.replace(profile, id=model + "-twin")
    profiles = [*CAT.models, twin]
    s = state(30_000, 25_000, model, effort)
    stay = expected_cost(s, Action(model, effort), profiles, 1_000, 0.5)
    switch = expected_cost(s, Action(twin.id, effort), profiles, 1_000, 0.5)
    assert stay.total_usd < switch.total_usd
    assert stay.output_usd == switch.output_usd


def test_ollama_switch_costs_latency_not_money() -> None:
    p = dataclasses.replace(
        CAT.get("qwen3-coder:30b"),
        latency_base_ms=100.0,
        latency_ms_per_input_token=0.1,
        latency_ms_per_output_token=20.0,
        latency_switch_ms=5000.0,
    )
    twin = dataclasses.replace(p, id="twin")
    s = state(10_000, 9_000, p.id, "medium")
    stay = expected_cost(s, Action(p.id), [p, twin], 500, 0.5)
    switch = expected_cost(s, Action("twin"), [p, twin], 500, 0.5)
    assert stay.total_usd == switch.total_usd == 0.0
    assert stay.latency_ms is not None and switch.latency_ms is not None
    assert switch.latency_ms - stay.latency_ms == pytest.approx(5000.0 + 0.1 * 9_000)


def test_realized_matches_expected_when_predictions_match() -> None:
    for s, a in [
        (state(50_000, 45_000, OPUS, "medium"), Action(OPUS, "medium")),
        (state(50_000, 45_000, OPUS, "medium"), Action(SONNET, "high")),
        (state(150_000, 140_000, HAIKU, "low"), Action(HAIKU, "low")),
        (state(300, 0, None), Action(HAIKU, "low")),
        (state(50_000, 45_000, FLASH, "high"), Action(FLASH, "high")),
    ]:
        b = expected_cost(s, a, CAT, 1_234, 0.8)
        u = b.attempts[0].usage
        outcome = Outcome(
            "d",
            1.0,
            True,
            final_model=a.model,
            input_tokens=int(u.input_tokens),
            output_tokens=int(u.output_tokens),
            cache_read_tokens=int(u.cache_read_tokens),
            cache_write_tokens=int(u.cache_write_tokens),
        )
        assert realized_cost(outcome, CAT.get(a.model)) == pytest.approx(b.total_usd, abs=TOL)


def test_realized_deepseek_normalised_usage_agrees() -> None:
    # DeepSeek reports prompt_tokens including cached tokens and no writes; normalised, the
    # misses are input_tokens. Same cost as the planned counts, which classify them as writes.
    s = state(50_000, 45_000, FLASH, "high")
    b = expected_cost(s, Action(FLASH, "high"), CAT, 2_000, 0.5)
    outcome = Outcome(
        "d", 1.0, True, input_tokens=5_000, cache_read_tokens=45_000, output_tokens=2_000
    )
    assert realized_cost(outcome, CAT.get(FLASH)) == pytest.approx(b.total_usd, abs=TOL)


def test_realized_refuses_escalated_and_mismatch() -> None:
    o = Outcome("d", 1.0, True, escalated=True, input_tokens=10)
    with pytest.raises(CostModelError, match="escalated"):
        realized_cost(o, CAT.get(HAIKU))
    with pytest.raises(CostModelError, match="does not match"):
        realized_cost(Outcome("d", 1.0, True, final_model=OPUS), CAT.get(HAIKU))


def test_multi_attempt_priced_per_attempt() -> None:
    # Two 60K Haiku attempts: per attempt both are lower tier; summed they would cross 100K.
    u = AttemptUsage(input_tokens=60_000, output_tokens=100)
    per_attempt = 2 * price_usage(CAT.get(HAIKU), u).total_usd
    summed = price_usage(CAT.get(HAIKU), AttemptUsage(input_tokens=120_000, output_tokens=200))
    assert per_attempt == pytest.approx(2 * (60_000 * 0.10 + 100 * 0.5) / 1e6)
    assert summed.total_usd > per_attempt


# --- semantics ------------------------------------------------------------------------------


def test_owner_effort_rules() -> None:
    sonnet = CAT.get(SONNET)
    assert owns_cache(state(10, 5, SONNET, "low"), Action(SONNET, "low"), sonnet)
    assert not owns_cache(state(10, 5, SONNET, "low"), Action(SONNET, "high"), sonnet)
    assert not owns_cache(state(10, 5, SONNET, None), Action(SONNET, "low"), sonnet)
    keep = dataclasses.replace(sonnet, effort_change_keeps_cache=True)
    assert owns_cache(state(10, 5, SONNET, None), Action(SONNET, "high"), keep)
    coder = CAT.get("qwen3-coder:30b")
    assert owns_cache(state(10, 5, coder.id, None), Action(coder.id), coder)
    assert not owns_cache(state(10, 5, OPUS, "low"), Action(SONNET, "low"), sonnet)


def test_cache_minimum() -> None:
    haiku = CAT.get(HAIKU)
    below = planned_usage(state(511, 0, None), Action(HAIKU, "low"), haiku, 10)
    assert (below.input_tokens, below.cache_write_tokens) == (511, 0)
    at = planned_usage(state(512, 0, None), Action(HAIKU, "low"), haiku, 10)
    assert (at.input_tokens, at.cache_write_tokens) == (0, 512)
    assert state_after_failure(state(511), Action(HAIKU, "low"), haiku).cached_prefix_tokens == 0
    assert state_after_failure(state(512), Action(HAIKU, "low"), haiku).cached_prefix_tokens == 512


def test_state_after_failure_records_failure() -> None:
    s = state(1_000, 200, OPUS, "low")
    s.tools.append("Bash")
    after = state_after_failure(s, Action(SONNET, "high"), CAT.get(SONNET))
    assert after.current_model == SONNET and after.current_effort == "high"
    assert after.cached_prefix_tokens == 1_000
    assert after.failed_attempts == [Action(SONNET, "high")]
    assert s.failed_attempts == [] and s.cached_prefix_tokens == 200
    after.tools.append("x")
    assert s.tools == ["Bash"]


def test_callable_p_sees_running_state() -> None:
    seen: list[tuple[str | None, int, int]] = []

    def predictor(st_: StepState, step: Action) -> float:
        seen.append((st_.current_model, len(st_.failed_attempts), st_.cached_prefix_tokens))
        return 0.3 if st_.failed_attempts else 0.5

    a = Action(HAIKU, "low", Action(SONNET, "low", Action(OPUS, "low")))
    b = expected_cost(state(5_000, 0, None), a, CAT, 100, predictor)
    assert seen == [(None, 0, 0), (HAIKU, 1, 5_000), (SONNET, 2, 5_000)]
    assert b.p_success == pytest.approx(1 - 0.5 * 0.7 * 0.7)
    assert [x.reach_prob for x in b.attempts] == pytest.approx([1.0, 0.5, 0.35])


def test_zero_prompt_charges_output_only() -> None:
    b = expected_cost(state(0), Action(OPUS, "low"), CAT, 1_000, 1.0)
    assert b.total_usd == pytest.approx(1_000 * 20 / 1e6)
    assert b.input_usd == b.cache_usd == 0.0


def test_local_profile_compute_price_through_catalog() -> None:
    text = """
[[model]]
id = "local-x"
provider = "ollama"
local = true
price_in = 0.05
price_out = 0.2
context_window = 32768
efforts = ["medium"]
effort_output_mult = { medium = 1.0 }
latency_base_ms = 100.0
latency_ms_per_input_token = 0.2
latency_ms_per_output_token = 30.0
latency_switch_ms = 6000.0
verified = true
"""
    import tomllib

    cat = Catalog.from_dict(tomllib.loads(text))
    b = expected_cost(state(1_000), Action("local-x"), cat, 100, 1.0)
    assert b.total_usd == pytest.approx((1_000 * 0.05 + 100 * 0.2) / 1e6)
    assert b.latency_ms == pytest.approx(100 + 0.2 * 1_000 + 30 * 100 + 6000)
    assert Catalog.from_dict(cat.to_dict()) == cat


def test_no_latency_without_estimate() -> None:
    b = expected_cost(state(1_000), Action(OPUS, "low"), CAT, 100, 1.0)
    assert b.latency_ms is None and b.first_attempt_latency_ms is None


def test_catalog_rejects_partial_latency_and_negative_minimum() -> None:
    base = {
        "id": "m",
        "provider": "p",
        "price_in": 1.0,
        "price_out": 1.0,
        "context_window": 10,
        "verified": True,
    }
    with pytest.raises(CatalogError, match="latency"):
        Catalog.from_dict({"model": [{**base, "latency_base_ms": 1.0}]})
    with pytest.raises(CatalogError, match="cache_min_tokens"):
        Catalog.from_dict({"model": [{**base, "cache_min_tokens": -1}]})


def test_default_catalog_loads_without_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cat = load_default_catalog()
    assert all(m.cache_min_tokens == 512 for m in cat.models if m.provider == "anthropic")
    assert all(m.latency_base_ms is None for m in cat.models)


# --- edges that must raise ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: expected_cost(state(10, 11), Action(OPUS, "low"), CAT, 1, 0.5), "exceeds"),
        (lambda: expected_cost(state(10), Action("nope"), CAT, 1, 0.5), "not in the catalog"),
        (lambda: expected_cost(state(10), Action(OPUS, "none"), CAT, 1, 0.5), "not supported"),
        (lambda: expected_cost(state(10), Action(OPUS, "low"), CAT, 1, 1.5), r"\[0.0, 1.0\]"),
        (lambda: expected_cost(state(10), Action(OPUS, "low"), CAT, 1, -0.1), r"\[0.0, 1.0\]"),
        (lambda: expected_cost(state(10), Action(OPUS, "low"), CAT, 1, math.nan), "finite"),
        (lambda: expected_cost(state(10), Action(OPUS, "low"), CAT, -1, 0.5), ">= 0"),
        (
            lambda: expected_cost(
                state(10), Action(OPUS, "low", Action(OPUS, "high")), CAT, 1, 0.5
            ),
            "ambiguous",
        ),
        (
            lambda: expected_cost(
                state(10), Action(OPUS, "low", Action(OPUS, "high")), CAT, 1, [0.5]
            ),
            "entries",
        ),
        (
            lambda: expected_cost(
                state(10), Action(OPUS, "low", Action(OPUS, "high")), CAT, 1, lambda s, a: 2.0
            ),
            "step 0",
        ),
    ],
)
def test_bad_inputs_raise(call: Callable[[], object], match: str) -> None:
    with pytest.raises(CostModelError, match=match):
        call()


def test_negative_state_and_bad_profiles_raise() -> None:
    with pytest.raises(CostModelError):
        expected_cost(state(-1), Action(OPUS, "low"), CAT, 1, 0.5)
    missing = dataclasses.replace(CAT.get(OPUS), effort_output_mult={"low": 1.0})
    with pytest.raises(CostModelError, match="effort_output_mult"):
        expected_cost(state(10), Action(OPUS, "high"), [missing], 1, 0.5)
    discount = dataclasses.replace(CAT.get(OPUS), cache_write_mult=0.5)
    with pytest.raises(CostModelError, match="cache_write_mult"):
        expected_cost(state(10), Action(OPUS, "low"), [discount], 1, 0.5)
    negative = dataclasses.replace(CAT.get(OPUS), price_in=-1.0)
    with pytest.raises(CostModelError, match="price_in"):
        expected_cost(state(10), Action(OPUS, "low"), [negative], 1, 0.5)
    with pytest.raises(CostModelError, match="duplicate"):
        expected_cost(state(10), Action(OPUS, "low"), [CAT.get(OPUS)] * 2, 1, 0.5)
    with pytest.raises(CostModelError):
        AttemptUsage(input_tokens=-1)
    with pytest.raises(ValueError):
        StepState("h", "s", 0, "main", 1, current_effort="ultra")


# --- properties -----------------------------------------------------------------------------

POOL = [m for m in CAT.models if not m.holdout]
steps = st.sampled_from([Action(m.id, e) for m in POOL for e in m.efforts])
probs = st.floats(min_value=0.0, max_value=1.0)
tokens = st.integers(min_value=0, max_value=400_000)
outs = st.floats(min_value=0.0, max_value=50_000.0)


@st.composite
def situations(draw: st.DrawFn, max_steps: int = 4) -> tuple[StepState, Action, list[float]]:
    prompt = draw(tokens)
    cached = draw(st.integers(min_value=0, max_value=prompt))
    owner = draw(st.none() | steps)
    s = state(
        prompt,
        cached,
        None if owner is None else owner.model,
        None if owner is None else draw(st.none() | st.just(owner.effort)),
    )
    chain = draw(st.lists(steps, min_size=1, max_size=max_steps))
    action: Action | None = None
    for step in reversed(chain):
        action = Action(step.model, step.effort, action)
    assert action is not None
    return s, action, draw(st.lists(probs, min_size=len(chain), max_size=len(chain)))


@settings(max_examples=300)
@given(situations(), outs)
def test_cascade_cost_between_first_step_and_sum(
    sit: tuple[StepState, Action, list[float]], est: float
) -> None:
    s, a, ps = sit
    b = expected_cost(s, a, CAT, est, ps)
    assert b.first_attempt_usd <= b.total_usd <= b.max_usd
    assert b.max_usd == pytest.approx(sum(x.total_usd for x in b.attempts))
    assert b.total_usd == b.first_attempt_usd + b.escalation_usd
    assert 0.0 <= b.p_success <= 1.0


@settings(max_examples=300)
@given(situations(), outs)
def test_costs_non_negative(sit: tuple[StepState, Action, list[float]], est: float) -> None:
    s, a, ps = sit
    b = expected_cost(s, a, CAT, est, ps)
    for x in b.attempts:
        c = x.cost
        assert min(c.input_usd, c.cache_read_usd, c.cache_write_usd, c.output_usd) >= 0.0
    assert b.escalation_usd >= 0.0


@settings(max_examples=200)
@given(situations(), outs)
def test_deeper_cascade_never_cheaper_than_prefix(
    sit: tuple[StepState, Action, list[float]], est: float
) -> None:
    s, a, ps = sit
    chain = a.chain()
    full = expected_cost(s, a, CAT, est, ps).total_usd
    for k in range(1, len(chain)):
        prefix: Action | None = None
        for step in reversed(chain[:k]):
            prefix = Action(step.model, step.effort, prefix)
        assert prefix is not None
        assert expected_cost(s, prefix, CAT, est, ps[:k]).total_usd <= full


@settings(max_examples=300)
@given(situations(), outs, st.integers(min_value=0, max_value=50_000), outs)
def test_monotone_in_output_and_prompt(
    sit: tuple[StepState, Action, list[float]], est: float, extra_prompt: int, extra_out: float
) -> None:
    s, a, ps = sit
    base = expected_cost(s, a, CAT, est, ps).total_usd
    assert expected_cost(s, a, CAT, est + extra_out, ps).total_usd >= base
    if s.prompt_tokens >= 512:  # crossing the cache minimum can legitimately lower a cascade
        bigger = dataclasses.replace(s, prompt_tokens=s.prompt_tokens + extra_prompt)
        assert expected_cost(bigger, a, CAT, est, ps).total_usd >= base * (1 - 1e-12)


@settings(max_examples=300)
@given(situations(max_steps=1), outs, st.integers(min_value=0, max_value=50_000))
def test_more_cache_never_costs_more(
    sit: tuple[StepState, Action, list[float]], est: float, more: int
) -> None:
    s, a, ps = sit
    base = expected_cost(s, a, CAT, est, ps).total_usd
    cached = min(s.prompt_tokens, s.cached_prefix_tokens + more)
    warmer = dataclasses.replace(s, cached_prefix_tokens=cached)
    assert expected_cost(warmer, a, CAT, est, ps).total_usd <= base * (1 + 1e-12)


price_bump = st.floats(min_value=0.0, max_value=10.0)


@settings(max_examples=300)
@given(
    situations(), outs, st.sampled_from(["price_in", "price_out", "cache_write_mult"]), price_bump
)
def test_monotone_in_prices(
    sit: tuple[StepState, Action, list[float]], est: float, name: str, bump: float
) -> None:
    s, a, ps = sit
    base = expected_cost(s, a, CAT, est, ps).total_usd
    bumped = []
    for m in CAT.models:
        changes: dict[str, object] = {name: getattr(m, name) + bump}
        if name in ("price_in", "price_out"):
            tiers = tuple(
                dataclasses.replace(t, **{name: getattr(t, name) + bump}) for t in m.price_tiers
            )
            changes["price_tiers"] = tiers
        bumped.append(dataclasses.replace(m, **changes))  # type: ignore[arg-type]
    assert expected_cost(s, a, bumped, est, ps).total_usd >= base * (1 - 1e-12)


def test_empty_effort_output_mult_means_no_adjustment() -> None:
    # An empty table is "unmeasured": it prices like an explicit 1.0 for every effort.
    explicit = CAT.get(OPUS)
    assert set(explicit.effort_output_mult.values()) == {1.0}
    empty = dataclasses.replace(explicit, effort_output_mult={})
    s = state(10_000)
    a = Action(OPUS, "high")
    assert (
        expected_cost(s, a, [empty], 500, 0.5).total_usd
        == expected_cost(s, a, [explicit], 500, 0.5).total_usd
    )
