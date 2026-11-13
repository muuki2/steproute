from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steproute.core import NoFeasibleAction, enumerate_actions, filter_feasible
from steproute.core.catalog import Catalog, load_default_catalog
from steproute.core.schema import EFFORTS, Action, ModelProfile, StepState, effort_rank


def mp(
    mid: str,
    tier: int,
    efforts: tuple[str, ...] = ("low", "medium", "high"),
    *,
    local: bool = False,
    tools: bool = True,
    holdout: bool = False,
    context_window: int = 100_000,
) -> ModelProfile:
    return ModelProfile(
        id=mid,
        provider="p",
        price_in=1.0,
        price_out=2.0,
        context_window=context_window,
        local=local,
        efforts=efforts,
        tier=tier,
        tools=tools,
        holdout=holdout,
    )


def state(**kw: object) -> StepState:
    base: dict[str, object] = {
        "harness": "h",
        "session_id": "s",
        "step_idx": 0,
        "kind": "edit",
        "prompt_tokens": 1000,
    }
    base.update(kw)
    return StepState(**base)  # type: ignore[arg-type]


def depth(a: Action) -> int:
    return len(a.chain())


def test_count_four_models_three_efforts() -> None:
    cat = Catalog(tuple(mp(m, t) for t, m in enumerate("abcd")))
    # Hand count, 4 models with distinct tiers, 3 efforts each (low < medium < high).
    # Depth 1: 4 models x 3 efforts = 12.
    # Depth 2, same model: an ordered pair of strictly rising efforts is C(3,2) = 3
    #   (low>med, low>high, med>high), so 4 x 3 = 12.
    # Depth 2, different models: the second model must have a higher tier, so 6 ordered
    #   model pairs (a>b, a>c, a>d, b>c, b>d, c>d), each with 3 x 3 efforts = 54.
    # Depth 2 total 12 + 54 = 66; depth 1 + 2 = 78.
    acts = enumerate_actions(cat, 2)
    assert sum(depth(a) == 1 for a in acts) == 12
    assert sum(depth(a) == 2 for a in acts) == 66
    assert len(acts) == 78


def test_count_with_tier_tie() -> None:
    # tiers: a=1, b=2, c=2, d=3, three efforts each. Depth 1: 12. Same-model depth 2: 12.
    # Cross-model strictly higher tier: (a,b) (a,c) (a,d) (b,d) (c,d) = 5 pairs; b and c tie
    # so neither escalates to the other. 5 x 9 = 45. Depth 2 = 57; total 69.
    cat = Catalog((mp("a", 1), mp("b", 2), mp("c", 2), mp("d", 3)))
    acts = enumerate_actions(cat, 2)
    assert sum(depth(a) == 1 for a in acts) == 12
    assert sum(depth(a) == 2 for a in acts) == 57
    assert len(acts) == 69
    labels = {a.label() for a in acts}
    assert "b (low) -> c (high)" not in labels
    assert "c (low) -> b (high)" not in labels


def test_count_depth_three() -> None:
    # x (tier 1): low, high. y (tier 2): low, medium, high. Steps x.l x.h y.l y.m y.h (5).
    # Depth 2 (a then strictly stronger b): x.l>x.h, x.l>y*(3), x.h>y*(3), y.l>y.m, y.l>y.h,
    #   y.m>y.h = 1+3+3+3 = 10.
    # Depth 3: x.l>x.h>y* = 3; x.l>y.l>y.m, x.l>y.l>y.h, x.l>y.m>y.h = 3;
    #   x.h>y.l>y.m, x.h>y.l>y.h, x.h>y.m>y.h = 3; y.l>y.m>y.h = 1. Total 10.
    cat = Catalog((mp("x", 1, ("low", "high")), mp("y", 2)))
    by_depth = [0, 0, 0, 0]
    for a in enumerate_actions(cat, 3):
        by_depth[depth(a)] += 1
    assert by_depth[1:] == [5, 10, 10]
    # Depth 4: x.l>x.h>y.l>y.m, x.l>x.h>y.l>y.h, x.l>x.h>y.m>y.h, x.l>y.l>y.m>y.h,
    #   x.h>y.l>y.m>y.h = 5. Depth 5: x.l>x.h>y.l>y.m>y.h = 1. Totals: 30 and 31.
    assert len(enumerate_actions(cat, 4)) == 30
    assert len(enumerate_actions(cat, 9)) == 31


def test_default_catalog_counts() -> None:
    cat = load_default_catalog(warn_unverified=False)
    assert [len(enumerate_actions(cat, d)) for d in (1, 2, 3)] == [34, 555, 5399]
    assert len(enumerate_actions(cat, 1, include_holdout=True)) == 35


def test_order_is_deterministic_and_documented() -> None:
    cat = Catalog((mp("hi", 2, ("low", "high")), mp("lo", 1, ("high", "low"))))
    labels = [a.label() for a in enumerate_actions(cat, 2)]
    assert labels[:4] == ["lo (low)", "lo (high)", "hi (low)", "hi (high)"]
    assert labels[4] == "lo (low) -> lo (high)"
    assert labels[5:8] == [
        "lo (low) -> hi (low)",
        "lo (low) -> hi (high)",
        "lo (high) -> hi (low)",
    ]
    assert enumerate_actions(cat, 2) == enumerate_actions(list(cat.models), 2)


def test_holdout_excluded_by_default() -> None:
    cat = Catalog((mp("a", 1), mp("h", 2, holdout=True)))
    default = enumerate_actions(cat)
    assert all(s.model != "h" for a in default for s in a.chain())
    with_holdout = enumerate_actions(cat, include_holdout=True)
    assert any(s.model == "h" for a in with_holdout for s in a.chain())


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_max_depth_validated(bad: object) -> None:
    with pytest.raises(ValueError, match="max_depth"):
        enumerate_actions(Catalog((mp("a", 1),)), bad)  # type: ignore[arg-type]


def test_empty_catalog() -> None:
    assert enumerate_actions(Catalog(()), 3) == ()


@st.composite
def catalogs(draw: st.DrawFn) -> Catalog:
    n = draw(st.integers(1, 5))
    models = []
    for i in range(n):
        efforts = draw(st.lists(st.sampled_from(EFFORTS), min_size=1, max_size=4, unique=True))
        models.append(
            mp(
                f"m{i}",
                draw(st.integers(0, 3)),  # small range forces tier ties
                tuple(efforts),
                local=draw(st.booleans()),
                tools=draw(st.booleans()),
                holdout=draw(st.booleans()),
                context_window=draw(st.sampled_from([1_000, 50_000, 200_000])),
            )
        )
    return Catalog(tuple(models))


@settings(max_examples=60, deadline=None)
@given(catalogs(), st.integers(1, 3), st.booleans())
def test_property_enumeration(cat: Catalog, max_depth: int, holdout: bool) -> None:
    acts = enumerate_actions(cat, max_depth, include_holdout=holdout)
    assert len(set(acts)) == len(acts)
    pool = [m for m in cat.models if holdout or not m.holdout]
    for a in acts:
        chain = a.chain()
        assert 1 <= len(chain) <= max_depth
        for prev, nxt in zip(chain, chain[1:], strict=False):
            if prev.model == nxt.model:
                assert effort_rank(nxt.effort) > effort_rank(prev.effort)
            else:
                assert cat.get(nxt.model).tier > cat.get(prev.model).tier
        for step in chain:
            assert step.effort in cat.get(step.model).efforts
            assert holdout or not cat.get(step.model).holdout
    singles = [a for a in acts if a.escalate_to is None]
    assert sorted((a.model, a.effort) for a in singles) == sorted(
        (m.id, e) for m in pool for e in m.efforts
    )


@settings(max_examples=60, deadline=None)
@given(catalogs(), st.booleans())
def test_property_sensitive_never_cloud(cat: Catalog, include_holdout: bool) -> None:
    acts = enumerate_actions(cat, 3, include_holdout=True)
    sensitive = state(sensitive=True)
    try:
        kept = filter_feasible(sensitive, acts, cat, include_holdout=include_holdout)
    except NoFeasibleAction:
        kept = ()
    for a in kept:
        assert all(cat.get(s.model).local for s in a.chain())
    # allow cannot re-admit a cloud model
    cloud = [m.id for m in cat.models if not m.local]
    if cloud:
        try:
            kept = filter_feasible(sensitive, acts, cat, allow=cloud, include_holdout=True)
        except NoFeasibleAction:
            kept = ()
        assert kept == ()


def test_sensitive_default_catalog_no_cloud_at_any_depth() -> None:
    cat = load_default_catalog(warn_unverified=False)
    acts = enumerate_actions(cat, 3)
    kept = filter_feasible(state(sensitive=True), acts, cat)
    assert kept
    assert all(cat.get(s.model).local for a in kept for s in a.chain())
    # a cloud model appearing only as the fallback is removed too
    local_then_cloud = Action("qwen3.5:9b", "medium", Action("claude-opus-5-5", "high"))
    assert local_then_cloud in acts
    assert local_then_cloud not in kept
    with pytest.raises(NoFeasibleAction):
        filter_feasible(state(sensitive=True), [local_then_cloud], cat)


def test_allow_cannot_readmit_cloud_when_sensitive() -> None:
    cat = load_default_catalog(warn_unverified=False)
    acts = enumerate_actions(cat, 2)
    with pytest.raises(NoFeasibleAction) as exc:
        filter_feasible(state(sensitive=True), acts, cat, allow=["claude-opus-5-5"])
    assert exc.value.removed["sensitive"] > 0


def test_filter_constraints_and_order() -> None:
    cat = Catalog(
        (
            mp("small", 1, local=True, context_window=2_000, tools=False),
            mp("mid", 2, local=True),
            mp("big", 3),
            mp("held", 4, holdout=True),
        )
    )
    acts = enumerate_actions(cat, 2, include_holdout=True)

    def models(kept: tuple[Action, ...]) -> set[str]:
        return {s.model for a in kept for s in a.chain()}

    everyone = {"small", "mid", "big", "held"}
    assert models(filter_feasible(state(), acts, cat)) == everyone - {"held"}
    assert models(filter_feasible(state(), acts, cat, include_holdout=True)) == everyone
    assert "small" not in models(filter_feasible(state(prompt_tokens=5_000), acts, cat))
    assert "small" in models(filter_feasible(state(prompt_tokens=2_000), acts, cat))
    assert "small" not in models(filter_feasible(state(tools=["bash"]), acts, cat))
    assert models(filter_feasible(state(), acts, cat, allow=["mid", "big"])) == {"mid", "big"}
    assert models(filter_feasible(state(), acts, cat, deny=["big"])) == {"small", "mid"}
    both = filter_feasible(state(), acts, cat, allow=["mid", "big"], deny=["big"])
    assert models(both) == {"mid"}
    kept = filter_feasible(state(), acts, cat)
    assert list(kept) == [a for a in acts if a in kept]  # input order preserved


def test_no_feasible_action_reports_counts_without_prompt() -> None:
    cat = Catalog((mp("a", 1), mp("b", 2)))
    acts = enumerate_actions(cat, 2)
    secret = "TOP SECRET PROMPT"
    with pytest.raises(NoFeasibleAction) as exc:
        filter_feasible(state(sensitive=True, task_text=secret), acts, cat)
    err = exc.value
    assert err.candidates == len(acts) == err.removed["sensitive"]
    assert f"all {len(acts)} candidates removed: {len(acts)} sensitive" in str(err)
    assert secret not in str(err)


def test_empty_input_raises() -> None:
    with pytest.raises(NoFeasibleAction, match="no candidates"):
        filter_feasible(state(), [], Catalog((mp("a", 1),)))


def test_programming_errors_raise_value_error() -> None:
    cat = Catalog((mp("a", 1, ("low",)),))
    with pytest.raises(ValueError, match="unknown model 'zzz'"):
        filter_feasible(state(), [Action("zzz", "low")], cat)
    with pytest.raises(ValueError, match="does not list effort 'high'"):
        filter_feasible(state(), [Action("a", "high")], cat)
    # also when the bad step is a fallback
    with pytest.raises(ValueError, match="unknown model 'zzz'"):
        filter_feasible(state(), [Action("a", "low", Action("zzz", "high"))], cat)
    with pytest.raises(ValueError, match="allow list"):
        filter_feasible(state(), [Action("a", "low")], cat, allow=["nope"])
    with pytest.raises(ValueError, match="deny list"):
        filter_feasible(state(), [Action("a", "low")], cat, deny=["nope"])
