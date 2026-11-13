"""RulePolicy: table of states, config-only changes, property checks, shipped config."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steproute.core import (
    Action,
    ModelProfile,
    NoFeasibleAction,
    Policy,
    RuleConfig,
    RuleConfigError,
    RulePolicy,
    StepState,
    filter_feasible,
    load_rule_config,
)
from steproute.core.catalog import Catalog, load_default_catalog
from steproute.core.schema import effort_rank

DEFAULT = load_default_catalog(warn_unverified=False)


def chain(*steps: tuple[str, str]) -> Action:
    action: Action | None = None
    for model, effort in reversed(steps):
        action = Action(model, effort, action)
    assert action is not None
    return action


def state(**kw: Any) -> StepState:
    base: dict[str, Any] = {
        "harness": "t",
        "session_id": "s",
        "step_idx": 1,
        "kind": "main",
        "prompt_tokens": 1000,
    }
    base.update(kw)
    return StepState(**base)


SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"
OPUS, FABLE = "claude-opus-5-5", "claude-fable-5-2"
V4, QCODER, GPT = "deepseek-v4-pro", "qwen3-coder:30b", "gpt-oss:20b"


@dataclass
class Case:
    id: str
    state: StepState
    expected: Action
    config: dict[str, Any] = field(default_factory=dict)  # parsed-TOML style overrides
    rule: str = "default"
    fallback: bool = False


CASES = [
    Case("plain main step", state(), chain((SONNET, "medium"), (SONNET, "high"), (OPUS, "low"))),
    Case(
        "read-only subagent",
        state(kind="subagent", tools=["Read", "Grep"]),
        chain((HAIKU, "low"), (HAIKU, "medium"), (SONNET, "low")),
        rule="readonly_subagent",
    ),
    Case(
        "read-only tools match case-insensitively",
        state(kind="Subagent", tools=["GREP", "test_run"]),
        chain((HAIKU, "low"), (HAIKU, "medium"), (SONNET, "low")),
        rule="readonly_subagent",
    ),
    Case(
        "subagent with a write tool",
        state(kind="subagent", tools=["Read", "Edit"]),
        chain((SONNET, "medium"), (SONNET, "high"), (OPUS, "low")),
    ),
    Case(
        "subagent with no tools",
        state(kind="subagent"),
        chain((SONNET, "medium"), (SONNET, "high"), (OPUS, "low")),
    ),
    Case(
        "read-only tools on a main step",
        state(tools=["Read"]),
        chain((SONNET, "medium"), (SONNET, "high"), (OPUS, "low")),
    ),
    Case(
        "stay on a same-tier current model that is not the preferred one",
        state(current_model=V4),
        chain((V4, "low"), (V4, "high"), (OPUS, "low")),
    ),
    Case(
        "stay keeps current effort",
        state(current_model=V4, current_effort="high"),
        chain((V4, "high"), (V4, "max"), (OPUS, "low")),
    ),
    Case(
        "stay on the preferred model keeps its effort",
        state(current_model=SONNET, current_effort="high"),
        chain((SONNET, "high"), (SONNET, "xhigh"), (OPUS, "low")),
    ),
    Case(
        "current model on a different tier",
        state(current_model=HAIKU, current_effort="high"),
        chain((SONNET, "medium"), (SONNET, "high"), (OPUS, "low")),
    ),
    Case(
        "sensitive state falls back to a local tier",
        state(sensitive=True),
        chain((QCODER, "medium")),
        fallback=True,
    ),
    Case(
        "sensitive state is local at every rung",
        state(sensitive=True),
        chain((GPT, "medium"), (GPT, "high"), (QCODER, "medium")),
        config={"default": {"tier": 2}},
    ),
    Case(
        "prompt too large for the local tier",
        state(prompt_tokens=100_000),
        chain((HAIKU, "medium"), (HAIKU, "high"), (SONNET, "low")),
        config={"default": {"tier": 3}},
        fallback=True,
    ),
    Case(
        "holdout current model is never chosen",
        state(current_model="devstral:24b"),
        chain((QCODER, "medium"), (HAIKU, "low")),
        config={"default": {"tier": 3}},
    ),
    Case(
        "cascade rung skipped: no higher effort",
        state(current_model=SONNET, current_effort="max"),
        chain((SONNET, "max"), (OPUS, "low")),
    ),
    Case(
        "cascade rung skipped: no higher tier",
        state(),
        chain((FABLE, "medium"), (FABLE, "high")),
        config={"default": {"tier": 7}},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_table(case: Case) -> None:
    pol = RulePolicy(DEFAULT, load_rule_config(case.config))
    d = pol.decide(case.state)
    assert d.action == case.expected
    assert d.candidates == [d.action]
    assert d.propensity == 1.0
    assert d.policy == "rules"
    assert d.predicted_success is None and d.predicted_cost is None
    note = d.state.features["rules"]
    assert note["rule"] == case.rule
    assert note["fallback"] is case.fallback
    if d.state.sensitive:
        for step in d.action.chain():
            assert DEFAULT.get(step.model).local
    filter_feasible(d.state, [d.action], DEFAULT)


def test_table_has_enough_states() -> None:
    assert len(CASES) >= 12


def test_prompt_too_large_for_everything_raises() -> None:
    with pytest.raises(NoFeasibleAction):
        RulePolicy(DEFAULT).decide(state(prompt_tokens=5_000_000))
    with pytest.raises(NoFeasibleAction):  # sensitive: only local models, all too small
        RulePolicy(DEFAULT).decide(state(sensitive=True, prompt_tokens=100_000))


def test_tools_required_when_preferred_model_has_no_tools() -> None:
    def m(mid: str, tier: int, tools: bool) -> ModelProfile:
        return ModelProfile(mid, "p", 1.0, 1.0, 100_000, tier=tier, tools=tools)

    cat = Catalog((m("a", 5, False), m("b", 5, True), m("c", 6, True)))
    cfg = RuleConfig(provider_order=("p",))
    pol = RulePolicy(cat, cfg)
    assert pol.decide(state()).action.model == "a"  # catalog order, no tools needed
    d = pol.decide(state(tools=["Edit"]))
    assert d.action.model == "b" and not d.state.features["rules"]["fallback"]
    cat2 = Catalog((m("a", 5, False), m("c", 6, True)))
    d2 = RulePolicy(cat2, cfg).decide(state(tools=["Edit"]))
    assert d2.action.model == "c" and d2.state.features["rules"]["fallback"]


def test_marks_sensitive_text_before_routing() -> None:
    secret = "aws key AKIAIOSFODNN7EXAMPLE and password=hunter2hunter2"
    d = RulePolicy(DEFAULT).decide(state(task_text=secret))
    assert d.state.sensitive
    assert DEFAULT.get(d.action.model).local


def test_injectable_clock_and_ids_and_protocol() -> None:
    pol = RulePolicy(DEFAULT, clock=lambda: 12.5, new_id=lambda: "id-1", name="r")
    as_policy: Policy = pol  # structural check by mypy
    assert as_policy is pol
    d = pol.decide(state())
    assert (d.ts, d.decision_id, d.policy) == (12.5, "id-1", "r")
    pol.report(None)  # type: ignore[arg-type]


def test_state_features_are_not_mutated() -> None:
    s = state(features={"k": 1})
    d = RulePolicy(DEFAULT).decide(s)
    assert s.features == {"k": 1}
    assert d.state.features["k"] == 1


# --- config only ------------------------------------------------------------------------------


def test_config_changes_decisions_without_code_change(tmp_path: Any) -> None:
    sub = state(kind="subagent", tools=["Read"])
    plain = state()
    base = RulePolicy(DEFAULT)
    assert base.decide(plain).action.model == SONNET
    assert base.decide(sub).action.model == HAIKU

    path = tmp_path / "rules.toml"
    path.write_text(
        """
[default]
tier = 6
effort = "high"
[cheap]
tier = 5
read_only_tools = ["bash"]
[cascade]
enabled = false
""",
        encoding="utf-8",
    )
    pol = RulePolicy(DEFAULT, load_rule_config(path))
    assert pol.decide(plain).action == Action(OPUS, "high")
    # "Read" is no longer read-only, so the subagent gets the default rule
    assert pol.decide(sub).action == Action(OPUS, "high")
    assert pol.decide(state(kind="subagent", tools=["Bash"])).action == Action(SONNET, "low")


def test_config_errors() -> None:
    with pytest.raises(RuleConfigError, match="unknown key"):
        load_rule_config({"default": {"tierr": 5}})
    with pytest.raises(RuleConfigError, match="unknown section"):
        load_rule_config({"nope": {}})
    with pytest.raises(RuleConfigError):
        load_rule_config({"default": {"tier": "5"}})
    with pytest.raises(RuleConfigError, match="unknown effort"):
        load_rule_config({"default": {"effort": "turbo"}})
    with pytest.raises(RuleConfigError, match="provider_order"):
        RulePolicy(DEFAULT, load_rule_config({"models": {"provider_order": ["nobody"]}}))


# --- shipped config ---------------------------------------------------------------------------


def test_default_config_is_package_data_and_matches_dataclass_defaults() -> None:
    res = resources.files("steproute.catalogs").joinpath("rules.toml")
    assert res.is_file()
    assert load_rule_config() == RuleConfig()


def test_default_tiers_resolve_on_default_catalog() -> None:
    cfg = load_rule_config()
    assert {m.tier for m in DEFAULT if m.tier == cfg.default_tier} == {5}
    assert {m.tier for m in DEFAULT if m.tier == cfg.cheap_tier} == {4}


# --- properties -------------------------------------------------------------------------------

MODELS = [m.id for m in DEFAULT.models]
TOOLS = ["Read", "grep", "Edit", "Bash", "search", "pytest"]


@settings(max_examples=300, deadline=None)
@given(
    kind=st.sampled_from(["main", "subagent", "Subagent"]),
    tools=st.lists(st.sampled_from(TOOLS), max_size=3),
    sensitive=st.booleans(),
    tokens=st.sampled_from([0, 1000, 40_000, 2_000_000]),
    current=st.one_of(st.none(), st.sampled_from(MODELS)),
    effort=st.one_of(st.none(), st.sampled_from(["none", "low", "medium", "high", "max"])),
    default_tier=st.integers(0, 8),
    cascade=st.booleans(),
)
def test_every_decision_is_feasible_and_cascades_increase(
    kind: str,
    tools: list[str],
    sensitive: bool,
    tokens: int,
    current: str | None,
    effort: str | None,
    default_tier: int,
    cascade: bool,
) -> None:
    cfg = dataclasses.replace(RuleConfig(), default_tier=default_tier, cascade_enabled=cascade)
    pol = RulePolicy(DEFAULT, cfg)
    s = state(
        kind=kind,
        tools=tools,
        sensitive=sensitive,
        prompt_tokens=tokens,
        current_model=current,
        current_effort=effort,
    )
    try:
        d = pol.decide(s)
    except NoFeasibleAction:
        return
    assert filter_feasible(d.state, [d.action], DEFAULT) == (d.action,)
    steps = d.action.chain()
    assert not any(DEFAULT.get(x.model).holdout for x in steps)
    for a, b in zip(steps, steps[1:], strict=False):
        pa, pb = DEFAULT.get(a.model), DEFAULT.get(b.model)
        if a.model == b.model:
            assert effort_rank(b.effort) > effort_rank(a.effort)
        else:
            assert pb.tier > pa.tier
    assert d.state.sensitive or not sensitive
