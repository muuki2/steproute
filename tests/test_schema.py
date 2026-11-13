"""Trace schema: round-trips, forward compatibility, JSON Schema drift, doc examples."""

import dataclasses
import json
import re
import warnings
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from hypothesis import given
from hypothesis import strategies as st

from steproute.core import (
    EFFORTS,
    SCHEMA_VERSION,
    Action,
    Decision,
    ModelProfile,
    Outcome,
    PriceTier,
    StepState,
    UnknownFieldWarning,
    effort_rank,
    new_decision_id,
)
from steproute.core.schema_gen import (
    REGENERATE_CMD,
    SCHEMA_PATH,
    generate_json_schema_text,
    load_json_schema,
)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_MD = ROOT / "docs" / "SCHEMA.md"

# --- strategies -----------------------------------------------------------------------------

text = st.text(max_size=20)
finite = st.floats(allow_nan=False, allow_infinity=False)
ints = st.integers(min_value=0, max_value=2**40)
json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | finite | text,
    lambda children: st.lists(children, max_size=3) | st.dictionaries(text, children, max_size=3),
    max_leaves=8,
)
json_dicts = st.dictionaries(text, json_values, max_size=3)

actions = st.recursive(
    st.builds(Action, model=text, effort=st.sampled_from(EFFORTS)),
    lambda inner: st.builds(
        Action, model=text, effort=st.sampled_from(EFFORTS), escalate_to=st.one_of(st.none(), inner)
    ),
    max_leaves=4,
)
profiles = st.builds(
    ModelProfile,
    id=text,
    provider=text,
    price_in=finite,
    price_out=finite,
    context_window=ints,
    local=st.booleans(),
    cache_write_mult=finite,
    cache_read_mult=finite,
    efforts=st.lists(st.sampled_from(EFFORTS), max_size=6).map(tuple),
    effort_output_mult=st.dictionaries(text, finite, max_size=3),
    meta=json_dicts,
    tier=ints,
    tools=st.booleans(),
    holdout=st.booleans(),
    verified=st.booleans(),
    price_tiers=st.lists(
        st.builds(PriceTier, above_tokens=ints, price_in=finite, price_out=finite), max_size=2
    ).map(tuple),
    effort_map=st.dictionaries(text, text, max_size=3),
    effort_change_keeps_cache=st.booleans(),
    cache_min_tokens=ints,
    latency_base_ms=st.none() | finite,
    latency_switch_ms=st.none() | finite,
)
states = st.builds(
    StepState,
    harness=text,
    session_id=text,
    step_idx=ints,
    kind=text,
    prompt_tokens=ints,
    task_id=st.none() | text,
    cached_prefix_tokens=ints,
    current_model=st.none() | text,
    current_effort=st.none() | st.sampled_from(EFFORTS),
    failed_attempts=st.lists(actions, max_size=2),
    tools=st.lists(text, max_size=3),
    sensitive=st.booleans(),
    source=st.sampled_from(["replay", "personal"]),
    task_text=st.none() | text,
    features=json_dicts,
)
decisions = st.builds(
    Decision,
    decision_id=text,
    ts=finite,
    state=states,
    candidates=st.lists(actions, max_size=3),
    action=actions,
    policy=text,
    propensity=finite,
    predicted_success=st.none() | finite,
    predicted_cost=st.none() | finite,
)
outcomes = st.builds(
    Outcome,
    decision_id=text,
    ts=finite,
    success=st.none() | st.booleans(),
    signals=json_dicts,
    final_model=st.none() | text,
    escalated=st.booleans(),
    input_tokens=ints,
    output_tokens=ints,
    cache_read_tokens=ints,
    cache_write_tokens=ints,
    cost_usd=st.none() | finite,
    latency_ms=st.none() | finite,
)

ALL = [
    (Action, actions),
    (ModelProfile, profiles),
    (StepState, states),
    (Decision, decisions),
    (Outcome, outcomes),
]


def _roundtrip(obj: Any) -> None:
    d = obj.to_dict()
    assert type(obj).from_dict(d) == obj
    assert type(obj).from_dict(json.loads(json.dumps(d))) == obj


@pytest.mark.parametrize(("cls", "strategy"), ALL, ids=lambda x: getattr(x, "__name__", ""))
def test_property_roundtrip(cls: type, strategy: st.SearchStrategy[Any]) -> None:
    @given(strategy)
    def check(obj: Any) -> None:
        _roundtrip(obj)

    check()


# --- hand-written cases ---------------------------------------------------------------------


def cascade3() -> Action:
    c = Action("claude-opus-4-1", "max")
    b = Action("claude-sonnet-4-5", "high", c)
    return Action("ollama/devstral:24b@q4", "low", b)


def make_decision() -> Decision:
    a = cascade3()
    return Decision(
        decision_id=new_decision_id(),
        ts=1767225600.5,
        state=StepState("proxy", "s1", 3, "main", 1200, tools=["Bash"], features={"x": [1, 2]}),
        candidates=[Action("m"), a],
        action=a,
        policy="rule",
        propensity=0.5,
        predicted_success=0.7,
    )


def test_roundtrip_each_type() -> None:
    _roundtrip(cascade3())
    _roundtrip(ModelProfile("m", "p", 3.0, 15.0, 200_000, efforts=("low", "high")))
    _roundtrip(StepState("h", "s", 0, "plan", 10, task_text="hi"))
    _roundtrip(make_decision())
    _roundtrip(Outcome("d", 1.0, None))
    _roundtrip(Outcome("d", 1.0, True, cost_usd=0.0, latency_ms=12.5, signals={"a": None}))


def test_three_level_cascade() -> None:
    a = cascade3()
    assert a.escalate_to is not None and a.escalate_to.escalate_to is not None
    assert a.escalate_to.escalate_to.escalate_to is None
    _roundtrip(a)
    assert Action.from_dict(a.to_dict()).chain() == a.chain()


def test_to_dict_emits_every_field_json_native() -> None:
    p = ModelProfile("m", "p", 1.0, 2.0, 10, efforts=("low", "high"))
    d = p.to_dict()
    assert set(d) == {f.name for f in dataclasses.fields(ModelProfile)}
    assert d["efforts"] == ["low", "high"]
    o = Outcome("d", 1.0, None).to_dict()
    assert o["success"] is None and o["cost_usd"] is None
    assert Action("m").to_dict() == {"model": "m", "effort": "medium", "escalate_to": None}


def test_action_chain_label_hash() -> None:
    a = cascade3()
    assert [s.model for s in a.chain()] == [
        "ollama/devstral:24b@q4",
        "claude-sonnet-4-5",
        "claude-opus-4-1",
    ]
    assert all(s.escalate_to is None for s in a.chain())
    assert [s.effort for s in a.chain()] == ["low", "high", "max"]
    assert a.label() == (
        "ollama/devstral:24b@q4 (low) -> claude-sonnet-4-5 (high) -> claude-opus-4-1 (max)"
    )
    assert Action("m").label() == "m (medium)"
    assert len({a, cascade3(), Action("m")}) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.model = "x"  # type: ignore[misc]


def test_effort_vocabulary() -> None:
    assert EFFORTS == ("none", "low", "medium", "high", "xhigh", "max")
    assert effort_rank("none") < effort_rank("high") < effort_rank("xhigh") < effort_rank("max")
    with pytest.raises(ValueError):
        effort_rank("ultra")
    with pytest.raises(ValueError):
        Action("m", "ultra")
    with pytest.raises(ValueError):
        Action.from_dict({"model": "m", "effort": "ultra"})


def test_source_restricted() -> None:
    assert StepState("h", "s", 0, "main", 1).source == "personal"
    with pytest.raises(ValueError):
        StepState("h", "s", 0, "main", 1, source="other")
    with pytest.raises(ValueError):
        StepState.from_dict({**StepState("h", "s", 0, "main", 1).to_dict(), "source": "x"})


def test_new_decision_id_unique() -> None:
    assert new_decision_id() != new_decision_id()


# --- forward compatibility and errors -------------------------------------------------------


def test_unknown_keys_warn_and_are_ignored() -> None:
    d = make_decision().to_dict()
    d["future"] = 1
    d["state"]["future_state"] = 2
    d["action"]["future_action"] = 3
    with pytest.warns(UnknownFieldWarning) as rec:
        got = Decision.from_dict(d)
    assert len(rec) == 3
    assert issubclass(UnknownFieldWarning, UserWarning)
    assert not hasattr(got, "future")


def test_known_keys_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Decision.from_dict(make_decision().to_dict())


def test_missing_required_is_error() -> None:
    with pytest.raises(ValueError, match="model"):
        Action.from_dict({})
    d = make_decision().to_dict()
    del d["propensity"]
    with pytest.raises(ValueError, match="propensity"):
        Decision.from_dict(d)


def test_defaults_applied_when_optional_keys_missing() -> None:
    o = Outcome.from_dict({"decision_id": "d", "ts": 1, "success": True})
    assert o == Outcome("d", 1.0, True)


def test_wrong_types_rejected() -> None:
    with pytest.raises(TypeError):
        Outcome.from_dict({"decision_id": "d", "ts": 1, "success": "yes"})
    with pytest.raises(TypeError):
        Outcome.from_dict({"decision_id": "d", "ts": 1, "success": True, "input_tokens": True})


def test_non_json_native_values_rejected() -> None:
    with pytest.raises(TypeError):
        Outcome("d", 1.0, True, signals={"x": object()}).to_dict()


# --- JSON Schema ----------------------------------------------------------------------------


def test_json_schema_is_up_to_date() -> None:
    current = SCHEMA_PATH.read_text(encoding="utf-8")
    assert current == generate_json_schema_text(), (
        f"{SCHEMA_PATH.name} drifted from the dataclasses in steproute.core.schema. "
        f"Regenerate it with: {REGENERATE_CMD}"
    )


def test_json_schema_shape() -> None:
    schema = load_json_schema()
    assert set(schema["$defs"]) == {
        "Action",
        "PriceTier",
        "ModelProfile",
        "StepState",
        "Decision",
        "Outcome",
    }
    esc = schema["$defs"]["Action"]["properties"]["escalate_to"]
    assert {"$ref": "#/$defs/Action"} in esc["anyOf"]
    for d in schema["$defs"].values():
        assert d.get("additionalProperties", True) is True
    jsonschema.Draft202012Validator.check_schema(schema)


def _validator(name: str) -> jsonschema.Draft202012Validator:
    schema = load_json_schema()
    return jsonschema.Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": schema["$defs"]})


@pytest.mark.parametrize(("cls", "strategy"), ALL, ids=lambda x: getattr(x, "__name__", ""))
def test_to_dict_validates_against_schema(cls: type, strategy: st.SearchStrategy[Any]) -> None:
    validator = _validator(cls.__name__)

    @given(strategy)
    def check(obj: Any) -> None:
        validator.validate(json.loads(json.dumps(obj.to_dict())))

    check()


def test_schema_rejects_bad_records() -> None:
    v = _validator("Action")
    assert not v.is_valid({"model": "m", "effort": "ultra"})
    assert not v.is_valid({"effort": "low"})
    assert v.is_valid({"model": "m", "unknown": 1})


# --- docs/SCHEMA.md -------------------------------------------------------------------------


def _doc_examples() -> dict[str, Any]:
    md = SCHEMA_MD.read_text(encoding="utf-8")
    found = re.findall(r"```json example=(\w+)\n(.*?)\n```", md, flags=re.S)
    return {name: json.loads(body) for name, body in found}


def test_doc_examples_validate_and_roundtrip() -> None:
    ex = _doc_examples()
    assert set(ex) == {"decision", "outcome"}
    _validator("Decision").validate(ex["decision"])
    _validator("Outcome").validate(ex["outcome"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dec = Decision.from_dict(ex["decision"])
        out = Outcome.from_dict(ex["outcome"])
    assert dec.to_dict() == ex["decision"]
    assert out.to_dict() == ex["outcome"]
    assert dec.decision_id == out.decision_id


@pytest.mark.parametrize("cls", [Action, PriceTier, ModelProfile, StepState, Decision, Outcome])
def test_doc_documents_every_field(cls: type) -> None:
    md = SCHEMA_MD.read_text(encoding="utf-8")
    section = re.search(rf"^## {cls.__name__}\n(.*?)(?=^## )", md, flags=re.S | re.M)
    assert section, f"no '## {cls.__name__}' section in docs/SCHEMA.md"
    for f in dataclasses.fields(cls):  # type: ignore[arg-type]
        assert f"| `{f.name}` |" in section.group(1), f"{cls.__name__}.{f.name} undocumented"


def test_doc_mentions_version() -> None:
    assert f"`{SCHEMA_VERSION}`" in SCHEMA_MD.read_text(encoding="utf-8")
