"""Target policies: what the evaluator scores. Standard library only.

Every policy is reduced to one question: given a state, what probability does it give each
action? ``ScoredPolicy`` answers it with ``action_probabilities``; a policy with only
``decide`` (``RulePolicy``, ``FixedPolicy``) is treated as deterministic. An empty
distribution means the policy cannot act in that state (``NoFeasibleAction``).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from steproute.core.actions import NoFeasibleAction
from steproute.core.catalog import Catalog
from steproute.core.rules import RulePolicy, load_rule_config
from steproute.core.schema import EFFORTS, Action, StepState

__all__ = [
    "Distribution",
    "DistributionTarget",
    "FixedTarget",
    "OracleTarget",
    "PolicySpecError",
    "TargetPolicy",
    "as_target",
    "parse_action",
    "parse_policy_spec",
]

Distribution = tuple[tuple[Action, float], ...]


class PolicySpecError(ValueError):
    """A ``--policy`` specification cannot be parsed or built."""


@runtime_checkable
class TargetPolicy(Protocol):
    """A policy as the evaluator sees it."""

    name: str
    #: True only for the hindsight oracle, which looks at outcomes and is not deployable.
    hindsight: bool

    def distribution(self, state: StepState) -> Distribution: ...


def _normalise(dist: Sequence[tuple[Action, float]], name: str) -> Distribution:
    merged: dict[Action, float] = {}
    for action, p in dist:
        if not isinstance(action, Action):
            raise TypeError(f"policy {name!r} returned {action!r}, not an Action")
        if isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p) or p < 0:
            raise ValueError(f"policy {name!r} gave {action.label()} probability {p!r}")
        if p > 0:
            merged[action] = merged.get(action, 0.0) + float(p)
    total = sum(merged.values())
    if merged and abs(total - 1.0) > 1e-6:
        raise ValueError(f"policy {name!r}: probabilities sum to {total}, not 1")
    return tuple(merged.items())


@dataclass
class FixedTarget:
    """Always ``action``, in every state. Feasibility is not checked."""

    action: Action
    name: str = ""
    hindsight: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"fixed:{self.action.label()}"

    def distribution(self, state: StepState) -> Distribution:
        return ((self.action, 1.0),)


@dataclass
class DistributionTarget:
    """A target from a function ``state -> [(action, probability), ...]``."""

    fn: Callable[[StepState], Sequence[tuple[Action, float]]]
    name: str
    hindsight: bool = False

    def distribution(self, state: StepState) -> Distribution:
        try:
            dist = self.fn(state)
        except NoFeasibleAction:
            return ()
        return _normalise(dist, self.name)


@dataclass
class OracleTarget:
    """The hindsight oracle (grid only): per task, the single configuration with the highest
    success, ties to the cheapest. It reads outcomes, so it is an upper reference, never a
    policy; the grid evaluator implements it directly."""

    name: str = "oracle"
    hindsight: bool = True

    def distribution(self, state: StepState) -> Distribution:
        raise TypeError("the oracle has no state-only distribution; it is evaluated on a grid")


def as_target(policy: Any, name: str | None = None) -> TargetPolicy:
    """Wrap a policy object as a ``TargetPolicy``.

    - Already a ``TargetPolicy``: returned as is.
    - Has ``action_probabilities(state)`` (``ScoredPolicy``): that distribution is used.
    - Has ``decide(state)``: treated as **deterministic**, probability 1 on
      ``decide(state).action``. A policy that randomises inside ``decide`` must expose
      ``action_probabilities`` instead, or the evaluation is wrong.
    - A plain callable ``state -> [(action, p), ...]``.
    """
    if isinstance(policy, TargetPolicy):
        return policy
    label = name or str(getattr(policy, "name", None) or type(policy).__name__)
    probs = getattr(policy, "action_probabilities", None)
    if callable(probs):
        return DistributionTarget(lambda s: probs(s), label)
    decide = getattr(policy, "decide", None)
    if callable(decide):
        return DistributionTarget(lambda s: ((decide(s).action, 1.0),), label)
    if callable(policy):
        return DistributionTarget(policy, label)
    raise TypeError(f"{policy!r} is not a policy: no action_probabilities, decide or __call__")


def parse_action(text: str) -> Action:
    """``EFFORT:MODEL`` steps joined by ``,`` into a single action or cascade.

    The effort comes first because model ids may contain ``:`` (Ollama tags); commas do not
    occur in model ids. Example: ``low:qwen3.5:9b,medium:claude-sonnet-5-5``.
    """
    steps: list[tuple[str, str]] = []
    for part in text.split(","):
        effort, sep, model = part.strip().partition(":")
        if not sep or not model:
            raise PolicySpecError(f"bad step {part!r}: expected EFFORT:MODEL")
        if effort not in EFFORTS:
            raise PolicySpecError(f"bad step {part!r}: unknown effort {effort!r}")
        steps.append((model, effort))
    action: Action | None = None
    for model, effort in reversed(steps):
        action = Action(model, effort, action)
    assert action is not None
    return action


def parse_policy_spec(spec: str, catalog: Callable[[], Catalog]) -> TargetPolicy:
    """Build a target from a CLI specification.

    - ``fixed:EFFORT:MODEL`` or ``cascade:EFFORT:MODEL,EFFORT:MODEL[,...]``
    - ``rules`` (the shipped rule config) or ``rules:PATH`` (a rules TOML)
    - ``oracle`` (grid only; hindsight reference, not a policy)
    - ``python:MODULE:ATTR``: ``ATTR`` is a policy object, or a factory called with the
      catalog that returns one.

    ``catalog`` is called only when a catalog is needed.
    """
    kind, _, rest = spec.partition(":")
    if kind in ("fixed", "cascade"):
        action = parse_action(rest)
        if kind == "fixed" and action.escalate_to is not None:
            raise PolicySpecError("fixed: takes one EFFORT:MODEL step; use cascade: for more")
        return FixedTarget(action, name=spec)
    if kind == "rules":
        config = load_rule_config(Path(rest)) if rest else load_rule_config()
        return as_target(RulePolicy(catalog(), config), name=spec)
    if kind == "oracle" and not rest:
        return OracleTarget()
    if kind == "python":
        module, sep, attr = rest.rpartition(":")
        if not sep or not module or not attr:
            raise PolicySpecError("python: takes MODULE:ATTR")
        try:
            obj = getattr(importlib.import_module(module), attr)
        except (ImportError, AttributeError) as exc:
            raise PolicySpecError(f"cannot import {rest!r}: {exc}") from None
        is_policy = any(callable(getattr(obj, m, None)) for m in ("action_probabilities", "decide"))
        if not is_policy and not isinstance(obj, TargetPolicy) and callable(obj):
            obj = obj(catalog())
        return as_target(obj, name=spec)
    raise PolicySpecError(
        f"unknown policy {spec!r}; expected fixed:, cascade:, rules[:PATH], oracle or python:"
    )
