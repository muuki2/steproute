"""Trace schema: the record types shared by every collector, store and policy.

Plain stdlib dataclasses with explicit ``to_dict`` / ``from_dict``. Serialisation is driven by
the dataclass fields and their type annotations, so adding a defaulted field to a type needs
no other change: ``to_dict``, ``from_dict`` and the JSON Schema generator all pick it up.

Units: money is USD, prices are per million tokens, time is milliseconds, timestamps are Unix
seconds (float).
"""

from __future__ import annotations

import dataclasses
import types
import typing
import uuid
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

__all__ = [
    "EFFORTS",
    "SCHEMA_VERSION",
    "SOURCES",
    "Action",
    "CandidateSet",
    "Decision",
    "ModelProfile",
    "Outcome",
    "PriceTier",
    "StepState",
    "UnknownFieldWarning",
    "effort_rank",
    "new_decision_id",
]

SCHEMA_VERSION = "0.1"

#: Effort levels, weakest to strongest. Providers without effort levels use ``medium`` only.
EFFORTS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh", "max")

#: Allowed values of ``StepState.source``.
SOURCES: tuple[str, ...] = ("replay", "personal")


class UnknownFieldWarning(UserWarning):
    """A record contained a key this schema version does not know; the key was ignored."""


def effort_rank(effort: str) -> int:
    """Position of ``effort`` in ``EFFORTS`` (0 is weakest). Raises ``ValueError`` if unknown."""
    try:
        return EFFORTS.index(effort)
    except ValueError:
        raise ValueError(f"unknown effort {effort!r}; expected one of {EFFORTS}") from None


def new_decision_id() -> str:
    """A fresh, globally unique decision id (32 hex characters)."""
    return uuid.uuid4().hex


# --- generic field-driven codec -------------------------------------------------------------


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, _Record):
        return value.to_dict()
    if isinstance(value, list | tuple):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"dict keys must be str, got {k!r}")
            out[k] = _encode(v)
        return out
    raise TypeError(f"value of type {type(value).__name__} is not JSON-native: {value!r}")


def _decode(hint: Any, value: Any, where: str) -> Any:
    if hint is Any:
        return value
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _decode(inner, value, where)
    if origin is list or origin is tuple:
        if not isinstance(value, list | tuple):
            raise TypeError(f"{where}: expected a list, got {type(value).__name__}")
        item = args[0]
        items = [_decode(item, v, f"{where}[{i}]") for i, v in enumerate(value)]
        return items if origin is list else tuple(items)
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected an object, got {type(value).__name__}")
        return {str(k): _decode(args[1], v, f"{where}.{k}") for k, v in value.items()}
    if isinstance(hint, type) and issubclass(hint, _Record):
        return hint.from_dict(value)
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected a boolean, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported annotation {hint!r}")  # pragma: no cover


_HINTS: dict[str, dict[str, Any]] = {}


def field_hints(cls: type[Any]) -> dict[str, Any]:
    """Resolved type annotations of a record class, keyed by field name."""
    if cls.__qualname__ not in _HINTS:
        _HINTS[cls.__qualname__] = typing.get_type_hints(cls)
    return _HINTS[cls.__qualname__]


class _Record:
    """Mixin giving a dataclass explicit, field-driven ``to_dict`` / ``from_dict``."""

    def to_dict(self) -> dict[str, Any]:
        """Every field, JSON-native (``None`` as null, tuples as lists)."""
        obj = cast("DataclassInstance", self)
        return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Inverse of ``to_dict``.

        Unknown keys are ignored with an ``UnknownFieldWarning``; missing required keys raise
        ``ValueError``.
        """
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__}: expected an object, got {type(data).__name__}")
        flds = dataclasses.fields(cast("type[DataclassInstance]", cls))
        known = {f.name for f in flds}
        unknown = sorted(set(data) - known)
        if unknown:
            warnings.warn(
                f"{cls.__name__}: ignoring unknown field(s) {unknown} "
                f"(schema version {SCHEMA_VERSION})",
                UnknownFieldWarning,
                stacklevel=2,
            )
        hints = field_hints(cls)
        missing = [
            f.name
            for f in flds
            if f.name not in data
            and f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        ]
        if missing:
            raise ValueError(f"{cls.__name__}: missing required field(s) {missing}")
        kwargs = {
            name: _decode(hints[name], data[name], f"{cls.__name__}.{name}")
            for name in known
            if name in data
        }
        return cls(**kwargs)


# --- record types ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action(_Record):
    """A routing action: a model, an effort level, and an optional fallback tried on failure."""

    model: str
    effort: str = field(default="medium", metadata={"enum": EFFORTS})
    escalate_to: Action | None = None

    def __post_init__(self) -> None:
        effort_rank(self.effort)

    def chain(self) -> tuple[Action, ...]:
        """The single-step actions in order, each with ``escalate_to=None``."""
        steps: list[Action] = []
        node: Action | None = self
        while node is not None:
            steps.append(Action(node.model, node.effort))
            node = node.escalate_to
        return tuple(steps)

    def label(self) -> str:
        """Compact display string, e.g. ``a (high) -> b (low)``.

        For display and grouping only; not meant to be parsed back (model ids may contain
        ``:`` ``.`` ``@`` ``/``).
        """
        return " -> ".join(f"{a.model} ({a.effort})" for a in self.chain())


@dataclass(frozen=True)
class PriceTier(_Record):
    """Prices that apply to a request whose prompt is longer than ``above_tokens``."""

    above_tokens: int
    price_in: float
    price_out: float


@dataclass
class ModelProfile(_Record):
    """What the router knows about one model. Prices are USD per million tokens."""

    id: str
    provider: str
    price_in: float
    price_out: float
    context_window: int
    local: bool = False
    cache_write_mult: float = 1.0
    cache_read_mult: float = 1.0
    efforts: tuple[str, ...] = ("medium",)
    effort_output_mult: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    tier: int = 0
    tools: bool = True
    holdout: bool = False
    verified: bool = False
    price_tiers: tuple[PriceTier, ...] = ()
    effort_map: dict[str, str] = field(default_factory=dict)
    effort_change_keeps_cache: bool = False
    cache_min_tokens: int = 0
    latency_base_ms: float | None = None
    latency_ms_per_input_token: float | None = None
    latency_ms_per_output_token: float | None = None
    latency_switch_ms: float | None = None

    def prices_for(self, prompt_tokens: int) -> tuple[float, float]:
        """``(price_in, price_out)`` in force for a request with ``prompt_tokens`` of prompt.

        ``prompt_tokens`` counts every input token, cache reads and writes included. The base
        prices apply up to and including the first tier's ``above_tokens``; a request above a
        threshold pays that tier's rate on the whole request.
        """
        price_in, price_out = self.price_in, self.price_out
        for t in self.price_tiers:
            if prompt_tokens > t.above_tokens:
                price_in, price_out = t.price_in, t.price_out
        return price_in, price_out


@dataclass
class StepState(_Record):
    """Harness-neutral description of one routable step."""

    harness: str
    session_id: str
    step_idx: int
    kind: str
    prompt_tokens: int
    task_id: str | None = None
    cached_prefix_tokens: int = 0
    current_model: str | None = None
    current_effort: str | None = field(default=None, metadata={"enum": (None, *EFFORTS)})
    failed_attempts: list[Action] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    sensitive: bool = False
    source: str = field(default="personal", metadata={"enum": SOURCES})
    task_text: str | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.current_effort is not None:
            effort_rank(self.current_effort)
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")


@dataclass(frozen=True)
class CandidateSet(_Record):
    """A compact, verifiable description of a decision's candidate set.

    The candidates are ``filter_feasible(state, enumerate_actions(catalog, max_depth,
    include_holdout=...), catalog, allow=..., deny=..., include_holdout=...)`` for the
    decision's own state. ``fingerprint`` hashes that definition (the catalog fields that
    shape the set, plus the parameters below), so a reader can rebuild the list from the state
    and a catalog and check it is the same one; ``count`` is the list's length. See
    ``docs/POLICY.md`` and ``steproute.core.policy.rebuild_candidates``.
    """

    fingerprint: str
    count: int
    max_depth: int
    include_holdout: bool = False
    allow: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()


@dataclass
class Decision(_Record):
    """A routing decision, logged before the step runs.

    ``candidates`` may be empty when ``candidate_set`` is set: the explicit list was left out
    to keep the record small and can be rebuilt with ``rebuild_candidates``.
    """

    decision_id: str
    ts: float
    state: StepState
    candidates: list[Action]
    action: Action
    policy: str
    propensity: float
    predicted_success: float | None = None
    predicted_cost: float | None = None
    candidate_set: CandidateSet | None = None


@dataclass
class Outcome(_Record):
    """What happened after a decision, joined to it by ``decision_id``."""

    decision_id: str
    ts: float
    success: bool | None
    signals: dict[str, Any] = field(default_factory=dict)
    final_model: str | None = None
    escalated: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float | None = None


#: Every record type, in definition order (the JSON Schema ``$defs`` follow this order).
RECORD_TYPES: tuple[type[_Record], ...] = (
    Action,
    PriceTier,
    ModelProfile,
    StepState,
    CandidateSet,
    Decision,
    Outcome,
)
