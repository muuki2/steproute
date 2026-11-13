"""Rule baseline policy: no learning, deterministic. Standard library only. See ``docs/RULES.md``.

``RulePolicy`` encodes common model-selection guidance (a mid-tier model by default, the cheap
tier for read-only subagent work, an escalation ladder) as tiers, not model names. Every
threshold lives in a TOML config (``RuleConfig``); the shipped default is
``steproute/catalogs/rules.toml``. The policy is the floor learned policies must beat.
"""

from __future__ import annotations

import dataclasses
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from steproute.core.actions import filter_feasible
from steproute.core.catalog import Catalog
from steproute.core.privacy import PrivacyConfig, mark_before_routing
from steproute.core.schema import (
    EFFORTS,
    Action,
    Decision,
    ModelProfile,
    Outcome,
    StepState,
    effort_rank,
    new_decision_id,
)

__all__ = ["RULES_FEATURE", "RuleConfig", "RuleConfigError", "RulePolicy", "load_rule_config"]

#: Key under ``Decision.state.features`` where the policy records which rule fired.
RULES_FEATURE = "rules"


class RuleConfigError(ValueError):
    """The rules config is invalid (unknown key, wrong type, unknown effort or provider)."""


@dataclass(frozen=True)
class RuleConfig:
    """Every choice the rules make. Defaults here equal the shipped ``rules.toml``."""

    default_tier: int = 5
    default_effort: str = "medium"
    cheap_tier: int = 4
    cheap_effort: str = "low"
    subagent_kinds: tuple[str, ...] = ("subagent",)
    read_only_tools: tuple[str, ...] = (
        "search", "read", "grep", "glob", "ls", "list", "find", "cat",
        "websearch", "webfetch", "notebookread",
        "test", "tests", "run_tests", "test_run", "pytest",
    )  # fmt: skip
    stay_enabled: bool = True
    stay_keep_effort: bool = True
    cascade_enabled: bool = True
    cascade_effort_up: bool = True
    cascade_next_tier: bool = True
    cascade_next_tier_effort: str = "low"
    prefer_current: bool = True
    provider_order: tuple[str, ...] = ("anthropic", "deepseek", "ollama")

    def __post_init__(self) -> None:
        for name in ("default_effort", "cheap_effort", "cascade_next_tier_effort"):
            if getattr(self, name) not in EFFORTS:
                raise RuleConfigError(f"{name}: unknown effort {getattr(self, name)!r}")
        for name in ("default_tier", "cheap_tier"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise RuleConfigError(f"{name}: must be an integer >= 0, got {v!r}")


# section -> {toml key: (field name, kind)}
_SCHEMA: dict[str, dict[str, tuple[str, str]]] = {
    "default": {"tier": ("default_tier", "int"), "effort": ("default_effort", "str")},
    "cheap": {
        "tier": ("cheap_tier", "int"),
        "effort": ("cheap_effort", "str"),
        "subagent_kinds": ("subagent_kinds", "strs"),
        "read_only_tools": ("read_only_tools", "strs"),
    },
    "stay": {"enabled": ("stay_enabled", "bool"), "keep_effort": ("stay_keep_effort", "bool")},
    "cascade": {
        "enabled": ("cascade_enabled", "bool"),
        "effort_up": ("cascade_effort_up", "bool"),
        "next_tier": ("cascade_next_tier", "bool"),
        "next_tier_effort": ("cascade_next_tier_effort", "str"),
    },
    "models": {
        "prefer_current": ("prefer_current", "bool"),
        "provider_order": ("provider_order", "strs"),
    },
}


def _coerce(where: str, value: Any, kind: str) -> Any:
    if kind == "int" and isinstance(value, int) and not isinstance(value, bool):
        return value
    if kind == "str" and isinstance(value, str):
        return value
    if kind == "bool" and isinstance(value, bool):
        return value
    if kind == "strs" and isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise RuleConfigError(f"{where}: wrong type or value {value!r} (expected {kind})")


def load_rule_config(source: str | Path | Mapping[str, Any] | None = None) -> RuleConfig:
    """Load a ``RuleConfig``.

    ``source`` is a path to a TOML file, an already-parsed mapping, or ``None`` for the config
    shipped in the package (``steproute/catalogs/rules.toml``). Missing keys keep their
    defaults; unknown sections or keys raise ``RuleConfigError``.
    """
    if source is None:
        text = resources.files("steproute.catalogs").joinpath("rules.toml").read_text("utf-8")
        data: Mapping[str, Any] = tomllib.loads(text)
    elif isinstance(source, Mapping):
        data = source
    else:
        try:
            data = tomllib.loads(Path(source).read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise RuleConfigError(f"{source}: invalid TOML: {exc}") from None
    values: dict[str, Any] = {}
    for section, body in data.items():
        if section not in _SCHEMA:
            raise RuleConfigError(f"unknown section [{section}]; expected {sorted(_SCHEMA)}")
        if not isinstance(body, Mapping):
            raise RuleConfigError(f"[{section}] must be a table")
        for key, value in body.items():
            if key not in _SCHEMA[section]:
                raise RuleConfigError(
                    f"[{section}] unknown key {key!r}; expected {sorted(_SCHEMA[section])}"
                )
            field_name, kind = _SCHEMA[section][key]
            values[field_name] = _coerce(f"[{section}] {key}", value, kind)
    return RuleConfig(**values)


def _nearest_effort(profile: ModelProfile, wanted: str) -> str:
    """The model's supported effort closest in rank to ``wanted`` (ties go lower)."""
    if wanted in profile.efforts:
        return wanted
    r = effort_rank(wanted)
    return min(profile.efforts, key=lambda e: (abs(effort_rank(e) - r), effort_rank(e)))


class RulePolicy:
    """Rule-based policy with propensity 1.0 (see ``docs/RULES.md``).

    ``decide`` marks the state (sensitive detection) first, keeps only models that pass
    ``filter_feasible``, picks a target tier by rule, falls back to the nearest feasible tier
    (higher tiers first, then lower), stays on ``current_model`` when it is in the chosen tier,
    and optionally attaches an escalation cascade. The rule that fired is recorded in
    ``Decision.state.features["rules"]``. Raises ``NoFeasibleAction`` if nothing is feasible.
    ``report`` does nothing.
    """

    def __init__(
        self,
        catalog: Catalog | Iterable[ModelProfile],
        config: RuleConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = new_decision_id,
        name: str = "rules",
        privacy: PrivacyConfig | None = None,
    ) -> None:
        self.catalog = catalog if isinstance(catalog, Catalog) else Catalog(tuple(catalog))
        self.config = config or load_rule_config()
        self.name = name
        self.privacy = privacy or PrivacyConfig()
        self._clock = clock
        self._new_id = new_id
        cfg = self.config
        providers = {m.provider for m in self.catalog.models}
        unknown = [p for p in cfg.provider_order if p not in providers]
        if unknown:
            raise RuleConfigError(f"provider_order names providers not in the catalog: {unknown}")
        rank = {p: i for i, p in enumerate(cfg.provider_order)}
        pool = [m for m in self.catalog.models if not m.holdout]
        ordered = sorted(
            enumerate(pool), key=lambda im: (rank.get(im[1].provider, len(rank)), im[0])
        )
        by_tier: dict[int, list[ModelProfile]] = {}
        for _, m in ordered:
            by_tier.setdefault(m.tier, []).append(m)
        self._by_tier = {t: tuple(ms) for t, ms in by_tier.items()}
        self._tiers = tuple(sorted(self._by_tier))
        self._probe = tuple(Action(m.id, m.efforts[0]) for m in pool)
        self._profiles = {m.id: m for m in pool}
        self._readonly = frozenset(t.casefold() for t in cfg.read_only_tools)
        self._kinds = frozenset(k.casefold() for k in cfg.subagent_kinds)

    def _pick(self, tier: int, feasible: frozenset[str], current: str | None) -> ModelProfile:
        if self.config.prefer_current and current is not None and current in feasible:
            cur = self._profiles[current]
            if cur.tier == tier:
                return cur
        for m in self._by_tier[tier]:
            if m.id in feasible:
                return m
        raise AssertionError("tier has no feasible model")  # pragma: no cover

    def _target(self, state: StepState) -> tuple[str, int, str]:
        cfg = self.config
        if (
            state.kind.casefold() in self._kinds
            and state.tools
            and all(t.casefold() in self._readonly for t in state.tools)
        ):
            return "readonly_subagent", cfg.cheap_tier, cfg.cheap_effort
        return "default", cfg.default_tier, cfg.default_effort

    def decide(self, state: StepState) -> Decision:
        state = mark_before_routing(state, self.privacy)
        cfg = self.config
        feasible = frozenset(a.model for a in filter_feasible(state, self._probe, self.catalog))
        tiers = [t for t in self._tiers if any(m.id in feasible for m in self._by_tier[t])]
        rule, want_tier, want_effort = self._target(state)
        fallback = want_tier not in tiers
        if fallback:
            higher = [t for t in tiers if t > want_tier]
            lower = [t for t in reversed(tiers) if t < want_tier]
            tier = (higher + lower)[0]
        else:
            tier = want_tier
        cur = state.current_model
        model = self._pick(tier, feasible, cur)
        stayed = cfg.stay_enabled and cur == model.id
        if stayed and cfg.stay_keep_effort and state.current_effort in model.efforts:
            effort = str(state.current_effort)
        else:
            effort = _nearest_effort(model, want_effort)

        steps = [Action(model.id, effort)]
        if cfg.cascade_enabled:
            if cfg.cascade_effort_up:
                up = [e for e in model.efforts if effort_rank(e) > effort_rank(effort)]
                if up:
                    steps.append(Action(model.id, min(up, key=effort_rank)))
            if cfg.cascade_next_tier:
                nxt = [t for t in tiers if t > model.tier]
                if nxt:
                    m2 = self._pick(nxt[0], feasible, cur)
                    steps.append(Action(m2.id, _nearest_effort(m2, cfg.cascade_next_tier_effort)))
        action = steps[-1]
        for s in reversed(steps[:-1]):
            action = Action(s.model, s.effort, action)

        note = {
            "rule": rule,
            "target_tier": want_tier,
            "tier": tier,
            "fallback": fallback,
            "stayed": stayed,
        }
        marked = dataclasses.replace(state, features={**state.features, RULES_FEATURE: note})
        return Decision(
            decision_id=self._new_id(),
            ts=self._clock(),
            state=marked,
            candidates=[action],
            action=action,
            policy=self.name,
            propensity=1.0,
        )

    def report(self, outcome: Outcome) -> None:
        return None
