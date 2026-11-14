"""What every external baseline shares: the task split, the inputs, the policy and the report.

Needs numpy (the ``ml`` extra), nothing heavier.

- ``BaselineData``: a replay grid cut into the same task roles as the tabular predictor
  (``steproute.ml.splits.assign_roles`` with the same seed and probe count, so the ``test``
  tasks are the ones issue 21 scores), with the per-task outcome of every configuration.
- ``ContextEncoder``: the numeric routing state every baseline is given in place of prompt text:
  the tabular predictor's state block (``steproute.ml.features``), standardised on the training
  tasks, with a one-hot copy of each small integer column so linear models can use it.
- ``ArmPolicy``: a deterministic ``TargetPolicy`` that maps a state to one configuration.
- ``baseline_report``: scores the policies on the ``test`` tasks next to the rule baseline and the
  hindsight oracle, and stamps anything made from synthetic data.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from steproute.core.catalog import Catalog
from steproute.core.rules import RuleConfigError
from steproute.core.schema import Action, StepState
from steproute.eval.data import Cell, ConfigKey, GridData, task_state
from steproute.eval.grid import evaluate_grid
from steproute.eval.plot import render_svg
from steproute.eval.report import build_report
from steproute.eval.results import EvalSettings, Evaluation
from steproute.eval.targets import DistributionTarget, TargetPolicy, parse_policy_spec
from steproute.ml.data import probe_table_from_grid
from steproute.ml.features import (
    FeatureBuilder,
    FeatureSpec,
    ProbeTable,
    config_order,
    task_feature_keys,
)
from steproute.ml.splits import Fractions, Roles, assign_roles
from steproute.ml.stamp import STAMP, is_synthetic

__all__ = [
    "ArmPolicy",
    "BaselineData",
    "ContextEncoder",
    "Upstream",
    "arm_label",
    "baseline_report",
]


def arm_label(arm: ConfigKey) -> str:
    """``EFFORT:MODEL``, the evaluator's spelling of a configuration."""
    return f"{arm[1]}:{arm[0]}"


@dataclass(frozen=True)
class Upstream:
    """Where a baseline comes from. Mirrored in ``docs/BASELINES.md``."""

    name: str
    repository: str
    commit: str
    version: str
    licence: str
    paper: str
    use: str  # "dependency", "adapted" or "note"

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


# --- data -----------------------------------------------------------------------------------


class BaselineData:
    """A grid with task roles, restricted to tasks every configuration ran cleanly.

    Roles come from ``assign_roles`` exactly as ``steproute.ml.model.train`` draws them (same
    seed, probe count and fractions), so a baseline and the tabular predictor trained with the
    same seed are scored on the same ``test`` tasks. A task enters a role's list only when
    every configuration's cell is complete (labelled and costed), which is also what the
    evaluator's headline requires.

    ``exclude_configs`` drops configurations from the arms (and, in ``baseline_report``, from
    the frontier), for a sparse grid where some configurations ran on a few tasks only.
    """

    def __init__(
        self,
        grid: GridData,
        catalog: Catalog,
        *,
        seed: int = 0,
        n_probes: int = 50,
        fractions: Fractions | None = None,
        exclude_configs: Sequence[ConfigKey] = (),
    ) -> None:
        missing = sorted({m for m, _ in grid.configs() if m not in catalog})
        if missing:
            raise ValueError(f"the grid has models that the catalog does not: {missing}")
        self.grid = grid
        self.catalog = catalog
        self.seed = seed
        self.roles: Roles = assign_roles(
            sorted(grid.tasks),
            seed=seed,
            n_probe=n_probes,
            fractions=fractions or Fractions(),
            grid_splits={t: grid.tasks[t].split for t in grid.tasks},
        )
        self.excluded: tuple[ConfigKey, ...] = tuple(exclude_configs)
        #: Every configuration not excluded, cheapest tier and effort first (``config_order``).
        self.arms: tuple[ConfigKey, ...] = tuple(
            config_order([c for c in grid.configs() if c not in self.excluded], catalog)
        )
        if len(self.arms) < 2:
            raise ValueError("a baseline needs at least two configurations to route between")
        self._cells: dict[tuple[str, ConfigKey], Cell] = grid.cells()
        self._complete = {
            t
            for t in grid.tasks
            if all((c := self._cells.get((t, a))) is not None and c.complete for a in self.arms)
        }

    @property
    def synthetic(self) -> bool:
        return is_synthetic(self.grid.header)

    def tasks(self, role: str) -> list[str]:
        return [t for t in self.roles.tasks(role) if t in self._complete]

    def state(self, task: str) -> StepState:
        return task_state(self.grid.tasks[task])

    def outcomes(self, tasks: Sequence[str]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(success, cost)`` matrices, one row per task and one column per arm."""
        s = np.empty((len(tasks), len(self.arms)), dtype=np.float64)
        c = np.empty_like(s)
        for i, t in enumerate(tasks):
            for j, a in enumerate(self.arms):
                cell = self._cells[(t, a)]
                assert cell.success is not None and cell.cost is not None
                s[i, j] = cell.success
                c[i, j] = cell.cost
        return s, c

    def probes(self) -> ProbeTable:
        return probe_table_from_grid(self.grid, self.roles.probe_order)

    def counts(self) -> dict[str, int]:
        return {r: len(self.tasks(r)) for r in ("probe", "train", "tune", "calibration", "test")}


# --- inputs ---------------------------------------------------------------------------------


@dataclass
class ContextEncoder:
    """The routing state as a fixed-length numeric vector, fitted on training states.

    The columns are the tabular predictor's state block (``FeatureSpec.state_names``: step kind,
    prompt and cache sizes, tools, step index, sensitivity, each numeric ``state.features``
    key, and the failed-attempt summary). Missing values become 0, columns constant on the
    training states are dropped, the rest are z-scored. Each kept column whose training values
    are at most ``max_levels`` distinct integers also gets a one-hot copy (an unseen value is all
    zeros), so a linear or factorised model is not limited to a monotone effect of, say, a
    difficulty hint. No prompt text, embedding or task id is used.
    """

    catalog: Catalog
    max_levels: int = 8
    names: tuple[str, ...] = field(default=(), init=False)
    _builder: FeatureBuilder | None = field(default=None, init=False, repr=False)
    _cols: NDArray[np.intp] = field(default_factory=lambda: np.zeros(0, np.intp), init=False)
    _mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), init=False)
    _std: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0), init=False)
    _levels: list[tuple[int, tuple[float, ...]]] = field(default_factory=list, init=False)
    _step: Action | None = field(default=None, init=False, repr=False)

    def _raw(self, states: Sequence[StepState]) -> NDArray[np.float64]:
        assert self._builder is not None and self._step is not None
        n = len(self._builder.spec.state_names)
        out = self._builder.matrix([(s, self._step) for s in states])[:, :n]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, states: Sequence[StepState]) -> ContextEncoder:
        if not states:
            raise ValueError("no training states to fit the context encoder on")
        model = self.catalog.models[0]
        self._step = Action(model.id, model.efforts[0] if model.efforts else "medium")
        spec = FeatureSpec(task_keys=task_feature_keys(states))
        self._builder = FeatureBuilder(spec, self.catalog)
        raw = self._raw(states)
        std = raw.std(axis=0)
        self._cols = np.flatnonzero(std > 1e-12).astype(np.intp)
        self._mean = raw[:, self._cols].mean(axis=0)
        self._std = std[self._cols]
        all_names = spec.state_names
        names = [all_names[c] for c in self._cols]
        self._levels = []
        for j, c in enumerate(self._cols):
            values = np.unique(raw[:, c])
            if len(values) <= self.max_levels and np.all(values == np.round(values)):
                self._levels.append((j, tuple(float(v) for v in values)))
                names += [f"{all_names[c]}=={v:g}" for v in values]
        self.names = tuple(names)
        return self

    @property
    def dim(self) -> int:
        return len(self.names)

    def transform_many(self, states: Sequence[StepState]) -> NDArray[np.float64]:
        if self._builder is None:
            raise RuntimeError("call fit() first")
        raw = self._raw(states)[:, self._cols]
        z = (raw - self._mean) / self._std
        blocks = [z]
        for j, values in self._levels:
            blocks.append(
                (raw[:, j : j + 1] == np.asarray(values, dtype=np.float64)[None, :]).astype(
                    np.float64
                )
            )
        return np.concatenate(blocks, axis=1) if blocks else z

    def transform(self, state: StepState) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self.transform_many([state])[0]
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "source": "the tabular predictor's state block (steproute.ml.features), no text",
            "columns": list(self.names),
            "standardised_on": "train tasks",
            "one_hot_max_levels": self.max_levels,
        }


# --- the policy -----------------------------------------------------------------------------


@dataclass
class ArmPolicy:
    """Deterministic: ``choose(state)`` names one configuration, given probability 1.

    Answers are cached by the encoded state, so the same state always gets the same arm even
    when the upstream router breaks ties at random.
    """

    name: str
    choose: Callable[[NDArray[np.float64]], ConfigKey]
    encoder: ContextEncoder
    hindsight: bool = False
    _cache: dict[bytes, ConfigKey] = field(default_factory=dict, init=False, repr=False)

    def arm(self, state: StepState) -> ConfigKey:
        x = self.encoder.transform(state)
        key = x.tobytes()
        hit = self._cache.get(key)
        if hit is None:
            hit = self.choose(x)
            self._cache[key] = hit
        return hit

    def distribution(self, state: StepState) -> tuple[tuple[Action, float], ...]:
        model, effort = self.arm(state)
        return ((Action(model, effort), 1.0),)


# --- the report -----------------------------------------------------------------------------


def stamp_report(report: dict[str, Any]) -> None:
    """``steproute.ml.evaluate.stamp_report`` (imported late: it pulls in the tabular model)."""
    from steproute.ml.evaluate import stamp_report as _stamp

    _stamp(report)


def baseline_report(
    data: BaselineData,
    policies: Sequence[TargetPolicy],
    provenance: Mapping[str, Any],
    *,
    rules: str | None = None,
    settings: EvalSettings | None = None,
    command: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str, Evaluation]:
    """Score ``policies`` on the ``test`` tasks with the rule baseline and the oracle.

    Returns ``(report dict, SVG text, Evaluation)``. Every single configuration is the frontier.
    Anything made from a synthetic grid is stamped ``SYNTHETIC DATA. NOT A RESULT.`` in the
    JSON (top level, every verdict and summary line) and in the SVG.
    """
    from steproute.ml.evaluate import stamp_svg

    st = settings or EvalSettings()
    grid = data.grid
    tasks = {
        t: dataclasses.replace(task, split=data.roles.role.get(t, "excluded"))
        for t, task in grid.tasks.items()
    }
    # Only the file's name goes into the report, never the directory it was read from.
    scored = GridData(
        os.path.basename(grid.source),
        grid.runs,
        tasks,
        grid.header,
        grid.skipped_lines,
        grid.sha256,
    )
    targets: list[TargetPolicy] = list(policies)
    skipped: str | None = None
    try:
        rule_target = parse_policy_spec(
            f"rules:{rules}" if rules else "rules", lambda: data.catalog
        )
        if isinstance(rule_target, DistributionTarget):
            rule_target.name = "rules"  # not the file path: reports must not carry local paths
        targets.append(rule_target)
    except RuleConfigError as exc:
        if rules:
            raise
        skipped = f"The rule baseline was not run: {exc}. Pass a rules file for this catalog."
    targets.append(parse_policy_spec("oracle", lambda: data.catalog))
    ev = evaluate_grid(scored, targets, st, split="test", exclude_configs=data.excluded)
    if skipped:
        ev.notes.append(skipped)
    ev.provenance["baseline"] = dict(provenance)
    ev.provenance["baseline"]["tasks_by_role"] = data.counts()
    ev.provenance["baseline"]["split_seed"] = data.seed
    ev.provenance["baseline"]["excluded_configs"] = [arm_label(a) for a in data.excluded]
    report = build_report([ev], command=command)
    svg = render_svg(ev, title=STAMP if data.synthetic else None)
    if data.synthetic:
        stamp_report(report)
        svg = stamp_svg(svg)
    return report, svg, ev
