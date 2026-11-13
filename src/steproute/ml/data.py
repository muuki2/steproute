"""Training and evaluation tables from the evaluator's replay grid (``docs/EVALUATION.md``).

The grid is the only data format: ``load_grid`` reads it, and a row here is one ``(task,
configuration)`` cell. The label is the **first labelled run** of the cell (repeats beyond the
first are not used in v1); a cell with no labelled run gives no row (an unlabelled run is not
a failure).

Two kinds of rows, both read off the grid:

- **fresh** rows: the task's start state (exactly what the evaluator hands a policy,
  ``task_state``) and one configuration. Label: did that configuration solve the task.
- **conditional** rows (cascades): the start state after configurations ``A, ...`` ran and
  *failed*, built by ``state_after_failure`` as ``ScoredPolicy`` builds it, and a stronger
  configuration ``C`` (the ``enumerate_actions`` rule). Label: did ``C`` solve the task, on
  the tasks where ``A, ...`` all failed. That is the conditional probability the evaluator's
  own cascade simulation realises (a later step is a fresh, independent run, and a cascade
  succeeds if any step's run did), so it is estimated from the paired runs the grid has.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from steproute.core.catalog import Catalog
from steproute.core.cost import state_after_failure
from steproute.core.schema import Action, StepState
from steproute.eval.data import GridData, task_state
from steproute.ml.features import (
    ConfigKey,
    FeatureBuilder,
    ProbeResult,
    ProbeTable,
    config_order,
    stronger,
)

__all__ = ["Dataset", "build_dataset", "first_outcomes", "probe_table_from_grid"]


@dataclass
class Dataset:
    """Feature rows with labels and the bookkeeping needed to score and bootstrap them."""

    X: NDArray[np.float64]
    y: NDArray[np.int8]
    task_ids: list[str]
    history: list[int]  # number of failed earlier attempts in the row's state
    config: list[ConfigKey]

    def __len__(self) -> int:
        return len(self.task_ids)


def first_outcomes(grid: GridData) -> dict[tuple[str, ConfigKey], ProbeResult]:
    """Per ``(task, configuration)``, the first labelled run's success and cost."""
    best: dict[tuple[str, ConfigKey], tuple[int, ProbeResult]] = {}
    for r in grid.runs:
        if r.success is None:
            continue
        key = (r.task_id, r.config)
        if key not in best or r.repeat < best[key][0]:
            best[key] = (r.repeat, ProbeResult(r.success, r.cost_usd))
    return {k: v[1] for k, v in best.items()}


def probe_table_from_grid(grid: GridData, order: Sequence[str]) -> ProbeTable:
    """The probe table: every configuration's first labelled result on each probe task."""
    wanted = set(order)
    results: dict[ConfigKey, dict[str, ProbeResult]] = {}
    for (task, cfg), res in first_outcomes(grid).items():
        if task in wanted:
            results.setdefault(cfg, {})[task] = res
    return ProbeTable(order, results)


def _cascades(
    configs: Sequence[ConfigKey],
    outcomes: Mapping[ConfigKey, ProbeResult],
    catalog: Catalog,
    max_history: int,
) -> list[tuple[tuple[ConfigKey, ...], ConfigKey]]:
    """``(failed history, next configuration)`` for every chain the grid can score: each
    history step failed on this task and each step is stronger than the one before."""
    out: list[tuple[tuple[ConfigKey, ...], ConfigKey]] = []

    def extend(history: tuple[ConfigKey, ...]) -> None:
        last = history[-1]
        for nxt in configs:
            if nxt in history or nxt not in outcomes:
                continue
            if not stronger(catalog.get(last[0]), last[1], catalog.get(nxt[0]), nxt[1]):
                continue
            out.append((history, nxt))
            if len(history) < max_history and outcomes[nxt].success is False:
                extend((*history, nxt))

    for a in configs:
        if a in outcomes and outcomes[a].success is False and max_history >= 1:
            extend((a,))
    return out


def build_dataset(
    grid: GridData,
    tasks: Sequence[str],
    catalog: Catalog,
    builder: FeatureBuilder,
    *,
    seed: int,
    max_history: int = 1,
    history_cap: int | None = None,
    probe_sizes: Sequence[int | None] | None = None,
    exclude_models: Sequence[str] = (),
) -> Dataset:
    """Rows for ``tasks`` (in sorted order), configurations in id-free order.

    ``max_history`` is the longest failed history to build conditional rows for (0: fresh rows
    only). ``history_cap`` caps the conditional rows per task by a seeded sample. With
    ``probe_sizes``, each row draws its probe count from it (probe dropout, so the model
    sees the cold-start cases); otherwise every row uses ``builder.probe_k``. Configurations of
    ``exclude_models`` give no rows, neither as the step nor in a failed history (a held-out
    model: its probe results can still describe it).
    """
    outcomes = first_outcomes(grid)
    skip = set(exclude_models)
    all_cfgs = config_order([c for c in grid.configs() if c[0] not in skip], catalog)
    pairs: list[tuple[StepState, Action]] = []
    ks: list[int | None] = []
    y: list[int] = []
    task_ids: list[str] = []
    hist: list[int] = []
    cfg_of: list[ConfigKey] = []
    for t in sorted(tasks):
        rng = random.Random(f"{seed}:rows:{t}")
        cell = {c: outcomes[(t, c)] for c in all_cfgs if (t, c) in outcomes}
        if not cell:
            continue
        start = task_state(grid.tasks[t])

        def draw(rng: random.Random = rng) -> int | None:
            return builder.probe_k if probe_sizes is None else rng.choice(list(probe_sizes))

        for c in all_cfgs:
            if c in cell:
                res = cell[c]
                assert res.success is not None
                pairs.append((start, Action(*c)))
                ks.append(draw())
                y.append(int(res.success))
                task_ids.append(t)
                hist.append(0)
                cfg_of.append(c)
        if max_history >= 1:
            chains = _cascades(all_cfgs, cell, catalog, max_history)
            if history_cap is not None and len(chains) > history_cap:
                keep = sorted(rng.sample(range(len(chains)), history_cap))
                chains = [chains[i] for i in keep]  # id-free order is kept
            states: dict[tuple[ConfigKey, ...], StepState] = {}
            for history, nxt in chains:
                st = states.get(history)
                if st is None:
                    st = start
                    for h in history:
                        st = state_after_failure(st, Action(*h), catalog.get(h[0]))
                    states[history] = st
                res = cell[nxt]
                assert res.success is not None
                pairs.append((st, Action(*nxt)))
                ks.append(draw())
                y.append(int(res.success))
                task_ids.append(t)
                hist.append(len(history))
                cfg_of.append(nxt)
    X = builder.matrix(pairs, ks)
    return Dataset(X, np.asarray(y, dtype=np.int8), task_ids, hist, cfg_of)
