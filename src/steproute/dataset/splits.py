"""Fixed splits by task. Standard library only.

The split is a function of the task set alone (task ids, strata, a seed and the fractions),
never of which runs happen to exist, so every release made from the same task set gives every
task the same split. Within a stratum the tasks are ordered by a keyed hash of the task id
and cut in order; the number of tasks per split is the largest-remainder share of the
stratum, with ties and leftovers going to the split that is furthest behind its overall share.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

__all__ = [
    "DEFAULT_FRACTIONS",
    "DEFAULT_SEED",
    "HOLDOUT_SPLIT",
    "TASK_SPLITS",
    "UNSTRATIFIED",
    "all_splits",
    "assign_splits",
]

#: Splits that tasks are assigned to, in order.
TASK_SPLITS: tuple[str, ...] = ("train", "validation", "test")
#: The split holding every run of the held-out model, whatever the task.
HOLDOUT_SPLIT = "holdout_model"
UNSTRATIFIED = "unstratified"

DEFAULT_SEED = 19
DEFAULT_FRACTIONS: Mapping[str, float] = {"train": 0.6, "validation": 0.2, "test": 0.2}


def all_splits() -> tuple[str, ...]:
    return (*TASK_SPLITS, HOLDOUT_SPLIT)


def _key(seed: int, task_id: str) -> str:
    return hashlib.sha256(f"steproute-split:{seed}:{task_id}".encode()).hexdigest()


def assign_splits(
    tasks: Sequence[tuple[str, str | None]],
    *,
    seed: int = DEFAULT_SEED,
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
) -> dict[str, str]:
    """``{task_id: split}`` for ``(task_id, stratum)`` pairs. No task gets two splits.

    Raises ``ValueError`` on a duplicate task id or fractions that are not usable.
    """
    if set(fractions) != set(TASK_SPLITS) or min(fractions.values()) < 0:
        raise ValueError(f"fractions must give a non-negative share to each of {TASK_SPLITS}")
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("fractions must not all be zero")
    share = {s: fractions[s] / total for s in TASK_SPLITS}
    ids = [t for t, _ in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate task id")

    strata: dict[str, list[str]] = {}
    for task_id, stratum in tasks:
        strata.setdefault(stratum or UNSTRATIFIED, []).append(task_id)

    out: dict[str, str] = {}
    cum_ideal = dict.fromkeys(TASK_SPLITS, 0.0)
    cum_given = dict.fromkeys(TASK_SPLITS, 0)
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda t: (_key(seed, t), t))
        n = len(members)
        ideal = {s: n * share[s] for s in TASK_SPLITS}
        counts = {s: int(ideal[s]) for s in TASK_SPLITS}
        for s in TASK_SPLITS:
            cum_ideal[s] += ideal[s]
        left = n - sum(counts.values())
        order = sorted(
            TASK_SPLITS,
            key=lambda s: (
                -((ideal[s] - counts[s]) + (cum_ideal[s] - cum_given[s] - counts[s])),
                TASK_SPLITS.index(s),
            ),
        )
        for s in order[:left]:
            counts[s] += 1
        i = 0
        for s in TASK_SPLITS:
            for task_id in members[i : i + counts[s]]:
                out[task_id] = s
            i += counts[s]
            cum_given[s] += counts[s]
    return out
