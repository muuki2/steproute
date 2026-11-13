"""Task-level splits for the tabular predictor. Standard library only.

Rows are never split: every run of a task, under every configuration and in every cascade
history, goes to the same role. Roles:

- ``probe``: the fixed probe set. Its results become the probe features of every
  configuration. A probe task is in **no** other role, so no probe task is a training,
  calibration or test row of any configuration it describes.
- ``train``: fits the trees.
- ``tune``: chooses the number of boosting rounds (early stopping). Never calibrates.
- ``calibration``: fits the calibrator. Disjoint from ``train`` and ``tune`` (the trees have
  not seen it) and from ``test``.
- ``test``: scored once, in ``evaluate``.
- ``excluded``: tasks left out (the rest of a probe group that was cut to size, or tasks
  whose grid split is not one of ours).

A ``group`` is a set of tasks that must stay together (a repository, a stratum, a session
family). With no grouping each task is its own group. Groups are assigned as wholes.

If the grid's tasks carry the splits of issue 19 (``train`` and ``test`` both present), they
are respected: ``test`` stays ``test``, ``validation`` becomes ``calibration``, and the probe,
train, tune and (when there is no ``validation``) calibration roles are carved out of
``train``. Otherwise roles are drawn at random from the seed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["ROLES", "Fractions", "Roles", "assign_roles"]

ROLES = ("probe", "train", "tune", "calibration", "test", "excluded")


@dataclass(frozen=True)
class Fractions:
    """Share of the non-probe tasks that goes to each role (normalised)."""

    train: float = 0.55
    tune: float = 0.1
    calibration: float = 0.15
    test: float = 0.2

    def __post_init__(self) -> None:
        if min(self.train, self.tune, self.calibration, self.test) < 0 or self.total() <= 0:
            raise ValueError("fractions must be non-negative and not all zero")

    def total(self) -> float:
        return self.train + self.tune + self.calibration + self.test


@dataclass
class Roles:
    """The role of every task, plus how it was decided."""

    role: dict[str, str]
    probe_order: tuple[str, ...]
    source: str  # "grid splits" or "seeded random"
    notes: list[str] = field(default_factory=list)

    def tasks(self, role: str) -> list[str]:
        return sorted(t for t, r in self.role.items() if r == role)

    def counts(self) -> dict[str, int]:
        return {r: sum(1 for v in self.role.values() if v == r) for r in ROLES}

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"role": sorted(self.role.items()), "probe_order": list(self.probe_order)},
            separators=(",", ":"),
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "notes": list(self.notes),
            "probe_order": list(self.probe_order),
            "role": dict(sorted(self.role.items())),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Roles:
        role = data["role"]
        order = data["probe_order"]
        notes = data.get("notes", [])
        assert isinstance(role, dict) and isinstance(order, list) and isinstance(notes, list)
        return cls(
            {str(k): str(v) for k, v in role.items()},
            tuple(str(t) for t in order),
            str(data["source"]),
            [str(n) for n in notes],
        )


def _groups(tasks: Sequence[str], group_of: Callable[[str], str] | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in sorted(tasks):
        out.setdefault(t if group_of is None else group_of(t), []).append(t)
    return out


def _shuffled(keys: Sequence[str], seed: int, salt: str) -> list[str]:
    rng = random.Random(f"{seed}:{salt}")
    out = sorted(keys)
    rng.shuffle(out)
    return out


def _deal(
    groups: Mapping[str, list[str]], order: Sequence[str], targets: Mapping[str, float]
) -> dict[str, str]:
    """Give each group, in ``order``, to the role furthest below its target count."""
    got = dict.fromkeys(targets, 0)
    role: dict[str, str] = {}
    for g in order:
        size = len(groups[g])
        pick = max(targets, key=lambda r: (targets[r] - got[r], -list(targets).index(r)))
        got[pick] += size
        for t in groups[g]:
            role[t] = pick
    return role


def assign_roles(
    task_ids: Sequence[str],
    *,
    seed: int,
    n_probe: int,
    fractions: Fractions | None = None,
    group_of: Callable[[str], str] | None = None,
    grid_splits: Mapping[str, str | None] | None = None,
) -> Roles:
    """Assign every task a role. Deterministic in ``(task ids, seed, arguments)``.

    ``n_probe`` is the size of the probe set (0 for none). ``group_of`` maps a task id to its
    group. ``grid_splits`` maps a task id to its grid split name, if the grid has them.
    """
    fr = fractions or Fractions()
    notes: list[str] = []
    ids = sorted(set(task_ids))
    role: dict[str, str] = {}
    pool = ids
    use_grid = False
    if grid_splits is not None:
        names = {grid_splits.get(t) for t in ids}
        if "train" in names and "test" in names:
            use_grid = True
        else:
            notes.append(
                "the grid's splits were not used: they need both 'train' and 'test' "
                f"(found {sorted(str(n) for n in names)})"
            )
    if use_grid:
        assert grid_splits is not None
        pool = []
        for t in ids:
            s = grid_splits.get(t)
            if s == "test":
                role[t] = "test"
            elif s == "validation":
                role[t] = "calibration"
            elif s == "train":
                pool.append(t)
            else:
                role[t] = "excluded"
        has_validation = any(grid_splits.get(t) == "validation" for t in ids)
        # The grid decides test (and validation); only what is left of train is carved up.
        fr = Fractions(fr.train, fr.tune, 0.0 if has_validation else fr.calibration, 0.0)
    groups = _groups(pool, group_of)
    order = _shuffled(list(groups), seed, "groups")

    # Probe tasks: whole groups, in shuffled order, until n_probe tasks. A group that does not
    # fit whole is cut to size and its remainder excluded (it cannot be both probe and other).
    probe: list[str] = []
    rest: list[str] = []
    for g in order:
        members = groups[g]
        if len(probe) >= n_probe:
            rest.append(g)
        elif len(probe) + len(members) <= n_probe:
            probe.extend(_shuffled(members, seed, f"probe:{g}"))
        else:
            cut = _shuffled(members, seed, f"probe:{g}")
            take = n_probe - len(probe)
            probe.extend(cut[:take])
            for t in cut[take:]:
                role[t] = "excluded"
            notes.append(
                f"group {g!r} was cut to fit the probe set; its {len(cut) - take} other "
                "tasks are excluded"
            )
    for t in probe:
        role[t] = "probe"
    if len(probe) < n_probe:
        notes.append(f"only {len(probe)} probe tasks available; {n_probe} asked for")

    targets = {
        "train": fr.train / fr.total(),
        "tune": fr.tune / fr.total(),
        "calibration": fr.calibration / fr.total(),
        "test": fr.test / fr.total(),
    }
    n_rest = sum(len(groups[g]) for g in rest)
    scaled = {r: v * n_rest for r, v in targets.items() if v > 0}
    role.update(_deal(groups, rest, scaled))
    return Roles(
        role=role,
        probe_order=tuple(probe),
        source="grid splits" if use_grid else "seeded random",
        notes=notes,
    )
