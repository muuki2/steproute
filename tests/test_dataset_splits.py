"""Splits by task for the dataset release (issue 19). Standard library only."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

from steproute.dataset.splits import DEFAULT_FRACTIONS, TASK_SPLITS, assign_splits
from steproute.tasks.manifest import load_default_manifest


def _tasks(pilot: bool = False) -> list[tuple[str, str | None]]:
    m = load_default_manifest()
    return [(t.id, t.stratum) for t in (m.pilot() if pilot else m)]


def test_every_task_gets_exactly_one_split() -> None:
    tasks = _tasks()
    split = assign_splits(tasks)
    assert set(split) == {t for t, _ in tasks}
    assert set(split.values()) <= set(TASK_SPLITS)


def test_the_split_is_reproducible_and_depends_on_the_seed() -> None:
    tasks = _tasks()
    assert assign_splits(tasks) == assign_splits(list(reversed(tasks)))
    assert assign_splits(tasks, seed=1) != assign_splits(tasks, seed=2)


def test_full_task_set_is_balanced_over_strata() -> None:
    tasks = _tasks()
    split = assign_splits(tasks)
    n = len(tasks)
    sizes = Counter(split.values())
    for s in TASK_SPLITS:
        assert abs(sizes[s] - n * DEFAULT_FRACTIONS[s]) <= 1
    by_stratum = Counter(st for _, st in tasks)
    for stratum, count in by_stratum.items():
        for s in TASK_SPLITS:
            got = sum(1 for t, st in tasks if st == stratum and split[t] == s)
            assert abs(got - count * DEFAULT_FRACTIONS[s]) < 1.5, (stratum, s)


def test_pilot_tasks_alone_can_be_split() -> None:
    tasks = _tasks(pilot=True)
    sizes = Counter(assign_splits(tasks).values())
    assert sum(sizes.values()) == len(tasks)
    assert all(sizes[s] > 0 for s in TASK_SPLITS)


def test_without_pyarrow_the_package_imports_and_build_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from steproute.cli import main
    from steproute.dataset import MissingParquetSupport, tables

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(MissingParquetSupport):
        tables.check_available()
    assert main(["dataset", "build", "--from", str(tmp_path), "--out", str(tmp_path / "r")]) == 2


def test_bad_fractions_and_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError):
        assign_splits([("a", None)], fractions={"train": 1.0})
    with pytest.raises(ValueError):
        assign_splits([("a", None), ("a", None)])
