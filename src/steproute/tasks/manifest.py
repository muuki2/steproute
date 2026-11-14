"""Task manifests: which tasks exist, which backend checks them, and how they are grouped.

A manifest is a TOML file with one ``[meta]`` table and a list of ``[[task]]`` tables. It is
hand-written (or generated) config, so loading is strict: unknown keys raise ``ManifestError``.
Standard library only. Format: ``docs/TASKSET.md``, "Manifest format".
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = [
    "BACKENDS",
    "DEFAULT_MANIFEST",
    "ManifestError",
    "Task",
    "TaskManifest",
    "load_default_manifest",
    "load_manifest",
]

#: Backends a manifest may name. ``swebench`` runs the official harness in Docker;
#: ``local`` copies a directory and runs a command (no container).
BACKENDS = ("swebench", "local")

#: The issue-17 selection, shipped as package data.
DEFAULT_MANIFEST = "swebench-verified-v1.toml"

_META_KEYS = {
    "name",
    "benchmark",
    "dataset",
    "dataset_revision",
    "split",
    "backend",
    "image_arch",
    "seed",
    "experiments_commit",
    "full_size",
    "pilot_size",
    "timeout_s",
}
_TASK_KEYS = {
    "id",
    "stratum",
    "fits32k",
    "pilot",
    "image",
    "repo",
    "test_cmd",
    "timeout_s",
    "gold_patch",
    "prompt",
    "broken",
    "replaced_by",
}


class ManifestError(ValueError):
    """The manifest is invalid. The message names the task and key where possible."""


@dataclass(frozen=True)
class Task:
    """One task. ``repo``, ``test_cmd`` and ``gold_patch`` are used by the local backend only;
    ``image`` (the evaluation image, overriding the dataset's) by the swebench backend only.

    ``repo`` and ``gold_patch`` are resolved against the manifest's directory. In ``test_cmd``
    the token ``{python}`` is replaced by the running interpreter.
    """

    id: str
    backend: str
    stratum: str | None = None
    fits32k: bool | None = None
    pilot: bool = False
    image: str | None = None
    repo: Path | None = None
    test_cmd: tuple[str, ...] = ()
    timeout_s: float = 1800.0
    gold_patch: Path | None = None
    #: The problem statement given to the agent (local tasks; SWE-bench tasks take it from
    #: the dataset row).
    prompt: str | None = None
    #: Why the task cannot be graded correctly (for example its gold patch fails on the
    #: image). A broken task keeps its record but is not part of the set: it is left out of
    #: iteration, ``len``, ``pilot()`` and every plan.
    broken: str | None = None
    #: For a broken task: the id of the task that took its place in the set.
    replaced_by: str | None = None


@dataclass(frozen=True)
class TaskManifest:
    """A loaded manifest. Iterates over its working tasks in file order; the broken ones are
    in ``broken_tasks`` and still found by ``get`` and ``in`` (so ``steproute tasks check``
    can show why they are broken)."""

    meta: Mapping[str, Any]
    tasks: tuple[Task, ...]
    path: Path | None = None
    broken_tasks: tuple[Task, ...] = ()
    _index: dict[str, Task] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._index.update({t.id: t for t in (*self.tasks, *self.broken_tasks)})

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks)

    def __contains__(self, task_id: object) -> bool:
        return task_id in self._index

    def get(self, task_id: str) -> Task:
        try:
            return self._index[task_id]
        except KeyError:
            raise KeyError(task_id) from None

    @property
    def name(self) -> str:
        return str(self.meta.get("name", self.path.stem if self.path else "manifest"))

    def pilot(self) -> tuple[Task, ...]:
        """The pilot slice, in file order."""
        return tuple(t for t in self.tasks if t.pilot)


def _check_type(value: Any, kind: type | tuple[type, ...], where: str) -> Any:
    if isinstance(value, bool) and kind in (int, float, (int, float)):
        raise ManifestError(f"{where}: expected a number, got a boolean")
    if not isinstance(value, kind):
        raise ManifestError(f"{where}: expected {kind}, got {type(value).__name__}")
    return value


def _task(raw: Any, index: int, meta: Mapping[str, Any], base: Path | None) -> Task:
    if not isinstance(raw, dict):
        raise ManifestError(f"task {index}: must be a table")
    tid = raw.get("id")
    if not isinstance(tid, str) or not tid:
        raise ManifestError(f"task {index}: 'id' must be a non-empty string")
    unknown = sorted(set(raw) - _TASK_KEYS)
    if unknown:
        raise ManifestError(f"task {tid!r}: unknown key(s) {', '.join(unknown)}")
    backend = str(meta.get("backend", "swebench"))
    where = f"task {tid!r}"

    def path(key: str) -> Path | None:
        if key not in raw:
            return None
        p = Path(_check_type(raw[key], str, f"{where}, {key!r}"))
        return p if p.is_absolute() or base is None else base / p

    test_cmd = raw.get("test_cmd", [])
    _check_type(test_cmd, list, f"{where}, 'test_cmd'")
    for part in test_cmd:
        _check_type(part, str, f"{where}, 'test_cmd'")
    timeout = raw.get("timeout_s", meta.get("timeout_s", 1800.0))
    _check_type(timeout, (int, float), f"{where}, 'timeout_s'")
    if timeout <= 0:
        raise ManifestError(f"{where}, 'timeout_s': must be positive")
    stratum = raw.get("stratum")
    if stratum is not None:
        _check_type(stratum, str, f"{where}, 'stratum'")
    fits = raw.get("fits32k")
    if fits is not None:
        _check_type(fits, bool, f"{where}, 'fits32k'")
    image = raw.get("image")
    if image is not None:
        _check_type(image, str, f"{where}, 'image'")
    prompt = raw.get("prompt")
    if prompt is not None:
        _check_type(prompt, str, f"{where}, 'prompt'")
    broken = raw.get("broken")
    if broken is not None:
        _check_type(broken, str, f"{where}, 'broken'")
        if not broken.strip():
            raise ManifestError(f"{where}, 'broken': give the reason")
    replaced_by = raw.get("replaced_by")
    if replaced_by is not None:
        _check_type(replaced_by, str, f"{where}, 'replaced_by'")
        if broken is None:
            raise ManifestError(f"{where}: 'replaced_by' needs 'broken'")
    task = Task(
        id=tid,
        backend=backend,
        stratum=stratum,
        fits32k=fits,
        pilot=bool(_check_type(raw.get("pilot", False), bool, f"{where}, 'pilot'")),
        image=image,
        repo=path("repo"),
        test_cmd=tuple(test_cmd),
        timeout_s=float(timeout),
        gold_patch=path("gold_patch"),
        prompt=prompt,
        broken=broken,
        replaced_by=replaced_by,
    )
    if task.broken is not None and task.pilot:
        raise ManifestError(f"{where}: a broken task cannot be in the pilot")
    if backend == "local" and (task.repo is None or not task.test_cmd):
        raise ManifestError(f"{where}: a local task needs 'repo' and 'test_cmd'")
    return task


def parse_manifest(data: Mapping[str, Any], *, path: Path | None = None) -> TaskManifest:
    """Validate a parsed manifest. ``path`` (the file) anchors relative task paths."""
    unknown = sorted(set(data) - {"meta", "task"})
    if unknown:
        raise ManifestError(f"unknown top-level key(s) {', '.join(unknown)}")
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        raise ManifestError("'meta' must be a table")
    bad = sorted(set(meta) - _META_KEYS)
    if bad:
        raise ManifestError(f"meta: unknown key(s) {', '.join(bad)}")
    backend = meta.get("backend", "swebench")
    if backend not in BACKENDS:
        raise ManifestError(f"meta, 'backend': must be one of {', '.join(BACKENDS)}")
    raw_tasks = data.get("task", [])
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ManifestError("a manifest needs at least one [[task]]")
    base = path.parent if path is not None else None
    every = tuple(_task(raw, i, meta, base) for i, raw in enumerate(raw_tasks))
    seen: set[str] = set()
    for t in every:
        if t.id in seen:
            raise ManifestError(f"task {t.id!r}: duplicate id")
        seen.add(t.id)
    tasks = tuple(t for t in every if t.broken is None)
    broken = tuple(t for t in every if t.broken is not None)
    for t in broken:
        if t.replaced_by is not None and t.replaced_by not in {w.id for w in tasks}:
            raise ManifestError(
                f"task {t.id!r}: replaced_by {t.replaced_by!r} is not a working task"
            )
    if not tasks:
        raise ManifestError("a manifest needs at least one working [[task]]")
    for key, actual in (("full_size", len(tasks)), ("pilot_size", sum(t.pilot for t in tasks))):
        if key in meta and meta[key] != actual:
            raise ManifestError(f"meta, {key!r}: says {meta[key]}, the file has {actual} working")
    return TaskManifest(meta=dict(meta), tasks=tasks, path=path, broken_tasks=broken)


def load_manifest(path: str | Path) -> TaskManifest:
    """Load and validate a manifest file."""
    p = Path(path)
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"{p}: {exc}") from exc
    return parse_manifest(data, path=p)


def load_default_manifest() -> TaskManifest:
    """The issue-17 SWE-bench Verified selection shipped with the package."""
    ref = resources.files("steproute.tasks") / "manifests" / DEFAULT_MANIFEST
    data = tomllib.loads(ref.read_text(encoding="utf-8"))
    return parse_manifest(data)
