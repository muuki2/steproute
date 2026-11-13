"""Benchmark tasks with automatic pass/fail checks (issue 17).

* ``manifest``: which tasks exist and how each is checked (``load_manifest``).
* ``check``: prepare a workspace, apply a candidate patch, run the check (``run_check``),
  with a ``local`` backend (a directory and a command) and a ``swebench`` backend (the
  official SWE-bench harness in Docker).
* ``estimate``: the replay-grid cost estimate from ``cost_assumptions.toml`` and the catalog.

Standard library only. See ``docs/TASKSET.md``.
"""

from steproute.tasks.check import (
    CHECK_STATUSES,
    CheckError,
    CheckResult,
    LocalBackend,
    SwebenchBackend,
    run_check,
)
from steproute.tasks.manifest import (
    ManifestError,
    Task,
    TaskManifest,
    load_default_manifest,
    load_manifest,
)

__all__ = [
    "CHECK_STATUSES",
    "CheckError",
    "CheckResult",
    "LocalBackend",
    "ManifestError",
    "SwebenchBackend",
    "Task",
    "TaskManifest",
    "load_default_manifest",
    "load_manifest",
    "run_check",
]
