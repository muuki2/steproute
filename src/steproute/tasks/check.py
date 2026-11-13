"""Run a task's automatic check on a candidate patch and return pass or fail.

Two backends share one result type:

* ``LocalBackend``: copy the task's directory into a fresh workspace, ``git apply`` the patch,
  run the task's command with a timeout. Exit code 0 is a pass. No container.
* ``SwebenchBackend``: hand the patch to the official SWE-bench harness
  (``swebench.harness.run_evaluation``), which runs the instance's evaluation image in Docker,
  and read its per-instance ``report.json``. The harness is an external command, so steproute
  does not depend on it.

``CheckResult.signals()`` gives the ``Outcome.signals`` for the label (``tests_passed``).
Standard library only. See ``docs/TASKSET.md``, "The loader".
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from steproute.core.labels import TESTS_PASSED
from steproute.tasks.manifest import Task

__all__ = [
    "CHECK_STATUSES",
    "DEFAULT_HARNESS",
    "Backend",
    "CheckError",
    "CheckResult",
    "LocalBackend",
    "SwebenchBackend",
    "fetch_rows",
    "load_rows",
    "run_check",
    "run_command",
]

#: ``passed``: the check passed. ``failed``: it ran and failed. ``patch_failed``: the patch did
#: not apply. ``timeout``: the check ran out of time. ``error``: the check itself broke
#: (infrastructure, missing image, harness crash), so nothing is known about the patch.
CHECK_STATUSES = ("passed", "failed", "patch_failed", "timeout", "error")

#: How the swebench backend invokes the harness. Any prefix that ends in the harness module
#: works, for example ``uvx --from swebench==5.0.2 python -m swebench.harness.run_evaluation``.
DEFAULT_HARNESS: tuple[str, ...] = ("python", "-m", "swebench.harness.run_evaluation")

# Markers the harness writes to run_instance.log (swebench/harness/constants, v5.0.x).
_APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
_TIMED_OUT = "Test timed out after"

_LOG_CHARS = 4000


class CheckError(RuntimeError):
    """The check could not be set up (missing directory, no git, unknown task row)."""


@dataclass(frozen=True)
class CheckResult:
    """The result of checking one patch on one task."""

    task_id: str
    status: str
    backend: str
    detail: str = ""
    duration_ms: float = 0.0
    log_tail: str = ""  # last part of the test or harness output; may contain repo code

    def __post_init__(self) -> None:
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"unknown check status {self.status!r}")

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    @property
    def tests_passed(self) -> bool | None:
        """The task-level label: True if passed, False if the patch failed the check in any way
        the patch is responsible for (tests failed, did not apply, hung), None on ``error``."""
        if self.status == "error":
            return None
        return self.status == "passed"

    def signals(self) -> dict[str, Any]:
        """``Outcome.signals`` for this result. ``check_status`` is informational only."""
        out: dict[str, Any] = {"check_status": self.status}
        if self.tests_passed is not None:
            out[TESTS_PASSED] = self.tests_passed
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "backend": self.backend,
            "tests_passed": self.tests_passed,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 1),
        }


class Backend(Protocol):
    name: str

    def check(self, task: Task, patch: str, *, timeout_s: float | None = None) -> CheckResult: ...


def _tail(text: str, n: int = _LOG_CHARS) -> str:
    return text[-n:]


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None  # None when it timed out
    output: str
    timed_out: bool


#: Variables that point git at one particular repository (the list `git rev-parse
#: --local-env-vars` prints). Git exports them to hooks, and a caller may have set them.
GIT_LOCAL_ENV_VARS: tuple[str, ...] = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
)


def workspace_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment for a command run in a task workspace.

    It is ``env`` (the caller's environment by default) without git's repository-local
    variables. With ``GIT_DIR`` inherited, a ``git init`` or ``git apply`` in the workspace
    would act on the caller's repository instead: run from a git hook, it has flipped that
    repository's ``core.bare``.
    """
    base = dict(os.environ if env is None else env)
    for name in GIT_LOCAL_ENV_VARS:
        base.pop(name, None)
    return base


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    env: Mapping[str, str] | None = None,
    stdin: str | None = None,
) -> CommandResult:
    """Run ``cmd`` in its own process group; on timeout kill the whole group.

    The command never inherits git's repository-local variables (``workspace_env``).
    """
    proc = subprocess.Popen(
        list(cmd),
        cwd=cwd,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=workspace_env(env),
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(stdin.encode() if stdin is not None else None, timeout=timeout_s)
        return CommandResult(proc.returncode, out.decode("utf-8", "replace"), False)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        return CommandResult(None, out.decode("utf-8", "replace"), True)


# --- local backend ------------------------------------------------------------------------------


class LocalBackend:
    """Check a task that is a plain directory plus a test command. No container."""

    name = "local"

    def __init__(self, *, workdir: Path | None = None, keep_workspace: bool = False) -> None:
        self.workdir = workdir
        self.keep_workspace = keep_workspace
        if shutil.which("git") is None:
            raise CheckError("the local backend needs git on PATH")

    def prepare(self, task: Task) -> Path:
        """A fresh copy of the task's directory, initialised as its own git repository so a
        patch applies relative to it and never to an enclosing checkout."""
        if task.repo is None or not task.repo.is_dir():
            raise CheckError(f"task {task.id!r}: repo directory {task.repo} does not exist")
        root = Path(tempfile.mkdtemp(prefix=f"steproute-{task.id}-", dir=self.workdir))
        ws = root / "repo"
        shutil.copytree(task.repo, ws, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        r = run_command(["git", "init", "-q"], cwd=ws, timeout_s=30)
        if r.returncode != 0:
            raise CheckError(f"git init failed: {r.output.strip()}")
        return ws

    def apply(self, ws: Path, patch: str) -> tuple[bool, str]:
        """Apply a unified diff with ``git apply``, then ``patch --fuzz`` as a fallback (the
        same order as the SWE-bench harness). An empty patch applies trivially."""
        if not patch.strip():
            return True, "empty patch"
        pfile = ws.parent / "candidate.patch"
        pfile.write_text(patch if patch.endswith("\n") else patch + "\n", encoding="utf-8")
        attempts = [["git", "apply", "--verbose", "--whitespace=nowarn", str(pfile)]]
        if shutil.which("patch"):
            attempts.append(["patch", "--batch", "--forward", "--fuzz=5", "-p1", "-i", str(pfile)])
        log = []
        for cmd in attempts:
            r = run_command(cmd, cwd=ws, timeout_s=60)
            log.append(f"$ {' '.join(cmd[:2])}\n{r.output}")
            if r.returncode == 0:
                return True, "\n".join(log)
            run_command(["git", "checkout", "--", "."], cwd=ws, timeout_s=30)
        return False, "\n".join(log)

    def check(self, task: Task, patch: str, *, timeout_s: float | None = None) -> CheckResult:
        start = time.monotonic()

        def result(status: str, detail: str, log: str = "") -> CheckResult:
            ms = (time.monotonic() - start) * 1000.0
            return CheckResult(task.id, status, self.name, detail, ms, _tail(log))

        ws = self.prepare(task)
        try:
            ok, log = self.apply(ws, patch)
            if not ok:
                return result("patch_failed", "the patch does not apply", log)
            cmd = [sys.executable if part == "{python}" else part for part in task.test_cmd]
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            limit = timeout_s if timeout_s is not None else task.timeout_s
            r = run_command(cmd, cwd=ws, timeout_s=limit, env=env)
            if r.timed_out:
                return result("timeout", f"the check ran longer than {limit:g} s", r.output)
            if r.returncode == 0:
                return result("passed", "exit code 0", r.output)
            return result("failed", f"exit code {r.returncode}", r.output)
        finally:
            if not self.keep_workspace:
                shutil.rmtree(ws.parent, ignore_errors=True)


# --- swebench backend ---------------------------------------------------------------------------

Runner = Callable[[Sequence[str], Path, float], CommandResult]


def _default_runner(cmd: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
    return run_command(cmd, cwd=cwd, timeout_s=timeout_s)


def load_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Dataset rows keyed by instance id, from a JSONL file (as ``fetch_rows`` writes) or a
    JSON list."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise CheckError(f"{p}: expected a list of rows")
    return {str(r["instance_id"]): r for r in rows}


def _urllib_get_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "steproute"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_rows(
    dest: str | Path,
    *,
    dataset: str = "SWE-bench/SWE-bench_Verified",
    split: str = "test",
    get_json: Callable[[str], Any] | None = None,
) -> int:
    """Download every row of ``dataset`` from the Hugging Face datasets-server into ``dest``
    (JSONL). For SWE-bench Verified that is 500 rows, about 9 MB. Returns the row count."""
    get = get_json or _urllib_get_json
    base = "https://datasets-server.huggingface.co/rows"
    name = urllib.parse.quote(dataset, safe="")
    rows: list[Any] = []
    while True:
        page = get(
            f"{base}?dataset={name}&config=default&split={split}&offset={len(rows)}&length=100"
        )
        batch = [r["row"] for r in page.get("rows", [])]
        rows += batch
        if len(batch) < 100:
            break
    out = Path(dest)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return len(rows)


class SwebenchBackend:
    """Grade a patch with the official SWE-bench harness (Docker).

    The instance row (with its ``eval_script`` and test lists) comes from ``rows``. When the
    task names an ``image``, the row's ``image`` is replaced by it before the harness sees it:
    the harness takes the image name from the dataset row verbatim, so this is how a native
    arm64 image is used instead of the dataset's x86_64 one.
    """

    name = "swebench"

    def __init__(
        self,
        rows: Mapping[str, Mapping[str, Any]],
        *,
        harness: Sequence[str] = DEFAULT_HARNESS,
        workdir: Path | None = None,
        runner: Runner | None = None,
        model_name: str = "steproute",
        setup_allowance_s: float = 1800.0,
        keep_logs: bool = False,
    ) -> None:
        self.rows = rows
        self.harness = tuple(harness)
        self.workdir = workdir
        self.runner = runner or _default_runner
        self.model_name = model_name
        self.setup_allowance_s = setup_allowance_s
        self.keep_logs = keep_logs

    def gold_patch(self, task: Task) -> str:
        return str(self._row(task)["patch"])

    def _row(self, task: Task) -> Mapping[str, Any]:
        try:
            return self.rows[task.id]
        except KeyError:
            msg = f"no dataset row for {task.id!r}; run `steproute tasks fetch`"
            raise CheckError(msg) from None

    def command(self, task: Task, run_dir: Path, run_id: str, timeout_s: float) -> list[str]:
        return [
            *self.harness,
            "--dataset_name",
            str(run_dir / "dataset.json"),
            "--split",
            "test",
            "--predictions_path",
            str(run_dir / "predictions.jsonl"),
            "--instance_ids",
            task.id,
            "--run_id",
            run_id,
            "--max_workers",
            "1",
            "--timeout",
            str(int(timeout_s)),
        ]

    def _instance_dir(self, run_dir: Path, run_id: str, task: Task) -> Path | None:
        # logs/evaluation from the unreleased harness on, logs/run_evaluation in 5.0.x
        for base in ("logs/evaluation", "logs/run_evaluation"):
            d = run_dir / base / run_id / self.model_name.replace("/", "__") / task.id
            if d.is_dir():
                return d
        return None

    def check(self, task: Task, patch: str, *, timeout_s: float | None = None) -> CheckResult:
        start = time.monotonic()

        def result(status: str, detail: str, log: str = "") -> CheckResult:
            ms = (time.monotonic() - start) * 1000.0
            return CheckResult(task.id, status, self.name, detail, ms, _tail(log))

        row = dict(self._row(task))
        if not patch.strip():
            # The harness skips empty predictions and counts them as unresolved.
            return result("failed", "empty patch: not evaluated, counted as unresolved")
        if task.image:
            row["image"] = task.image
        limit = timeout_s if timeout_s is not None else task.timeout_s
        run_id = f"steproute-{uuid.uuid4().hex[:12]}"
        run_dir = Path(tempfile.mkdtemp(prefix=f"steproute-swebench-{task.id}-", dir=self.workdir))
        try:
            (run_dir / "dataset.json").write_text(json.dumps([row]), encoding="utf-8")
            pred = {"instance_id": task.id, "model_name_or_path": self.model_name}
            pred["model_patch"] = patch
            (run_dir / "predictions.jsonl").write_text(json.dumps(pred) + "\n", encoding="utf-8")
            cmd = self.command(task, run_dir, run_id, limit)
            try:
                r = self.runner(cmd, run_dir, limit + self.setup_allowance_s)
            except FileNotFoundError as exc:
                return result("error", f"harness not found: {exc}")
            if r.timed_out:
                # The harness enforces the test timeout itself; outliving the allowance means
                # it hung in setup (image pull or build), which says nothing about the patch.
                why = "the harness hung past its allowance and was killed"
                return result("error", why, r.output)
            inst = self._instance_dir(run_dir, run_id, task)
            report_path = inst / "report.json" if inst else None
            if report_path is not None and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8")).get(task.id, {})
                if report.get("infra_failure"):
                    why = report.get("infra_failure_reason", "infrastructure failure")
                    return result("error", f"harness: {why}", r.output)
                if not report.get("patch_successfully_applied", False):
                    return result("patch_failed", "the harness could not apply the patch")
                if report.get("resolved"):
                    return result("passed", "resolved (FAIL_TO_PASS and PASS_TO_PASS pass)")
                return result("failed", "not resolved", r.output)
            # No report: the harness stops before grading on a patch failure or a timeout.
            log = ""
            if inst is not None and (inst / "run_instance.log").exists():
                log = (inst / "run_instance.log").read_text(encoding="utf-8", errors="replace")
            if _APPLY_PATCH_FAIL in log:
                return result("patch_failed", "the harness could not apply the patch", log)
            if _TIMED_OUT in log:
                return result("timeout", f"the tests ran longer than {limit:g} s", log)
            return result(
                "error",
                f"the harness wrote no report (exit code {r.returncode})",
                log or r.output,
            )
        finally:
            if not self.keep_logs:
                shutil.rmtree(run_dir, ignore_errors=True)


def run_check(
    task: Task,
    patch: str,
    backend: Backend,
    *,
    timeout_s: float | None = None,
) -> CheckResult:
    """Check ``patch`` on ``task`` with ``backend``; refuse a backend the task was not made for."""
    if task.backend != backend.name:
        raise CheckError(f"task {task.id!r} is a {task.backend} task, not {backend.name}")
    return backend.check(task, patch, timeout_s=timeout_s)
