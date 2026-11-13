"""Where the agent's commands run: a task container, or a plain local directory (tests).

Both start from a clean checkout with one commit, so the final patch is ``git diff`` against
it. Every command, git included, goes through ``steproute.tasks.check.run_command``, which
drops git's repository-local variables (``GIT_DIR`` and friends), so nothing here can act on
the repository the runner was started from. Neither passes the runner's environment to the
commands: a command the model writes can never read an API key (``env`` inside the workspace
shows only a fixed, harmless set).
Standard library only.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from steproute.tasks.check import run_command
from steproute.tasks.manifest import Task

__all__ = ["DockerWorkspace", "ExecResult", "LocalWorkspace", "Workspace", "WorkspaceError"]

#: The only environment a command sees (plus ``PATH``, and ``HOME`` in a local workspace).
SAFE_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PAGER": "cat",
    "MANPAGER": "cat",
    "GIT_PAGER": "cat",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}

#: Where the docker CLI runs: never inside the surrounding repository.
_NEUTRAL_CWD = Path(tempfile.gettempdir())

_GIT_IDENT = ["-c", "user.name=steproute", "-c", "user.email=", "-c", "commit.gpgsign=false"]


class WorkspaceError(RuntimeError):
    """The workspace could not be set up or used: an infrastructure error, not the model's."""


@dataclass(frozen=True)
class ExecResult:
    output: str
    returncode: int | None  # None when the command timed out
    timed_out: bool


class Workspace(Protocol):
    def execute(self, command: str, *, timeout_s: float) -> ExecResult: ...

    def diff(self) -> str: ...

    def close(self) -> None: ...


class LocalWorkspace:
    """A fresh copy of a local task's directory, committed once, in a temporary directory."""

    def __init__(self, task: Task, *, workdir: Path | None = None) -> None:
        if task.repo is None or not task.repo.is_dir():
            raise WorkspaceError(f"task {task.id!r}: repo directory {task.repo} does not exist")
        if shutil.which("git") is None:
            raise WorkspaceError("a local workspace needs git on PATH")
        self.root = Path(tempfile.mkdtemp(prefix=f"steproute-replay-{task.id}-", dir=workdir))
        self.path = self.root / "repo"
        shutil.copytree(task.repo, self.path, ignore=shutil.ignore_patterns("__pycache__", ".git"))
        self.env = {
            **SAFE_ENV,
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
        }
        for cmd in (
            ["git", "init", "-q"],
            ["git", "add", "-A"],
            ["git", *_GIT_IDENT, "commit", "-q", "-m", "base", "--allow-empty"],
        ):
            r = run_command(cmd, cwd=self.path, timeout_s=60, env=self.env)
            if r.returncode != 0:
                self.close()
                raise WorkspaceError(f"workspace setup failed: {' '.join(cmd[:2])}")

    def execute(self, command: str, *, timeout_s: float) -> ExecResult:
        r = run_command(["bash", "-c", command], cwd=self.path, timeout_s=timeout_s, env=self.env)
        return ExecResult(r.output, r.returncode, r.timed_out)

    def diff(self) -> str:
        run_command(["git", "add", "-N", "."], cwd=self.path, timeout_s=60, env=self.env)
        r = run_command(["git", "diff", "--no-color"], cwd=self.path, timeout_s=60, env=self.env)
        if r.returncode != 0:
            raise WorkspaceError("git diff failed in the workspace")
        return r.output

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class DockerWorkspace:
    """The task's evaluation image, started detached; commands run with ``docker exec``.

    **Never run against a real daemon in this repository's tests** (Docker is not running
    here): the command lines follow mini-SWE-agent's ``DockerEnvironment``, and the smoke test
    uses ``LocalWorkspace``. ``BASH_ENV`` makes ``bash -c`` source the image's ``.bashrc``, so
    the SWE-bench ``testbed`` conda environment is active.
    """

    def __init__(
        self,
        image: str,
        *,
        docker: str = "docker",
        cwd: str = "/testbed",
        lifetime_s: int = 7200,
        start_timeout_s: float = 600.0,
    ) -> None:
        self.docker = docker
        self.cwd = cwd
        self.name = f"steproute-replay-{uuid.uuid4().hex[:12]}"
        cmd = [
            docker,
            "run",
            "-d",
            "--rm",
            "--name",
            self.name,
            "--network",
            "none",
            "-w",
            cwd,
            image,
            "sleep",
            str(lifetime_s),
        ]
        try:
            r = run_command(cmd, cwd=_NEUTRAL_CWD, timeout_s=start_timeout_s)
        except OSError as exc:
            raise WorkspaceError(
                f"could not start the task container: {type(exc).__name__}"
            ) from None
        if r.timed_out or r.returncode != 0:
            raise WorkspaceError(f"could not start the task container: {r.output.strip()[:300]}")
        base = self.execute(
            f"git {' '.join(_GIT_IDENT)} add -A && git {' '.join(_GIT_IDENT)} commit -q "
            "--allow-empty -m base",
            timeout_s=120,
        )
        if base.returncode != 0:
            self.close()
            raise WorkspaceError("could not commit the container's starting state")

    def execute(self, command: str, *, timeout_s: float) -> ExecResult:
        envs = [
            x
            for k, v in {**SAFE_ENV, "BASH_ENV": "/root/.bashrc"}.items()
            for x in ("-e", f"{k}={v}")
        ]
        cmd = [self.docker, "exec", "-w", self.cwd, *envs, self.name, "bash", "-c", command]
        r = run_command(cmd, cwd=_NEUTRAL_CWD, timeout_s=timeout_s)
        return ExecResult(r.output, r.returncode, r.timed_out)

    def diff(self) -> str:
        r = self.execute("git add -N . && git diff --no-color", timeout_s=120)
        if r.returncode != 0:
            raise WorkspaceError("git diff failed in the container")
        return r.output

    def close(self) -> None:
        try:
            run_command([self.docker, "rm", "-f", self.name], cwd=_NEUTRAL_CWD, timeout_s=120)
        except OSError:
            pass
