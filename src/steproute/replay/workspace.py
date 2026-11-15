"""Where the agent's commands run: a task container, or a plain local directory (tests).

Both start from a clean checkout with one commit, so the final patch is ``git diff`` against
it. Both put an ``apply_patch`` command (``steproute.replay.apply_patch``) on the commands'
``PATH``, outside the repository, so it never shows in the diff. Every command, git included,
goes through ``steproute.tasks.check.run_command``, which drops git's repository-local
variables (``GIT_DIR`` and friends), so nothing here can act on the repository the runner was
started from. Neither passes the runner's environment to the
commands: a command the model writes can never read an API key (``env`` inside the workspace
shows only a fixed, harmless set).
Standard library only.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from steproute.tasks.check import run_command
from steproute.tasks.manifest import Task

__all__ = [
    "APPLY_PATCH_PATH",
    "DockerWorkspace",
    "ExecResult",
    "LocalWorkspace",
    "Workspace",
    "WorkspaceError",
    "apply_patch_source",
]

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


#: Where a task container gets the ``apply_patch`` command: on the image's default ``PATH``,
#: outside ``/testbed``.
APPLY_PATCH_PATH = "/usr/local/bin/apply_patch"

#: The interpreters a container tries for ``apply_patch``, in order. The image's system Python
#: comes first: it is the same in every SWE-bench image and is not the task's environment, so
#: nothing the model installs or removes in ``testbed`` changes the command. ``python3`` on
#: ``PATH`` is the last resort (the script runs on Python 3.6 and later).
_CONTAINER_PYTHONS = ("/usr/bin/python3", "/opt/miniconda3/bin/python3", "/usr/local/bin/python3")


def apply_patch_source() -> str:
    """The ``apply_patch`` script, without its ``#!`` line."""
    return (Path(__file__).parent / "apply_patch.py").read_text("utf-8")


def _install_script(python: str) -> str:
    return f"#!{python}\n{apply_patch_source()}"


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
        # apply_patch in a private bin directory beside the repository, run by this Python.
        bindir = self.root / "bin"
        bindir.mkdir()
        tool = bindir / "apply_patch"
        tool.write_text(_install_script(sys.executable), "utf-8")
        tool.chmod(0o755)
        self.env = {
            **SAFE_ENV,
            "PATH": os.pathsep.join([str(bindir), os.environ.get("PATH", "/usr/bin:/bin")]),
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
        self._install_apply_patch()
        base = self.execute(
            f"git {' '.join(_GIT_IDENT)} add -A && git {' '.join(_GIT_IDENT)} commit -q "
            "--allow-empty -m base",
            timeout_s=120,
        )
        if base.returncode != 0:
            self.close()
            raise WorkspaceError("could not commit the container's starting state")

    def _install_apply_patch(self) -> None:
        """Write ``apply_patch`` to ``APPLY_PATCH_PATH`` in the container, run by the first
        interpreter of ``_CONTAINER_PYTHONS`` it has. Plain ``sh``, without the image's
        ``.bashrc``, so the task's environment plays no part."""
        tries = " ".join(_CONTAINER_PYTHONS)
        target = APPLY_PATCH_PATH
        script = (
            f'py=; for p in {tries}; do if [ -x "$p" ]; then py=$p; break; fi; done; '
            'if [ -z "$py" ]; then py="/usr/bin/env python3"; fi; '
            f"mkdir -p {Path(target).parent} && "
            f'{{ echo "#!$py"; cat; }} > {target} && chmod 755 {target}'
        )
        cmd = [self.docker, "exec", "-i", self.name, "sh", "-c", script]
        try:
            r = run_command(cmd, cwd=_NEUTRAL_CWD, timeout_s=120, stdin=apply_patch_source())
        except OSError as exc:
            self.close()
            raise WorkspaceError(f"could not install apply_patch: {type(exc).__name__}") from None
        if r.timed_out or r.returncode != 0:
            self.close()
            raise WorkspaceError(f"could not install apply_patch: {r.output.strip()[:300]}")

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
