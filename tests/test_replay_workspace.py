"""Task workspaces never act on the surrounding repository, and never see the runner's keys.

Git exports ``GIT_DIR`` and friends to hooks; a ``git init`` or ``git commit`` in a scratch
workspace that inherits them acts on the caller's repository instead. Every command the
replay runner runs in a workspace goes through ``steproute.tasks.check.run_command``, which
drops them. The Docker workspace is driven by a stand-in ``docker`` executable (no daemon).
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from steproute.replay.workspace import SAFE_ENV, DockerWorkspace, LocalWorkspace
from steproute.tasks.check import GIT_LOCAL_ENV_VARS, workspace_env
from steproute.tasks.manifest import load_manifest

MANIFEST = Path(__file__).parent / "fixtures" / "replay" / "manifest.toml"


def _outer_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A scratch repository with a linked worktree: (its config file, the worktree git dir)."""
    env = workspace_env()
    outer = tmp_path / "outer"
    ident = ["-c", "user.name=tester", "-c", "user.email="]
    for cmd in (
        ["git", "init", "-q", str(outer)],
        ["git", "-C", str(outer), *ident, "commit", "-q", "--allow-empty", "-m", "base"],
        ["git", "-C", str(outer), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "side"],
    ):
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    return outer / ".git" / "config", outer / ".git" / "worktrees" / "wt"


def _head(repo_git: Path) -> str:
    return subprocess.run(
        ["git", "--git-dir", str(repo_git), "rev-parse", "HEAD"],
        check=True,
        env=workspace_env(),
        capture_output=True,
        text=True,
    ).stdout


def test_local_workspace_leaves_the_callers_repository_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, gitdir = _outer_repo(tmp_path)
    before, head = config.read_text(), _head(gitdir)
    monkeypatch.setenv("GIT_DIR", str(gitdir))
    monkeypatch.setenv("GIT_INDEX_FILE", str(gitdir / "index"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "wt"))
    task = load_manifest(MANIFEST).get("add-a")
    ws = LocalWorkspace(task)  # git init, add, commit
    try:
        r = ws.execute("echo changed >> calc.py && git status --porcelain && env", timeout_s=30)
        assert r.returncode == 0
        assert "calc.py" in r.output
        assert not any(f"{name}=" in r.output for name in GIT_LOCAL_ENV_VARS)
        assert "+changed" in ws.diff()
    finally:
        ws.close()
    assert config.read_text() == before
    assert _head(gitdir) == head


def test_local_workspace_commands_see_only_a_fixed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "-".join(["fake", "host", "secret", "q" * 10])
    monkeypatch.setenv("STEPROUTE_TEST_SECRET", secret)
    ws = LocalWorkspace(load_manifest(MANIFEST).get("add-a"))
    try:
        out = ws.execute("env", timeout_s=30).output
    finally:
        ws.close()
    names = {line.split("=", 1)[0] for line in out.splitlines() if "=" in line}
    assert secret not in out
    # bash adds a few of its own (PWD, SHLVL, _); nothing comes from the host but PATH
    assert names - {"PWD", "SHLVL", "_", "OLDPWD"} <= {*SAFE_ENV, "PATH", "HOME"}


_FAKE_DOCKER = """\
import json, os, sys
log = {log!r}
with open(log, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(),
        "git_env": sorted(k for k in os.environ if k in {names!r})}}) + "\\n")
if sys.argv[1] == "run":
    print("container-id")
sys.exit(0)
"""


def test_docker_workspace_command_lines_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "docker.log"
    fake = tmp_path / "docker"
    fake.write_text(
        f"#!{sys.executable}\n" + _FAKE_DOCKER.format(log=str(log), names=list(GIT_LOCAL_ENV_VARS)),
        "utf-8",
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    _, gitdir = _outer_repo(tmp_path)
    monkeypatch.setenv("GIT_DIR", str(gitdir))
    secret = "-".join(["fake", "host", "secret", "w" * 10])
    monkeypatch.setenv("STEPROUTE_TEST_SECRET", secret)
    repo_root = Path(__file__).resolve().parent.parent

    ws = DockerWorkspace("example/image:tag", docker=str(fake))
    ws.execute("ls", timeout_s=30)
    ws.diff()
    ws.close()

    calls = [json.loads(line) for line in log.read_text().splitlines()]
    verbs = [c["argv"][0] for c in calls]
    assert verbs[0] == "run" and verbs[-1] == "rm" and "exec" in verbs
    run = calls[0]["argv"]
    assert run[:3] == ["run", "-d", "--rm"] and "--network" in run and "none" in run
    for c in calls:
        assert c["git_env"] == []  # the docker CLI never sees GIT_DIR
        assert not Path(c["cwd"]).resolve().is_relative_to(repo_root)
        assert secret not in json.dumps(c["argv"])
        if c["argv"][0] == "exec":
            forwarded = {
                c["argv"][i + 1].split("=", 1)[0] for i, a in enumerate(c["argv"]) if a == "-e"
            }
            assert forwarded == {*SAFE_ENV, "BASH_ENV"}
