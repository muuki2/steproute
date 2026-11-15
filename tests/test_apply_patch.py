"""The apply_patch command that task workspaces offer (issue 75).

Run as a command, as the agent runs it: the script on standard input or as one argument, in a
scratch directory. Plus a check that the script parses as Python 3.6, the oldest interpreter in
the task images.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "steproute" / "replay" / "apply_patch.py"


def run(cwd: Path, patch: str, *, as_argument: bool = False) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, str(SCRIPT)]
    if as_argument:
        argv.append(patch)
    return subprocess.run(
        argv,
        cwd=cwd,
        input=None if as_argument else patch,
        capture_output=True,
        text=True,
        timeout=30,
    )


def patch(*lines: str) -> str:
    return "\n".join(["*** Begin Patch", *lines, "*** End Patch"]) + "\n"


MOD = "def f():\n    return 1\n\n\ndef g():\n    return 1\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    d = tmp_path / "repo"
    (d / "pkg").mkdir(parents=True)
    (d / "pkg" / "mod.py").write_text(MOD)
    (d / "old.txt").write_text("bye\n")
    return d


def test_update_on_stdin(repo: Path) -> None:
    r = run(repo, patch("*** Update File: pkg/mod.py", "@@", "-    return 1", "+    return 2"))
    assert r.returncode == 0, r.stderr
    assert r.stdout == "Success. Updated the following files:\nM pkg/mod.py\n"
    # the first match: f, not g
    assert (repo / "pkg" / "mod.py").read_text() == MOD.replace("return 1", "return 2", 1)


def test_update_as_an_argument(repo: Path) -> None:
    p = patch("*** Update File: pkg/mod.py", "@@ def g():", "-    return 1", "+    return 3")
    r = run(repo, p, as_argument=True)
    assert r.returncode == 0, r.stderr
    # the context line after @@ selects g
    assert (repo / "pkg" / "mod.py").read_text() == MOD[: MOD.rindex("1")] + "3\n"


def test_add_and_delete(repo: Path) -> None:
    r = run(
        repo,
        patch("*** Add File: new/dir/a.py", "+x = 1", "+", "+y = 2", "*** Delete File: old.txt"),
    )
    assert r.returncode == 0, r.stderr
    assert "A new/dir/a.py\nD old.txt" in r.stdout
    assert (repo / "new" / "dir" / "a.py").read_text() == "x = 1\n\ny = 2\n"
    assert not (repo / "old.txt").exists()


def test_several_hunks_and_files(repo: Path) -> None:
    (repo / "b.txt").write_text("one\ntwo\nthree\nfour\nfive\n")
    r = run(
        repo,
        patch(
            "*** Update File: pkg/mod.py",
            "@@ def f():",
            "-    return 1",
            "+    return 10",
            "@@ def g():",
            "-    return 1",
            "+    return 20",
            "*** Update File: b.txt",
            "@@",
            " one",
            "-two",
            "+TWO",
            "@@",
            " four",
            "-five",
            "+FIVE",
            "+six",
        ),
    )
    assert r.returncode == 0, r.stderr
    text = (repo / "pkg" / "mod.py").read_text()
    assert "return 10" in text and "return 20" in text and "return 1\n" not in text
    assert (repo / "b.txt").read_text() == "one\nTWO\nthree\nfour\nFIVE\nsix\n"


def test_move_end_of_file_and_unified_ranges(repo: Path) -> None:
    (repo / "c.txt").write_text("x\nend\nx\nend\n")
    r = run(
        repo,
        patch(
            "*** Update File: c.txt",
            "*** Move to: d/c2.txt",
            "@@ -3,2 +3,2 @@",
            " x",
            "-end",
            "+END",
            "*** End of File",
        ),
    )
    assert r.returncode == 0, r.stderr
    assert "M c.txt -> d/c2.txt" in r.stdout
    assert not (repo / "c.txt").exists()
    # anchored at the end: the second "end", not the first
    assert (repo / "d" / "c2.txt").read_text() == "x\nend\nx\nEND\n"


def test_trailing_whitespace_crlf_and_no_final_newline(repo: Path) -> None:
    (repo / "w.txt").write_text("a  \nb\t\nc")
    (repo / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    r = run(
        repo,
        patch(
            "*** Update File: w.txt",
            " a",
            "-b",
            "+B",
            "*** Update File: crlf.txt",
            "-two",
            "+TWO",
        ),
    )
    assert r.returncode == 0, r.stderr
    assert (repo / "w.txt").read_text() == "a  \nB\nc"
    assert (repo / "crlf.txt").read_bytes() == b"one\r\nTWO\r\n"


def test_a_hunk_that_does_not_match_changes_nothing(repo: Path) -> None:
    before = {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()}
    r = run(
        repo,
        patch(
            "*** Add File: added.py",
            "+x = 1",
            "*** Update File: pkg/mod.py",
            "@@",
            " def f():",
            "-    return 42",
            "+    return 2",
        ),
    )
    assert r.returncode == 1
    # the file and the first line that did not match
    assert "pkg/mod.py" in r.stderr and "'    return 42'" in r.stderr
    assert "No files were changed." in r.stderr
    assert {p: p.read_bytes() for p in repo.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("*** Update File: /etc/passwd", "absolute paths are not allowed"),
        ("*** Delete File: ../outside.txt", "leaves the current directory"),
        ("*** Add File: pkg/../../outside.txt", "leaves the current directory"),
        ("*** Delete File: link/secret.txt", "leaves the current directory"),
        ("*** Add File: old.txt", "already exists"),
        ("*** Delete File: missing.txt", "no such file"),
        ("*** Update File: missing.txt", "no such file"),
    ],
)
def test_refusals(repo: Path, section: str, message: str) -> None:
    outside = repo.parent / "outside.txt"
    outside.write_text("keep\n")
    (repo.parent / "elsewhere").mkdir()
    (repo.parent / "elsewhere" / "secret.txt").write_text("keep\n")
    os.symlink(repo.parent / "elsewhere", repo / "link")
    body = [section] + (["+x"] if "Add" in section else [])
    if "Update" in section:
        body += ["@@", "-a", "+b"]
    r = run(repo, patch(*body))
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert outside.read_text() == "keep\n"
    assert (repo.parent / "elsewhere" / "secret.txt").read_text() == "keep\n"
    assert (repo / "old.txt").read_text() == "bye\n"


@pytest.mark.parametrize(
    ("text", "code", "message"),
    [
        ("", 2, "no patch given"),
        ("*** Update File: a\n", 1, "must start with"),
        ("*** Begin Patch\n*** Delete File: old.txt\n", 1, "must end with"),
        ("*** Begin Patch\nhello\n*** End Patch\n", 1, "expected"),
        ("*** Begin Patch\n*** Update File: old.txt\n*bye\n*** End Patch\n", 1, "hunk line"),
    ],
)
def test_malformed_input(repo: Path, text: str, code: int, message: str) -> None:
    r = run(repo, text)
    assert r.returncode == code and message in r.stderr, r.stderr
    assert (repo / "old.txt").read_text() == "bye\n"


def test_the_script_is_python_3_6_syntax() -> None:
    """Task images run Python 3.6 and later; the script must not need anything newer."""
    source = SCRIPT.read_text("utf-8")
    tree = ast.parse(source, feature_version=(3, 6))
    newer = (ast.JoinedStr, ast.NamedExpr, ast.AnnAssign)
    assert not [type(n).__name__ for n in ast.walk(tree) if isinstance(n, newer)]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            assert node.returns is None, node.name
            assert all(a.annotation is None for a in node.args.args), node.name
    # standard library only
    imports = {
        alias.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for alias in n.names
    }
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports <= {"os", "re", "sys"}
