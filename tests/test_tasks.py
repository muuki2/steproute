"""Issue 17: task manifests, the pass/fail check (local and swebench backends), the estimate."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from steproute.cli import main
from steproute.core.catalog import load_default_catalog
from steproute.core.labels import label
from steproute.core.schema import ModelProfile, Outcome
from steproute.tasks import (
    CheckError,
    CheckResult,
    LocalBackend,
    ManifestError,
    SwebenchBackend,
    load_default_manifest,
    load_manifest,
    run_check,
)
from steproute.tasks.check import GIT_LOCAL_ENV_VARS, fetch_rows, load_rows, workspace_env
from steproute.tasks.estimate import (
    Assumptions,
    build_plans,
    estimate_plan,
    estimate_run,
    load_assumptions,
    months_needed,
)
from steproute.tasks.manifest import parse_manifest

FIX = Path(__file__).parent / "fixtures" / "tasks"
MANIFEST = FIX / "manifest.toml"


def patch(name: str) -> str:
    return (FIX / f"{name}.patch").read_text()


# --- manifest -----------------------------------------------------------------------------------


def test_default_manifest_is_the_issue_17_selection() -> None:
    m = load_default_manifest()
    assert len(m) == 150 and len(m.pilot()) == 20
    assert m.meta["seed"] == 17 and m.meta["dataset"] == "SWE-bench/SWE-bench_Verified"
    assert len({t.id for t in m}) == 150
    strata = Counter(t.stratum for t in m)
    assert strata == {"easy": 40, "medium": 45, "hard": 50, "unsolved": 15}
    assert Counter(t.stratum for t in m.pilot()) == {
        "easy": 5,
        "medium": 6,
        "hard": 7,
        "unsolved": 2,
    }
    for t in m:
        assert t.backend == "swebench"
        assert t.image and ".arm64." in t.image
        assert not t.id.startswith("pylint-dev__")  # GPL-2.0, excluded
    # Easy and medium pilot tasks are mostly ones local models can hold in 32K (10 of 11; the
    # repository-diversity pass picked one long scikit-learn task).
    assert sum(bool(t.fits32k) for t in m.pilot() if t.stratum in ("easy", "medium")) >= 10


def test_local_manifest_resolves_paths() -> None:
    m = load_manifest(MANIFEST)
    t = m.get("tiny-add")
    assert t.backend == "local" and t.repo == FIX / "tiny_add"
    assert t.gold_patch == FIX / "fix.patch" and t.timeout_s == 60
    assert "tiny-add" in m and "nope" not in m
    with pytest.raises(KeyError):
        m.get("nope")


@pytest.mark.parametrize(
    "data, match",
    [
        ({"task": [{"id": "a", "colour": 1}]}, "unknown key"),
        ({"meta": {"bogus": 1}, "task": [{"id": "a"}]}, "unknown key"),
        ({"task": [{"id": "a"}, {"id": "a"}]}, "duplicate"),
        ({"meta": {"full_size": 2}, "task": [{"id": "a"}]}, "full_size"),
        ({"meta": {"backend": "local"}, "task": [{"id": "a"}]}, "needs 'repo'"),
        ({"meta": {"backend": "k8s"}, "task": [{"id": "a"}]}, "backend"),
        ({"task": []}, "at least one"),
        ({"task": [{"id": "a", "timeout_s": 0}]}, "positive"),
        ({"task": [{"id": "a", "pilot": "yes"}]}, "pilot"),
    ],
)
def test_manifest_is_strict(data: dict, match: str) -> None:
    with pytest.raises(ManifestError, match=match):
        parse_manifest(data)


# --- local backend: the end-to-end check ---------------------------------------------------------


@pytest.fixture
def tiny() -> tuple:
    m = load_manifest(MANIFEST)
    return m.get("tiny-add"), LocalBackend()


def test_local_pass(tiny: tuple) -> None:
    task, backend = tiny
    r = run_check(task, patch("fix"), backend)
    assert r.status == "passed" and r.passed and r.tests_passed is True
    assert r.signals() == {"check_status": "passed", "tests_passed": True}
    assert label(r.signals()).success is True


def test_local_fail(tiny: tuple) -> None:
    task, backend = tiny
    r = run_check(task, patch("wrong"), backend)
    assert r.status == "failed" and r.tests_passed is False
    assert "FAILED" in r.log_tail or "AssertionError" in r.log_tail
    assert label(r.signals()).success is False


def test_local_patch_does_not_apply(tiny: tuple) -> None:
    task, backend = tiny
    r = run_check(task, patch("bad"), backend)
    assert r.status == "patch_failed" and r.tests_passed is False


def test_local_timeout_kills_the_tests(tiny: tuple) -> None:
    task, backend = tiny
    r = run_check(task, patch("hang"), backend, timeout_s=1.0)
    assert r.status == "timeout" and r.tests_passed is False
    assert 900 <= r.duration_ms < 20_000


def test_local_empty_patch_runs_the_unpatched_tests(tiny: tuple) -> None:
    task, backend = tiny
    assert run_check(task, "", backend).status == "failed"


def test_local_leaves_the_source_alone_and_cleans_up(tiny: tuple, tmp_path: Path) -> None:
    task, _ = tiny
    before = (task.repo / "calc.py").read_text()
    backend = LocalBackend(workdir=tmp_path)
    assert backend.check(task, patch("fix")).passed
    assert (task.repo / "calc.py").read_text() == before
    assert list(tmp_path.iterdir()) == []


def test_local_keep_workspace(tiny: tuple, tmp_path: Path) -> None:
    task, _ = tiny
    LocalBackend(workdir=tmp_path, keep_workspace=True).check(task, patch("fix"))
    (ws,) = tmp_path.iterdir()
    assert "a + b" in (ws / "repo" / "calc.py").read_text()


def _outer_repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A scratch repository with a linked worktree; returns (config file, worktree git dir)."""
    env = workspace_env()
    outer = tmp_path / "outer"
    ident = ["-c", "user.name=tester", "-c", "user.email=tester@example.invalid"]
    for cmd in (
        ["git", "init", "-q", str(outer)],
        ["git", "-C", str(outer), *ident, "commit", "-q", "--allow-empty", "-m", "base"],
        ["git", "-C", str(outer), "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "side"],
    ):
        subprocess.run(cmd, check=True, env=env, capture_output=True)
    return outer / ".git" / "config", outer / ".git" / "worktrees" / "wt"


def test_local_check_leaves_the_callers_repository_alone(
    tiny: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Git exports GIT_DIR to hooks. If the workspace's `git init` inherited it, it would act on
    # that repository and set its core.bare, which happened to this project's own repository
    # when the test suite ran from the pre-push hook.
    config, gitdir = _outer_repo_with_worktree(tmp_path)
    before = config.read_text()
    assert "bare = false" in before
    monkeypatch.setenv("GIT_DIR", str(gitdir))
    monkeypatch.setenv("GIT_INDEX_FILE", str(gitdir / "index"))
    task, _ = tiny
    work = tmp_path / "work"
    work.mkdir()
    assert LocalBackend(workdir=work).check(task, patch("fix")).passed
    assert config.read_text() == before


def test_workspace_env_drops_every_git_local_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GIT_LOCAL_ENV_VARS:
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("STEPROUTE_KEEP_ME", "1")
    env = workspace_env()
    assert not set(GIT_LOCAL_ENV_VARS) & set(env)
    assert env["STEPROUTE_KEEP_ME"] == "1"
    assert "GIT_DIR" not in workspace_env({"GIT_DIR": "x", "A": "b"})
    # The list is the one git itself reports.
    listed = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True, env=env
    ).stdout.split()
    assert set(listed) <= set(GIT_LOCAL_ENV_VARS)


def test_outcome_labelled_from_the_check(tiny: tuple) -> None:
    task, backend = tiny
    r = backend.check(task, patch("fix"))
    o = Outcome(decision_id="d", ts=0.0, success=None, signals=r.signals())
    assert label(o.signals).reason == "task_check:tests_passed"


def test_run_check_refuses_the_wrong_backend(tiny: tuple) -> None:
    task, _ = tiny
    with pytest.raises(CheckError, match="local task"):
        run_check(task, "", SwebenchBackend({}))


def test_missing_repo_is_a_check_error(tmp_path: Path) -> None:
    m = parse_manifest(
        {"meta": {"backend": "local"}, "task": [{"id": "x", "repo": "gone", "test_cmd": ["true"]}]},
        path=tmp_path / "m.toml",
    )
    with pytest.raises(CheckError, match="does not exist"):
        LocalBackend().check(m.get("x"), "")


def test_check_result_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        CheckResult("t", "maybe", "local")
    err = CheckResult("t", "error", "swebench")
    assert err.tests_passed is None and err.signals() == {"check_status": "error"}
    assert label(err.signals()).success is None


# --- swebench backend against a stand-in harness ------------------------------------------------

ROW = {
    "instance_id": "astropy__astropy-12907",
    "image": "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
    "patch": "diff --git a/x b/x\n",
    "eval_script": "#!/bin/bash\n",
    "FAIL_TO_PASS": ["t1"],
    "PASS_TO_PASS": [],
}
ARM = "swebench/sweb.eval.arm64.astropy_1776_astropy-12907:latest"


def swe_task():
    m = parse_manifest({"task": [{"id": ROW["instance_id"], "image": ARM, "timeout_s": 900}]})
    return m.get(ROW["instance_id"])


def swe_backend(tmp_path: Path) -> SwebenchBackend:
    return SwebenchBackend(
        {ROW["instance_id"]: ROW},
        harness=[sys.executable, str(FIX / "fake_harness.py")],
        workdir=tmp_path,
        keep_logs=True,
    )


@pytest.mark.parametrize(
    "marker, status",
    [
        ("ok", "passed"),
        ("NEWLAYOUT", "passed"),
        ("UNRESOLVED", "failed"),
        ("NOAPPLY", "patch_failed"),
        ("HANG", "timeout"),
        ("INFRA", "error"),
        ("CRASH", "error"),
    ],
)
def test_swebench_report_mapping(tmp_path: Path, marker: str, status: str) -> None:
    r = run_check(swe_task(), f"diff --git a/f b/f\n# {marker}\n", swe_backend(tmp_path))
    assert r.status == status, r.detail
    assert r.backend == "swebench"


def test_swebench_passes_the_arm64_image_and_one_instance(tmp_path: Path) -> None:
    swe_backend(tmp_path).check(swe_task(), "diff --git a/f b/f\n")
    (run_dir,) = tmp_path.iterdir()
    seen = json.loads((run_dir / "seen.json").read_text())
    assert seen == {"image": ARM, "ids": [ROW["instance_id"]], "timeout": "900"}


def test_swebench_empty_patch_is_unresolved_without_running(tmp_path: Path) -> None:
    r = swe_backend(tmp_path).check(swe_task(), "  \n")
    assert r.status == "failed" and "empty patch" in r.detail
    assert list(tmp_path.iterdir()) == []


def test_swebench_missing_harness_and_row(tmp_path: Path) -> None:
    b = SwebenchBackend({ROW["instance_id"]: ROW}, harness=["/nonexistent/harness"])
    assert b.check(swe_task(), "diff\n").status == "error"
    with pytest.raises(CheckError, match="no dataset row"):
        SwebenchBackend({}).check(swe_task(), "diff\n")
    assert b.gold_patch(swe_task()) == ROW["patch"]


def test_fetch_rows_pages_and_round_trips(tmp_path: Path) -> None:
    urls: list[str] = []

    def get(url: str) -> dict:
        urls.append(url)
        offset = int(url.split("offset=")[1].split("&")[0])
        n = 100 if offset < 200 else 7
        return {"rows": [{"row": {"instance_id": f"i{offset + k}"}} for k in range(n)]}

    dest = tmp_path / "rows.jsonl"
    assert fetch_rows(dest, get_json=get) == 207
    assert len(urls) == 3 and "SWE-bench%2FSWE-bench_Verified" in urls[0]
    rows = load_rows(dest)
    assert len(rows) == 207 and rows["i206"] == {"instance_id": "i206"}


# --- estimate -----------------------------------------------------------------------------------


def _claude45(pid: str, price_in: float, price_out: float) -> ModelProfile:
    return ModelProfile(
        id=pid,
        provider="anthropic",
        price_in=price_in,
        price_out=price_out,
        context_window=200_000,
        cache_write_mult=1.25,
        cache_read_mult=0.1,
        efforts=("high",),
        cache_min_tokens=1024,
    )


# 20 quantiles of api_calls on all 500 Verified instances and the published mean $/instance
# (SWE-bench/experiments, mini-SWE-agent 2.0.0 runs, at Claude 4.5/4.6 prices).
CALIBRATION = {
    "haiku45": (1, 5, 0.331, [25, 32, 37, 41, 44, 47, 50, 54, 58, 61,
                              65, 69, 71, 75, 78, 83, 89, 95, 106, 137]),
    "sonnet45": (3, 15, 0.658, [23, 27, 30, 32, 34, 36, 38, 40, 41, 44,
                                46, 48, 50, 52, 55, 59, 63, 68, 77, 96]),
    "opus45": (5, 25, 0.754, [13, 16, 19, 20, 21, 23, 24, 25, 27, 28,
                              30, 32, 34, 36, 38, 41, 45, 51, 59, 74]),
    "opus46": (5, 25, 0.552, [7, 9, 11, 13, 15, 16, 18, 19, 21, 22,
                              24, 26, 29, 31, 35, 38, 45, 51, 62, 80]),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(CALIBRATION))
def test_central_scenario_reproduces_published_costs(name: str) -> None:
    pin, pout, measured, quantiles = CALIBRATION[name]
    base = load_assumptions()
    a = Assumptions(
        scenarios=base.scenarios,
        steps={name: tuple(quantiles)},
        output_mult=base.output_mult,
        steps_mult=base.steps_mult,
        local={},
        run_cost_cap_usd=None,
    )
    est = estimate_run(_claude45(name, pin, pout), "high", a, "central").usd
    assert abs(est - measured) / measured < 0.25, (est, measured)


def test_estimate_orders_and_sensitivities() -> None:
    cat = load_default_catalog()
    a = load_assumptions()
    sonnet = cat.get("claude-sonnet-5-5")
    low, mid, high = (estimate_run(sonnet, "high", a, s).usd for s in ("low", "central", "high"))
    assert low < mid < high
    assert estimate_run(sonnet, "max", a).usd > estimate_run(sonnet, "medium", a).usd
    assert estimate_run(sonnet, "high", a, caching=False).usd > 3 * mid
    # every paid run stops at the cost cap (plus at most the step that crossed it)
    for p in cat:
        for e in p.efforts:
            assert estimate_run(p, e, a, "high").usd <= 3.0 + 0.5
    local = estimate_run(cat.get("qwen3-coder:30b"), "medium", a)
    assert local.usd == 0 and local.seconds > a.eval_seconds and local.capped_share > 0


def test_pilot_fits_the_caps_with_margin() -> None:
    cat = load_default_catalog()
    plans = build_plans(cat)
    est = estimate_plan(plans["pilot"], cat, load_assumptions())
    assert est.usd["anthropic"]["central"] < 0.6 * 80
    assert est.usd["deepseek"]["central"] < 20
    assert est.usd["ollama"]["high"] == 0
    dense = estimate_plan(plans["dense"], cat, load_assumptions())
    sparse = estimate_plan(plans["sparse"], cat, load_assumptions())
    assert sparse.usd["anthropic"]["central"] < dense.usd["anthropic"]["central"]
    assert dense.runs["anthropic"] == 150 * 20 and dense.runs["deepseek"] == 150 * 8
    assert months_needed(81, 80) == 2 and months_needed(10, 80) == 1


# --- CLI ----------------------------------------------------------------------------------------


def test_cli_check_exit_codes(capsys: pytest.CaptureFixture[str]) -> None:
    base = ["tasks", "check", "tiny-add", "--manifest", str(MANIFEST)]
    assert main([*base, "--patch", "gold"]) == 0
    assert main([*base, "--patch", str(FIX / "wrong.patch"), "--json"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert json.loads(out[-1])["tests_passed"] is False
    assert main([*base, "--patch", "empty"]) == 1
    assert main(["tasks", "check", "nope", "--manifest", str(MANIFEST), "--patch", "empty"]) == 2


def test_cli_list_and_estimate(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["tasks", "list", "--pilot"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 20
    assert main(["tasks", "estimate", "--plan", "pilot", "--per-config"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out and "claude-fable-5-2" in out


def test_cli_swebench_check_needs_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(
        [
            "tasks",
            "check",
            "astropy__astropy-12907",
            "--patch",
            "gold",
            "--rows",
            str(tmp_path / "none.jsonl"),
        ]
    )
    assert rc == 2 and "tasks fetch" in capsys.readouterr().err
