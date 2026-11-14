"""The dataset release (issue 19), end to end on a replay output directory made by the replay
runner against the loopback mock upstream. No provider, no network beyond loopback.

Acceptance criteria: the loader reproduces the datasheet's summary table
(``test_loader_reproduces_the_datasheet_table``) and no task is in two splits
(``test_no_task_is_in_two_splits``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("litellm")
pytest.importorskip("pyarrow")

from proxy_harness import NetworkGuard  # noqa: E402
from replay_harness import (  # noqa: E402
    FIXTURES,
    ScriptedAgent,
    config_for,
    environ_with,
    plan_file,
    runner,
)
from steproute.cli import main  # noqa: E402
from steproute.collectors.mock_upstream import MockUpstream  # noqa: E402
from steproute.core.catalog import load_default_catalog  # noqa: E402
from steproute.dataset import (  # noqa: E402
    ReleaseError,
    ReleaseRefused,
    build_release,
    load,
)
from steproute.dataset.splits import HOLDOUT_SPLIT, TASK_SPLITS  # noqa: E402
from steproute.replay.plan import load_plan_file  # noqa: E402
from steproute.tasks.estimate import load_assumptions  # noqa: E402
from steproute.tasks.manifest import TaskManifest, load_manifest  # noqa: E402

MANIFEST = FIXTURES.parent / "dataset" / "manifest.toml"
CONFIGS = [
    ("qwen3-coder:30b", "medium"),
    ("claude-haiku-5-5", "low"),
    ("claude-haiku-5-5", "high"),
    ("devstral:24b", "medium"),
]


@pytest.fixture(scope="module")
def manifest() -> TaskManifest:
    return load_manifest(MANIFEST)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory, manifest: TaskManifest) -> Path:
    tmp = tmp_path_factory.mktemp("replay")
    out = tmp / "out"
    plan = load_plan_file(
        plan_file(tmp, CONFIGS), manifest, load_default_catalog(), load_assumptions()
    )
    with pytest.MonkeyPatch.context() as mp, MockUpstream() as mock:
        NetworkGuard().install(mp)
        mock.set_responder(ScriptedAgent())
        code = runner(plan, manifest, config_for(mock), out, environ=environ_with("anthropic")).run(
            execute=True
        )
    assert code == 0
    return out


@pytest.fixture(scope="module")
def release_dir(
    tmp_path_factory: pytest.TempPathFactory, run_dir: Path, manifest: TaskManifest
) -> Path:
    out = tmp_path_factory.mktemp("release") / "release"
    build_release(
        [run_dir], out, manifest=manifest, catalog=load_default_catalog(), allow_mock=True
    )
    return out


def _copy(run_dir: Path, tmp_path: Path) -> Path:
    dst = tmp_path / "copy"
    shutil.copytree(run_dir, dst)
    return dst


def _build(src: Path, tmp_path: Path, manifest: TaskManifest, **kw: Any) -> Path:
    out = tmp_path / "rel"
    kw.setdefault("allow_mock", True)
    build_release([src], out, manifest=manifest, catalog=load_default_catalog(), **kw)
    return out


def _edit_lines(path: Path, fn: Any) -> None:
    lines = path.read_text("utf-8").splitlines()
    path.write_text("\n".join(fn(lines)) + "\n", "utf-8")


# --- the release ----------------------------------------------------------------------------


def test_release_layout_and_checksums(release_dir: Path) -> None:
    manifest = json.loads((release_dir / "manifest.json").read_text("utf-8"))
    assert manifest["format"] == "steproute-dataset"
    assert manifest["mock"] is True
    assert {
        "DATASHEET.md",
        "splits.json",
        "schema/trace.schema.json",
        "schema/tables.json",
        "data/runs.parquet",
        *(f"data/steps-{s}.parquet" for s in (*TASK_SPLITS, HOLDOUT_SPLIT)),
    } == set(manifest["files"])
    assert load(release_dir).manifest["counts"] == manifest["counts"]


def test_loader_reproduces_the_datasheet_table(release_dir: Path) -> None:
    rel = load(release_dir)
    assert rel.summary_block() == rel.datasheet_block()
    # the table is computed, not typed: check one row against the source grid by hand
    runs = [r for r in rel.runs if r["model"] == "claude-haiku-5-5" and r["effort"] == "low"]
    passed = sum(1 for r in runs if r["success"])
    assert f"| {len(runs)} | {len(runs)} | {passed} |" in rel.datasheet
    assert "MOCK DATA" in rel.datasheet


def test_no_task_is_in_two_splits(release_dir: Path) -> None:
    rel = load(release_dir)
    seen: dict[str, str] = {}
    for name in TASK_SPLITS:
        for tid in rel.splits[name].tasks:
            assert seen.setdefault(tid, name) == name, tid
        for row in rel.splits[name].steps:
            assert row["task_id"] in rel.splits[name].tasks
    assert len(seen) == len(rel.tasks) - len(set(rel.tasks) - set(seen))
    # the held-out model is in its own split and nowhere else
    held = rel.splits[HOLDOUT_SPLIT]
    assert {r["model"] for r in held.runs} == {"devstral:24b"}
    assert {r["model"] for r in held.steps} == {"devstral:24b"}
    for name in TASK_SPLITS:
        assert "devstral:24b" not in {r["model"] for r in rel.splits[name].runs}
        assert "devstral:24b" not in {r["model"] for r in rel.splits[name].steps}
    assert {r["task_id"] for r in held.runs} == set(rel.tasks)


def test_release_has_both_outcomes_and_only_scrubbed_replay_data(release_dir: Path) -> None:
    rel = load(release_dir)
    assert {r["success"] for r in rel.runs} == {True, False}
    assert {r["source"] for r in rel.steps} == {"replay"}
    assert all(r["has_outcome"] for r in rel.steps)
    assert all(r["ts"] % 3600 == 0 for r in rel.steps)
    ids = {r["decision_id"] for r in rel.steps}
    assert len(ids) == len(rel.steps) and all(len(i) == 32 for i in ids)
    assert not any(r["session_id"].startswith("replay-") for r in rel.steps)
    # a step's label is its run's label
    by_run = {r["run_id"]: r["success"] for r in rel.runs}
    assert all(s["success"] == by_run[s["run_id"]] for s in rel.steps)


def test_build_is_reproducible(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest, release_dir: Path
) -> None:
    again = _build(run_dir, tmp_path, manifest)
    a = json.loads((release_dir / "manifest.json").read_text("utf-8"))["files"]
    b = json.loads((again / "manifest.json").read_text("utf-8"))["files"]
    assert a == b


def test_check_command_and_cli_build(
    tmp_path: Path, run_dir: Path, release_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["dataset", "check", str(release_dir)]) == 0
    assert "ok:" in capsys.readouterr().out
    out = tmp_path / "cli-release"
    args = [
        "dataset",
        "build",
        "--from",
        str(run_dir),
        "--out",
        str(out),
        "--manifest",
        str(MANIFEST),
    ]
    assert main(args) == 2  # a mock run is refused without --allow-mock
    err = capsys.readouterr().err
    assert "mock" in err and "nothing was written" in err and not out.exists()
    assert main([*args, "--allow-mock"]) == 0
    assert (out / "DATASHEET.md").is_file()


def test_tampering_is_detected(tmp_path: Path, release_dir: Path) -> None:
    dst = tmp_path / "tampered"
    shutil.copytree(release_dir, dst)
    p = dst / "DATASHEET.md"
    p.write_text(p.read_text("utf-8") + "x", "utf-8")
    with pytest.raises(ReleaseError, match="checksum"):
        load(dst)
    (dst / "extra.txt").write_text("x", "utf-8")
    with pytest.raises(ReleaseError):
        load(dst)


# --- refusals: loud, named, nothing partial -------------------------------------------------


def _refused(src: Path, tmp_path: Path, manifest: TaskManifest, **kw: Any) -> ReleaseRefused:
    with pytest.raises(ReleaseRefused) as exc:
        _build(src, tmp_path, manifest, **kw)
    assert not (tmp_path / "rel").exists()
    assert not (tmp_path / "rel.partial").exists()
    return exc.value


def test_a_personal_record_is_refused_with_file_and_id(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    f = next((src / "traces").glob("*.jsonl"))
    victim = ""

    def edit(lines: list[str]) -> list[str]:
        nonlocal victim
        for i, line in enumerate(lines):
            if '"type":"decision"' in line:
                victim = json.loads(line)["data"]["decision_id"]
                lines[i] = line.replace('"source":"replay"', '"source":"personal"')
                break
        return lines

    _edit_lines(f, edit)
    err = str(_refused(src, tmp_path, manifest))
    assert f.name in err and victim in err and "not a replay record" in err


def test_an_orphan_outcome_is_refused(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    f = next((src / "traces").glob("*.jsonl"))
    first = next(
        i for i, ln in enumerate(f.read_text("utf-8").splitlines()) if '"type":"decision"' in ln
    )
    _edit_lines(f, lambda lines: [ln for i, ln in enumerate(lines) if i != first])
    err = str(_refused(src, tmp_path, manifest))
    assert "outcome without a decision" in err


def test_unknown_plan_identity_is_refused(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    (src / "plan.json").unlink()
    assert "plan.json" in str(_refused(src, tmp_path, manifest))
    (src / "plan.json").write_text(json.dumps({"plan": "x"}), "utf-8")
    assert "incomplete" in str(_refused(src, tmp_path, manifest))
    doc = json.loads((run_dir / "plan.json").read_text("utf-8"))
    doc["manifest"] = "some-other-task-set"
    doc.pop("plans")
    (src / "plan.json").write_text(json.dumps(doc), "utf-8")
    assert "some-other-task-set" in str(_refused(src, tmp_path, manifest))


def test_unknown_schema_version_is_refused(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    f = next((src / "traces").glob("*.jsonl"))
    _edit_lines(f, lambda lines: [lines[0].replace('"v":"0.1"', '"v":"9.9"'), *lines[1:]])
    err = str(_refused(src, tmp_path, manifest))
    assert f.name in err and "schema version" in err and "9.9" in err


def test_a_secret_in_a_record_fails_the_scrubber(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    f = next((src / "traces").glob("*.jsonl"))
    fake = "sk-" + "a1" * 12  # built at run time: no credential-shaped literal in the repository
    _edit_lines(
        f,
        lambda lines: [
            lines[0].replace('"policy":"replay-fixed"', f'"policy":"{fake}"'),
            *lines[1:],
        ],
    )
    err = str(_refused(src, tmp_path, manifest))
    assert f.name in err and "privacy scrubber" in err and fake not in err


def test_the_held_out_model_in_the_pool_store_is_refused(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    src = _copy(run_dir, tmp_path)
    held = next((src / "traces-holdout").glob("*.jsonl"))
    shutil.copy(held, src / "traces" / "traces-2000-01-01.jsonl")
    err = str(_refused(src, tmp_path, manifest))
    assert "wrong store" in err


def test_a_task_outside_the_task_set_is_refused(tmp_path: Path, run_dir: Path) -> None:
    other = load_manifest(FIXTURES / "manifest.toml")  # names a different task set
    err = str(_refused(run_dir, tmp_path, other))
    assert "dataset-fixture" in err or "replay-smoke" in err


def test_an_existing_output_is_never_overwritten(
    tmp_path: Path, run_dir: Path, manifest: TaskManifest
) -> None:
    out = tmp_path / "rel"
    out.mkdir()
    with pytest.raises(ReleaseRefused, match="already exists"):
        build_release(
            [run_dir], out, manifest=manifest, catalog=load_default_catalog(), allow_mock=True
        )
