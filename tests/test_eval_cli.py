"""`steproute eval` end to end on generated data, the report's shape, and the plot."""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from steproute.cli import main
from steproute.eval.args import add_arguments
from steproute.eval.report import DEFINITIONS, METRICS_VERSION
from steproute.eval.results import EvalSettings
from steproute.eval.synthetic import write_example


@pytest.fixture(scope="module")
def example(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_example(tmp_path_factory.mktemp("synthetic"), n_tasks=150, seed=2)


def _run(example: dict[str, Path], out: Path, *extra: str) -> int:
    return main(
        [
            "eval",
            "--policy", "fixed:medium:cloud-mid",
            "--policy", "cascade:medium:local-small,low:cloud-small",
            "--policy", f"rules:{example['rules']}",
            "--policy", "oracle",
            "--data", str(example["grid"]),
            "--data", str(example["traces"]),
            "--catalog", str(example["catalog"]),
            "--out", str(out),
            "--bootstrap", "200",
            *extra,
        ]
    )  # fmt: skip


def test_cli_end_to_end(
    example: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "report.json"
    assert _run(example, out) == 0
    printed = capsys.readouterr()
    assert "beats every single configuration" in printed.out
    assert "the oracle needs a grid" in printed.err

    report = json.loads(out.read_text())
    assert report["format"] == "steproute-eval-report"
    assert report["metrics_version"] == METRICS_VERSION
    assert report["definitions"] == DEFINITIONS
    assert report["settings"]["bootstrap_replicates"] == 200
    grid, logged = report["evaluations"]
    assert (grid["kind"], grid["level"]) == ("grid", "task")
    assert (logged["kind"], logged["level"]) == ("logged", "step")
    assert "NOT measured here" in grid["scope"]

    # Provenance: names, counts, what was unlabelled or unsupported.
    prov = grid["provenance"]
    assert prov["source"] == str(example["grid"]) and len(prov["sha256"]) == 64
    assert prov["runs"] == 150 * 6 and prov["tasks_evaluated"] == 150
    assert prov["unlabelled_runs"] == 0
    lprov = logged["provenance"]
    assert lprov["decisions"] == 150 * 2 * 4
    assert lprov["catalog"] == str(example["catalog"])
    assert lprov["candidate_set_issues"] == {}

    # Every estimate has a value, an interval and n; the frontier and verdicts are present.
    pol = {p["name"]: p for p in grid["policies"]}
    sr = pol["fixed:medium:cloud-mid"]["metrics"]["success_rate"]
    assert set(sr) >= {"value", "ci", "n"}
    assert len(grid["frontier"]["configurations"]) == 6
    assert grid["frontier"]["envelope"][0]["cost"] == 0.0
    verdicts = {(v["kind"], v["policy"]): v for v in report["verdicts"]}
    v = verdicts[("grid", "cascade:medium:local-small,low:cloud-small")]
    assert v["status"] == "beats_frontier" and v["beats_every_single_configuration"] is True
    assert v["sentence"] in report["summary"]
    assert verdicts[("grid", "oracle")]["hindsight"] is True
    # On the logs the cascade was never taken: refused, with the reason, not dropped.
    lv = verdicts[("logged", "cascade:medium:local-small,low:cloud-small")]
    assert lv["status"] == "not_shown" and "never took" in lv["sentence"]
    excluded = set(logged["frontier"]["excluded"])
    assert not excluded & {v["label"] for v in logged["frontier"]["envelope"]}
    lpol = {p["name"]: p for p in logged["policies"]}
    assert set(lpol["fixed:medium:cloud-mid"]["estimators"]) == {"ips", "snips", "dr"}
    assert grid["ranking"][0]["rank"] == 1

    # One plot per dataset, valid SVG naming every policy, with a text legend.
    for i in (1, 2):
        svg = tmp_path / f"report-{i}.svg"
        root = ET.parse(svg).getroot()
        text = " ".join(t.text or "" for t in root.iter("{http://www.w3.org/2000/svg}text"))
        assert "fixed:medium:cloud-mid" in text
        assert "single configuration" in text and "frontier" in text
    grid_svg = (tmp_path / "report-1.svg").read_text()
    assert "beats the frontier" in grid_svg and "hindsight reference" in grid_svg
    assert "prefers-color-scheme: dark" in grid_svg


def test_cli_is_deterministic(example: dict[str, Path], tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert _run(example, a, "--no-plot") == 0
    assert _run(example, b, "--no-plot") == 0
    ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
    for j in (ja, jb):
        j.pop("generated_at")
        j.pop("command")
    assert ja == jb
    assert not (tmp_path / "a.svg").exists()


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = str(tmp_path / "r.json")
    assert main(["eval", "--policy", "bogus", "--data", out, "--out", out]) == 2
    assert "unknown policy" in capsys.readouterr().err
    missing = str(tmp_path / "missing.jsonl")
    assert main(["eval", "--policy", "fixed:low:m", "--data", missing, "--out", out]) == 2
    assert "no such file" in capsys.readouterr().err
    assert main(["eval", "--policy", "fixed:bad:m", "--data", missing, "--out", out]) == 2


def test_cli_defaults_match_settings() -> None:
    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args(["--policy", "oracle", "--data", "x", "--out", "y"])
    d = EvalSettings()
    for name in ("confidence", "bootstrap", "seed", "min_ess", "min_clusters", "min_margin"):
        assert getattr(args, name) == getattr(d, name), name
    assert args.headline == d.headline and args.clip is None and args.cost_weight is None


def test_eval_import_is_stdlib_only() -> None:
    code = "import sys, steproute.eval, steproute.cli; print('\\n'.join(sys.modules))"
    base = "import sys; print('\\n'.join(sys.modules))"
    run = [sys.executable, "-c"]
    loaded = set(
        subprocess.run([*run, code], capture_output=True, text=True, check=True).stdout.split()
    )
    baseline = set(
        subprocess.run([*run, base], capture_output=True, text=True, check=True).stdout.split()
    )
    tops = {m.split(".")[0] for m in loaded - baseline}
    assert not sorted(tops - set(sys.stdlib_module_names) - {"steproute"})
