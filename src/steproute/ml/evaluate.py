"""Evaluate a trained tabular model: calibration metrics, and the evaluator's report.

Two outputs, both on the model's own ``test`` tasks (never read in training):

- ``calibration_report``: accuracy and calibration of the probabilities, before and after
  calibration, with task-cluster bootstrap intervals (``docs/MODEL-CARD-TABULAR.md``).
- ``evaluator_report``: the predictor inside ``ScoredPolicy``, scored by ``steproute.eval``
  against every single configuration, next to the rule baseline.

Anything made from synthetic data is stamped ``SYNTHETIC DATA. NOT A RESULT.`` in the JSON, in
every verdict sentence, in the SVG and in the Markdown, so it cannot be mistaken for a result.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.policy import ScoredPolicy
from steproute.core.rules import RuleConfigError
from steproute.eval.data import GridData
from steproute.eval.grid import evaluate_grid
from steproute.eval.plot import render_svg
from steproute.eval.report import build_report
from steproute.eval.results import EvalSettings, Evaluation
from steproute.eval.stats import cluster_bootstrap, cluster_index, dot, percentile_interval
from steproute.eval.targets import (
    DistributionTarget,
    TargetPolicy,
    as_target,
    parse_policy_spec,
)
from steproute.ml.calibration import ECE_BINS, auc, brier, ece, log_loss, reliability
from steproute.ml.data import Dataset, build_dataset
from steproute.ml.features import FeatureBuilder
from steproute.ml.model import TabularModel, TabularPredictor
from steproute.ml.stamp import STAMP, is_synthetic

__all__ = [
    "calibration_markdown",
    "calibration_report",
    "evaluator_report",
    "stamp_report",
    "stamp_svg",
]

_EPS = 1e-12
_WINDOW = 0.05

#: Row groups reported: a first step (nothing failed before), and a step after one failure.
_GROUPS: tuple[tuple[str, str, Callable[[int], bool]], ...] = (
    ("first_step", "first step (no earlier failure)", lambda h: h == 0),
    ("after_one_failure", "second step, after one failed attempt", lambda h: h == 1),
)


# --- calibration metrics with bootstrap -----------------------------------------------------


class _Accumulator:
    """Per-task sums of everything the metrics need, so a bootstrap replicate is a dot product
    of task multiplicities with these sums rather than a pass over every row."""

    def __init__(
        self, tasks: Sequence[str], p: Sequence[float], y: Sequence[int], taus: Sequence[float]
    ):
        self.index, self.n_tasks = cluster_index(tasks)
        c = self.n_tasks
        self.rows = [0.0] * c
        self.sq = [0.0] * c
        self.ll = [0.0] * c
        self.sy = [0.0] * c
        self.sp = [0.0] * c
        self.b_cnt = [[0.0] * c for _ in range(ECE_BINS)]
        self.b_sp = [[0.0] * c for _ in range(ECE_BINS)]
        self.b_sy = [[0.0] * c for _ in range(ECE_BINS)]
        self.taus = tuple(taus)
        self.w_cnt = [[0.0] * c for _ in taus]
        self.w_sp = [[0.0] * c for _ in taus]
        self.w_sy = [[0.0] * c for _ in taus]
        for t, pi, yi in zip(self.index, p, y, strict=True):
            q = min(max(pi, _EPS), 1.0 - _EPS)
            self.rows[t] += 1
            self.sq[t] += (pi - yi) ** 2
            self.ll[t] -= math.log(q if yi else 1.0 - q)
            self.sy[t] += yi
            self.sp[t] += pi
            b = min(int(pi * ECE_BINS), ECE_BINS - 1)
            self.b_cnt[b][t] += 1
            self.b_sp[b][t] += pi
            self.b_sy[b][t] += yi
            for k, tau in enumerate(self.taus):
                if abs(pi - tau) <= _WINDOW:
                    self.w_cnt[k][t] += 1
                    self.w_sp[k][t] += pi
                    self.w_sy[k][t] += yi

    def stats(self, counts: Sequence[int]) -> dict[str, float]:
        n = dot(counts, self.rows)
        out = {
            "brier": dot(counts, self.sq) / n,
            "log_loss": dot(counts, self.ll) / n,
            "mean_predicted": dot(counts, self.sp) / n,
            "observed": dot(counts, self.sy) / n,
        }
        total = 0.0
        for b in range(ECE_BINS):
            nb = dot(counts, self.b_cnt[b])
            if nb:
                total += (
                    nb / n * abs(dot(counts, self.b_sy[b]) / nb - dot(counts, self.b_sp[b]) / nb)
                )
        out["ece"] = total
        for k, tau in enumerate(self.taus):
            nw = dot(counts, self.w_cnt[k])
            out[f"gap@{tau:g}"] = (
                dot(counts, self.w_sy[k]) / nw - dot(counts, self.w_sp[k]) / nw if nw else math.nan
            )
            out[f"n@{tau:g}"] = nw
        return out


def _interval(samples: Sequence[Mapping[str, float]], key: str, level: float) -> list[float] | None:
    iv = percentile_interval([s[key] for s in samples], level)
    return [iv.lo, iv.hi] if iv.estimable else None


def _variant(p: Sequence[float], y: Sequence[int]) -> dict[str, Any]:
    return {
        "n": len(y),
        "mean_predicted": sum(p) / len(p),
        "observed": sum(y) / len(y),
        "brier": brier(p, y),
        "log_loss": log_loss(p, y),
        "ece": ece(p, y),
        "auc": auc(p, y),
        "accuracy_at_0.5": sum((pi >= 0.5) == bool(yi) for pi, yi in zip(p, y, strict=True))
        / len(y),
        "reliability": reliability(p, y),
    }


def _group_report(
    tasks: Sequence[str],
    raw: Sequence[float],
    cal: Sequence[float],
    y: Sequence[int],
    taus: Sequence[float],
    replicates: int,
    seed: int,
    level: float,
) -> dict[str, Any]:
    acc_raw = _Accumulator(tasks, raw, y, taus)
    acc_cal = _Accumulator(tasks, cal, y, taus)
    n = acc_raw.n_tasks

    def stat(counts: Sequence[int]) -> dict[str, float]:
        r, c = acc_raw.stats(counts), acc_cal.stats(counts)
        out = {f"raw.{k}": v for k, v in r.items()}
        out.update({f"cal.{k}": v for k, v in c.items()})
        for m in ("brier", "log_loss", "ece"):
            out[f"delta.{m}"] = c[m] - r[m]
        return out

    samples = cluster_bootstrap(n, replicates, seed, stat) if n > 1 and replicates > 0 else []
    point = stat([1] * n)
    base = sum(y) / len(y)
    brier_base = base * (1.0 - base)
    report: dict[str, Any] = {
        "n_rows": len(y),
        "n_tasks": n,
        "observed_success_rate": base,
        "brier_of_constant_base_rate": brier_base,
    }
    for name, probs in (("uncalibrated", raw), ("calibrated", cal)):
        v = _variant(probs, y)
        tag = "raw" if name == "uncalibrated" else "cal"
        for m in ("brier", "log_loss", "ece"):
            v[f"{m}_ci"] = _interval(samples, f"{tag}.{m}", level) if samples else None
        v["brier_skill_vs_base_rate"] = 1.0 - v["brier"] / brier_base if brier_base > 0 else None
        v["thresholds"] = [
            {
                "tau": tau,
                "window": [tau - _WINDOW, tau + _WINDOW],
                "n_rows": int(point[f"{tag}.n@{tau:g}"]),
                "observed_minus_predicted": (
                    None if math.isnan(point[f"{tag}.gap@{tau:g}"]) else point[f"{tag}.gap@{tau:g}"]
                ),
                "ci": _interval(samples, f"{tag}.gap@{tau:g}", level) if samples else None,
            }
            for tau in taus
        ]
        report[name] = v
    report["calibration_effect"] = {
        m: {
            "calibrated_minus_uncalibrated": point[f"delta.{m}"],
            "ci": _interval(samples, f"delta.{m}", level) if samples else None,
        }
        for m in ("brier", "log_loss", "ece")
    }
    return report


def _test_dataset(
    model: TabularModel, grid: GridData, catalog: Catalog, n_probes: int | None, seed: int
) -> Dataset:
    builder = FeatureBuilder(model.spec, catalog, model.probes, probe_k=n_probes)
    cfg = model.meta["train_config"]
    return build_dataset(
        grid,
        model.roles.tasks("test"),
        catalog,
        builder,
        seed=seed,
        max_history=int(cfg["max_history"]),
        history_cap=None,
        exclude_models=(),
    )


def calibration_report(
    model: TabularModel,
    grid: GridData,
    catalog: Catalog,
    *,
    n_probes: int | None = None,
    taus: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    replicates: int = 300,
    seed: int = 0,
    confidence: float = 0.95,
    holdout_models: Sequence[str] = (),
) -> dict[str, Any]:
    """Accuracy and calibration on the model's test tasks, before and after calibration.

    ``n_probes`` is how many probes describe each configuration (None: all, 0: none).
    ``holdout_models`` adds, for first-step rows of those models, the same metrics on their
    rows alone (the cold-start check).
    """
    ds = _test_dataset(model, grid, catalog, n_probes, seed)
    synthetic = model.synthetic or is_synthetic(grid.header)
    out: dict[str, Any] = {
        "stamp": STAMP if synthetic else None,
        "synthetic": synthetic,
        "data_fingerprint": model.meta["data_fingerprint"],
        "n_probes_used": n_probes if n_probes is not None else len(model.probes),
        "test_tasks": len(model.roles.tasks("test")),
        "calibrator": model.calibrator.to_dict(),
        "binning": f"{ECE_BINS} equal-width bins of the predicted probability",
        "interval": (
            f"{confidence:.0%} percentile interval, {replicates} task-cluster bootstrap replicates"
        ),
        "threshold_window": f"rows with |predicted - tau| <= {_WINDOW}",
        "groups": {},
    }
    if len(ds) == 0:
        out["error"] = "no test rows"
        return out
    margins = model.margins(ds.X)
    raw = model.raw_probabilities(margins)
    cal = model.calibrated(margins)
    y = [int(v) for v in ds.y.tolist()]
    for key, label, pick in _GROUPS:
        rows = [i for i, h in enumerate(ds.history) if pick(h)]
        if not rows:
            continue
        g = _group_report(
            [ds.task_ids[i] for i in rows],
            [raw[i] for i in rows],
            [cal[i] for i in rows],
            [y[i] for i in rows],
            taus,
            replicates,
            seed,
            confidence,
        )
        g["description"] = label
        out["groups"][key] = g
    per_config: list[dict[str, Any]] = []
    for cfg in sorted({c for c, h in zip(ds.config, ds.history, strict=True) if h == 0}):
        rows = [i for i, c in enumerate(ds.config) if c == cfg and ds.history[i] == 0]
        yy = [y[i] for i in rows]
        per_config.append(
            {
                "model": cfg[0],
                "effort": cfg[1],
                "held_out": cfg[0] in holdout_models,
                "n": len(rows),
                "observed": sum(yy) / len(yy),
                "mean_calibrated": sum(cal[i] for i in rows) / len(rows),
                "brier": brier([cal[i] for i in rows], yy),
                "auc_within_config": auc([cal[i] for i in rows], yy),
            }
        )
    out["per_configuration_first_step"] = per_config
    return out


# --- the evaluator's report -----------------------------------------------------------------


def stamp_report(report: dict[str, Any]) -> None:
    """Mark an evaluator report as synthetic, in place: a top-level notice, and the stamp at
    the start of every verdict sentence and summary line."""
    report["stamp"] = STAMP
    report["synthetic_data"] = True
    report["notice"] = (
        f"{STAMP} This report was made from the evaluator's synthetic generator. It says "
        "nothing about real models."
    )
    for v in report.get("verdicts", []):
        v["sentence"] = f"{STAMP} {v['sentence']}"
    report["summary"] = [f"{STAMP} {s}" for s in report.get("summary", [])]
    for ev in report.get("evaluations", []):
        ev["stamp"] = STAMP
        for pol in ev.get("policies", []):
            verdict = pol.get("verdict")
            if isinstance(verdict, dict) and "sentence" in verdict:
                verdict["sentence"] = f"{STAMP} {verdict['sentence']}"


def stamp_svg(svg: str) -> str:
    """Add a diagonal watermark and a bottom line carrying the stamp to an SVG."""
    marker = 'viewBox="0 0 '
    i = svg.index(marker) + len(marker)
    w_s, h_s = svg[i : svg.index('"', i)].split()
    w, h = float(w_s), float(h_s)
    mark = (
        f'<g class="stamp" aria-label="{STAMP}">'
        f'<text transform="translate({w / 2:.0f},{h / 2:.0f}) rotate(-24)" text-anchor="middle" '
        'font-family="system-ui,sans-serif" font-size="44" font-weight="800" '
        f'style="fill:#c62828;fill-opacity:0.22">{STAMP}</text>'
        f'<text x="{w / 2:.0f}" y="{h - 8:.0f}" text-anchor="middle" '
        'font-family="system-ui,sans-serif" font-size="13" font-weight="700" '
        f'style="fill:#c62828">{STAMP}</text></g>'
    )
    return svg[: svg.rindex("</svg>")] + mark + "</svg>"


def evaluator_report(
    model: TabularModel,
    grid: GridData,
    catalog: Catalog,
    *,
    taus: Sequence[float] = (0.7, 0.8, 0.9),
    epsilon: float = 0.0,
    n_probes: int | None = None,
    rules: str | None = None,
    settings: EvalSettings | None = None,
    command: Sequence[str] | None = None,
) -> tuple[dict[str, Any], str, Evaluation]:
    """The predictor inside ``ScoredPolicy`` (calibrated and not, at each ``tau``) and the rule
    baseline, scored on the model's test tasks against every single configuration.

    Returns ``(report dict, SVG text, the Evaluation)``. Probe tasks and tasks of every other
    role are not scored: only ``test`` is.
    """
    st = settings or EvalSettings()
    tasks = {
        t: dataclasses.replace(task, split=model.roles.role.get(t, "excluded"))
        for t, task in grid.tasks.items()
    }
    # Only the file's name goes into the report, never the directory it was read from.
    data = GridData(
        os.path.basename(grid.source),
        grid.runs,
        tasks,
        grid.header,
        grid.skipped_lines,
        grid.sha256,
    )
    policies: list[TargetPolicy] = []
    for tau in taus:
        for calibrated in (True, False):
            pred = TabularPredictor(model, catalog, n_probes=n_probes, calibrated=calibrated)
            label = f"{pred.name} tau={tau:g}"
            policies.append(
                as_target(ScoredPolicy(pred, catalog, tau, epsilon, seed=0, name=label), label)
            )
    skipped: str | None = None
    try:
        rule_target = parse_policy_spec(f"rules:{rules}" if rules else "rules", lambda: catalog)
        if isinstance(rule_target, DistributionTarget):
            rule_target.name = "rules"  # not the file path: reports must not carry local paths
        policies.append(rule_target)
    except RuleConfigError as exc:
        if rules:
            raise
        skipped = f"The rule baseline was not run: {exc}. Pass a rules file for this catalog."
    policies.append(parse_policy_spec("oracle", lambda: catalog))
    ev = evaluate_grid(data, policies, st, split="test")
    if skipped:
        ev.notes.append(skipped)
    ev.provenance["model"] = {
        "data_fingerprint": model.meta["data_fingerprint"],
        "seed": model.meta["seed"],
        "n_probes_used": n_probes if n_probes is not None else len(model.probes),
        "calibration": model.calibrator.kind,
        "roles": model.roles.counts(),
        "synthetic": model.synthetic,
    }
    report = build_report([ev], command=command)
    synthetic = model.synthetic or is_synthetic(grid.header)
    svg = render_svg(ev, title=STAMP if synthetic else None)
    if synthetic:
        stamp_report(report)
        svg = stamp_svg(svg)
    return report, svg, ev


# --- markdown -------------------------------------------------------------------------------


def _fmt(v: float | None, digits: int = 4) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(ci: Sequence[float] | None, digits: int = 4) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def calibration_markdown(rep: dict[str, Any], title: str = "Calibration report") -> str:
    """The calibration report as Markdown, stamped when the data is synthetic."""
    stamp = rep.get("stamp")
    lines: list[str] = []
    if stamp:
        lines += [
            f"# {stamp}",
            "",
            f"> **{stamp}** Made from the evaluator's synthetic generator.",
            "",
        ]
    lines += [f"## {title}", ""]
    lines += [
        f"- Test tasks: {rep['test_tasks']}; probes describing each configuration: "
        f"{rep['n_probes_used']}",
        f"- Binning: {rep['binning']}",
        f"- Intervals: {rep['interval']}",
        f"- Threshold window: {rep['threshold_window']}",
        f"- Calibrator: {rep['calibrator']['kind']}"
        + (
            f" (a = {rep['calibrator']['a']:.3f}, b = {rep['calibrator']['b']:.3f})"
            if rep["calibrator"]["kind"] == "platt"
            else ""
        ),
        f"- Data fingerprint: `{rep['data_fingerprint']}`",
        "",
    ]
    for g in rep["groups"].values():
        if stamp:
            lines += [f"> {stamp}", ""]
        lines += [
            f"### {g['description']}",
            "",
            f"{g['n_rows']} rows from {g['n_tasks']} tasks; observed success rate "
            f"{g['observed_success_rate']:.3f}; Brier score of the constant base rate "
            f"{g['brier_of_constant_base_rate']:.4f}.",
            "",
            "| metric | uncalibrated | 95% interval | calibrated | 95% interval |",
            "|---|---|---|---|---|",
        ]
        u, c = g["uncalibrated"], g["calibrated"]
        for m, name in (("ece", "ECE"), ("brier", "Brier"), ("log_loss", "log loss")):
            lines.append(
                f"| {name} | {_fmt(u[m])} | {_ci(u[m + '_ci'])} | "
                f"{_fmt(c[m])} | {_ci(c[m + '_ci'])} |"
            )
        lines.append(f"| AUC | {_fmt(u['auc'])} | | {_fmt(c['auc'])} | |")
        lines.append(
            f"| accuracy at 0.5 | {_fmt(u['accuracy_at_0.5'])} | | {_fmt(c['accuracy_at_0.5'])} | |"
        )
        lines += ["", "Calibrated minus uncalibrated (negative is better):", ""]
        lines += ["| metric | difference | 95% interval |", "|---|---|---|"]
        for m, e in g["calibration_effect"].items():
            lines.append(f"| {m} | {e['calibrated_minus_uncalibrated']:+.4f} | {_ci(e['ci'])} |")
        lines += ["", "Near the policy's thresholds (observed minus predicted success):", ""]
        lines += [
            "| tau | rows in window | uncalibrated | 95% interval | calibrated | 95% interval |",
            "|---|---|---|---|---|---|",
        ]
        for tu, tc in zip(u["thresholds"], c["thresholds"], strict=True):
            lines.append(
                f"| {tu['tau']:g} | {tc['n_rows']} (calibrated), {tu['n_rows']} (raw) | "
                f"{_fmt(tu['observed_minus_predicted'], 3)} | {_ci(tu['ci'], 3)} | "
                f"{_fmt(tc['observed_minus_predicted'], 3)} | {_ci(tc['ci'], 3)} |"
            )
        lines += ["", "Reliability table (calibrated):", ""]
        lines += ["| predicted bin | rows | mean predicted | observed |", "|---|---|---|---|"]
        for r in c["reliability"]:
            if r["count"]:
                lines.append(
                    f"| [{r['bin_low']:.1f}, {r['bin_high']:.1f}) | {r['count']} | "
                    f"{r['mean_predicted']:.3f} | {r['observed']:.3f} |"
                )
        lines.append("")
    if rep.get("per_configuration_first_step"):
        if stamp:
            lines += [f"> {stamp}", ""]
        lines += [
            "### Per configuration, first step",
            "",
            "| configuration | held out | rows | observed | mean predicted | Brier "
            "| AUC within configuration |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in rep["per_configuration_first_step"]:
            lines.append(
                f"| {r['model']} ({r['effort']}) | {'yes' if r['held_out'] else 'no'} | {r['n']} | "
                f"{r['observed']:.3f} | {r['mean_calibrated']:.3f} | {r['brier']:.4f} | "
                f"{_fmt(r['auc_within_config'], 3)} |"
            )
        lines.append("")
    if stamp:
        lines += [f"**{stamp}**", ""]
    return "\n".join(lines)
