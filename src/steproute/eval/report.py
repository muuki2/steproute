"""The self-describing JSON report. Standard library only."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from typing import Any

from steproute import __version__
from steproute.eval.results import Evaluation

__all__ = ["DEFINITIONS", "METRICS_VERSION", "REPORT_FORMAT", "build_report", "write_report"]

REPORT_FORMAT = "steproute-eval-report"
#: Bump when any definition below changes meaning, so old and new reports are not compared.
METRICS_VERSION = "1"

#: One line per metric; the full text is docs/EVALUATION.md, "Metric definitions".
DEFINITIONS: dict[str, str] = {
    "unit": "A task on the grid; a logged step for off-policy estimates.",
    "success_rate": (
        "Mean success over units. Unlabelled outcomes are excluded from the headline and "
        "bounded in the 'unlabelled_sensitivity' diagnostic; they are never failures."
    ),
    "cost_per_task": "Mean realized cost in USD per task, failures included.",
    "cost_per_step": "Mean realized cost in USD per logged step, failures included.",
    "cost_per_solved": (
        "Total spend divided by units solved (mean cost / success rate), so failed attempts "
        "are paid for. Null when nothing is solved; an unbounded interval is flagged."
    ),
    "quality_retention": (
        "Success rate divided by that of the strongest single configuration (highest "
        "success, then cheapest) on the same units. Not capped: above 1 means the policy "
        "solves more than the strongest configuration."
    ),
    "strongest_solved_retained": (
        "Grid only: of the tasks the strongest configuration solves, the share the policy "
        "also solves (at most 1)."
    ),
    "switches_per_task": (
        "Grid: expected model changes between cascade attempts within a task (0 for a single "
        "configuration)."
    ),
    "config_changes_per_task": "Grid: expected (model, effort) changes between attempts.",
    "switch_rate_per_step": (
        "Logged: probability that the policy's first step uses a model other than the "
        "state's cache owner, averaged over logged states."
    ),
    "added_latency_p50_ms": (
        "Grid: median over tasks of the latency of cascade attempts before the one that "
        "produced the result (routing-decision time is not in the data). Null on logged data."
    ),
    "added_latency_p95_ms": "As added_latency_p50_ms, the 95th percentile (lower quantile).",
    "total_latency_p50_ms": "Grid: median end-to-end latency of the policy's runs per task.",
    "total_latency_p95_ms": "Grid: 95th percentile end-to-end latency per task.",
    "frontier_margin": (
        "Success rate minus the frontier's success rate at the policy's spend, in "
        "success-rate units. The frontier is the upper concave envelope of every single "
        "configuration's (spend, success) point and the origin."
    ),
    "cost_saving_at_equal_success": (
        "Grid: the frontier's least spend reaching the policy's success rate, minus the "
        "policy's spend (USD per task). Null and flagged when no configuration reaches it."
    ),
    "utility": "success_rate - cost_weight * cost, only when a cost weight is given.",
    "verdict": (
        "beats_frontier when the margin's interval lies above 0; does_not_beat when it lies "
        "at or below 0 (a tie does not beat); not_shown otherwise, or when the data cannot "
        "support a verdict (reasons listed)."
    ),
    "interval": (
        "Percentile cluster bootstrap at the stated confidence: whole tasks (or sessions "
        "without a task id) are resampled, paired across policies and configurations."
    ),
    "ess": "Kish effective sample size of the importance weights, (sum w)^2 / sum w^2.",
}


def build_report(
    evaluations: Sequence[Evaluation], *, command: Sequence[str] | None = None
) -> dict[str, Any]:
    """The report as a JSON-ready dict."""
    verdicts = []
    for ev in evaluations:
        for p in ev.policies:
            verdicts.append(
                {
                    "data": ev.provenance.get("source"),
                    "kind": ev.kind,
                    "level": ev.level,
                    "policy": p.name,
                    "hindsight": p.hindsight,
                    "status": p.verdict.status,
                    "beats_every_single_configuration": p.verdict.status == "beats_frontier",
                    "sentence": p.verdict.sentence,
                }
            )
    settings = evaluations[0].settings.to_json() if evaluations else {}
    return {
        "format": REPORT_FORMAT,
        "metrics_version": METRICS_VERSION,
        "steproute_version": __version__,
        "generated_at": time.time(),
        "command": None if command is None else list(command),
        "settings": settings,
        "definitions": DEFINITIONS,
        "verdicts": verdicts,
        "summary": [v["sentence"] for v in verdicts],
        "evaluations": [ev.to_json() for ev in evaluations],
    }


def write_report(path: str | os.PathLike[str], report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
