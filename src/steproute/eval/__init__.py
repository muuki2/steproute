"""Offline evaluation. Standard library only; see ``docs/EVALUATION.md``.

Two evaluators, one per kind of data:

- ``evaluate_grid``: task-level, full information, on the replay grid (``load_grid``).
- ``evaluate_logged``: step-level, off-policy, on logged decisions (``load_logged``).

Both compare every policy with every single ``(model, effort)`` configuration and return an
``Evaluation``; ``build_report`` turns evaluations into the JSON report and ``write_svg``
draws the frontier plot.
"""

from steproute.eval.data import (
    GRID_FORMAT,
    GRID_VERSION,
    DataFormatError,
    GridData,
    GridRun,
    GridTask,
    LoggedData,
    LoggedRow,
    detect_kind,
    load_grid,
    load_logged,
    logged_from_pairs,
    task_state,
    write_grid,
)
from steproute.eval.frontier import Point, envelope, margin, pareto
from steproute.eval.grid import evaluate_grid
from steproute.eval.ope import CellMeanModel, FittableRewardModel, RewardModel, evaluate_logged
from steproute.eval.plot import render_svg, write_svg
from steproute.eval.report import DEFINITIONS, METRICS_VERSION, build_report, write_report
from steproute.eval.results import (
    BEATS,
    DOES_NOT_BEAT,
    NOT_SHOWN,
    Estimate,
    EvalSettings,
    Evaluation,
    PolicyResult,
    Verdict,
)
from steproute.eval.targets import (
    DistributionTarget,
    FixedTarget,
    OracleTarget,
    TargetPolicy,
    as_target,
    parse_action,
    parse_policy_spec,
)

__all__ = [
    "BEATS",
    "DEFINITIONS",
    "DOES_NOT_BEAT",
    "GRID_FORMAT",
    "GRID_VERSION",
    "METRICS_VERSION",
    "NOT_SHOWN",
    "CellMeanModel",
    "DataFormatError",
    "DistributionTarget",
    "Estimate",
    "EvalSettings",
    "Evaluation",
    "FittableRewardModel",
    "FixedTarget",
    "GridData",
    "GridRun",
    "GridTask",
    "LoggedData",
    "LoggedRow",
    "OracleTarget",
    "Point",
    "PolicyResult",
    "RewardModel",
    "TargetPolicy",
    "Verdict",
    "as_target",
    "build_report",
    "detect_kind",
    "envelope",
    "evaluate_grid",
    "evaluate_logged",
    "load_grid",
    "load_logged",
    "logged_from_pairs",
    "margin",
    "pareto",
    "parse_action",
    "parse_policy_spec",
    "render_svg",
    "task_state",
    "write_grid",
    "write_report",
    "write_svg",
]
