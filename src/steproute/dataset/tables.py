"""The two Parquet tables of a release and their column definitions.

``pyarrow`` is imported inside the functions, so importing this module (and the rest of the
package) needs no optional extra. The column list is plain data and doubles as the
documentation written to ``schema/tables.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "RUN_COLUMNS",
    "STEP_COLUMNS",
    "MissingParquetSupport",
    "read_rows",
    "table_docs",
    "write_rows",
]


class MissingParquetSupport(ImportError):
    """The ``dataset`` extra (pyarrow) is not installed."""


#: (name, type, meaning). Types: str, int, float, bool, list[str].
STEP_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("decision_id", "str", "keyed hash of the decision id"),
    ("run_id", "str", "the run this step belongs to (join key to the runs table)"),
    ("session_id", "str", "keyed hash of the session id"),
    ("task_id", "str", "task id of the task set (a public benchmark id, kept in clear)"),
    ("stratum", "str", "difficulty stratum of the task in the task set (may be null)"),
    ("task_split", "str", "the split the task belongs to: train, validation or test"),
    ("split", "str", "the split of this file; holdout_model for the held-out model"),
    ("ts", "float", "decision time, UTC seconds, rounded down to the bucket (default 1 hour)"),
    ("step_idx", "int", "index of the step in the run, from 0"),
    ("kind", "str", "step kind from the collector"),
    ("harness", "str", "harness that produced the step"),
    ("source", "str", "always replay"),
    ("prompt_tokens", "int", "prompt size of the step"),
    ("cached_prefix_tokens", "int", "tokens of the prompt that were a cached prefix"),
    ("sensitive", "bool", "the step was flagged sensitive"),
    ("tools", "list[str]", "tool names offered at the step"),
    ("model", "str", "model of the logged action"),
    ("effort", "str", "effort of the logged action"),
    ("policy", "str", "logging policy"),
    ("propensity", "float", "probability of the logged action under the logging policy"),
    ("candidates_json", "str", "JSON list of the candidate actions"),
    ("features_json", "str", "JSON of the scalar features that survived scrubbing"),
    ("has_outcome", "bool", "an outcome record is joined to this decision"),
    ("success", "bool", "task-level result copied onto the step; null if unlabelled"),
    ("final_model", "str", "model that served the step"),
    ("escalated", "bool", "the step escalated to a fallback"),
    ("input_tokens", "int", "uncached input tokens"),
    ("output_tokens", "int", "output tokens"),
    ("cache_read_tokens", "int", "input tokens read from cache"),
    ("cache_write_tokens", "int", "input tokens written to cache"),
    ("cost_usd", "float", "realised cost of the step in USD, as billed"),
    ("latency_ms", "float", "latency of the step in milliseconds"),
    ("signals_json", "str", "JSON of the boolean and numeric outcome signals"),
)

RUN_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("run_id", "str", "unique run id"),
    ("session_id", "str", "keyed hash of the session id (same as in the steps table)"),
    ("task_id", "str", "task id of the task set"),
    ("stratum", "str", "difficulty stratum of the task (may be null)"),
    ("task_split", "str", "the split the task belongs to: train, validation or test"),
    ("split", "str", "split of the run; holdout_model for every run of the held-out model"),
    ("model", "str", "model, fixed for the whole run"),
    ("effort", "str", "effort, fixed for the whole run"),
    ("repeat", "int", "repeat number of the cell, from 0"),
    ("success", "bool", "task check result; null means unlabelled (not a failure)"),
    ("status", "str", "run status from the replay runner (passed, failed, ...)"),
    ("steps", "int", "number of steps of the run"),
    ("attempts", "int", "attempts made for the cell"),
    ("cost_usd", "float", "billed cost of the run in USD"),
    ("list_cost_usd", "float", "the same usage at list (peak) price; compare models on this"),
    ("latency_ms", "float", "wall-clock time of the run in milliseconds"),
    ("ts", "float", "end of the run, UTC seconds, rounded down to the bucket"),
)


def table_docs() -> dict[str, list[dict[str, str]]]:
    def cols(spec: Sequence[tuple[str, str, str]]) -> list[dict[str, str]]:
        return [{"name": n, "type": t, "description": d} for n, t, d in spec]

    return {"steps": cols(STEP_COLUMNS), "runs": cols(RUN_COLUMNS)}


def _pa() -> Any:
    try:
        import pyarrow
    except ImportError:
        raise MissingParquetSupport(
            "reading and writing a release needs the dataset extra: "
            "pip install 'steproute[dataset]'"
        ) from None
    return pyarrow


def _pa_type(pa: Any, name: str) -> Any:
    return {
        "str": pa.string(),
        "int": pa.int64(),
        "float": pa.float64(),
        "bool": pa.bool_(),
        "list[str]": pa.list_(pa.string()),
    }[name]


def _schema(pa: Any, spec: Sequence[tuple[str, str, str]]) -> Any:
    return pa.schema([pa.field(n, _pa_type(pa, t)) for n, t, _ in spec])


def check_available() -> None:
    _pa()


def write_rows(path: Path, kind: str, rows: Sequence[dict[str, Any]]) -> None:
    """Write ``rows`` as one Parquet file; ``kind`` is ``steps`` or ``runs``."""
    pa = _pa()
    import pyarrow.parquet as pq

    spec = STEP_COLUMNS if kind == "steps" else RUN_COLUMNS
    schema = _schema(pa, spec)
    names = [n for n, _, _ in spec]
    table = pa.Table.from_pylist([{n: r[n] for n in names} for r in rows], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")


def read_rows(path: Path, kind: str) -> list[dict[str, Any]]:
    """Read a table back as a list of row dicts; the file must have exactly the expected
    columns and types."""
    pa = _pa()
    import pyarrow.parquet as pq

    spec = STEP_COLUMNS if kind == "steps" else RUN_COLUMNS
    table = pq.read_table(path)
    if table.schema != _schema(pa, spec):
        raise ValueError(f"{path}: columns differ from the {kind} table of this release format")
    return list(table.to_pylist())
