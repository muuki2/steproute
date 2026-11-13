"""Trace store tests.

Refresh the stats snapshot with:
    STEPROUTE_UPDATE_SNAPSHOTS=1 pytest tests/test_store.py
"""

import multiprocessing
import os
import warnings
from pathlib import Path

import pytest

from steproute.cli import main
from steproute.core import Action, Decision, Outcome, StepState, TraceStore
from steproute.core.store import CorruptLineWarning

SNAPSHOT = Path(__file__).parent / "snapshots" / "traces_stats.txt"


def _dec(i: str, model: str = "m1", effort: str = "medium", n: int = 1) -> Decision:
    a = Action(model=model, effort=effort)
    st = StepState(harness="h", session_id="s", step_idx=0, kind="edit", prompt_tokens=10)
    return Decision(
        decision_id=i, ts=1.0, state=st, candidates=[a] * n, action=a, policy="p", propensity=1.0
    )


def _out(i: str, success: bool | None, cost: float | None = None) -> Outcome:
    return Outcome(decision_id=i, ts=2.0, success=success, cost_usd=cost)


def _worker(path: str, w: int, n: int) -> None:
    store = TraceStore(path)
    for k in range(n):
        store.log_decision(_dec(f"{w}-{k}", n=50))


def test_concurrent_appends(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(str(tmp_path), w, 1000)) for w in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0
    store = TraceStore(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recs = list(store.records())
    assert len(recs) == 8000
    assert sum(len(f.read_text().splitlines()) for f in store.files()) == 8000
    assert store.skipped_lines == 0


def test_truncated_tail_skipped_and_repaired(tmp_path: Path) -> None:
    store = TraceStore(tmp_path, clock=lambda: 0.0)
    store.log_decision(_dec("a"))
    f = store.files()[0]
    f.write_bytes(f.read_bytes() + b'{"type":"decision","v":"0.1","da')
    with pytest.warns(CorruptLineWarning, match=r"traces-1970-01-01\.jsonl:2"):
        assert len(list(store.records())) == 1
    assert store.skipped_lines == 1
    store.log_decision(_dec("b"))  # must not be glued onto the partial line
    with pytest.warns(CorruptLineWarning):
        ids = [r.decision_id for r in store.records() if isinstance(r, Decision)]
    assert ids == ["a", "b"]
    assert store.skipped_lines == 1


def test_corrupt_middle_and_unknown_type(tmp_path: Path) -> None:
    store = TraceStore(tmp_path, clock=lambda: 0.0)
    store.log_decision(_dec("a"))
    f = store.files()[0]
    with open(f, "ab") as fh:
        fh.write(b"garbage\n" + b'{"type":"weird","v":"9","data":{}}\n')
    store.log_decision(_dec("b"))
    with pytest.warns(CorruptLineWarning) as w:
        assert len(list(store.records())) == 2
    assert len(w) == 2
    assert store.skipped_lines == 2


def test_daily_rotation_and_joined(tmp_path: Path) -> None:
    now = [0.0]
    store = TraceStore(tmp_path, clock=lambda: now[0])
    store.log_decision(_dec("a"))
    store.log_decision(_dec("b"))
    now[0] = 86400.0 * 2
    store.log_outcome(_out("a", False))
    store.log_outcome(_out("a", True))
    store.log_outcome(_out("zzz", True))
    assert [f.name for f in store.files()] == [
        "traces-1970-01-01.jsonl",
        "traces-1970-01-03.jsonl",
    ]
    pairs = list(store.joined())
    assert [(d.decision_id, o.success if o else None) for d, o in pairs] == [
        ("a", True),
        ("b", None),
    ]


def test_stats_snapshot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = TraceStore(tmp_path, clock=lambda: 0.0)
    combos = [("haiku", "low"), ("haiku", "low"), ("opus", "high"), ("opus", "high")]
    for i, (m, e) in enumerate(combos):
        store.log_decision(_dec(f"d{i}", m, e))
    store.log_outcome(_out("d0", False, 0.01))
    store.log_outcome(_out("d0", True, 0.01))
    store.log_outcome(_out("d1", False, 0.0125))
    store.log_outcome(_out("d2", True))
    store.log_outcome(_out("orphan", True, 1.0))
    assert main(["traces", "stats", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    if os.environ.get("STEPROUTE_UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(out)
    assert out == SNAPSHOT.read_text()


def test_stats_empty_missing_and_bad_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["traces", "stats", str(tmp_path / "nope")]) == 0
    assert "decisions: 0" in capsys.readouterr().out
    f = tmp_path / "file"
    f.write_text("x")
    assert main(["traces", "stats", str(f)]) != 0
    assert capsys.readouterr().err.count("\n") == 1


def test_cost_counts_last_outcome_only(tmp_path: Path) -> None:
    from steproute.core.store import compute_stats

    store = TraceStore(tmp_path, clock=lambda: 0.0)
    store.log_decision(_dec("a"))
    store.log_outcome(_out("a", None, 0.5))
    store.log_outcome(_out("a", True, 0.5))
    st = compute_stats(store)
    assert st.total_cost_usd == 0.5
    assert (st.outcomes, st.superseded_outcomes, st.outcomes_without_cost) == (2, 1, 0)
