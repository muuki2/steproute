"""Check a trace store written by a scripted dsh session (scripts/dsh_session.sh, or the
in-process session scripts/dsh_inprocess_session.sh).

    uv run python scripts/check_dsh_traces.py STORE [--expect N] [--mode shadow|route]

Prints one row per decision with its joined outcome and exits 0 only when at least N
decisions (default 3) have a joined outcome, every outcome of a step that ran has non-zero
tokens, and, with ``--mode route``, every applied decision's model is the model dsh ran
(``final_model``, from the session log's ``assistant/message`` source). With
``--allow-unjoined`` (a session whose service was killed) a decision without an outcome is
listed but is not a failure: its outcome was due after the service died.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from steproute.core import Decision, Outcome
from steproute.core.store import TraceStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--expect", type=int, default=3)
    ap.add_argument("--mode", choices=["shadow", "route"])
    ap.add_argument("--allow-unjoined", action="store_true")
    args = ap.parse_args(argv)
    decisions: dict[str, Decision] = {}
    outcomes: dict[str, Outcome] = {}
    for r in TraceStore(args.store).records():
        if isinstance(r, Decision):
            decisions[r.decision_id] = r
        elif isinstance(r, Outcome):
            outcomes[r.decision_id] = r  # the last outcome per decision counts
    joined = 0
    problems: list[str] = []
    for did, d in decisions.items():
        o = outcomes.get(did)
        rec = f"{d.action.model} ({d.action.effort})"
        st = d.state
        head = (
            f"{did[:8]}  {st.kind:<13} step_idx={st.step_idx} prompt={st.prompt_tokens} "
            f"cached={st.cached_prefix_tokens} owner={st.current_model}/{st.current_effort}  "
            f"recommended {rec}"
        )
        if o is None:
            print(f"{head}  NO OUTCOME")
            if not args.allow_unjoined:
                problems.append(f"{did[:8]} has no outcome")
            continue
        joined += 1
        s = o.signals
        print(
            f"{head}  applied={s.get('applied')} ({s.get('apply_reason')})  "
            f"selected {s.get('selected_route')}/{s.get('selected_effort')}  "
            f"ran {s.get('executed_route')}/{s.get('executed_effort')} -> {o.final_model}  "
            f"in={o.input_tokens} out={o.output_tokens} cache_read={o.cache_read_tokens}  "
            f"tools={s.get('tool_call_count')} decide_ms={s.get('decide_ms')} "
            f"wait_ms={s.get('plugin_wait_ms')}  "
            f"end={s.get('end_reason')}  cost={o.cost_usd}  success={o.success}"
        )
        ran = s.get("step_ran", True)
        if ran and (
            o.input_tokens + o.cache_read_tokens + o.cache_write_tokens == 0 or o.output_tokens == 0
        ):
            problems.append(f"{did[:8]} ran but has zero tokens")
        if args.mode and s.get("routing_mode") != args.mode:
            problems.append(f"{did[:8]} was logged in mode {s.get('routing_mode')}")
        if args.mode == "route" and s.get("applied") and o.final_model != d.action.model:
            problems.append(f"{did[:8]} applied {d.action.model} but {o.final_model} ran")
        if ran and s.get("selection_ran") is False:
            problems.append(f"{did[:8]}: the step's fixed selection is not what ran")
    print(f"decisions {len(decisions)}, joined {joined}")
    for p in problems:
        print(f"problem: {p}")
    return 0 if joined >= args.expect and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
