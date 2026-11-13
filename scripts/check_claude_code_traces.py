"""Check a trace store written by a scripted Claude Code session (scripts/claude_code_session.sh).

    uv run python scripts/check_claude_code_traces.py STORE [--expect N] [--mode shadow|route]

Prints one row per decision with its joined outcome and exits 0 only when at least N
decisions (default 3) have a joined outcome with non-zero input and output tokens, and, with
``--mode route``, every applied decision's model is the model the API reported.
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
    args = ap.parse_args(argv)
    decisions: dict[str, Decision] = {}
    outcomes: dict[str, Outcome] = {}
    for r in TraceStore(args.store).records():
        if isinstance(r, Decision):
            decisions[r.decision_id] = r
        elif isinstance(r, Outcome):
            outcomes[r.decision_id] = r  # the last outcome per decision counts
    good = 0
    problems: list[str] = []
    for did, d in decisions.items():
        o = outcomes.get(did)
        rec = f"{d.action.model} ({d.action.effort})"
        if o is None:
            print(f"{did[:8]}  recommended {rec}  NO OUTCOME")
            problems.append(f"{did[:8]} has no outcome")
            continue
        s = o.signals
        print(
            f"{did[:8]}  {d.state.features.get('subagent_type')}  recommended {rec}  "
            f"applied={s.get('applied')} ({s.get('apply_reason')})  ran {o.final_model}  "
            f"in={o.input_tokens} out={o.output_tokens} cache_read={o.cache_read_tokens} "
            f"cache_write={o.cache_write_tokens}  requests={s.get('requests')} "
            f"via {s.get('usage_source')}  decide_ms={s.get('decide_ms')}  "
            f"end={s.get('end_reason')}  success={o.success}"
        )
        if o.input_tokens + o.cache_read_tokens + o.cache_write_tokens > 0 and o.output_tokens > 0:
            good += 1
        else:
            problems.append(f"{did[:8]} has zero tokens")
        if args.mode and s.get("routing_mode") != args.mode:
            problems.append(f"{did[:8]} was logged in mode {s.get('routing_mode')}")
        if args.mode == "route" and s.get("applied") and o.final_model != d.action.model:
            problems.append(f"{did[:8]} applied {d.action.model} but {o.final_model} ran")
    print(f"decisions {len(decisions)}, joined with non-zero tokens {good}")
    for p in problems:
        print(f"problem: {p}")
    return 0 if good >= args.expect and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
