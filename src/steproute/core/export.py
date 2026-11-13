"""Export of publishable traces. Standard library only.

Decision D5: only traces from benchmark replays are published. ``export_replay`` has no
parameter that can include personal records.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from steproute.core.privacy import ScrubOptions, scrub_decision, scrub_outcome
from steproute.core.schema import Decision
from steproute.core.store import TraceStore, encode_line

__all__ = ["ExportSummary", "export_replay"]


@dataclass(frozen=True)
class ExportSummary:
    decisions: int = 0
    outcomes: int = 0
    withheld_decisions: int = 0
    withheld_outcomes: int = 0

    @property
    def exported(self) -> int:
        return self.decisions + self.outcomes

    @property
    def withheld(self) -> int:
        return self.withheld_decisions + self.withheld_outcomes


def export_replay(
    store: TraceStore, out: str | os.PathLike[str], *, scrub: ScrubOptions | None = None
) -> ExportSummary:
    """Write the replay records of ``store`` to ``out`` as JSONL in the store's line format.

    A decision is included only when ``state.source == "replay"`` (a decision id that also
    appears with any other source is withheld). An outcome is included only when its decision
    was; orphan outcomes are withheld because their source is unknown. With ``scrub`` the
    records are scrubbed first.
    """
    replay: set[str] = set()
    other: set[str] = set()
    for rec in store.records():
        if isinstance(rec, Decision):
            (replay if rec.state.source == "replay" else other).add(rec.decision_id)
    allowed = replay - other

    target = Path(out)
    tmp = target.with_name(target.name + ".tmp")
    dec = out_n = wdec = wout = 0
    try:
        with open(tmp, "wb") as f:
            for rec in store.records():
                if isinstance(rec, Decision):
                    if rec.decision_id in allowed and rec.state.source == "replay":
                        d = scrub_decision(rec, scrub) if scrub else rec
                        f.write(encode_line("decision", d.to_dict()))
                        dec += 1
                    else:
                        wdec += 1
                elif rec.decision_id in allowed:
                    o = scrub_outcome(rec, scrub) if scrub else rec
                    f.write(encode_line("outcome", o.to_dict()))
                    out_n += 1
                else:
                    wout += 1
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return ExportSummary(dec, out_n, wdec, wout)
