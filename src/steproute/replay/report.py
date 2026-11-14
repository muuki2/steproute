"""The pilot report: what a finished pilot measured, and the sparse-grid estimate redone from it.

``pilot_report`` reads an output directory (``runs.jsonl``, ``ledger.jsonl``, ``plan.json``,
``gate.json``) and returns Markdown for ``docs/results/pilot.md``. A directory whose paid
providers pointed at loopback (the mock upstream) is marked MOCK DATA at the top, so a test
result can never be mistaken for a real pilot. Standard library only.

The revised estimate: each sparse-grid configuration measured in the pilot is priced at its
measured mean cost per run; one that was not measured gets its ``steproute tasks estimate``
central value times the measured realised/estimated ratio of the same model (or, without
any, of the same provider; without that, the estimate unchanged, flagged).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog
from steproute.replay.config import DEFAULT_CAPS
from steproute.replay.journal import DEFERRED, LABELLED, Journal
from steproute.replay.ledger import Ledger
from steproute.replay.plan import ReplayPlan
from steproute.tasks.estimate import months_needed
from steproute.tasks.manifest import TaskManifest

__all__ = ["ConfigStats", "measure", "pilot_report"]


@dataclass
class ConfigStats:
    model: str
    effort: str
    provider: str
    runs: int = 0
    labelled: int = 0
    passed: int = 0
    cost: float = 0.0
    #: The same usage at the catalog's peak (list) price; ``cost`` is what was billed.
    list_cost: float = 0.0
    steps: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output_tokens: int = 0
    missing_usage: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    est_central: float = 0.0
    by_stratum: dict[str, list[int]] = field(default_factory=dict)  # stratum -> [passed, labelled]

    @property
    def success_rate(self) -> float | None:
        return self.passed / self.labelled if self.labelled else None

    @property
    def mean_cost(self) -> float | None:
        return self.cost / self.runs if self.runs else None

    @property
    def cache_share(self) -> float | None:
        total = self.input_tokens + self.cache_read + self.cache_write
        return self.cache_read / total if total else None

    @property
    def ratio(self) -> float | None:
        """Realised over estimated, both at list price (the estimate is at peak prices)."""
        if not self.runs or self.est_central <= 0:
            return None
        return (self.list_cost / self.runs) / self.est_central


def measure(
    journal: Journal, catalog: Catalog, manifest: TaskManifest, plan: ReplayPlan | None
) -> dict[tuple[str, str], ConfigStats]:
    """Per-configuration statistics over the last attempt of every finished cell."""
    est = {(s.model, s.effort): s.est_usd[0] for s in plan.specs} if plan else {}
    out: dict[tuple[str, str], ConfigStats] = {}
    for attempts in journal.by_cell().values():
        last = attempts[-1]
        if last.end is None or last.status in DEFERRED:
            continue  # pending: stopped for budget or balance, run again on resume
        end = last.end
        _, model, effort, _ = last.cell
        prov = catalog.get(model).provider if model in catalog else "?"
        st = out.setdefault(
            (model, effort),
            ConfigStats(model, effort, prov, est_central=est.get((model, effort), 0.0)),
        )
        status = str(end.get("status"))
        st.statuses[status] = st.statuses.get(status, 0) + 1
        st.runs += 1
        st.cost += float(end.get("cost_usd") or 0.0)
        st.list_cost += float(end.get("list_cost_usd", end.get("cost_usd")) or 0.0)
        st.steps += int(end.get("steps") or 0)
        u = end.get("usage") or {}
        st.input_tokens += int(u.get("input", 0))
        st.cache_read += int(u.get("cache_read", 0))
        st.cache_write += int(u.get("cache_write", 0))
        st.output_tokens += int(u.get("output", 0))
        st.missing_usage += int(end.get("calls_missing_usage", 0))
        if status in LABELLED:
            st.labelled += 1
            st.passed += status == "passed"
            tid = last.cell[0]
            stratum = manifest.get(tid).stratum if tid in manifest else None
            cell = st.by_stratum.setdefault(stratum or "?", [0, 0])
            cell[0] += status == "passed"
            cell[1] += 1
    return out


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def _usd(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"${x:,.{nd}f}"


def pilot_report(
    out: Path,
    *,
    catalog: Catalog,
    manifest: TaskManifest,
    pilot: ReplayPlan,
    sparse: ReplayPlan,
    caps: Mapping[str, float] = DEFAULT_CAPS,
) -> str:
    journal = Journal.open(out / "runs.jsonl")
    ledger = Ledger(out / "ledger.jsonl")
    ident: dict[str, Any] = {}
    if (out / "plan.json").exists():
        ident = json.loads((out / "plan.json").read_text("utf-8"))
    stats = measure(journal, catalog, manifest, pilot)
    lines: list[str] = []
    if ident.get("mock", False):
        lines += [
            "> **MOCK DATA. NOT A PILOT.** Every model in this run was the loopback mock upstream;",
            "> no provider was called and every number below is synthetic.",
            "",
        ]
    lines += [
        "# Pilot results (issue 18)",
        "",
        f"Output directory: `{out.name}`. Plan: `{ident.get('plan', '?')}`, manifest "
        f"`{ident.get('manifest', manifest.name)}`. Only the last attempt of each finished cell "
        "counts; costs are realised provider cost from the ledger (usage priced with the catalog; "
        "a call without usage is charged its worst case and counted in *missing usage*). "
        "*$ billed* is what was charged, off-peak calls at off-peak prices; *$ at list price* "
        "is the same usage at the catalog's peak (list) price, the number to compare across "
        "providers and with the estimates.",
        "",
        "## Per configuration",
        "",
        "| Model | Effort | Runs | Success (labelled) | Unlabelled | $/run billed | $ total billed "
        "| $ total at list price | Est. $/run (list) | Realised / est. (list) | Steps/run "
        "| Cache-read share | Missing usage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, effort), st in sorted(
        stats.items(), key=lambda kv: (kv[1].provider != "ollama", kv[1].provider, kv[0])
    ):
        unl = st.runs - st.labelled
        lines.append(
            f"| {model} | {effort} | {st.runs} "
            f"| {_pct(st.success_rate)} ({st.passed}/{st.labelled}) "
            f"| {unl} | {_usd(st.mean_cost, 4)} | {_usd(st.cost, 2)} | {_usd(st.list_cost, 2)} "
            f"| {_usd(st.est_central, 4)} "
            f"| {'n/a' if st.ratio is None else f'{st.ratio:.2f}'} "
            f"| {st.steps / st.runs if st.runs else 0:.1f} | {_pct(st.cache_share)} "
            f"| {st.missing_usage} |"
        )
    lines += [
        "",
        "## Spend by provider and month (ledger)",
        "",
        "| Provider | Month | $ |",
        "|---|---|---|",
    ]
    for (prov, month), usd in sorted(ledger.by_month().items()):
        lines.append(f"| {prov} | {month} | {_usd(usd, 2)} |")
    if ledger.open_intents():
        lines.append(f"\n{len(ledger.open_intents())} unsettled call(s) are charged at worst case.")
    periods = ledger.by_period()
    if any(k[1] in ("peak", "offpeak") for k in periods):
        lines += [
            "",
            "## Spend by time of day (ledger)",
            "",
            "| Provider | Period | $ billed | $ at list price |",
            "|---|---|---|---|",
        ]
        for (prov, period), (billed, listed) in sorted(periods.items()):
            lines.append(f"| {prov} | {period} | {_usd(billed, 2)} | {_usd(listed, 2)} |")
    pending: dict[str, int] = {}
    for attempts in journal.by_cell().values():
        if attempts[-1].status in DEFERRED:
            pending[str(attempts[-1].status)] = pending.get(str(attempts[-1].status), 0) + 1
    if pending:
        lines += [
            "",
            "Pending cells stopped early (not failures, not counted above; a resume runs them): "
            + ", ".join(f"{k} {v}" for k, v in sorted(pending.items()))
            + ".",
        ]
    status_tot: dict[str, int] = {}
    for st in stats.values():
        for k, v in st.statuses.items():
            status_tot[k] = status_tot.get(k, 0) + v
    lines += ["", "## Run statuses", "", "| Status | Runs |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(status_tot.items())]
    lines += ["", "## Success by stratum (labelled runs, all configurations of a model)", ""]
    by_model: dict[str, dict[str, list[int]]] = {}
    for st in stats.values():
        m = by_model.setdefault(st.model, {})
        for k, (p, n) in st.by_stratum.items():
            c = m.setdefault(k, [0, 0])
            c[0] += p
            c[1] += n
    strata = ["easy", "medium", "hard", "unsolved"]
    lines += ["| Model | " + " | ".join(strata) + " |", "|---|" + "---|" * len(strata)]
    for model, m in sorted(by_model.items()):
        cells = [f"{m[s][0]}/{m[s][1]}" if s in m else "-" for s in strata]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    if (out / "gate.json").exists():
        g = json.loads((out / "gate.json").read_text("utf-8"))
        lines += [
            "",
            "## Stage gate",
            "",
            f"{'Cut' if g.get('cut') else 'Passed'}: {g.get('reason')}",
        ]

    # --- revised sparse estimate ---
    model_ratio: dict[str, list[float]] = {}
    prov_ratio: dict[str, list[float]] = {}
    for st in stats.values():
        if st.ratio is not None and st.provider not in ("ollama",):
            model_ratio.setdefault(st.model, [0.0, 0.0])
            model_ratio[st.model][0] += st.list_cost
            model_ratio[st.model][1] += st.est_central * st.runs
            prov_ratio.setdefault(st.provider, [0.0, 0.0])
            prov_ratio[st.provider][0] += st.list_cost
            prov_ratio[st.provider][1] += st.est_central * st.runs
    doc_est: dict[str, float] = {}
    new_est: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    counts: dict[tuple[str, str], int] = {}
    for s in sparse.specs:
        counts[(s.model, s.effort)] = counts.get((s.model, s.effort), 0) + 1
    for s in sparse.specs:
        if s.local:
            continue
        doc_est[s.provider] = doc_est.get(s.provider, 0.0) + s.est_usd[0]
        cs = stats.get((s.model, s.effort))
        src = sources.setdefault(
            s.provider, {"measured": 0, "model ratio": 0, "provider ratio": 0, "unmeasured": 0}
        )
        if cs is not None and cs.runs:
            new_est[s.provider] = new_est.get(s.provider, 0.0) + cs.list_cost / cs.runs
            src["measured"] += 1
        elif s.model in model_ratio and model_ratio[s.model][1] > 0:
            r = model_ratio[s.model][0] / model_ratio[s.model][1]
            new_est[s.provider] = new_est.get(s.provider, 0.0) + r * s.est_usd[0]
            src["model ratio"] += 1
        elif s.provider in prov_ratio and prov_ratio[s.provider][1] > 0:
            r = prov_ratio[s.provider][0] / prov_ratio[s.provider][1]
            new_est[s.provider] = new_est.get(s.provider, 0.0) + r * s.est_usd[0]
            src["provider ratio"] += 1
        else:
            new_est[s.provider] = new_est.get(s.provider, 0.0) + s.est_usd[0]
            src["unmeasured"] += 1
    lines += [
        "",
        "## Sparse grid (option B), re-estimated from the pilot",
        "",
        "Runs priced at their measured $/run at list (peak) price where the pilot ran that "
        "configuration, otherwise "
        "the task-set estimate times the measured ratio of the same model (or provider). "
        "Months at the per-provider monthly cap.",
        "",
        "| Provider | Runs | Task-set central $ | Re-estimated $ | Months at cap | "
        "Measured / model ratio / provider ratio / unmeasured runs |",
        "|---|---|---|---|---|---|",
    ]
    for prov in sorted(doc_est):
        runs = sum(1 for s in sparse.specs if s.provider == prov)
        cap = caps.get(prov, 0.0)
        src = sources[prov]
        lines.append(
            f"| {prov} | {runs} | {_usd(doc_est[prov], 0)} | {_usd(new_est.get(prov), 0)} | "
            f"{months_needed(new_est.get(prov, 0.0), cap) if cap else 'n/a'} | "
            f"{src['measured']} / {src['model ratio']} / {src['provider ratio']} / "
            f"{src['unmeasured']} |"
        )
    local_runs = sum(1 for s in sparse.specs if s.local)
    lines += [
        "",
        f"Local runs in the sparse grid: {local_runs} (free; wall time not measured here).",
        "",
    ]
    return "\n".join(lines)
