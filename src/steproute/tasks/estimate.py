"""Cost and time estimate for the replay grid, from editable assumptions and catalog prices.

Standard library only. Prices come from the catalog; everything else (steps per run, tokens per
step, cache hit rate, effort multipliers) from ``cost_assumptions.toml``. Each step of a run is
priced with ``steproute.core.cost.price_usage``, so price tiers (Haiku above 100K) and cache
multipliers are the same as everywhere else in steproute. See ``docs/TASKSET.md``.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from steproute.core.catalog import Catalog
from steproute.core.cost import AttemptUsage, price_usage
from steproute.core.schema import ModelProfile

__all__ = [
    "PLANS",
    "SCENARIOS",
    "Assumptions",
    "Block",
    "Plan",
    "PlanEstimate",
    "RunEstimate",
    "Scenario",
    "estimate_plan",
    "estimate_run",
    "load_assumptions",
    "plan_configs",
]

SCENARIOS = ("low", "central", "high")


@dataclass(frozen=True)
class Scenario:
    """Token shape of one run. Prompt at step k is ``prompt_base + growth_per_step * k``."""

    prompt_base: float
    growth_per_step: float
    output_per_step: float
    cache_hit: float
    steps_mult: float


@dataclass(frozen=True)
class LocalSpeed:
    prefill_tps: float
    gen_tps: float


@dataclass(frozen=True)
class Assumptions:
    scenarios: Mapping[str, Scenario]
    steps: Mapping[str, tuple[int, ...]]  # model id (or "default") -> step-count quantiles
    output_mult: Mapping[str, float]
    steps_mult: Mapping[str, float]
    local: Mapping[str, LocalSpeed]  # model id -> throughput
    step_limit: int = 250
    run_cost_cap_usd: float | None = 3.0
    local_context_cap: int = 32768
    eval_seconds: float = 120.0

    def quantiles(self, model: str) -> tuple[int, ...]:
        q = self.steps.get(model) or self.steps.get("default")
        if not q:
            raise KeyError(f"no step quantiles for {model!r} and no [steps.default]")
        return q


def _num(table: Mapping[str, Any], key: str, where: str) -> float:
    v = table.get(key)
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise ValueError(f"{where}: {key!r} must be a number")
    return float(v)


def parse_assumptions(data: Mapping[str, Any]) -> Assumptions:
    """Validate parsed ``cost_assumptions.toml`` data."""
    scen: dict[str, Scenario] = {}
    for name, t in data.get("scenario", {}).items():
        where = f"scenario.{name}"
        s = Scenario(
            prompt_base=_num(t, "prompt_base", where),
            growth_per_step=_num(t, "growth_per_step", where),
            output_per_step=_num(t, "output_per_step", where),
            cache_hit=_num(t, "cache_hit", where),
            steps_mult=_num(t, "steps_mult", where),
        )
        if not 0.0 <= s.cache_hit <= 1.0:
            raise ValueError(f"{where}: cache_hit must be in [0, 1]")
        scen[name] = s
    missing = [s for s in SCENARIOS if s not in scen]
    if missing:
        raise ValueError(f"missing scenario(s): {', '.join(missing)}")
    steps = {
        model: tuple(int(x) for x in t["quantiles"]) for model, t in data.get("steps", {}).items()
    }
    effort = data.get("effort", {})
    local = {
        str(t["model"]): LocalSpeed(
            _num(t, "prefill_tps", f"local.{k}"), _num(t, "gen_tps", f"local.{k}")
        )
        for k, t in data.get("local", {}).items()
    }
    limits = data.get("limits", {})
    cap = limits.get("run_cost_cap_usd")
    return Assumptions(
        scenarios=scen,
        steps=steps,
        output_mult={k: float(v) for k, v in effort.get("output_mult", {}).items()},
        steps_mult={k: float(v) for k, v in effort.get("steps_mult", {}).items()},
        local=local,
        step_limit=int(limits.get("step_limit", 250)),
        run_cost_cap_usd=float(cap) if cap is not None else None,
        local_context_cap=int(limits.get("local_context_cap", 32768)),
        eval_seconds=float(limits.get("eval_seconds", 120.0)),
    )


def load_assumptions(path: str | Path | None = None) -> Assumptions:
    """Load an assumptions file, or the default one shipped with the package."""
    if path is None:
        text = (resources.files("steproute.tasks") / "cost_assumptions.toml").read_text("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    return parse_assumptions(tomllib.loads(text))


@dataclass(frozen=True)
class RunEstimate:
    """Expected values for one run of one configuration, averaged over the step quantiles."""

    usd: float
    seconds: float  # wall time: local generation and prefill, plus eval_seconds
    steps: float
    input_tokens: float  # sum of prompt tokens over all steps
    output_tokens: float
    capped_share: float  # share of quantiles stopped by the cost cap or the local context cap


def estimate_run(
    profile: ModelProfile,
    effort: str,
    assumptions: Assumptions,
    scenario: str = "central",
    *,
    caching: bool = True,
) -> RunEstimate:
    """Expected cost and time of one run. ``caching=False`` models a scaffold or proxy that sets
    no Anthropic cache breakpoints (every prompt token billed at the input price)."""
    s = assumptions.scenarios[scenario]
    out = s.output_per_step * assumptions.output_mult.get(effort, 1.0)
    smult = s.steps_mult * assumptions.steps_mult.get(effort, 1.0)
    speed = assumptions.local.get(profile.id)
    cap = assumptions.run_cost_cap_usd
    totals = [0.0] * 5  # usd, seconds, steps, input, output
    capped = 0
    qs = assumptions.quantiles(profile.id)
    # Only explicit caching (a write premium, i.e. Anthropic) can be switched off by the
    # scaffold; DeepSeek caches automatically and a local server reuses its KV cache.
    use_cache = caching or profile.cache_write_mult <= 1.0
    for q in qs:
        n = min(assumptions.step_limit, max(1, round(q * smult)))
        usd = secs = inp = 0.0
        prev = 0.0
        steps = 0
        stopped = False
        for k in range(n):
            prompt = s.prompt_base + s.growth_per_step * k
            if profile.local and prompt + out > assumptions.local_context_cap:
                stopped = True
                break
            hit = use_cache and k > 0 and prev >= profile.cache_min_tokens
            read = s.cache_hit * prev if hit else 0.0
            if profile.cache_write_mult > 1.0 and use_cache and prompt >= profile.cache_min_tokens:
                usage = AttemptUsage(
                    cache_read_tokens=read, cache_write_tokens=prompt - read, output_tokens=out
                )
            else:
                usage = AttemptUsage(
                    input_tokens=prompt - read, cache_read_tokens=read, output_tokens=out
                )
            usd += price_usage(profile, usage).total_usd
            if speed is not None:
                secs += (prompt - read) / speed.prefill_tps + out / speed.gen_tps
            inp += prompt
            prev = prompt
            steps += 1
            if cap is not None and not profile.local and usd >= cap:
                stopped = True
                break
        capped += stopped
        for i, v in enumerate((usd, secs + assumptions.eval_seconds, steps, inp, out * steps)):
            totals[i] += v
    m = len(qs)
    usd_m, secs_m, steps_m, in_m, out_m = (t / m for t in totals)
    return RunEstimate(usd_m, secs_m, steps_m, in_m, out_m, capped_share=capped / m)


# --- plans -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    """``tasks`` tasks, each run under every configuration in ``configs``."""

    tasks: int
    configs: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Plan:
    name: str
    description: str
    blocks: tuple[Block, ...]


def _default_effort(p: ModelProfile) -> str:
    d = p.meta.get("default_effort")
    if isinstance(d, str) and d in p.efforts:
        return d
    return p.efforts[len(p.efforts) // 2]


def plan_configs(catalog: Catalog, efforts: str, *, holdout: bool = False) -> tuple[Any, ...]:
    """Configurations of the pool: ``efforts`` is ``all``, ``default``, ``rest`` (all but the
    default) or a comma-separated list (efforts a model lacks are skipped)."""
    out: list[tuple[str, str]] = []
    models = list(catalog) if holdout else list(catalog.pool())
    for p in models:
        d = _default_effort(p)
        if efforts == "all":
            chosen = list(p.efforts)
        elif efforts == "default":
            chosen = [d]
        elif efforts == "rest":
            chosen = [e for e in p.efforts if e != d]
        else:
            wanted = [e.strip() for e in efforts.split(",")]
            chosen = [e for e in p.efforts if e in wanted] or [d]
        out += [(p.id, e) for e in chosen]
    return tuple(out)


#: Pilot configurations of the paid models (docs/TASKSET.md, "Pilot design"); every local
#: configuration also runs. Models not listed here run no configuration in the pilot.
PILOT_EFFORTS: Mapping[str, tuple[str, ...]] = {
    "claude-haiku-5-5": ("low", "medium", "high", "xhigh", "max"),
    "claude-sonnet-5-5": ("low", "high"),
    "claude-opus-5-5": ("medium",),
    "claude-fable-5-2": ("low",),
    "deepseek-flash": ("none", "low", "high", "max"),
    "deepseek-v4-pro": ("high",),
}


def _pilot_configs(catalog: Catalog) -> tuple[tuple[str, str], ...]:
    out = []
    for m, e in plan_configs(catalog, "all", holdout=True):
        p = catalog.get(m)
        if p.local or e in PILOT_EFFORTS.get(m, ()):
            out.append((m, e))
    return tuple(out)


def _split(configs: Sequence[tuple[str, str]], catalog: Catalog, local: bool) -> tuple[Any, ...]:
    return tuple(c for c in configs if catalog.get(c[0]).local == local)


def build_plans(catalog: Catalog, n_tasks: int = 150, pilot_tasks: int = 20) -> dict[str, Plan]:
    """The options in ``docs/TASKSET.md``. Local configurations (the holdout included) cost
    nothing but time, so every plan runs all of them on all of its tasks."""
    every = plan_configs(catalog, "all", holdout=True)
    local_all = _split(every, catalog, local=True)
    sweep = max(1, round(n_tasks / 5))
    small = max(1, round(n_tasks * 0.4))
    return {
        "dense": Plan("dense", f"{n_tasks} tasks x every configuration", (Block(n_tasks, every),)),
        "sparse": Plan(
            "sparse",
            f"{n_tasks} tasks x every local configuration and each paid model at its default "
            f"effort, plus the paid models' other efforts on {sweep} tasks",
            (
                Block(n_tasks, local_all),
                Block(n_tasks, _split(plan_configs(catalog, "default"), catalog, local=False)),
                Block(sweep, _split(plan_configs(catalog, "rest"), catalog, local=False)),
            ),
        ),
        "small": Plan("small", f"{small} tasks x every configuration", (Block(small, every),)),
        "pilot": Plan(
            "pilot",
            f"{pilot_tasks} pilot tasks x every local configuration and the pilot's paid "
            "configurations",
            (Block(pilot_tasks, _pilot_configs(catalog)),),
        ),
    }


PLANS = ("dense", "sparse", "small", "pilot")


@dataclass(frozen=True)
class PlanEstimate:
    plan: Plan
    # provider -> scenario -> USD; provider -> runs; local wall hours per scenario
    usd: Mapping[str, Mapping[str, float]]
    runs: Mapping[str, int]
    hours: Mapping[str, Mapping[str, float]]
    per_config: Sequence[tuple[str, str, int, Mapping[str, RunEstimate]]]


def estimate_plan(
    plan: Plan,
    catalog: Catalog,
    assumptions: Assumptions,
    *,
    scenarios: Iterable[str] = SCENARIOS,
    caching: bool = True,
) -> PlanEstimate:
    scen = tuple(scenarios)
    usd: dict[str, dict[str, float]] = {}
    hours: dict[str, dict[str, float]] = {}
    runs: dict[str, int] = {}
    rows: list[tuple[str, str, int, Mapping[str, RunEstimate]]] = []
    cache: dict[tuple[str, str, str], RunEstimate] = {}
    for block in plan.blocks:
        for model, effort in block.configs:
            p = catalog.get(model)
            ests = {}
            for sc in scen:
                key = (model, effort, sc)
                if key not in cache:
                    cache[key] = estimate_run(p, effort, assumptions, sc, caching=caching)
                ests[sc] = cache[key]
                usd.setdefault(p.provider, {}).setdefault(sc, 0.0)
                usd[p.provider][sc] += block.tasks * ests[sc].usd
                hours.setdefault(p.provider, {}).setdefault(sc, 0.0)
                hours[p.provider][sc] += block.tasks * ests[sc].seconds / 3600.0
            runs[p.provider] = runs.get(p.provider, 0) + block.tasks
            rows.append((model, effort, block.tasks, ests))
    return PlanEstimate(plan=plan, usd=usd, runs=runs, hours=hours, per_config=rows)


def months_needed(usd: float, monthly_cap: float) -> int:
    """Calendar months to spend ``usd`` at ``monthly_cap`` per month (at least 1)."""
    return max(1, math.ceil(usd / monthly_cap)) if monthly_cap > 0 else 0
