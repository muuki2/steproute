"""Synthetic worlds with a known ground truth, for testing the evaluator. Stdlib only.

The world is small and discrete so every true value is an exact finite sum, not a Monte Carlo
estimate:

- A task has a hidden difficulty level ``d`` (finite set, known probabilities) and an
  observed ``hint`` that equals ``d`` with probability ``hint_accuracy`` and is otherwise
  uniform over the other levels. Policies see only the hint (``StepState.features``).
- A configuration ``k`` succeeds on a task with probability ``sigmoid(ability_k - delta_d -
  kappa)``. Every configuration faces the same ``delta_d``, so a task that defeats one model
  is harder for all of them: failures are correlated across models through the task.
  Given the task, runs are independent.
- Cost is ``price_k * tokens_d * noise`` with log-normal noise of mean 1, so costs differ by
  orders of magnitude between configurations and grow with difficulty. One configuration is
  local and free.
- Logged sessions add steps: the cache owner is the previous step's model, and a step that
  changes model pays a switch tax. The logging policy is epsilon-greedy with exact,
  logged propensities.

``python -m steproute.eval.synthetic --out DIR`` writes a grid, a trace store and a catalog
for trying ``steproute eval``.
"""

from __future__ import annotations

import argparse
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from steproute.core.catalog import Catalog
from steproute.core.schema import EFFORTS, Action, Decision, ModelProfile, Outcome, StepState
from steproute.core.store import TraceStore
from steproute.eval.data import GridRun, GridTask, write_grid
from steproute.eval.frontier import Point, envelope, success_at
from steproute.eval.targets import Distribution, DistributionTarget, TargetPolicy

__all__ = [
    "RULES_TOML",
    "SynthConfig",
    "SynthLogging",
    "SynthWorld",
    "Truth",
    "catalog",
    "catalog_toml",
    "default_world",
    "hint_policy",
    "make_grid",
    "make_logged",
    "optimal_task_policy",
    "task_actions",
    "true_logged_value",
    "true_margin",
    "true_task_value",
    "write_example",
]

PROMPT_TOKENS = 1000  # constant, so the routing state does not leak the hidden difficulty


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class SynthConfig:
    model: str
    effort: str
    ability: float
    usd_per_mtok: float
    ms_per_ktok: float
    local: bool = False

    @property
    def action(self) -> Action:
        return Action(self.model, self.effort)


@dataclass(frozen=True)
class SynthWorld:
    configs: tuple[SynthConfig, ...]
    difficulty: tuple[float, ...]
    level_probs: tuple[float, ...]
    tokens: tuple[float, ...]
    hint_accuracy: float = 0.8
    cost_sigma: float = 0.3
    #: (kind, extra difficulty) per step of a logged session.
    step_kinds: tuple[tuple[str, float], ...] = (
        ("plan", -0.5),
        ("main", 0.0),
        ("main", 0.3),
        ("tool_followup", -1.0),
    )
    step_token_frac: float = 0.1
    switch_tax: float = 0.25

    @property
    def levels(self) -> int:
        return len(self.difficulty)

    def config(self, action: Action) -> SynthConfig:
        for c in self.configs:
            if c.model == action.model and c.effort == action.effort:
                return c
        raise KeyError(action.label())

    def p(self, d: int, cfg: SynthConfig, kappa: float = 0.0) -> float:
        return _sigmoid(cfg.ability - self.difficulty[d] - kappa)

    def run_cost(self, d: int, cfg: SynthConfig) -> float:
        """Expected cost of one full-task run."""
        return cfg.usd_per_mtok * self.tokens[d] / 1e6

    def hint_probs(self, d: int) -> list[float]:
        other = (1.0 - self.hint_accuracy) / (self.levels - 1)
        return [self.hint_accuracy if h == d else other for h in range(self.levels)]

    def task_state(self, task_id: str, hint: int) -> StepState:
        return StepState(
            harness="synthetic",
            session_id=task_id,
            step_idx=0,
            kind="main",
            prompt_tokens=PROMPT_TOKENS,
            task_id=task_id,
            source="replay",
            features={"hint": hint},
        )


def default_world() -> SynthWorld:
    """Six configurations over four models, five difficulty levels."""
    return SynthWorld(
        configs=(
            SynthConfig("local-small", "medium", 0.0, 0.0, 30.0, local=True),
            SynthConfig("cloud-small", "low", 0.7, 0.5, 4.0),
            SynthConfig("cloud-small", "high", 1.3, 1.2, 8.0),
            SynthConfig("cloud-mid", "medium", 2.1, 4.0, 10.0),
            SynthConfig("cloud-big", "medium", 3.0, 15.0, 14.0),
            SynthConfig("cloud-big", "high", 3.4, 25.0, 25.0),
        ),
        difficulty=(-2.0, -0.5, 1.0, 2.5, 4.0),
        level_probs=(0.25, 0.25, 0.2, 0.15, 0.15),
        tokens=(20_000.0, 40_000.0, 80_000.0, 150_000.0, 250_000.0),
    )


def _lognormal(rng: random.Random, sigma: float) -> float:
    return math.exp(rng.gauss(-0.5 * sigma * sigma, sigma)) if sigma > 0 else 1.0


def _draw(rng: random.Random, probs: Sequence[float]) -> int:
    u = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if u < acc:
            return i
    return len(probs) - 1


# --- task level --------------------------------------------------------------------------


def task_actions(world: SynthWorld) -> tuple[Action, ...]:
    """Every single configuration plus every two-step cascade to a stronger one."""
    singles = [c.action for c in world.configs]
    cascades = [
        Action(a.model, a.effort, b.action)
        for a in world.configs
        for b in world.configs
        if b.ability > a.ability
    ]
    return (*singles, *cascades)


def _task_value(world: SynthWorld, d: int, action: Action) -> tuple[float, float]:
    reach, succ, cost = 1.0, 0.0, 0.0
    for step in action.chain():
        cfg = world.config(step)
        p = world.p(d, cfg)
        cost += reach * world.run_cost(d, cfg)
        succ += reach * p
        reach *= 1.0 - p
    return succ, cost


@dataclass(frozen=True)
class Truth:
    success: float
    cost: float

    @property
    def cost_per_solved(self) -> float:
        return self.cost / self.success if self.success > 0 else math.inf


def true_task_value(world: SynthWorld, target: TargetPolicy) -> Truth:
    """Exact population value of a task-level policy that reads only the hint."""
    s = c = 0.0
    for d, qd in enumerate(world.level_probs):
        for h, ph in enumerate(world.hint_probs(d)):
            for a, p in target.distribution(world.task_state("truth", h)):
                ts, tc = _task_value(world, d, a)
                s += qd * ph * p * ts
                c += qd * ph * p * tc
    return Truth(s, c)


def true_margin(truth: Truth, configs: Sequence[tuple[str, Truth]]) -> float:
    env = envelope([Point(label, t.cost, t.success) for label, t in configs])
    return truth.success - success_at(env, truth.cost)


def hint_policy(name: str, choice: Callable[[int], Action | Distribution]) -> TargetPolicy:
    """A policy that maps the hint to an action (or a distribution over actions)."""

    def fn(state: StepState) -> Distribution:
        out = choice(int(state.features["hint"]))
        return ((out, 1.0),) if isinstance(out, Action) else out

    return DistributionTarget(fn, name)


def optimal_task_policy(
    world: SynthWorld, cost_weight: float, actions: Sequence[Action] | None = None
) -> tuple[TargetPolicy, dict[int, Action]]:
    """The deterministic task-level policy maximising ``success - cost_weight * cost`` given
    the hint: the known optimum the evaluator must rank first."""
    acts = tuple(actions) if actions is not None else task_actions(world)
    table: dict[int, Action] = {}
    for h in range(world.levels):
        post = [qd * world.hint_probs(d)[h] for d, qd in enumerate(world.level_probs)]
        z = sum(post)

        def utility(a: Action, post: list[float] = post, z: float = z) -> float:
            total = 0.0
            for d, w in enumerate(post):
                s, c = _task_value(world, d, a)
                total += w / z * (s - cost_weight * c)
            return total

        table[h] = max(acts, key=utility)
    return hint_policy("optimal", lambda h: table[h]), table


def make_grid(
    world: SynthWorld,
    n_tasks: int,
    seed: int,
    *,
    unlabelled_rate: float = 0.0,
    split: str | None = "test",
) -> tuple[list[GridRun], list[GridTask]]:
    """A dense grid: every task run once under every configuration."""
    rng = random.Random(seed)
    runs: list[GridRun] = []
    tasks: list[GridTask] = []
    for i in range(n_tasks):
        tid = f"t{i:05d}"
        d = _draw(rng, world.level_probs)
        h = _draw(rng, world.hint_probs(d))
        tasks.append(GridTask(tid, split, world.task_state(tid, h), {"hint": h}))
        for cfg in world.configs:
            ok = rng.random() < world.p(d, cfg)
            noise = _lognormal(rng, world.cost_sigma)
            label: bool | None = ok
            if unlabelled_rate and rng.random() < unlabelled_rate:
                label = None
            runs.append(
                GridRun(
                    task_id=tid,
                    model=cfg.model,
                    effort=cfg.effort,
                    success=label,
                    cost_usd=world.run_cost(d, cfg) * noise,
                    latency_ms=cfg.ms_per_ktok * world.tokens[d] / 1000 * noise,
                    run_id=f"{tid}/{cfg.model}/{cfg.effort}",
                    session_id=f"{tid}/{cfg.model}/{cfg.effort}",
                )
            )
    return runs, tasks


# --- step level --------------------------------------------------------------------------


@dataclass
class SynthLogging:
    """Epsilon-greedy logging policy over the single configurations.

    Greedy: the configuration ``min(hint + 1, K - 1)`` in ability order, except that the
    current configuration is kept when it is within one place of that (a crude cache-aware
    rule). With probability ``epsilon`` a configuration is drawn uniformly instead. The
    distribution depends on the state's ``current_model``, so the logged state distribution
    depends on earlier logged actions, as in real sessions.
    """

    world: SynthWorld
    epsilon: float = 0.3
    name: str = "synthetic-logging"
    hindsight: bool = False

    def _ordered(self) -> list[SynthConfig]:
        return sorted(self.world.configs, key=lambda c: c.ability)

    def distribution(self, state: StepState) -> Distribution:
        cfgs = self._ordered()
        k = len(cfgs)
        target = min(int(state.features["hint"]) + 1, k - 1)
        g = target
        for i, c in enumerate(cfgs):
            if (
                c.model == state.current_model
                and c.effort == state.current_effort
                and abs(i - target) <= 1
            ):
                g = i
        return tuple(
            (c.action, (1.0 - self.epsilon) * (i == g) + self.epsilon / k)
            for i, c in enumerate(cfgs)
        )


def _step_state(
    world: SynthWorld, task_id: str, session: str, t: int, hint: int, cur: Action | None
) -> StepState:
    return StepState(
        harness="synthetic",
        session_id=session,
        step_idx=t,
        kind=world.step_kinds[t][0],
        prompt_tokens=PROMPT_TOKENS,
        task_id=task_id,
        current_model=None if cur is None else cur.model,
        current_effort=None if cur is None else cur.effort,
        source="replay",
        features={"hint": hint},
    )


def _step_cost(world: SynthWorld, d: int, cfg: SynthConfig, cur: Action | None) -> float:
    switch = cur is not None and cur.model != cfg.model
    tokens = world.tokens[d] * world.step_token_frac
    return cfg.usd_per_mtok * tokens / 1e6 * (1.0 + world.switch_tax * switch)


def make_logged(
    world: SynthWorld,
    n_tasks: int,
    seed: int,
    *,
    logging: SynthLogging | None = None,
    sessions_per_task: int = 2,
    unlabelled_rate: float = 0.0,
    logged_propensity: Callable[[Distribution, Action], float] | None = None,
) -> list[tuple[Decision, Outcome]]:
    """Logged step-level sessions. ``logged_propensity`` replaces the true propensity in
    the log (to simulate a collector that logs the wrong number); by default the true one is
    logged."""
    rng = random.Random(seed)
    log = logging or SynthLogging(world)
    actions = [c.action for c in world.configs]
    out: list[tuple[Decision, Outcome]] = []
    n = 0
    for i in range(n_tasks):
        tid = f"t{i:05d}"
        d = _draw(rng, world.level_probs)
        h = _draw(rng, world.hint_probs(d))
        for j in range(sessions_per_task):
            session = f"{tid}-s{j}"
            cur: Action | None = None
            for t, (_, kappa) in enumerate(world.step_kinds):
                state = _step_state(world, tid, session, t, h, cur)
                dist = log.distribution(state)
                idx = _draw(rng, [p for _, p in dist])
                action, prop = dist[idx]
                cfg = world.config(action)
                ok = rng.random() < world.p(d, cfg, kappa)
                noise = _lognormal(rng, world.cost_sigma)
                cost = _step_cost(world, d, cfg, cur) * noise
                label: bool | None = ok
                if unlabelled_rate and rng.random() < unlabelled_rate:
                    label = None
                did = f"d{n:07d}"
                n += 1
                logged = prop if logged_propensity is None else logged_propensity(dist, action)
                decision = Decision(
                    decision_id=did,
                    ts=1_767_225_600.0 + n,
                    state=state,
                    candidates=list(actions),
                    action=action,
                    policy=log.name,
                    propensity=logged,
                )
                outcome = Outcome(
                    decision_id=did,
                    ts=1_767_225_600.5 + n,
                    success=label,
                    final_model=action.model,
                    cost_usd=cost,
                    latency_ms=cfg.ms_per_ktok * world.tokens[d] * world.step_token_frac / 1000,
                )
                out.append((decision, outcome))
                cur = action
    return out


def true_logged_value(
    world: SynthWorld, target: TargetPolicy, logging: SynthLogging | None = None
) -> Truth:
    """Exact per-step value of ``target`` in the logging policy's state distribution.

    This is what step-level off-policy estimates estimate: the expected success and cost of
    one step routed by ``target`` in a state drawn as the logged sessions draw them. The
    cache owner of each state follows the *logging* policy's earlier choices; the value of
    running ``target`` for whole sessions is a different quantity (see docs).
    """
    log = logging or SynthLogging(world)
    T = len(world.step_kinds)
    s = c = 0.0
    for d, qd in enumerate(world.level_probs):
        for h, ph in enumerate(world.hint_probs(d)):
            w = qd * ph
            curs: dict[Action | None, float] = {None: 1.0}
            for t, (_, kappa) in enumerate(world.step_kinds):
                nxt: dict[Action | None, float] = {}
                for cur, m in curs.items():
                    state = _step_state(world, "truth", "truth", t, h, cur)
                    for a, p in target.distribution(state):
                        cfg = world.config(a)
                        s += w * m * p * world.p(d, cfg, kappa) / T
                        c += w * m * p * _step_cost(world, d, cfg, cur) / T
                    for a, p in log.distribution(state):
                        nxt[a] = nxt.get(a, 0.0) + m * p
                curs = nxt
    return Truth(s, c)


# --- files -------------------------------------------------------------------------------


def catalog(world: SynthWorld) -> Catalog:
    """A catalog naming the world's models, so catalog-based policies can run on it."""
    by_model: dict[str, list[SynthConfig]] = {}
    for c in world.configs:
        by_model.setdefault(c.model, []).append(c)
    order = sorted(by_model, key=lambda m: max(c.ability for c in by_model[m]))
    models = []
    for tier, m in enumerate(order, start=1):
        cfgs = by_model[m]
        efforts = tuple(sorted({c.effort for c in cfgs}, key=EFFORTS.index))
        price = max(c.usd_per_mtok for c in cfgs)
        models.append(
            ModelProfile(
                id=m,
                provider="ollama" if cfgs[0].local else "synthetic",
                price_in=price,
                price_out=price,
                context_window=1_000_000,
                local=cfgs[0].local,
                efforts=efforts,
                effort_output_mult={e: 1.0 for e in efforts},
                tier=tier,
                verified=True,
            )
        )
    return Catalog(tuple(models))


def catalog_toml(world: SynthWorld) -> str:
    """``catalog(world)`` as TOML for ``--catalog``."""
    lines = ["# Synthetic catalog written by steproute.eval.synthetic. Not real models.", ""]
    for m in catalog(world).models:
        efforts = ", ".join(f'"{e}"' for e in m.efforts)
        mult = ", ".join(f"{e} = 1.0" for e in m.efforts)
        lines += [
            "[[model]]",
            f'id = "{m.id}"',
            f'provider = "{m.provider}"',
            f"price_in = {m.price_in}",
            f"price_out = {m.price_out}",
            f"context_window = {m.context_window}",
            f"local = {'true' if m.local else 'false'}",
            f"tier = {m.tier}",
            f"efforts = [{efforts}]",
            f"effort_output_mult = {{ {mult} }}",
            "verified = true",
            "",
        ]
    return "\n".join(lines)


def write_example(out: Path, n_tasks: int = 200, seed: int = 0) -> dict[str, Path]:
    """Write ``grid.jsonl``, ``traces/`` and ``catalog.toml`` under ``out``."""
    world = default_world()
    out.mkdir(parents=True, exist_ok=True)
    runs, tasks = make_grid(world, n_tasks, seed)
    grid = out / "grid.jsonl"
    write_grid(grid, runs, tasks, {"generator": "steproute.eval.synthetic", "seed": seed})
    traces = out / "traces"
    store = TraceStore(traces, clock=lambda: 1_767_225_600.0)
    for dec, outc in make_logged(world, n_tasks, seed + 1):
        store.log_decision(dec)
        store.log_outcome(outc)
    cat = out / "catalog.toml"
    cat.write_text(catalog_toml(world), encoding="utf-8")
    rules = out / "rules.toml"
    rules.write_text(RULES_TOML, encoding="utf-8")
    return {"grid": grid, "traces": traces, "catalog": cat, "rules": rules}


#: A rule config for the synthetic catalog (its tiers run 1 to 4, its providers are
#: ``ollama`` and ``synthetic``), for ``--policy rules:PATH``.
RULES_TOML = """\
# Rule config for the synthetic catalog written by steproute.eval.synthetic.
[default]
tier = 3
effort = "medium"
[cheap]
tier = 2
effort = "low"
[models]
provider_order = ["synthetic", "ollama"]
"""


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a synthetic dataset for steproute eval.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    paths = write_example(args.out, args.tasks, args.seed)
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
