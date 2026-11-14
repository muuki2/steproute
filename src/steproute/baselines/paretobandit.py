"""ParetoBandit as an evaluator policy, through the ``paretobandit`` package (``baselines`` extra).

ParetoBandit (arXiv 2604.00136) is a disjoint LinUCB router: one ridge regression per model
(arm) predicts the reward from a context vector, and the arm picked is

    argmax_m  x . theta_m + alpha * sqrt(x' A_m^-1 x) - lambda_c * normcost(m)

where ``normcost`` is the arm's price on a log scale between two anchors. A budget pacer can add
to ``lambda_c`` online; it is not used here (see ``docs/BASELINES.md``).

The mapping onto our setting:

- **Arms** are configurations, ``(model, effort)``, so effort is routed too. An arm's price is
  its mean realised cost per task on the training tasks (registered as the arm's per-1k-token
  price, so ``normcost`` is the log of that mean between the cheapest and dearest arm, the
  paper's "cost range anchored to arm cost extremes"). A free arm sits at the floor.
- **Context** is ``ContextEncoder`` plus the bias term the library expects last
  (``FeatureService.for_precomputed``): numeric state, no prompt embedding.
- **Reward** is task success, 0 or 1 (the library clips to [0, 1]).
- **Training**, two variants, both on the ``train`` tasks only:

  - ``full``: every arm is updated with its own outcome on every training task. The grid is
    full information, so this is the offline warm start the library builds its priors from,
    with nothing withheld.
  - ``online``: the router routes the training tasks one at a time (seeded order) and sees only
    the outcome of the arm it picked (``route`` then ``process_feedback``): the bandit feedback it
    was built for.

  After training the router is frozen; the evaluator asks it for one arm per test state.
- **Hyperparameters**: ``alpha`` is the library's tuned value for each start, 0.01 for the
  warm (``full``) start (``BEST_K3_HPARAMS``) and 0.05 for the cold (``online``) start
  (``BEST_K3_TABULA_RASA_HPARAMS``); ``init_lambda = 1.0`` (default);
  ``forgetting_factor = 1.0`` (the data is stationary; the default 0.997 is for drifting
  traffic), cold start (``priors="none"``: the shipped priors are for other
  models). ``lambda_c`` is swept (``COST_PENALTIES``), one policy each.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from steproute.baselines.common import (
    ArmPolicy,
    BaselineData,
    ContextEncoder,
    Upstream,
    arm_label,
)
from steproute.eval.data import ConfigKey

__all__ = ["COST_PENALTIES", "UPSTREAM", "BanditArms", "build_policies", "train_router"]

UPSTREAM = Upstream(
    name="ParetoBandit",
    repository="https://github.com/ParetoBandit/ParetoBandit",
    commit="72ccbefe5cb858c95e00368e2b68de3c2b28a2ee",
    version="paretobandit 0.1.1 (PyPI; src/ identical to tag v0.1.1, e20a5979)",
    licence="Apache-2.0",
    paper="arXiv 2604.00136",
    use="dependency",
)

#: lambda_c values: 0 is quality only, 0.3 the library's default.
COST_PENALTIES: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0)
#: Exploration coefficient per training mode: the library's own tuned values.
ALPHA: dict[str, float] = {"full": 0.01, "online": 0.05}
INIT_LAMBDA = 1.0
FORGETTING = 1.0
MODES = ("full", "online")


def _arm_id(arm: ConfigKey) -> str:
    return f"{arm[0]}|{arm[1]}"


@dataclass(frozen=True)
class BanditArms:
    """The arms and their registry entries (prices from training-task mean cost)."""

    arms: tuple[ConfigKey, ...]
    mean_cost: tuple[float, ...]

    @classmethod
    def from_data(cls, data: BaselineData, tasks: Sequence[str]) -> BanditArms:
        _, cost = data.outcomes(tasks)
        return cls(data.arms, tuple(float(v) for v in cost.mean(axis=0)))

    def registry(self) -> dict[str, dict[str, Any]]:
        # The router's normalised cost reads (input + output) / 2 per 1M tokens / 1000 as the
        # per-1k price; registering 1000 * mean task cost makes that per-1k price the mean cost.
        return {
            _arm_id(a): {
                "model_id": _arm_id(a),
                "input_cost_per_m": 1000.0 * c,
                "output_cost_per_m": 1000.0 * c,
            }
            for a, c in zip(self.arms, self.mean_cost, strict=True)
        }

    def anchors(self) -> tuple[float, float]:
        """``(floor, ceiling)`` for the log cost scale: the cheapest paid and dearest arm."""
        paid = [c for c in self.mean_cost if c > 0]
        if not paid:
            return 1e-6, 1e-5
        lo, hi = min(paid), max(paid)
        return lo, hi if hi > lo else lo * 10.0

    def by_id(self) -> dict[str, ConfigKey]:
        return {_arm_id(a): a for a in self.arms}


def _with_bias(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([x, np.ones(1)])


def train_router(
    arms: BanditArms,
    contexts: NDArray[np.float64],
    success: NDArray[np.float64],
    *,
    mode: str,
    cost_penalty: float,
    seed: int,
) -> Any:
    """A trained ``pareto_bandit.BanditRouter``. ``contexts`` has one row per training task
    (without the bias column); ``success`` one row per task and one column per arm."""
    from pareto_bandit import BanditRouter, RouterConfig
    from pareto_bandit.feature_service import FeatureService

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    floor, ceiling = arms.anchors()
    router = BanditRouter.create(
        model_registry=arms.registry(),
        priors="none",
        feature_service=FeatureService.for_precomputed(dimension=contexts.shape[1] + 1),
        alpha=ALPHA[mode],
        init_lambda=INIT_LAMBDA,
        forgetting_factor=FORGETTING,
        bandit_seed=seed,
        cost_penalty=cost_penalty,
        config=RouterConfig(market_cost_floor=floor, market_cost_ceiling=ceiling),
    )
    ids = [_arm_id(a) for a in arms.arms]
    order = list(range(len(contexts)))
    random.Random(f"{seed}:paretobandit").shuffle(order)
    for i in order:
        x = _with_bias(contexts[i])
        if mode == "full":
            for j, arm_id in enumerate(ids):
                router.update(arm_id, x, float(success[i, j]))
        else:
            picked, log = router.route(x)
            router.process_feedback(log.request_id, float(success[i, ids.index(picked)]))
    return router


def _chooser(router: Any, by_id: dict[str, ConfigKey]) -> Any:
    def choose(x: NDArray[np.float64]) -> ConfigKey:
        picked, _ = router.route(_with_bias(x))
        return by_id[str(picked)]

    return choose


def build_policies(
    data: BaselineData,
    encoder: ContextEncoder,
    *,
    seed: int = 0,
    cost_penalties: Sequence[float] = COST_PENALTIES,
    modes: Sequence[str] = MODES,
) -> tuple[list[ArmPolicy], dict[str, Any]]:
    """One policy per ``(mode, lambda_c)``, trained on the ``train`` tasks.

    ``encoder`` must already be fitted on the training states.
    """
    train = data.tasks("train")
    if not train:
        raise ValueError("no complete training tasks")
    contexts = encoder.transform_many([data.state(t) for t in train])
    success, _ = data.outcomes(train)
    arms = BanditArms.from_data(data, train)
    policies: list[ArmPolicy] = []
    for mode in modes:
        for lam in cost_penalties:
            router = train_router(arms, contexts, success, mode=mode, cost_penalty=lam, seed=seed)
            policies.append(
                ArmPolicy(
                    f"paretobandit {mode} lambda_c={lam:g}", _chooser(router, arms.by_id()), encoder
                )
            )
    floor, ceiling = arms.anchors()
    provenance = {
        "upstream": UPSTREAM.to_dict(),
        "trained_on": "train tasks only (retrained from scratch; no shipped priors or PCA)",
        "modes": {
            "full": "every arm updated with its outcome on every training task",
            "online": "route, then reward of the picked arm only (bandit feedback)",
        },
        "hyperparameters": {
            "alpha": dict(ALPHA),
            "init_lambda": INIT_LAMBDA,
            "forgetting_factor": FORGETTING,
            "cost_penalty_sweep": list(cost_penalties),
            "priors": "none",
            "budget_pacer": None,
            "cost_anchors_per_task_usd": [floor, ceiling],
        },
        "arms": {arm_label(a): c for a, c in zip(arms.arms, arms.mean_cost, strict=True)},
        "arm_price": "mean realised cost per task on the train tasks (USD)",
        "context": encoder.describe(),
    }
    return policies, provenance
