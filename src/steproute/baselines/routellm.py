"""RouteLLM's matrix-factorisation router, two-model setting (``baselines`` extra: torch).

RouteLLM (arXiv 2406.18665) routes each query to a strong or a weak model: a router estimates
the strong model's win rate on the query, and the query goes to the strong model when that rate
is at least a threshold, calibrated so a chosen share of queries goes strong. The ``mf`` router,
the paper's best, learns a model embedding ``P[m]`` and a projection of the prompt embedding,
trained on pairwise battles with a Bradley-Terry loss; it is trained on many models and then
asked about one pair. The model and training loop are ``_vendor/routellm_mf.py``, adapted from
RouteLLM at the commit in ``UPSTREAM`` (Apache-2.0).

The mapping onto our setting:

- **Battles** come from the grid: on each ``train`` task, every ordered pair of configurations
  where one succeeded and the other failed is one battle won by the one that succeeded. Pairs
  where both succeeded or both failed are ties and are dropped, as upstream drops ties. Upstream
  trains on Chatbot Arena human preferences (plus judge-labelled data); nothing of that is used.
- **Models** are every configuration ``(model, effort)`` in the grid (the MF table spans all of
  them, as upstream's spans 64 Arena models).
- **Prompt embeddings** are ``ContextEncoder`` vectors (numeric state, no text). Upstream embeds
  prompts with OpenAI ``text-embedding-3-small``, a paid API call; that is removed.
- **The pair**: by default the strong arm is the configuration with the highest success rate on
  the training tasks (ties to the cheaper), and the weak arm is the cheapest configuration by
  mean training cost (often a free local model). ``pair`` overrides both.
- **Thresholds** are calibrated on the training tasks, as upstream calibrates on Arena prompts:
  for each share of strong calls in ``STRONG_SHARES``, the threshold is the ``1 - share``
  quantile of the training win rates. One policy per share.
- **Hyperparameters** are upstream's example values (``train_matrix_factorization.py``): dim
  128, projection on, Adam lr 3e-4, weight decay 1e-5, noise 0.1, batch 64, 100 epochs. CPU,
  seeded.
"""

from __future__ import annotations

from collections.abc import Sequence
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

__all__ = [
    "HPARAMS",
    "STRONG_SHARES",
    "UPSTREAM",
    "battles",
    "build_policies",
    "default_pair",
    "thresholds",
]

UPSTREAM = Upstream(
    name="RouteLLM mf",
    repository="https://github.com/lm-sys/RouteLLM",
    commit="0b64fdafe049e596a3f5657c219329f24af24198",
    version="routellm 0.2.0 (source at the commit; the package is not installed)",
    licence="Apache-2.0",
    paper="arXiv 2406.18665",
    use="adapted",
)

HPARAMS: dict[str, float | int] = {
    "dim": 128,
    "batch_size": 64,
    "num_epochs": 100,
    "alpha": 0.1,
    "lr": 3e-4,
    "weight_decay": 1e-5,
}
#: Shares of tasks sent to the strong arm; one calibrated threshold (and policy) each.
STRONG_SHARES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)


def battles(success: NDArray[np.float64]) -> list[tuple[int, int, int]]:
    """``(winner arm, loser arm, task row)`` for every discordant ordered pair on every task."""
    out: list[tuple[int, int, int]] = []
    n, k = success.shape
    for i in range(n):
        for a in range(k):
            for b in range(k):
                if a != b and success[i, a] > 0.5 and success[i, b] <= 0.5:
                    out.append((a, b, i))
    return out


def default_pair(
    arms: Sequence[ConfigKey], success: NDArray[np.float64], cost: NDArray[np.float64]
) -> tuple[int, int]:
    """``(weak, strong)`` arm indices: the cheapest arm, and the most successful one."""
    rate, mean_cost = success.mean(axis=0), cost.mean(axis=0)
    strong = min(range(len(arms)), key=lambda j: (-rate[j], mean_cost[j], j))
    weak = min((j for j in range(len(arms)) if j != strong), key=lambda j: (mean_cost[j], j))
    return weak, strong


def thresholds(win_rates: Sequence[float], shares: Sequence[float]) -> list[float]:
    """Upstream's calibration: the ``1 - share`` quantile of the calibration win rates."""
    v = np.asarray(win_rates, dtype=np.float64)
    return [float(np.quantile(v, 1.0 - s)) for s in shares]


def _resolve_pair(arms: Sequence[ConfigKey], pair: tuple[str, str]) -> tuple[int, int]:
    labels = [arm_label(a) for a in arms]
    try:
        return labels.index(pair[0]), labels.index(pair[1])
    except ValueError:
        raise ValueError(f"pair {pair} not among the grid's configurations {labels}") from None


def build_policies(
    data: BaselineData,
    encoder: ContextEncoder,
    *,
    seed: int = 0,
    shares: Sequence[float] = STRONG_SHARES,
    pair: tuple[str, str] | None = None,
) -> tuple[list[ArmPolicy], dict[str, Any]]:
    """Train the MF model on training battles and return one policy per strong share.

    ``pair`` is ``(weak, strong)`` as ``EFFORT:MODEL`` labels; by default ``default_pair``.
    """
    import torch

    from steproute.baselines._vendor.routellm_mf import MFModel, train_loops

    train, tune = data.tasks("train"), data.tasks("tune")
    if not train:
        raise ValueError("no complete training tasks")
    arms = data.arms
    contexts = encoder.transform_many([data.state(t) for t in train])
    success, cost = data.outcomes(train)
    weak, strong = _resolve_pair(arms, pair) if pair else default_pair(arms, success, cost)
    games = battles(success)
    if not games:
        raise ValueError("no discordant pairs in the training tasks: nothing to train on")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    net = MFModel(dim=int(HPARAMS["dim"]), num_models=len(arms), prompts=contexts)
    losses = train_loops(
        net,
        games,
        lr=float(HPARAMS["lr"]),
        weight_decay=float(HPARAMS["weight_decay"]),
        alpha=float(HPARAMS["alpha"]),
        num_epochs=int(HPARAMS["num_epochs"]),
        batch_size=int(HPARAMS["batch_size"]),
        generator=generator,
    )

    def win_rate(x: NDArray[np.float64]) -> float:
        return net.pred_win_rate(strong, weak, x)

    calib = [win_rate(x) for x in contexts]
    cuts = thresholds(calib, shares)

    tune_acc: float | None = None
    if tune:
        s_tune, _ = data.outcomes(tune)
        x_tune = encoder.transform_many([data.state(t) for t in tune])
        tune_games = battles(s_tune)
        if tune_games:
            right = sum(net.pred_win_rate(a, b, x_tune[i]) > 0.5 for a, b, i in tune_games)
            tune_acc = right / len(tune_games)

    policies: list[ArmPolicy] = []
    for share, cut in zip(shares, cuts, strict=True):

        def choose(x: NDArray[np.float64], cut: float = cut) -> ConfigKey:
            return arms[strong] if win_rate(x) >= cut else arms[weak]

        policies.append(ArmPolicy(f"routellm mf strong_share={share:g}", choose, encoder))
    provenance = {
        "upstream": UPSTREAM.to_dict(),
        "adapted_code": "steproute/baselines/_vendor/routellm_mf.py (Apache-2.0 notice kept)",
        "trained_on": "battles from the train tasks; retrained from scratch, no pretrained "
        "checkpoint, no OpenAI embeddings",
        "pair": {"weak": arm_label(arms[weak]), "strong": arm_label(arms[strong])},
        "battles": len(games),
        "hyperparameters": dict(HPARAMS),
        "strong_shares": list(shares),
        "thresholds": cuts,
        "final_training_loss": losses[-1],
        "tune_pairwise_accuracy": tune_acc,
        "context": encoder.describe(),
    }
    return policies, provenance
