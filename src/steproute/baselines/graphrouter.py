"""LLMRouter's ``graphrouter`` as an evaluator policy (``baselines`` extra: torch, torch-geometric).

GraphRouter (arXiv 2410.03834), as LLMRouter implements it (``llmrouter/models/graphrouter``):
queries and LLMs are nodes, every query is joined to every LLM by an edge carrying the observed
``performance``, two GeneralConv layers give each LLM a node state, and an edge's score is
``sigmoid(mean(query_features * llm_state))``. It is trained with binary cross-entropy to score
1 on each query's best LLM (one-hot argmax of ``performance``) and routes a new query to the LLM
with the highest score. The model and its training loop are ``_vendor/graphrouter_gnn.py``,
adapted from LLMRouter at the commit in ``UPSTREAM`` (MIT). The data preparation of
``router.py`` is re-done here for our grid.

The mapping onto our setting:

- **LLM nodes are configurations** ``(model, effort)``. Their features are the tabular
  predictor's configuration block (profile and probe results on the probe tasks: tier, local,
  prices, context window, effort rank and multiplier, provider, probe success and cost), min-max
  scaled across configurations, as upstream scales its LLM embeddings. Upstream uses a text
  embedding of a model description, or random features when none is given.
- **Query nodes are tasks**; their features are ``ContextEncoder`` (numeric state, no prompt
  text), min-max scaled on the training queries. Upstream uses a Longformer embedding of the
  query and refits the scaler on training plus test queries; here the scaler is fitted on the
  training queries alone, so a test state's answer does not depend on the other test states.
- **Performance** is upstream's cost-aware form (``benchmark_pipeline/run_pipeline.py``),
  ``alpha * norm(success) - beta * norm(cost)``, min-max over the training edges, with
  ``alpha = 1 - beta`` (upstream's example is 0.6 and 0.4) and ``beta`` swept (``BETAS``), one
  trained model and one policy per ``beta``. Success is 0 or 1 here, so for ``beta < alpha`` the
  label is the cheapest configuration that solved the task, whatever ``beta`` is; only
  ``beta > alpha`` can prefer a cheap failure. Ties in the argmax go to the earlier
  configuration, cheapest tier and effort first.
- **Splits**: query nodes are the ``train`` tasks (training edges) and the ``tune`` tasks
  (validation edges; upstream draws 20% of its training queries at random instead). The best
  epoch by validation value is kept.
- **Hyperparameters** are upstream's defaults (``configs/model_config_train/graphrouter.yaml``):
  hidden 64, AdamW lr 1e-3, weight decay 1e-4, 100 epochs, 4 masked samples per step, edge mask
  rate 0.3. CPU only, seeded.
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
from steproute.core.schema import Action
from steproute.eval.data import ConfigKey
from steproute.ml.features import FeatureBuilder, FeatureSpec

__all__ = [
    "BETAS",
    "HPARAMS",
    "UPSTREAM",
    "build_policies",
    "llm_features",
    "minmax",
    "performance",
]

UPSTREAM = Upstream(
    name="LLMRouter graphrouter",
    repository="https://github.com/ulab-uiuc/LLMRouter",
    commit="c65ee0122bc9e84f71d571fcdec4a948481b6e88",
    version="llmrouter-lib 0.4.0 (source at the commit; the package is not installed)",
    licence="MIT",
    paper="arXiv 2410.03834",
    use="adapted",
)

#: Upstream's training defaults (configs/model_config_train/graphrouter.yaml).
HPARAMS: dict[str, float | int] = {
    "hidden_dim": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "train_epoch": 100,
    "batch_size": 4,
    "train_mask_rate": 0.3,
}
#: Cost weights in ``alpha * norm(success) - beta * norm(cost)``, with ``alpha = 1 - beta``.
BETAS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)


def minmax(x: NDArray[np.float64], ref: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Scale columns of ``x`` to [0, 1] by the range of ``ref`` (default ``x``); constant -> 0."""
    r = x if ref is None else ref
    lo, hi = r.min(axis=0), r.max(axis=0)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    out: NDArray[np.float64] = (x - lo) / span
    return out


def performance(
    success: NDArray[np.float64], cost: NDArray[np.float64], beta: float, alpha: float = 1.0
) -> NDArray[np.float64]:
    """Upstream's cost-aware performance, min-max over all given edges."""

    def norm(v: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = (v - v.min()) / (v.max() - v.min() + 1e-12)
        return out

    return alpha * norm(success) - beta * norm(cost)


def llm_features(data: BaselineData) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """One row per arm: the tabular predictor's configuration block, min-max scaled."""
    spec = FeatureSpec()
    builder = FeatureBuilder(spec, data.catalog, data.probes())
    rows = np.stack([builder.config_vector(Action(m, e), None) for m, e in data.arms])
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
    return minmax(rows), spec.config_names


def _train_one(
    q_train: NDArray[np.float64],
    q_val: NDArray[np.float64],
    llm: NDArray[np.float64],
    perf_train: NDArray[np.float64],
    perf_val: NDArray[np.float64],
    seed: int,
) -> tuple[Any, Any, float]:
    """Train one GNN; return ``(predictor, llm node states, best validation value)``."""
    import torch

    from steproute.baselines._vendor.graphrouter_gnn import GNNPredictor, build_graph

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    n_train, n_llm = perf_train.shape
    n_val = len(perf_val)
    queries = np.vstack([q_train, q_val])
    perf = np.vstack([perf_train, perf_val])
    label = np.eye(n_llm)[np.argmax(perf, axis=1)]
    edges = (n_train + n_val) * n_llm
    train_mask = torch.zeros(edges)
    train_mask[: n_train * n_llm] = 1
    val_mask = torch.zeros(edges)
    val_mask[n_train * n_llm :] = 1

    def graph(edge_mask: Any) -> Any:
        return build_graph(
            queries, llm, perf.reshape(-1), label.reshape(-1), train_mask, val_mask, edge_mask
        )

    train_data, val_data = graph(train_mask), graph(val_mask)
    config = {
        "learning_rate": HPARAMS["learning_rate"],
        "weight_decay": HPARAMS["weight_decay"],
        "train_epoch": HPARAMS["train_epoch"],
        "batch_size": HPARAMS["batch_size"],
        "train_mask_rate": HPARAMS["train_mask_rate"],
        "llm_num": n_llm,
    }
    predictor = GNNPredictor(
        query_feature_dim=queries.shape[1],
        llm_feature_dim=llm.shape[1],
        hidden_features_size=int(HPARAMS["hidden_dim"]),
        in_edges_size=1,
        config=config,
    )
    best = predictor.train_validate(train_data, val_data, generator)
    # At routing time every training and validation edge is visible, as in upstream predict.
    states = predictor.llm_states(train_data)
    return predictor, states, best


def build_policies(
    data: BaselineData,
    encoder: ContextEncoder,
    *,
    seed: int = 0,
    betas: Sequence[float] = BETAS,
) -> tuple[list[ArmPolicy], dict[str, Any]]:
    """One trained GNN and policy per ``beta``. ``encoder`` must be fitted on training states."""
    train, val = data.tasks("train"), data.tasks("tune")
    if not train or not val:
        raise ValueError("graphrouter needs complete train and tune tasks")
    raw_train = encoder.transform_many([data.state(t) for t in train])
    raw_val = encoder.transform_many([data.state(t) for t in val])
    q_ref = raw_train
    q_train, q_val = minmax(raw_train, q_ref), minmax(raw_val, q_ref)
    llm, llm_names = llm_features(data)
    s_tr, c_tr = data.outcomes(train)
    s_va, c_va = data.outcomes(val)
    s_all, c_all = np.vstack([s_tr, s_va]), np.vstack([c_tr, c_va])
    arms = data.arms
    policies: list[ArmPolicy] = []
    validation: dict[str, float] = {}
    for beta in betas:
        perf = performance(s_all, c_all, beta, alpha=1.0 - beta)
        predictor, states, best = _train_one(
            q_train, q_val, llm, perf[: len(train)], perf[len(train) :], seed
        )
        validation[f"{beta:g}"] = best

        def choose(
            x: NDArray[np.float64], predictor: Any = predictor, states: Any = states
        ) -> ConfigKey:
            scores = predictor.score_queries(minmax(x[None, :], q_ref), states)
            return arms[int(scores[0].argmax())]

        policies.append(ArmPolicy(f"graphrouter beta={beta:g}", choose, encoder))
    provenance = {
        "upstream": UPSTREAM.to_dict(),
        "adapted_code": "steproute/baselines/_vendor/graphrouter_gnn.py (MIT notice kept)",
        "trained_on": "train tasks (training edges) and tune tasks (validation edges); "
        "retrained from scratch, no pretrained weights or text encoder",
        "hyperparameters": {**HPARAMS, "alpha": "1 - beta", "beta_sweep": list(betas)},
        "best_validation_value_by_beta": validation,
        "llm_node_features": list(llm_names),
        "arms": [arm_label(a) for a in arms],
        "context": encoder.describe(),
    }
    return policies, provenance
