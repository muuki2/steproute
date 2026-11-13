"""Benchmark ScoredPolicy.decide with the tabular predictor on the default catalog.

    uv run --extra ml python scripts/bench_tabular.py [--n 500] [--depth 2]

Trains a small model on the evaluator's synthetic grid (in a temporary directory; nothing is
kept), then times ``decide`` on the built-in catalog, with the model's features described only
by the profiles (no probe results, the cold-start case), two ways: ``predict`` once per
candidate prefix, and ``predict_many`` once per decision. Prints p50 / p95 / max in ms. The
trained model has no meaning for the default catalog's models; only its cost is measured.
Timings are for this machine; no test asserts on them.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from steproute.core import Action, CandidateSpec, ScoredPolicy, StepState
from steproute.core.catalog import load_default_catalog
from steproute.eval.data import load_grid
from steproute.eval.synthetic import catalog as synth_catalog
from steproute.eval.synthetic import default_world, write_example
from steproute.ml.model import TabularModel, TabularPredictor, TrainConfig, train


class PerCall:
    """The same predictor with ``predict_many`` hidden, so ScoredPolicy calls ``predict``."""

    def __init__(self, inner: TabularPredictor) -> None:
        self._inner = inner

    def predict(self, state: StepState, action_step: Action) -> float:
        return self._inner.predict(state, action_step)


def bench(fn: Callable[[], object], n: int) -> tuple[float, float, float]:
    for _ in range(min(20, n)):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    return statistics.median(times), times[int(0.95 * (len(times) - 1))], times[-1]


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=300, help="boosting rounds to time (300)")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        paths = write_example(Path(tmp), n_tasks=400, seed=0)
        grid = load_grid(paths["grid"])
        cfg = TrainConfig(seed=0, num_boost_round=args.rounds, early_stopping_rounds=0)
        result = train(grid, synth_catalog(default_world()), cfg)
        result.model.save(Path(tmp) / "model")
        model = TabularModel.load(Path(tmp) / "model")

    catalog = load_default_catalog(warn_unverified=False)
    spec = CandidateSpec(max_depth=args.depth)
    state = StepState(
        "proxy",
        "bench-session",
        3,
        "main",
        20_000,
        cached_prefix_tokens=18_000,
        current_model="claude-sonnet-5-5",
        current_effort="high",
        tools=["Bash", "Edit"],
    )
    predictor = TabularPredictor(model, catalog)
    batch = ScoredPolicy(predictor, catalog, tau=0.9, epsilon=0.05, spec=spec, seed=1)
    single = ScoredPolicy(PerCall(predictor), catalog, tau=0.9, epsilon=0.05, spec=spec, seed=1)  # type: ignore[arg-type]
    assert batch.score(state) == single.score(state), "batch and per-call paths differ"
    n_nodes = len(batch.score(state))
    print(f"python {platform.python_version()} on {platform.machine()}")
    print(
        f"default catalog, depth {args.depth}: {n_nodes} feasible candidates; "
        f"{model.booster.num_trees()} trees"
    )
    for label, policy in (("per-call predict", single), ("batch predict_many", batch)):
        n = max(5, args.n // 10) if label.startswith("per-call") else args.n
        p50, p95, worst = bench(lambda p=policy: p.decide(state), n)  # type: ignore[misc]
        print(f"decide, {label:20s} p50 {p50:8.3f} ms  p95 {p95:8.3f} ms  max {worst:8.3f} ms")


if __name__ == "__main__":
    main()
