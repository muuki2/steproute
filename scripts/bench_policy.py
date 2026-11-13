"""Benchmark ScoredPolicy.decide on the default catalog, and the size of a logged decision.

    uv run python scripts/bench_policy.py [--depth 2] [--n 2000] [--text-chars 20000]

Prints p50 / p95 / max latency of ``decide`` with a trivial predictor, once with no task text
and once with ``--text-chars`` characters of ordinary (non-secret) code and prose in
``task_text`` plus five file paths, so sensitive detection scans everything; the same for the naive
path (``filter_feasible`` plus one ``expected_cost`` per candidate) for comparison, and the
serialised size of one decision with and without the explicit candidate list. Timings are for
this machine; no test asserts on them.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import time
from collections.abc import Callable

from steproute.core import (
    Action,
    CandidateSpec,
    ScoredPolicy,
    StepState,
    expected_cost,
    filter_feasible,
    rebuild_candidates,
)
from steproute.core.catalog import load_default_catalog
from steproute.core.store import encode_line


class Trivial:
    def predict(self, state: StepState, action_step: Action) -> float:
        return 0.7


def bench(fn: Callable[[], object], n: int) -> tuple[float, float, float]:
    for _ in range(min(50, n)):
        fn()  # warm up
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    return statistics.median(times), times[int(0.95 * (len(times) - 1))], times[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--text-chars", type=int, default=20_000)
    args = ap.parse_args()

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
    policy = ScoredPolicy(Trivial(), catalog, tau=0.9, epsilon=0.05, spec=spec, seed=1)
    n_feasible = len(policy.score(state))
    print(f"python {platform.python_version()} on {platform.machine()}")
    print(f"default catalog, depth {args.depth}: {n_feasible} feasible candidates")

    p50, p95, worst = bench(lambda: policy.decide(state), args.n)
    print(f"ScoredPolicy.decide, no text  p50 {p50:.3f} ms  p95 {p95:.3f} ms  max {worst:.3f} ms")

    chunk = (
        "def handler(request, context):\n"
        "    # Look up the user record and return the rendered page.\n"
        "    user = db.users.get(request.args['id'])\n"
        "    return render('profile.html', user=user, items=load_items(user))\n"
        "The test in tests/test_views.py fails with a KeyError on line 42; please fix it.\n"
    )
    text = (chunk * (args.text_chars // len(chunk) + 1))[: args.text_chars]
    paths = ["src/app/views.py", "src/app/db.py", "tests/test_views.py", "README.md", "setup.cfg"]
    with_text = dataclasses.replace(state, task_text=text, features={"paths": paths})
    assert not policy.decide(with_text).state.sensitive  # the scan finds nothing: worst case
    p50, p95, worst = bench(lambda: policy.decide(with_text), args.n)
    print(
        f"ScoredPolicy.decide, {len(text)}-char text + {len(paths)} paths  "
        f"p50 {p50:.3f} ms  p95 {p95:.3f} ms  max {worst:.3f} ms"
    )

    actions = spec.enumerate(catalog)
    trivial = Trivial()

    def naive() -> object:
        feas = filter_feasible(state, actions, catalog)
        return min(
            (expected_cost(state, a, catalog, 1000.0, trivial.predict) for a in feas),
            key=lambda b: b.total_usd,
        )

    p50, p95, worst = bench(naive, max(50, args.n // 10))
    print(f"naive expected_cost  p50 {p50:.3f} ms  p95 {p95:.3f} ms  max {worst:.3f} ms")

    compact = policy.decide(state)
    full = dataclasses.replace(compact, candidates=list(rebuild_candidates(compact, catalog)))
    for label, d in (("explicit list", full), ("compact", compact)):
        line = encode_line("decision", d.to_dict())
        print(f"decision record, {label}: {len(line)} bytes")
    print("candidate_set:", json.dumps(compact.to_dict()["candidate_set"]))


if __name__ == "__main__":
    main()
