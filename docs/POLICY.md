# Policy

`steproute.core.policy` holds the slot every learned component plugs into (`Predictor`), the object every collector calls (`Policy`), and the constrained-argmin policy built from them (`ScoredPolicy`). Standard library only, `mypy --strict` clean.

```python
from steproute.core import ScoredPolicy
from steproute.core.catalog import load_default_catalog

catalog = load_default_catalog()

policy = ScoredPolicy(predictor, catalog, tau=0.9, epsilon=0.05, seed=1)
decision = policy.decide(state)  # the caller logs it: store.log_decision(decision)
...
policy.report(outcome)  # the caller logs it too: store.log_outcome(outcome)
```

The policy never writes to the trace store. It returns a `Decision`; the caller logs it.

## Protocols

| Protocol | Methods | Notes |
|---|---|---|
| `Predictor` | `predict(state, action_step) -> float` | Success probability of one `(model, effort)` step (`escalate_to` is `None`). Must return a number in [0, 1]; anything else raises `CostModelError`. |
| `UpdatablePredictor` | `predict`, plus `update(decision, outcome) -> None` | `update` is optional: a `Predictor` without it is fine. `runtime_checkable`. |
| `Policy` | `decide(state) -> Decision`, `report(outcome) -> None` | Implemented by `ScoredPolicy` and `FixedPolicy`; issue 8's rule baseline implements it too. |

**The state passed to `predict`.** Each step is predicted in the state it actually runs in. For the second step of a cascade that is `state_after_failure(state, first_step, profile)`: the cache now belongs to the first model, and `failed_attempts` lists the failed first step. So `predict` returns the probability of success *given the earlier failures*, which is exactly the callable form of `p_success` that `expected_cost` takes. Overall success is then `1 - prod(1 - p_i)` with conditional `p_i`, which is exact. Multiplying independent per-step probabilities instead would overstate success because failures are correlated.

## Decision rule

`ScoredPolicy(predictor, catalog, tau, epsilon, *, spec=CandidateSpec(), seed=None, est_output_tokens=1000.0, inline_candidates_max=16, max_pending=10_000, pending_ttl_s=3600.0, clock=time.time, new_id=new_decision_id, name="scored")`

1. **Candidates.** `spec.enumerate(catalog)` (that is `enumerate_actions(catalog, max_depth, include_holdout=...)`) once at construction; per decision, `filter_feasible(state, candidates, catalog, allow, deny, include_holdout)`. `NoFeasibleAction` propagates to the caller (the HTTP service in issue 11 handles fail-open). Every branch, exploration included, chooses only from this filtered set, so a sensitive state never gets a cloud model in any step.
2. **Score.** For each feasible action: expected cost `c` (USD, the cache-aware cost model of `docs/COST-MODEL.md`) and overall success `p`.
3. **Greedy choice.** Among actions with `p >= tau`, the lowest `c`. If none reaches `tau`, the highest `p`.
4. **Exploration.** With probability `epsilon`, an action drawn uniformly from the feasible set replaces the greedy one.
5. **Decision.** `action`, `propensity` (below), `predicted_success = p`, `predicted_cost = c` of the chosen action, `candidate_set`, and the candidates themselves when the set is small.

**Tie-break (stable, documented, the same for every `tau`).**

- Qualified branch: lowest cost; on equal cost the higher predicted success; then enumeration order (earlier wins).
- Fallback branch (nothing reaches `tau`): highest predicted success; on equal success the lower cost; then enumeration order.

Because the greedy choice is unique and the order does not depend on `tau`, raising `tau` never lowers the predicted success of the greedy action (acceptance criterion 3). Proof sketch: with `tau1 < tau2`, the qualified set at `tau2` is a subset of the one at `tau1`. If the `tau1` choice still qualifies at `tau2` it is still the minimum of the smaller set under the same total order, so nothing changes; otherwise its success is below `tau2`, which the `tau2` choice reaches (or, if nothing qualifies at `tau2`, the `tau2` choice has the highest success of all). The hypothesis test `test_raising_tau_never_lowers_predicted_success` checks it over random catalogs, predictors (including coarse ones with many exact ties), depths and cache states.

Other read-only methods: `score(state)` (every feasible action with cost and success, enumeration order), `greedy(state)`, and `action_probabilities(state)`, the policy's full distribution `pi(a | state)`, which off-policy evaluation of this policy as a *target* needs (issue 20).

## Propensity

With a deterministic greedy action `g` and `n` feasible actions, the probability that the policy takes `a` is

```
P(a) = (1 - epsilon) * [a == g] + epsilon / n
```

The greedy action can also be drawn by the exploration branch; the formula counts both. The logged `propensity` is `P(chosen action)`, the true probability, whichever branch ran. The values over the feasible set sum to 1. `epsilon = 0` gives propensity 1.0; `epsilon = 1` gives `1/n` for everything.

The exploration index is `floor(b * n / 2^64)` for a uniform 64-bit `b`, so its deviation from exact uniformity is at most `n / 2^64`; the explore test `u < epsilon` uses a 53-bit uniform. Both are far below anything an estimator can see.

Acceptance criterion 2 is `test_empirical_frequencies_match_propensities`: 20,000 seeded decisions on the worked-example state below. For every feasible action, the empirical frequency is within 5 binomial standard deviations, `5 * sqrt(p (1 - p) / 20000)` (at most 0.018), of the logged propensity; every logged propensity equals `action_probabilities` exactly; and the cases `epsilon = 0` (all 20,000 are the greedy action) and `epsilon = 1` (each `1/6`) are checked as well. The test is seeded, so it cannot flake.

## Worked example

Reproduced by `test_example_scores_match_hand_arithmetic`, `test_example_choice` and `test_example_propensities` in `tests/test_policy.py`.

Catalog: two models with cache multipliers 1.0 (no read discount, no write premium), so who owns the cache never changes a price.

| Model | tier | efforts | price in / out (USD per M) | output multiplier |
|---|---|---|---|---|
| `small` | 1 | low, high | 1 / 5 | low 1.0, high 2.0 |
| `big` | 2 | medium | 5 / 25 | 1.0 |

State: 10,000 prompt tokens, nothing cached, `est_output_tokens = 1000`.

Step costs: `small low = 10,000 * 1e-6 + 1,000 * 5e-6 = 0.010 + 0.005 = 0.015`; `small high = 0.010 + 2,000 * 5e-6 = 0.020`; `big medium = 10,000 * 5e-6 + 1,000 * 25e-6 = 0.050 + 0.025 = 0.075`.

Stub predictor (probability given the failures listed):

| Step | fresh | after `small low` failed | after `small high` failed |
|---|---|---|---|
| `small low` | 0.60 | | |
| `small high` | 0.75 | 0.50 | |
| `big medium` | 0.90 | 0.80 | 0.70 |

The six candidates at depth 2, cost `c1 + (1 - p1) c2`, success `1 - (1 - p1)(1 - p2)`:

| # | Action | Expected cost | Success |
|---|---|---|---|
| 0 | small low | 0.015 | 0.60 |
| 1 | small high | 0.020 | 0.75 |
| 2 | big medium | 0.075 | 0.90 |
| 3 | small low -> small high | 0.015 + 0.40 * 0.020 = 0.023 | 1 - 0.40 * 0.50 = 0.80 |
| 4 | small low -> big medium | 0.015 + 0.40 * 0.075 = 0.045 | 1 - 0.40 * 0.20 = 0.92 |
| 5 | small high -> big medium | 0.020 + 0.25 * 0.075 = 0.03875 | 1 - 0.25 * 0.30 = 0.925 |

Choices:

- `tau = 0.91`: qualified #4 and #5. **A cascade wins:** #5 `small high -> big medium` at 0.03875, cheaper than #4 (0.045) and than `big medium` alone (0.075, which also misses `tau`). With `tau = 0.85` `big medium` qualifies but #5 is still cheapest.
- `tau = 0.78`: #3 `small low -> small high` (0.023) qualifies and wins.
- `tau = 0.70`: #1 `small high` (0.020). `tau = 0.50`: #0 `small low`.
- `tau = 0.95` (or 1.0): **nothing reaches `tau`**, so the highest success wins: #5 at 0.925.

Propensity at `tau = 0.91`, `epsilon = 0.1`, `n = 6`: #5 gets `0.9 + 0.1 / 6 = 0.91667`; each of the other five gets `0.1 / 6 = 0.01667`; total 1.

## Design decisions

### 1. Latency: a shared-work scoring path

Calling `expected_cost` once per candidate on the default catalog at depth 2 (555 feasible candidates) takes about 7.5 ms (p50) on an M4 Pro, already over the budget with a trivial predictor. It rebuilds the post-failure state and re-prices the first step for every cascade, and validates every number again.

**Chosen.** `_Scorer`, built once per (catalog, spec): the candidate list is enumerated once and turned into a prefix tree in which every distinct chain prefix is one node (555 nodes at depth 2). Per decision:

- the feasibility filter result is cached under the only three things `filter_feasible` reads from the state (`sensitive`, whether `tools` is non-empty, and which context windows `prompt_tokens` exceeds). The cache cannot grow past `4 * (distinct context windows + 1)` entries. The first state with a given key still goes through `filter_feasible` itself; `test_feasibility_cache_equals_filter_feasible` checks the cached result against a direct call;
- `state_after_failure` runs once per first step that has children (34), not once per cascade;
- `predict` runs once per prefix node;
- each step is priced once per distinct cache situation (does the step own the cache, and how much is cached), with the real `planned_usage` and `price_usage`, so pricing logic is not duplicated;
- the per-candidate fold is the same arithmetic in the same order as `expected_cost`, so the results agree exactly, not approximately. `test_fast_path_equals_expected_cost_for_every_candidate` asserts `==` on cost and success for every candidate, at depths 1, 2 and 3, on five states (cold, warm cache, prompt below the cache minimum, a prompt in a higher price tier beyond the local context caps, a local current model), with a predictor that depends on the failure history.

**Measured** with `scripts/bench_policy.py` (3,000 timed `decide` calls after 50 warm-up calls, `time.perf_counter`, a constant predictor, default catalog at depth 2, a warm-cache 20,000-token state with tools, `epsilon = 0.05`), Apple M4 Pro:

| | p50 | p95 |
|---|---|---|
| `ScoredPolicy.decide`, Python 3.12 | 0.60 ms | 0.66 ms |
| `ScoredPolicy.decide`, Python 3.13 | 0.57 ms | 0.64 ms |
| naive `filter_feasible` + `expected_cost` per candidate, Python 3.12 | 8.6 ms | 9.3 ms |

No unit test asserts on timing.

**Rejected.** Reimplementing the cost arithmetic inline (faster still, but a second copy of the pricing rules that could drift from `cost.py`); pruning candidates before scoring (needs the predictor's values, so not safe in general); caching scores across decisions (the predictor and the state change every call).

### 2. Size of the logged decision: fingerprint plus rebuild

With 555 candidates one decision serialises to **65,944 bytes** as a store line. Most of it is the candidate list.

**Chosen.** A new defaulted field `Decision.candidate_set: CandidateSet | null` (schema stays backward compatible; `docs/SCHEMA.md`). It records `fingerprint`, `count`, `max_depth`, `include_holdout`, `allow` and `deny`. The explicit `candidates` list is still written when the set has at most `inline_candidates_max` (default 16) entries, and left empty otherwise; `inline_candidates_max=None` always writes it. `rebuild_candidates(decision, catalog)` rebuilds the list from the logged state and a catalog and verifies the fingerprint, the count, and the explicit list when present, raising `CandidateSetMismatch` otherwise. The same decision is then **824 bytes** (most of that is the state). The parameters are stored in clear, so rebuilding needs only the decision and a catalog, not the policy's configuration.

The fingerprint is `v1:` plus the first 32 hex characters of SHA-256 over a canonical JSON of: per model in file order, `id`, `tier`, `efforts`, `local`, `context_window`, `tools`, `holdout` (everything that decides which actions exist, their order, and which survive the filter); the spec parameters; and `ENUMERATION_VERSION`, which must be bumped whenever `enumerate_actions` or `filter_feasible` change their output, so old fingerprints stop verifying rather than silently matching a different list.

**Rejected.** Hashing the whole catalog including prices and `meta`: a price correction would make every earlier decision unverifiable although its candidate set is unchanged. (Re-scoring an old decision needs the old prices anyway, and those live in the catalog's git history.) Storing candidate indices or a bitmask: not readable without the catalog either, and still hundreds of bytes. Always omitting the list: small sets are cheap and self-describing, so they stay explicit.

Caveat: the rebuild uses the *logged* state. `TraceStore.log_decision` runs `redact_state`, which can set `sensitive` when it detects sensitive text. If a collector calls `decide` before `mark_sensitive`, the logged state differs from the one the policy saw, and verification fails (by count) instead of returning a wrong list. Collectors should call `mark_sensitive` before `decide`, which is also what privacy needs.

### 3. `est_output_tokens`: an injectable estimate with a placeholder default

**Chosen.** `est_output_tokens` is a number or a callable `(state) -> float`, applied to every step before the per-effort multiplier. A state can override it with `state.features["est_output_tokens"]`, which wins over the estimator and is logged with the state, so the decision stays reproducible. The default is `DEFAULT_EST_OUTPUT_TOKENS = 1000`.

**The default is a placeholder, not a measurement.** Nobody has measured output lengths for these harnesses. Replace it with an estimator fitted on the pilot replay (issue 18), per step kind at least. A bad value raises `CostModelError`.

**Rejected.** A per-(state, step) estimator: output length per model is not measured either, and a per-decision constant is what lets each step be priced once. It can be added later without changing the protocols. Leaving the estimate to every caller: every collector would then invent its own number.

### 4. `report(outcome)` and `update(decision, outcome)`: a bounded map of recent decisions

**Chosen.** If the predictor has `update`, `decide` keeps the decision in an insertion-ordered map keyed by `decision_id`, bounded by `max_pending` (default 10,000) and by age `pending_ttl_s` (default 3,600 s, against the injected clock). Inserting evicts entries older than the TTL and then the oldest beyond the count bound. `report(outcome)`:

- unknown id (never seen, evicted, or already consumed): counted (`stats().unknown_reports`) and ignored, never raised;
- expired (older than the TTL at report time): removed, counted (`expired_reports`), ignored;
- `success is None`: counted (`unlabelled_reports`) and ignored, and the decision stays pending so a later labelled outcome can still be used;
- otherwise the decision is removed and `update(decision, outcome)` is called once, outside the lock. Exceptions from `update` propagate.

So a predictor sees at most one outcome per decision: the first labelled one. If the predictor has no `update`, nothing is stored and `report` is a no-op.

Memory bound: a compact pending decision takes about 0.7 KB of its own (measured with `tracemalloc`, default catalog at depth 2) and keeps its state alive, typically another 1 KB or so (more with many tools or features). The default bound is therefore roughly 10 to 20 MB, and less when the TTL expires entries first.

**Rejected.** Changing `report` to take the decision too (breaks the `Policy` protocol and pushes the join onto every collector); reading decisions back from the trace store (the policy must not depend on the store, and it is slow); an unbounded map (a collector that never reports would leak memory).

### 5. Randomness and threads: counter-based draws

**Chosen.** Decision `k` (counted from 0 in the order calls enter `decide`, taken from a counter under a lock) draws its randomness from `blake2b(f"{seed}:{k}")` only: a 53-bit uniform for "explore?" and a 64-bit integer for "which action?". There is no shared generator state to corrupt.

"Deterministic under a seed" means: single-threaded, the same seed and the same sequence of states (and the same injected clock and id factory) give byte-identical decisions; different seeds give different sequences. Concurrently, each decision is still a pure function of its state, its `k` and the seed, so the multiset of draws is the same as sequentially; only which state receives which `k` depends on scheduling (tested). `seed=None` draws a seed from `os.urandom`; it is exposed as `policy.seed`.

**Rejected.** A shared `random.Random` with a lock (same determinism, but decision `k`'s draw would depend on how many numbers earlier decisions consumed); thread-local generators (results depend on which thread serves a request); deriving the draw from the `decision_id` (ids are random `uuid4` by default, so the sequence would not be reproducible, and making ids deterministic from the seed would collide across runs that reuse a seed).

### 6. Time and ids are injectable

`clock` (default `time.time`) stamps `Decision.ts` and drives the TTL; `new_id` (default `new_decision_id`) makes decision ids. Tests inject both, so decisions are byte-for-byte reproducible.

## FixedPolicy

`FixedPolicy(action, catalog, *, allow=None, deny=None, include_holdout=False, clock, new_id, name="fixed")` always returns `action` with propensity 1.0 and `candidates = [action]`, after `filter_feasible`: on a state that forbids the action (for example a cloud model on a sensitive state) `decide` raises `NoFeasibleAction`, and the caller must handle it. An action that names an unknown model or effort, or that the allow, deny or holdout settings always refuse, raises `ValueError` at construction. `report` does nothing. Used by the replay runner (issue 18) and as the service's fail-open default (issue 11): give that default a local model, or the fail-open path itself fails on sensitive states.

## Limitations

- **Deterministic predictors only, for exact propensities.** The propensity formula assumes the greedy choice is a deterministic function of the state. A predictor that samples (Thompson sampling, issue 28) makes the greedy choice random, and the logged propensity is then wrong. Such a policy needs its own propensity estimate. Within one decision `predict` is called once per distinct prefix, so a sampling predictor at least gets one draw per step.
- **Predictor call count.** `predict` is called once per chain prefix: 555 times per decision at depth 2 on the default catalog. Cheap for table or rule predictors, too slow for a model with per-call overhead (LightGBM, a GNN). Those need a batched predict, or distillation to a table (issue 27); the protocol can gain an optional batch method without breaking anything.
- **Latency is not part of the rule.** The cost model can estimate latency, but the policy minimises USD only. No default profile carries a latency estimate yet.
- **One output estimate per decision**, the same for every model, adjusted only by the effort multiplier; the default is a placeholder.
- **`report` memory is per process.** Pending decisions are lost on restart; outcomes reported afterwards are counted as unknown.
- **Exploration is uniform** over the feasible set, which at depth 2 is mostly cascades with expensive top-tier fallbacks. Fine for coverage and for IPS, wasteful in money at high `epsilon`.
- The state object is stored in the decision by reference; a caller that mutates it after `decide` changes the logged record.
