# Actions

`steproute.core.actions` produces the candidate set every policy chooses from and the hard filter that removes candidates which must never be chosen. Standard library only.

```python
from steproute.core import enumerate_actions, filter_feasible, NoFeasibleAction

candidates = enumerate_actions(catalog, max_depth=2)  # compute once per catalog, reuse
allowed = filter_feasible(state, candidates, catalog)  # cheap linear pass, per decision
```

## Enumeration

An action is a step `(model, effort)`, optionally followed by a fallback step tried only if the attempt fails. `max_depth` is the total number of steps in a chain and must be at least 1.

- Only efforts a model lists in `efforts` are generated.
- **Strictly stronger.** Step B may follow step A only if B is the same model at a higher effort (by `effort_rank`), or a different model with a higher `tier`, at any effort that model supports. So chains cannot cycle.
- **Ties.** Two different models with the same `tier` never escalate to each other, in either direction. Each can still be escalated to by lower tiers and escalate to higher ones.
- **Holdout.** Models with `holdout = true` are left out unless `include_holdout=True` (they are reserved for the cold-start experiment).
- **Order.** Deterministic: singles first, then depth 2, and so on. Within a depth, chains are ordered lexicographically by step, where steps follow catalog tier order (ties in file order) and then effort rank.
- Accepts a `Catalog` or any iterable of `ModelProfile`. The result is an immutable tuple; compute it once and reuse it.

## Hard constraints

`filter_feasible(state, actions, catalog, *, allow=None, deny=None, include_holdout=False)` keeps an action only if every step of its chain passes all of:

1. `state.sensitive` implies the model is `local`.
2. `state.prompt_tokens` is at most the model's `context_window`.
3. If `state.tools` is non-empty, the model has `tools = true`.
4. The model is in `allow` (when given) and not in `deny`. A model in both is denied.
5. The model is not a holdout, unless `include_holdout`.

Constraints only remove; `allow` can never re-admit a model another constraint refuses, so a sensitive state never yields an action containing a cloud model, even as a fallback. Input order is preserved.

An action naming a model missing from the catalog, or an effort the model does not list, is a programming error and raises `ValueError`, as do `allow`/`deny` entries not in the catalog.

`NoFeasibleAction` is raised when nothing is left (also for empty input). `err.candidates` is the number offered and `err.removed` maps each constraint (`sensitive`, `context`, `tools`, `allow`, `deny`, `holdout`) to how many candidates violated it; one candidate can violate several, so the counts can sum to more than `candidates`. The message, e.g. `all 34 candidates removed: 34 sensitive`, never includes prompt text.

## Default catalog action counts

Holdout models excluded (9 models):

| max_depth | actions |
|---|---|
| 1 | 34 |
| 2 | 555 |
| 3 | 5399 |

Counts are totals up to that depth (depth 3 includes depths 1 and 2). They follow the editable tier assumptions in `docs/CATALOG.md`.
