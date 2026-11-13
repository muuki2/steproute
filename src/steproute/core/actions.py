"""Action enumeration and hard constraints. Standard library only. See ``docs/ACTIONS.md``.

``enumerate_actions`` builds the candidate set every policy chooses from.
``filter_feasible`` removes candidates that must never be chosen (privacy, context size,
tools, allow/deny lists). Policies may trust that anything it returns is allowed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from steproute.core.catalog import Catalog
from steproute.core.schema import Action, ModelProfile, StepState, effort_rank

__all__ = ["CONSTRAINTS", "NoFeasibleAction", "enumerate_actions", "filter_feasible"]

# Constraint names, as used in ``NoFeasibleAction.removed``.
CONSTRAINTS = ("sensitive", "context", "tools", "allow", "deny", "holdout")


class NoFeasibleAction(Exception):
    """The hard filter left no candidate.

    ``candidates`` is the number of candidates offered. ``removed`` maps each constraint name
    to the number of candidates that violated it; one candidate can violate several, so the
    counts may add up to more than ``candidates``. Contains no prompt text.
    """

    def __init__(self, candidates: int, removed: dict[str, int]):
        self.candidates = candidates
        self.removed = dict(removed)
        if candidates == 0:
            message = "no feasible action: no candidates were given"
        else:
            parts = ", ".join(f"{n} {name}" for name, n in self.removed.items() if n)
            message = f"no feasible action: all {candidates} candidates removed: {parts}"
        super().__init__(message)


def _as_catalog(profiles: Catalog | Iterable[ModelProfile]) -> Catalog:
    return profiles if isinstance(profiles, Catalog) else Catalog(tuple(profiles))


def enumerate_actions(
    profiles: Catalog | Iterable[ModelProfile],
    max_depth: int = 2,
    *,
    include_holdout: bool = False,
) -> tuple[Action, ...]:
    """All single steps and escalation cascades of up to ``max_depth`` steps.

    A step is a ``(model, effort)`` pair, using only efforts the model lists. Step B may follow
    step A only if it is strictly stronger: the same model at a higher effort, or a model with
    a higher ``tier`` at any of its efforts. Equal-tier models never escalate to each other, so
    chains cannot cycle. ``max_depth`` is the total number of steps in a chain (at least 1).

    Holdout models are skipped unless ``include_holdout``.

    Order: by depth (singles first); within a depth, lexicographic over steps, where steps are
    ordered by catalog tier order (ties in file order), then effort rank. Deterministic.
    The result is immutable; compute it once per catalog and reuse it.
    """
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError(f"max_depth must be an integer >= 1, got {max_depth!r}")
    catalog = _as_catalog(profiles)
    steps: list[tuple[int, int, Action]] = []  # (model position, tier, single action)
    for pos, p in enumerate(m for m in catalog if include_holdout or not m.holdout):
        for effort in sorted(p.efforts, key=effort_rank):
            steps.append((pos, p.tier, Action(p.id, effort)))

    def stronger(a: int, b: int) -> bool:
        pa, ta, sa = steps[a]
        pb, tb, sb = steps[b]
        if pa == pb:
            return effort_rank(sb.effort) > effort_rank(sa.effort)
        return tb > ta

    successors = [[j for j in range(len(steps)) if stronger(i, j)] for i in range(len(steps))]
    level: list[tuple[tuple[int, ...], Action]] = [((i,), s[2]) for i, s in enumerate(steps)]
    out: list[Action] = [a for _, a in level]
    for _ in range(max_depth - 1):
        nxt: list[tuple[tuple[int, ...], Action]] = []
        for idxs, _ in level:
            for j in successors[idxs[-1]]:
                chain = (*idxs, j)
                action = steps[chain[-1]][2]
                for k in reversed(chain[:-1]):
                    s = steps[k][2]
                    action = Action(s.model, s.effort, action)
                nxt.append((chain, action))
        if not nxt:
            break
        out.extend(a for _, a in nxt)
        level = nxt
    return tuple(out)


def filter_feasible(
    state: StepState,
    actions: Sequence[Action],
    catalog: Catalog | Iterable[ModelProfile],
    *,
    allow: Iterable[str] | None = None,
    deny: Iterable[str] | None = None,
    include_holdout: bool = False,
) -> tuple[Action, ...]:
    """Keep the actions whose every step passes all hard constraints, in input order.

    Per step: a sensitive state needs a local model; ``state.prompt_tokens`` must fit the
    model's ``context_window``; a state with tools needs ``tools = true``; the model must be in
    ``allow`` (when given) and not in ``deny`` (deny wins); holdout models are refused unless
    ``include_holdout``. ``allow`` can only narrow: it never re-admits a model another
    constraint refuses.

    Raises ``ValueError`` if an action names a model missing from the catalog or an effort the
    model does not list (a programming error), or if ``allow``/``deny`` name unknown models.
    Raises ``NoFeasibleAction`` if nothing is left, including for empty input.
    """
    cat = _as_catalog(catalog)
    by_id = {m.id: m for m in cat.models}
    allow_set = None if allow is None else frozenset(allow)
    deny_set = frozenset(deny) if deny is not None else frozenset()
    for label, ids in (("allow", allow_set or ()), ("deny", deny_set)):
        unknown = sorted(i for i in ids if i not in by_id)
        if unknown:
            raise ValueError(f"{label} list names models not in the catalog: {unknown}")

    # Per-model verdict, computed once: the constraints the model violates.
    tools_needed = bool(state.tools)
    verdict: dict[str, tuple[str, ...]] = {}
    for m in by_id.values():
        bad = []
        if state.sensitive and not m.local:
            bad.append("sensitive")
        if state.prompt_tokens > m.context_window:
            bad.append("context")
        if tools_needed and not m.tools:
            bad.append("tools")
        if allow_set is not None and m.id not in allow_set:
            bad.append("allow")
        if m.id in deny_set:
            bad.append("deny")
        if m.holdout and not include_holdout:
            bad.append("holdout")
        verdict[m.id] = tuple(bad)

    kept: list[Action] = []
    removed = dict.fromkeys(CONSTRAINTS, 0)
    total = 0
    for action in actions:
        total += 1
        violated: set[str] = set()
        node: Action | None = action
        while node is not None:
            profile = by_id.get(node.model)
            if profile is None:
                raise ValueError(f"action {action.label()!r} names unknown model {node.model!r}")
            if node.effort not in profile.efforts:
                raise ValueError(
                    f"action {action.label()!r}: model {node.model!r} does not list "
                    f"effort {node.effort!r} (has {list(profile.efforts)})"
                )
            violated.update(verdict[node.model])
            node = node.escalate_to
        if violated:
            for name in violated:
                removed[name] += 1
        else:
            kept.append(action)
    if not kept:
        raise NoFeasibleAction(total, removed)
    return tuple(kept)
