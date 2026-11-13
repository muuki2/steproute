"""Feature builder for the tabular predictor: ``StepState`` + ``ModelProfile`` + probe results.

One row describes one question, "does this ``(model, effort)`` step succeed in this state".
The row has three blocks:

- **state**: step kind, prompt size, cache fields, tools, step index, the task features the
  grid passes through (numeric ones, named in the ``FeatureSpec``), and what the earlier failed
  attempts of a cascade were like;
- **configuration**: the profile (tier, local or cloud, prices, context window, effort rank and
  output multiplier, provider family, size) and the **probe features**: how this configuration
  did on a small fixed set of probe tasks;
- **relation**: whether the step owns the cache, how it compares with the failed attempts.

**No model id is ever a feature.** A configuration is described only by what it is (its
profile) and what it did on the probes, so the trees cannot memorise "claude-x" and say nothing
about an unseen model. Ids are used only to look a profile or a probe result up. The provider
is a one-hot over a fixed family vocabulary with an ``other`` bucket. ``docs/MODEL-CARD-
TABULAR.md`` records the decisions and what was rejected.

Missing values are ``NaN`` (LightGBM routes them down a learned branch): a configuration with
no probe results, a task feature the state lacks, a profile without ``meta.size_gb``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from steproute.core.catalog import Catalog
from steproute.core.schema import Action, ModelProfile, StepState, effort_rank

__all__ = [
    "FEATURE_VERSION",
    "KINDS",
    "PROVIDERS",
    "ConfigKey",
    "FeatureBuilder",
    "FeatureSpec",
    "ProbeResult",
    "ProbeStats",
    "ProbeTable",
    "config_order",
    "stronger",
    "task_feature_keys",
]

#: Version of the feature definitions. Bump it when a feature's meaning or order changes, so a
#: saved model built on the old definition is refused rather than fed the wrong columns.
FEATURE_VERSION = 1

#: Known step kinds (``docs/SCHEMA.md``); anything else is the ``other`` bucket.
KINDS: tuple[str, ...] = ("main", "subagent", "tool_followup", "plan")
#: Provider families; anything else is the ``other`` bucket.
PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "google", "deepseek", "ollama")

ConfigKey = tuple[str, str]
_NAN = float("nan")


# --- probes ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """One configuration's result on one probe task. ``None`` means unknown."""

    success: bool | None
    cost_usd: float | None = None


@dataclass(frozen=True)
class ProbeStats:
    """What a configuration did on the first ``k`` probe tasks.

    ``n`` is how many of them it has a labelled result for; ``rate`` and ``log_cost`` are NaN
    when ``n`` is 0 (the cold-start representation: no results, not "all failed").
    """

    n: int
    rate: float
    log_cost: float


class ProbeTable:
    """Results of configurations on a fixed, ordered probe set.

    ``order`` lists the probe task ids; the first ``k`` of them are "k probes", so the sets for
    k = 5, 20, 50 are nested and the same for every configuration. ``results[config][task_id]``
    holds what that configuration did; a configuration may lack results for some probes.
    """

    def __init__(
        self,
        order: Sequence[str],
        results: Mapping[ConfigKey, Mapping[str, ProbeResult]] | None = None,
    ) -> None:
        if len(set(order)) != len(order):
            raise ValueError("probe task ids must be unique")
        self.order: tuple[str, ...] = tuple(order)
        self.results: dict[ConfigKey, dict[str, ProbeResult]] = {
            k: dict(v) for k, v in (results or {}).items()
        }
        self._cache: dict[tuple[ConfigKey, int], ProbeStats] = {}

    def __len__(self) -> int:
        return len(self.order)

    def stats(self, config: ConfigKey, k: int | None = None) -> ProbeStats:
        """Probe features of ``config`` over the first ``k`` probes (all when ``k`` is None)."""
        k = len(self.order) if k is None else max(0, min(int(k), len(self.order)))
        key = (config, k)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rows = self.results.get(config, {})
        n = wins = 0
        costs: list[float] = []
        for tid in self.order[:k]:
            r = rows.get(tid)
            if r is None or r.success is None:
                continue
            n += 1
            wins += bool(r.success)
            if r.cost_usd is not None:
                costs.append(math.log1p(1000.0 * r.cost_usd))
        out = ProbeStats(
            n=n,
            rate=wins / n if n else _NAN,
            log_cost=sum(costs) / len(costs) if costs else _NAN,
        )
        self._cache[key] = out
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "results": [
                {
                    "model": m,
                    "effort": e,
                    "runs": {t: [r.success, r.cost_usd] for t, r in sorted(rows.items())},
                }
                for (m, e), rows in sorted(self.results.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProbeTable:
        results: dict[ConfigKey, dict[str, ProbeResult]] = {}
        for entry in data["results"]:
            rows = {
                str(t): ProbeResult(
                    None if v[0] is None else bool(v[0]), None if v[1] is None else float(v[1])
                )
                for t, v in entry["runs"].items()
            }
            results[(str(entry["model"]), str(entry["effort"]))] = rows
        return cls([str(t) for t in data["order"]], results)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# --- the spec -------------------------------------------------------------------------------


def task_feature_keys(states: Iterable[StepState]) -> tuple[str, ...]:
    """Sorted names of the numeric (or boolean) entries seen in ``state.features``.

    These are the task features the grid passes through. Strings, lists and objects are
    ignored (a free-text or high-cardinality field would let trees memorise tasks).
    """
    keys: set[str] = set()
    for s in states:
        for k, v in s.features.items():
            if isinstance(v, bool | int | float) and math.isfinite(float(v)):
                keys.add(k)
    return tuple(sorted(keys))


@dataclass(frozen=True)
class FeatureSpec:
    """Everything that fixes the meaning and order of the columns. Saved with a model."""

    version: int = FEATURE_VERSION
    kinds: tuple[str, ...] = KINDS
    providers: tuple[str, ...] = PROVIDERS
    task_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def state_names(self) -> tuple[str, ...]:
        return (
            *(f"kind_{k}" for k in (*self.kinds, "other")),
            "log_prompt_tokens",
            "cached_frac",
            "log_cached_tokens",
            "n_tools",
            "step_idx",
            "sensitive",
            *(f"task_{k}" for k in self.task_keys),
            "n_failed",
            "failed_max_tier",
            "failed_max_effort_rank",
            "failed_max_probe_rate",
        )

    @property
    def config_names(self) -> tuple[str, ...]:
        return (
            "tier",
            "local",
            "log_price_in",
            "log_price_out",
            "log_context_window",
            "effort_rank",
            "effort_output_mult",
            "cache_read_mult",
            "supports_tools",
            "log_size_gb",
            *(f"provider_{p}" for p in (*self.providers, "other")),
            "probe_n",
            "probe_rate",
            "probe_log_cost",
        )

    @property
    def relation_names(self) -> tuple[str, ...]:
        return (
            "owns_cache",
            "same_effort_as_current",
            "model_already_failed",
            "tier_minus_failed_max",
            "probe_rate_minus_failed_max",
        )

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.state_names, *self.config_names, *self.relation_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kinds": list(self.kinds),
            "providers": list(self.providers),
            "task_keys": list(self.task_keys),
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeatureSpec:
        return cls(
            version=int(data["version"]),
            kinds=tuple(str(v) for v in data["kinds"]),
            providers=tuple(str(v) for v in data["providers"]),
            task_keys=tuple(str(v) for v in data["task_keys"]),
        )

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


# --- helpers --------------------------------------------------------------------------------


def stronger(a: ModelProfile, ea: str, b: ModelProfile, eb: str) -> bool:
    """Whether step ``b`` may follow failed step ``a`` in a cascade (the rule of
    ``enumerate_actions``): the same model at a higher effort, or a higher tier."""
    if a.id == b.id:
        return effort_rank(eb) > effort_rank(ea)
    return b.tier > a.tier


def config_order(configs: Iterable[ConfigKey], catalog: Catalog) -> list[ConfigKey]:
    """Configurations in an order that does not depend on their ids.

    Sorted by profile attributes (tier, effort, prices, context, size); the id is only the last
    tie-break. Training iterates in this order, so renaming every model leaves the training
    rows, and therefore the trained model, unchanged.
    """

    def key(c: ConfigKey) -> tuple[Any, ...]:
        p = catalog.get(c[0])
        return (
            p.tier,
            effort_rank(c[1]),
            p.price_out,
            p.price_in,
            p.context_window,
            p.local,
            _size_gb(p) if _size_gb(p) == _size_gb(p) else -1.0,
            p.provider,
            c[0],
        )

    return sorted(configs, key=key)


def _size_gb(p: ModelProfile) -> float:
    v = p.meta.get("size_gb")
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(float(v)):
        return _NAN
    return float(v)


def _num(v: Any) -> float:
    if isinstance(v, bool | int | float) and math.isfinite(float(v)):
        return float(v)
    return _NAN


# --- the builder ----------------------------------------------------------------------------


class FeatureBuilder:
    """Turns ``(state, step)`` pairs into feature rows.

    ``catalog`` supplies profiles (looked up by id, never used as a feature). ``probes`` holds
    the probe results; ``probe_k`` is how many probes to use (None: all of them, 0: none, which
    is the cold-start representation).
    """

    def __init__(
        self,
        spec: FeatureSpec,
        catalog: Catalog,
        probes: ProbeTable | None = None,
        probe_k: int | None = None,
    ) -> None:
        self.spec = spec
        self.catalog = catalog
        self.probes = probes if probes is not None else ProbeTable(())
        self.probe_k = probe_k
        self._names = spec.names
        self._ns = len(spec.state_names)
        self._nc = len(spec.config_names)
        self._nr = len(spec.relation_names)
        self._cfg_vec: dict[tuple[ConfigKey, int | None], NDArray[np.float64]] = {}
        self._profiles: dict[str, ModelProfile] = {m.id: m for m in catalog.models}
        self._kind_index = {k: i for i, k in enumerate(spec.kinds)}

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def _profile(self, model: str) -> ModelProfile:
        try:
            return self._profiles[model]
        except KeyError:
            raise KeyError(f"model {model!r} is not in the catalog") from None

    def config_vector(self, step: Action, k: int | None) -> NDArray[np.float64]:
        """The configuration block: profile and probe features of one step."""
        key = ((step.model, step.effort), k)
        hit = self._cfg_vec.get(key)
        if hit is not None:
            return hit
        p = self._profile(step.model)
        size = _size_gb(p)
        prov = [0.0] * (len(self.spec.providers) + 1)
        prov[
            self.spec.providers.index(p.provider)
            if p.provider in self.spec.providers
            else len(self.spec.providers)
        ] = 1.0
        ps = self.probes.stats((step.model, step.effort), k)
        vec = np.array(
            [
                float(p.tier),
                float(p.local),
                math.log1p(p.price_in),
                math.log1p(p.price_out),
                math.log(max(p.context_window, 1)),
                float(effort_rank(step.effort)),
                float(p.effort_output_mult.get(step.effort, 1.0)),
                float(p.cache_read_mult),
                float(p.tools),
                math.log1p(size) if size == size else _NAN,
                *prov,
                float(ps.n),
                ps.rate,
                ps.log_cost,
            ],
            dtype=np.float64,
        )
        self._cfg_vec[key] = vec
        return vec

    def _state_vector(
        self, state: StepState, k: int | None
    ) -> tuple[NDArray[np.float64], tuple[str, ...], float, float]:
        """The state block, the failed models, and the strongest failed tier and probe rate."""
        kinds = [0.0] * (len(self.spec.kinds) + 1)
        kinds[self._kind_index.get(state.kind, len(self.spec.kinds))] = 1.0
        prompt = max(state.prompt_tokens, 0)
        cached = max(state.cached_prefix_tokens, 0)
        failed_tier = failed_eff = failed_rate = _NAN
        failed_models: list[str] = []
        for a in state.failed_attempts:
            prof = self._profiles.get(a.model)
            failed_models.append(a.model)
            if prof is not None:
                failed_tier = (
                    prof.tier if failed_tier != failed_tier else max(failed_tier, prof.tier)
                )
            e = float(effort_rank(a.effort))
            failed_eff = e if failed_eff != failed_eff else max(failed_eff, e)
            r = self.probes.stats((a.model, a.effort), k).rate
            if r == r:
                failed_rate = r if failed_rate != failed_rate else max(failed_rate, r)
        vec = np.array(
            [
                *kinds,
                math.log1p(prompt),
                min(cached / prompt, 1.0) if prompt else 0.0,
                math.log1p(cached),
                float(len(state.tools)),
                float(state.step_idx),
                float(state.sensitive),
                *(_num(state.features.get(key)) for key in self.spec.task_keys),
                float(len(state.failed_attempts)),
                failed_tier,
                failed_eff,
                failed_rate,
            ],
            dtype=np.float64,
        )
        return vec, tuple(failed_models), failed_tier, failed_rate

    def matrix(
        self,
        pairs: Sequence[tuple[StepState, Action]],
        probe_k: Sequence[int | None] | None = None,
    ) -> NDArray[np.float64]:
        """Feature rows for ``pairs``, one per ``(state, step)``, columns in ``names`` order.

        ``probe_k`` gives the probe count per row (training augmentation); by default every
        row uses the builder's ``probe_k``.
        """
        n = len(pairs)
        out = np.empty((n, self._ns + self._nc + self._nr), dtype=np.float64)
        if n == 0:
            return out
        ks = [self.probe_k] * n if probe_k is None else list(probe_k)
        if len(ks) != n:
            raise ValueError("probe_k must have one entry per row")
        s_index: dict[tuple[int, int | None], int] = {}
        s_rows: list[NDArray[np.float64]] = []
        s_meta: list[tuple[str, str | None, tuple[str, ...], float, float]] = []
        c_index: dict[tuple[ConfigKey, int | None], int] = {}
        c_rows: list[NDArray[np.float64]] = []
        c_tier: list[float] = []
        c_rate: list[float] = []
        si = np.empty(n, dtype=np.intp)
        ci = np.empty(n, dtype=np.intp)
        rel = np.empty((n, self._nr), dtype=np.float64)
        for i, (state, step) in enumerate(pairs):
            k = ks[i]
            skey = (id(state), k)
            j = s_index.get(skey)
            if j is None:
                j = len(s_rows)
                s_index[skey] = j
                vec, failed, ftier, frate = self._state_vector(state, k)
                s_rows.append(vec)
                s_meta.append(
                    (state.current_model or "", state.current_effort, failed, ftier, frate)
                )
            ckey = ((step.model, step.effort), k)
            m = c_index.get(ckey)
            if m is None:
                m = len(c_rows)
                c_index[ckey] = m
                c_rows.append(self.config_vector(step, k))
                c_tier.append(self._profile(step.model).tier)
                c_rate.append(self.probes.stats((step.model, step.effort), k).rate)
            si[i] = j
            ci[i] = m
            cur_model, cur_effort, failed, ftier, frate = s_meta[j]
            rel[i, 0] = float(bool(cur_model) and cur_model == step.model)
            rel[i, 1] = float(cur_effort == step.effort) if cur_model == step.model else _NAN
            rel[i, 2] = float(step.model in failed) if failed else _NAN
            rel[i, 3] = c_tier[m] - ftier
            rel[i, 4] = c_rate[m] - frate
        out[:, : self._ns] = np.stack(s_rows)[si]
        out[:, self._ns : self._ns + self._nc] = np.stack(c_rows)[ci]
        out[:, self._ns + self._nc :] = rel
        return out
