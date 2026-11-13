"""Model catalog: ``ModelProfile``s loaded from a TOML file and validated.

A catalog is hand-written config, so unlike trace records it is strict: unknown keys, bad
efforts, negative prices and the like raise ``CatalogError`` naming the model and the field.
Standard library only. See ``docs/CATALOG.md``.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from steproute.core.schema import EFFORTS, ModelProfile, PriceTier

__all__ = [
    "Catalog",
    "CatalogError",
    "UnverifiedCatalogWarning",
    "load_catalog",
    "load_default_catalog",
]

_PROFILE_KEYS = {f.name for f in dataclasses.fields(ModelProfile)}
_TIER_KEYS = {f.name for f in dataclasses.fields(PriceTier)}
# A latency estimate is all three of these or none; latency_switch_ms is optional on its own.
_LATENCY_REQUIRED = ("latency_base_ms", "latency_ms_per_input_token", "latency_ms_per_output_token")
_LATENCY_FIELDS = (*_LATENCY_REQUIRED, "latency_switch_ms")


class CatalogError(ValueError):
    """The catalog is invalid. ``model`` and ``field`` say where (either may be ``None``)."""

    def __init__(self, message: str, *, model: str | None = None, field: str | None = None):
        where = "".join(
            [f"model {model!r}" if model else "", f", field {field!r}" if model and field else ""]
        )
        if not model and field:
            where = f"field {field!r}"
        super().__init__(f"{where}: {message}" if where else message)
        self.model = model
        self.field = field


class UnverifiedCatalogWarning(UserWarning):
    """The catalog contains models with ``verified = false``; their numbers are unchecked."""


def _fail(model: str, field: str, message: str) -> CatalogError:
    return CatalogError(message, model=model, field=field)


def _validate(p: ModelProfile) -> None:
    mid = p.id
    if not p.id:
        raise _fail("", "id", "id must not be empty")
    if not p.provider:
        raise _fail(mid, "provider", "provider must not be empty")
    for name in ("price_in", "price_out", "cache_write_mult", "cache_read_mult"):
        if getattr(p, name) < 0:
            raise _fail(mid, name, f"must not be negative, got {getattr(p, name)}")
    if p.context_window <= 0:
        raise _fail(mid, "context_window", f"must be positive, got {p.context_window}")
    if p.tier < 0:
        raise _fail(mid, "tier", f"must not be negative, got {p.tier}")
    if not p.efforts:
        raise _fail(mid, "efforts", "must list at least one effort")
    for e in p.efforts:
        if e not in EFFORTS:
            raise _fail(mid, "efforts", f"unknown effort {e!r}; expected one of {EFFORTS}")
    if len(set(p.efforts)) != len(p.efforts):
        raise _fail(mid, "efforts", "duplicate effort")
    for name in ("effort_output_mult", "effort_map"):
        mapping: dict[str, Any] = getattr(p, name)
        for k, v in mapping.items():
            if k not in p.efforts:
                raise _fail(mid, name, f"key {k!r} is not in efforts {list(p.efforts)}")
            if name == "effort_output_mult" and v < 0:
                raise _fail(mid, name, f"{k!r} must not be negative, got {v}")
            if name == "effort_map" and not v:
                raise _fail(mid, name, f"{k!r} must map to a non-empty string")
    if p.cache_min_tokens < 0:
        raise _fail(mid, "cache_min_tokens", f"must not be negative, got {p.cache_min_tokens}")
    latency = {name: getattr(p, name) for name in _LATENCY_FIELDS}
    for name, v in latency.items():
        if v is not None and v < 0:
            raise _fail(mid, name, f"must not be negative, got {v}")
    given = [name for name in _LATENCY_REQUIRED if latency[name] is not None]
    if given and len(given) != len(_LATENCY_REQUIRED):
        missing = [n for n in _LATENCY_REQUIRED if latency[n] is None]
        raise _fail(mid, missing[0], f"latency estimate is partial; set all of {_LATENCY_REQUIRED}")
    previous = 0
    for i, t in enumerate(p.price_tiers):
        loc = f"price_tiers[{i}]"
        if t.above_tokens <= 0:
            raise _fail(mid, loc, f"above_tokens must be positive, got {t.above_tokens}")
        if i and t.above_tokens <= previous:
            raise _fail(mid, loc, "above_tokens must be strictly increasing")
        if t.price_in < 0 or t.price_out < 0:
            raise _fail(mid, loc, "prices must not be negative")
        previous = t.above_tokens


def _profile_from_table(table: Any, index: int) -> ModelProfile:
    if not isinstance(table, dict):
        raise CatalogError(f"model entry {index} must be a table")
    mid = str(table.get("id", f"#{index}"))
    unknown = sorted(set(table) - _PROFILE_KEYS)
    if unknown:
        raise _fail(mid, unknown[0], f"unknown key(s) {unknown}")
    tiers = table.get("price_tiers", [])
    if not isinstance(tiers, list):
        raise _fail(mid, "price_tiers", "must be an array of tables")
    for i, t in enumerate(tiers):
        bad = sorted(set(t) - _TIER_KEYS) if isinstance(t, dict) else []
        if bad:
            raise _fail(mid, f"price_tiers[{i}].{bad[0]}", f"unknown key(s) {bad}")
    try:
        profile = ModelProfile.from_dict(table)
    except (TypeError, ValueError) as exc:
        text = str(exc)
        hit = re.search(r"ModelProfile\.(\w+)|PriceTier\.(\w+)|field\(s\) \['(\w+)'", text)
        field = next((g for g in hit.groups() if g), None) if hit else None
        raise CatalogError(text, model=mid, field=field) from None
    _validate(profile)
    return profile


@dataclass
class Catalog:
    """Profiles in file order, unique by id."""

    models: tuple[ModelProfile, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for m in self.models:
            if m.id in seen:
                raise CatalogError("duplicate model id", model=m.id, field="id")
            seen.add(m.id)

    def __len__(self) -> int:
        return len(self.models)

    def __contains__(self, model_id: object) -> bool:
        return any(m.id == model_id for m in self.models)

    def __iter__(self) -> Iterator[ModelProfile]:
        """Profiles in tier order, weakest first; ties keep file order."""
        return iter(sorted(self.models, key=lambda m: m.tier))

    def get(self, model_id: str) -> ModelProfile:
        """The profile with this id. Raises ``KeyError`` if absent."""
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def pool(self) -> tuple[ModelProfile, ...]:
        """Non-holdout profiles in tier order: what training and normal routing may use."""
        return tuple(m for m in self if not m.holdout)

    def local_models(self) -> tuple[ModelProfile, ...]:
        """Profiles with ``local = true``, in tier order (holdout models included)."""
        return tuple(m for m in self if m.local)

    def unverified(self) -> tuple[str, ...]:
        """Ids of models with ``verified = false``."""
        return tuple(m.id for m in self.models if not m.verified)

    def to_dict(self) -> dict[str, Any]:
        """Same shape as the TOML file: ``{"model": [profile dicts]}``."""
        return {"model": [m.to_dict() for m in self.models]}

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, warn_unverified: bool = True) -> Catalog:
        """Build and validate a catalog. Emits one ``UnverifiedCatalogWarning`` if needed."""
        if not isinstance(data, dict):
            raise CatalogError("catalog must be a table")
        unknown = sorted(set(data) - {"model"})
        if unknown:
            raise CatalogError(f"unknown top-level key(s) {unknown}; only [[model]] is allowed")
        tables = data.get("model", [])
        if not isinstance(tables, list):
            raise CatalogError("'model' must be an array of tables ([[model]])")
        catalog = cls(tuple(_profile_from_table(t, i) for i, t in enumerate(tables)))
        bad = catalog.unverified()
        if bad and warn_unverified:
            warnings.warn(
                f"catalog has unverified models (verified = false): {', '.join(bad)}",
                UnverifiedCatalogWarning,
                stacklevel=2,
            )
        return catalog


def load_catalog(path: str | Path, *, warn_unverified: bool = True) -> Catalog:
    """Load and validate a TOML catalog file."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise CatalogError(f"{path}: invalid TOML: {exc}") from None
    return Catalog.from_dict(data, warn_unverified=warn_unverified)


def load_default_catalog(*, warn_unverified: bool = True) -> Catalog:
    """Load the catalog shipped inside the package (``steproute/catalogs/default.toml``)."""
    text = resources.files("steproute.catalogs").joinpath("default.toml").read_text("utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover
        raise CatalogError(f"default catalog: invalid TOML: {exc}") from None
    return Catalog.from_dict(data, warn_unverified=warn_unverified)
