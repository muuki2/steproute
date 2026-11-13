"""Model catalog: loading, validation, tiered prices, round-trips, the unverified warning."""

import copy
import os
import warnings
from pathlib import Path
from typing import Any

import pytest

from steproute.core import EFFORTS, ModelProfile, PriceTier
from steproute.core.catalog import (
    Catalog,
    CatalogError,
    UnverifiedCatalogWarning,
    load_catalog,
    load_default_catalog,
)

MINIMAL = """
[[model]]
id = "a"
provider = "p"
price_in = 1.0
price_out = 2.0
context_window = 1000
tier = 2
efforts = ["low", "high"]
effort_output_mult = { low = 1.0, high = 1.0 }
effort_map = { low = "lo" }
verified = true

[[model.price_tiers]]
above_tokens = 100
price_in = 3.0
price_out = 4.0

[[model]]
id = "b"
provider = "p"
price_in = 0
price_out = 0
context_window = 1000
tier = 1
local = true
holdout = true
verified = true
"""


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "c.toml"
    p.write_text(text, encoding="utf-8")
    return p


def raw(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "m",
        "provider": "p",
        "price_in": 1.0,
        "price_out": 1.0,
        "context_window": 10,
        "verified": True,
    }
    base.update(over)
    return base


def build(*models: dict[str, Any]) -> Catalog:
    return Catalog.from_dict({"model": list(models)})


# --- loading and round trips ------------------------------------------------------------------


def test_minimal_loads_and_orders_by_tier(tmp_path: Path) -> None:
    cat = load_catalog(write(tmp_path, MINIMAL))
    assert [m.id for m in cat] == ["b", "a"]
    assert cat.get("a").price_tiers == (PriceTier(100, 3.0, 4.0),)
    assert [m.id for m in cat.pool()] == ["a"]
    assert [m.id for m in cat.local_models()] == ["b"]
    assert "a" in cat
    assert len(cat) == 2
    with pytest.raises(KeyError):
        cat.get("zzz")


def test_roundtrip_dict_and_toml(tmp_path: Path) -> None:
    cat = load_catalog(write(tmp_path, MINIMAL))
    assert Catalog.from_dict(cat.to_dict()) == cat
    again = Catalog.from_dict(copy.deepcopy(cat.to_dict()))
    assert again.to_dict() == cat.to_dict()


def test_default_catalog_loads_validates_roundtrips() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the shipped catalog is fully verified: no warning
        cat = load_default_catalog()
    assert Catalog.from_dict(cat.to_dict()) == cat
    ids = {m.id for m in cat}
    assert ids == {
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
        "claude-opus-5-5",
        "claude-fable-5-2",
        "deepseek-flash",
        "deepseek-v4-pro",
        "qwen3.5:9b",
        "gpt-oss:20b",
        "qwen3-coder:30b",
        "devstral:24b",
    }
    assert [m.id for m in cat if m.holdout] == ["devstral:24b"]
    assert "devstral:24b" not in {m.id for m in cat.pool()}
    assert all(m.context_window == 32768 for m in cat.local_models())
    assert all(m.price_in == 0 for m in cat.local_models())
    for m in cat:
        assert set(m.efforts) <= set(EFFORTS)
        assert m.verified and m.tools


def test_default_catalog_values_from_notes() -> None:
    cat = load_default_catalog()
    haiku = cat.get("claude-haiku-5-5")
    assert (haiku.price_in, haiku.price_out, haiku.cache_read_mult) == (0.10, 0.50, 0.1)
    assert cat.get("claude-fable-5-2").cache_read_mult == 0.025
    flash, pro = cat.get("deepseek-flash"), cat.get("deepseek-v4-pro")
    assert flash.cache_read_mult == pytest.approx(0.006 / 0.30)
    assert pro.cache_read_mult == pytest.approx(0.044 / 1.32)
    assert flash.cache_write_mult == 1.0
    assert flash.efforts == ("none", "low", "high", "max")
    assert flash.meta["price_in_offpeak"] == 0.15
    assert cat.get("qwen3-coder:30b").efforts == ("medium",)
    assert cat.get("qwen3-coder:30b").effort_map == {}


# --- tiered prices ----------------------------------------------------------------------------


def test_tier_boundary_at_exactly_100k_prompt_tokens() -> None:
    haiku = load_default_catalog().get("claude-haiku-5-5")
    assert haiku.prices_for(0) == (0.10, 0.50)
    assert haiku.prices_for(99_999) == (0.10, 0.50)
    assert haiku.prices_for(100_000) == (0.10, 0.50)  # exactly 100K is still the lower tier
    assert haiku.prices_for(100_001) == (0.50, 2.50)
    assert haiku.prices_for(900_000) == (0.50, 2.50)


def test_untiered_model_has_one_price() -> None:
    sonnet = load_default_catalog().get("claude-sonnet-5-5")
    assert sonnet.prices_for(1) == sonnet.prices_for(900_000) == (2.0, 10.0)


def test_multiple_tiers() -> None:
    p = ModelProfile(
        "m",
        "p",
        1.0,
        1.0,
        10,
        price_tiers=(PriceTier(10, 2.0, 2.0), PriceTier(20, 3.0, 3.0)),
    )
    assert [p.prices_for(n)[0] for n in (10, 11, 20, 21)] == [1.0, 2.0, 2.0, 3.0]


# --- the unverified warning -------------------------------------------------------------------


def test_unverified_catalog_warns_exactly_once_listing_ids() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        build(raw(id="x", verified=False), raw(id="y"), raw(id="z", verified=False))
    assert len(w) == 1
    assert issubclass(w[0].category, UnverifiedCatalogWarning)
    msg = str(w[0].message)
    assert "x" in msg and "z" in msg and "y," not in msg


def test_unverified_default_flag_is_false() -> None:
    assert ModelProfile("m", "p", 0, 0, 1).verified is False


def test_load_unverified_file_warns_once(tmp_path: Path) -> None:
    text = MINIMAL.replace("verified = true", "verified = false")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_catalog(write(tmp_path, text))
    assert [type(x.message) for x in w] == [UnverifiedCatalogWarning]


def test_verified_catalog_does_not_warn(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_catalog(write(tmp_path, MINIMAL))


# --- validation -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"colour": 1}, "colour"),
        ({"efforts": ["ultra"]}, "efforts"),
        ({"efforts": []}, "efforts"),
        ({"efforts": ["low", "low"]}, "efforts"),
        ({"effort_output_mult": {"high": 1.0}}, "effort_output_mult"),
        ({"effort_output_mult": {"medium": -1.0}}, "effort_output_mult"),
        # Partly filled: "low" has a multiplier, "high" does not.
        (
            {"efforts": ["low", "high"], "effort_output_mult": {"low": 1.0}},
            "effort_output_mult",
        ),
        ({"effort_map": {"high": "x"}}, "effort_map"),
        ({"price_in": -0.1}, "price_in"),
        ({"price_out": -1}, "price_out"),
        ({"cache_write_mult": -1}, "cache_write_mult"),
        ({"cache_read_mult": -0.5}, "cache_read_mult"),
        ({"context_window": 0}, "context_window"),
        ({"context_window": -5}, "context_window"),
        ({"tier": -1}, "tier"),
        ({"price_in": "free"}, "price_in"),
        (
            {"price_tiers": [{"above_tokens": 5, "price_in": 1, "price_out": 1}] * 2},
            "price_tiers[1]",
        ),
        (
            {
                "price_tiers": [
                    {"above_tokens": 9, "price_in": 1, "price_out": 1},
                    {"above_tokens": 5, "price_in": 1, "price_out": 1},
                ]
            },
            "price_tiers[1]",
        ),
        ({"price_tiers": [{"above_tokens": 5, "price_in": -1, "price_out": 1}]}, "price_tiers[0]"),
        ({"price_tiers": [{"above_tokens": 5, "price_in": 1, "price_out": 1, "x": 1}]}, "x"),
    ],
)
def test_invalid_model_names_model_and_field(over: dict[str, Any], field: str) -> None:
    with pytest.raises(CatalogError) as ei:
        build(raw(**over))
    assert ei.value.model == "m"
    assert field in (ei.value.field or "")
    assert "'m'" in str(ei.value)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(CatalogError, match="duplicate") as ei:
        build(raw(), raw())
    assert ei.value.model == "m"


def test_unknown_top_level_key_and_bad_toml(tmp_path: Path) -> None:
    with pytest.raises(CatalogError, match="top-level"):
        Catalog.from_dict({"models": []})
    with pytest.raises(CatalogError, match="invalid TOML"):
        load_catalog(write(tmp_path, "[[model"))


def test_missing_required_field() -> None:
    with pytest.raises(CatalogError, match="missing"):
        Catalog.from_dict({"model": [{"id": "q"}]})


# --- packaging --------------------------------------------------------------------------------


def test_default_toml_is_package_data() -> None:
    import steproute.catalogs as pkg

    assert (Path(pkg.__file__).parent / "default.toml").is_file()


@pytest.mark.skipif(not os.environ.get("STEPROUTE_TEST_OLLAMA"), reason="needs a real Ollama")
def test_real_ollama_tool_call() -> None:  # pragma: no cover
    from steproute.core.toolcheck import check_model

    model = os.environ.get("STEPROUTE_TEST_OLLAMA_MODEL", "qwen3-coder:30b")
    result = check_model(model)
    assert result.ok, result.detail
