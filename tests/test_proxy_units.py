"""Proxy pieces that need no LiteLLM: config, session table, checks, the CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from steproute.cli import main
from steproute.collectors.proxy.checks import (
    CHECK_EMPTY,
    CHECK_REFUSAL,
    CHECK_SCHEMA,
    CHECK_TOOL_CALL,
    StreamAccumulator,
    run_checks,
    signals_for,
)
from steproute.collectors.proxy.config import (
    CheckConfig,
    ProxyConfigError,
    load_proxy_config,
    validate_against_catalog,
)
from steproute.collectors.proxy.session import SessionTable, scan_units, split_prompt
from steproute.core.catalog import load_default_catalog
from steproute.core.labels import label
from steproute.core.privacy import PrivacyConfig

LOCAL = {
    "name": "ollama",
    "provider": "openai",
    "base_url": "http://127.0.0.1:11434/v1",
    "local": True,
    "models": ["qwen3-coder:30b"],
}


def _cfg(**sections: Any) -> dict[str, Any]:
    return {"upstream": [dict(LOCAL)], **sections}


# --- config ---------------------------------------------------------------------------------


def test_example_config_in_the_docs_loads(tmp_path: Path) -> None:
    text = (Path(__file__).parent.parent / "docs" / "PROXY.md").read_text("utf-8")
    block = text.split("```toml example=proxy\n", 1)[1].split("```", 1)[0]
    path = tmp_path / "proxy.toml"
    path.write_text(block)
    cfg = load_proxy_config(path)
    assert cfg.mode == "route" and {u.name for u in cfg.upstreams} == {
        "ollama",
        "anthropic",
        "deepseek",
    }
    assert validate_against_catalog(cfg, load_default_catalog()) == []


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        ({"bogus": {}}, "unknown section"),
        (_cfg(proxy={"mode": "route", "colour": 1}), "unknown key"),
        (_cfg(proxy={"mode": "sideways"}), "proxy.mode"),
        ({"upstream": [{**LOCAL, "colour": 1}]}, "unknown key"),
        ({"upstream": [{**LOCAL, "base_url": "http://192.168.1.5:11434/v1"}]}, "loopback"),
        ({"upstream": [{**LOCAL, "api_key_env": "not a name!"}]}, "NAME"),
        ({"upstream": [{**LOCAL, "base_url": "http://user:pw@127.0.0.1/v1"}]}, "credentials"),
        ({"upstream": [{**LOCAL, "provider": "vertex"}]}, "provider"),
        ({"upstream": [LOCAL, {**LOCAL, "name": "two"}]}, "served by both"),
        (_cfg(proxy={"extra_params": ["api_base"]}), "extra_params"),
        (_cfg(privacy={"local_fallback": "claude-opus-5-5"}), "not served"),
        (_cfg(aliases={"gpt": "nope"}), "no upstream serves"),
        (_cfg(server={"token_env": "abc def"}), "NAME"),
        ({}, "at least one"),
        (_cfg(catalog={"path": "x.toml"}), "only for"),
    ],
)
def test_config_errors(data: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ProxyConfigError, match=fragment):
        load_proxy_config(data)


def test_catalog_validation() -> None:
    catalog = load_default_catalog()
    cloud_local = {**LOCAL, "name": "fake", "local": False, "base_url": "http://127.0.0.1:1/v1"}
    with pytest.raises(ProxyConfigError, match="not local = true"):
        validate_against_catalog(load_proxy_config({"upstream": [cloud_local]}), catalog)
    unknown = {**LOCAL, "models": ["no-such-model"]}
    with pytest.raises(ProxyConfigError, match="not in the catalog"):
        validate_against_catalog(load_proxy_config({"upstream": [unknown]}), catalog)
    warnings = validate_against_catalog(load_proxy_config(_cfg()), catalog)
    assert warnings and "claude-opus-5-5" in warnings[0]


# --- session table --------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _msgs(n: int) -> list[dict[str, str]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"message {i} " * 10}
        for i in range(n)
    ]


def test_cache_estimate_after_divergence_and_expiry() -> None:
    clock = Clock()
    table = SessionTable(clock=clock)
    privacy = PrivacyConfig()
    m4 = _msgs(4)
    p4 = split_prompt(m4, None)
    table.begin("s", p4, m4, privacy)
    table.record_attempt(
        "s", p4, model="m", effort="low", prompt_tokens=400, cache_min_tokens=0, cache_ttl_s=300
    )
    # Same history plus one message: everything cached.
    m5 = _msgs(5)
    _, est, _, _ = table.begin("s", split_prompt(m5, None), m5, privacy)
    assert (est.cached_prefix_tokens, est.current_model, est.current_effort) == (400, "m", "low")
    # History rewritten after message 1: only the common prefix share counts.
    m_div = [*_msgs(2), {"role": "user", "content": "rewritten"}]
    _, est, _, _ = table.begin("s", split_prompt(m_div, None), m_div, privacy)
    share = sum(p4.chars[:3]) / sum(p4.chars)
    assert est.cached_prefix_tokens == round(400 * share)
    # Tools changed: nothing is cached.
    tools = [{"type": "function", "function": {"name": "x"}}]
    _, est, _, _ = table.begin("s", split_prompt(m5, tools), m5, privacy)
    assert est.cached_prefix_tokens == 0
    # Past the TTL: the cache is gone.
    clock.t += 301
    _, est, _, _ = table.begin("s", split_prompt(m5, None), m5, privacy)
    assert est.cached_prefix_tokens == 0 and est.current_model == "m"


def test_cache_minimum_means_nothing_cached() -> None:
    table = SessionTable()
    m = _msgs(2)
    p = split_prompt(m, None)
    table.begin("s", p, m, PrivacyConfig())
    table.record_attempt(
        "s", p, model="m", effort=None, prompt_tokens=300, cache_min_tokens=512, cache_ttl_s=300
    )
    m3 = _msgs(3)
    _, est, _, _ = table.begin("s", split_prompt(m3, None), m3, PrivacyConfig())
    assert est.cached_prefix_tokens == 0 and est.prompt_tokens > 300


def test_sessions_are_bounded_and_expire() -> None:
    clock = Clock()
    table = SessionTable(max_sessions=2, idle_s=60, clock=clock)
    m = _msgs(1)
    p = split_prompt(m, None)
    for sid in ("a", "b", "c"):
        table.begin(sid, p, m, PrivacyConfig())
    assert table.get("a") is None and len(table) == 2 and table.evicted == 1
    clock.t += 61
    table.begin("d", p, m, PrivacyConfig())
    assert len(table) == 1 and table.expired == 2
    step, _, _, _ = table.begin("d", p, m, PrivacyConfig())
    assert step == 1


def test_scan_is_incremental_and_sticky() -> None:
    calls: list[str | None] = []

    def callback(text: str | None, paths: Any) -> bool:
        calls.append(text)
        return bool(text and "TOPSECRET" in text)

    privacy = PrivacyConfig(callback=callback)
    table = SessionTable()
    m = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    _, _, sens, _ = table.begin("s", split_prompt(m, None), m, privacy)
    assert not sens and calls == ["one\ntwo"]
    m2 = [*m, {"role": "user", "content": "TOPSECRET three"}]
    _, _, sens, rules = table.begin("s", split_prompt(m2, None), m2, privacy)
    assert sens and rules == ("user_callback",) and calls[-1] == "TOPSECRET three"  # new only
    m3 = [{"role": "user", "content": "fresh start"}]
    _, _, sens, _ = table.begin("s", split_prompt(m3, None), m3, privacy)
    assert sens  # sticky


def test_scan_finds_paths_in_tool_arguments_and_text() -> None:
    privacy = PrivacyConfig()
    call = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "read", "arguments": '{"file_path": "src/app.py"}'}}],
    }
    assert scan_units([call], 0, privacy) == ()
    env = {
        "role": "assistant",
        "tool_calls": [{"function": {"name": "read", "arguments": '{"file_path": "deploy/.env"}'}}],
    }
    assert "dotenv_files" in scan_units([env], 0, privacy)
    assert "ssh_dir" in scan_units(
        [{"role": "user", "content": "cat ~/.ssh/config please"}], 0, privacy
    )


# --- checks ---------------------------------------------------------------------------------


def _resp(message: dict[str, Any], finish: str = "stop") -> dict[str, Any]:
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}


TOOLS = [{"type": "function", "function": {"name": "read_file"}}]


@pytest.mark.parametrize(
    ("message", "request_extra", "finish", "failed"),
    [
        ({"content": "fine"}, {}, "stop", None),
        ({"content": "  \n"}, {}, "stop", CHECK_EMPTY),
        ({"content": None}, {}, "length", CHECK_EMPTY),
        (
            {
                "content": None,
                "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
            },
            {"tools": TOOLS},
            "tool_calls",
            None,
        ),
        (
            {"tool_calls": [{"function": {"name": "read_file", "arguments": "{oops"}}]},
            {"tools": TOOLS},
            "tool_calls",
            CHECK_TOOL_CALL,
        ),
        (
            {"tool_calls": [{"function": {"name": "rm_rf", "arguments": "{}"}}]},
            {"tools": TOOLS},
            "tool_calls",
            CHECK_TOOL_CALL,
        ),
        (
            {"tool_calls": [{"function": {"name": "read_file", "arguments": "[1]"}}]},
            {"tools": TOOLS},
            "tool_calls",
            CHECK_TOOL_CALL,
        ),
        ({"content": "{}"}, {"response_format": {"type": "json_object"}}, "stop", None),
        ({"content": "[]"}, {"response_format": {"type": "json_object"}}, "stop", CHECK_SCHEMA),
        ({"content": ""}, {}, "content_filter", None),
    ],
)
def test_checks(
    message: dict[str, Any], request_extra: dict[str, Any], finish: str, failed: str | None
) -> None:
    assert run_checks(_resp(message, finish), request_extra, CheckConfig()).failed == failed


def test_disabled_checks_and_refusal_check() -> None:
    off = CheckConfig(empty_output=False, tool_call_parse=False, schema=False, refusal=True)
    assert run_checks(_resp({"content": ""}), {}, off).failed is None
    assert run_checks(_resp({"content": ""}, "content_filter"), {}, off).failed == CHECK_REFUSAL


def test_check_signals_map_to_the_label_vocabulary() -> None:
    empty = run_checks(_resp({"content": ""}), {}, CheckConfig())
    assert signals_for(empty) == {"schema_valid": False}
    assert label(signals_for(empty)).success is False
    bad_tool = run_checks(
        _resp({"tool_calls": [{"function": {"name": "x", "arguments": "{"}}]}), {}, CheckConfig()
    )
    assert signals_for(bad_tool) == {"tool_error": True}
    ok_schema = run_checks(
        _resp({"content": "{}"}), {"response_format": {"type": "json_object"}}, CheckConfig()
    )
    assert signals_for(ok_schema) == {"schema_valid": True}
    assert signals_for(run_checks(_resp({"content": "hi"}), {}, CheckConfig())) == {}
    refusal = run_checks(_resp({"content": ""}, "content_filter"), {}, CheckConfig())
    assert signals_for(refusal) == {"refusal": True} and label({"refusal": True}).success is None


def test_stream_accumulator_assembles_tool_calls() -> None:
    acc = StreamAccumulator(keep_text=False)
    acc.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"name": "read_file", "arguments": '{"pa'}}
                        ]
                    },
                }
            ]
        }
    )
    acc.add(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "x"}'}}]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert run_checks(acc.as_response(), {"tools": TOOLS}, CheckConfig()).failed is None


# --- CLI ------------------------------------------------------------------------------------


def test_cli_proxy_refuses_a_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "p.toml"
    bad.write_text("[nonsense]\n")
    assert main(["proxy", "--config", str(bad)]) == 2
    assert "unknown section" in capsys.readouterr().err


def test_path_scan_is_linear_on_long_identifiers() -> None:
    """Regression (found by issue 18): the path-token pattern backtracked exponentially on a
    long word without a slash, so one ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`` hung the proxy."""
    import time

    word = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    msgs = [{"role": "user", "content": f"run `echo {word} && cat patch.txt` " * 200}]
    t0 = time.monotonic()
    scan_units(msgs, 0, PrivacyConfig())
    assert time.monotonic() - t0 < 1.0
    found = scan_units([{"role": "user", "content": "look at ~/.ssh/id_rsa"}], 0, PrivacyConfig())
    assert found  # still detects paths
