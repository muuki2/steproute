"""The MCP server: advisor logic, SDK wiring and a replay of a recorded stdio session.

The advisor tests need no SDK. The wire tests skip when the ``mcp`` extra is not installed.
Fake secrets are assembled at runtime so no credential-shaped literal is in this file.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from steproute.collectors.mcp_server import (
    LIST_MODELS_DESCRIPTION,
    REPORT_OUTCOME_DESCRIPTION,
    ROUTE_TASK_DESCRIPTION,
    TIMED_OUT_SIGNAL,
    Advisor,
    McpConfig,
    McpConfigError,
)
from steproute.core import TraceStore
from steproute.core.catalog import load_default_catalog
from steproute.core.service import ServiceConfig, build_service

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_session.json"
LAUNCH = "import sys; from steproute.cli import main; sys.exit(main(sys.argv[1:]))"
CATALOG = load_default_catalog(warn_unverified=False)


def fake_secret() -> str:
    return "AK" + "IA" + "A" * 16  # matches the aws_access_key_id rule; built at runtime


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


class Broken:
    name = "broken"

    def decide(self, state: Any) -> Any:
        raise RuntimeError("boom")

    def report(self, outcome: Any) -> None:
        return None


def make(tmp_path: Path, clock: Clock | None = None, policy: Any = None, **kw: Any) -> Advisor:
    service, _ = build_service(
        ServiceConfig(store_path=tmp_path / "traces"), policy=policy, catalog=CATALOG
    )
    return Advisor(service, McpConfig(**kw), clock=clock or Clock(), session_id="mcp-test")


def joined(advisor: Advisor) -> list[Any]:
    return list(TraceStore(advisor.service.store.path).joined())


def test_every_route_task_is_logged_and_joins_with_its_outcome(tmp_path: Path) -> None:
    adv = make(tmp_path)
    a = adv.route_task("summarise the tests", kind="subagent", tools=["Read"], est_size="small")
    b = adv.route_task("write a parser", est_size=12000)
    assert a["recommendation"] and b["recommendation"]
    assert a["decision_id"] != b["decision_id"]
    assert adv.report_outcome(a["decision_id"], True, {"tests_passed": True})["recorded"]
    pairs = {d.decision_id: o for d, o in joined(adv)}
    assert set(pairs) == {a["decision_id"], b["decision_id"]}
    assert pairs[a["decision_id"]] is not None and pairs[a["decision_id"]].success is True
    assert pairs[b["decision_id"]] is None
    decisions = {d.decision_id: d for d, _ in joined(adv)}
    st = decisions[a["decision_id"]].state
    assert (st.harness, st.session_id, st.step_idx, st.kind, st.source) == (
        "mcp",
        "mcp-test",
        0,
        "subagent",
        "personal",
    )
    assert (st.prompt_tokens, st.tools) == (2000, ["Read"])
    assert decisions[b["decision_id"]].state.step_idx == 1
    assert decisions[b["decision_id"]].state.prompt_tokens == 12000


def test_result_shape_and_reason(tmp_path: Path) -> None:
    adv = make(tmp_path)
    r = adv.route_task("look around", kind="subagent", tools=["Read", "Grep"], est_size="small")
    assert r["model"] == "claude-haiku-5-5" and r["effort"] == "low"
    assert r["cascade"] == [
        {"model": "claude-haiku-5-5", "effort": "medium"},
        {"model": "claude-sonnet-5-5", "effort": "low"},
    ]
    assert r["reason"] == "Read-only subagent work, so a cheaper tier (4)."
    assert r["sensitive"] is False and r["local"] is False
    d = adv.route_task("design a thing")
    assert d["reason"] == "No special rule applies, so the default tier (5)."


def test_timeout_closes_unreported_decisions_with_success_none(tmp_path: Path) -> None:
    clock = Clock()
    adv = make(tmp_path, clock, timeout_s=1800)
    old = adv.route_task("first")["decision_id"]
    clock.now += 1000
    fresh = adv.route_task("second")["decision_id"]
    assert adv.pending == 2
    clock.now += 900  # old is now 1900 s old, fresh 900 s
    assert adv.expire() == 1
    by_id = {d.decision_id: o for d, o in joined(adv)}
    assert by_id[old].success is None and by_id[old].signals == {TIMED_OUT_SIGNAL: True}
    assert by_id[fresh] is None
    assert adv.pending == 1
    # a late report for the closed decision is still stored and replaces the timeout outcome
    assert adv.report_outcome(old, False, {"tool_error": True})["recorded"]
    by_id = {d.decision_id: o for d, o in joined(adv)}
    assert by_id[old].success is False
    # the next tool call sweeps too
    clock.now += 5000
    adv.list_models()
    adv.route_task("third")
    assert {d.decision_id: o for d, o in joined(adv)}[fresh].signals == {TIMED_OUT_SIGNAL: True}


def test_close_all_on_shutdown(tmp_path: Path) -> None:
    adv = make(tmp_path)
    ids = [adv.route_task(f"t{i}")["decision_id"] for i in range(3)]
    adv.report_outcome(ids[0], True)
    assert adv.close_all() == 2
    outcomes = {d.decision_id: o for d, o in joined(adv)}
    assert outcomes[ids[0]].success is True
    assert all(outcomes[i].signals == {TIMED_OUT_SIGNAL: True} for i in ids[1:])
    assert adv.close_all() == 0


def test_secret_in_description_is_local_only(tmp_path: Path) -> None:
    adv = make(tmp_path)
    r = adv.route_task(f"deploy using key {fake_secret()} to the server")
    assert r["sensitive"] is True and r["local"] is True
    local = {m.id for m in CATALOG if m.local}
    assert r["model"] in local
    assert {s["model"] for s in r["cascade"]} <= local
    assert "sensitive" in r["reason"] and "local" in r["reason"]
    stored = joined(adv)[0][0]
    assert stored.state.sensitive is True


def test_prompt_text_is_not_stored_by_default(tmp_path: Path) -> None:
    adv = make(tmp_path)
    marker = "zanzibar-unique-phrase-" + "x" * 5
    adv.route_task(f"please refactor {marker}")
    adv.report_outcome(adv.route_task(f"also {marker}")["decision_id"], True, {"note": "ok"})
    blob = "".join(p.read_text() for p in Path(adv.service.store.path).rglob("*") if p.is_file())
    assert blob and marker not in blob
    assert joined(adv)[0][0].state.task_text is None


def test_unknown_decision_id_is_not_found_and_not_stored(tmp_path: Path) -> None:
    adv = make(tmp_path)
    adv.route_task("x")
    r = adv.report_outcome("f" * 32, True)
    assert r["found"] is False and r["recorded"] is False
    assert all(o is None for _, o in joined(adv))


def test_explicit_success_is_not_overridden_and_unknown_signals_are_kept(tmp_path: Path) -> None:
    adv = make(tmp_path)
    did = adv.route_task("x")["decision_id"]
    adv.report_outcome(did, False, {"tests_passed": True, "my_metric": 7})
    out = joined(adv)[0][1]
    assert out.success is False and out.signals == {"tests_passed": True, "my_metric": 7}
    did2 = adv.route_task("y")["decision_id"]
    assert adv.report_outcome(did2, None, {"tests_passed": True})["success"] is True  # labelled
    did3 = adv.route_task("z")["decision_id"]
    assert adv.report_outcome(did3)["success"] is None


def test_no_decision_gives_a_clear_result(tmp_path: Path) -> None:
    adv = make(tmp_path, policy=Broken())
    r = adv.route_task("anything")
    assert r["recommendation"] is False and r["decision_id"] is None and r["model"] is None
    assert "own default model" in r["reason"] and r["sensitive"] is False
    s = adv.route_task(f"use {fake_secret()}")
    assert s["recommendation"] is False and s["sensitive"] is True
    assert "local" in s["reason"]
    assert joined(adv) == [] and adv.pending == 0


def test_est_size_parsing(tmp_path: Path) -> None:
    adv = make(tmp_path, size_tokens={"small": 1, "medium": 2, "large": 3})
    assert [adv.parse_size(v) for v in ("small", "Large", 5, "700", None)] == [1, 3, 5, 700, 2]
    for bad in ("huge", -1, True, "1.5"):
        with pytest.raises(ValueError):
            adv.parse_size(bad)  # type: ignore[arg-type]
    with pytest.raises(McpConfigError):
        McpConfig(size_tokens={"small": 1})
    with pytest.raises(McpConfigError):
        McpConfig(timeout_s=0)


def test_list_models_excludes_holdout(tmp_path: Path) -> None:
    models = make(tmp_path).list_models()["models"]
    assert {m["id"] for m in models} == {m.id for m in CATALOG.pool()}
    assert all(set(m) == {"id", "provider", "tier", "efforts", "local"} for m in models)
    assert any(m["local"] for m in models) and any(not m["local"] for m in models)


def test_descriptions_say_what_a_stranger_needs() -> None:
    for text in (ROUTE_TASK_DESCRIPTION, REPORT_OUTCOME_DESCRIPTION, LIST_MODELS_DESCRIPTION):
        assert "cannot change" in text or "does not change" in text
    assert "BEFORE" in ROUTE_TASK_DESCRIPTION and "subagent" in ROUTE_TASK_DESCRIPTION
    assert "yourself" in ROUTE_TASK_DESCRIPTION
    assert "ALWAYS" in REPORT_OUTCOME_DESCRIPTION and "failures" in REPORT_OUTCOME_DESCRIPTION


# --- wire level (needs the mcp extra) ------------------------------------------------------


def test_sdk_server_registers_three_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    from steproute.collectors.mcp_server import build_mcp_server

    async def go() -> None:
        async with Client(build_mcp_server(make(tmp_path))) as client:
            tools = {t.name: t for t in (await client.list_tools()).tools}
            assert set(tools) == {"route_task", "report_outcome", "list_models"}
            assert tools["route_task"].description == ROUTE_TASK_DESCRIPTION
            assert set(tools["route_task"].input_schema["properties"]) == {
                "description",
                "kind",
                "tools",
                "est_size",
            }
            assert tools["route_task"].input_schema["required"] == ["description"]
            r = await client.call_tool("route_task", {"description": "hi", "est_size": "large"})
            assert r.structured_content["recommendation"] is True
            bad = await client.call_tool("route_task", {"description": "hi", "est_size": "huge"})
            assert bad.is_error

    anyio.run(go)


ID = re.compile(r"\b[0-9a-f]{32}\b")
VOLATILE = {"serverInfo": {"version": "<v>"}}


def normalise(value: Any) -> Any:
    if isinstance(value, str):
        return ID.sub("<id>", value)
    if isinstance(value, list):
        return [normalise(v) for v in value]
    if isinstance(value, dict):
        return {k: normalise(v) for k, v in value.items()}
    return value


def test_replay_of_recorded_stdio_session(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    recorded = json.loads(FIXTURE.read_text())["messages"]
    proc = subprocess.Popen(
        [sys.executable, "-c", LAUNCH, "mcp", "--store", str(tmp_path / "traces")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin and proc.stdout
    live_id: str | None = None
    recorded_id: str | None = None
    replayed = 0
    try:
        pending: list[dict[str, Any]] = []
        for entry in recorded:
            msg = entry["message"]
            if entry["dir"] == "out":
                if recorded_id and live_id:
                    msg = json.loads(json.dumps(msg).replace(recorded_id, live_id))
                proc.stdin.write(json.dumps(msg) + "\n")
                proc.stdin.flush()
                if "id" in msg:
                    pending.append(entry)
                continue
            got = json.loads(proc.stdout.readline())
            assert normalise(got) == normalise(msg), f"response {msg.get('id')} differs"
            text = json.dumps(msg)
            if recorded_id is None and '\\"decision_id\\": \\"' in text:
                recorded_id = ID.search(text.split("decision_id")[1]).group(0)  # type: ignore[union-attr]
                live_id = ID.search(json.dumps(got).split("decision_id")[1]).group(0)  # type: ignore[union-attr]
            replayed += 1
    finally:
        proc.stdin.close()
        proc.wait(10)
    assert replayed == sum(1 for e in recorded if e["dir"] == "in") == 5
    assert proc.returncode == 0
    store = TraceStore(tmp_path / "traces")
    ((decision, outcome),) = list(store.joined())
    assert outcome is not None and outcome.success is True
    assert decision.decision_id == live_id
