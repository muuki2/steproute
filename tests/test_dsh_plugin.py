"""The dsh plugin against the real service (no dsh involved).

`tests/dsh/e2e_driver.mjs` loads the plugin's entry module on a fake cordis context, with its
real HTTP client, and drives a fake dsh agent loop (`tests/dsh/fake_dsh.mjs`) at a
`steproute serve` built from the plugin's own service config (`plugins/dsh/service/`). This
checks what the Node unit tests with a fake service cannot: that the state the plugin sends
passes the service's schema, that the config restricts decisions to models dsh can run, that
decisions and outcomes join in the store with the catalog's prices, and that stopping the
service between turns leaves the session running on the user's model. The unit tests are in
tests/dsh/ (run by scripts/ci.sh). Skipped when Node is not installed.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from steproute.core import Decision, Outcome
from steproute.core.service import build_service, load_service_config, make_server
from steproute.core.store import TraceStore, compute_stats

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "dsh"
DRIVER = ROOT / "tests" / "dsh" / "e2e_driver.mjs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node is not installed")


class Service:
    def __init__(self, store: Path) -> None:
        cfg = load_service_config(PLUGIN / "service" / "steproute.toml")
        cfg = dataclasses.replace(cfg, port=0, store_path=store)
        service, token = build_service(cfg, environ={})
        self.server = make_server(service, token, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture
def service(tmp_path: Path) -> Iterator[Service]:
    s = Service(tmp_path / "traces")
    try:
        yield s
    finally:
        s.stop()


def records(store: Path) -> tuple[list[Decision], dict[str, Outcome]]:
    decisions: list[Decision] = []
    outcomes: dict[str, Outcome] = {}
    for r in TraceStore(store).records():
        if isinstance(r, Decision):
            decisions.append(r)
        elif isinstance(r, Outcome):
            outcomes[r.decision_id] = r
    # A turn's first decision is asked for without waiting, so the store's order is not
    # always step order.
    decisions.sort(key=lambda d: d.state.step_idx)
    return decisions, outcomes


def drive(url: str, mode: str) -> dict[str, Any]:
    assert NODE is not None
    out = subprocess.run(
        [NODE, str(DRIVER), url, mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    result: dict[str, Any] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize("mode", ["shadow", "route"])
def test_three_steps_join_with_the_real_service(
    service: Service, tmp_path: Path, mode: str
) -> None:
    result = drive(service.url, mode)
    decisions, outcomes = records(tmp_path / "traces")
    assert len(decisions) == 3
    assert all(d.decision_id in outcomes for d in decisions)
    assert compute_stats(TraceStore(tmp_path / "traces")).join_rate == 1.0
    # Only models of the DeepSeek route are candidates.
    for d in decisions:
        assert d.action.model.startswith("deepseek-")
        assert d.state.harness == "dsh"
        assert d.state.task_text is None  # sent for sensitive detection, dropped by the store
    kinds = [d.state.kind for d in decisions]
    assert kinds == ["main", "tool_followup", "tool_followup"]
    second = decisions[1].state
    assert second.current_model == "deepseek-flash"
    assert second.cached_prefix_tokens == 2000
    assert "a.txt" in second.features["paths"]
    for d in decisions:
        o = outcomes[d.decision_id]
        assert o.cost_usd is not None and o.cost_usd > 0  # priced from the catalog
        assert o.input_tokens > 0 and o.output_tokens > 0
        assert o.signals["routing_mode"] == mode
    if mode == "shadow":
        assert result["selects"] == []
        assert result["ran"] == ["deepseek-flash"] * 3
    else:
        # Applied between steps: the steps after a tool batch ran on the decision's model.
        for d in decisions[1:]:
            o = outcomes[d.decision_id]
            assert o.signals["applied"] is True
            assert o.final_model == d.action.model
            assert o.signals["selection_ran"] is True
        assert result["ran"][1:] == [decisions[1].action.model, decisions[2].action.model]


def test_killing_the_service_mid_session_does_not_interrupt_it(tmp_path: Path) -> None:
    """Criterion 2 without dsh: `steproute serve` as its own process, killed with SIGKILL."""
    assert NODE is not None
    serve = Path(sys.executable).parent / "steproute"
    service = subprocess.Popen(
        [str(serve), "serve", "--config", str(PLUGIN / "service" / "steproute.toml"), "--port", "0"]
        + ["--store", str(tmp_path / "traces")],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert service.stdout is not None
    line = service.stdout.readline().strip()
    url = line.removeprefix("steproute serve: listening on ")
    proc = subprocess.Popen(
        [NODE, str(DRIVER), url, "route", "--pause"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None and proc.stdin is not None
    try:
        assert proc.stdout.readline().strip() == "PAUSED"
        service.kill()
        service.wait(10)
        proc.stdin.write("go\n")
        proc.stdin.flush()
        out, err = proc.communicate(timeout=60)
    finally:
        for p in (proc, service):
            if p.poll() is None:
                p.kill()
                p.wait(10)
    assert proc.returncode == 0, err
    result = json.loads(out.strip().splitlines()[-1])
    # Six steps ran. The first step after the kill starts a turn, so it runs on the model last
    # selected (its selection is read before anything can be awaited); from the next step on
    # the session is back on the user's model. No step waited longer than a refused
    # connection takes.
    assert len(result["ran"]) == 6
    assert result["ran"][2] == result["ran"][3]
    assert result["ran"][4:] == ["deepseek-flash"] * 2
    assert all(r["postExecuteMs"] < 200 for r in result["record"][2:])
    decisions, outcomes = records(tmp_path / "traces")
    assert len(decisions) == 3
    assert all(d.decision_id in outcomes for d in decisions)
