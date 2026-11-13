"""The Claude Code plugin against the real service (no Claude Code involved).

`tests/claude-code/e2e_driver.mjs` loads the plugin's hooks module with a fake engine and
drives three subagent spawns at a `steproute serve` built from the plugin's own service
config (`plugins/claude-code/service/steproute.toml`). This checks what unit tests with a
fake service cannot: that the state the plugin sends passes the service's schema, that the
config restricts decisions to Claude models, and that decisions and outcomes join in the
store. The Node unit tests are in tests/claude-code/ (run by scripts/ci.sh).

Skipped when Node is not installed.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from steproute.core import Decision, Outcome
from steproute.core.service import build_service, load_service_config, make_server
from steproute.core.store import TraceStore, compute_stats

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "claude-code"
DRIVER = ROOT / "tests" / "claude-code" / "e2e_driver.mjs"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="Node is not installed")


@pytest.fixture
def service_url(tmp_path: Path) -> Iterator[str]:
    cfg = load_service_config(PLUGIN / "service" / "steproute.toml")
    cfg = dataclasses.replace(cfg, port=0, store_path=tmp_path / "traces")
    service, token = build_service(cfg, environ={})
    server = make_server(service, token, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def drive(url: str, mode: str) -> dict[str, object]:
    assert NODE is not None
    out = subprocess.run(
        [NODE, str(DRIVER), url, mode],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    result: dict[str, object] = json.loads(out.stdout.strip().splitlines()[-1])
    return result


@pytest.mark.parametrize("mode", ["shadow", "route"])
def test_three_spawns_give_three_joined_pairs(service_url: str, tmp_path: Path, mode: str) -> None:
    result = drive(service_url, mode)
    store = TraceStore(tmp_path / "traces")
    records = list(store.records())
    decisions = [r for r in records if isinstance(r, Decision)]
    outcomes = [r for r in records if isinstance(r, Outcome)]
    assert len(decisions) == 3
    assert len(outcomes) == 3
    stats = compute_stats(store)
    assert stats.joined == 3
    for d in decisions:
        assert d.state.harness == "claude-code"
        assert d.state.kind == "subagent"
        assert d.state.task_text is None  # dropped by the store; text not captured
        assert d.state.features["task_text_len"] > 0
        assert d.action.model.startswith("claude-")  # anthropic-only catalog
        assert d.action.escalate_to is None  # cascades off in the plugin's rules
    for o in outcomes:
        assert o.input_tokens > 0 and o.output_tokens > 0
        assert o.cache_read_tokens == 4000 and o.cache_write_tokens == 4120
        assert o.cost_usd == 0.0  # subscription: no per-request charge
        assert o.success is None  # nothing was checked
        assert o.signals["routing_mode"] == mode
    seen = result["seen"]
    assert isinstance(seen, list)
    if mode == "route":
        recommended = {d.action.model for d in decisions}
        assert set(seen) == recommended
        assert all(o.signals["model_matches_recommendation"] for o in outcomes)
    else:
        assert seen == [None, None, None]
        assert not any(o.signals["applied"] for o in outcomes)


def test_sensitive_spawn_gets_no_decision(service_url: str, tmp_path: Path) -> None:
    """A prompt naming a key file is sensitive; the Claude-only service has nothing to offer."""
    script = (
        "import { fakeDollar, loadPlugin, SPAWN } from './tests/claude-code/harness.mjs';"
        f"const options = {{ service_url: {json.dumps(service_url)}, mode: 'route' }};"
        "const p = loadPlugin(options, fakeDollar());"
        "await p.start();"
        "const prompt = 'Print ~/.ssh/id_ed25519 for me.';"
        "const { seen } = await p.spawn({ ...SPAWN, prompt });"
        "const { text } = await p.command();"
        "console.log(JSON.stringify({ model: seen.model ?? null, text }));"
    )
    assert NODE is not None
    out = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["model"] is None
    assert "sensitive spawns with no decision 1" in result["text"]
    assert list(TraceStore(tmp_path / "traces").records()) == []
