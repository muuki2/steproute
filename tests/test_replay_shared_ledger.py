"""Issue 60: one spend ledger for every output directory.

Acceptance: two runners in different output directories, started at once, cannot together
pass a provider's monthly cap or its total budget (``test_two_runners_in_two_directories_...``,
real processes); a crash between intent and settle is still charged at its worst case
(``test_a_runner_killed_between_intent_and_settle_...``); ``steproute replay plan`` shows
month-to-date and all-time spend from the shared ledger (``test_plan_shows_spend_...``).

Every ledger here is under ``tmp_path``; ``conftest.py`` fails a test that touches the default
location. Subprocesses are coordinated through pipes, never sleeps.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from steproute.cli import main
from steproute.replay.config import parse_replay_config
from steproute.replay.ledger import (
    LEDGER_ENV,
    CapExceeded,
    CapGuard,
    Ledger,
    UnknownSpend,
    default_ledger_path,
    read_lines,
    resolve_ledger_path,
)
from steproute.tasks.check import workspace_env

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src"


def _env() -> dict[str, str]:
    env = workspace_env()  # carries conftest's XDG_STATE_HOME: never the real default
    env["PYTHONPATH"] = os.pathsep.join([str(TESTS), str(SRC)])
    return env


def _spawn(tmp: Path, name: str, code: str, *args: str) -> subprocess.Popen[str]:
    script = tmp / f"{name}.py"
    script.write_text(code, "utf-8")
    with open(tmp / f"{name}.err", "w") as err:  # the child keeps its own copy
        return subprocess.Popen(
            [sys.executable, str(script), *args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
            env=_env(),
        )


def _expect(p: subprocess.Popen[str], line: str, tmp: Path, name: str) -> None:
    assert p.stdout is not None
    got = p.stdout.readline().strip()
    if got != line:
        p.kill()
        p.wait()
        raise AssertionError(f"{name}: expected {line!r}, got {got!r}\n{_err(tmp, name)}")


def _err(tmp: Path, name: str) -> str:
    path = tmp / f"{name}.err"
    return path.read_text("utf-8")[-3000:] if path.exists() else ""


# --- where the ledger lives -------------------------------------------------------------------


def test_the_default_is_the_per_user_state_directory() -> None:
    assert default_ledger_path({"XDG_STATE_HOME": "/s"}) == Path("/s/steproute/ledger.jsonl")
    # A relative XDG_STATE_HOME is invalid by the specification and ignored.
    assert default_ledger_path({"XDG_STATE_HOME": "rel", "HOME": "/h"}) == Path(
        "/h/.local/state/steproute/ledger.jsonl"
    )
    assert default_ledger_path({"HOME": "/h"}) == Path("/h/.local/state/steproute/ledger.jsonl")


def test_ledger_location_precedence(tmp_path: Path) -> None:
    env = {LEDGER_ENV: str(tmp_path / "env.jsonl"), "XDG_STATE_HOME": str(tmp_path / "x")}
    flag, conf = tmp_path / "flag.jsonl", tmp_path / "conf.jsonl"
    assert resolve_ledger_path(explicit=flag, configured=conf, environ=env)[0] == flag
    assert resolve_ledger_path(configured=conf, environ=env)[0] == conf
    assert resolve_ledger_path(environ=env)[0] == tmp_path / "env.jsonl"
    path, source = resolve_ledger_path(environ={"XDG_STATE_HOME": str(tmp_path / "x")})
    assert path == tmp_path / "x" / "steproute" / "ledger.jsonl" and "default" in source
    # [ledger] path in the config is relative to the config file
    cfg = parse_replay_config({"ledger": {"path": "state/l.jsonl"}}, base=tmp_path)
    assert cfg.ledger_path == tmp_path / "state" / "l.jsonl"
    with pytest.raises(ValueError, match=r"\[ledger\] takes only path"):
        parse_replay_config({"ledger": {"where": "x"}})


def test_the_conftest_guard_points_the_default_away_from_home() -> None:
    path, _ = resolve_ledger_path(environ=os.environ)
    assert LEDGER_ENV not in os.environ
    assert str(path).startswith(os.environ["XDG_STATE_HOME"])
    assert not path.exists()


# --- one ledger, several writers --------------------------------------------------------------


def test_two_guards_on_one_ledger_see_each_others_reservations(tmp_path: Path) -> None:
    """Two runners (two Ledger and CapGuard objects, as in two processes) on one file: what
    one holds, the other cannot take. Per-directory ledgers would have admitted both."""
    path = tmp_path / "state" / "ledger.jsonl"
    a = CapGuard(Ledger(path, out=tmp_path / "a"), {"anthropic": 10.0}, {})
    b = CapGuard(Ledger(path, out=tmp_path / "b"), {"anthropic": 10.0}, {})
    assert a.admit("anthropic", "ra", 6.0) == 6.0
    with pytest.raises(CapExceeded, match=r"in flight \(every runner on the ledger\) \$6"):
        b.admit("anthropic", "rb", 6.0)
    # spend by A's run is inside its reservation: B's headroom does not change
    led_a = a.ledger
    led_a.settle(led_a.intend("anthropic", 1.0, "ra"), 2.5)
    assert b.headroom("anthropic") == pytest.approx(4.0)
    a.release("ra")
    assert b.headroom("anthropic") == pytest.approx(7.5)
    assert b.admit("anthropic", "rb", 6.0) == 6.0
    # an intent written by another directory counts against the month and the total
    assert Ledger(path).spent("anthropic") == pytest.approx(2.5)


def test_the_total_budget_counts_every_directory(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    pilot = Ledger(path, out=tmp_path / "pilot", mirror=tmp_path / "pilot" / "ledger.jsonl")
    pilot.settle(pilot.intend("deepseek", 1.0, "p1"), 7.0)
    grid = CapGuard(Ledger(path, out=tmp_path / "grid"), {"deepseek": 10.0}, {"deepseek": 10.0})
    assert grid.headroom("deepseek") == pytest.approx(3.0)
    with pytest.raises(CapExceeded, match="total budget"):
        grid.admit("deepseek", "g1", 3.5)
    # the per-directory view: the pilot's own file holds only the pilot's lines
    rows, _ = read_lines(tmp_path / "pilot" / "ledger.jsonl")
    assert [r["kind"] for r in rows] == ["intent", "settle"]
    assert {r["out"] for r in rows} == {str(tmp_path / "pilot")}
    assert Ledger(path).by_out() == {(str(tmp_path / "pilot"), "deepseek"): 7.0}


_HOLDER = """
import sys
from pathlib import Path
from steproute.replay.ledger import CapGuard, Ledger
path, out, what = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
led = Ledger(path, out=out, mirror=out / "ledger.jsonl")
if what == "reserve":
    CapGuard(led, {"deepseek": 10.0}, {"deepseek": 10.0}).admit("deepseek", "dead1", 6.0)
    print("held", flush=True)
else:
    led.intend("deepseek", 2.5, "run-x")  # the call starts; the process dies during it
    print("intent", flush=True)
sys.stdin.readline()  # wait to be killed
"""


def test_a_dead_runners_reservation_is_released_and_a_live_ones_is_not(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    p = _spawn(tmp_path, "holder", _HOLDER, str(path), str(tmp_path / "a"), "reserve")
    try:
        _expect(p, "held", tmp_path, "holder")
        guard = CapGuard(Ledger(path, out=tmp_path / "b"), {"deepseek": 10.0}, {"deepseek": 10.0})
        # alive: its $6 stays held
        assert guard.headroom("deepseek") == pytest.approx(4.0)
        with pytest.raises(CapExceeded):
            guard.admit("deepseek", "b1", 6.0)
    finally:
        p.send_signal(signal.SIGKILL)  # no release line, no cleanup: the kernel drops its lock
        p.wait()
    assert guard.admit("deepseek", "b1", 6.0) == 6.0
    assert [r.run_id for r in guard.reclaimed] == ["dead1"]
    rows, _ = read_lines(path)
    rel = [r for r in rows if r["kind"] == "release" and r["run_id"] == "dead1"]
    assert len(rel) == 1 and "without releasing" in rel[0]["reason"]


def test_a_reservation_whose_owner_cannot_be_checked_stays_held(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    holder = Ledger(path, out=tmp_path / "a")
    CapGuard(holder, {"deepseek": 10.0}).admit("deepseek", "r1", 6.0)
    owner = holder.reservations["r1"].owner
    (holder.owners_dir / f"{owner}.lock").unlink()  # the owner file is gone: in doubt
    other = Ledger(path, out=tmp_path / "b")
    assert other.owner_state(owner) == "unknown"
    assert CapGuard(other, {"deepseek": 10.0}).headroom("deepseek") == pytest.approx(4.0)
    # an owner id that is not one of ours (a hand-edited line) is never taken as dead
    assert other.owner_state("../../etc") == "unknown"


def test_a_runner_killed_between_intent_and_settle_is_charged_at_worst_case(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    out_a = tmp_path / "a"
    p = _spawn(tmp_path, "crash", _HOLDER, str(path), str(out_a), "intent")
    try:
        _expect(p, "intent", tmp_path, "crash")
    finally:
        p.send_signal(signal.SIGKILL)
        p.wait()
    shared = Ledger(path)
    assert shared.spent_total("deepseek") == pytest.approx(2.5)
    assert [i.run_id for i in shared.open_intents()] == ["run-x"]
    assert Ledger(out_a / "ledger.jsonl").spent_total("deepseek") == pytest.approx(2.5)
    guard = CapGuard(Ledger(path, out=tmp_path / "b"), {"deepseek": 10.0}, {"deepseek": 10.0})
    assert guard.headroom("deepseek") == pytest.approx(7.5)


def test_a_crash_between_the_shared_and_the_mirror_write_is_repaired(tmp_path: Path) -> None:
    path = tmp_path / "state" / "ledger.jsonl"
    out = tmp_path / "out"
    led = Ledger(path, out=out, mirror=out / "ledger.jsonl")
    led.settle(led.intend("anthropic", 1.0, "r1"), 0.5)
    lost = led.intend("anthropic", 1.0, "r2")
    led.mirror = None  # the process dies after the shared write of the settle
    led.settle(lost, 0.25)
    assert Ledger(out / "ledger.jsonl").spent("anthropic") == pytest.approx(1.5)
    again = Ledger(path, out=out, mirror=out / "ledger.jsonl")
    assert again.repair_mirror() == 1
    assert Ledger(out / "ledger.jsonl").spent("anthropic") == pytest.approx(0.75)
    assert again.repair_mirror() == 0


# --- bringing an existing directory in --------------------------------------------------------

#: Lines exactly as the runner before issue 60 wrote them (ledger.py at 7ff8441): no ``out``;
#: a settle from before issue 57 without ``period`` or ``list_usd``; an off-peak settle; an
#: intent never settled (a crash); and a torn last line.
_OLD = [
    {
        "v": 1,
        "kind": "intent",
        "id": "i1",
        "ts": 1_794_000_000.0,
        "month": "2026-11",
        "provider": "deepseek",
        "usd": 3.08,
        "run_id": "r1",
    },
    {"v": 1, "kind": "settle", "id": "i1", "ts": 1_794_000_010.0, "usd": 0.04, "flags": []},
    {
        "v": 1,
        "kind": "intent",
        "id": "i2",
        "ts": 1_794_000_020.0,
        "month": "2026-11",
        "provider": "deepseek",
        "usd": 3.08,
        "run_id": "r1",
    },
    {
        "v": 1,
        "kind": "settle",
        "id": "i2",
        "ts": 1_794_000_030.0,
        "usd": 0.02,
        "flags": [],
        "period": "offpeak",
        "list_usd": 0.04,
    },
    {
        "v": 1,
        "kind": "intent",
        "id": "i3",
        "ts": 1_794_000_040.0,
        "month": "2026-11",
        "provider": "deepseek",
        "usd": 3.08,
        "run_id": "r2",
    },
]


def _old_directory(out: Path) -> Path:
    out.mkdir(parents=True)
    local = out / "ledger.jsonl"
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in _OLD)
    local.write_text(text + '{"v":1,"kind":"settle","id":"i3","us', "utf-8")
    return local


def test_import_brings_an_old_directory_in_exactly_once(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    local = _old_directory(out)
    before = local.read_bytes()
    path = tmp_path / "state" / "ledger.jsonl"
    led = Ledger(path)
    assert len(led.unknown_spend(local)) == 3
    res = led.import_from(local, out=out)
    assert (res.intents, res.settles, res.already) == (3, 2, 0)
    assert local.read_bytes() == before  # the directory's own file is not touched
    for again in (Ledger(path), led):
        assert again.unknown_spend(local) == []
        # the worst case of the open intent, the realised cost of the others
        assert again.spent_total("deepseek") == pytest.approx(0.04 + 0.02 + 3.08)
        assert again.spent("deepseek", "2026-11") == pytest.approx(3.14)
        assert again.by_period() == {
            ("deepseek", "unknown"): (0.04, 0.04),
            ("deepseek", "offpeak"): (0.02, 0.04),
        }
    res2 = Ledger(path).import_from(local, out=out)
    assert (res2.intents, res2.settles, res2.already) == (0, 0, 5)
    assert Ledger(path).spent_total("deepseek") == pytest.approx(3.14)
    rows, _ = read_lines(path)
    assert sum(1 for r in rows if r["kind"] == "import") == 1
    assert {r["out"] for r in rows if r["kind"] == "intent"} == {str(out)}


def test_import_adds_a_settle_the_shared_ledger_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "pilot"
    local = _old_directory(out)
    path = tmp_path / "state" / "ledger.jsonl"
    Ledger(path).import_from(local, out=out)
    with open(local, "ab") as f:  # the old runner settled i3 later (after a torn line)
        f.write(
            b"\n"
            + json.dumps(
                {"v": 1, "kind": "settle", "id": "i3", "ts": 1.0, "usd": 0.5, "flags": []}
            ).encode()
            + b"\n"
        )
    res = Ledger(path).import_from(local, out=out)
    assert (res.intents, res.settles) == (0, 1)
    assert Ledger(path).spent_total("deepseek") == pytest.approx(0.06 + 0.5)


def test_cli_import_status_and_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "pilot"
    _old_directory(out)
    path = tmp_path / "state" / "ledger.jsonl"
    assert main(["replay", "status", "--out", str(out), "--ledger", str(path)]) == 0
    text = capsys.readouterr().out
    assert "not created yet" in text and "steproute replay ledger import --out" in text
    assert main(["replay", "ledger", "import", "--out", str(out), "--ledger", str(path)]) == 0
    text = capsys.readouterr().out
    assert "3 call(s) and 2 settlement(s)" in text and "every call" in text
    assert main(["replay", "ledger", "import", "--out", str(out), "--ledger", str(path)]) == 0
    assert "0 call(s) and 0 settlement(s); 5 line(s) were already there" in (
        capsys.readouterr().out
    )
    assert main(["replay", "ledger", "show", "--ledger", str(path)]) == 0
    text = capsys.readouterr().out
    assert "spend deepseek   all time  $3.1400" in text and str(out) in text
    assert "1 unsettled call(s)" in text
    assert main(["replay", "status", "--out", str(out), "--ledger", str(path)]) == 0
    assert "does not know" not in capsys.readouterr().out


# --- the runner and the CLI -------------------------------------------------------------------


def test_plan_shows_spend_from_the_shared_ledger(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "state" / "ledger.jsonl"
    other = Ledger(path, out=tmp_path / "earlier")  # spent by a run in another directory
    other.settle(other.intend("deepseek", 1.0, "e1"), 2.5)
    other.settle(other.intend("anthropic", 1.0, "e2"), 1.25)
    month = time.strftime("%Y-%m", time.gmtime())
    out = tmp_path / "new"
    monkeypatch.setenv(LEDGER_ENV, str(path))  # also through the environment variable
    assert main(["replay", "plan", "--plan", "pilot", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert f"spend ledger: {path} (${LEDGER_ENV})" in text
    assert f"month-to-date {month} $1.25 (all directories)" in text
    assert f"month-to-date {month} $2.50 (all directories)" in text
    assert "spent so far $2.50, left $7.50" in text
    assert not out.exists()
    # the config's [ledger] path wins over the environment
    cfg = tmp_path / "replay.toml"
    cfg.write_text('[ledger]\npath = "elsewhere/ledger.jsonl"\n', "utf-8")
    assert main(["replay", "plan", "--plan", "pilot", "--out", str(out), "--config", str(cfg)]) == 0
    text = capsys.readouterr().out
    assert "elsewhere" in text and "not created yet: nothing spent" in text
    assert not (tmp_path / "elsewhere").exists()  # a dry run creates nothing


def test_a_real_run_refuses_a_directory_with_unknown_spend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from replay_harness import MANIFEST, build, runner
    from steproute.replay.config import ReplayConfig

    out = tmp_path / "pilot"
    _old_directory(out)
    plan, manifest = build(tmp_path, [("qwen3-coder:30b", "medium")])
    cfg = ReplayConfig(environment="local", manage_ollama=False)
    path = tmp_path / "state" / "ledger.jsonl"
    buf = io.StringIO()
    assert runner(plan, manifest, cfg, out, environ={}, stdout=buf, ledger_path=path).run() == 0
    assert "a real run would stop here" in buf.getvalue()
    assert "steproute replay ledger import --out" in buf.getvalue()
    with pytest.raises(UnknownSpend, match="does not know"):
        runner(plan, manifest, cfg, out, environ={}, ledger_path=path).run(execute=True)
    assert not (out / "runs.jsonl").exists()  # nothing ran
    # the CLI says the same, exit 2
    toml = tmp_path / "replay.toml"
    toml.write_text('[environment]\nkind = "local"\nmanage_ollama = false\n', "utf-8")
    args = ["--plan", str(tmp_path / "smoke.toml"), "--manifest", str(MANIFEST)]
    args += ["--config", str(toml), "--out", str(out), "--ledger", str(path), "--execute"]
    assert main(["replay", "run", *args]) == 2
    err = capsys.readouterr().err
    assert "does not know" in err and "steproute replay ledger import --out" in err
    assert not (out / "runs.jsonl").exists()
    # preflight names it too
    from steproute.replay.preflight import _ledger_check

    assert not _ledger_check(path, out).ok
    Ledger(path).import_from(out / "ledger.jsonl", out=out)
    assert _ledger_check(path, out).ok


# --- the acceptance test: two runners, two directories, one budget -----------------------------

_RUNNER = """
import io, json, sys
from pathlib import Path
from types import SimpleNamespace

from replay_harness import build, config_for, environ_with, runner
from steproute.replay import ledger as ledger_mod
from steproute.replay.config import Limits

tmp, out, ledger, base, abase, provider, limit = sys.argv[1:8]
mock = SimpleNamespace(base_url=base, anthropic_base_url=abase)
model = "claude-haiku-5-5" if provider == "anthropic" else "deepseek-flash"
limits = Limits(call_backoff_s=0.0, cache_canary_runs=0, run_cap_usd=0.01,
                min_run_budget_usd=0.001, max_prompt_tokens=20_000, max_output_tokens=1_000)
if provider == "anthropic":
    cfg = config_for(mock, limits=limits, caps={"anthropic": float(limit), "deepseek": 10.0})
else:
    cfg = config_for(mock, limits=limits, caps={"anthropic": 80.0, "deepseek": 10.0},
                     totals={"deepseek": float(limit)})
plan, manifest = build(Path(tmp), [(model, "low")])

orig = ledger_mod.CapGuard.admit
gate = {"open": False}

def admit(self, *a, **kw):
    if not gate["open"]:  # the first admission of this runner waits for the start signal
        print("ready", flush=True)
        assert sys.stdin.readline().strip() == "go"
        gate["open"] = True
    return orig(self, *a, **kw)

ledger_mod.CapGuard.admit = admit
buf = io.StringIO()
code = runner(plan, manifest, cfg, Path(out), environ=environ_with(provider), stdout=buf,
              ledger_path=Path(ledger)).run(execute=True)
print(json.dumps({"code": code, "text": buf.getvalue()[-4000:]}), flush=True)
"""


def _never_jointly_passed(path: Path, provider: str, limit: float) -> float:
    """Replays the shared ledger line by line: spend (open intents at their worst case) plus
    what every live reservation could still spend never passes ``limit``."""
    rows, bad = read_lines(path)
    assert bad == 0
    charged: dict[str, tuple[str, float]] = {}  # intent id -> (run id, usd)
    held: dict[str, float] = {}
    peak = 0.0
    for r in rows:
        if r["kind"] == "intent" and r["provider"] == provider:
            charged[r["id"]] = (r["run_id"], r["usd"])
        elif r["kind"] == "settle" and r["id"] in charged:
            charged[r["id"]] = (charged[r["id"]][0], r["usd"])
        elif r["kind"] == "reserve" and r["provider"] == provider:
            held[r["run_id"]] = r["usd"]
        elif r["kind"] == "release":
            held.pop(r["run_id"], None)
        by_run: dict[str, float] = {}
        for run_id, usd in charged.values():
            by_run[run_id] = by_run.get(run_id, 0.0) + usd
        spent = sum(by_run.values())
        outstanding = sum(max(0.0, usd - by_run.get(rid, 0.0)) for rid, usd in held.items())
        peak = max(peak, spent + outstanding)
    assert peak <= limit + 1e-9, f"spend + reservations reached ${peak:.4f} > ${limit}"
    return sum(usd for _, usd in charged.values())


@pytest.mark.parametrize("provider", ["anthropic", "deepseek"])
def test_two_runners_in_two_directories_cannot_jointly_pass_a_cap_or_a_total(
    tmp_path: Path, provider: str
) -> None:
    """Two runner processes, each in its own output directory with a plan that alone fits
    the limit (three cells of about $0.005), start at once on one shared ledger whose limit
    is $0.02: the monthly cap (Anthropic) or the total budget (DeepSeek). Without the shared
    ledger each would have counted from $0 and together they would have spent about $0.03."""
    pytest.importorskip("litellm")
    from replay_harness import ScriptedAgent
    from steproute.collectors.mock_upstream import MockUpstream

    limit = 0.02
    ledger = tmp_path / "state" / "ledger.jsonl"
    with MockUpstream() as mock:
        mock.set_responder(ScriptedAgent(extra_steps=20))
        procs = {}
        for name in ("one", "two"):
            work = tmp_path / name
            work.mkdir()
            procs[name] = _spawn(
                tmp_path,
                f"runner-{name}",
                _RUNNER,
                str(work),
                str(work / "out"),
                str(ledger),
                mock.base_url,
                mock.anthropic_base_url,
                provider,
                str(limit),
            )
        try:
            for name, p in procs.items():  # both are at their first admission ...
                _expect(p, "ready", tmp_path, f"runner-{name}")
            for p in procs.values():  # ... and both go
                assert p.stdin is not None
                p.stdin.write("go\n")
                p.stdin.flush()
            results = {}
            for name, p in procs.items():
                assert p.stdout is not None
                line = p.stdout.readline()
                assert p.wait(timeout=600) == 0, _err(tmp_path, f"runner-{name}")
                results[name] = json.loads(line)
        finally:
            for p in procs.values():
                if p.poll() is None:
                    p.kill()
                    p.wait()
    spent = _never_jointly_passed(ledger, provider, limit)
    assert 0 < spent <= limit
    # both directories keep their own view, and together they are the shared ledger
    per_dir = [Ledger(tmp_path / n / "out" / "ledger.jsonl").spent_total(provider) for n in procs]
    assert sum(per_dir) == pytest.approx(spent)
    assert all(d > 0 for d in per_dir) or any(r["code"] == 4 for r in results.values())
    # the budget stopped someone: six cells cannot all fit in $0.02
    assert any(r["code"] == 4 for r in results.values()), results
    # no reservation is left behind
    shared = Ledger(ledger)
    assert shared.live_reservations() == []
