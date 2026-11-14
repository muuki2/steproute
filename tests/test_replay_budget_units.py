"""Issue 57 pieces that need no LiteLLM: totals and reduced budgets in the guard, the replay
config's [totals] / [schedule] / floor, and the --env-file parser. Fake values only."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from steproute.cli import main
from steproute.replay.config import ReplayConfigError, Schedule, parse_replay_config
from steproute.replay.envfile import EnvFileError, apply_env, load_env_file, parse_env
from steproute.replay.journal import DEFERRED, Journal
from steproute.replay.ledger import CapExceeded, CapGuard, Ledger


def _fake(name: str) -> str:
    return "-".join(["fake", name, "value", "k" * 10])


# --- the guard -----------------------------------------------------------------------------


def test_a_total_budget_counts_every_month(tmp_path: Path) -> None:
    t = {"now": 1_793_000_000.0}  # some day in October 2026
    led = Ledger(tmp_path / "ledger.jsonl", clock=lambda: t["now"])
    led.settle(led.intend("deepseek", 5.0, "r1"), 6.0)
    t["now"] += 40 * 86400  # next month: the monthly cap is fresh, the total is not
    assert led.spent("deepseek") == 0.0 and led.spent_total("deepseek") == 6.0
    guard = CapGuard(led, {"deepseek": 10.0}, {"deepseek": 10.0})
    assert guard.headroom("deepseek") == pytest.approx(4.0)
    with pytest.raises(CapExceeded, match="all-time .* total budget"):
        guard.admit("deepseek", "r2", 4.5)
    without_total = CapGuard(led, {"deepseek": 10.0})
    assert without_total.admit("deepseek", "r3", 4.5) == 4.5


def test_admit_with_a_floor_grants_the_headroom_left(tmp_path: Path) -> None:
    led = Ledger(tmp_path / "ledger.jsonl")
    guard = CapGuard(led, {"deepseek": 10.0}, {"deepseek": 10.0})
    led.settle(led.intend("deepseek", 1.0, "old"), 8.0)
    # 2.0 left: a 3.08 bound does not fit; with a floor of 0.58 it gets exactly 2.0.
    assert guard.admit("deepseek", "a", 3.08, least=0.58) == pytest.approx(2.0)
    assert guard.headroom("deepseek") == pytest.approx(0.0)
    with pytest.raises(CapExceeded, match="smallest budget"):
        guard.admit("deepseek", "b", 3.08, least=0.58)
    guard.release("a")
    led.settle(led.intend("deepseek", 1.0, "c"), 1.5)  # 0.5 left: below the floor
    with pytest.raises(CapExceeded):
        guard.admit("deepseek", "d", 3.08, least=0.58)
    assert guard.admit("deepseek", "e", 0.4, least=0.58) == 0.4  # a full bound that fits


def test_deferred_attempts_do_not_finish_a_cell(tmp_path: Path) -> None:
    j = Journal.open(tmp_path / "runs.jsonl")
    cell = {"task_id": "t", "model": "deepseek-flash", "effort": "low", "repeat": 0}
    for i, status in enumerate(DEFERRED):
        j.record_start({"run_id": f"r{i}", "cell": cell, "attempt": i + 1, "session_id": "s"})
        j.record_end(f"r{i}", {"status": status, "success": None})
    assert list(j.finished()) == []
    assert not any(a.terminal for a in j.attempts.values())


# --- config --------------------------------------------------------------------------------


def test_totals_schedule_and_floor_in_the_config() -> None:
    cfg = parse_replay_config(
        {
            "totals": {"deepseek": 6.0},
            "schedule": {"deepseek": {"start_margin_s": 1800, "call_margin_s": 120}},
            "limits": {"min_run_budget_usd": 0.25},
        }
    )
    assert cfg.total("deepseek") == 6.0
    assert cfg.schedule("deepseek") == Schedule(True, 1800.0, 120.0)
    assert cfg.start_margin("deepseek") == 1800.0
    assert cfg.limits.min_run_budget_usd == 0.25
    off = parse_replay_config({"schedule": {"deepseek": {"offpeak_only": False}}})
    assert not off.schedule("deepseek").offpeak_only
    with pytest.raises(ReplayConfigError, match="allow_raise"):
        parse_replay_config({"totals": {"deepseek": 20.0}})
    assert (
        parse_replay_config({"totals": {"deepseek": 20.0, "allow_raise": True}}).total("deepseek")
        == 20.0
    )
    with pytest.raises(ReplayConfigError, match="min_run_budget_usd"):
        parse_replay_config({"limits": {"min_run_budget_usd": 4.0}})
    with pytest.raises(ReplayConfigError, match="unknown"):
        parse_replay_config({"schedule": {"deepseek": {"peak_ok": True}}})
    with pytest.raises(ReplayConfigError):
        parse_replay_config({"totals": {"ollama": 1.0}})


# --- --env-file ------------------------------------------------------------------------------


def test_env_file_format() -> None:
    a, b, c, d = _fake("a"), _fake("b"), _fake("c"), _fake("d")
    text = (
        "# a comment\n"
        "\n"
        f"FIRST={a}\n"
        f"export SECOND='{b}'\n"
        f'  THIRD="{c}"  \n'
        "INTERP=$FIRST\n"
        "EMPTY=\n"
        f"LAST={d}"  # no final newline
    )
    assert parse_env(text) == {
        "FIRST": a,
        "SECOND": b,
        "THIRD": c,
        "INTERP": "$FIRST",  # no interpolation
        "EMPTY": "",
        "LAST": d,
    }


@pytest.mark.parametrize(
    "line", ["NO_EQUALS_SIGN", "1BAD=x", "BAD NAME=x", "QUOTE='unbalanced", "export"]
)
def test_a_malformed_line_names_the_line_number_only(line: str) -> None:
    secret = _fake("hidden")
    text = f"OK={secret}\n# comment\n{line}{secret}\n"
    with pytest.raises(EnvFileError) as exc:
        parse_env(text)
    assert "line 3" in str(exc.value)
    assert secret not in str(exc.value) and line not in str(exc.value)


def test_an_unreadable_env_file_is_an_error_without_content(tmp_path: Path) -> None:
    with pytest.raises(EnvFileError, match="cannot read"):
        load_env_file(tmp_path / "missing.env")


def test_a_variable_already_set_wins() -> None:
    env = {"KEEP": "from-the-environment"}
    used = apply_env({"KEEP": _fake("file"), "NEW": _fake("new")}, env)
    assert env == {"KEEP": "from-the-environment", "NEW": _fake("new")}
    assert used == [_fake("new")]


def test_env_file_on_the_cli_sets_the_key_without_showing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    value = _fake("deepseek")
    env_file = tmp_path / "keys.env"
    env_file.write_text(f"export DEEPSEEK_API_KEY='{value}'\n", "utf-8")
    plan = tmp_path / "p.toml"
    plan.write_text(
        'name = "p"\n[[block]]\ntasks = ["astropy__astropy-12907"]\n'
        'configs = [["deepseek-flash", "low"]]\n'
    )
    try:
        # a dry run (it prints the key status); preflight would also probe Docker
        main(
            [
                "replay",
                "run",
                "--plan",
                str(plan),
                "--out",
                str(tmp_path / "out"),
                "--env-file",
                str(env_file),
            ]
        )
        out = capsys.readouterr()
        assert "DEEPSEEK_API_KEY: set" in out.out and "DRY RUN" in out.out
        assert value not in out.out and value not in out.err
    finally:
        os.environ.pop("DEEPSEEK_API_KEY", None)
    bad = tmp_path / "bad.env"
    bad.write_text(f"{value}\n", "utf-8")
    for cmd in ("run", "preflight"):  # the file is read before anything else is checked
        rc = main(
            [
                "replay",
                cmd,
                "--plan",
                str(plan),
                "--out",
                str(tmp_path / "o2"),
                "--env-file",
                str(bad),
            ]
        )
        err = capsys.readouterr()
        assert rc == 2 and "line 1" in err.err
        assert value not in err.out and value not in err.err
