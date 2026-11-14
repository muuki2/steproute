"""Test harness for the replay runner: a scripted mock agent on the loopback mock upstream.

The mock "model" is a responder that reads the conversation it is sent: the TASK-KEY in the
task message picks the fix, and the number of assistant turns so far picks the step:

0. write calc.py (right fix for keys a and b, wrong fix for c);
1. ``git diff -- calc.py > patch.txt``;
2. submit (``echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt``).

Usage is scripted to grow per step, with cached reads after the first step, as a real
provider would report. Every model is the mock on 127.0.0.1; no other connection is opened.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from steproute.collectors.mock_upstream import MockUpstream, Received, Reply
from steproute.core.catalog import load_default_catalog
from steproute.replay.config import Limits, ProviderConfig, ReplayConfig, Schedule
from steproute.replay.plan import ReplayPlan, load_plan_file
from steproute.replay.runner import Runner, RunnerDeps
from steproute.tasks.estimate import load_assumptions
from steproute.tasks.manifest import TaskManifest, load_manifest

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MANIFEST = FIXTURES / "manifest.toml"

GOOD = "    return a + b"
BAD = "    return a * b"


def fake_key(name: str) -> str:
    """A fake key built at run time (no credential-shaped literal in the repository)."""
    return "-".join(["fake", name, "replay", "x" * 12])


def write_command(body_line: str) -> str:
    lines = [
        '"""A tiny task for the local check backend: add() has a bug."""',
        "",
        "",
        "def add(a, b):",
        body_line,
    ]
    quoted = " ".join("'" + x.replace("'", "'\\''") + "'" for x in lines)
    return f"printf '%s\\n' {quoted} > calc.py"


def _assistant_turns(body: dict[str, Any]) -> int:
    return sum(
        1 for m in body.get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant"
    )


@dataclass
class ScriptedAgent:
    """The responder. ``overrides`` maps a task key to a function returning a reply (or
    ``None`` to use the default script) for a given step."""

    overrides: dict[str, Callable[[int, Received], Reply | None]] = field(default_factory=dict)
    delay_s: float = 0.0
    extra_steps: int = 0  # harmless `ls` steps before the fix (longer runs)
    usage: bool = True
    scale: int = 1  # multiplies the reported prompt sizes (expensive runs)
    seen: list[tuple[str, int]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, rec: Received) -> Reply | None:
        text = json.dumps(rec.body)
        m = re.search(r"TASK-KEY: (\w)", text)
        key = m.group(1) if m else "?"
        step = _assistant_turns(rec.body)
        with self.lock:
            self.seen.append((key, step))
        if self.delay_s:
            time.sleep(self.delay_s)
        if key in self.overrides:
            r = self.overrides[key](step, rec)
            if r is not None:
                return r
        prompt = (2000 + 600 * step) * self.scale
        cached = 0 if step == 0 else (2000 + 600 * (step - 1)) * self.scale
        common = {
            "prompt_tokens": prompt,
            "completion_tokens": 120,
            "cached_tokens": cached,
            "cache_write_tokens": prompt - cached,
            "usage": self.usage,
        }
        s = step - self.extra_steps
        if s < 0:
            cmd = "ls"
        elif s == 0:
            cmd = write_command(BAD if key == "c" else GOOD)
        elif s == 1:
            cmd = "git diff -- calc.py > patch.txt"
        else:
            cmd = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt"
        return Reply(
            kind="tool",
            tool_name="bash",
            tool_args=json.dumps({"command": cmd}),
            tool_text="THOUGHT: next step.",
            **common,  # type: ignore[arg-type]
        )


def config_for(
    mock: MockUpstream,
    *,
    caps: dict[str, float] | None = None,
    limits: Limits | None = None,
    paid: int = 1,
    totals: dict[str, float] | None = None,
    schedules: dict[str, Schedule] | None = None,
) -> ReplayConfig:
    """A config on the mock. Tests that are not about the off-peak schedule run DeepSeek at
    any hour (``schedules = {}``); the schedule's own tests pass the default explicitly."""
    return ReplayConfig(
        caps=caps or {"anthropic": 80.0, "deepseek": 10.0},
        totals=totals if totals is not None else {"deepseek": 10.0},
        schedules=schedules if schedules is not None else {},
        limits=limits or Limits(call_backoff_s=0.0, cache_canary_runs=0),
        providers={
            "anthropic": ProviderConfig(
                "anthropic", "STEPROUTE_TEST_ANTHROPIC", mock.anthropic_base_url
            ),
            "deepseek": ProviderConfig("deepseek", "STEPROUTE_TEST_DEEPSEEK", mock.base_url),
            "ollama": ProviderConfig("ollama", None, mock.base_url),
        },
        paid_concurrency=paid,
        environment="local",
        manage_ollama=False,
    )


def environ_with(*providers: str) -> dict[str, str]:
    env = {}
    if "anthropic" in providers:
        env["STEPROUTE_TEST_ANTHROPIC"] = fake_key("anthropic")
    if "deepseek" in providers:
        env["STEPROUTE_TEST_DEEPSEEK"] = fake_key("deepseek")
    return env


def plan_file(
    tmp: Path, configs: list[tuple[str, str]], tasks: list[str] | None = None, name: str = "smoke"
) -> Path:
    p = tmp / f"{name}.toml"
    t = json.dumps(tasks) if tasks else '"all"'
    cfg = json.dumps([list(c) for c in configs])
    p.write_text(f'name = "{name}"\n[[block]]\ntasks = {t}\nconfigs = {cfg}\n', "utf-8")
    return p


def build(
    tmp: Path,
    configs: list[tuple[str, str]],
    *,
    tasks: list[str] | None = None,
) -> tuple[ReplayPlan, TaskManifest]:
    manifest = load_manifest(MANIFEST)
    plan = load_plan_file(
        plan_file(tmp, configs, tasks), manifest, load_default_catalog(), load_assumptions()
    )
    return plan, manifest


def runner(
    plan: ReplayPlan,
    manifest: TaskManifest,
    config: ReplayConfig,
    out: Path,
    *,
    environ: dict[str, str],
    stdout: Any = None,
    **deps: Any,
) -> Runner:
    import io

    return Runner(
        plan,
        config=config,
        catalog=load_default_catalog(),
        manifest=manifest,
        out=out,
        deps=RunnerDeps(environ=environ, out=stdout or io.StringIO(), sleep=lambda s: None, **deps),
    )
