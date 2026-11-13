"""Pre-flight: what is missing before a real run. Read-only; changes nothing, spends nothing.

Checks: the Docker daemon, the task images, the SWE-bench harness and dataset rows, each
paid provider's key (set or not; the value is never read beyond "non-empty"), Ollama and its
models, and free disk. No provider is called: a key's validity cannot be checked for free.
Standard library only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from steproute.replay.config import ReplayConfig
from steproute.replay.plan import ReplayPlan
from steproute.tasks.check import workspace_env
from steproute.tasks.manifest import TaskManifest

__all__ = ["Check", "MIN_FREE_GB", "preflight"]

#: The SWE-bench docs ask for 120 GB free (docs/TASKSET.md section 8).
MIN_FREE_GB = 120.0

Exec = Callable[[Sequence[str], float], tuple[int, str]]
Fetch = Callable[[str, float], str]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _exec(cmd: Sequence[str], timeout: float) -> tuple[int, str]:
    try:
        r = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, env=workspace_env()
        )
    except FileNotFoundError:
        return 127, "not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"
    return r.returncode, (r.stdout + r.stderr).strip()


def _fetch(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return str(r.read().decode())


def preflight(
    plan: ReplayPlan,
    manifest: TaskManifest,
    config: ReplayConfig,
    *,
    environ: Mapping[str, str],
    out: Path,
    run: Exec = _exec,
    fetch: Fetch = _fetch,
    disk_free_gb: Callable[[Path], float] | None = None,
) -> list[Check]:
    checks: list[Check] = []
    tasks = {s.task_id for s in plan.specs}
    swebench = [t for t in manifest if t.id in tasks and t.backend == "swebench"]

    for prov in plan.paid_providers():
        env = config.provider(prov).api_key_env
        ok = bool(env and environ.get(env))
        checks.append(
            Check(
                f"key {prov}",
                ok,
                f"{env} is {'set' if ok else 'NOT set'} (value not shown, validity not checked)",
            )
        )

    if swebench or config.environment == "docker":
        code, text = run([config.docker, "info", "--format", "{{.ServerVersion}}"], 20)
        docker_ok = code == 0
        checks.append(
            Check(
                "docker daemon",
                docker_ok,
                f"server {text}" if docker_ok else f"not reachable ({text[:120]})",
            )
        )
        if docker_ok:
            missing = []
            for t in swebench:
                if t.image:
                    c, _ = run([config.docker, "image", "inspect", t.image], 20)
                    if c != 0:
                        missing.append(t.image)
            checks.append(
                Check(
                    "task images",
                    not missing,
                    f"all {len(swebench)} present"
                    if not missing
                    else f"{len(missing)} of {len(swebench)} not pulled, e.g. {missing[0]}",
                )
            )
        else:
            checks.append(Check("task images", False, "cannot check without the daemon"))
    if swebench:
        rows = config.rows
        rows_ok = rows is not None and rows.exists()
        checks.append(
            Check(
                "dataset rows",
                rows_ok,
                str(rows) if rows_ok else "[check] rows missing; run `steproute tasks fetch`",
            )
        )
        code, text = run([*config.harness, "--help"], 60)
        checks.append(
            Check(
                "swebench harness",
                code == 0,
                " ".join(config.harness) if code == 0 else f"not runnable ({text[:120]})",
            )
        )

    local_models = sorted({s.model for s in plan.specs if s.local})
    if local_models:
        base = (config.provider("ollama").base_url or "").rstrip("/").removesuffix("/v1")
        try:
            tags = json.loads(fetch(f"{base}/api/tags", 5))
            have = {str(m.get("name")) for m in tags.get("models", [])}
            missing_m = [m for m in local_models if m not in have]
            checks.append(Check("ollama", True, f"reachable at {base}"))
            checks.append(
                Check(
                    "ollama models",
                    not missing_m,
                    "all pulled: " + ", ".join(local_models)
                    if not missing_m
                    else "not pulled: " + ", ".join(missing_m),
                )
            )
        except (OSError, ValueError) as exc:
            checks.append(Check("ollama", False, f"not reachable at {base} ({type(exc).__name__})"))
        checks.append(
            Check(
                "ollama settings",
                True,
                "cannot be read from outside: start it with OLLAMA_CONTEXT_LENGTH="
                f"{config.limits.local_context_cap} OLLAMA_MAX_LOADED_MODELS=1",
                required=False,
            )
        )

    gb = (disk_free_gb or _free_gb)(out)
    need = MIN_FREE_GB if swebench else 1.0
    checks.append(Check("free disk", gb >= need, f"{gb:.0f} GB free at {out} (need {need:.0f} GB)"))
    if shutil.which("git") is None:
        checks.append(Check("git", False, "git is not on PATH"))
    return checks


def _free_gb(path: Path) -> float:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9
