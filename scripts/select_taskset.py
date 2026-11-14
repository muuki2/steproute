"""Draw the issue-17 task subset from SWE-bench Verified, reproducibly.

    uv run python scripts/select_taskset.py --cache .cache/taskset          # fetch, select, print
    uv run python scripts/select_taskset.py --cache .cache/taskset --write  # and write the manifest

Inputs (all public, all small, fetched once into --cache; about 10 MB in total):

* SWE-bench Verified rows from the Hugging Face datasets-server (500 rows, about 9 MB of JSON).
  Only instance_id, repo, difficulty, the problem statement length and the gold patch size are
  used. Nothing from the rows is written to the repository except the instance ids.
* The names of the native arm64 evaluation images in the ``swebench`` Docker Hub namespace, and
  for every Verified id whether ``ghcr.io/epoch-research/swe-bench.eval.arm64.<id>`` exists
  (one anonymous manifest request per id; nothing is pulled).
* Per-instance resolved flags and API-call counts of public submissions in the
  SWE-bench/experiments repository, pinned to one commit (``EXPERIMENTS_COMMIT``).

The public results are used only as an estimate of difficulty for our pool: a task solved by
older or weaker models is likely easy for ours, and one solved only by frontier models is likely
hard. That is a guess about our pool, not a measurement of it; the pilot measures it.

Selection (docs/TASKSET.md, "How the subset is drawn"):

1. Universe: Verified instances with a native arm64 image (official ``swebench/sweb.eval.arm64.*``
   preferred, else Epoch's), minus pylint (GPL-2.0, so its code in traces is copyleft).
2. Stratum per task from the panel: easy, medium, hard, unsolved.
3. Within each stratum, allocate across repositories in proportion to the square root of the
   repository's count (so django does not dominate), then sample with ``random.Random(SEED)``.
4. The pilot is drawn from the full set, stratum by stratum, preferring tasks whose estimated
   context fits the 32K local cap for the easy and medium strata.
5. Broken tasks (``BROKEN``, issue 58): a task whose gold patch fails on its arm64 image can
   never be resolved. The draw of steps 3 and 4 is made first, exactly as before, so no other
   task moves. Each drawn task listed in ``BROKEN`` is then replaced, taking its pilot slot if
   it had one, by the first candidate of ``replacement_order``: same stratum, not drawn, not
   broken, not from a repository whose tests reach the network (``NETWORK_REPOS``), in an order
   fixed by ``random.Random(f"{SEED}:replace:{broken id}")``. A candidate that then fails its
   own check (gold must pass, baseline must fail) is added to ``BROKEN`` and the next one is
   taken. The manifest keeps a record of every broken task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 17
DATASET = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"  # HF main, 2026-08-16
EXPERIMENTS_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"  # 2026-09-03
MANIFEST = Path("src/steproute/tasks/manifests/swebench-verified-v1.toml")

FULL_SIZE = {"easy": 40, "medium": 45, "hard": 50, "unsolved": 15}  # 150
PILOT_SIZE = {"easy": 5, "medium": 6, "hard": 7, "unsolved": 2}  # 20

# Public submissions used as the difficulty panel (evaluation/verified/<name>).
# devstral results are deliberately left out: devstral:24b is our held-out model (decision D2).
WEAK = [
    "20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct",
    "20250807_mini-v1.7.0_gpt-oss-120b",
    "20250807_mini-v1.7.0_gpt-5-nano",
    "20250805_openhands-Qwen3-Coder-30B-A3B-Instruct",
]
MID = ["20260217_mini-v2.0.0_claude-4-5-haiku-high"]
STRONG = [
    "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "20260217_mini-v2.0.0_claude-4-6-opus",
    "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
    "20260217_mini-v2.0.0_gpt-5-2-high",
    "20260217_mini-v2.0.0_gemini-3-flash-high",
    "20260217_mini-v2.0.0_minimax-2-5-high",
    "20260217_mini-v2.0.0_glm-5-high",
    "20260217_mini-v2.0.0_kimi-k2-5-high",
    "20260217_mini-v2.0.0_deepseek-3-2-high",
]
# Analogs whose per-instance step counts feed the cost assumptions (docs/TASKSET.md).
STEP_ANALOGS = {
    "claude-haiku-5-5": "20260217_mini-v2.0.0_claude-4-5-haiku-high",
    "claude-sonnet-5-5": "20260217_mini-v2.0.0_claude-4-5-sonnet-high",
    "claude-opus-5-5": "20260217_mini-v2.0.0_claude-4-5-opus-high",
    "claude-fable-5-2": "20260217_mini-v2.0.0_claude-4-6-opus",
    "deepseek": "20260217_mini-v2.0.0_deepseek-3-2-high",
    "local": "20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct",
}

# Context estimate for the 32K local cap: measured in public mini-SWE-agent trajectories
# (docs/TASKSET.md, cost model): about 2,000 tokens of prompt, then about 550 per step.
CTX_BASE = 2000
CTX_PER_STEP = 550
LOCAL_CAP = 32768

ARM64_PREFIXES = [
    "astropy",
    "sympy",
    "sphinx",
    "matplotlib",
    "scikit",
    "pydata",
    "pytest",
    "pylint",
    "psf",
    "mwaskom",
    "pallets",
] + [f"django_1776_django-{d}" for d in ("9", "10", "11", "12", "13", "14", "15", "16", "17")]


def curl_json(url: str, dest: Path) -> Any:
    """GET a URL into dest once (curl: the system Python may lack CA certificates)."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", url, "-o", str(dest)], check=True)
    return json.loads(dest.read_text())


def load_rows(cache: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, 500, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset=SWE-bench%2FSWE-bench_Verified"
            f"&config=default&split=test&offset={offset}&length=100"
        )
        data = curl_json(url, cache / f"verified_rows_{offset}.json")
        rows += [r["row"] for r in data["rows"]]
    if len(rows) != 500 or len({r["instance_id"] for r in rows}) != 500:
        sys.exit("expected 500 distinct Verified rows")
    return rows


def load_arm64(cache: Path) -> set[str]:
    ids: set[str] = set()
    for pre in ARM64_PREFIXES:
        url = (
            "https://hub.docker.com/v2/repositories/swebench/"
            f"?page_size=100&name=sweb.eval.arm64.{pre}"
        )
        data = curl_json(url, cache / "arm64" / f"{pre}.json")
        if data.get("count", 0) > 100:
            sys.exit(f"prefix {pre} has more than 100 images; split it")
        for r in data.get("results", []):
            name = r["name"]
            if name.startswith("sweb.eval.arm64."):
                ids.add(name[len("sweb.eval.arm64.") :].replace("_1776_", "__"))
    return ids


_OCI_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json"
)


def _epoch_has(iid: str) -> bool:
    repo = f"epoch-research/swe-bench.eval.arm64.{iid}"
    tok = subprocess.run(
        ["curl", "-sf", "-m", "30", f"https://ghcr.io/token?scope=repository:{repo}:pull"],
        capture_output=True,
    )
    if tok.returncode != 0:
        return False
    token = json.loads(tok.stdout)["token"]
    head = subprocess.run(
        ["curl", "-s", "-m", "30", "-o", "/dev/null", "-w", "%{http_code}", "-I"]
        + ["-H", f"Authorization: Bearer {token}", "-H", f"Accept: {_OCI_ACCEPT}"]
        + [f"https://ghcr.io/v2/{repo}/manifests/latest"],
        capture_output=True,
        text=True,
    )
    return head.stdout.strip() == "200"


def load_epoch_arm64(cache: Path, ids: list[str]) -> set[str]:
    """Verified ids with an Epoch arm64 image, cached in ``epoch_arm64.json``."""
    from concurrent.futures import ThreadPoolExecutor

    dest = cache / "epoch_arm64.json"
    if not dest.exists():
        with ThreadPoolExecutor(16) as ex:
            present = dict(zip(ids, ex.map(_epoch_has, ids), strict=True))
        dest.write_text(json.dumps(present, indent=0, sort_keys=True))
    present = json.loads(dest.read_text())
    return {k for k, v in present.items() if v}


def image_name(iid: str, official: set[str]) -> str:
    if iid in official:
        return f"swebench/sweb.eval.arm64.{iid.replace('__', '_1776_')}:latest"
    return f"ghcr.io/epoch-research/swe-bench.eval.arm64.{iid}:latest"


EXCLUDED_REPOS = {"pylint-dev/pylint"}  # GPL-2.0

# Tasks that grade wrongly on their arm64 image, found with `steproute tasks check` (gold patch
# must pass, baseline patch must fail). Add an id here, with the reason, and regenerate.
BROKEN: dict[str, str] = {
    "psf__requests-1921": (
        "the gold patch fails on the arm64 image: 3 of 107 PASS_TO_PASS tests fail (one on the "
        "image's pytest 8.3.4, two with CERTIFICATE_VERIFY_FAILED reaching httpbin.org)"
    ),
    "psf__requests-1766": (
        "the gold patch fails on the arm64 image: 3 of 91 tests fail (test_conflicting_post_params "
        "and test_POSTBIN_GET_POST_FILES_WITH_DATA on the image's pytest 8.3.4, "
        "test_mixed_case_scheme_acceptable with CERTIFICATE_VERIFY_FAILED)"
    ),
    "psf__requests-2317": (
        "the gold patch fails on the arm64 image: 9 of 143 tests fail (pytest 8.3.4 string "
        "raises, CERTIFICATE_VERIFY_FAILED, timeouts reaching httpbin.org); the run takes 4 minutes"
    ),
    "psf__requests-2931": (
        "the gold patch fails on the arm64 image: 2 tests fail and 81 error, because the "
        "pytest-httpbin 'httpbin' fixture is recursive on the image's pytest 8.3.4 and the "
        "timeout tests need the network"
    ),
    "psf__requests-5414": (
        "the gold patch fails on the arm64 image: 4 tests fail and 158 error, because the "
        "pytest-httpbin 'httpbin' fixture is recursive on the image's pytest 8.3.4 and the "
        "timeout tests need the network"
    ),
}
# Repositories whose tests reach the live network are never chosen as a replacement.
NETWORK_REPOS = {"psf/requests"}


def load_panel(cache: Path, sub: str) -> dict[str, dict[str, Any]]:
    """instance_id -> {"resolved": bool, "api_calls": int | None}."""
    base = (
        "https://raw.githubusercontent.com/SWE-bench/experiments/"
        f"{EXPERIMENTS_COMMIT}/evaluation/verified/{sub}"
    )
    try:
        pid = curl_json(f"{base}/per_instance_details.json", cache / "exp" / sub / "pid.json")
        return {
            k: {"resolved": bool(v.get("resolved")), "api_calls": v.get("api_calls")}
            for k, v in pid.items()
        }
    except subprocess.CalledProcessError:
        res = curl_json(f"{base}/results/results.json", cache / "exp" / sub / "results.json")
        solved = set(res.get("resolved", []))
        return {k: {"resolved": True, "api_calls": None} for k in solved}


def patch_size(patch: str) -> tuple[int, int]:
    files = sum(1 for line in patch.splitlines() if line.startswith("diff --git"))
    lines = sum(
        1
        for line in patch.splitlines()
        if (line.startswith("+") or line.startswith("-")) and not line.startswith(("+++", "---"))
    )
    return files, lines


def stratify(rows: list[dict[str, Any]], panel: dict[str, dict[str, dict[str, Any]]]):
    out = []
    for r in rows:
        iid = r["instance_id"]

        def solved(sub: str, iid: str = iid) -> bool:
            return bool(panel[sub].get(iid, {}).get("resolved"))

        weak = sum(solved(s) for s in WEAK)
        mid = sum(solved(s) for s in MID)
        strong = sum(solved(s) for s in STRONG)
        if weak >= 2:
            stratum = "easy"
        elif mid:
            stratum = "medium"
        elif strong:
            stratum = "hard"
        else:
            stratum = "unsolved"
        calls = [
            panel[s][iid]["api_calls"]
            for s in WEAK + MID + STRONG
            if solved(s) and panel[s].get(iid, {}).get("api_calls")
        ]
        steps = statistics.median(calls) if calls else None
        issue_tokens = math.ceil(len(r["problem_statement"]) / 4)
        ctx = CTX_BASE + issue_tokens + CTX_PER_STEP * (steps if steps is not None else 60)
        files, lines = patch_size(r["patch"])
        out.append(
            {
                "id": iid,
                "repo": r["repo"],
                "difficulty": r["difficulty"],
                "stratum": stratum,
                "weak": weak,
                "strong": strong,
                "haiku": mid,
                "steps": steps,
                "ctx": ctx,
                "fits32k": ctx <= LOCAL_CAP - 4096,
                "files": files,
                "lines": lines,
            }
        )
    return out


def allocate(groups: dict[str, list[Any]], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n across groups, weighted by sqrt(group size)."""
    total = sum(len(v) for v in groups.values())
    if n >= total:
        return {k: len(v) for k, v in groups.items()}
    weights = {k: math.sqrt(len(v)) for k, v in groups.items()}
    wsum = sum(weights.values())
    raw = {k: n * w / wsum for k, w in weights.items()}
    alloc = {k: min(len(groups[k]), int(raw[k])) for k in groups}
    while sum(alloc.values()) < n:
        room = [k for k in groups if alloc[k] < len(groups[k])]
        k = max(room, key=lambda k: (raw[k] - alloc[k], k))
        alloc[k] += 1
    return alloc


def select(tasks: list[dict[str, Any]], rng: random.Random):
    full: list[dict[str, Any]] = []
    for stratum, n in FULL_SIZE.items():
        pool = sorted((t for t in tasks if t["stratum"] == stratum), key=lambda t: t["id"])
        by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for t in pool:
            by_repo[t["repo"]].append(t)
        alloc = allocate(dict(sorted(by_repo.items())), n)
        for repo in sorted(by_repo):
            full += rng.sample(by_repo[repo], alloc[repo])
    pilot: list[dict[str, Any]] = []
    for stratum, n in PILOT_SIZE.items():
        members = sorted((t for t in full if t["stratum"] == stratum), key=lambda t: t["id"])
        rng.shuffle(members)
        if stratum in ("easy", "medium"):
            members.sort(key=lambda t: not t["fits32k"])  # stable: fitting tasks first
        chosen: list[dict[str, Any]] = []
        repos: Counter[str] = Counter()
        for t in members:  # one pass preferring unseen repositories, then fill
            if len(chosen) < n and repos[t["repo"]] == 0:
                chosen.append(t)
                repos[t["repo"]] += 1
        for t in members:
            if len(chosen) < n and t not in chosen:
                chosen.append(t)
        pilot += chosen
    return sorted(full, key=lambda t: t["id"]), sorted(pilot, key=lambda t: t["id"])


def replacement_order(
    broken_id: str, stratum: str, universe: list[dict[str, Any]], drawn: set[str]
) -> list[dict[str, Any]]:
    """Candidates to replace ``broken_id``, best first. Deterministic."""
    pool = sorted(
        (
            t
            for t in universe
            if t["stratum"] == stratum
            and t["id"] not in drawn
            and t["id"] not in BROKEN
            and t["repo"] not in NETWORK_REPOS
        ),
        key=lambda t: t["id"],
    )
    random.Random(f"{SEED}:replace:{broken_id}").shuffle(pool)
    return pool


def replace_broken(
    full: list[dict[str, Any]], pilot: list[dict[str, Any]], universe: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Swap every drawn task listed in BROKEN for its replacement (module docstring, step 5).
    Returns the new full set, the new pilot and ``{broken id: replacement id}``."""
    drawn = {t["id"] for t in full}
    swaps: dict[str, str] = {}
    for t in sorted(full, key=lambda t: t["id"]):
        if t["id"] not in BROKEN:
            continue
        cands = replacement_order(t["id"], t["stratum"], universe, drawn)
        if not cands:
            sys.exit(f"no replacement for {t['id']}")
        swaps[t["id"]] = cands[0]["id"]
        drawn.add(cands[0]["id"])
        nxt = [c["id"] for c in cands[1:4]]
        print(f"replace {t['id']} ({t['stratum']}) by {cands[0]['id']}; next candidates: {nxt}")
    by_id = {t["id"]: t for t in universe}

    def swap(ts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = [by_id[swaps[t["id"]]] if t["id"] in swaps else t for t in ts]
        return sorted(out, key=lambda t: t["id"])

    return swap(full), swap(pilot), swaps


def quantiles20(xs: list[float]) -> list[int]:
    xs = sorted(xs)
    n = len(xs)
    return [int(xs[min(n - 1, int((i + 0.5) / 20 * n))]) for i in range(20)]


def write_manifest(
    full: list[dict[str, Any]],
    pilot: list[dict[str, Any]],
    official: set[str],
    broken: list[dict[str, Any]],
    swaps: dict[str, str],
) -> None:
    pilot_ids = {t["id"] for t in pilot}
    lines = [
        "# SWE-bench Verified subset for the steproute replay grid (issue 17).",
        "# Generated by scripts/select_taskset.py; do not edit by hand. Why: docs/TASKSET.md.",
        "# Only instance ids and our own derived labels are stored here. The task data itself",
        "# (problem statements, patches, tests) stays on Hugging Face and is fetched at run time.",
        "#",
        "# stratum: easy | medium | hard | unsolved, estimated from public per-instance results of",
        "#   other models (SWE-bench/experiments). An estimate of difficulty for our pool, not a",
        "#   measurement.",
        "# fits32k: our estimate that a short successful trajectory fits the 32K local context.",
        "# image: the native arm64 evaluation image (official swebench namespace where it exists,",
        "#   else Epoch AI's best-effort, untested build). Validate with the gold patch first.",
        "# broken: the task grades wrongly on its image (BROKEN in the script). Its record is",
        "#   kept, it is not part of the set (full_size and pilot_size count working tasks only)",
        "#   and no plan can include it. replaced_by: the task that took its place.",
        "",
        "[meta]",
        'name = "swebench-verified-v1"',
        'benchmark = "SWE-bench Verified"',
        f'dataset = "{DATASET}"',
        f'dataset_revision = "{DATASET_REVISION}"',
        'split = "test"',
        'backend = "swebench"',
        'image_arch = "arm64"',
        f"seed = {SEED}",
        f'experiments_commit = "{EXPERIMENTS_COMMIT}"',
        f"full_size = {len(full)}",
        f"pilot_size = {len(pilot)}",
        "",
    ]
    for t in sorted(full + broken, key=lambda t: t["id"]):
        lines += [
            "[[task]]",
            f'id = "{t["id"]}"',
            f'stratum = "{t["stratum"]}"',
            f"fits32k = {str(t['fits32k']).lower()}",
            f"pilot = {str(t['id'] in pilot_ids).lower()}",
            f'image = "{image_name(t["id"], official)}"',
        ]
        if t["id"] in BROKEN:
            lines.append(f'broken = "{BROKEN[t["id"]]}"')
            if t["id"] in swaps:
                lines.append(f'replaced_by = "{swaps[t["id"]]}"')
        lines.append("")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("\n".join(lines))


def report(label: str, chosen: list[dict[str, Any]], panel) -> None:
    print(f"\n== {label}: {len(chosen)} tasks")
    print("  strata:", dict(Counter(t["stratum"] for t in chosen)))
    print("  repos:", dict(Counter(t["repo"] for t in chosen).most_common()))
    print("  human difficulty:", dict(Counter(t["difficulty"] for t in chosen).most_common()))
    print(f"  fits 32K (est.): {sum(t['fits32k'] for t in chosen)}")
    print("  fits 32K by stratum:", dict(Counter(t["stratum"] for t in chosen if t["fits32k"])))
    ids = [t["id"] for t in chosen]
    for name, sub in [("weak-qwen2.5", WEAK[0]), ("haiku-4.5", MID[0]), ("opus-4.5", STRONG[0])]:
        rate = sum(bool(panel[sub].get(i, {}).get("resolved")) for i in ids) / len(ids)
        print(f"  public resolve rate {name}: {rate:.0%}")
    for model, sub in STEP_ANALOGS.items():
        calls = [panel[sub][i]["api_calls"] for i in ids if panel[sub].get(i, {}).get("api_calls")]
        print(f"  step quantiles ({model} <- {sub}): {quantiles20(calls)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, required=True, help="download cache directory")
    ap.add_argument("--write", action="store_true", help=f"write {MANIFEST}")
    args = ap.parse_args()
    rows = load_rows(args.cache)
    official = load_arm64(args.cache)
    epoch = load_epoch_arm64(args.cache, sorted(r["instance_id"] for r in rows))
    arm64 = official | epoch
    panel = {s: load_panel(args.cache, s) for s in WEAK + MID + STRONG}
    tasks = stratify(rows, panel)
    print(f"arm64 images: official {len(official)}, Epoch {len(epoch)}, union {len(arm64)}")
    universe = [t for t in tasks if t["id"] in arm64 and t["repo"] not in EXCLUDED_REPOS]
    print(f"Verified: {len(tasks)}; arm64 and not excluded: {len(universe)}")
    print("all 500 strata:", dict(Counter(t["stratum"] for t in tasks)))
    print("arm64 strata:", dict(Counter(t["stratum"] for t in universe)))
    fits = Counter(t["stratum"] for t in universe if t["fits32k"])
    print("arm64 fits32k by stratum:", dict(fits))
    full, pilot = select(universe, random.Random(SEED))
    drawn_broken = [t for t in full if t["id"] in BROKEN]
    full, pilot, swaps = replace_broken(full, pilot, universe)
    by_id = {t["id"]: t for t in universe}
    broken = drawn_broken + [by_id[i] for i in sorted(BROKEN) if i not in swaps and i in by_id]
    report("full set", full, panel)
    report("pilot", pilot, panel)
    if args.write:
        write_manifest(full, pilot, official, broken, swaps)
        print(f"\nwrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
