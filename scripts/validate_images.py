"""Validate the arm64 task images: gold must pass, baseline must fail (issue 64).

For every working task of the manifest that has no line in the results file yet, the driver
makes the image local (Docker Hub: `docker pull`; ghcr.io: `scripts/fetch_ghcr_image.py` then
`docker load`, the tar and blobs deleted), runs `steproute tasks check` with the gold patch and
with `--patch baseline`, and appends one JSON line to the results file. Resumable: a task with a
line is skipped. One downloader thread, one check at a time.

Rate limits: on Docker Hub `toomanyrequests` (or ghcr HTTP 429) that source is paused for
`--backoff` seconds and the other source goes on. Nothing is retried in a loop.

Wall clock: no download starts after `--stop-downloads`, no task starts after `--stop`, and every
subprocess is killed at `--kill` (all HH:MM, local time, today). Run it as

    caffeinate -i uv run python scripts/validate_images.py --results docs/taskset-checks.jsonl

Tasks already checked by hand (the pilot and the first replacements) are skipped through
`--skip` (default list below) and through their `pilot = true` flag.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from steproute.tasks.manifest import Task, load_default_manifest

ROOT = Path(__file__).resolve().parent.parent
CHECKED_BY_HAND = (
    "psf__requests-1142",
    "psf__requests-6028",
    "sympy__sympy-16450",
    "django__django-16139",
    "sympy__sympy-15875",
    "sympy__sympy-23950",
)
HUB = "hub"
GHCR = "ghcr"


def source(task: Task) -> str:
    return GHCR if (task.image or "").startswith("ghcr.io/") else HUB


def at(hhmm: str) -> float:
    h, m = hhmm.split(":")
    now = dt.datetime.now()
    return now.replace(hour=int(h), minute=int(m), second=0, microsecond=0).timestamp()


def local(image: str | None) -> bool:
    r = subprocess.run(["docker", "image", "inspect", image or "none"], capture_output=True)
    return r.returncode == 0


def run(cmd: list[str], limit: float, cwd: Path = ROOT) -> tuple[int, str]:
    """Run a command, kill it after `limit` seconds. Returns (code, combined output)."""
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=max(1.0, limit), check=False
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return 124, out + "\n[killed at the wall-clock limit]"
    return r.returncode, r.stdout + r.stderr


class Driver:
    def __init__(self, args: argparse.Namespace) -> None:
        self.results = Path(args.results)
        self.stop_dl = at(args.stop_downloads)
        self.stop = at(args.stop)
        self.kill = at(args.kill)
        self.backoff = float(args.backoff)
        self.paused_until: dict[str, float] = {HUB: 0.0, GHCR: 0.0}
        self.lock = threading.Lock()
        self.skip = set(CHECKED_BY_HAND) | set(args.skip)
        self.tmp = Path(tempfile.mkdtemp(prefix="validate-images-"))
        self.pull_info: dict[str, dict[str, object]] = {}

    def done(self) -> set[str]:
        if not self.results.exists():
            return set()
        return {json.loads(x)["task"] for x in self.results.read_text().splitlines() if x.strip()}

    def pending(self) -> list[Task]:
        done = self.done()
        return [
            t
            for t in load_default_manifest()
            if not t.pilot and t.id not in self.skip and t.id not in done
        ]

    def log(self, msg: str) -> None:
        print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)

    def fetch(self, task: Task) -> None:
        src = source(task)
        image = task.image or ""
        t0 = time.time()
        if src == HUB:
            code, out = run(["docker", "pull", "-q", image], self.kill - time.time())
        else:
            tar = self.tmp / f"{task.id}.tar"
            blobs = self.tmp / f"{task.id}-blobs"
            code, out = run(
                [
                    sys.executable,
                    "scripts/fetch_ghcr_image.py",
                    image,
                    str(tar),
                    str(blobs),
                ],
                self.kill - time.time(),
            )
            if code == 0:
                code, out2 = run(["docker", "load", "-i", str(tar)], self.kill - time.time())
                out += out2
            run(["rm", "-rf", str(tar), str(blobs)], 120)
        low = out.lower()
        if code != 0 and ("toomanyrequests" in low or "429" in low or "rate limit" in low):
            with self.lock:
                self.paused_until[src] = time.time() + self.backoff
            self.log(f"{src} rate limited; paused {self.backoff:.0f}s")
        elif code != 0:
            self.log(f"fetch failed {task.id}: {out.strip()[-300:]}")
            with self.lock:
                self.paused_until[src] = time.time() + 120  # not a loop: wait a little
        with self.lock:
            self.pull_info[task.id] = {"pull_s": round(time.time() - t0, 1), "pull_ok": code == 0}

    def downloader(self) -> None:
        failed: dict[str, int] = {}
        while time.time() < self.stop_dl:
            todo = [t for t in self.pending() if not local(t.image) and failed.get(t.id, 0) < 3]
            now = time.time()
            ready = [t for t in todo if self.paused_until[source(t)] <= now]
            if not todo:
                return
            if not ready:
                time.sleep(60)
                continue
            # stay at most a few tasks ahead of the checker
            ahead = sum(1 for t in self.pending()[:6] if local(t.image))
            if ahead >= 4:
                time.sleep(20)
                continue
            t = ready[0]
            self.log(f"fetching {t.id} ({source(t)})")
            self.fetch(t)
            if not local(t.image):
                failed[t.id] = failed.get(t.id, 0) + 1

    def check(self, task: Task, patch: str) -> dict[str, object]:
        t0 = time.time()
        code, out = run(
            ["uv", "run", "steproute", "tasks", "check", task.id, "--patch", patch, "--json"],
            self.kill - time.time(),
        )
        rec: dict[str, object] = {"seconds": round(time.time() - t0, 1), "exit": code}
        for line in reversed(out.splitlines()):
            if line.startswith("{"):
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                rec["status"] = d.get("status")
                rec["detail"] = str(d.get("detail", ""))[-600:]
                return rec
        rec["status"] = "no-json"
        rec["detail"] = out.strip()[-600:]
        return rec

    def main(self) -> int:
        dl = threading.Thread(target=self.downloader, daemon=True)
        dl.start()
        while time.time() < self.stop:
            todo = self.pending()
            if not todo:
                self.log("all tasks have a result")
                break
            ready = [t for t in todo if local(t.image)]
            if not ready:
                if not dl.is_alive():
                    self.log("nothing local and the downloader has stopped")
                    break
                time.sleep(15)
                continue
            t = ready[0]
            started = dt.datetime.now().isoformat(timespec="seconds")
            self.log(f"checking {t.id}")
            gold = self.check(t, "gold")
            base = self.check(t, "baseline")
            rec = {
                "task": t.id,
                "image": t.image,
                "source": source(t),
                "gold": gold.get("status"),
                "gold_s": gold["seconds"],
                "gold_detail": gold.get("detail"),
                "baseline": base.get("status"),
                "baseline_s": base["seconds"],
                "baseline_detail": base.get("detail"),
                "pull": self.pull_info.get(t.id),
                "started": started,
                "finished": dt.datetime.now().isoformat(timespec="seconds"),
            }
            with self.results.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            self.log(f"{t.id}: gold={rec['gold']} baseline={rec['baseline']}")
        return 0


def cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="docs/taskset-checks.jsonl")
    p.add_argument("--stop-downloads", default="19:15")
    p.add_argument("--stop", default="19:30")
    p.add_argument("--kill", default="19:45")
    p.add_argument("--backoff", default="1800")
    p.add_argument("--skip", nargs="*", default=[])
    return Driver(p.parse_args()).main()


if __name__ == "__main__":
    raise SystemExit(cli())
