"""Benchmark POST /decide of the local service with the rule baseline and the default catalog.

    uv run python scripts/bench_service.py [--n 3000] [--text-chars 20000]

Starts ``steproute serve`` as a separate process (so the client and the server do not share a
GIL), on a free loopback port with a temporary trace store, then sends requests over ONE
kept-alive HTTP/1.1 connection and times each round trip (send, wait, read the full response;
the request body is encoded beforehand). Reports p50 / p95 / p99 / max, once with no
``task_text`` (the typical case) and once with ``--text-chars`` characters of ordinary
non-secret code and prose plus five file paths, so sensitive detection scans everything.
Timings are for this machine; no test asserts on them.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CODE = "import sys; from steproute.cli import main; sys.exit(main(sys.argv[1:]))"
CHUNK = (
    "def handler(request, context):\n"
    "    # Look up the user record and return the rendered page.\n"
    "    user = db.users.get(request.args['id'])\n"
    "    return render('profile.html', user=user, items=load_items(user))\n"
    "The test in tests/test_views.py fails with a KeyError on line 42; please fix it.\n"
)
PATHS = ["src/app/views.py", "src/app/db.py", "tests/test_views.py", "README.md", "setup.cfg"]
WARMUP = 200


def percentile(sorted_ms: list[float], q: float) -> float:
    return sorted_ms[min(len(sorted_ms) - 1, int(q * (len(sorted_ms) - 1) + 0.5))]


def measure(port: int, body: dict[str, object], n: int) -> list[float]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    times: list[float] = []
    for i in range(n + WARMUP):
        payload = json.dumps({**body, "step_idx": i}).encode()
        t = time.perf_counter()
        conn.request("POST", "/decide", body=payload, headers=headers)
        resp = conn.getresponse()
        resp.read()
        dt = (time.perf_counter() - t) * 1000.0
        if resp.status != 200:
            raise SystemExit(f"unexpected status {resp.status}")
        if i >= WARMUP:
            times.append(dt)
    conn.close()
    return sorted(times)


def report(label: str, times: list[float]) -> None:
    print(
        f"{label:<34} p50 {percentile(times, 0.50):6.3f} ms  "
        f"p95 {percentile(times, 0.95):6.3f} ms  p99 {percentile(times, 0.99):6.3f} ms  "
        f"max {times[-1]:6.3f} ms  (n={len(times)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--text-chars", type=int, default=20_000)
    args = ap.parse_args()

    state: dict[str, object] = {
        "harness": "bench",
        "session_id": "bench-session",
        "step_idx": 0,
        "kind": "main",
        "prompt_tokens": 20_000,
        "cached_prefix_tokens": 18_000,
        "current_model": "claude-sonnet-5-5",
        "current_effort": "high",
        "tools": ["Bash", "Edit"],
    }
    text = (CHUNK * (args.text_chars // len(CHUNK) + 1))[: args.text_chars]
    with_text = {**state, "task_text": text, "features": {"paths": PATHS}}

    with tempfile.TemporaryDirectory() as store:
        proc = subprocess.Popen(
            [sys.executable, "-c", CODE, "serve", "--port", "0", "--store", str(Path(store))],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONWARNINGS": "ignore"},
        )
        try:
            assert proc.stdout is not None
            port = int(proc.stdout.readline().rsplit(":", 1)[1])
            print(
                f"python {platform.python_version()} on {platform.machine()}, {platform.system()}"
            )
            print("steproute serve: rule baseline, default catalog, trace store on local disk")
            report("no task_text", measure(port, state, args.n))
            report(f"{len(text)}-char task_text + 5 paths", measure(port, with_text, args.n))
        finally:
            proc.terminate()
            proc.wait(10)
            if proc.stdout:
                proc.stdout.close()


if __name__ == "__main__":
    main()
