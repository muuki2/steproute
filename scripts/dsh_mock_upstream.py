"""Run the repository's mock upstream as a scripted coding agent's model, for a real dsh session
(scripts/dsh_session.sh). Loopback only; no key is checked and nothing leaves the machine.

    uv run python scripts/dsh_mock_upstream.py --out RECEIVED.json

Prints ``mock upstream: <base url>`` (the Anthropic Messages root dsh-llm-deepseek appends
``/v1/messages`` to) and serves until SIGTERM or SIGINT, then writes what each request asked
for (model, thinking, effort) to RECEIVED.json.

Per user turn the scripted model reads ``a.txt``, then ``b.txt`` (one ``read`` tool call per
step, the tool name of dsh-tool-fs), then answers in one sentence: three steps per turn.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path
from typing import Any

from steproute.collectors.mock_upstream import MockUpstream, Received, Reply


def _tool_results_since_user_text(body: dict[str, Any]) -> int:
    """Tool results after the last user message that carries text (the turn's prompt)."""
    n = 0
    for msg in reversed(body.get("messages") or []):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if msg.get("role") != "user":
            continue
        if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in blocks):
            n += sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "tool_result")
            continue
        if any(isinstance(b, dict) and b.get("type") == "text" and b.get("text") for b in blocks):
            break
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    counter = {"calls": 0}

    def respond(rec: Received) -> Reply | None:
        if not rec.path.rstrip("/").endswith("/v1/messages"):
            return None
        done = _tool_results_since_user_text(rec.body)
        counter["calls"] += 1
        n = counter["calls"]
        prompt = 1200 + 150 * n
        cached = 0 if n == 1 else 1000
        if done < 2:
            name = "a.txt" if done == 0 else "b.txt"
            return Reply(
                kind="tool",
                tool_name="read",
                tool_args=json.dumps({"path": name}),
                tool_id=f"toolu_mock_{n}",
                tool_text=f"Reading {name}.",
                prompt_tokens=prompt,
                cached_tokens=cached,
                completion_tokens=25,
            )
        return Reply(
            kind="text",
            content="Both files are short notes about a town and a shop.",
            prompt_tokens=prompt,
            cached_tokens=cached,
            completion_tokens=15,
        )

    upstream = MockUpstream().start()
    upstream.set_responder(respond)
    print(f"mock upstream: {upstream.anthropic_base_url}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stop.wait()
    received = [
        {
            "path": r.path,
            "model": r.body.get("model"),
            "thinking": r.body.get("thinking"),
            "output_config": r.body.get("output_config"),
            "stream": r.body.get("stream"),
            "has_session_log_field": "dsh_session_log" in r.body,
            "has_plugin_inventory_field": "dsh_plugin_packages" in r.body,
        }
        for r in upstream.received
    ]
    upstream.stop()
    args.out.write_text(json.dumps(received, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
