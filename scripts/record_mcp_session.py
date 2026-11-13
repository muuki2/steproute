"""Record a real stdio session with the MCP server into tests/fixtures/mcp_session.json.

Run it with the mcp extra installed:

    uv run --extra mcp python scripts/record_mcp_session.py

It launches ``steproute mcp`` as a subprocess and drives it with the official SDK client
(``mcp.ClientSession`` over ``mcp.client.stdio.stdio_client``). The streams are wrapped so
every JSON-RPC message in both directions is captured as it crosses the wire. Nothing leaves
the machine and no model provider is called.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mcp_session.json"
LAUNCH = "import sys; from steproute.cli import main; sys.exit(main(sys.argv[1:]))"


class Tee:
    """Wraps an SDK stream and logs each message that passes through."""

    def __init__(self, inner: Any, direction: str, log: list[dict[str, Any]]) -> None:
        self._inner, self._direction, self._log = inner, direction, log

    def _note(self, item: Any) -> None:
        message = getattr(item, "message", None)
        if message is not None:
            data = message.model_dump(by_alias=True, mode="json", exclude_none=True)
            self._log.append({"dir": self._direction, "message": data})

    async def send(self, item: Any) -> None:
        self._note(item)
        await self._inner.send(item)

    async def receive(self) -> Any:
        item = await self._inner.receive()
        self._note(item)
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __aiter__(self) -> Tee:
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.receive()
        except Exception as exc:
            if type(exc).__name__ in ("EndOfStream", "ClosedResourceError"):
                raise StopAsyncIteration from None
            raise

    async def __aenter__(self) -> Tee:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._inner.aclose()


async def record(store: str) -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    params = StdioServerParameters(
        command=sys.executable, args=["-c", LAUNCH, "mcp", "--store", store]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(Tee(read, "in", log), Tee(write, "out", log)) as session:
            await session.initialize()
            await session.list_tools()
            await session.call_tool("list_models", {})
            routed = await session.call_tool(
                "route_task",
                {
                    "description": "Read the repo and summarise the test layout",
                    "kind": "subagent",
                    "tools": ["Read", "Grep"],
                    "est_size": "small",
                },
            )
            decision_id = routed.structured_content["decision_id"]  # type: ignore[index]
            await session.call_tool(
                "report_outcome",
                {"decision_id": decision_id, "success": True, "signals": {"tests_passed": True}},
            )
    return log


def main() -> None:
    with tempfile.TemporaryDirectory() as store:
        log = asyncio.run(record(store))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"messages": log}, indent=2, sort_keys=True) + "\n")
    print(f"recorded {len(log)} messages to {OUT}")


if __name__ == "__main__":
    main()
