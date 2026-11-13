"""MCP server: lets any MCP client ask steproute which model and effort to use.

Two layers, so the logic is testable without the SDK:

* ``Advisor``: plain Python (no ``mcp`` import). It builds a ``StepState`` from a tool call,
  asks an embedded ``steproute.core.service.Service`` for the decision (so privacy rules,
  labelling, costing and trace storage are exactly those of ``steproute serve``), and tracks
  decisions that have not been reported yet.
* ``build_mcp_server`` / ``run_stdio``: register the three tools on the official ``mcp``
  SDK's ``MCPServer``. The SDK is imported lazily, so ``steproute --version`` works without
  the ``mcp`` extra.

The server only advises. An MCP server cannot change the host's model; the calling model has
to apply the recommendation itself. See ``docs/MCP.md``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from steproute import __version__
from steproute.core.schema import Outcome, StepState
from steproute.core.service import FAIL_OPEN_PREFIX, NoActionAllowed, Service

if TYPE_CHECKING:
    from mcp.server import MCPServer

log = logging.getLogger("steproute.mcp")

DEFAULT_TIMEOUT_S = 30 * 60.0
#: Placeholder token counts for the size words. Rough guesses, not measurements.
DEFAULT_SIZE_TOKENS: Mapping[str, int] = {"small": 2_000, "medium": 8_000, "large": 32_000}
TIMED_OUT_SIGNAL = "timed_out"

ROUTE_TASK_DESCRIPTION = """\
Ask the steproute router which model and reasoning effort to use for a task. Call this BEFORE \
you spawn a subagent or start a new task (a new subtask, a plan, a large edit), and use the \
answer to pick the subagent's model and effort. This tool only ADVISES: it cannot change \
your model or any setting. You must apply the recommendation yourself, for example by \
choosing the model when you spawn a subagent. If it says the task is sensitive, use only a \
local model for it. If it returns no recommendation, use your own default model (or keep the \
work local when it says the task is sensitive). The result includes a decision_id: keep it, \
and call report_outcome with it when the task is finished.

Arguments: description is a short summary of the task (it is checked for secrets and is not \
stored by default); kind is "main", "subagent", "plan" or "tool_followup" (use "subagent" \
when you are about to spawn one); tools lists the names of the tools the task will use; \
est_size is the estimated prompt size in tokens (an integer), or one of the words "small", \
"medium", "large" if you cannot estimate it.\
"""

REPORT_OUTCOME_DESCRIPTION = """\
Tell the steproute router how a task went, using the decision_id that route_task returned. \
ALWAYS call this once the task you asked about has finished, whether it worked or not: \
honest results, including failures, are how the router learns which model suits which task. \
success is true, false, or null if you cannot tell. signals is an optional object of facts \
you observed, such as tests_passed (bool), schema_valid (bool), tool_error (bool), \
compile_error (bool), user_rejected (bool), user_rewrote (bool), user_approved (bool), \
retried (bool), retry_count (int), judge_score (0 to 1); unknown keys are kept. This tool \
only records the result; it does not change any model. An unknown decision_id is reported \
back as not found and nothing is stored. A decision you never report is closed as unknown \
after a timeout.\
"""

LIST_MODELS_DESCRIPTION = """\
List the models steproute can recommend, with provider, tier (higher means more capable and \
more expensive), supported effort levels and whether each runs locally. Call it when you \
want to know which models a route_task answer can name. It is read-only and only informs you: \
steproute cannot change your model, you apply recommendations yourself.\
"""


class McpConfigError(ValueError):
    """Invalid MCP-specific settings."""


@dataclass(frozen=True)
class McpConfig:
    """MCP-specific settings; routing settings live in ``ServiceConfig``."""

    harness: str = "mcp"
    timeout_s: float = DEFAULT_TIMEOUT_S
    size_tokens: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_SIZE_TOKENS))
    sweep_interval_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.harness:
            raise McpConfigError("harness must not be empty")
        if not self.timeout_s > 0:
            raise McpConfigError("timeout must be positive")
        if not self.sweep_interval_s > 0:
            raise McpConfigError("sweep interval must be positive")
        if set(self.size_tokens) != set(DEFAULT_SIZE_TOKENS):
            raise McpConfigError("size_tokens needs exactly: small, medium, large")
        for word, n in self.size_tokens.items():
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise McpConfigError(f"size {word!r} must be a non-negative integer")


class Advisor:
    """The tool logic: route, report, list, and close unreported decisions."""

    def __init__(
        self,
        service: Service,
        config: McpConfig | None = None,
        *,
        clock: Callable[[], float] = time.time,
        session_id: str | None = None,
    ) -> None:
        self.service = service
        self.config = config or McpConfig()
        self._clock = clock
        self.session_id = session_id or f"mcp-{secrets.token_hex(4)}"
        self._lock = threading.Lock()
        self._step = 0
        self._pending: dict[str, float] = {}  # decision_id -> time it was issued
        self._closed: set[str] = set()  # issued, then timed out or reported
        self.timed_out = 0

    # --- route_task ---

    def parse_size(self, est_size: int | str | None) -> int:
        """Prompt tokens from ``est_size``: an int, a digit string, or small/medium/large."""
        if est_size is None:
            return self.config.size_tokens["medium"]
        if isinstance(est_size, bool):
            raise ValueError("est_size must be an integer token count or small/medium/large")
        if isinstance(est_size, int):
            if est_size < 0:
                raise ValueError("est_size must not be negative")
            return est_size
        word = str(est_size).strip().casefold()
        if word in self.config.size_tokens:
            return self.config.size_tokens[word]
        if word.isdecimal():
            return int(word)
        raise ValueError(
            f"est_size {est_size!r} is not an integer token count or small/medium/large"
        )

    def route_task(
        self,
        description: str,
        kind: str = "main",
        tools: list[str] | None = None,
        est_size: int | str | None = None,
    ) -> dict[str, Any]:
        self.expire()
        tokens = self.parse_size(est_size)
        with self._lock:
            idx = self._step
            self._step += 1
        state = StepState(
            harness=self.config.harness,
            session_id=self.session_id,
            step_idx=idx,
            kind=kind or "main",
            prompt_tokens=tokens,
            tools=[str(t) for t in tools or []],
            source="personal",
            task_text=description,
        )
        try:
            decision = self.service.decide(state)
        except NoActionAllowed as exc:
            return _no_advice(exc.sensitive)
        with self._lock:
            self._pending[decision.decision_id] = self._clock()
        steps = decision.action.chain()
        sensitive = decision.state.sensitive
        model = self.service.catalog.get(steps[0].model)
        return {
            "recommendation": True,
            "decision_id": decision.decision_id,
            "model": steps[0].model,
            "effort": steps[0].effort,
            "local": model.local,
            "cascade": [{"model": s.model, "effort": s.effort} for s in steps[1:]],
            "reason": explain(decision.policy, decision.state.features.get("rules"), sensitive),
            "sensitive": sensitive,
            "note": "Advice only: apply it yourself, then call report_outcome with the "
            "decision_id when the task is finished.",
        }

    # --- report_outcome ---

    def report_outcome(
        self,
        decision_id: str,
        success: bool | None = None,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.expire()
        with self._lock:
            known = decision_id in self._pending or decision_id in self._closed
        if not known:
            return {
                "recorded": False,
                "found": False,
                "message": f"decision_id {decision_id!r} is not known to this server; nothing "
                "was stored. Use the decision_id returned by route_task in this session.",
            }
        outcome = self.service.report(
            Outcome(
                decision_id=decision_id,
                ts=self._clock(),
                success=success,
                signals=dict(signals or {}),
            )
        )
        with self._lock:
            self._pending.pop(decision_id, None)
            self._closed.add(decision_id)
        return {"recorded": True, "found": True, "success": outcome.success}

    # --- list_models ---

    def list_models(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": m.id,
                    "provider": m.provider,
                    "tier": m.tier,
                    "efforts": list(m.efforts),
                    "local": m.local,
                }
                for m in self.service.catalog.pool()
            ]
        }

    # --- unreported decisions ---

    def expire(self) -> int:
        """Close decisions unreported for longer than the timeout. Returns how many."""
        now = self._clock()
        with self._lock:
            due = [d for d, t in self._pending.items() if now - t >= self.config.timeout_s]
        return self._close(due)

    def close_all(self) -> int:
        """Close every pending decision (shutdown). Returns how many."""
        with self._lock:
            due = list(self._pending)
        return self._close(due)

    def _close(self, ids: list[str]) -> int:
        closed = 0
        for did in ids:
            with self._lock:
                if self._pending.pop(did, None) is None:
                    continue  # reported or closed concurrently
                self._closed.add(did)
            try:
                self.service.report(
                    Outcome(
                        decision_id=did,
                        ts=self._clock(),
                        success=None,
                        signals={TIMED_OUT_SIGNAL: True},
                    )
                )
            except Exception as exc:  # shutdown must not fail on a full disk
                log.error("could not close decision %s: %s", did, type(exc).__name__)
                continue
            closed += 1
            self.timed_out += 1
        return closed

    @property
    def pending(self) -> int:
        return len(self._pending)


def _no_advice(sensitive: bool) -> dict[str, Any]:
    if sensitive:
        advice = (
            "steproute has no recommendation. This task looks sensitive: keep it on a local "
            "model, or do not send it to a cloud model."
        )
    else:
        advice = (
            "steproute has no recommendation right now. Use your own default model for this task."
        )
    return {
        "recommendation": False,
        "decision_id": None,
        "model": None,
        "effort": None,
        "cascade": [],
        "sensitive": sensitive,
        "reason": advice,
        "note": "Nothing was logged, so there is nothing to report for this call.",
    }


def explain(policy: str, note: object, sensitive: bool) -> str:
    """One plain-words sentence from what the policy recorded."""
    if policy.startswith(FAIL_OPEN_PREFIX):
        why = policy[len(FAIL_OPEN_PREFIX) :].replace("_", " ")
        sentence = f"The router could not decide ({why}), so a configured fallback is used"
    elif isinstance(note, dict):
        if note.get("rule") == "readonly_subagent":
            sentence = f"Read-only subagent work, so a cheaper tier ({note.get('tier')})"
        else:
            sentence = f"No special rule applies, so the default tier ({note.get('tier')})"
        if note.get("fallback"):
            sentence += (
                f" was used because tier {note.get('target_tier')} had no allowed model "
                "for this task"
            )
        elif note.get("stayed"):
            sentence += " and the current model is kept"
    else:
        sentence = f"Chosen by the {policy!r} policy"
    if sensitive:
        sentence += "; the task looks sensitive, so only local models are allowed"
    return sentence + "."


# --- MCP wiring (the only part that needs the SDK) ----------------------------------------


def build_mcp_server(advisor: Advisor) -> MCPServer:
    """Register the three tools on an SDK ``MCPServer`` (imported here, lazily)."""
    from mcp.server import MCPServer

    @contextlib.asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        async def sweep() -> None:
            while True:
                await asyncio.sleep(advisor.config.sweep_interval_s)
                advisor.expire()

        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            advisor.close_all()

    server: MCPServer = MCPServer(
        "steproute",
        version=__version__,
        instructions=(
            "steproute recommends a model and effort per task. It only advises: apply its "
            "recommendations yourself. Call route_task before spawning a subagent or starting "
            "a task, and report_outcome when it finishes."
        ),
        lifespan=lifespan,
        log_level="WARNING",
    )

    @server.tool(name="route_task", description=ROUTE_TASK_DESCRIPTION)
    def route_task(
        description: str,
        kind: str = "main",
        tools: list[str] | None = None,
        est_size: int | str | None = None,
    ) -> dict[str, Any]:
        return advisor.route_task(description, kind, tools, est_size)

    @server.tool(name="report_outcome", description=REPORT_OUTCOME_DESCRIPTION)
    def report_outcome(
        decision_id: str,
        success: bool | None = None,
        signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return advisor.report_outcome(decision_id, success, signals)

    @server.tool(name="list_models", description=LIST_MODELS_DESCRIPTION)
    def list_models() -> dict[str, Any]:
        return advisor.list_models()

    return server


def run_stdio(advisor: Advisor) -> None:
    """Serve over stdio until the client disconnects; then close pending decisions."""
    server = build_mcp_server(advisor)
    try:
        server.run("stdio")
    finally:
        advisor.close_all()
