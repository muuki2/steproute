"""Where the proxy's decisions come from: an in-process ``Service`` or a running
``steproute serve`` over HTTP, behind one small interface. Standard library only.

Both raise ``SourceError`` for anything that leaves the proxy without a decision; the proxy
then fails open (``docs/PROXY.md``, "Fail-open").
"""

from __future__ import annotations

import http.client
import json
from typing import Protocol
from urllib.parse import urlsplit

from steproute.core.schema import Decision, Outcome, StepState
from steproute.core.service import NoActionAllowed, Service

__all__ = ["DecisionSource", "HttpSource", "InProcessSource", "SourceError"]


class SourceError(Exception):
    """No decision (or the outcome was not accepted). ``reason`` is a short code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DecisionSource(Protocol):
    name: str

    def decide(self, state: StepState) -> Decision: ...

    def report(self, outcome: Outcome) -> Outcome: ...


class InProcessSource:
    """A ``Service`` in the proxy process: same policy, budget, fail-open and store rules."""

    name = "inprocess"

    def __init__(self, service: Service) -> None:
        self.service = service

    def decide(self, state: StepState) -> Decision:
        try:
            return self.service.decide(state)
        except NoActionAllowed as exc:
            raise SourceError("no_action_allowed" if exc.configured else "no_fallback") from None
        except Exception as exc:  # the service is a helper; never take the request down
            raise SourceError(f"error:{type(exc).__name__}") from None

    def report(self, outcome: Outcome) -> Outcome:
        try:
            return self.service.report(outcome)
        except Exception as exc:
            raise SourceError(f"error:{type(exc).__name__}") from None


class HttpSource:
    """``POST /decide`` and ``POST /report`` on a running ``steproute serve``."""

    name = "http"

    def __init__(
        self,
        url: str,
        *,
        token: str | None = None,
        decide_timeout_s: float = 0.25,
        report_timeout_s: float = 2.0,
    ) -> None:
        parts = urlsplit(url)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("decisions.url must be an http:// URL")
        self.host = parts.hostname
        self.port = parts.port or 80
        self.prefix = parts.path.rstrip("/")
        self.token = token
        self.decide_timeout_s = decide_timeout_s
        self.report_timeout_s = report_timeout_s

    def _post(self, path: str, body: dict[str, object], timeout: float) -> dict[str, object]:
        data = json.dumps(body, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        try:
            conn.request("POST", self.prefix + path, body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
        except TimeoutError:
            raise SourceError("timeout") from None
        except OSError as exc:
            raise SourceError(f"unreachable:{type(exc).__name__}") from None
        except http.client.HTTPException as exc:
            raise SourceError(f"http:{type(exc).__name__}") from None
        finally:
            conn.close()
        if resp.status != 200:
            raise SourceError(f"status:{resp.status}")
        try:
            obj = json.loads(raw)
        except ValueError:
            raise SourceError("bad_json") from None
        if not isinstance(obj, dict):
            raise SourceError("bad_json")
        return obj

    def decide(self, state: StepState) -> Decision:
        obj = self._post("/decide", state.to_dict(), self.decide_timeout_s)
        try:
            return Decision.from_dict(obj)
        except (TypeError, ValueError):
            raise SourceError("bad_decision") from None

    def report(self, outcome: Outcome) -> Outcome:
        obj = self._post("/report", outcome.to_dict(), self.report_timeout_s)
        try:
            return Outcome.from_dict(obj)
        except (TypeError, ValueError):
            raise SourceError("bad_outcome") from None
