"""The proxy's request logic, without HTTP: state, decision, rewrite, cascade, outcome.

``Proxy.handle`` turns one chat-completions request into a ``Response``. The HTTP layer
writes it (all at once, or chunk by chunk for a stream) and then calls ``Response.finish``,
which reports the outcome, so reporting never adds latency for the client. See
``docs/PROXY.md`` for every rule implemented here.
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from steproute.collectors.proxy.checks import (
    CheckResult,
    StreamAccumulator,
    normalise_usage,
    run_checks,
    signals_for,
)
from steproute.collectors.proxy.config import FORWARDED_PARAMS, ProxyConfig, UpstreamConfig
from steproute.collectors.proxy.session import SessionTable, session_id_from_messages, split_prompt
from steproute.collectors.proxy.source import DecisionSource, SourceError
from steproute.collectors.proxy.upstream import UpstreamError, UpstreamPool
from steproute.core.catalog import Catalog
from steproute.core.cost import AttemptUsage, price_usage
from steproute.core.privacy import PrivacyConfig
from steproute.core.schema import EFFORTS, SOURCES, Decision, ModelProfile, Outcome, StepState

__all__ = ["SHADOW_PREFIX", "Proxy", "Response"]

log = logging.getLogger("steproute.proxy")

#: Outcome id of a shadow-mode step: ``"shadow:" + <id of the recommendation>``. No decision
#: has this id, so the outcome of the client's own model never joins the router's action.
SHADOW_PREFIX = "shadow:"
_MAX_OUTCOMES = 4096


def default_effort(profile: ModelProfile) -> str:
    d = profile.meta.get("default_effort")
    if isinstance(d, str) and d in profile.efforts:
        return d
    return "medium" if "medium" in profile.efforts else profile.efforts[0]


def _error_body(status: int, kind: str, message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": kind, "code": status}}


def _sse(obj: Mapping[str, Any] | str) -> bytes:
    text = obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))
    return f"data: {text}\n\n".encode()


@dataclass
class Attempt:
    """One planned or executed upstream call."""

    model: str
    effort: str | None
    rewrite: bool
    skipped: str | None = None  # "no_upstream" or "privacy": never called
    ran: bool = False  # the upstream call was made
    usage: AttemptUsage | None = None
    cost_usd: float = 0.0
    error: UpstreamError | None = None
    check: CheckResult | None = None
    dropped: tuple[str, ...] = ()
    #: Who placed Anthropic cache markers (``Prepared.cache``) and whether the provider keeps
    #: the prompt cached after this call (``Prepared.cacheable``).
    cache: str = "none"
    cacheable: bool = True
    latency_ms: float = 0.0

    @property
    def called(self) -> bool:
        return self.ran

    @property
    def responded(self) -> bool:
        return self.called and self.error is None


@dataclass
class Response:
    """What the HTTP layer sends. Exactly one of ``body`` and ``stream`` is set."""

    status: int
    headers: dict[str, str]
    body: dict[str, Any] | None = None
    stream: Iterator[bytes] | None = None
    _finish: Callable[[], None] | None = field(default=None, repr=False)

    def finish(self) -> None:
        """Report the outcome. Call once, after the response was written (or failed)."""
        fn, self._finish = self._finish, None
        if fn is not None:
            fn()


@dataclass
class _Step:
    """Everything one request carries from ``handle`` to ``finish``."""

    session_id: str
    prompt: Any
    params: dict[str, Any]
    client_model: str
    mode: str
    outcome_id: str | None
    sensitive: bool
    attempts: list[Attempt]
    t0: float
    stream: bool = False
    signals: dict[str, Any] = field(default_factory=dict)


class Proxy:
    """The proxy logic. Thread-safe; one instance serves every request."""

    def __init__(
        self,
        config: ProxyConfig,
        *,
        source: DecisionSource,
        catalog: Catalog,
        privacy: PrivacyConfig,
        pool: UpstreamPool | None = None,
        session_salt: str | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.source = source
        self.catalog = catalog
        self.privacy = privacy
        self.pool = pool or UpstreamPool()
        self.salt = session_salt or secrets.token_hex(16)
        self._clock = clock
        self.sessions = SessionTable(
            max_sessions=config.max_sessions,
            idle_s=config.session_idle_s,
            chars_per_token=config.chars_per_token,
            clock=clock,
        )
        self.allowed_params = FORWARDED_PARAMS | set(config.extra_params)
        self._outcomes: OrderedDict[str, Outcome] = OrderedDict()
        self._lock = threading.Lock()
        self.counters = {
            "requests": 0,
            "source_failures": 0,
            "upstream_errors": 0,
            "privacy_blocked": 0,
            "report_failures": 0,
            "fail_open": 0,
        }
        self._names: dict[str, str] = {}
        for up in config.upstreams:
            for m in up.models:
                self._names.setdefault(m, m)
                self._names.setdefault(up.upstream_model(m), m)
        self._names.update(config.aliases)

    def _count(self, key: str) -> None:
        with self._lock:
            self.counters[key] += 1

    # --- helpers ---

    def resolve(self, name: str) -> str | None:
        """The catalog id for a client's model name (served id, alias, upstream name)."""
        return self._names.get(name)

    def models(self) -> list[dict[str, Any]]:
        out = []
        for up in self.config.upstreams:
            for m in up.models:
                out.append({"id": m, "object": "model", "created": 0, "owned_by": up.name})
        for alias in self.config.aliases:
            out.append({"id": alias, "object": "model", "created": 0, "owned_by": "alias"})
        return out

    def health(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self.counters)
        return {
            "status": "ok",
            "mode": self.config.mode,
            "source": self.source.name,
            "sessions": len(self.sessions),
            **counters,
        }

    def _reasoning(self, up: UpstreamConfig, model: str, effort: str | None) -> str | None:
        if effort is None:
            return None
        value = self.catalog.get(model).effort_map.get(effort)
        if value is None:
            return None  # an effort missing from the map means: send no reasoning parameter
        return up.reasoning_values.get(value, value)

    # --- the request ---

    def handle(self, body: Mapping[str, Any], headers: Mapping[str, str]) -> Response:
        self._count("requests")
        h = {k.lower(): v for k, v in headers.items()}
        messages = body.get("messages")
        model = body.get("model")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(model, str)
            or not model
        ):
            return Response(
                400,
                {},
                _error_body(
                    400, "invalid_request_error", "'model' and a non-empty 'messages' are required"
                ),
            )
        stream = body.get("stream") is True
        source = h.get("x-steproute-source", "personal")
        if source not in SOURCES:
            return Response(
                400,
                {},
                _error_body(
                    400,
                    "invalid_request_error",
                    f"X-Steproute-Source must be one of {list(SOURCES)}",
                ),
            )
        params = {k: v for k, v in body.items() if k in self.allowed_params}
        dropped_client = sum(
            1 for k in body if k not in self.allowed_params and k not in ("model", "stream")
        )
        tools = body.get("tools") if isinstance(body.get("tools"), list) else None
        client_id = self.resolve(model)
        client_effort = body.get("reasoning_effort")
        if client_effort not in EFFORTS:
            client_effort = None
        client_up = self.config.upstream_for(client_id) if client_id else None
        client_local = client_up is not None and client_up.local

        sid = h.get(self.config.session_header.lower()) or session_id_from_messages(
            messages, self.salt
        )
        prompt = split_prompt(messages, tools)
        step_idx, est, sensitive, _rules = self.sessions.begin(sid, prompt, messages, self.privacy)
        if est.retry_of:
            self._late(est.retry_of, {"retried": True})

        mode = self.config.mode
        features: dict[str, Any] = {"mode": mode, "client_model": client_id, "stream": stream}
        if mode == "shadow" and sensitive and not client_local:
            # The client's model may not see this step, so the router's choice runs: the
            # decision is executed and logged as a route-mode decision.
            mode = "route"
            features.update(mode="route", forced_route="sensitive")
        if dropped_client:
            features["dropped_client_params"] = dropped_client
        tool_names = [
            t["function"]["name"]
            for t in tools or []
            if isinstance(t, Mapping)
            and isinstance(t.get("function"), Mapping)
            and isinstance(t["function"].get("name"), str)
        ]
        state = StepState(
            harness=self.config.harness,
            session_id=sid,
            step_idx=step_idx,
            kind=h.get("x-steproute-kind", "main") or "main",
            prompt_tokens=est.prompt_tokens,
            task_id=h.get("x-steproute-task") or None,
            cached_prefix_tokens=est.cached_prefix_tokens,
            current_model=est.current_model,
            current_effort=est.current_effort,
            tools=tool_names,
            sensitive=sensitive,
            source=source,
            task_text=self._task_text(messages),
            features={"proxy": features},
        )

        decision: Decision | None = None
        try:
            decision = self.source.decide(state)
        except SourceError as exc:
            self._count("source_failures")
            log.warning(
                "decision source %s failed (%s); failing open", self.source.name, exc.reason
            )

        plan = self._plan(decision, mode, client_id, client_effort, client_local, sensitive)
        if isinstance(plan, Response):
            return plan
        attempts, mode_label, outcome_id = plan
        self.sessions.set_decision(sid, outcome_id)
        step = _Step(
            sid,
            prompt,
            params,
            model,
            mode_label,
            outcome_id,
            sensitive,
            attempts,
            self._clock(),
            stream,
        )
        if mode_label == "fail_open":
            self._count("fail_open")
        if stream:
            return self._run_stream(step, client_id, client_effort, client_local)
        return self._run(step, client_id, client_effort, client_local)

    def _task_text(self, messages: list[Any]) -> str | None:
        if self.config.task_text != "last_user":
            return None
        for m in reversed(messages):
            if isinstance(m, Mapping) and m.get("role") == "user":
                c = m.get("content")
                if isinstance(c, str):
                    return c
                if isinstance(c, list):
                    return "\n".join(
                        p["text"]
                        for p in c
                        if isinstance(p, Mapping) and isinstance(p.get("text"), str)
                    )
        return None

    def _plan(
        self,
        decision: Decision | None,
        mode: str,
        client_id: str | None,
        client_effort: str | None,
        client_local: bool,
        sensitive: bool,
    ) -> tuple[list[Attempt], str, str | None] | Response:
        unknown = Response(
            404,
            {},
            _error_body(404, "model_not_found", "the requested model is not served by this proxy"),
        )
        if decision is not None and mode == "route":
            steps = [Attempt(a.model, a.effort, True) for a in decision.action.chain()]
            return steps, "route", decision.decision_id
        if decision is not None:  # shadow
            if client_id is None:
                return unknown
            return (
                [Attempt(client_id, client_effort, False)],
                "shadow",
                SHADOW_PREFIX + decision.decision_id,
            )
        # No decision: fail open to the client's own model, subject to the privacy rule.
        if client_id is not None and (not sensitive or client_local):
            return [Attempt(client_id, client_effort, False)], "fail_open", None
        if sensitive and self.config.local_fallback:
            m = self.config.local_fallback
            return [Attempt(m, default_effort(self.catalog.get(m)), True)], "fail_open", None
        if sensitive:
            self._count("privacy_blocked")
            return Response(
                503,
                {},
                _error_body(
                    503,
                    "steproute_sensitive",
                    "this request looks sensitive, no routing decision is available and no local "
                    "model is configured ([privacy] local_fallback); it was not sent anywhere",
                ),
            )
        return unknown

    def _blocked(self, a: Attempt, sensitive: bool) -> UpstreamConfig | None:
        up = self.config.upstream_for(a.model)
        if up is None:
            a.skipped = "no_upstream"
            return None
        if sensitive and not up.local:
            # Belt and braces: the policy and the plan never get here, but no code path may
            # send a sensitive step to a non-local upstream.
            a.skipped = "privacy"
            self._count("privacy_blocked")
            log.error("refused to send a sensitive step to non-local upstream %r", up.name)
            return None
        return up

    def _price(self, a: Attempt) -> None:
        if a.usage is not None:
            a.cost_usd = price_usage(self.catalog.get(a.model), a.usage).total_usd

    def _record(self, step: _Step, a: Attempt, up: UpstreamConfig) -> None:
        if a.usage is None:
            return
        profile = self.catalog.get(a.model)
        self.sessions.record_attempt(
            step.session_id,
            step.prompt,
            model=a.model,
            effort=a.effort,
            prompt_tokens=int(a.usage.prompt_tokens),
            cache_min_tokens=profile.cache_min_tokens,
            cache_ttl_s=up.cache_ttl_s,
            cacheable=a.cacheable,
        )

    def _fallback_attempt(
        self, step: _Step, client_id: str | None, client_effort: str | None, client_local: bool
    ) -> Attempt | None:
        """After every routed attempt errored: the client's own request, if allowed."""
        if step.mode != "route" or self.config.on_upstream_error != "client_model":
            return None
        if client_id is None or (step.sensitive and not client_local):
            return None
        if any(a.called and not a.rewrite and a.model == client_id for a in step.attempts):
            return None
        step.signals["fallback_to_client"] = True
        a = Attempt(client_id, client_effort, False)
        step.attempts.append(a)
        return a

    # --- non-streaming ---

    def _call(self, step: _Step, a: Attempt) -> dict[str, Any] | None:
        up = self._blocked(a, step.sensitive)
        if up is None:
            return None
        prepared = self.pool.prepare(
            up,
            a.model,
            step.params,
            rewrite=a.rewrite,
            reasoning=self._reasoning(up, a.model, a.effort),
        )
        a.dropped, a.cache, a.cacheable = prepared.dropped, prepared.cache, prepared.cacheable
        a.ran = True
        t0 = self._clock()
        try:
            resp = self.pool.complete(prepared)
        except UpstreamError as exc:
            a.error = exc
            a.latency_ms = (self._clock() - t0) * 1000
            self._count("upstream_errors")
            log.warning("upstream %r failed: %s %s", up.name, exc.status, exc.kind)
            return None
        a.latency_ms = (self._clock() - t0) * 1000
        a.usage = normalise_usage(resp.get("usage"), up.provider)
        self._price(a)
        self._record(step, a, up)
        a.check = run_checks(resp, step.params, self.config.checks)
        return resp

    def _run(
        self, step: _Step, client_id: str | None, client_effort: str | None, client_local: bool
    ) -> Response:
        final: dict[str, Any] | None = None
        last: Attempt | None = None
        planned = list(step.attempts)
        for i, a in enumerate(planned):
            resp = self._call(step, a)
            if resp is None:
                continue
            final, last = resp, a
            assert a.check is not None
            if a.check.failed is None or step.mode != "route":
                break
            if i + 1 < len(planned):
                log.info(
                    "check %s failed on %s; trying the next cascade step", a.check.failed, a.model
                )
        if final is None:
            extra = self._fallback_attempt(step, client_id, client_effort, client_local)
            if extra is not None:
                resp = self._call(step, extra)
                if resp is not None:
                    final, last = resp, extra
        headers = self._headers(step, last)
        if final is None or last is None:
            err = next((a.error for a in reversed(step.attempts) if a.error is not None), None)
            status = err.status if err else 502
            message = (
                f"upstream failed ({err.kind}): {err}"
                if err
                else "no upstream could serve this request (see the proxy log)"
            )
            resp_err = Response(status, headers, _error_body(status, "upstream_error", message))
            resp_err._finish = lambda: self._finish(step, None)
            return resp_err
        final["model"] = step.client_model
        out = Response(200, headers, final)
        out._finish = lambda: self._finish(step, last)
        return out

    def _headers(self, step: _Step, last: Attempt | None) -> dict[str, str]:
        h = {"X-Steproute-Session": step.session_id, "X-Steproute-Mode": step.mode}
        if step.outcome_id:
            h["X-Steproute-Decision-Id"] = step.outcome_id
        if last is not None:
            h["X-Steproute-Model"] = last.model
            if last.effort:
                h["X-Steproute-Effort"] = last.effort
        h["X-Steproute-Attempts"] = str(sum(1 for a in step.attempts if a.called))
        return h

    # --- streaming ---

    def _run_stream(
        self, step: _Step, client_id: str | None, client_effort: str | None, client_local: bool
    ) -> Response:
        """One attempt that produces bytes. A step that fails *before* its first chunk (an
        HTTP error, a refused connection) moves on to the next cascade step; once a byte has
        gone out there is no cascade (``docs/PROXY.md``, "Streaming")."""
        opened: Iterator[dict[str, Any]] | None = None
        chosen: Attempt | None = None
        up_used: UpstreamConfig | None = None
        candidates = list(step.attempts)
        for a in candidates:
            up = self._blocked(a, step.sensitive)
            if up is None:
                continue
            prepared = self.pool.prepare(
                up,
                a.model,
                step.params,
                rewrite=a.rewrite,
                reasoning=self._reasoning(up, a.model, a.effort),
            )
            a.dropped, a.cache, a.cacheable = prepared.dropped, prepared.cache, prepared.cacheable
            a.ran = True
            try:
                opened = self.pool.stream(prepared)
            except UpstreamError as exc:
                a.error = exc
                self._count("upstream_errors")
                log.warning("upstream %r failed: %s %s", up.name, exc.status, exc.kind)
                if step.mode != "route":
                    break
                continue
            chosen, up_used = a, up
            break
        if opened is None:
            extra = self._fallback_attempt(step, client_id, client_effort, client_local)
            if extra is not None:
                up = self._blocked(extra, step.sensitive)
                if up is not None:
                    prepared = self.pool.prepare(
                        up, extra.model, step.params, rewrite=False, reasoning=None
                    )
                    extra.cache, extra.cacheable = prepared.cache, prepared.cacheable
                    extra.ran = True
                    try:
                        opened = self.pool.stream(prepared)
                        chosen, up_used = extra, up
                    except UpstreamError as exc:
                        extra.error = exc
                        self._count("upstream_errors")
        headers = self._headers(step, chosen)
        if opened is None or chosen is None or up_used is None:
            err = next((a.error for a in reversed(step.attempts) if a.error is not None), None)
            status = err.status if err else 502
            message = (
                f"upstream failed ({err.kind}): {err}"
                if err
                else "no upstream could serve this request"
            )
            r = Response(status, headers, _error_body(status, "upstream_error", message))
            r._finish = lambda: self._finish(step, None)
            return r
        options = step.params.get("stream_options")
        client_usage = isinstance(options, Mapping) and options.get("include_usage") is True
        rf = step.params.get("response_format")
        acc = StreamAccumulator(keep_text=self.config.checks.schema and isinstance(rf, Mapping))
        attempt, upstream = chosen, up_used
        t0 = self._clock()
        captured: dict[str, Any] = {}

        def gen() -> Iterator[bytes]:
            try:
                for chunk in opened:
                    usage = chunk.get("usage")
                    if isinstance(usage, Mapping) and "prompt_tokens" in usage:
                        captured["usage"] = dict(usage)
                    acc.add(chunk)
                    if not client_usage:
                        chunk.pop("usage", None)
                        if not chunk.get("choices"):
                            continue
                    chunk["model"] = step.client_model
                    yield _sse(chunk)
                yield _sse("[DONE]")
            except UpstreamError as exc:
                attempt.error = exc
                step.signals["stream_error"] = True
                self._count("upstream_errors")
                yield _sse(
                    _error_body(
                        exc.status, "upstream_error", f"upstream failed mid-stream ({exc.kind})"
                    )
                )
            except GeneratorExit:
                step.signals["client_disconnected"] = True
                raise
            finally:
                attempt.latency_ms = (self._clock() - t0) * 1000

        def finish() -> None:
            attempt.usage = normalise_usage(captured.get("usage"), upstream.provider)
            self._price(attempt)
            self._record(step, attempt, upstream)
            if attempt.error is None:
                attempt.check = run_checks(acc.as_response(), step.params, self.config.checks)
            self._finish(step, attempt)

        headers["Content-Type"] = "text/event-stream"
        headers["Cache-Control"] = "no-cache"
        return Response(200, headers, stream=gen(), _finish=finish)

    # --- outcome ---

    def _finish(self, step: _Step, last: Attempt | None) -> None:
        if step.outcome_id is None:
            return  # fail-open without a decision: nothing was logged, so nothing to join
        called = [a for a in step.attempts if a.called]
        signals: dict[str, Any] = dict(step.signals)
        signals["proxy_mode"] = step.mode
        signals["attempts"] = len(called)
        failed = [a.check.failed for a in called if a.check is not None and a.check.failed]
        if failed:
            signals["failed_checks"] = failed
        errors = sum(1 for a in called if a.error is not None)
        if errors:
            signals["upstream_errors"] = errors
        skipped = sorted({a.skipped for a in step.attempts if a.skipped})
        if skipped:
            signals["skipped"] = skipped
        dropped = sorted({d for a in called for d in a.dropped})
        if dropped:
            signals["params_dropped"] = dropped
        if step.stream:
            signals["stream"] = True
        if last is not None and last.cache != "none":
            signals["cache_markers"] = last.cache
        # Label evidence comes from the final answer only, and never from the client's own
        # model standing in for a failed routed chain.
        if (
            last is not None
            and last.check is not None
            and not step.signals.get("fallback_to_client")
        ):
            signals.update(signals_for(last.check))
        responded = [a for a in called if a.usage is not None]
        outcome = Outcome(
            decision_id=step.outcome_id,
            ts=self._clock(),
            success=None,
            signals=signals,
            final_model=last.model if last is not None else None,
            escalated=len(called) > 1,
            input_tokens=int(sum(a.usage.input_tokens for a in responded if a.usage)),
            output_tokens=int(sum(a.usage.output_tokens for a in responded if a.usage)),
            cache_read_tokens=int(sum(a.usage.cache_read_tokens for a in responded if a.usage)),
            cache_write_tokens=int(sum(a.usage.cache_write_tokens for a in responded if a.usage)),
            cost_usd=sum(a.cost_usd for a in responded) if responded else None,
            latency_ms=(self._clock() - step.t0) * 1000,
        )
        self._submit(outcome)

    def _submit(self, outcome: Outcome) -> Outcome | None:
        with self._lock:
            self._outcomes[outcome.decision_id] = outcome
            self._outcomes.move_to_end(outcome.decision_id)
            while len(self._outcomes) > _MAX_OUTCOMES:
                self._outcomes.popitem(last=False)
        try:
            return self.source.report(outcome)
        except SourceError as exc:
            self._count("report_failures")
            log.warning("could not report an outcome (%s)", exc.reason)
            return None

    def _late(
        self, decision_id: str, signals: Mapping[str, Any], success: bool | None = None
    ) -> Outcome | None:
        with self._lock:
            old = self._outcomes.get(decision_id)
        if old is None:
            return None
        merged = Outcome(
            decision_id=old.decision_id,
            ts=self._clock(),
            success=success,
            signals={**old.signals, **signals},
            final_model=old.final_model,
            escalated=old.escalated,
            input_tokens=old.input_tokens,
            output_tokens=old.output_tokens,
            cache_read_tokens=old.cache_read_tokens,
            cache_write_tokens=old.cache_write_tokens,
            cost_usd=old.cost_usd,
            latency_ms=old.latency_ms,
        )
        stored = self._submit(merged)
        return stored or merged

    def report(self, body: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """``POST /report``: late signals for an outcome this proxy emitted."""
        extra = sorted(set(body) - {"decision_id", "signals", "success"})
        did, signals, success = (
            body.get("decision_id"),
            body.get("signals", {}),
            body.get("success"),
        )
        if (
            extra
            or not isinstance(did, str)
            or not isinstance(signals, Mapping)
            or not (success is None or isinstance(success, bool))
        ):
            return 400, {
                "error": "bad_request",
                "message": "expected {decision_id: str, signals: object, success?: bool|null}",
            }
        try:
            json.dumps(signals, allow_nan=False)
        except (TypeError, ValueError):
            return 400, {"error": "bad_request", "message": "signals must be JSON values"}
        merged = self._late(did, signals, success)
        if merged is None:
            return 404, {
                "error": "unknown_decision",
                "message": "this proxy holds no outcome for that decision_id (expired, restarted, "
                "or never emitted); report to the service directly with a full Outcome",
            }
        return 200, merged.to_dict()
