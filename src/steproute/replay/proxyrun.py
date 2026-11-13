"""One run's private proxy, and the chat client that enforces every limit on every call.

Each run gets its own in-process proxy (``steproute.collectors.proxy``) on a fresh loopback
port, built for exactly one configuration:

* the decision source is the service with a ``FixedPolicy`` of the run's ``(model, effort)``,
  so every step is a normal route-mode decision and outcome in the trace store, with
  ``source = "replay"``;
* the only upstream serves only the run's model, so no other model is reachable;
* ``on_no_decision = "error"`` and ``on_upstream_error = "error"``: the proxy never fails open
  to another configuration and never falls back to the client's model;
* a fresh ``SessionTable``, so nothing about the cache carries over from another run.

``ProxyChatClient`` makes each call through it and, around each paid call, writes the ledger
intent (worst case) before and the settled cost after. It checks the response headers and
the outcome against the planned configuration and raises ``ConfigMismatch`` on any
difference. Standard library only (the proxy itself needs the ``[proxy]`` extra at call time).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from steproute.collectors.proxy.config import CheckConfig, ProxyConfig, UpstreamConfig
from steproute.collectors.proxy.engine import Proxy
from steproute.collectors.proxy.server import make_proxy_server
from steproute.collectors.proxy.source import InProcessSource
from steproute.collectors.proxy.upstream import UpstreamPool
from steproute.core.catalog import Catalog
from steproute.core.cost import AttemptUsage, price_usage
from steproute.core.policy import FixedPolicy, Policy
from steproute.core.privacy import PrivacyConfig
from steproute.core.schema import Action, Outcome
from steproute.core.service import Service
from steproute.core.store import TraceStore
from steproute.replay.agent import ChatReply, RunStopped
from steproute.replay.config import Limits, ProviderConfig
from steproute.replay.ledger import Ledger
from steproute.replay.plan import RunSpec

__all__ = [
    "CallRecord",
    "ConfigMismatch",
    "ProxyChatClient",
    "RunProxy",
    "Redactor",
    "worst_call_usd",
]


class ConfigMismatch(RuntimeError):
    """What executed is not what was planned. The run fails and the runner stops."""


class Redactor:
    """Replaces every configured key value with ``[redacted]`` in text the runner writes."""

    def __init__(self, secrets: Sequence[str]) -> None:
        self.secrets = sorted({s for s in secrets if len(s) >= 6}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for s in self.secrets:
            text = text.replace(s, "[redacted]")
        return text


def worst_call_usd(profile_price: Any, limits: Limits, *, local: bool) -> float:
    """The most one call can cost: a ``max_prompt_tokens`` prompt billed entirely at the
    highest input rate (a cache write where writes cost extra, at the highest price tier) plus
    ``max_output_tokens`` of output. Free for a local model."""
    if local:
        return 0.0
    prompt = float(limits.max_prompt_tokens)
    if profile_price.cache_write_mult > 1.0:
        usage = AttemptUsage(cache_write_tokens=prompt, output_tokens=limits.max_output_tokens)
    else:
        usage = AttemptUsage(input_tokens=prompt, output_tokens=limits.max_output_tokens)
    return float(price_usage(profile_price, usage).total_usd)


class _Capture(InProcessSource):
    """Records each outcome as the proxy produced it (before the service labels and prices it),
    so the runner sees a missing usage as missing, never as $0."""

    def __init__(self, service: Service) -> None:
        super().__init__(service)
        self.outcomes: dict[str, Outcome] = {}
        self.decisions: dict[str, Action] = {}
        self._cv = threading.Condition()

    def decide(self, state: Any) -> Any:
        d = super().decide(state)
        with self._cv:
            self.decisions[d.decision_id] = d.action
        return d

    def report(self, outcome: Outcome) -> Outcome:
        with self._cv:
            self.outcomes.setdefault(outcome.decision_id, outcome)
            self._cv.notify_all()
        return super().report(outcome)

    def wait(self, decision_id: str, timeout_s: float) -> Outcome | None:
        deadline = time.monotonic() + timeout_s
        with self._cv:
            while decision_id not in self.outcomes:
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return self.outcomes[decision_id]


def upstream_for(spec: RunSpec, provider: ProviderConfig) -> UpstreamConfig:
    if spec.provider == "anthropic":
        return UpstreamConfig(
            name="anthropic",
            provider="anthropic",
            models=(spec.model,),
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            drop_params=("temperature", "top_p", "top_k"),
            forced_tool_choice="auto",
            cache_ttl_s=300.0,
            timeout_s=provider.timeout_s,
        )
    if spec.provider == "deepseek":
        return UpstreamConfig(
            name="deepseek",
            provider="deepseek",
            models=(spec.model,),
            base_url=provider.base_url,
            api_key_env=provider.api_key_env,
            cache_ttl_s=3600.0,
            timeout_s=provider.timeout_s,
        )
    if spec.local:
        return UpstreamConfig(
            name=spec.provider,
            provider="openai",
            models=(spec.model,),
            base_url=provider.base_url,
            local=True,
            cache_ttl_s=300.0,
            timeout_s=provider.timeout_s,
        )
    raise ValueError(f"no upstream recipe for provider {spec.provider!r}")


class RunProxy:
    """The proxy of one run, serving on ``127.0.0.1:<port>``. A context manager."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        catalog: Catalog,
        store: TraceStore,
        provider: ProviderConfig,
        environ: Mapping[str, str],
        policy: Policy | None = None,
    ) -> None:
        self.spec = spec
        action = Action(spec.model, spec.effort)
        fixed = policy or FixedPolicy(
            action, catalog, include_holdout=spec.holdout, name="replay-fixed"
        )
        service = Service(
            policy=fixed,
            catalog=catalog,
            store=store,
            privacy=PrivacyConfig(),
            default_action=None,
            local_fallback=None,
            budget_ms=5000.0,
        )
        self.capture = _Capture(service)
        up = upstream_for(spec, provider)
        env = {up.api_key_env: environ.get(up.api_key_env, "")} if up.api_key_env else {}
        config = ProxyConfig(
            host="127.0.0.1",
            port=0,
            mode="route",
            harness="replay",
            on_upstream_error="error",
            on_no_decision="error",
            checks=CheckConfig(),
            upstreams=(up,),
        )
        self.proxy = Proxy(
            config,
            source=self.capture,
            catalog=catalog,
            privacy=PrivacyConfig(),
            pool=UpstreamPool(env),
        )
        self._server = make_proxy_server(self.proxy, None, host="127.0.0.1", port=0)
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )

    def __enter__(self) -> RunProxy:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def label(
        self, decision_ids: Sequence[str], signals: Mapping[str, Any], success: bool | None
    ) -> int:
        """Copy the task-level result onto every step's outcome (``POST /report`` semantics)."""
        n = 0
        for did in decision_ids:
            body: dict[str, Any] = {"decision_id": did, "signals": dict(signals)}
            if success is not None:
                body["success"] = success
            status, _ = self.proxy.report(body)
            n += status == 200
        return n


@dataclass
class CallRecord:
    decision_id: str | None
    usd: float
    flags: list[str]
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


_TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 520, 529}


@dataclass
class ProxyChatClient:
    """The ``ChatClient`` of one run. Raises ``RunStopped`` to end the run."""

    spec: RunSpec
    proxy: RunProxy
    session_id: str
    run_id: str
    limits: Limits
    worst_call: float
    ledger: Ledger | None
    stop: threading.Event
    started: float = field(default_factory=time.monotonic)
    sleep: Callable[[float], None] = time.sleep
    outcome_timeout_s: float = 30.0
    calls: list[CallRecord] = field(default_factory=list)
    spent: float = 0.0
    _last_prompt_tokens: int = 0
    _last_chars: int = 0

    # --- limits ---

    def _prompt_estimate(self, messages: Sequence[Mapping[str, Any]]) -> tuple[int, float]:
        chars = len(json.dumps(list(messages), separators=(",", ":")))
        if self._last_prompt_tokens > 0 and self._last_chars > 0:
            ratio = min(8.0, max(1.0, self._last_chars / self._last_prompt_tokens))
        else:
            ratio = 4.0
        if not self.spec.local:
            ratio /= 2.0  # paid: over-estimate on purpose; this bounds the worst call
        new = max(0, chars - self._last_chars) if self._last_prompt_tokens else chars
        return chars, self._last_prompt_tokens + new / ratio

    def _check_limits(self, messages: Sequence[Mapping[str, Any]]) -> int:
        if self.stop.is_set():
            raise RunStopped("interrupted", "the runner is stopping")
        if self.spent >= self.limits.run_cap_usd:
            raise RunStopped(
                "run_cap",
                f"run spend ${self.spent:.4f} reached the ${self.limits.run_cap_usd:g} cap",
            )
        if time.monotonic() - self.started > self.limits.run_timeout_s:
            raise RunStopped("wall_time", f"the run passed {self.limits.run_timeout_s:g} s")
        chars, est = self._prompt_estimate(messages)
        if self.spec.local:
            cap = self.limits.local_context_cap - 2048
            if est > cap:
                raise RunStopped(
                    "context_limit", f"prompt of about {est:.0f} tokens passes the local cap"
                )
        elif est > self.limits.max_prompt_tokens:
            raise RunStopped(
                "context_limit",
                f"prompt of about {est:.0f} tokens would pass max_prompt_tokens "
                f"({self.limits.max_prompt_tokens})",
            )
        return chars

    # --- one call ---

    def _post(self, body: Mapping[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        data = json.dumps(body).encode()
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(data)),
            "X-Steproute-Session": self.session_id,
            "X-Steproute-Task": self.spec.task_id,
            "X-Steproute-Source": "replay",
            "X-Steproute-Kind": "main",
        }
        conn = http.client.HTTPConnection(
            "127.0.0.1", self.proxy.port, timeout=self.limits.run_timeout_s
        )
        try:
            conn.request("POST", "/v1/chat/completions", body=data, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            hdrs = {k.lower(): v for k, v in resp.getheaders()}
        finally:
            conn.close()
        try:
            obj = json.loads(raw)
        except ValueError:
            obj = {}
        return resp.status, hdrs, obj if isinstance(obj, dict) else {}

    def _verify(self, headers: Mapping[str, str], outcome: Outcome | None) -> None:
        mode = headers.get("x-steproute-mode")
        model = headers.get("x-steproute-model")
        effort = headers.get("x-steproute-effort")
        planned = (self.spec.model, self.spec.effort)
        problems = []
        if mode != "route":
            problems.append(f"proxy mode {mode!r} (expected 'route')")
        if (model, effort) != planned:
            problems.append(f"executed {model} ({effort})")
        if outcome is not None and outcome.final_model not in (None, self.spec.model):
            problems.append(f"outcome final_model {outcome.final_model}")
        did = headers.get("x-steproute-decision-id")
        action = self.proxy.capture.decisions.get(did or "")
        if action is not None and (action.model, action.effort) != planned:
            problems.append(f"decision {action.model} ({action.effort})")
        if problems:
            raise ConfigMismatch(
                f"run {self.run_id}: planned {planned[0]} ({planned[1]}) but " + "; ".join(problems)
            )

    def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> ChatReply:
        chars = self._check_limits(messages)
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": list(messages),
            "tools": list(tools),
            "max_tokens": self.limits.max_output_tokens,
        }
        attempt = 0
        while True:
            intent = None
            if self.ledger is not None and not self.spec.local:
                intent = self.ledger.intend(self.spec.provider, self.worst_call, self.run_id)
            t0 = time.monotonic()
            try:
                status, headers, obj = self._post(body)
            except (OSError, http.client.HTTPException) as exc:
                # The local proxy itself did not answer: nothing reached the provider through
                # it that we know of, but a call may have been in flight. Charge the worst case.
                if intent is not None and self.ledger is not None:
                    self.ledger.settle(intent, self.worst_call, ["proxy_unreachable"])
                    self._charged(self.worst_call)
                raise RunStopped(
                    "infra_error", f"the run's proxy failed: {type(exc).__name__}"
                ) from None
            latency = (time.monotonic() - t0) * 1000
            if status == 200:
                did = headers.get("x-steproute-decision-id")
                outcome = self.proxy.capture.wait(did, self.outcome_timeout_s) if did else None
                usd, flags, rec = self._cost(outcome)
                rec.latency_ms = latency
                if intent is not None and self.ledger is not None:
                    self.ledger.settle(intent, usd, flags)
                self._charged(usd)
                self.calls.append(rec)
                self._verify(headers, outcome)
                if rec.input_tokens + rec.cache_read_tokens + rec.cache_write_tokens > 0:
                    self._last_prompt_tokens = (
                        rec.input_tokens + rec.cache_read_tokens + rec.cache_write_tokens
                    )
                    self._last_chars = chars
                choices = obj.get("choices") or [{}]
                first = choices[0] if isinstance(choices[0], Mapping) else {}
                message = first.get("message") if isinstance(first.get("message"), Mapping) else {}
                return ChatReply(dict(message or {}), first.get("finish_reason"))
            # An error answer: the provider either refused before doing work (not billed) or
            # timed out (possibly billed): a timeout is charged at the worst case.
            err = obj.get("error") if isinstance(obj.get("error"), Mapping) else {}
            kind = str((err or {}).get("type", ""))
            timeout = status in (408, 504)
            usd = self.worst_call if timeout else 0.0
            if intent is not None and self.ledger is not None:
                self.ledger.settle(intent, usd, ["timeout"] if timeout else [f"http_{status}"])
            self._charged(usd)
            if kind == "steproute_sensitive":
                raise RunStopped(
                    "blocked_sensitive",
                    "the proxy found secret-like content and refused to send it to a cloud model",
                )
            if kind == "steproute_no_decision":
                raise RunStopped("infra_error", "the fixed policy returned no decision")
            if status in _TRANSIENT and attempt < self.limits.call_retries:
                attempt += 1
                self.sleep(self.limits.call_backoff_s * attempt)
                if self.spent >= self.limits.run_cap_usd:
                    raise RunStopped("run_cap", "run cap reached while retrying a call")
                continue
            raise RunStopped(
                "provider_error", f"provider answered HTTP {status} ({kind or 'error'})"
            )

    def _charged(self, usd: float) -> None:
        self.spent += usd

    def _cost(self, outcome: Outcome | None) -> tuple[float, list[str], CallRecord]:
        did = outcome.decision_id if outcome else None
        if outcome is None:
            return (
                self.worst_call,
                ["outcome_missing"],
                CallRecord(did, self.worst_call, ["outcome_missing"]),
            )
        tokens = outcome.input_tokens + outcome.cache_read_tokens + outcome.cache_write_tokens
        rec = CallRecord(
            did,
            0.0,
            [],
            outcome.input_tokens,
            outcome.cache_read_tokens,
            outcome.cache_write_tokens,
            outcome.output_tokens,
        )
        if outcome.cost_usd is None or tokens <= 0 or outcome.signals.get("usage_missing"):
            # Missing usage: never zero. Charge the worst case and flag it.
            rec.usd, rec.flags = self.worst_call, ["missing_usage"]
            return self.worst_call, ["missing_usage"], rec
        rec.usd = float(outcome.cost_usd)
        return rec.usd, [], rec
