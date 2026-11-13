"""Local HTTP service: ``POST /decide``, ``POST /report``, ``GET /models``, ``GET /healthz``.

Standard library only (``http.server``). See ``docs/SERVICE.md`` for the API, the config and
the fail-open rules. The pieces:

* ``ServiceConfig`` / ``load_service_config``: the TOML config, strict about unknown keys.
* ``Service``: the HTTP-free logic (decide with a time budget and a privacy-safe fail-open,
  report with labelling and cost filling). Unit-testable without a socket.
* ``make_server`` / ``serve``: the threaded HTTP/1.1 front end.

The service sits in the request path of every LLM call of its clients, so it must answer
quickly, never take the client down, and never weaken the privacy constraints: a sensitive
state never receives a cloud model, not even through the fail-open path.
"""

from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import json
import logging
import os
import signal
import socket
import threading
import time
import tomllib
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

from steproute import __version__
from steproute.core.actions import NoFeasibleAction, filter_feasible
from steproute.core.catalog import Catalog, load_catalog, load_default_catalog
from steproute.core.cost import CostModelError, realized_cost
from steproute.core.labels import apply_label
from steproute.core.policy import FixedPolicy, Policy
from steproute.core.privacy import PrivacyConfig, mark_before_routing, redact_state
from steproute.core.rules import RuleConfigError, RulePolicy, load_rule_config
from steproute.core.schema import (
    EFFORTS,
    SCHEMA_VERSION,
    Action,
    Decision,
    Outcome,
    StepState,
    UnknownFieldWarning,
    new_decision_id,
)
from steproute.core.store import TraceStore

__all__ = [
    "FAIL_OPEN_PREFIX",
    "NoActionAllowed",
    "Service",
    "ServiceConfig",
    "ServiceConfigError",
    "build_service",
    "load_service_config",
    "make_server",
    "serve",
]

log = logging.getLogger("steproute.service")

#: ``Decision.policy`` of a fail-open decision is ``"fail_open:<reason>"``.
FAIL_OPEN_PREFIX = "fail_open:"

DEFAULT_PORT = 8787
DEFAULT_BUDGET_MS = 50.0
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_MAX_ABANDONED = 8
#: Seconds a connection may stay silent (idle keep-alive or a stalled upload) before it is cut.
CONNECTION_TIMEOUT_S = 30.0


class ServiceConfigError(ValueError):
    """The service configuration is invalid; the service must not start."""


class NoActionAllowed(Exception):
    """Neither the policy nor the configured fallbacks can route this state (HTTP 503)."""

    def __init__(self, sensitive: bool, configured: bool = True) -> None:
        self.sensitive = sensitive
        #: False when the operator configured no fallback action at all.
        self.configured = configured
        super().__init__("no action is allowed for this state")


# --- config ---------------------------------------------------------------------------------

ActionSpec = tuple[str, str]  # (model, effort)


@dataclass(frozen=True)
class ServiceConfig:
    """Everything ``steproute serve`` reads from its TOML file. Every field is optional."""

    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    token_env: str | None = None
    store_path: Path | None = None
    catalog_path: Path | None = None
    policy_kind: str = "rules"
    rules_config: Path | None = None
    policy_action: ActionSpec | None = None
    budget_ms: float = DEFAULT_BUDGET_MS
    max_abandoned: int = DEFAULT_MAX_ABANDONED
    default_action: ActionSpec | None = None
    local_fallback: ActionSpec | None = None
    capture_text: bool = False
    hash_text: bool = False
    salt_env: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.port, bool) or not 0 <= self.port <= 65535:
            raise ServiceConfigError(f"server.port must be 0..65535, got {self.port!r}")
        if self.max_body_bytes < 1:
            raise ServiceConfigError("server.max_body_bytes must be at least 1")
        if not self.budget_ms > 0:
            raise ServiceConfigError("failopen.budget_ms must be positive")
        if self.max_abandoned < 1:
            raise ServiceConfigError("failopen.max_abandoned must be at least 1")
        if self.policy_kind not in ("rules", "fixed"):
            raise ServiceConfigError(
                f"policy.kind must be 'rules' or 'fixed', got {self.policy_kind!r}"
            )
        for key, value in (("auth.token_env", self.token_env), ("privacy.salt_env", self.salt_env)):
            if value is not None and not _is_env_name(value):
                # Deliberately not echoed: someone may have pasted the secret itself.
                raise ServiceConfigError(
                    f"{key} must be the NAME of an environment variable (letters, digits, _), "
                    "not a secret value"
                )


def _is_env_name(value: str) -> bool:
    return (
        bool(value)
        and not value[0].isdigit()
        and all(c.isascii() and (c.isalnum() or c == "_") for c in value)
    )


# section -> {toml key: (ServiceConfig field, kind)}
_SCHEMA: dict[str, dict[str, tuple[str, str]]] = {
    "server": {
        "host": ("host", "str"),
        "port": ("port", "int"),
        "max_body_bytes": ("max_body_bytes", "int"),
    },
    "auth": {"token_env": ("token_env", "str")},
    "store": {"path": ("store_path", "path")},
    "catalog": {"path": ("catalog_path", "path")},
    "policy": {
        "kind": ("policy_kind", "str"),
        "rules_config": ("rules_config", "path"),
        "action": ("policy_action", "action"),
    },
    "failopen": {
        "budget_ms": ("budget_ms", "num"),
        "max_abandoned": ("max_abandoned", "int"),
        "default": ("default_action", "action"),
        "local_fallback": ("local_fallback", "action"),
    },
    "privacy": {
        "capture_text": ("capture_text", "bool"),
        "hash_text": ("hash_text", "bool"),
        "salt_env": ("salt_env", "str"),
    },
}


def _coerce(where: str, value: Any, kind: str, base: Path) -> Any:
    ok = False
    if kind == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif kind == "num":
        ok = isinstance(value, int | float) and not isinstance(value, bool)
        if ok:
            return float(value)
    elif kind == "bool":
        ok = isinstance(value, bool)
    elif kind == "str":
        ok = isinstance(value, str) and bool(value)
    elif kind == "path":
        ok = isinstance(value, str) and bool(value)
        if ok:
            return (base / Path(value).expanduser()).resolve()
    elif kind == "action":
        if not isinstance(value, Mapping):
            raise ServiceConfigError(f"{where}: expected a table with 'model' and 'effort'")
        extra = sorted(set(value) - {"model", "effort"})
        if extra:
            raise ServiceConfigError(f"{where}: unknown key(s) {extra}; expected model, effort")
        model, effort = value.get("model"), value.get("effort", "medium")
        if not isinstance(model, str) or not model:
            raise ServiceConfigError(f"{where}: 'model' is required and must be a string")
        if effort not in EFFORTS:
            raise ServiceConfigError(f"{where}: effort must be one of {list(EFFORTS)}")
        return (model, effort)
    if not ok:
        raise ServiceConfigError(f"{where}: wrong type or empty value (expected {kind})")
    return value


def load_service_config(source: str | Path | Mapping[str, Any] | None = None) -> ServiceConfig:
    """Load a ``ServiceConfig`` from a TOML file, a parsed mapping, or ``None`` (defaults).

    Missing keys keep their defaults; unknown sections or keys raise ``ServiceConfigError``.
    Relative paths in a file resolve against the file's directory (against the current
    directory for a mapping).
    """
    base = Path.cwd()
    if source is None:
        data: Mapping[str, Any] = {}
    elif isinstance(source, Mapping):
        data = source
    else:
        path = Path(source)
        try:
            data = tomllib.loads(path.read_text("utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ServiceConfigError(f"{path}: invalid TOML: {exc}") from None
        except OSError as exc:
            raise ServiceConfigError(f"cannot read config: {exc}") from None
        base = path.resolve().parent
    values: dict[str, Any] = {}
    for section, body in data.items():
        if section not in _SCHEMA:
            raise ServiceConfigError(f"unknown section [{section}]; expected {sorted(_SCHEMA)}")
        if not isinstance(body, Mapping):
            raise ServiceConfigError(f"[{section}] must be a table")
        for key, value in body.items():
            if key not in _SCHEMA[section]:
                raise ServiceConfigError(
                    f"[{section}] unknown key {key!r}; expected {sorted(_SCHEMA[section])}"
                )
            field_name, kind = _SCHEMA[section][key]
            values[field_name] = _coerce(f"[{section}] {key}", value, kind, base)
    return ServiceConfig(**values)


# --- the HTTP-free logic --------------------------------------------------------------------

_PROBE = StepState("probe", "probe", 0, "main", 0)


def _check_action(label: str, spec: ActionSpec, catalog: Catalog, *, local: bool) -> Action:
    action = Action(*spec)
    try:
        filter_feasible(_PROBE, (action,), catalog)
    except ValueError as exc:
        raise ServiceConfigError(f"{label}: {exc}") from None
    except NoFeasibleAction:
        raise ServiceConfigError(
            f"{label}: {action.label()} is never allowed (holdout models are refused)"
        ) from None
    if local and not catalog.get(action.model).local:
        raise ServiceConfigError(f"{label}: {action.model!r} is not a local model")
    return action


class _Call:
    """One policy call running in a worker thread, with a lock-protected hand-off."""

    __slots__ = ("abandoned", "done", "error", "lock", "result")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.done = False
        self.abandoned = False
        self.result: object = None
        self.error: BaseException | None = None


class Service:
    """Routing decisions and outcome recording, without any HTTP.

    ``decide`` runs ``policy.decide`` in a worker thread and waits at most ``budget_ms``. On an
    exception, a timeout, a malformed result or a result that breaks the privacy rule, it
    answers with the configured default action (or the local fallback) instead; see
    ``docs/SERVICE.md``. The request path logs the decision it returns; the worker never logs
    anything, so the late result of an abandoned call can never be recorded.
    """

    def __init__(
        self,
        *,
        policy: Policy,
        catalog: Catalog,
        store: TraceStore,
        privacy: PrivacyConfig,
        default_action: Action | None,
        local_fallback: Action | None,
        budget_ms: float = DEFAULT_BUDGET_MS,
        max_abandoned: int = DEFAULT_MAX_ABANDONED,
        clock: Callable[[], float] = time.time,
        new_id: Callable[[], str] = new_decision_id,
    ) -> None:
        self.policy = policy
        self.catalog = catalog
        self.store = store
        self.privacy = privacy
        self.default_action = default_action
        self.local_fallback = local_fallback
        self.budget_s = budget_ms / 1000.0
        self.max_abandoned = max_abandoned
        self._clock = clock
        self._new_id = new_id
        self._lock = threading.Lock()
        self._abandoned = 0
        self.fail_open_count = 0
        self.store_errors = 0

    @property
    def policy_name(self) -> str:
        return str(getattr(self.policy, "name", type(self.policy).__name__))

    @property
    def abandoned_running(self) -> int:
        """Abandoned policy calls that are still running."""
        return self._abandoned

    # --- decide ---

    def decide(self, state: StepState) -> Decision:
        """The decision to return for ``state``, already written to the store.

        Raises ``NoActionAllowed`` when the policy failed and no fallback action is allowed.
        The returned decision is the stored form: ``task_text`` is dropped unless the privacy
        config captures text.
        """
        decision, reason = self._run_policy(state)
        if decision is None:
            decision = self._fail_open(state, reason)
        stored = dataclasses.replace(
            decision, state=redact_state(decision.state, self.store.privacy)
        )
        try:
            self.store.log_decision(stored)
        except Exception as exc:  # a full disk must not stop routing
            self.store_errors += 1
            log.error("could not write a decision to the trace store: %s", type(exc).__name__)
        return stored

    def _run_policy(self, state: StepState) -> tuple[Decision | None, str]:
        with self._lock:
            stuck = self._abandoned >= self.max_abandoned
        if stuck:
            log.warning("fail-open: %d abandoned policy calls are still running", self._abandoned)
            return None, "stuck"
        call = _Call()

        def work() -> None:
            result: object = None
            error: BaseException | None = None
            try:
                result = self.policy.decide(state)
            except BaseException as exc:
                error = exc
            with call.lock:
                call.result, call.error, call.done = result, error, True
                if call.abandoned:
                    with self._lock:
                        self._abandoned -= 1

        thread = threading.Thread(target=work, name="steproute-policy", daemon=True)
        thread.start()
        thread.join(self.budget_s)
        with call.lock:
            if not call.done:
                call.abandoned = True
                with self._lock:
                    self._abandoned += 1
                log.warning(
                    "fail-open: policy %r exceeded the %.0f ms budget; the call is abandoned",
                    self.policy_name,
                    self.budget_s * 1000,
                )
                return None, "timeout"
            result, error = call.result, call.error
        if error is not None:
            reason = "no_feasible" if isinstance(error, NoFeasibleAction) else "error"
            # Only the exception type: its message could contain request content.
            log.warning("fail-open: policy %r raised %s", self.policy_name, type(error).__name__)
            return None, reason
        if not isinstance(result, Decision):
            log.error("fail-open: policy %r returned %s", self.policy_name, type(result).__name__)
            return None, "bad_result"
        if (state.sensitive or result.state.sensitive) and not self._all_local(result.action):
            log.error(
                "fail-open: policy %r chose a non-local model for a sensitive state",
                self.policy_name,
            )
            return None, "policy_violation"
        return result, ""

    def _all_local(self, action: Action) -> bool:
        for step in action.chain():
            if step.model not in self.catalog or not self.catalog.get(step.model).local:
                return False
        return True

    def _fail_open(self, state: StepState, reason: str) -> Decision:
        """The fallback decision; every candidate goes through the hard constraints."""
        marked = mark_before_routing(state, self.privacy)
        self.fail_open_count += 1
        if self.default_action is None and self.local_fallback is None:
            log.error("fail-open (%s): no fallback action is configured", reason)
            raise NoActionAllowed(marked.sensitive, configured=False)
        for action in (self.default_action, self.local_fallback):
            if action is None:
                continue
            try:
                filter_feasible(marked, (action,), self.catalog)
            except NoFeasibleAction:
                continue
            # Belt and braces: the filter already enforces this.
            if marked.sensitive and not self._all_local(action):
                continue  # pragma: no cover
            return Decision(
                decision_id=self._new_id(),
                ts=self._clock(),
                state=marked,
                candidates=[action],
                action=action,
                policy=f"{FAIL_OPEN_PREFIX}{reason}",
                propensity=1.0,
            )
        log.error("fail-open (%s): no configured fallback action is allowed for this state", reason)
        raise NoActionAllowed(marked.sensitive)

    # --- report ---

    def report(self, outcome: Outcome) -> Outcome:
        """Label and price ``outcome``, store it, tell the policy, return what was stored."""
        outcome = apply_label(outcome)
        if outcome.cost_usd is None and not outcome.escalated and outcome.final_model:
            if outcome.final_model in self.catalog:
                try:
                    cost = realized_cost(outcome, self.catalog.get(outcome.final_model))
                    outcome = dataclasses.replace(outcome, cost_usd=cost)
                except CostModelError:
                    pass  # leave it unset rather than guess
        self.store.log_outcome(outcome)
        try:
            self.policy.report(outcome)
        except Exception as exc:  # the outcome is stored; a learner's failure is not the client's
            log.warning("policy %r failed in report: %s", self.policy_name, type(exc).__name__)
        return outcome

    # --- read-only views ---

    def models(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.catalog]

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "schema_version": SCHEMA_VERSION,
            "policy": self.policy_name,
            "fail_open_count": self.fail_open_count,
            "abandoned_policy_calls": self._abandoned,
            "store_errors": self.store_errors,
        }


def build_service(
    config: ServiceConfig,
    *,
    environ: Mapping[str, str] | None = None,
    policy: Policy | None = None,
    catalog: Catalog | None = None,
) -> tuple[Service, bytes | None]:
    """Validate ``config`` and build the ``Service`` plus the bearer token (or ``None``).

    ``policy`` and ``catalog`` override what the config selects (used by tests and embedders;
    an injected policy should use the same ``PrivacyConfig`` as the config). Raises
    ``ServiceConfigError`` for anything that must stop the service from starting.
    """
    env = os.environ if environ is None else environ
    if catalog is None:
        try:
            catalog = (
                load_catalog(config.catalog_path) if config.catalog_path else load_default_catalog()
            )
        except (ValueError, OSError) as exc:
            raise ServiceConfigError(f"catalog: {exc}") from None

    token: bytes | None = None
    if config.token_env:
        value = env.get(config.token_env, "")
        if not value:
            raise ServiceConfigError(
                f"auth.token_env names {config.token_env}, which is not set or is empty"
            )
        token = value.encode("utf-8")
    if not _is_loopback(config.host) and token is None:
        raise ServiceConfigError(
            f"refusing to listen on non-loopback address {config.host!r} without a bearer "
            "token; set [auth] token_env"
        )

    salt = ""
    if config.salt_env:
        salt = env.get(config.salt_env, "")
        if not salt:
            raise ServiceConfigError(
                f"privacy.salt_env names {config.salt_env}, which is not set or is empty"
            )
    try:
        privacy = PrivacyConfig(
            capture_text=config.capture_text, hash_text=config.hash_text, salt=salt
        )
    except ValueError as exc:
        raise ServiceConfigError(f"privacy: {exc}") from None

    default = (
        _check_action("failopen.default", config.default_action, catalog, local=False)
        if config.default_action
        else None
    )
    fallback = (
        _check_action("failopen.local_fallback", config.local_fallback, catalog, local=True)
        if config.local_fallback
        else None
    )
    if fallback is None and (default is None or not catalog.get(default.model).local):
        log.warning(
            "no local fail-open action is configured: if the policy fails, a sensitive state "
            "gets HTTP 503 %s instead of a decision",
            "no_action_allowed" if default else "no_fallback_configured",
        )

    if policy is None:
        try:
            if config.policy_kind == "rules":
                policy = RulePolicy(catalog, load_rule_config(config.rules_config), privacy=privacy)
            else:
                if config.policy_action is None:
                    raise ServiceConfigError("policy.kind = 'fixed' requires [policy] action")
                action = _check_action("policy.action", config.policy_action, catalog, local=False)
                policy = FixedPolicy(action, catalog, privacy=privacy)
        except RuleConfigError as exc:
            raise ServiceConfigError(f"policy: {exc}") from None
        except ValueError as exc:
            if isinstance(exc, ServiceConfigError):
                raise
            raise ServiceConfigError(f"policy: {exc}") from None

    store_path = config.store_path or Path.home() / ".steproute" / "traces"
    store = TraceStore(store_path, privacy=privacy)
    service = Service(
        policy=policy,
        catalog=catalog,
        store=store,
        privacy=privacy,
        default_action=default,
        local_fallback=fallback,
        budget_ms=config.budget_ms,
        max_abandoned=config.max_abandoned,
    )
    return service, token


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# --- HTTP front end -------------------------------------------------------------------------

_JSON = "application/json"
_ROUTES: dict[str, tuple[str, str]] = {
    "/decide": ("POST", "decide"),
    "/report": ("POST", "report"),
    "/models": ("GET", "models"),
    "/healthz": ("GET", "healthz"),
}


def _reject_constant(name: str) -> Any:
    raise ValueError("non-finite number")


def _safe_message(exc: Exception) -> str:
    """A validation message that names the field but never repeats the value."""
    text = str(exc)
    if "missing required field(s)" in text:
        return text  # field names only
    head, sep, _ = text.partition(": ")
    if sep and head and all(c.isalnum() or c in "._[]" for c in head):
        return f"invalid value at {head}"
    return "invalid request body"


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], service: Service, token: bytes | None) -> None:
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.service = service
        self.token = token
        self.max_body = DEFAULT_MAX_BODY_BYTES
        self.allowed_hosts: frozenset[str] | None = None
        super().__init__(address, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError | TimeoutError):
            return
        log.error("connection handler failed: %s", type(exc).__name__)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = CONNECTION_TIMEOUT_S
    server: _Server

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # pragma: no cover
            pass

    def log_message(self, format: str, *args: Any) -> None:
        return None  # never log request lines or bodies

    # Every method goes through one dispatcher so unsupported ones get a 405, not a 501.
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_HEAD(self) -> None:
        self._dispatch("HEAD")

    def do_OPTIONS(self) -> None:
        self._dispatch("OPTIONS")

    # --- responses ---

    def _send(
        self,
        status: int,
        body: Mapping[str, Any],
        *,
        close: bool = False,
        extra: tuple[tuple[str, str], ...] = (),
    ) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        lines = [
            f"HTTP/1.1 {status} {HTTPStatus(status).phrase}",
            f"Content-Type: {_JSON}",
            f"Content-Length: {len(payload)}",
            "Cache-Control: no-store",
        ]
        lines.extend(f"{k}: {v}" for k, v in extra)
        if close:
            lines.append("Connection: close")
            self.close_connection = True
        head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
        try:
            self.wfile.write(head + payload)  # one write: no Nagle/delayed-ACK stall
        except OSError:
            self.close_connection = True

    def _error(
        self,
        status: int,
        code: str,
        message: str,
        *,
        close: bool = False,
        extra: tuple[tuple[str, str], ...] = (),
        **fields: Any,
    ) -> None:
        self._send(status, {"error": code, "message": message, **fields}, close=close, extra=extra)

    # --- request handling ---

    def _dispatch(self, method: str) -> None:
        try:
            self._handle(method)
        except Exception as exc:
            log.error("request failed: %s", type(exc).__name__)
            self._error(500, "internal_error", "internal error", close=True)

    def _handle(self, method: str) -> None:
        srv = self.server
        # A request body we never read must not be parsed as the next request.
        unread = method not in ("GET", "HEAD") or "content-length" in self.headers
        path = urlsplit(self.path).path
        route = _ROUTES.get(path)

        if srv.token is None and srv.allowed_hosts is not None:
            host = self.headers.get("Host")
            if host is not None and _host_name(host) not in srv.allowed_hosts:
                return self._error(403, "forbidden_host", "unexpected Host header", close=unread)
        public = path == "/healthz" and method == "GET"
        if srv.token is not None and not public and not self._authorized(srv.token):
            return self._error(
                401,
                "unauthorized",
                "missing or invalid bearer token",
                close=unread,
                extra=(("WWW-Authenticate", "Bearer"),),
            )
        if route is None:
            return self._error(404, "not_found", "unknown path", close=unread)
        if method != route[0]:
            return self._error(
                405,
                "method_not_allowed",
                f"use {route[0]} for {path}",
                close=unread,
                extra=(("Allow", route[0]),),
            )
        name = route[1]
        if name == "healthz":
            return self._send(200, srv.service.health())
        if name == "models":
            return self._send(200, {"models": srv.service.models()})
        body = self._read_json()
        if body is None:
            return
        if name == "decide":
            self._decide(body)
        else:
            self._report(body)

    def _authorized(self, token: bytes) -> bool:
        value = self.headers.get("Authorization", "")
        scheme, _, credential = value.partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(credential.strip().encode("utf-8", "replace"), token)

    def _read_json(self) -> dict[str, Any] | None:
        """The parsed JSON object body, or ``None`` after an error response was sent."""
        if "transfer-encoding" in self.headers:
            self._error(
                411,
                "length_required",
                "send Content-Length; chunked bodies are not supported",
                close=True,
            )
            return None
        length_text = self.headers.get("Content-Length")
        if length_text is None:
            self._error(411, "length_required", "Content-Length is required", close=True)
            return None
        if not length_text.isascii() or not length_text.isdigit():
            self._error(400, "bad_request", "invalid Content-Length", close=True)
            return None
        length = int(length_text)
        if length > self.server.max_body:
            self._error(
                413,
                "payload_too_large",
                f"body exceeds {self.server.max_body} bytes",
                close=True,
            )
            return None
        ctype = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if ctype != _JSON:
            self._error(
                415, "unsupported_media_type", "Content-Type must be application/json", close=True
            )
            return None
        raw = self.rfile.read(length)
        if len(raw) < length:
            self.close_connection = True  # the client went away mid-body
            return None
        try:
            obj = json.loads(raw, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            self._error(400, "bad_json", "request body is not valid JSON")
            return None
        if not isinstance(obj, dict):
            self._error(400, "bad_request", "request body must be a JSON object")
            return None
        return obj

    def _decide(self, body: dict[str, Any]) -> None:
        try:
            state = StepState.from_dict(body)
        except (TypeError, ValueError) as exc:
            return self._error(400, "bad_request", _safe_message(exc))
        try:
            decision = self.server.service.decide(state)
        except NoActionAllowed as exc:
            if not exc.configured:
                return self._error(
                    503,
                    "no_fallback_configured",
                    "steproute has no decision and no fail-open action is configured; "
                    "use the model you would have used without steproute (local or refuse "
                    "if sensitive is true)",
                    sensitive=exc.sensitive,
                )
            return self._error(
                503,
                "no_action_allowed",
                "the policy failed and no configured fail-open action is allowed for this "
                "state; use your own model (local or refuse if sensitive is true)",
                sensitive=exc.sensitive,
            )
        self._send(200, decision.to_dict())

    def _report(self, body: dict[str, Any]) -> None:
        try:
            outcome = Outcome.from_dict(body)
        except (TypeError, ValueError) as exc:
            return self._error(400, "bad_request", _safe_message(exc))
        try:
            stored = self.server.service.report(outcome)
        except OSError as exc:
            log.error("could not write an outcome to the trace store: %s", type(exc).__name__)
            return self._error(503, "store_unavailable", "the outcome could not be stored")
        self._send(200, stored.to_dict())


def _host_name(header: str) -> str:
    header = header.strip().lower()
    if header.startswith("["):
        return header[1:].partition("]")[0]
    return header.partition(":")[0]


def make_server(
    service: Service,
    token: bytes | None,
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> ThreadingHTTPServer:
    """Bind a threaded server for ``service``; the caller runs ``serve_forever``.

    Refuses a non-loopback ``host`` without a token. ``server.server_address[1]`` is the bound
    port (useful with ``port=0``).
    """
    if not _is_loopback(host) and token is None:
        raise ServiceConfigError(
            f"refusing to listen on non-loopback address {host!r} without a bearer token"
        )
    server = _Server((host, port), service, token)
    server.max_body = max_body_bytes
    if token is None and _is_loopback(host):
        server.allowed_hosts = frozenset({"localhost", "127.0.0.1", "::1", host.lower()})
    return server


def serve(
    config: ServiceConfig,
    *,
    environ: Mapping[str, str] | None = None,
    out: TextIO | None = None,
) -> int:
    """Run the service until SIGINT or SIGTERM. Returns the process exit code."""
    import sys

    out = out or sys.stdout
    # Unknown keys are tolerated by design; do not spam stderr from the request path.
    # Process-wide, so it lives here and not in make_server.
    warnings.filterwarnings("ignore", category=UnknownFieldWarning)
    service, token = build_service(config, environ=environ)
    server = make_server(
        service,
        token,
        host=config.host,
        port=config.port,
        max_body_bytes=config.max_body_bytes,
    )
    host, port = str(server.server_address[0]), server.server_address[1]
    shown = f"[{host}]" if ":" in host else host
    print(f"steproute serve: listening on http://{shown}:{port}", file=out, flush=True)
    log.info("policy %r, store %s", service.policy_name, service.store.path)

    stopping = threading.Event()

    def stop(signum: int, frame: Any) -> None:
        if not stopping.is_set():
            stopping.set()
            # shutdown() blocks until serve_forever returns, so never call it on this thread.
            threading.Thread(target=server.shutdown, daemon=True).start()

    previous: dict[signal.Signals, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, stop)
        except ValueError:  # not the main thread
            pass
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    log.info("stopped")
    return 0
