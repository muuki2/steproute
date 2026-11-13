"""HTTP front end of the proxy: ``POST /v1/chat/completions``, ``GET /v1/models``,
``POST /report``, ``GET /healthz``. Standard library ``http.server``, one thread per
connection, HTTP/1.1 with keep-alive.

Streaming responses use chunked transfer encoding; each upstream chunk is written and flushed
as soon as it arrives, so nothing is buffered beyond one chunk.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import signal
import socket
import sys
import threading
import warnings
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO
from urllib.parse import urlsplit

from steproute.collectors.proxy.config import (
    ProxyConfig,
    ProxyConfigError,
    is_loopback_host,
    validate_against_catalog,
)
from steproute.collectors.proxy.engine import Proxy, Response
from steproute.collectors.proxy.source import DecisionSource, HttpSource, InProcessSource
from steproute.collectors.proxy.upstream import UpstreamPool
from steproute.core.catalog import Catalog, load_catalog, load_default_catalog
from steproute.core.privacy import PrivacyConfig
from steproute.core.schema import UnknownFieldWarning
from steproute.core.service import ServiceConfigError, build_service, load_service_config

__all__ = ["build_proxy", "make_proxy_server", "run_proxy"]

log = logging.getLogger("steproute.proxy")
CONNECTION_TIMEOUT_S = 600.0  # long generations: an idle socket is cut only after this


def build_proxy(
    config: ProxyConfig,
    *,
    environ: Mapping[str, str] | None = None,
    source: DecisionSource | None = None,
    catalog: Catalog | None = None,
    privacy: PrivacyConfig | None = None,
) -> tuple[Proxy, bytes | None]:
    """Validate ``config`` and build the ``Proxy`` and its bearer token (or ``None``).

    ``source``, ``catalog`` and ``privacy`` override what the config selects (tests).
    Raises ``ProxyConfigError``.
    """
    env = os.environ if environ is None else environ

    def secret(key: str, name: str | None) -> str | None:
        if name is None:
            return None
        value = env.get(name, "")
        if not value:
            raise ProxyConfigError(f"{key} names {name}, which is not set or is empty")
        return value

    token = secret("server.token_env", config.token_env)
    if not is_loopback_host(config.host) and token is None:
        raise ProxyConfigError(
            f"refusing to listen on non-loopback address {config.host!r} without a bearer "
            "token; set [server] token_env"
        )
    for up in config.upstreams:
        secret(f"upstream {up.name!r} api_key_env", up.api_key_env)
    salt = secret("session.salt_env", config.session_salt_env)

    if source is None:
        if config.source == "inprocess":
            try:
                service, _ = build_service(load_service_config(config.service_config), environ=env)
            except ServiceConfigError as exc:
                raise ProxyConfigError(f"decisions.service_config: {exc}") from None
            source = InProcessSource(service)
            catalog = catalog or service.catalog
            privacy = privacy or service.privacy
        else:
            try:
                source = HttpSource(
                    config.service_url,
                    token=secret("decisions.token_env", config.service_token_env),
                    decide_timeout_s=config.decide_timeout_ms / 1000,
                    report_timeout_s=config.report_timeout_ms / 1000,
                )
            except ValueError as exc:
                raise ProxyConfigError(str(exc)) from None
    if catalog is None:
        if isinstance(source, InProcessSource):
            catalog = source.service.catalog
        else:
            try:
                catalog = (
                    load_catalog(config.catalog_path)
                    if config.catalog_path
                    else load_default_catalog()
                )
            except (ValueError, OSError) as exc:
                raise ProxyConfigError(f"catalog: {exc}") from None
    if privacy is None:
        privacy = source.service.privacy if isinstance(source, InProcessSource) else PrivacyConfig()
    for warning in validate_against_catalog(config, catalog):
        log.warning("%s", warning)
    if config.local_fallback is None:
        log.warning(
            "no [privacy] local_fallback: if decisions fail, a sensitive request whose model "
            "is not local is refused (HTTP 503) instead of being routed"
        )
    proxy = Proxy(
        config,
        source=source,
        catalog=catalog,
        privacy=privacy,
        pool=UpstreamPool(env),
        session_salt=salt,
    )
    return proxy, token.encode() if token else None


_ROUTES = {
    "/v1/chat/completions": "POST",
    "/chat/completions": "POST",
    "/v1/models": "GET",
    "/models": "GET",
    "/report": "POST",
    "/healthz": "GET",
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self, address: tuple[str, int], proxy: Proxy, token: bytes | None, max_body: int
    ) -> None:
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        self.proxy = proxy
        self.token = token
        self.max_body = max_body
        self.allowed_hosts: frozenset[str] | None = None
        super().__init__(address, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError | TimeoutError):
            return
        log.error("connection handler failed: %s", type(exc).__name__)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = CONNECTION_TIMEOUT_S
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        return None  # never log request lines or bodies

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _send_json(
        self,
        status: int,
        body: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
        close: bool = False,
    ) -> None:
        payload = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(payload)

    def _err(self, status: int, code: str, message: str, close: bool = False) -> None:
        self._send_json(
            status, {"error": {"message": message, "type": code, "code": status}}, close=close
        )

    def _dispatch(self, method: str) -> None:
        try:
            self._handle(method)
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            log.error("request failed: %s", type(exc).__name__)
            try:
                self._err(500, "internal_error", "internal error", close=True)
            except OSError:
                self.close_connection = True

    def _handle(self, method: str) -> None:
        srv = self.server
        path = urlsplit(self.path).path
        unread = method == "POST"
        if srv.token is None and srv.allowed_hosts is not None:
            host = (self.headers.get("Host") or "").strip().lower()
            name = host[1:].partition("]")[0] if host.startswith("[") else host.partition(":")[0]
            if host and name not in srv.allowed_hosts:
                return self._err(403, "forbidden_host", "unexpected Host header", close=unread)
        if srv.token is not None and path != "/healthz":
            scheme, _, cred = self.headers.get("Authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                cred.strip().encode(), srv.token
            ):
                return self._err(
                    401, "unauthorized", "missing or invalid bearer token", close=unread
                )
        want = _ROUTES.get(path)
        if want is None:
            return self._err(404, "not_found", "unknown path", close=unread)
        if method != want:
            return self._err(405, "method_not_allowed", f"use {want} for {path}", close=unread)
        if path == "/healthz":
            return self._send_json(200, srv.proxy.health())
        if path.endswith("/models"):
            return self._send_json(200, {"object": "list", "data": srv.proxy.models()})
        body = self._read_json()
        if body is None:
            return
        if path == "/report":
            status, out = srv.proxy.report(body)
            return self._send_json(status, out)
        self._respond(srv.proxy.handle(body, dict(self.headers.items())))

    def _read_json(self) -> dict[str, Any] | None:
        length_text = self.headers.get("Content-Length")
        if "transfer-encoding" in self.headers or length_text is None:
            self._err(411, "length_required", "send Content-Length", close=True)
            return None
        if not length_text.isascii() or not length_text.isdigit():
            self._err(400, "invalid_request_error", "invalid Content-Length", close=True)
            return None
        length = int(length_text)
        if length > self.server.max_body:
            self._err(
                413, "payload_too_large", f"body exceeds {self.server.max_body} bytes", close=True
            )
            return None
        ctype = self.headers.get("Content-Type", "").partition(";")[0].strip().lower()
        if ctype != "application/json":
            self._err(
                415, "unsupported_media_type", "Content-Type must be application/json", close=True
            )
            return None
        raw = self.rfile.read(length)
        if len(raw) < length:
            self.close_connection = True
            return None
        try:
            obj = json.loads(raw)
        except (ValueError, RecursionError):
            self._err(400, "invalid_request_error", "request body is not valid JSON")
            return None
        if not isinstance(obj, dict):
            self._err(400, "invalid_request_error", "request body must be a JSON object")
            return None
        return obj

    def _respond(self, r: Response) -> None:
        try:
            if r.stream is None:
                self._send_json(r.status, r.body or {}, r.headers)
                return
            self.send_response(r.status)
            for k, v in r.headers.items():
                self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            try:
                for piece in r.stream:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                close = getattr(r.stream, "close", None)
                if close is not None:
                    close()  # marks the step as client_disconnected
                self.close_connection = True
        finally:
            r.finish()


def make_proxy_server(
    proxy: Proxy,
    token: bytes | None,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    max_body_bytes: int = 32 * 1024 * 1024,
) -> ThreadingHTTPServer:
    """Bind the proxy; the caller runs ``serve_forever``. ``server_address[1]`` is the port."""
    if not is_loopback_host(host) and token is None:
        raise ProxyConfigError(
            f"refusing to listen on non-loopback address {host!r} without a token"
        )
    server = _Server((host, port), proxy, token, max_body_bytes)
    if token is None:
        server.allowed_hosts = frozenset({"localhost", "127.0.0.1", "::1", host.lower()})
    return server


def run_proxy(
    config: ProxyConfig, *, environ: Mapping[str, str] | None = None, out: TextIO | None = None
) -> int:
    """Run the proxy until SIGINT or SIGTERM. Returns the exit code."""
    out = out or sys.stdout
    warnings.filterwarnings("ignore", category=UnknownFieldWarning)
    proxy, token = build_proxy(config, environ=environ)
    from steproute.collectors.proxy.upstream import load_litellm

    load_litellm()  # fail at start, not on the first request, if the extra is missing
    server = make_proxy_server(
        proxy, token, host=config.host, port=config.port, max_body_bytes=config.max_body_bytes
    )
    host, port = str(server.server_address[0]), server.server_address[1]
    shown = f"[{host}]" if ":" in host else host
    print(
        f"steproute proxy: listening on http://{shown}:{port}/v1 (mode {config.mode})",
        file=out,
        flush=True,
    )
    stopping = threading.Event()

    def stop(signum: int, frame: Any) -> None:
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    previous: dict[signal.Signals, Any] = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, stop)
        except ValueError:
            pass
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
