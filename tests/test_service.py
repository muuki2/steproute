"""The local HTTP service: contract tests, fail-open, privacy, concurrency.

The server under test is started on a free loopback port inside each test; nothing leaves the
machine. Fake secrets are assembled at runtime so no credential-shaped literal is in the file.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from steproute.core import (
    Action,
    Decision,
    FixedPolicy,
    NoFeasibleAction,
    Outcome,
    RulePolicy,
    StepState,
    TraceStore,
)
from steproute.core.catalog import Catalog, load_default_catalog
from steproute.core.schema_gen import load_json_schema
from steproute.core.service import (
    Service,
    ServiceConfig,
    ServiceConfigError,
    build_service,
    load_service_config,
    make_server,
)

CATALOG = load_default_catalog(warn_unverified=False)
CLOUD_ONLY = Catalog(tuple(m for m in CATALOG.models if not m.local))
SCHEMA = load_json_schema()
TOKEN = "t" + "k" * 20


def validator(name: str) -> Draft202012Validator:
    return Draft202012Validator({"$ref": f"#/$defs/{name}", "$defs": SCHEMA["$defs"]})


def fake_secret() -> str:
    return "AK" + "IA" + "A" * 16  # matches the aws_access_key_id rule; built at runtime


BODY: dict[str, Any] = {
    "harness": "test",
    "session_id": "s1",
    "step_idx": 0,
    "kind": "main",
    "prompt_tokens": 1000,
    "tools": ["Edit"],
}


class Client:
    def __init__(self, port: int, token: str | None = None) -> None:
        self.conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        *,
        raw: bytes | None = None,
        headers: dict[str, str] | None = None,
        token: str | None = "default",
    ) -> tuple[int, Any, http.client.HTTPResponse]:
        h = {"Content-Type": "application/json", **(headers or {})}
        tok = self.token if token == "default" else token
        if tok is not None:
            h["Authorization"] = f"Bearer {tok}"
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        self.conn.request(method, path, body=data, headers=h)
        resp = self.conn.getresponse()
        payload = resp.read()
        return resp.status, json.loads(payload) if payload else None, resp

    def close(self) -> None:
        self.conn.close()


class Running:
    def __init__(self, service: Service, token: str | None, tmp: Path, **kw: Any) -> None:
        self.service = service
        self.server = make_server(
            service, token.encode() if token else None, host="127.0.0.1", port=0, **kw
        )
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()
        self.token = token
        self.store = service.store

    def client(self, token: str | None = "default") -> Client:
        return Client(self.port, self.token if token == "default" else token)

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


def config(tmp: Path, **kw: Any) -> ServiceConfig:
    base: dict[str, Any] = {"store_path": tmp / "traces", "port": 0}
    base.update(kw)
    return ServiceConfig(**base)


@pytest.fixture
def start(tmp_path: Path) -> Iterator[Callable[..., Running]]:
    running: list[Running] = []

    def _start(
        cfg: ServiceConfig | None = None,
        *,
        policy: Any = None,
        catalog: Catalog | None = None,
        token: str | None = None,
        **kw: Any,
    ) -> Running:
        cfg = cfg or config(tmp_path)
        env = {"STEPROUTE_TEST_TOKEN": token} if token else {}
        if token and not cfg.token_env:
            cfg = ServiceConfig(**{**cfg.__dict__, "token_env": "STEPROUTE_TEST_TOKEN"})
        svc, tok = build_service(cfg, environ=env, policy=policy, catalog=catalog or CATALOG)
        r = Running(svc, token, tmp_path, **kw)
        running.append(r)
        return r

    yield _start
    for r in running:
        r.stop()


# --- contract tests -------------------------------------------------------------------------


def test_decide_contract(start: Callable[..., Running]) -> None:
    r = start()
    c = r.client()
    status, body, resp = c.request("POST", "/decide", BODY)
    assert status == 200
    assert resp.getheader("Content-Type") == "application/json"
    validator("Decision").validate(body)
    assert body["policy"] == "rules"
    assert body["state"]["session_id"] == "s1"
    # Keep-alive: a second request on the same connection works.
    status, body2, _ = c.request("POST", "/decide", {**BODY, "step_idx": 1})
    assert status == 200 and body2["decision_id"] != body["decision_id"]


def test_decide_tolerates_unknown_keys(start: Callable[..., Running]) -> None:
    status, body, _ = start().client().request("POST", "/decide", {**BODY, "future_field": 1})
    assert status == 200
    validator("Decision").validate(body)


def test_report_contract_and_join(start: Callable[..., Running]) -> None:
    r = start()
    c = r.client()
    _, decision, _ = c.request("POST", "/decide", BODY)
    out = {
        "decision_id": decision["decision_id"],
        "ts": time.time(),
        "success": None,
        "signals": {"tests_passed": True},
        "final_model": decision["action"]["model"],
        "input_tokens": 1000,
        "output_tokens": 500,
    }
    status, stored, _ = c.request("POST", "/report", out)
    assert status == 200
    validator("Outcome").validate(stored)
    assert stored["success"] is True  # labelled from the signals
    assert stored["cost_usd"] is not None and stored["cost_usd"] > 0  # filled from the catalog
    records = list(r.store.records())
    assert [type(x).__name__ for x in records] == ["Decision", "Outcome"]
    assert records[0].decision_id == records[1].decision_id  # type: ignore[union-attr]


def test_report_keeps_given_success_and_cost(start: Callable[..., Running]) -> None:
    c = start().client()
    out = {
        "decision_id": "d",
        "ts": 1.0,
        "success": False,
        "signals": {"tests_passed": True},
        "final_model": "claude-sonnet-5-5",
        "input_tokens": 10,
        "cost_usd": 0.123,
    }
    _, stored, _ = c.request("POST", "/report", out)
    assert stored["success"] is False and stored["cost_usd"] == 0.123


@pytest.mark.parametrize(
    "extra",
    [
        {"escalated": True},  # several attempts: not priced
        {"final_model": None},  # unknown model
        {"final_model": "no-such-model"},
    ],
)
def test_report_does_not_guess_cost(start: Callable[..., Running], extra: dict[str, Any]) -> None:
    out = {
        "decision_id": "d",
        "ts": 1.0,
        "success": None,
        "final_model": "claude-sonnet-5-5",
        "input_tokens": 10,
        **extra,
    }
    _, stored, _ = start().client().request("POST", "/report", out)
    assert stored["cost_usd"] is None


def test_report_orphan_is_stored(start: Callable[..., Running]) -> None:
    r = start()
    status, _, _ = r.client().request(
        "POST", "/report", {"decision_id": "never-decided", "ts": 1.0, "success": True}
    )
    assert status == 200
    assert len(list(r.store.records())) == 1


def test_models_contract(start: Callable[..., Running]) -> None:
    status, body, _ = start().client().request("GET", "/models")
    assert status == 200
    assert len(body["models"]) == len(CATALOG)
    for m in body["models"]:
        validator("ModelProfile").validate(m)
    holdout = {m["id"]: m["holdout"] for m in body["models"]}
    assert holdout["devstral:24b"] is True and holdout["claude-sonnet-5-5"] is False


def test_healthz_contract_needs_no_token(start: Callable[..., Running]) -> None:
    from steproute import __version__
    from steproute.core import SCHEMA_VERSION

    r = start(token=TOKEN)
    status, body, _ = r.client(token=None).request("GET", "/healthz", token=None)
    assert status == 200
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["policy"] == "rules"


# --- error paths ----------------------------------------------------------------------------


def test_bad_json(start: Callable[..., Running]) -> None:
    c = start().client()
    for raw in (b"{not json", b"[1,2]", b'{"a": NaN}', b"\xff\xfe"):
        status, body, _ = c.request("POST", "/decide", raw=raw)
        assert status == 400 and "error" in body
    # The connection is still usable after a body-level 400.
    assert c.request("POST", "/decide", BODY)[0] == 200


def test_wrong_types_and_no_echo(start: Callable[..., Running]) -> None:
    c = start().client()
    marker = "zz" + "marker" + "zz"
    status, body, _ = c.request("POST", "/decide", {**BODY, "prompt_tokens": marker})
    assert status == 400
    assert marker not in json.dumps(body)
    assert "prompt_tokens" in body["message"]
    status, body, _ = c.request("POST", "/decide", {**BODY, "source": marker})
    assert status == 400 and marker not in json.dumps(body)
    status, body, _ = c.request("POST", "/decide", {"harness": "x"})
    assert status == 400 and "session_id" in body["message"]
    status, body, _ = c.request("POST", "/report", {"decision_id": 5, "ts": 1, "success": None})
    assert status == 400


def test_body_too_large(start: Callable[..., Running]) -> None:
    r = start(max_body_bytes=2000)
    c = r.client()
    status, body, resp = c.request("POST", "/decide", {**BODY, "task_text": "x" * 5000})
    assert status == 413 and body["error"] == "payload_too_large"
    assert resp.getheader("Connection") == "close"
    # A fresh connection still works.
    assert r.client().request("POST", "/decide", BODY)[0] == 200


def test_content_type_and_length_required(start: Callable[..., Running]) -> None:
    c = start().client()
    status, _, _ = c.request("POST", "/decide", raw=b"{}", headers={"Content-Type": "text/plain"})
    assert status == 415
    c = start().client()
    c.conn.putrequest("POST", "/decide")
    c.conn.putheader("Content-Type", "application/json")
    c.conn.endheaders()
    resp = c.conn.getresponse()
    resp.read()
    assert resp.status == 411


def test_token_required(start: Callable[..., Running]) -> None:
    r = start(token=TOKEN)
    for path, method in (("/decide", "POST"), ("/report", "POST"), ("/models", "GET")):
        for tok in (None, "wrong", TOKEN + "x", ""):
            status, body, resp = r.client().request(method, path, BODY, token=tok)
            assert status == 401, (path, tok)
            assert body["error"] == "unauthorized"
            assert resp.getheader("WWW-Authenticate") == "Bearer"
    # Unknown paths do not reveal anything without the token either.
    assert r.client().request("GET", "/nope", token=None)[0] == 401
    assert r.client().request("POST", "/decide", BODY)[0] == 200
    assert r.client().request("GET", "/models")[0] == 200
    # Lower-case scheme is accepted (HTTP auth schemes are case-insensitive).
    status, _, _ = r.client().request(
        "GET", "/models", token=None, headers={"Authorization": f"bearer {TOKEN}"}
    )
    assert status == 200


def test_unknown_path_and_wrong_method(start: Callable[..., Running]) -> None:
    c = start().client()
    assert c.request("GET", "/nope")[0] == 404
    assert c.request("POST", "/nope", BODY)[0] == 404
    status, body, resp = c.request("GET", "/decide")
    assert status == 405 and resp.getheader("Allow") == "POST"
    status, _, resp = c.request("POST", "/models", BODY)
    assert status == 405 and resp.getheader("Allow") == "GET"
    assert c.request("DELETE", "/decide")[0] == 405
    assert c.request("POST", "/healthz", {})[0] == 405
    assert c.request("GET", "/healthz")[0] == 200


def test_foreign_host_header_refused_without_token(start: Callable[..., Running]) -> None:
    c = start().client()
    status, body, _ = c.request("GET", "/models", headers={"Host": "evil.example"})
    assert status == 403 and body["error"] == "forbidden_host"
    assert c.request("GET", "/models", headers={"Host": "localhost:1234"})[0] == 200


# --- fail-open ------------------------------------------------------------------------------


class SlowPolicy:
    name = "slow"

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.finished = threading.Event()
        self.inner = RulePolicy(CATALOG)

    def decide(self, state: StepState) -> Decision:
        time.sleep(self.delay)
        d = self.inner.decide(state)
        d.policy = self.name
        self.finished.set()
        return d

    def report(self, outcome: Outcome) -> None:
        return None


class RaisingPolicy:
    name = "raising"

    def decide(self, state: StepState) -> Decision:
        raise RuntimeError("secret text in an exception message " + str(state.task_text))

    def report(self, outcome: Outcome) -> None:
        raise RuntimeError("report failed")


CLOUD_DEFAULT = {"model": "claude-sonnet-5-5", "effort": "medium"}


def failopen_cfg(tmp: Path, **kw: Any) -> ServiceConfig:
    base: dict[str, Any] = {
        "budget_ms": 50.0,
        "default_action": ("claude-sonnet-5-5", "medium"),
        "local_fallback": ("qwen3-coder:30b", "medium"),
    }
    base.update(kw)
    return config(tmp, **base)


def test_slow_policy_triggers_default_action(start: Callable[..., Running], tmp_path: Path) -> None:
    pol = SlowPolicy(0.4)
    r = start(failopen_cfg(tmp_path), policy=pol)
    t0 = time.perf_counter()
    status, body, _ = r.client().request("POST", "/decide", BODY)
    elapsed = time.perf_counter() - t0
    assert status == 200
    assert elapsed < 0.3  # did not wait for the policy
    validator("Decision").validate(body)
    assert body["action"] == {"model": "claude-sonnet-5-5", "effort": "medium", "escalate_to": None}
    assert body["policy"] == "fail_open:timeout"
    assert r.service.abandoned_running == 1
    assert pol.finished.wait(5)
    deadline = time.time() + 5
    while r.service.abandoned_running and time.time() < deadline:
        time.sleep(0.01)
    assert r.service.abandoned_running == 0


def test_late_result_is_not_logged(start: Callable[..., Running], tmp_path: Path) -> None:
    pol = SlowPolicy(0.2)
    r = start(failopen_cfg(tmp_path, budget_ms=30.0), policy=pol)
    _, body, _ = r.client().request("POST", "/decide", BODY)
    assert body["policy"] == "fail_open:timeout"
    assert pol.finished.wait(5)
    time.sleep(0.1)
    decisions = [x for x in r.store.records() if isinstance(x, Decision)]
    assert [d.policy for d in decisions] == ["fail_open:timeout"]
    assert decisions[0].decision_id == body["decision_id"]


def test_abandoned_calls_are_bounded(start: Callable[..., Running], tmp_path: Path) -> None:
    pol = SlowPolicy(1.0)
    r = start(failopen_cfg(tmp_path, budget_ms=10.0, max_abandoned=2), policy=pol)
    reasons = [r.client().request("POST", "/decide", BODY)[1]["policy"] for _ in range(4)]
    assert reasons == ["fail_open:timeout"] * 2 + ["fail_open:stuck"] * 2
    assert r.service.abandoned_running == 2


def test_raising_policy_fails_open_and_logs_no_text(
    start: Callable[..., Running], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    r = start(failopen_cfg(tmp_path), policy=RaisingPolicy())
    marker = "promptmarker" + "123"
    with caplog.at_level("DEBUG", logger="steproute.service"):
        status, body, _ = r.client().request("POST", "/decide", {**BODY, "task_text": marker})
    assert status == 200 and body["policy"] == "fail_open:error"
    assert body["action"]["model"] == "claude-sonnet-5-5"
    assert body["state"]["task_text"] is None  # not echoed
    assert any("RuntimeError" in m for m in caplog.messages)
    assert marker not in caplog.text
    # report() failures of the policy do not fail the request.
    out = {"decision_id": body["decision_id"], "ts": 1.0, "success": True}
    assert r.client().request("POST", "/report", out)[0] == 200


def test_no_feasible_action_from_policy(start: Callable[..., Running], tmp_path: Path) -> None:
    class Nothing(RaisingPolicy):
        def decide(self, state: StepState) -> Decision:
            raise NoFeasibleAction(0, {})

    r = start(failopen_cfg(tmp_path), policy=Nothing())
    _, body, _ = r.client().request("POST", "/decide", BODY)
    assert body["policy"] == "fail_open:no_feasible"


def test_sensitive_state_never_gets_cloud_under_fail_open(
    start: Callable[..., Running], tmp_path: Path
) -> None:
    # The default action is a cloud model; the state is sensitive only through its text.
    r = start(failopen_cfg(tmp_path), policy=RaisingPolicy())
    c = r.client()
    status, body, _ = c.request("POST", "/decide", {**BODY, "task_text": "key " + fake_secret()})
    assert status == 200
    assert body["policy"] == "fail_open:error"
    assert body["state"]["sensitive"] is True
    assert body["action"]["model"] == "qwen3-coder:30b"
    # The same via the explicit flag, and via the time budget.
    r2 = start(failopen_cfg(tmp_path, budget_ms=10.0), policy=SlowPolicy(0.3))
    _, body, _ = r2.client().request("POST", "/decide", {**BODY, "sensitive": True})
    assert body["action"]["model"] == "qwen3-coder:30b"
    # And through a sensitive path.
    _, body, _ = c.request(
        "POST", "/decide", {**BODY, "features": {"paths": ["app/.env"]}, "tools": []}
    )
    assert body["state"]["sensitive"] is True and body["action"]["model"] == "qwen3-coder:30b"


def test_sensitive_with_no_allowed_fallback_is_503(
    start: Callable[..., Running], tmp_path: Path
) -> None:
    # A cloud-only catalog has no local model to offer, so the service may start.
    r = start(
        config(tmp_path, default_action=("claude-sonnet-5-5", "medium")),
        policy=RaisingPolicy(),
        catalog=CLOUD_ONLY,
    )
    c = r.client()
    status, body, _ = c.request("POST", "/decide", {**BODY, "sensitive": True})
    assert status == 503
    assert body["error"] == "no_action_allowed" and body["sensitive"] is True
    status, body, _ = c.request("POST", "/decide", BODY)  # not sensitive: the default is fine
    assert status == 200 and body["policy"] == "fail_open:error"
    assert list(r.store.records())  # the 503 left no decision behind, the 200 did
    assert len(list(r.store.records())) == 1


def test_default_infeasible_for_state_uses_fallback(
    start: Callable[..., Running], tmp_path: Path
) -> None:
    # 20k prompt tokens do not fit the default model's context but fit the fallback's.
    cat = Catalog(
        tuple(
            m.__class__(**{**m.__dict__, "context_window": 10_000})
            if m.id == "claude-sonnet-5-5"
            else m
            for m in CATALOG.models
        )
    )
    r = start(failopen_cfg(tmp_path), policy=RaisingPolicy(), catalog=cat)
    _, body, _ = r.client().request("POST", "/decide", {**BODY, "prompt_tokens": 20_000})
    assert body["action"]["model"] == "qwen3-coder:30b"


def test_policy_choosing_cloud_for_sensitive_is_overridden(
    start: Callable[..., Running], tmp_path: Path
) -> None:
    class Bad:
        name = "bad"

        def decide(self, state: StepState) -> Decision:
            a = Action("claude-sonnet-5-5", "medium")
            return Decision("d1", 1.0, state, [a], a, "bad", 1.0)

        def report(self, outcome: Outcome) -> None:
            return None

    r = start(failopen_cfg(tmp_path), policy=Bad())
    _, body, _ = r.client().request("POST", "/decide", {**BODY, "sensitive": True})
    assert body["policy"] == "fail_open:policy_violation"
    assert body["action"]["model"] == "qwen3-coder:30b"


def test_text_not_echoed_unless_captured(start: Callable[..., Running], tmp_path: Path) -> None:
    r = start()
    _, body, _ = r.client().request("POST", "/decide", {**BODY, "task_text": "hello there"})
    assert body["state"]["task_text"] is None
    assert body["state"]["features"]["task_text_len"] == 11
    stored = next(x for x in r.store.records() if isinstance(x, Decision))
    assert stored.state.task_text is None
    assert stored.to_dict() == body

    r = start(config(tmp_path, capture_text=True, store_path=tmp_path / "t2"))
    _, body, _ = r.client().request("POST", "/decide", {**BODY, "task_text": "hello there"})
    assert body["state"]["task_text"] == "hello there"


def test_fixed_policy_sensitive_gets_no_cloud(
    start: Callable[..., Running], tmp_path: Path
) -> None:
    cfg = failopen_cfg(tmp_path, policy_kind="fixed", policy_action=("claude-sonnet-5-5", "medium"))
    r = start(cfg)
    _, body, _ = r.client().request("POST", "/decide", BODY)
    assert body["policy"] == "fixed" and body["action"]["model"] == "claude-sonnet-5-5"
    _, body, _ = r.client().request("POST", "/decide", {**BODY, "sensitive": True})
    assert body["policy"] == "fail_open:no_feasible"
    assert body["action"]["model"] == "qwen3-coder:30b"


def test_store_failure_does_not_fail_decide(start: Callable[..., Running], tmp_path: Path) -> None:
    r = start()

    def boom(decision: Decision) -> None:
        raise OSError("disk full")

    r.store.log_decision = boom  # type: ignore[method-assign]
    status, _, _ = r.client().request("POST", "/decide", BODY)
    assert status == 200 and r.service.store_errors == 1


# --- concurrency ----------------------------------------------------------------------------


def test_concurrent_requests(start: Callable[..., Running]) -> None:
    r = start()
    errors: list[str] = []
    ids: list[str] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        c = r.client()
        try:
            for i in range(25):
                s, body, _ = c.request(
                    "POST", "/decide", {**BODY, "session_id": f"w{n}", "step_idx": i}
                )
                if s != 200:
                    errors.append(f"decide {s}")
                    continue
                s2, _, _ = c.request(
                    "POST",
                    "/report",
                    {"decision_id": body["decision_id"], "ts": 1.0, "success": True},
                )
                if s2 != 200:
                    errors.append(f"report {s2}")
                with lock:
                    ids.append(body["decision_id"])
        finally:
            c.close()

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(set(ids)) == 200
    recs = list(r.store.records())
    assert r.store.skipped_lines == 0
    assert sum(isinstance(x, Decision) for x in recs) == 200
    assert sum(isinstance(x, Outcome) for x in recs) == 200


# --- config and startup ---------------------------------------------------------------------


def test_config_file_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "svc.toml"
    f.write_text(
        """
[server]
host = "127.0.0.1"
port = 9001
max_body_bytes = 5000
[auth]
token_env = "MY_TOKEN_VAR"
[store]
path = "traces"
[policy]
kind = "fixed"
action = { model = "claude-sonnet-5-5", effort = "high" }
[failopen]
budget_ms = 25
max_abandoned = 3
default = { model = "claude-sonnet-5-5" }
local_fallback = { model = "qwen3-coder:30b", effort = "medium" }
[privacy]
capture_text = true
"""
    )
    cfg = load_service_config(f)
    assert cfg.port == 9001 and cfg.budget_ms == 25.0 and cfg.max_abandoned == 3
    assert cfg.store_path == (tmp_path / "traces").resolve()
    assert cfg.default_action == ("claude-sonnet-5-5", "medium")
    assert cfg.policy_action == ("claude-sonnet-5-5", "high")
    assert cfg.capture_text is True
    assert load_service_config(None) == ServiceConfig()


@pytest.mark.parametrize(
    "data",
    [
        {"nope": {}},
        {"server": {"prot": 1}},
        {"server": {"port": "80"}},
        {"server": {"port": 99999}},
        {"failopen": {"budget_ms": 0}},
        {"failopen": {"default": {"model": "m", "effort": "turbo"}}},
        {"failopen": {"default": {"model": "m", "extra": 1}}},
        {"policy": {"kind": "magic"}},
        {"auth": {"token_env": "not a valid name!"}},
        {"server": "x"},
    ],
)
def test_config_rejects(data: dict[str, Any]) -> None:
    with pytest.raises(ServiceConfigError):
        load_service_config(data)


def test_config_error_does_not_echo_pasted_secret() -> None:
    pasted = "pasted" + "-value-" + "123"
    with pytest.raises(ServiceConfigError) as ei:
        load_service_config({"auth": {"token_env": pasted}})
    assert pasted not in str(ei.value)


def test_startup_refusals(tmp_path: Path) -> None:
    # non-loopback without token
    with pytest.raises(ServiceConfigError, match="non-loopback"):
        build_service(config(tmp_path, host="0.0.0.0"), environ={})
    # token env named but unset
    with pytest.raises(ServiceConfigError, match="not set"):
        build_service(config(tmp_path, token_env="UNSET_VAR_X"), environ={})
    # unknown model / unsupported effort / holdout / non-local fallback
    for bad in (("no-such", "medium"), ("claude-sonnet-5-5", "none"), ("devstral:24b", "medium")):
        with pytest.raises(ServiceConfigError):
            build_service(
                config(tmp_path, default_action=bad, local_fallback=("qwen3-coder:30b", "medium")),
                environ={},
            )
    with pytest.raises(ServiceConfigError, match="not a local model"):
        build_service(
            config(
                tmp_path,
                default_action=("claude-sonnet-5-5", "medium"),
                local_fallback=("claude-haiku-5-5", "low"),
            ),
            environ={},
        )
    # cloud default, no fallback, catalog has local models
    with pytest.raises(ServiceConfigError, match="local_fallback"):
        build_service(config(tmp_path, default_action=("claude-sonnet-5-5", "medium")), environ={})
    # hash_text needs a salt
    with pytest.raises(ServiceConfigError):
        build_service(config(tmp_path, hash_text=True), environ={})
    # fixed policy needs an action
    with pytest.raises(ServiceConfigError, match="action"):
        build_service(config(tmp_path, policy_kind="fixed"), environ={})


def test_non_loopback_with_token_is_allowed_to_build(tmp_path: Path) -> None:
    svc, tok = build_service(config(tmp_path, host="0.0.0.0", token_env="T"), environ={"T": TOKEN})
    assert tok == TOKEN.encode()
    with pytest.raises(ServiceConfigError):
        make_server(svc, None, host="0.0.0.0", port=0)


def test_zero_config_defaults_fail_open_to_local(tmp_path: Path) -> None:
    svc, _ = build_service(config(tmp_path), environ={})
    assert svc.default_action.model == "qwen3-coder:30b"
    assert CATALOG.get(svc.default_action.model).local
    assert isinstance(svc.policy, RulePolicy)


def test_salt_comes_from_environment(tmp_path: Path) -> None:
    salt = "s" + "a" * 8
    svc, _ = build_service(
        config(tmp_path, hash_text=True, salt_env="SALT_VAR"), environ={"SALT_VAR": salt}
    )
    state = StepState("h", "s", 0, "main", 1, task_text="hello")
    d = svc.decide(state)
    assert "task_text_hash" in d.state.features and d.state.task_text is None


def test_service_is_usable_without_http(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "x")
    svc = Service(
        policy=FixedPolicy(Action("claude-sonnet-5-5", "medium"), CATALOG),
        catalog=CATALOG,
        store=store,
        privacy=store.privacy,
        default_action=Action("qwen3-coder:30b", "medium"),
        local_fallback=None,
    )
    assert svc.decide(StepState("h", "s", 0, "main", 1)).policy == "fixed"
    assert svc.health()["policy"] == "fixed"


# --- the command ----------------------------------------------------------------------------


@pytest.mark.parametrize("sig_name", ["SIGTERM", "SIGINT"])
def test_serve_command_starts_and_stops_on_signal(tmp_path: Path, sig_name: str) -> None:
    import os
    import signal
    import subprocess
    import sys

    code = "import sys; from steproute.cli import main; sys.exit(main(sys.argv[1:]))"
    proc = subprocess.Popen(
        [sys.executable, "-c", code, "serve", "--port", "0", "--store", str(tmp_path / "t")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        assert line.startswith("steproute serve: listening on http://127.0.0.1:")
        port = int(line.rsplit(":", 1)[1])
        status, body, _ = Client(port).request("GET", "/healthz")
        assert status == 200 and body["status"] == "ok"
        status, body, _ = Client(port).request("POST", "/decide", BODY)
        assert status == 200
        proc.send_signal(getattr(signal, sig_name))
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                stream.close()
    assert list((tmp_path / "t").glob("traces-*.jsonl"))


def test_serve_command_refuses_open_bind_without_token(capsys: pytest.CaptureFixture[str]) -> None:
    from steproute.cli import main

    assert main(["serve", "--host", "0.0.0.0", "--port", "0"]) == 2
    assert "non-loopback" in capsys.readouterr().err
