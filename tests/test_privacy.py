"""Privacy tests: text capture, sensitive detection, replay-only export, scrub.

Every fake secret is assembled at runtime from pieces so that no source line matches a real
credential format.
"""

import json
import random
import string
from pathlib import Path

import pytest

from steproute.cli import build_parser, main
from steproute.core import (
    Action,
    Decision,
    Outcome,
    PrivacyConfig,
    StepState,
    TraceStore,
    detect_sensitive,
    export_replay,
    mark_sensitive,
)
from steproute.core.privacy import (
    DEFAULT_SECRET_PATTERNS,
    FEATURE_TEXT_HASH,
    FEATURE_TEXT_LEN,
    ScrubOptions,
    scrub_decision,
)

_RNG = random.Random(1234)
_UP = string.ascii_uppercase + string.digits
_ALNUM = string.ascii_letters + string.digits


def _chars(alphabet: str, n: int) -> str:
    return "".join(_RNG.choice(alphabet) for _ in range(n))


def fake_aws() -> str:
    return "".join(["AK", "IA", _chars(_UP, 16)])


def fake_github() -> str:
    return "".join(["gh", "p_", _chars(_ALNUM, 36)])


def fake_sk() -> str:
    return "".join(["s", "k-", _chars(_ALNUM, 32)])


def fake_slack() -> str:
    return "".join(["xo", "xb-", _chars(string.digits, 12), "-", _chars(_ALNUM, 12)])


def fake_bearer() -> str:
    return "Bearer " + _chars(_ALNUM, 40)


def fake_pem_header() -> str:
    return "-----" + "BEGIN RSA PRIV" + "ATE KEY-----"


def fake_assignment() -> str:
    return "pass" + "word=" + _chars(_ALNUM, 14)


FAKES = [fake_aws, fake_github, fake_sk, fake_slack, fake_bearer, fake_pem_header, fake_assignment]


def _state(text: str | None = None, source: str = "personal", **kw: object) -> StepState:
    return StepState(
        harness="h",
        session_id=kw.pop("session_id", "sess-1"),  # type: ignore[arg-type]
        step_idx=0,
        kind="edit",
        prompt_tokens=10,
        source=source,
        task_text=text,
        **kw,  # type: ignore[arg-type]
    )


def _dec(i: str, state: StepState, ts: float = 1_700_000_123.0) -> Decision:
    a = Action("m1")
    return Decision(i, ts, state, [a], a, "p", 1.0)


# --- capture_text ---------------------------------------------------------------------------


def test_text_not_stored_by_default(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    st = _state("refactor the widget module", features={"a": 1})
    d = _dec("d1", st)
    store.log_decision(d)
    raw = b"".join(f.read_bytes() for f in store.files())
    assert b"widget" not in raw
    (rec,) = list(store.records())
    assert isinstance(rec, Decision)
    assert rec.state.task_text is None
    assert rec.state.features == {"a": 1, FEATURE_TEXT_LEN: len("refactor the widget module")}
    # the caller's objects are untouched
    assert st.task_text == "refactor the widget module"
    assert st.features == {"a": 1}
    assert d.state is st


def test_text_hash_is_salted_and_deterministic(tmp_path: Path) -> None:
    cfg = PrivacyConfig(hash_text=True, salt="s1")
    store = TraceStore(tmp_path, privacy=cfg)
    store.log_decision(_dec("d1", _state("same")))
    store.log_decision(_dec("d2", _state("same")))
    recs = [r for r in store.records() if isinstance(r, Decision)]
    h = recs[0].state.features[FEATURE_TEXT_HASH]
    assert h == recs[1].state.features[FEATURE_TEXT_HASH]
    assert "same" not in h and len(h) == 32
    other = TraceStore(tmp_path / "o", privacy=PrivacyConfig(hash_text=True, salt="s2"))
    other.log_decision(_dec("d1", _state("same")))
    (r2,) = list(other.records())
    assert isinstance(r2, Decision)
    assert r2.state.features[FEATURE_TEXT_HASH] != h


def test_hash_text_requires_salt() -> None:
    with pytest.raises(ValueError):
        PrivacyConfig(hash_text=True)


def test_capture_text_opt_in_stores_text(tmp_path: Path) -> None:
    store = TraceStore(tmp_path, privacy=PrivacyConfig(capture_text=True))
    store.log_decision(_dec("d1", _state("keep me")))
    (rec,) = list(store.records())
    assert isinstance(rec, Decision)
    assert rec.state.task_text == "keep me"
    assert FEATURE_TEXT_LEN not in rec.state.features


def test_secret_in_text_sets_sensitive_even_when_text_dropped(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    secret = fake_aws()
    store.log_decision(_dec("d1", _state(f"use {secret} here")))
    assert secret.encode() not in b"".join(f.read_bytes() for f in store.files())
    (rec,) = list(store.records())
    assert isinstance(rec, Decision)
    assert rec.state.sensitive


# --- detection ------------------------------------------------------------------------------


@pytest.mark.parametrize("make", FAKES)
def test_default_patterns_detect(make: object) -> None:
    secret = make()  # type: ignore[operator]
    det = detect_sensitive(f"prefix {secret} suffix", [])
    assert det.sensitive and det.rules
    assert all(secret not in r for r in det.rules)


def test_clean_text_not_sensitive() -> None:
    assert not detect_sensitive("add a unit test for parser", ["src/app.py"]).sensitive


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "proj/.env.local",
        "/home/u/.ssh/config",
        "~/.aws/credentials".replace("~", "home"),
        "cert/server.pem",
        "a/secrets/db.json",
        "C:\\Users\\u\\.env",
    ],
)
def test_default_globs_detect(path: str) -> None:
    det = detect_sensitive(None, [path])
    assert det.sensitive and det.rules


def test_glob_names_reported_not_paths() -> None:
    det = detect_sensitive(None, ["x/.env"])
    assert "dotenv_files" in det.rules and all("env" in r for r in det.rules)


def test_callback_adds_and_failure_fails_closed() -> None:
    cfg = PrivacyConfig(callback=lambda t, p: bool(t and "internal" in t))
    assert detect_sensitive("internal memo", [], cfg).rules == ("user_callback",)
    assert not detect_sensitive("hello", [], cfg).sensitive

    def boom(t: str | None, p: object) -> bool:
        raise RuntimeError("nope")

    det = detect_sensitive("hello", [], PrivacyConfig(callback=boom))  # type: ignore[arg-type]
    assert det.sensitive and det.rules == ("callback_error",)


def test_mark_sensitive_copies_and_never_clears() -> None:
    st = _state("hello")
    out = mark_sensitive(st, ["a/.env"])
    assert out.sensitive and not st.sensitive and out is not st
    already = _state("hello", sensitive=True)
    assert mark_sensitive(already, ["main.py"]).sensitive
    cfg = PrivacyConfig(callback=lambda t, p: False)
    assert mark_sensitive(already, [], cfg).sensitive


# --- export: replay only --------------------------------------------------------------------


def _mixed_store(path: Path) -> TraceStore:
    store = TraceStore(path)
    for i in range(3):
        store.log_decision(_dec(f"r{i}", _state("t", source="replay", session_id=f"rs{i}")))
        store.log_outcome(Outcome(f"r{i}", 2.0, True))
        store.log_decision(_dec(f"p{i}", _state("t", source="personal", session_id=f"ps{i}")))
        store.log_outcome(Outcome(f"p{i}", 2.0, True))
    store.log_outcome(Outcome("orphan", 2.0, True))
    return store


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("scrub", [False, True])
def test_export_has_zero_personal_records(tmp_path: Path, scrub: bool, capsys: object) -> None:
    _mixed_store(tmp_path / "s")
    out = tmp_path / "out.jsonl"
    argv = ["traces", "export", str(tmp_path / "s"), "--out", str(out)]
    if scrub:
        argv += ["--scrub", "--salt", "x"]
    assert main(argv) == 0
    recs = _read(out)
    assert len(recs) == 6
    ids = {r["data"]["decision_id"] for r in recs}  # type: ignore[index]
    assert len(ids) == 3
    if not scrub:
        assert ids == {"r0", "r1", "r2"}
    for r in recs:
        if r["type"] == "decision":
            assert r["data"]["state"]["source"] == "replay"  # type: ignore[index]
    summary = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "exported 6" in summary and "withheld 7" in summary


def test_export_withholds_ids_seen_with_personal_source(tmp_path: Path) -> None:
    store = TraceStore(tmp_path / "s")
    store.log_decision(_dec("x", _state(source="replay")))
    store.log_decision(_dec("x", _state(source="personal")))
    store.log_outcome(Outcome("x", 2.0, True))
    summary = export_replay(store, tmp_path / "o.jsonl")
    assert summary.exported == 0


def test_cli_has_no_option_to_export_personal() -> None:
    parser = build_parser()
    sub = next(a for a in parser._actions if a.dest == "command")
    traces = sub.choices["traces"]  # type: ignore[attr-defined]
    tsub = next(a for a in traces._actions if a.dest == "traces_command")
    export = tsub.choices["export"]
    flags = {o for a in export._actions for o in a.option_strings}
    assert flags == {"-h", "--help", "--out", "--scrub", "--salt"}
    with pytest.raises(SystemExit):
        main(["traces", "export", "x", "--out", "y", "--include-personal"])
    with pytest.raises(SystemExit):
        main(["traces", "export", "x", "--out", "y", "--source", "personal"])


def test_export_function_signature_has_no_source_parameter() -> None:
    import inspect

    assert set(inspect.signature(export_replay).parameters) == {"store", "out", "scrub"}


# --- scrub ----------------------------------------------------------------------------------


def test_seeded_secret_corpus_removed_by_scrub(tmp_path: Path) -> None:
    secrets_: list[str] = []

    def plant() -> str:
        value = FAKES[len(secrets_) % len(FAKES)]()
        secrets_.append(value)
        return value

    session = "session-" + _chars(_ALNUM, 12)
    task = "task-" + _chars(_ALNUM, 12)
    plain_strings = [session, task]  # must vanish via hashing, not pattern matching

    # Free-text carriers: plain strings with no pattern are also dropped from dicts.
    carrier = "free text " + _chars(_ALNUM, 10)
    plain_strings.append(carrier)

    st = _state(
        f"fix this {plant()} and {plant()}",
        source="replay",
        session_id=session,
        task_id=task,
        tools=[f"tool_{plant()}", "bash"],
        features={
            "ok": 3,
            "flag": True,
            "none": None,
            "s": f"leak {plant()}",
            "carrier": carrier,
            "nested": {"deep": f"x {plant()}", "n": 1},
            "lst": [f"y {plant()}"],
        },
    )
    st.harness = f"harness-{plant()}"
    d = _dec("r1", st)
    d.policy = f"pol-{plant()}"
    store = TraceStore(tmp_path / "s", privacy=PrivacyConfig(capture_text=True))
    store.log_decision(d)
    store.log_outcome(
        Outcome(
            "r1",
            1_700_000_500.0,
            True,
            signals={"msg": f"err {plant()}", "k": {"z": plant()}, "exit": 0},
            final_model=f"m-{plant()}",
        )
    )
    # the corpus really is in the store before export
    assert any(s.encode() in b"".join(f.read_bytes() for f in store.files()) for s in secrets_)

    out = tmp_path / "out.jsonl"
    export_replay(store, out, scrub=ScrubOptions(salt="salt-" + _chars(_ALNUM, 8)))
    raw = out.read_bytes()
    for s in [*secrets_, *plain_strings]:
        assert s.encode() not in raw, s
    recs = _read(out)
    assert [r["type"] for r in recs] == ["decision", "outcome"]
    rt = store._parse(raw.splitlines()[0])
    assert isinstance(rt, Decision)
    assert rt.state.task_text is None
    assert rt.state.features == {"ok": 3, "flag": True, "none": None}
    assert rt.decision_id != "r1"


def test_scrub_ids_consistent_and_salted() -> None:
    a = scrub_decision(_dec("1", _state(session_id="S", task_id="T")), ScrubOptions("k1"))
    b = scrub_decision(_dec("2", _state(session_id="S", task_id="T")), ScrubOptions("k1"))
    c = scrub_decision(_dec("3", _state(session_id="S", task_id="T")), ScrubOptions("k2"))
    assert a.state.session_id == b.state.session_id != c.state.session_id
    assert a.state.task_id == b.state.task_id != a.state.session_id
    assert a.state.task_id is not None and len(a.state.task_id) == 16
    assert a.decision_id != "1" and len(a.decision_id) == 32


def test_scrub_buckets_timestamps() -> None:
    d = scrub_decision(_dec("1", _state(), ts=7300.5), ScrubOptions("k"))
    assert d.ts == 7200.0
    d2 = scrub_decision(_dec("1", _state(), ts=7300.5), ScrubOptions("k", bucket_seconds=60))
    assert d2.ts == 7260.0
    with pytest.raises(ValueError):
        ScrubOptions("k", bucket_seconds=0)
    with pytest.raises(ValueError):
        ScrubOptions("")


def test_scrub_does_not_mutate_input() -> None:
    d = _dec("1", _state("txt", features={"s": "v"}))
    scrub_decision(d, ScrubOptions("k"))
    assert d.state.task_text == "txt" and d.state.features == {"s": "v"}


def test_salt_never_in_output_and_env_salt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: object
) -> None:
    _mixed_store(tmp_path / "s")
    salt = "salt-" + _chars(_ALNUM, 16)
    monkeypatch.setenv("STEPROUTE_SCRUB_SALT", salt)
    o1, o2 = tmp_path / "1.jsonl", tmp_path / "2.jsonl"
    assert main(["traces", "export", str(tmp_path / "s"), "--out", str(o1), "--scrub"]) == 0
    assert main(["traces", "export", str(tmp_path / "s"), "--out", str(o2), "--scrub"]) == 0
    assert o1.read_bytes() == o2.read_bytes()
    assert salt.encode() not in o1.read_bytes()
    assert salt not in capsys.readouterr().out  # type: ignore[attr-defined]
    monkeypatch.delenv("STEPROUTE_SCRUB_SALT")
    o3, o4 = tmp_path / "3.jsonl", tmp_path / "4.jsonl"
    main(["traces", "export", str(tmp_path / "s"), "--out", str(o3), "--scrub"])
    main(["traces", "export", str(tmp_path / "s"), "--out", str(o4), "--scrub"])
    assert o3.read_bytes() != o4.read_bytes()  # random salt per export


def test_default_patterns_have_unique_names() -> None:
    names = [r.name for r in DEFAULT_SECRET_PATTERNS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", ["DB_PASS" + "WORD", "AWS_SECRET_ACCESS_KEY", "my_api_key"])
def test_assignment_pattern_matches_env_style_names(name: str) -> None:
    assert detect_sensitive(f"{name}={_chars(_ALNUM, 12)}", []).sensitive


def test_scrubbed_export_still_joins_and_hides_session_in_decision_id(tmp_path: Path) -> None:
    session = "sess-" + _chars(_ALNUM, 14)
    store = TraceStore(tmp_path / "s")
    for step in range(3):
        did = f"{session}:{step}"
        store.log_decision(_dec(did, _state(source="replay", session_id=session)))
        store.log_outcome(Outcome(did, 2.0, True))
    out = tmp_path / "o.jsonl"
    export_replay(store, out, scrub=ScrubOptions("k"))
    assert session.encode() not in out.read_bytes()
    exported = TraceStore(tmp_path / "e")
    (exported.path).mkdir()
    (exported.path / "traces-2024-01-01.jsonl").write_bytes(out.read_bytes())
    pairs = list(exported.joined())
    assert len(pairs) == 3
    assert all(o is not None for _, o in pairs)
    assert len({d.decision_id for d, _ in pairs}) == 3


@pytest.mark.parametrize(
    "path",
    [
        "src/id_resolver.py",
        "src/id_registry.py",
        "src/environment.py",
        "docs/keys.md",
        "tests/test_secrets.py",
        "src/secret_santa.py",
        "app/envelope.py",
        "src/monkey.py",
        "docs/azure_notes.md",
        "src/sshd_config_parser.py",
        "README.md",
    ],
)
def test_ordinary_paths_not_flagged(path: str) -> None:
    assert not detect_sensitive(None, [path]).sensitive


@pytest.mark.parametrize(
    "path", ["home/u/.ssh_x/id_rsa", "id_ed25519", "k/id_ecdsa.pub", "k/id_dsa", "a/id_rsa_work"]
)
def test_real_key_file_names_flagged(path: str) -> None:
    assert detect_sensitive(None, [path]).sensitive


def test_salt_not_in_repr() -> None:
    salt = "salt-" + _chars(_ALNUM, 12)
    assert salt not in repr(PrivacyConfig(hash_text=True, salt=salt))
    assert salt not in repr(ScrubOptions(salt=salt))
