"""`steproute catalog show|check` with a fake transport (no sockets)."""

import json
from pathlib import Path
from typing import Any

import pytest

from steproute.cli import main
from steproute.core.toolcheck import TransportError, check_model


def tool_reply(args: Any = '{"city": "Paris"}', name: str = "get_weather") -> dict[str, Any]:
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }
    return {"choices": [{"message": msg}]}


def text_reply() -> dict[str, Any]:
    call = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    return {"choices": [{"message": {"role": "assistant", "content": call}}]}


class Fake:
    def __init__(self, replies: dict[str, Any]):
        self.replies = replies
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((url, body))
        r = self.replies[body["model"]]
        if isinstance(r, Exception):
            raise r
        return r  # type: ignore[no-any-return]


def test_structured_call_passes_and_request_shape() -> None:
    fake = Fake({"m": tool_reply()})
    r = check_model("m", transport=fake)
    assert r.ok
    url, body = fake.calls[0]
    assert url == "http://localhost:11434/v1/chat/completions"
    assert len(body["tools"]) == 1 and body["tools"][0]["type"] == "function"


def test_plain_text_call_fails() -> None:
    r = check_model("m", transport=Fake({"m": text_reply()}))
    assert not r.ok and "plain text" in r.detail


@pytest.mark.parametrize(
    "reply",
    [tool_reply("not json"), tool_reply('"str"'), tool_reply(name=""), {"choices": []}, {}],
)
def test_malformed_calls_fail(reply: dict[str, Any]) -> None:
    assert not check_model("m", transport=Fake({"m": reply})).ok


def test_http_error_and_unreachable() -> None:
    r = check_model("m", transport=Fake({"m": TransportError("HTTP 404 Not Found")}))
    assert not r.ok and not r.unreachable and "404" in r.detail
    r = check_model("m", transport=Fake({"m": TransportError("refused", unreachable=True)}))
    assert not r.ok and r.unreachable


CAT = """
[[model]]
id = "good"
provider = "ollama"
price_in = 0
price_out = 0
context_window = 100
local = true
verified = true

[[model]]
id = "bad"
provider = "ollama"
price_in = 0
price_out = 0
context_window = 100
local = true
verified = true

[[model]]
id = "cloud"
provider = "x"
price_in = 1
price_out = 1
context_window = 100
verified = true
"""


@pytest.fixture
def cat(tmp_path: Path) -> str:
    p = tmp_path / "c.toml"
    p.write_text(CAT, encoding="utf-8")
    return str(p)


def test_check_reports_per_model_and_exit_code(
    cat: str, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = Fake({"good": tool_reply(), "bad": text_reply()})
    code = main(["catalog", "check", "--catalog", cat], transport=fake)
    out = capsys.readouterr().out
    assert code == 1
    assert "PASS  good" in out and "FAIL  bad" in out
    assert 'set tools = false for "bad"' in out
    assert [c[1]["model"] for c in fake.calls] == ["good", "bad"]  # cloud model not checked


def test_check_all_pass_exit_zero(cat: str) -> None:
    fake = Fake({"good": tool_reply(), "bad": tool_reply()})
    assert main(["catalog", "check", "--catalog", cat], transport=fake) == 0


def test_check_http_error_no_traceback(cat: str, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake({"good": tool_reply(), "bad": TransportError("HTTP 500 Boom")})
    assert main(["catalog", "check", "--catalog", cat], transport=fake) == 1
    assert "HTTP 500" in capsys.readouterr().out


def test_check_unreachable_distinct_message(cat: str, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake({"good": TransportError("refused", unreachable=True)})
    assert main(["catalog", "check", "--catalog", cat], transport=fake) == 3
    assert "cannot reach" in capsys.readouterr().err


def test_check_refuses_cloud_model(cat: str, capsys: pytest.CaptureFixture[str]) -> None:
    fake = Fake({})
    assert main(["catalog", "check", "--catalog", cat, "--model", "cloud"], transport=fake) == 2
    assert fake.calls == [] and "proxy" in capsys.readouterr().err


def test_check_unknown_model(cat: str) -> None:
    assert main(["catalog", "check", "--catalog", cat, "--model", "nope"], transport=Fake({})) == 2


def test_show_default_catalog(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["catalog", "show"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["id", "provider", "tier", "efforts", "flags"]
    assert len(lines) == 11
    assert any("devstral:24b" in ln and "holdout" in ln for ln in lines)
    assert lines[1].startswith("qwen3.5:9b")


def test_bad_catalog_reports_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = tmp_path / "bad.toml"
    p.write_text('[[model]]\nid = "a"\nbogus = 1\n', encoding="utf-8")
    assert main(["catalog", "show", "--catalog", str(p)]) == 2
    assert "bogus" in capsys.readouterr().err
    assert main(["catalog", "show", "--catalog", str(tmp_path / "missing.toml")]) == 2


def test_json_body_is_serialisable() -> None:
    fake = Fake({"m": tool_reply()})
    check_model("m", transport=fake)
    json.dumps(fake.calls[0][1])
