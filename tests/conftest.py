"""Shared test setup. No test may read or write the real shared spend ledger.

Every test that needs a ledger passes an explicit path under its temporary directory. As a
backstop, the default location is pointed at a fresh directory for each test (through
``XDG_STATE_HOME``, with ``STEPROUTE_LEDGER`` removed), and a test that left anything there
fails: it used the default location by accident.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _no_default_ledger(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    state = tmp_path_factory.mktemp("default-state")
    monkeypatch.setenv("XDG_STATE_HOME", str(state))
    monkeypatch.delenv("STEPROUTE_LEDGER", raising=False)
    yield
    used = sorted(p.name for p in state.iterdir())
    assert not used, f"a test wrote to the default ledger location: {used}"
