"""Scaffold tests: `steproute.core` stays stdlib-only; the CLI works."""

import subprocess
import sys

import pytest

from steproute.cli import main


def _loaded_top_level(statement: str) -> set[str]:
    code = f"import sys; {statement}; print('\\n'.join(sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    return {name.split(".")[0] for name in out}


def test_core_import_is_stdlib_only() -> None:
    # Fresh interpreters: pytest itself is already loaded in this process.
    # The bare-interpreter baseline excludes venv startup hooks (_virtualenv, sitecustomize).
    baseline = _loaded_top_level("pass")
    loaded = _loaded_top_level("import steproute.core")
    allowed = set(sys.stdlib_module_names) | {"steproute"} | baseline
    offenders = sorted(loaded - allowed)
    assert not offenders, f"third-party modules imported by steproute.core: {offenders}"


def test_subpackages_importable() -> None:
    import steproute.collectors
    import steproute.eval
    import steproute.graph
    import steproute.ml  # noqa: F401


def test_cli_version() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_cli_no_command() -> None:
    assert main([]) == 0
