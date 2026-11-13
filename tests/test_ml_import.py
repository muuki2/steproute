"""The ml extra stays optional: the CLI imports and runs without lightgbm or numpy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=60
    )


def test_cli_imports_without_the_ml_extra() -> None:
    code = (
        "import sys\n"
        "sys.modules['lightgbm'] = None\n"  # makes `import lightgbm` raise ImportError
        "sys.modules['numpy'] = None\n"
        "import steproute.cli\n"
        "import steproute.ml.args\n"
        "loaded = [m for m in ('lightgbm', 'numpy', 'steproute.ml.model') if sys.modules.get(m)]\n"
        "assert not loaded, loaded\n"
        "try:\n"
        "    steproute.cli.main(['--version'])\n"
        "except SystemExit as e:\n"
        "    assert e.code == 0\n"
    )
    out = _run(code)
    assert out.returncode == 0, out.stderr
    assert "steproute" in out.stdout


def test_ml_command_names_the_missing_extra(tmp_path: Path) -> None:
    code = (
        "import sys\n"
        "sys.modules['lightgbm'] = None\n"
        "from steproute.cli import main\n"
        "sys.exit(main(sys.argv[1:]))\n"
    )
    out = _run(
        code, "ml", "train", "--data", str(tmp_path / "g.jsonl"), "--out", str(tmp_path / "m")
    )
    assert out.returncode == 2
    assert "steproute[ml]" in out.stderr
