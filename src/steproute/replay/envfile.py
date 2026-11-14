"""``--env-file``: load ``NAME=value`` lines for the runner's own environment. Standard library.

The format, and nothing more:

* ``NAME=value``, with an optional leading ``export``;
* the value may be wrapped in one pair of single or double quotes, taken literally;
* ``#`` starts a comment line; blank lines are skipped; the last line needs no newline.

No interpolation (``$OTHER`` stays as written) and no escapes. A line that is not one of
these is an error naming only its line number: the line may hold a key, so neither the line
nor any value is ever put in a message. See ``docs/REPLAY.md``, "Keys".
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, MutableMapping
from pathlib import Path

__all__ = ["EnvFileError", "apply_env", "load_env_file", "parse_env"]

_LINE = re.compile(r"^(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class EnvFileError(ValueError):
    """The env file cannot be used. The message never contains a value."""


def parse_env(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m is None:
            raise EnvFileError(f"env file line {n}: expected NAME=value")
        name, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value[:1] in ("'", '"') or value[-1:] in ("'", '"'):
            raise EnvFileError(f"env file line {n}: unbalanced quotes")
        out[name] = value
    return out


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Read and parse ``path``. Raises ``EnvFileError`` (never with a value in it)."""
    try:
        text = Path(path).read_text("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        reason = exc.strerror if isinstance(exc, OSError) and exc.strerror else type(exc).__name__
        raise EnvFileError(f"cannot read the env file: {reason}") from None
    return parse_env(text)


def apply_env(values: Mapping[str, str], environ: MutableMapping[str, str]) -> list[str]:
    """Set every value not already set in ``environ`` (a variable already in the environment
    wins). Returns the values now in force from the file, for the redactor."""
    used = []
    for name, value in values.items():
        if environ.get(name):
            continue
        environ[name] = value
        used.append(value)
    return used
