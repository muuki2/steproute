"""Generate the JSON Schema for the trace records from the dataclasses.

Regenerate the checked-in file with::

    uv run python -m steproute.core.schema_gen --write
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
import typing
from importlib import resources
from pathlib import Path
from typing import Any, cast

from steproute.core.schema import RECORD_TYPES, SCHEMA_VERSION, _Record, field_hints

SCHEMA_FILENAME = "trace.schema.json"
SCHEMA_PATH = Path(__file__).with_name(SCHEMA_FILENAME)
REGENERATE_CMD = "uv run python -m steproute.core.schema_gen --write"


def _hint_schema(hint: Any) -> dict[str, Any]:
    if hint is Any:
        return {}
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union or origin is types.UnionType:
        return {
            "anyOf": [_hint_schema(a) if a is not type(None) else {"type": "null"} for a in args]
        }
    if origin is list or origin is tuple:
        return {"type": "array", "items": _hint_schema(args[0])}
    if origin is dict:
        return {"type": "object", "additionalProperties": _hint_schema(args[1])}
    if isinstance(hint, type) and issubclass(hint, _Record):
        return {"$ref": f"#/$defs/{hint.__name__}"}
    scalars = {bool: "boolean", int: "integer", float: "number", str: "string"}
    return {"type": scalars[hint]}


def _record_schema(cls: type[_Record]) -> dict[str, Any]:
    hints = field_hints(cls)
    props: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cast("type[Any]", cls)):
        prop = _hint_schema(hints[f.name])
        if "enum" in f.metadata:
            prop["enum"] = list(f.metadata["enum"])
        props[f.name] = prop
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            required.append(f.name)
    # additionalProperties is deliberately left open: readers ignore unknown keys.
    return {"type": "object", "properties": props, "required": required}


def generate_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) for all record types, as a dict."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "steproute trace records",
        "description": f"Trace schema version {SCHEMA_VERSION}. Generated; do not edit by hand.",
        "$defs": {cls.__name__: _record_schema(cls) for cls in RECORD_TYPES},
    }


def generate_json_schema_text() -> str:
    """The generated schema serialised exactly as it is checked in."""
    return json.dumps(generate_json_schema(), indent=2) + "\n"


def load_json_schema() -> dict[str, Any]:
    """Load the checked-in JSON Schema shipped with the package."""
    text = resources.files("steproute.core").joinpath(SCHEMA_FILENAME).read_text("utf-8")
    return cast("dict[str, Any]", json.loads(text))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    text = generate_json_schema_text()
    if args == ["--write"]:
        SCHEMA_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {SCHEMA_PATH}")
        return 0
    if args == ["--check"]:
        if SCHEMA_PATH.read_text(encoding="utf-8") != text:
            print(f"{SCHEMA_FILENAME} is out of date; run: {REGENERATE_CMD}")
            return 1
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
