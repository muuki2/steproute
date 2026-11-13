# Contributing

## Setup

```
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

CI runs the same checks on Python 3.11 and 3.12.

## Conventions

- Python 3.11+, `uv`, `src/` layout, `pytest`, `ruff`, `mypy --strict` on `src/steproute/core`.
- `steproute.core` has no third-party runtime dependencies. Heavier parts are extras: `[proxy]`, `[mcp]`, `[ml]`, `[graph]`.
- Money is USD; prices are per million tokens; time is milliseconds; timestamps are Unix seconds (float).
- No network calls in unit tests. Provider calls are mocked or recorded.
- Never log prompt or completion text unless `capture_text=true` is set explicitly.

## Pull requests

One pull request per issue. Put `Closes #N` in the PR description. Work items are listed in [docs/ISSUES.md](docs/ISSUES.md).
