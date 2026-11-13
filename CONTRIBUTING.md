# Contributing

## Setup

```
uv sync --extra dev --extra ml
git config core.hooksPath .githooks
```

`--extra ml` installs LightGBM for the tabular predictor (`steproute.ml`); its tests are skipped without it. The second line enables the pre-push hook, which runs the CI pipeline before anything leaves your machine.

## CI is local

The repository uses no hosted CI. One script is the pipeline:

```
scripts/ci.sh
```

It runs, in order:

1. `ruff check` and `ruff format --check`
2. `mypy --strict` on `src/steproute/core` and `pytest`, on Python 3.11 and 3.12
3. a package build, then installs the wheel into a throwaway environment and runs it

Each interpreter gets its own environment under `.ci/`, installed from `uv.lock`, so your `.venv` is not touched. `CI_PYTHONS=3.12 scripts/ci.sh` runs a single interpreter.

After pushing, publish the result so the pull request shows it:

```
scripts/ci.sh --report
```

This posts a `local-ci` commit status on `HEAD`. It needs a clean working tree and the GitHub CLI (`gh`) signed in. If the same commit already passed in the pre-push hook, it reports without running again.

For a quick loop while editing, the individual tools work as usual: `uv run pytest`, `uv run ruff check .`, `uv run mypy`.

## Conventions

- Python 3.11+, `uv`, `src/` layout, `pytest`, `ruff`, `mypy --strict` on `src/steproute/core`.
- `steproute.core` has no third-party runtime dependencies. Heavier parts are extras: `[proxy]`, `[mcp]`, `[ml]`, `[graph]`.
- Money is USD; prices are per million tokens; time is milliseconds; timestamps are Unix seconds (float).
- No network calls in unit tests. Provider calls are mocked or recorded.
- Never log prompt or completion text unless `capture_text=true` is set explicitly.

## Pull requests

One pull request per issue. Put `Closes #N` in the PR description. A pull request is ready for review when `local-ci` is green on its head commit. Work items are listed in [docs/ISSUES.md](docs/ISSUES.md).
