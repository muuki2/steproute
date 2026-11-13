#!/usr/bin/env bash
# Install the official SWE-bench harness into a project-local environment, pinned.
#
#   scripts/install_swebench.sh
#
# Creates .cache/steproute/swebench-env (Python 3.12) and installs swebench==5.0.2 with every
# dependency at the exact version in scripts/swebench-requirements.txt. Nothing global: the
# environment lives in the project's .cache (ignored by git), and `steproute tasks check` and
# `steproute replay` find it there on their own (steproute.tasks.check.find_harness) when run
# from the project root. Remove it with `rm -rf .cache/steproute/swebench-env`.
#
# The requirements file was produced with
#   uv pip compile --python-version 3.12 --python-platform aarch64-apple-darwin --no-header \
#     <(echo swebench==5.0.2) -o scripts/swebench-requirements.txt
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

env=.cache/steproute/swebench-env
uv venv "$env" --python 3.12 --quiet --allow-existing
uv pip install --quiet --python "$env/bin/python" -r scripts/swebench-requirements.txt
"$env/bin/python" -c "import importlib.metadata as m; print('swebench', m.version('swebench'), 'in $env')"
