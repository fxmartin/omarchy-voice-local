#!/usr/bin/env bash
# The checks every change must pass locally, matching CONTRIBUTING.md and CI.
# Automation that looks for a repository quality gate runs this file.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run() { uv run --frozen --extra dev "$@"; }

run python -m compileall -q src tests tools
bash -n install.sh uninstall.sh share/install-paths.sh .githooks/pre-commit .githooks/pre-push
run python -m unittest discover -s tests
python3 tools/check_public_files.py --history HEAD
