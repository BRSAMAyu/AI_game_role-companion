#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pytest pytest-cov ruff

ruff check companion
pytest -q --maxfail=1 --disable-warnings --cov=companion --cov-report=term-missing

if [[ "${RUN_BENCH:-0}" != "0" ]]; then
  python bench/micro_bench_runtime.py --seconds 5
fi
