#!/usr/bin/env bash
set -euo pipefail

echo "[AI Companion] Starting backend bootstrap..."
if ! command -v python >/dev/null 2>&1; then
  echo "[AI Companion] Python is required but not found in PATH." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[AI Companion] Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "[AI Companion] Launching backend service..."
exec python -m companion.main
