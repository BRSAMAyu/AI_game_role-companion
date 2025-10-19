#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest pytest-cov ruff

ruff check companion
pytest -q --maxfail=1 --disable-warnings --cov=companion --cov-report=term-missing

if ($env:RUN_BENCH -and $env:RUN_BENCH -ne '0') {
    python bench/micro_bench_runtime.py --seconds 5
}
