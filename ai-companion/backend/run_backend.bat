@echo off
setlocal enabledelayedexpansion

echo [AI Companion] Starting backend bootstrap...
where python >nul 2>&1
if errorlevel 1 (
    echo [AI Companion] Python is required but was not found in PATH.
    exit /b 1
)

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo [AI Companion] Installing Python dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo [AI Companion] Launching backend service...
python -m companion.main
