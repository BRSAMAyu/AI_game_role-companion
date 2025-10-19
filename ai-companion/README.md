# AI Game Companion (P0 Skeleton)

This repository provides a minimal project skeleton for a cross-process "game companion" that reads on-screen text, produces concise summaries, and displays them via an Electron overlay while performing text-to-speech (TTS) playback.

## Project Layout

```
ai-companion/
  backend/
    companion/          # Python package containing the backend services
    requirements.txt    # Python dependencies
    run_backend.(sh|bat)# Helper scripts for launching the backend
  overlay/
    package.json        # Electron project configuration
    src/                # Overlay sources (main, preload, renderer)
```

## Prerequisites

- **Python 3.10+** with `pip`
- **Node.js 18+** with `npm`
- Windows 10/11 is the primary target platform, but the backend also supports macOS/Linux.
- Ensure GPU overlays or other screen-capture tools are disabled if they block MSS on Windows.

## Installation & Setup

### Backend (Python)

```bash
cd ai-companion/backend
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\\Scripts\\activate
pip install -r requirements.txt
python -m companion.main
```

Alternatively, use the helper scripts:

- Windows: `run_backend.bat`
- macOS/Linux: `./run_backend.sh`

These scripts validate Python availability, install dependencies, and start the backend with informative logging.

### Overlay (Electron)

```bash
cd ai-companion/overlay
npm install
npm run start
```

The overlay launches a transparent, always-on-top window and starts a WebSocket server on `ws://127.0.0.1:17865`. The Python backend connects as a client and sends `{ type: "showText", text, ms }` messages to show subtitles.

## Hotkeys & Controls

- **Alt+`** toggles whether the overlay ignores mouse input (mouse passthrough) on both backend (state tracking) and overlay (global shortcut). The backend also listens for the same hotkey to switch internal active state.

## Development Notes

- No business logic is implemented yet; all modules provide placeholders and minimal class structures for future expansion.
- The backend uses `loguru` for structured logging and `pydantic` for configuration via environment variables.
- Unit tests (pytest) cover the text aggregator and scene classifier placeholders.
- The overlay uses Electron with a lightweight renderer and exposes a WebSocket server via the `ws` package.

## Windows-Specific Considerations

- Run PowerShell or Command Prompt as Administrator to allow the `keyboard` library to register global hotkeys.
- If TTS voices are missing, install the desired language pack via Windows Settings.
- Ensure GPU drivers are up to date; MSS relies on DirectX for accelerated capture.

## Communication Flow

1. Electron overlay starts and hosts a WebSocket server (`ws://127.0.0.1:17865`).
2. Python backend connects to the WebSocket server and orchestrates screen capture, OCR, simple classification, and text aggregation.
3. Backend sends text payloads to the overlay, which displays them for the configured duration while TTS plays back the same text.

This skeleton is ready for implementing the full AI pipeline without requiring process injection or direct memory access.
