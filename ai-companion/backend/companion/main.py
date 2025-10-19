"""Entrypoint for backend services."""
from __future__ import annotations

import signal
import sys
import threading
import time

from .config import CompanionSettings
from .hotkeys import HotkeyManager
from .orchestrator import BackendOrchestrator
from .utils.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    settings = CompanionSettings()
    configure_logging()
    orchestrator = BackendOrchestrator(settings=settings)
    hotkeys = HotkeyManager()
    stop_event = threading.Event()

    def on_toggle(active: bool) -> None:
        logger.info("Companion active state changed: {state}", state=active)

    def signal_handler(_sig, _frame):
        logger.info("Shutdown requested")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    orchestrator.start()
    hotkeys.register_toggle(settings.hotkey_toggle, on_toggle)

    logger.info("Backend running. Press Ctrl+C to exit.")
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    finally:
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
