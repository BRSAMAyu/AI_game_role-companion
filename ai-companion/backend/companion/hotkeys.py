"""Global hotkey registration using keyboard library."""
from __future__ import annotations

from typing import Callable

import keyboard

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


class HotkeyManager:
    """Registers and manages global hotkeys."""

    def __init__(self) -> None:
        self._active = True
        logger.debug("HotkeyManager initialized")

    def register_toggle(self, hotkey: str, callback: Callable[[bool], None]) -> None:
        logger.info("Registering hotkey {hotkey}", hotkey=hotkey)

        def handler():
            self._active = not self._active
            logger.info("Hotkey {hotkey} toggled to {state}", hotkey=hotkey, state=self._active)
            callback(self._active)

        keyboard.add_hotkey(hotkey, handler)

    def is_active(self) -> bool:
        return self._active
