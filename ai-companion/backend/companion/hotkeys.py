"""Global hotkey registration using keyboard library."""
from __future__ import annotations

from typing import Callable, List, Tuple

import keyboard

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


class HotkeyManager:
    """Registers and manages global hotkeys."""

    def __init__(self) -> None:
        self._active = True
        self._registered: List[Tuple[str, Callable[[], None]]] = []
        logger.debug("HotkeyManager initialized")

    def register_hotkey(self, hotkey: str, callback: Callable[[], None]) -> None:
        """Register a generic hotkey callback."""

        keyboard.add_hotkey(hotkey, callback)
        self._registered.append((hotkey, callback))
        logger.info("Registered hotkey {hotkey}", hotkey=hotkey)

    def register_toggle(self, hotkey: str, callback: Callable[[bool], None]) -> None:
        """Register a toggle hotkey that flips the active state."""

        def handler() -> None:
            self._active = not self._active
            logger.info("Hotkey {hotkey} toggled to {state}", hotkey=hotkey, state=self._active)
            callback(self._active)

        self.register_hotkey(hotkey, handler)

    def is_active(self) -> bool:
        return self._active
