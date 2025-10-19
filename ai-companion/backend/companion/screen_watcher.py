"""Screen capture utilities leveraging MSS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from mss import mss

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class ScreenCapture:
    """Container for captured screen frame data."""

    image: np.ndarray
    monitor: dict


class ScreenWatcher:
    """Polls the screen at a configurable interval."""

    def __init__(self, monitor: Optional[int] = None):
        self._monitor = monitor
        self._mss = mss()
        logger.debug("Initialized ScreenWatcher for monitor={monitor}", monitor=monitor)

    def grab(self) -> ScreenCapture:
        """Capture a frame of the target monitor."""
        monitor = self._mss.monitors[self._monitor or 1]
        raw = self._mss.grab(monitor)
        image = np.array(raw)[:, :, :3]
        logger.trace("Captured frame size {shape}", shape=image.shape)
        return ScreenCapture(image=image, monitor=monitor)
