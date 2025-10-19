"""Screen capture utilities leveraging MSS."""
from __future__ import annotations

import threading
import time
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
    """Poll the screen on a background thread and expose motion metrics."""

    def __init__(self, monitor: Optional[int] = None, poll_interval: float = 0.2) -> None:
        self._monitor = monitor or 1
        self._mss = mss()
        self._poll_interval = max(0.02, float(poll_interval))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Condition(self._frame_lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_motion: float = 0.0
        self._previous_gray: Optional[np.ndarray] = None
        self._capture_count = 0
        logger.debug(
            "Initialized ScreenWatcher for monitor={monitor} (interval={interval})",
            monitor=self._monitor,
            interval=self._poll_interval,
        )

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background capture loop if not already running."""

        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="screen-watcher", daemon=True)
        self._thread.start()
        logger.debug("ScreenWatcher thread started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the background thread to stop and wait for completion."""

        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout)
        self._thread = None
        logger.debug("ScreenWatcher thread stopped")

    # ------------------------------------------------------------------
    def grab(self) -> ScreenCapture:
        """Capture a frame of the target monitor immediately."""

        monitor = self._resolve_monitor()
        raw = self._mss.grab(monitor)
        image = np.array(raw)[:, :, :3]
        logger.trace("Captured frame size {shape}", shape=image.shape)
        return ScreenCapture(image=image, monitor=monitor)

    def get_latest(self, timeout: Optional[float] = None) -> np.ndarray:
        """Return the most recently captured frame, waiting if necessary."""

        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._frame_lock:
            while self._latest_frame is None:
                if timeout is None:
                    self._frame_ready.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._frame_ready.wait(remaining)
            if self._latest_frame is None:
                raise RuntimeError("No frames captured yet")
            return self._latest_frame.copy()

    def get_motion_score(self) -> float:
        """Return the latest normalized motion score."""

        with self._frame_lock:
            return float(self._latest_motion)

    @property
    def capture_count(self) -> int:
        """Return the number of frames captured since start."""

        with self._frame_lock:
            return self._capture_count

    # ------------------------------------------------------------------
    def _resolve_monitor(self) -> dict:
        monitors = self._mss.monitors
        index = self._monitor
        if index >= len(monitors):
            index = 1
        return monitors[index]

    def _run(self) -> None:
        logger.debug("ScreenWatcher capture loop running")
        while not self._stop_event.is_set():
            try:
                capture = self.grab()
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.exception("Screen capture failed", error=str(exc))
                if self._stop_event.wait(self._poll_interval):
                    break
                continue

            gray = self._to_gray(capture.image)
            motion = self._compute_motion(gray)

            with self._frame_lock:
                self._latest_frame = capture.image
                self._latest_motion = motion
                self._capture_count += 1
                self._frame_ready.notify_all()

            if self._stop_event.wait(self._poll_interval):
                break

        logger.debug("ScreenWatcher capture loop exited")

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 2:
            return array
        if array.ndim == 3 and array.shape[2] >= 3:
            b, g, r = array[..., 0], array[..., 1], array[..., 2]
            return 0.114 * b + 0.587 * g + 0.299 * r
        return array.reshape(array.shape[0], -1)

    def _compute_motion(self, gray: np.ndarray) -> float:
        if gray is None or gray.size == 0:
            return 0.0
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            self._previous_gray = gray
            return 0.0
        diff = np.abs(gray - self._previous_gray)
        self._previous_gray = gray
        score = float(np.mean(diff) / 255.0)
        return float(min(max(score, 0.0), 1.0))


__all__ = ["ScreenWatcher", "ScreenCapture"]
