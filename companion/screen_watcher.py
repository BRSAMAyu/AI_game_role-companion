"""ScreenWatcher module capturing screen frames in the background."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import cv2
from loguru import logger
import mss


Region = Optional[Tuple[int, int, int, int]]


class ScreenWatcher:
    """Periodically captures the primary screen in a background thread."""

    def __init__(self, region: Region = None, interval_ms: int = 500) -> None:
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")

        self._region: Region = region
        self._interval: float = interval_ms / 1000.0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._previous_frame: Optional[np.ndarray] = None
        self._motion_score: float = 0.0
        self._frame_count: int = 0

    def start(self) -> None:
        """Start capturing frames in a background daemon thread."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                logger.warning("ScreenWatcher is already running")
                return

            logger.info("Starting ScreenWatcher thread")
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="ScreenWatcherThread", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the background capturing thread."""
        with self._lock:
            if not self._thread:
                return

            logger.info("Stopping ScreenWatcher thread")
            self._stop_event.set()
            thread = self._thread

        thread.join()
        with self._lock:
            self._thread = None
            self._latest_frame = None
            self._previous_frame = None
            self._motion_score = 0.0
            self._frame_count = 0

    def get_latest(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent BGR frame."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_motion_score(self) -> float:
        """Return the latest motion score between 0.0 and 1.0."""
        with self._lock:
            return float(self._motion_score)

    def get_frame_count(self) -> int:
        """Return the total number of frames captured since start."""
        with self._lock:
            return self._frame_count

    def _run(self) -> None:
        logger.debug("ScreenWatcher thread started")
        sct: Optional[mss.mss] = None
        monitor = None

        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            try:
                if sct is None:
                    logger.debug("Initializing MSS for screen capture")
                    sct = mss.mss()
                    monitor = self._build_monitor(sct)

                raw = sct.grab(monitor)
                frame = np.array(raw, dtype=np.uint8)
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                motion_score = self._compute_motion(frame)

                with self._lock:
                    self._latest_frame = frame.copy()
                    self._motion_score = motion_score
                    self._frame_count += 1

            except Exception as exc:
                logger.exception("Screen capture failed: {}", exc)
                self._reset_state()
                if sct is not None:
                    try:
                        sct.close()
                    except Exception:
                        logger.debug("Failed to close MSS cleanly", exc_info=True)
                    sct = None
                    monitor = None

                if self._stop_event.wait(1.0):
                    break
                continue

            elapsed = time.perf_counter() - start_time
            remaining = self._interval - elapsed
            if remaining > 0:
                self._stop_event.wait(remaining)

        if sct is not None:
            try:
                sct.close()
            except Exception:
                logger.debug("Failed to close MSS on shutdown", exc_info=True)

        logger.debug("ScreenWatcher thread exiting")

    def _build_monitor(self, sct: mss.mss) -> dict:
        if self._region is None:
            if len(sct.monitors) < 2:
                return sct.monitors[0]
            return sct.monitors[1]

        x, y, width, height = self._region
        return {"left": x, "top": y, "width": width, "height": height}

    def _compute_motion(self, frame: np.ndarray) -> float:
        with self._lock:
            previous = self._previous_frame
            self._previous_frame = frame.copy()

        if previous is None:
            return 0.0

        diff = cv2.absdiff(frame, previous)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

        changed_pixels = int(np.count_nonzero(thresh))
        total_pixels = thresh.size
        if total_pixels == 0:
            return 0.0
        return changed_pixels / float(total_pixels)

    def _reset_state(self) -> None:
        with self._lock:
            self._latest_frame = None
            self._previous_frame = None
            self._motion_score = 0.0
            self._frame_count = 0


if __name__ == "__main__":
    import signal
    import sys

    watcher = ScreenWatcher()
    watcher.start()

    last_count = 0
    last_time = time.perf_counter()

    def shutdown_handler(signum, frame):  # type: ignore[unused-argument]
        logger.info("Received signal %s, shutting down", signum)
        watcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            time.sleep(1.0)
            current_count = watcher.get_frame_count()
            now = time.perf_counter()
            elapsed = now - last_time
            fps = (current_count - last_count) / elapsed if elapsed > 0 else 0.0
            last_time = now
            last_count = current_count

            motion = watcher.get_motion_score()
            logger.info("FPS: {:.2f}, motion_score: {:.4f}", fps, motion)
    except KeyboardInterrupt:
        logger.info("Stopping due to keyboard interrupt")
    finally:
        watcher.stop()
