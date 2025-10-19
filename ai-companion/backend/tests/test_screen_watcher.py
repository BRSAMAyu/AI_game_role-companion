from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from companion import screen_watcher
from companion.screen_watcher import ScreenWatcher


class FakeGrab:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame

    def __array__(self, dtype=None):  # type: ignore[override]
        if dtype is None:
            return self._frame
        return np.asarray(self._frame, dtype=dtype)


class FakeMSS:
    def __init__(self, frames: List[np.ndarray]) -> None:
        self._frames = frames
        height, width = frames[0].shape[:2]
        self.monitors = [None, {"top": 0, "left": 0, "width": width, "height": height}]
        self._index = 0

    def grab(self, monitor):  # pragma: no cover - interface mirror
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index = min(self._index + 1, len(self._frames) - 1)
        return FakeGrab(frame)


@pytest.mark.parametrize("poll_interval", [0.03])
def test_screen_watcher_motion(monkeypatch: pytest.MonkeyPatch, poll_interval: float) -> None:
    base = np.zeros((60, 60, 4), dtype=np.uint8)
    base[:, :, :3] = (20, 20, 60)

    slight = np.zeros_like(base)
    slight[:, :, :3] = (120, 60, 20)

    large = np.zeros_like(base)
    large[:, :, :3] = (250, 200, 40)

    frames = [base, base, slight, slight, large, large]
    fake = FakeMSS(frames)
    monkeypatch.setattr(screen_watcher, "mss", lambda: fake)

    watcher = ScreenWatcher(poll_interval=poll_interval)
    watcher.start()
    try:
        frame = watcher.get_latest(timeout=1.0)
        assert frame.shape[:2] == (60, 60)

        deadline = time.time() + 2.0
        while watcher.capture_count < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert watcher.capture_count >= 2
        still_motion = watcher.get_motion_score()
        assert still_motion == pytest.approx(0.0, abs=0.03)

        deadline = time.time() + 2.0
        while watcher.capture_count < 3 and time.time() < deadline:
            time.sleep(0.02)
        mild_motion = watcher.get_motion_score()
        assert mild_motion > still_motion + 0.02

        deadline = time.time() + 2.0
        while watcher.capture_count < 5 and time.time() < deadline:
            time.sleep(0.02)
        intense_motion = watcher.get_motion_score()
        assert intense_motion > mild_motion + 0.02

    finally:
        count_before_stop = watcher.capture_count
        watcher.stop()
        time.sleep(0.1)
        assert watcher.capture_count == count_before_stop
