"""Micro benchmark exercising the companion runtime pipeline."""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import random
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from companion.config import CompanionSettings
from companion.main import CompanionRuntime
from companion.ocr_reader import OCRResult
from companion.orchestrator import DialogueOrchestrator


class BenchWatcher:
    def __init__(self) -> None:
        self._rng = random.Random(1234)
        self._shape = (120, 160, 3)
        self._current_motion = 0.05

    def start(self) -> None:  # pragma: no cover - runtime helper
        pass

    def stop(self) -> None:  # pragma: no cover - runtime helper
        pass

    def get_latest(self, timeout: float | None = None) -> np.ndarray:
        frame = np.zeros(self._shape, dtype=np.uint8)
        color = self._rng.randint(40, 200)
        frame[:, :] = (color, color // 2, color // 3)
        self._current_motion = self._rng.uniform(0.0, 0.15)
        return frame

    def get_motion_score(self) -> float:
        return self._current_motion


class BenchOCR:
    def __init__(self) -> None:
        self._scenes = collections.deque([
            ["危险来袭"],
            ["凯亚：注意[y=0.82]", "按F 继续[y=0.86]"],
            ["地图", "传送"],
            ["菜单"],
            [],
        ])

    def read(self, frame: np.ndarray) -> List[OCRResult]:
        self._scenes.rotate(-1)
        lines = self._scenes[0]
        return [OCRResult(text=line, confidence=0.9) for line in lines]


class BenchClassifier:
    def __init__(self) -> None:
        self._scenes = collections.deque([
            ("battle", 0.78),
            ("dialog", 0.7),
            ("map", 0.68),
            ("menu", 0.65),
            ("battle", 0.45),
            ("unknown", 0.5),
        ])

    def classify(self, frame: np.ndarray, ocr_lines: Iterable[str], motion_score: float) -> tuple[str, float]:
        self._scenes.rotate(-1)
        return self._scenes[0]

    def _compute_features(self, frame: np.ndarray, lines: Iterable[str], motion_score: float) -> dict:
        return {
            "red_ratio": 0.01,
            "low_texture_ratio": 0.6,
            "circle_score": 0.05,
            "brightness_var": 5.0,
            "brightness_delta": 1.0,
            "long_line_present": 0.0,
            "bottom_lines": 2.0,
            "line_count": float(len(list(lines))),
            "motion_score": float(motion_score),
        }


class BenchTTS:
    def __init__(self) -> None:
        self._executor = _NoopExecutor()
        self.calls: List[tuple[str, str]] = []

    def speak(self, text: str, mode: str = "dialog"):
        self.calls.append((text, mode))
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)
        return future


class _NoopExecutor:
    def shutdown(self, wait: bool = False) -> None:  # pragma: no cover
        pass


class BenchOverlay:
    def __init__(self) -> None:
        self.messages: List[dict] = []

    def send_text(self, text: str, ms: int, style: str) -> bool:
        self.messages.append({"text": text, "ms": ms, "style": style})
        return True

    def close(self) -> None:  # pragma: no cover
        pass


class AlwaysAllowCooldown:
    def allow(self, key: str, seconds: float) -> bool:
        return True

    def block_all(self, seconds: float) -> None:  # pragma: no cover
        pass

    def clear(self, key: str) -> None:  # pragma: no cover
        pass

    def reset(self) -> None:  # pragma: no cover
        pass


class RecentBypass:
    def add(self, text: str) -> None:  # pragma: no cover
        pass

    def seen(self, text: str) -> bool:
        return False


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    rank = max(0, min(len(values) - 1, int(round((pct / 100.0) * (len(values) - 1)))))
    return sorted(values)[rank]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a micro benchmark for the companion runtime.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Benchmark duration in seconds (default: 60)")
    parser.add_argument("--output", type=Path, default=Path("bench/runtime_hist.csv"), help="Histogram CSV output path")
    args = parser.parse_args()

    watcher = BenchWatcher()
    ocr = BenchOCR()
    classifier = BenchClassifier()
    tts = BenchTTS()
    overlay = BenchOverlay()
    cooldown = AlwaysAllowCooldown()
    recent = RecentBypass()
    orchestrator = DialogueOrchestrator()
    orchestrator._random.seed(1337)  # type: ignore[attr-defined]

    settings = CompanionSettings(overlay_display_ms=1800, screenshot_interval_ms=200)

    runtime = CompanionRuntime(
        settings,
        watcher=watcher,
        classifier=classifier,
        orchestrator=orchestrator,
        tts=tts,
        overlay=overlay,
        ocr_reader=ocr,
        cooldown=cooldown,
        recent_buffer=recent,
    )

    durations: List[float] = []
    histogram: collections.Counter[int] = collections.Counter()

    end_time = time.time() + max(1.0, args.seconds)
    while time.time() < end_time:
        start = time.perf_counter()
        runtime.run_once()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        durations.append(elapsed_ms)
        bucket = int(elapsed_ms // 10) * 10
        histogram[bucket] += 1

    runtime.shutdown()

    if durations:
        print(f"Iterations: {len(durations)}")
        print(f"P50: {percentile(durations, 50):.2f} ms")
        print(f"P95: {percentile(durations, 95):.2f} ms")
        print(f"P99: {percentile(durations, 99):.2f} ms")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["ms", "count"])
        for bucket in sorted(histogram):
            writer.writerow([bucket, histogram[bucket]])


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
