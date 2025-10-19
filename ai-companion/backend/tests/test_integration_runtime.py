from __future__ import annotations

import concurrent.futures
from typing import Iterable, List, Sequence

import numpy as np

from companion.config import CompanionSettings
from companion.main import CompanionRuntime
from companion.ocr_reader import OCRResult
from companion.orchestrator import DialogueOrchestrator


class FakeWatcher:
    def __init__(self, frames: Sequence[np.ndarray], motions: Sequence[float]) -> None:
        self._frames = list(frames)
        self._motions = list(motions)
        self._index = 0
        self._current = 0

    def start(self) -> None:  # pragma: no cover - interface stub
        pass

    def stop(self) -> None:  # pragma: no cover - interface stub
        pass

    def get_latest(self, timeout: float | None = None) -> np.ndarray:
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._current = min(self._index, len(self._motions) - 1)
        self._index = min(self._index + 1, len(self._frames) - 1)
        return frame

    def get_motion_score(self) -> float:
        return float(self._motions[self._current])


class FakeOCR:
    def __init__(self, sequences: Sequence[Iterable[str]]) -> None:
        self._sequences = list(sequences)
        self._index = 0

    def read(self, frame: np.ndarray) -> List[OCRResult]:
        lines = list(self._sequences[min(self._index, len(self._sequences) - 1)])
        self._index = min(self._index + 1, len(self._sequences) - 1)
        return [OCRResult(text=line, confidence=0.9) for line in lines]


class FakeClassifier:
    def __init__(self, outputs: Sequence[tuple[str, float]]) -> None:
        self._outputs = list(outputs)
        self._index = 0

    def classify(self, frame: np.ndarray, ocr_lines: Iterable[str], motion_score: float) -> tuple[str, float]:
        label, confidence = self._outputs[min(self._index, len(self._outputs) - 1)]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return label, confidence

    def _compute_features(self, frame: np.ndarray, lines: Iterable[str], motion_score: float) -> dict:
        return {
            "red_ratio": 0.0,
            "low_texture_ratio": 0.0,
            "circle_score": 0.0,
            "brightness_var": 0.0,
            "brightness_delta": 0.0,
            "long_line_present": 0.0,
            "bottom_lines": 0.0,
            "line_count": float(len(list(lines))),
            "motion_score": float(motion_score),
        }


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._executor = _ImmediateExecutor()

    def speak(self, text: str, mode: str = "dialog"):
        self.calls.append((text, mode))
        future: concurrent.futures.Future[None] = concurrent.futures.Future()
        future.set_result(None)
        return future


class _ImmediateExecutor:
    def shutdown(self, wait: bool = False) -> None:  # pragma: no cover - stub
        pass


class FakeOverlay:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def send_text(self, text: str, ms: int, style: str) -> bool:
        self.messages.append({"text": text, "ms": ms, "style": style})
        return True

    def close(self) -> None:  # pragma: no cover - stub
        pass


class DummyCooldown:
    def allow(self, key: str, seconds: float) -> bool:
        return True

    def block_all(self, seconds: float) -> None:  # pragma: no cover - stub
        pass

    def clear(self, key: str) -> None:  # pragma: no cover - stub
        pass

    def reset(self) -> None:  # pragma: no cover - stub
        pass


class DummyRecent:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add(self, text: str) -> None:
        self.added.append(text)

    def seen(self, text: str) -> bool:
        return False


def test_runtime_small_loop() -> None:
    steps = 8
    frames = [np.zeros((80, 80, 3), dtype=np.uint8) for _ in range(steps)]
    motions = [0.12, 0.02, 0.01, 0.02, 0.12, 0.01, 0.02, 0.02]
    ocr_sequences = [
        ["危险来袭"],
        ["凯亚：注意脚下[y=0.82]", "按F 继续[y=0.84]"],
        ["世界地图", "传送点"],
        ["加载中"],
        ["护盾告急"],
        ["打开背包整理物资"],
        ["剧情继续[y=0.82]", "按F 继续[y=0.86]"],
        ["??"],
    ]
    classifier_outputs = [
        ("battle", 0.8),
        ("dialog", 0.72),
        ("map", 0.7),
        ("battle", 0.4),
        ("battle", 0.82),
        ("menu", 0.7),
        ("dialog", 0.75),
        ("unknown", 0.45),
    ]

    watcher = FakeWatcher(frames, motions)
    ocr = FakeOCR(ocr_sequences)
    classifier = FakeClassifier(classifier_outputs)
    tts = FakeTTS()
    overlay = FakeOverlay()
    cooldown = DummyCooldown()
    recent = DummyRecent()
    orchestrator = DialogueOrchestrator()
    orchestrator._random.seed(2024)  # type: ignore[attr-defined]

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

    runtime.run_steps(steps)

    assert any(len(text) <= 12 for text, mode in tts.calls if mode == "battle")
    assert any("继续" in text for text, mode in tts.calls if mode == "dialog")
    assert any(20 <= len(text) <= 28 for text, mode in tts.calls if mode == "menu")

    assert len(overlay.messages) >= 3
    assert all(msg["style"] in {"battle", "default"} for msg in overlay.messages)

    assert len(tts.calls) < steps
    assert len(tts.calls) <= steps - 2  # unknown and low-confidence paths muted

    runtime.shutdown()
