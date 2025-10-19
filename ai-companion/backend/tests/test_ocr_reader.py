from __future__ import annotations

from typing import List

import numpy as np
import pytest

from companion import ocr_reader
from companion.ocr_reader import OCRReader


class FakeRapidOCR:
    def __call__(self, frame: np.ndarray):  # pragma: no cover - mimic external lib
        outputs: List = [
            ("继续冒险", 0.95, [[0, 0], [40, 0], [40, 20], [0, 20]]),
            ("准备出发", 0.96, [[0, 4], [40, 4], [40, 24], [0, 24]]),
            ("Vaporize", 0.88, [[0, 80], [60, 80], [60, 100], [0, 100]]),
            ("123456", 0.83, [[0, 120], [60, 120], [60, 140], [0, 140]]),
            ("ignored", 0.32, [[0, 160], [60, 160], [60, 180], [0, 180]]),
        ]
        return outputs, 0.01


def test_rapidocr_stub_filters_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_reader, "RapidOCR", FakeRapidOCR)
    reader = OCRReader(confidence_threshold=0.5, max_width=256)

    frame = np.ones((120, 160, 3), dtype=np.uint8) * 255
    detections = reader.detect(frame, merge_lines=True)

    texts = [item["text"] for item in detections]
    assert texts == ["继续冒险 准备出发", "Vaporize", "123456"]
    assert all(item["score"] >= 0.5 for item in detections)

    fragments = reader.read(frame)
    assert len(fragments) == 4
    assert fragments[0].bbox is not None

    assert reader.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    assert reader.detect(None) == []
