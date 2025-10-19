from __future__ import annotations

from companion.ocr_reader import OCRResult
from companion.text_aggregator import TextAggregator


def test_text_aggregator_window_and_language(fake_clock) -> None:
    aggregator = TextAggregator(window_seconds=2.0, time_func=fake_clock.time)

    fragments = [
        {"text": "继续冒险", "score": 0.92, "bbox": [0, 0, 40, 0, 40, 20, 0, 20]},
        {"text": "继续冒险", "score": 0.90, "bbox": [2, 1, 42, 1, 42, 21, 2, 21]},
        {"text": "***", "score": 0.80, "bbox": [0, 40, 30, 40, 30, 60, 0, 60]},
    ]
    aggregator.add(fragments)

    recent = aggregator.get_recent(now=fake_clock.time())
    assert recent == ["继续冒险"]
    assert aggregator.get_language(now=fake_clock.time()) == "zh"

    fake_clock.advance(1.0)
    aggregator.add([OCRResult(text="Hold E to glide", confidence=0.86, bbox=(0, 80, 120, 100))])

    recent = aggregator.get_recent(now=fake_clock.time())
    assert recent == ["继续冒险", "Hold E to glide"]
    assert aggregator.get_language(now=fake_clock.time()) == "en"

    fake_clock.advance(1.5)
    assert aggregator.get_recent(now=fake_clock.time()) == ["Hold E to glide"]

    fake_clock.advance(2.5)
    assert aggregator.get_recent(now=fake_clock.time()) == []
