"""Unit tests for text aggregator placeholders."""
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from companion.ocr_reader import OCRResult
from companion.text_aggregator import TextAggregator


def test_aggregator_concatenates_text():
    aggregator = TextAggregator()
    fragments = [
        OCRResult(text="Hello", confidence=0.9),
        OCRResult(text="world", confidence=0.8),
    ]
    result = aggregator.aggregate(fragments)
    assert result.text == "Hello world"
