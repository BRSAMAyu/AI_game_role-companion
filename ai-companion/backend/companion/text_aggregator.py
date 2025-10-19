"""Aggregates OCR results into concise summaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .ocr_reader import OCRResult
from .utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedText:
    """Represents processed text ready for downstream usage."""

    text: str


class TextAggregator:
    """Simple aggregator that concatenates OCR fragments."""

    def aggregate(self, fragments: Iterable[OCRResult]) -> AggregatedText:
        texts: List[str] = [fragment.text.strip() for fragment in fragments if fragment.text]
        combined = " ".join(texts).strip()
        logger.debug("Aggregated {count} fragments into {length} characters", count=len(texts), length=len(combined))
        return AggregatedText(text=combined)
