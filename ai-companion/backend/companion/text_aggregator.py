"""Aggregates OCR results into concise summaries."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import time
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .ocr_reader import OCRResult
from .utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedText:
    """Represents processed text ready for downstream usage."""

    text: str
    language: str = "unknown"


@dataclass
class _Entry:
    text: str
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]]
    timestamp: float
    language: str


class TextAggregator:
    """Aggregate OCR fragments with deduplication and language detection."""

    def __init__(
        self,
        window_seconds: float = 2.0,
        *,
        dedup_window: float = 1.2,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self._window = max(0.1, float(window_seconds))
        self._dedup_window = max(0.0, float(dedup_window))
        self._time = time_func or time.monotonic
        self._entries: Deque[_Entry] = deque()

    # ------------------------------------------------------------------
    def aggregate(self, fragments: Iterable[OCRResult]) -> AggregatedText:
        """Compatibility helper combining ``add`` and ``get_recent``."""

        self.add(fragments)
        texts = self.get_recent()
        language = self.get_language()
        combined = " ".join(texts).strip()
        logger.debug(
            "Aggregated {count} fragments into {length} characters",
            count=len(texts),
            length=len(combined),
        )
        return AggregatedText(text=combined, language=language)

    # ------------------------------------------------------------------
    def add(
        self,
        fragments: Iterable[OCRResult | Mapping[str, object]],
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add OCR fragments to the sliding window."""

        base_ts = self._time() if timestamp is None else float(timestamp)
        for index, fragment in enumerate(fragments):
            text, confidence, bbox = self._normalize_fragment(fragment)
            if not text:
                continue

            if self._is_noise(text):
                logger.trace("Skipping noisy fragment", fragment=text)
                continue

            point_ts = base_ts + index * 1e-3
            self._purge(point_ts)
            if self._is_duplicate(text, bbox, point_ts):
                logger.trace("Skipping duplicate fragment", fragment=text)
                continue

            language = self._detect_language(text)
            entry = _Entry(text=text, confidence=confidence, bbox=bbox, timestamp=point_ts, language=language)
            self._entries.append(entry)

    def get_recent(self, window_seconds: Optional[float] = None, *, now: Optional[float] = None) -> List[str]:
        """Return ordered distinct texts within the recent window."""

        current = self._time() if now is None else float(now)
        window = self._window if window_seconds is None else max(0.0, float(window_seconds))
        self._purge(current)

        texts: List[str] = []
        seen: set[str] = set()
        for entry in self._entries:
            if current - entry.timestamp > window:
                continue
            if entry.text in seen:
                continue
            seen.add(entry.text)
            texts.append(entry.text)
        return texts

    def get_language(self, *, now: Optional[float] = None) -> str:
        """Infer the dominant language from recent fragments."""

        current = self._time() if now is None else float(now)
        self._purge(current)
        if not self._entries:
            return "unknown"

        counter: Counter[str] = Counter()
        latest_ts: Dict[str, float] = {}
        for entry in self._entries:
            if current - entry.timestamp > self._window:
                continue
            if entry.language != "unknown":
                counter[entry.language] += 1
                latest_ts[entry.language] = max(latest_ts.get(entry.language, float("-inf")), entry.timestamp)
        if not counter:
            return "unknown"
        language = max(counter.keys(), key=lambda lang: (counter[lang], latest_ts.get(lang, float("-inf"))))
        return language

    # ------------------------------------------------------------------
    def _normalize_fragment(self, fragment: OCRResult | Mapping[str, object]) -> Tuple[str, float, Optional[Tuple[float, float, float, float]]]:
        if isinstance(fragment, OCRResult):
            text = (fragment.text or "").strip()
            bbox = self._normalize_bbox(fragment.bbox)
            return text, float(fragment.confidence), bbox

        mapping = fragment
        text = str(mapping.get("text", "")).strip()
        confidence = float(mapping.get("score", 1.0) or 1.0)
        bbox = self._normalize_bbox(mapping.get("bbox"))
        return text, confidence, bbox

    def _normalize_bbox(self, bbox: object | None) -> Optional[Tuple[float, float, float, float]]:
        if bbox is None:
            return None
        try:
            sequence: Sequence[float] = tuple(float(v) for v in bbox)  # type: ignore[arg-type]
        except TypeError:
            return None
        if len(sequence) < 4:
            return None
        xs = sequence[0::2]
        ys = sequence[1::2]
        return (min(xs), min(ys), max(xs), max(ys))

    def _is_noise(self, text: str) -> bool:
        if not text:
            return True
        cleaned = text.strip()
        if len(cleaned) == 1 and not cleaned.isalnum():
            return True
        alnum = sum(ch.isalnum() for ch in cleaned)
        cjk = sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in cleaned)
        return (alnum + cjk) == 0

    def _is_duplicate(
        self,
        text: str,
        bbox: Optional[Tuple[float, float, float, float]],
        timestamp: float,
    ) -> bool:
        if not self._entries:
            return False
        for entry in reversed(self._entries):
            if timestamp - entry.timestamp > self._dedup_window:
                break
            if entry.text == text:
                return True
            if bbox and entry.bbox:
                if self._bbox_iou(bbox, entry.bbox) >= 0.7:
                    return True
        return False

    def _bbox_iou(
        self,
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
        area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
        union = max(area_a + area_b - inter_area, 1e-6)
        return float(inter_area / union)

    def _detect_language(self, text: str) -> str:
        has_cjk = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
        has_kana = any(0x3040 <= ord(ch) <= 0x30FF for ch in text)
        has_alpha = any(ch.isalpha() for ch in text)
        if has_kana and has_cjk:
            return "ja"
        if has_kana:
            return "ja"
        if has_cjk:
            return "zh"
        if has_alpha:
            return "en"
        return "unknown"

    def _purge(self, now: float) -> None:
        if not self._entries:
            return
        window = max(self._window, self._dedup_window)
        while self._entries and now - self._entries[0].timestamp > window:
            self._entries.popleft()


__all__ = ["TextAggregator", "AggregatedText"]
