"""OCR interface built on top of RapidOCR."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .utils.logging_setup import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency import guard
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - fallback if dependency missing at import time
    RapidOCR = None  # type: ignore[assignment]
    logger.warning(
        "RapidOCR could not be imported; OCR functionality will be unavailable until dependencies are installed.",
    )

try:  # pragma: no cover - OpenCV is optional but preferred for resizing
    import cv2
except Exception:  # pragma: no cover - gracefully degrade if OpenCV is missing
    cv2 = None  # type: ignore[assignment]
    logger.warning("OpenCV is not available; OCR frames will not be resized for performance.")

_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_DEFAULT_MAX_WIDTH = 1280
_DEFAULT_MIN_DIMENSION = 20


@dataclass
class OCRResult:
    """Represents a detected text fragment."""

    text: str
    confidence: float
    bbox: Tuple[float, ...] | None = None


class OCRReader:
    """Wrapper around RapidOCR to simplify usage."""

    def __init__(
        self,
        *,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
        max_width: int = _DEFAULT_MAX_WIDTH,
        min_dimension: int = _DEFAULT_MIN_DIMENSION,
    ) -> None:
        if RapidOCR is None:
            raise RuntimeError(
                "RapidOCR dependency is not available. Install rapidocr-onnxruntime to enable OCR.",
            )
        self._engine = RapidOCR()
        self._confidence_threshold = confidence_threshold
        self._max_width = max_width
        self._min_dimension = min_dimension
        logger.debug(
            "RapidOCR engine initialized (confidence>={threshold}, max_width={width})",
            threshold=self._confidence_threshold,
            width=self._max_width,
        )

    def detect(self, frame: np.ndarray, *, merge_lines: bool = True) -> List[dict]:
        """Run OCR on a BGR frame and return detections.

        Parameters
        ----------
        frame:
            The input frame in BGR format.
        merge_lines:
            When ``True`` (default), detections whose center lines fall within a small
            vertical tolerance are merged into a single line of text.
        """

        if frame is None:
            logger.debug("Received None frame for OCR detection; skipping.")
            return []

        np_frame = np.asarray(frame)
        if np_frame.size == 0:
            logger.debug("Received empty frame for OCR detection; skipping.")
            return []

        if np_frame.ndim == 2:
            np_frame = np.repeat(np_frame[:, :, None], 3, axis=2)
        elif np_frame.ndim == 3 and np_frame.shape[2] > 3:
            np_frame = np_frame[:, :, :3]
        elif np_frame.ndim != 3 or np_frame.shape[2] != 3:
            logger.debug("Unsupported frame shape %s; skipping OCR.", np_frame.shape)
            return []

        height, width = np_frame.shape[:2]
        if height < self._min_dimension or width < self._min_dimension:
            logger.debug(
                "Frame too small for OCR (height=%s, width=%s); skipping.",
                height,
                width,
            )
            return []

        processed_frame, scale = self._prepare_frame(np_frame)

        try:
            outputs_raw, _elapsed = self._engine(processed_frame)
        except Exception as exc:  # pragma: no cover - defensive against backend failures
            logger.exception("RapidOCR inference failed: {error}", error=str(exc))
            return []

        detections = self._parse_outputs(outputs_raw, scale)
        if not detections:
            return []

        if merge_lines:
            return self._merge_line_detections(detections)

        return [self._format_detection(text, score, points) for text, score, points in detections]

    def read(self, image: np.ndarray) -> List[OCRResult]:
        """Compatibility wrapper returning :class:`OCRResult` instances."""

        detections = self.detect(image, merge_lines=False)
        results = [
            OCRResult(
                text=item["text"],
                confidence=float(item["score"]),
                bbox=tuple(float(v) for v in item.get("bbox", [])) if item.get("bbox") else None,
            )
            for item in detections
        ]
        logger.trace("OCR produced {count} fragments", count=len(results))
        return results

    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize large frames to improve throughput while tracking scale."""

        height, width = frame.shape[:2]
        if self._max_width and width > self._max_width and cv2 is not None:
            scale = self._max_width / float(width)
            new_size = (self._max_width, max(1, int(round(height * scale))))
            interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(frame, new_size, interpolation=interpolation)
            logger.trace("Resized frame from %sx%s to %sx%s", width, height, resized.shape[1], resized.shape[0])
            return resized, scale

        return frame, 1.0

    def _parse_outputs(
        self,
        outputs: Iterable,
        scale: float,
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Convert RapidOCR outputs into normalized tuples."""

        detections: List[Tuple[str, float, np.ndarray]] = []
        if not outputs:
            return detections

        for item in outputs:
            if not isinstance(item, Sequence) or len(item) < 3:
                continue

            text_raw, confidence_raw, bbox_raw = item[0], item[1], item[2]
            text = str(text_raw).strip()
            if not text:
                continue

            try:
                score = float(confidence_raw)
            except (TypeError, ValueError):
                continue

            if score < self._confidence_threshold:
                continue

            points = np.array(bbox_raw, dtype=float).reshape(-1, 2)
            if scale != 1.0:
                points = points / scale

            detections.append((text, score, points))

        return detections

    @staticmethod
    def _format_detection(text: str, score: float, points: np.ndarray) -> dict:
        bbox = points.reshape(-1).tolist()
        return {"text": text, "score": float(score), "bbox": [float(v) for v in bbox]}

    def _merge_line_detections(self, detections: List[Tuple[str, float, np.ndarray]]) -> List[dict]:
        if not detections:
            return []

        by_center = []
        heights: List[float] = []
        for text, score, points in detections:
            y_values = points[:, 1]
            center = float(np.mean(y_values))
            height = float(np.max(y_values) - np.min(y_values)) or 1.0
            heights.append(height)
            by_center.append((center, text, score, points))

        tolerance = max(10.0, median(heights) * 0.6 if heights else 10.0)
        by_center.sort(key=lambda item: item[0])

        merged: List[dict] = []
        current_texts: List[str] = []
        current_scores: List[float] = []
        current_points: List[np.ndarray] = []
        current_center: float | None = None

        def flush() -> None:
            nonlocal current_texts, current_scores, current_points, current_center
            if not current_texts:
                return
            all_points = np.vstack(current_points)
            min_x, min_y = np.min(all_points[:, 0]), np.min(all_points[:, 1])
            max_x, max_y = np.max(all_points[:, 0]), np.max(all_points[:, 1])
            bbox = [min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]
            merged.append(
                {
                    "text": " ".join(current_texts).strip(),
                    "score": float(np.mean(current_scores)),
                    "bbox": [float(v) for v in bbox],
                }
            )
            current_texts = []
            current_scores = []
            current_points = []
            current_center = None

        for center, text, score, points in by_center:
            if current_center is None or abs(center - current_center) > tolerance:
                flush()
                current_center = center

            current_texts.append(text)
            current_scores.append(score)
            current_points.append(points)
            current_center = (current_center * (len(current_scores) - 1) + center) / len(current_scores)

        flush()
        return merged


if __name__ == "__main__":  # pragma: no cover - manual usage example
    try:
        from .screen_watcher import ScreenWatcher

        watcher = ScreenWatcher()
        frame = watcher.get_latest()
        reader = OCRReader()
        results = reader.detect(frame, merge_lines=True)
        for entry in results[:5]:
            print(f"{entry['score']:.2f} - {entry['text']}")
    except Exception as exc:  # pragma: no cover - example should not crash callers
        print(f"OCR demonstration failed: {exc}")
