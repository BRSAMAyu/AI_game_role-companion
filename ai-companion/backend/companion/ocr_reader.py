"""OCR interface built on top of RapidOCR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from .utils.logging_setup import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - optional dependency import guard
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - fallback if dependency missing at import time
    RapidOCR = None  # type: ignore[assignment]
    logger.warning("RapidOCR could not be imported; OCR functionality will be unavailable until dependencies are installed.")


@dataclass
class OCRResult:
    """Represents a detected text fragment."""

    text: str
    confidence: float


class OCRReader:
    """Wrapper around RapidOCR to simplify usage."""

    def __init__(self) -> None:
        if RapidOCR is None:
            raise RuntimeError("RapidOCR dependency is not available. Install rapidocr-onnxruntime to enable OCR.")
        self._engine = RapidOCR()
        logger.debug("RapidOCR engine initialized")

    def read(self, image: np.ndarray) -> List[OCRResult]:
        """Run OCR over the provided image."""
        outputs: Iterable = self._engine(image)
        results: List[OCRResult] = []
        for text, confidence, _bbox in outputs:
            results.append(OCRResult(text=text, confidence=float(confidence)))
        logger.trace("OCR produced {count} fragments", count=len(results))
        return results
