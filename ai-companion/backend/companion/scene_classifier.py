"""Placeholder scene classification using simple heuristics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SceneInfo:
    """Minimal representation of a detected scene."""

    label: str
    confidence: float


class SceneClassifier:
    """Classifies basic scene contexts from aggregated text."""

    def classify(self, text: str) -> Optional[SceneInfo]:
        if not text:
            logger.trace("Scene classification skipped due to empty text")
            return None
        label = "dialogue" if "?" in text else "generic"
        confidence = 0.5 if label == "generic" else 0.7
        logger.debug("Classified scene as {label} with confidence {confidence}", label=label, confidence=confidence)
        return SceneInfo(label=label, confidence=confidence)
