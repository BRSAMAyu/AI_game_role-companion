"""Text to speech helper built on pyttsx3."""
from __future__ import annotations

from typing import Optional

import pyttsx3

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


class TextToSpeech:
    """Encapsulates pyttsx3 usage for synchronous playback."""

    def __init__(self, voice: Optional[str] = None) -> None:
        self.engine = pyttsx3.init()
        if voice:
            try:
                self.engine.setProperty("voice", voice)
                logger.debug("TTS voice set to {voice}", voice=voice)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to set requested voice {voice}: {error}", voice=voice, error=str(exc))
        logger.debug("TextToSpeech initialized")

    def say(self, text: str) -> None:
        if not text:
            logger.trace("Skipping TTS for empty text")
            return
        logger.debug("Speaking text of length {length}", length=len(text))
        self.engine.say(text)
        self.engine.runAndWait()
