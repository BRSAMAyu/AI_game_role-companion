"""Text to speech helper built on pyttsx3 with speed presets."""
from __future__ import annotations

import concurrent.futures
import threading
from concurrent.futures import Future
from typing import List, Optional

import pyttsx3

from .utils.logging_setup import get_logger

logger = get_logger(__name__)

_SPEED_MODES = {
    "battle": 40,
    "dialog": 10,
    "menu": 0,
    "map": 0,
    "menu|map": 0,
}

_STOP_CHARS = set("。？！!?.,;；，…")


def init_tts(voice: Optional[str] = None) -> pyttsx3.Engine:
    """Create and configure a pyttsx3 engine."""

    engine = pyttsx3.init()
    if voice:
        try:
            engine.setProperty("voice", voice)
            logger.debug("TTS voice set to {voice}", voice=voice)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to set requested voice {voice}: {error}", voice=voice, error=str(exc))
    return engine


class TextToSpeech:
    """Encapsulates pyttsx3 usage for asynchronous playback."""

    def __init__(self, voice: Optional[str] = None) -> None:
        self._engine = init_tts(voice)
        self._voice = voice
        self._base_rate = self._engine.getProperty("rate")
        self._rate_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        logger.debug(
            "TextToSpeech initialized",
            voice=voice,
            base_rate=self._base_rate,
        )

    def init_tts(self, voice: Optional[str] = None) -> None:
        """Reinitialize the engine, matching the public init_tts helper."""

        with self._rate_lock:
            self._engine = init_tts(voice or self._voice)
            self._voice = voice or self._voice
            self._base_rate = self._engine.getProperty("rate")
            logger.debug("TTS engine reinitialized", voice=self._voice, base_rate=self._base_rate)

    def speak(self, text: str, mode: str = "dialog") -> Future:
        """Queue text for playback and return a future for completion."""

        normalized = (text or "").strip()
        if not normalized:
            logger.trace("Skipping TTS for empty text")
            future: Future[None] = Future()
            future.set_result(None)
            return future

        mode_key = mode.lower().strip() or "dialog"
        logger.debug("Queueing speech", length=len(normalized), mode=mode_key)
        return self._executor.submit(self._run_playback, normalized, mode_key)

    def _run_playback(self, text: str, mode: str) -> None:
        try:
            rate = self._base_rate + _SPEED_MODES.get(mode, 0)
            chunks = self._chunk_text(text)
            with self._rate_lock:
                self._engine.setProperty("rate", rate)
                for chunk in chunks:
                    self._engine.say(chunk)
                self._engine.runAndWait()
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error("TTS playback failed", error=str(exc), mode=mode)
        finally:
            with self._rate_lock:
                self._engine.setProperty("rate", self._base_rate)

    def _chunk_text(self, text: str) -> List[str]:
        trimmed = text.strip()
        if len(trimmed) <= 24:
            return [trimmed]

        chunks: List[str] = []
        current: List[str] = []
        for char in trimmed:
            current.append(char)
            if len(current) >= 24 and (char in _STOP_CHARS or len(current) >= 32):
                segment = "".join(current).strip()
                if segment:
                    chunks.append(segment)
                current = []
        if current:
            segment = "".join(current).strip()
            if segment:
                chunks.append(segment)
        return chunks

    # Edge-TTS migration placeholder:
    # def speak_with_edge(self, text: str, voice: str = "zh-CN-XiaoxiaoNeural") -> Awaitable[None]:
    #     """Example signature for adopting edge-tts async playback."""
    #     raise NotImplementedError
