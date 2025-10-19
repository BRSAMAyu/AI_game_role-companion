"""Main orchestration pipeline placeholder."""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

from tenacity import RetryError, retry, stop_after_attempt, wait_fixed
from websocket import WebSocketApp

from .config import CompanionSettings
from .ocr_reader import OCRReader
from .screen_watcher import ScreenWatcher
from .scene_classifier import SceneClassifier
from .text_aggregator import TextAggregator
from .tts import TextToSpeech
from .utils.logging_setup import get_logger

logger = get_logger(__name__)


class BackendOrchestrator:
    """Coordinates screen capture, OCR, scene classification, and output."""

    def __init__(self, settings: CompanionSettings) -> None:
        self.settings = settings
        self.screen_watcher = ScreenWatcher()
        self.ocr_reader = OCRReader()
        self.aggregator = TextAggregator()
        self.scene_classifier = SceneClassifier()
        self.tts = TextToSpeech(voice=settings.tts_voice)
        self._ws_app: Optional[WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        logger.debug("BackendOrchestrator initialized")

    def start(self) -> None:
        self._connect_ws()
        logger.info("Backend orchestrator started")

    def stop(self) -> None:
        if self._ws_app:
            self._ws_app.close()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=1)
        logger.info("Backend orchestrator stopped")

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
    def _connect_ws(self) -> None:
        logger.info("Connecting to overlay WebSocket at {url}", url=self.settings.websocket_url)
        self._ws_app = WebSocketApp(self.settings.websocket_url)
        self._ws_thread = threading.Thread(target=self._ws_app.run_forever, daemon=True)
        self._ws_thread.start()
        time.sleep(0.5)
        if not self._ws_thread.is_alive():
            raise ConnectionError("Failed to start WebSocket thread")
        logger.success("Connected to overlay WebSocket")

    def send_overlay_text(self, text: str, duration_ms: Optional[int] = None) -> None:
        if not self._ws_app:
            logger.warning("Cannot send overlay text, WebSocket not connected")
            return
        payload = json.dumps({
            "type": "showText",
            "text": text,
            "ms": duration_ms or self.settings.overlay_display_ms,
        })
        try:
            self._ws_app.send(payload)
            logger.debug("Sent overlay payload of length {length}", length=len(payload))
        except RetryError as exc:  # pragma: no cover - placeholder for tenacity integration
            logger.error("Failed to send overlay payload: {error}", error=str(exc))
        except Exception as exc:  # pragma: no cover - general safety net
            logger.exception("Unexpected error sending overlay payload: {error}", error=str(exc))
