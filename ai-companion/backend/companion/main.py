"""Entrypoint orchestrating capture, classification, and playback."""
from __future__ import annotations

import contextlib
import signal
import sys
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import orjson
from websocket import WebSocketConnectionClosedException, create_connection

from .config import CompanionSettings
from .hotkeys import HotkeyManager
from .ocr_reader import OCRReader, OCRResult
from .orchestrator import DialogueOrchestrator
from .scene_classifier import SceneClassifier
from .screen_watcher import ScreenWatcher
from .tts import TextToSpeech
from .utils.logging_setup import configure_logging, get_logger
from .utils.state import Cooldown, RecentBuffer

logger = get_logger(__name__)


class OverlayClient:
    """Thin websocket client for communicating with the overlay."""

    def __init__(self, url: str, default_ms: int) -> None:
        self._url = url
        self._default_ms = max(100, int(default_ms))
        self._socket = None
        self._lock = threading.Lock()
        self._last_attempt = 0.0

    def send_text(self, text: str, ms: Optional[int], style: str) -> bool:
        payload = {
            "type": "showText",
            "text": text,
            "ms": int(ms) if ms else self._default_ms,
            "style": style,
        }
        message = orjson.dumps(payload)
        if not self._ensure_connection():
            return False

        try:
            assert self._socket is not None  # for type checkers
            self._socket.send(message)
            return True
        except WebSocketConnectionClosedException:
            logger.warning("Overlay connection closed; will attempt reconnect", url=self._url)
            self._socket = None
        except OSError as exc:
            logger.warning("Overlay send failed", error=str(exc))
            self._socket = None
        except Exception as exc:  # pragma: no cover - safety net for websocket stack
            logger.exception("Unexpected overlay send failure", error=str(exc))
            self._socket = None
        return False

    def close(self) -> None:
        with self._lock:
            if self._socket is not None:
                with contextlib.suppress(Exception):
                    self._socket.close()
                self._socket = None

    def _ensure_connection(self) -> bool:
        with self._lock:
            if self._socket is not None:
                return True

            now = time.monotonic()
            if now - self._last_attempt < 1.5:
                return False
            self._last_attempt = now
            try:
                self._socket = create_connection(self._url, timeout=3)
                logger.info("Connected to overlay", url=self._url)
                return True
            except Exception as exc:
                logger.debug("Failed to connect overlay", url=self._url, error=str(exc))
                self._socket = None
                return False


class CompanionRuntime:
    """High-level runtime tying capture, classification and responses together."""

    def __init__(self, settings: CompanionSettings) -> None:
        self._settings = settings
        self._watcher = ScreenWatcher()
        self._classifier = SceneClassifier()
        self._orchestrator = DialogueOrchestrator()
        self._tts = TextToSpeech(settings.tts_voice)
        self._overlay = OverlayClient(settings.websocket_url, settings.overlay_display_ms)
        self._cooldown = Cooldown()
        self._recent = RecentBuffer(window_seconds=30.0)
        self._active = True
        self._silence_until = 0.0
        self._silence_logged = False
        self._language = "zh"
        self._previous_gray: Optional[np.ndarray] = None
        self._tone_cooldowns: Dict[str, float] = {"battle": 2.0, "dialog": 4.5, "menu": 8.0}
        self._cooldown_policy: Dict[str, float] = {
            "battle": 2.0,
            "dialog": 4.5,
            "map": 6.0,
            "menu": 6.0,
            "calm": 3.5,
            "comfort": 3.5,
            "curious": 3.5,
            "urgent": 4.5,
            "guide": 6.0,
            "default": 3.0,
        }

        try:
            self._ocr = OCRReader()
        except RuntimeError as exc:
            logger.warning("OCR disabled: {error}", error=str(exc))
            self._ocr = None

    # ------------------------------------------------------------------
    # Hotkey integration
    def set_active(self, active: bool) -> None:
        self._active = active
        state = "active" if active else "paused"
        logger.info("Companion state updated", state=state)

    def trigger_emergency_silence(self, seconds: float = 5.0) -> None:
        duration = max(0.0, seconds)
        if duration <= 0:
            return
        self._cooldown.block_all(duration)
        until = time.monotonic() + duration
        self._silence_until = max(self._silence_until, until)
        self._silence_logged = True
        logger.warning("Emergency silence engaged", duration=duration)

    # ------------------------------------------------------------------
    # Main loop
    def run(self, stop_event: threading.Event) -> None:
        interval = max(0.05, self._settings.screenshot_interval_ms / 1000.0)
        logger.info("Runtime loop started", interval=interval)
        while not stop_event.is_set():
            start = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Runtime tick failed", error=str(exc))
            elapsed = time.monotonic() - start
            wait_time = max(0.0, interval - elapsed)
            stop_event.wait(wait_time)

    def shutdown(self) -> None:
        self._overlay.close()
        with contextlib.suppress(Exception):
            self._tts._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    def _tick(self) -> None:
        frame = self._capture_frame()
        if frame is None:
            return

        ocr_lines = self._run_ocr(frame)
        motion_score = self._compute_motion(frame)
        label, confidence = self._classifier.classify(frame, ocr_lines, motion_score)
        scene = self._apply_confidence_gate(label, confidence)
        features = self._collect_features(frame, ocr_lines, motion_score)

        logger.info(
            "Frame metrics",
            scene=scene,
            conf=round(float(confidence), 3),
            motion=round(float(motion_score), 4),
            red_ratio=round(float(features.get("red_ratio", 0.0)), 4),
            sobel_lowtex_ratio=round(float(features.get("low_texture_ratio", 0.0)), 4),
            lines_count=len(ocr_lines),
        )

        if not self._active:
            return

        if self._is_silenced():
            logger.debug("Output suppressed due to active silence window")
            return

        decision = self._orchestrator.decide(scene, ocr_lines, self._language, None, self._cooldown_ok)
        if decision.action != "speak" or not decision.text:
            return

        text = decision.text.strip()
        if not text:
            return

        if self._recent.seen(text):
            logger.info("Repeated text suppressed", sample=text[:32])
            return
        self._recent.add(text)

        style, tts_mode = self._scene_modes(scene)
        if not self._cooldown.allow(tts_mode, self._tone_cooldowns.get(tts_mode, 3.0)):
            logger.debug("Tone on cooldown", tone=tts_mode)
            return

        display_ms = self._display_duration(style)
        if not self._overlay.send_text(text, display_ms, style):
            logger.debug("Overlay unavailable; skipping display")

        self._tts.speak(text, mode=tts_mode)

    # ------------------------------------------------------------------
    # Helpers
    def _capture_frame(self) -> Optional[np.ndarray]:
        try:
            return self._watcher.get_latest()
        except Exception as exc:
            logger.error("Screen capture failed", error=str(exc))
        return None

    def _run_ocr(self, frame: np.ndarray) -> List[str]:
        if self._ocr is None:
            return []
        try:
            results: Iterable[OCRResult] = self._ocr.read(frame)
        except Exception as exc:
            logger.exception("OCR inference failed", error=str(exc))
            return []
        lines = [result.text.strip() for result in results if result.text and result.text.strip()]
        return lines

    def _compute_motion(self, frame: np.ndarray) -> float:
        gray = self._to_gray(frame)
        if gray is None:
            return 0.0
        if self._previous_gray is None or gray.shape != self._previous_gray.shape:
            self._previous_gray = gray
            return 0.0
        diff = np.abs(gray - self._previous_gray)
        motion = float(np.mean(diff > 25.0))
        self._previous_gray = gray
        return motion

    def _to_gray(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 2:
            gray = array
        elif array.ndim == 3 and array.shape[2] >= 3:
            # MSS captures in BGRA/BGR order; use first three channels.
            b, g, r = array[..., 0], array[..., 1], array[..., 2]
            gray = 0.114 * b + 0.587 * g + 0.299 * r
        else:
            gray = array.reshape(array.shape[0], -1)
        return gray

    def _apply_confidence_gate(self, label: str, confidence: float) -> str:
        thresholds = {"battle": 0.6, "dialog": 0.6, "map": 0.55, "menu": 0.55, "loading": 0.7}
        normalized = (label or "unknown").lower()
        threshold = thresholds.get(normalized)
        if threshold is not None and confidence < threshold:
            return "unknown"
        return normalized

    def _collect_features(
        self, frame: np.ndarray, lines: Iterable[str], motion_score: float
    ) -> Dict[str, float]:
        try:
            return self._classifier._compute_features(frame, list(lines), motion_score)  # type: ignore[attr-defined]
        except Exception:
            return {}

    def _cooldown_ok(self, key: str) -> bool:
        duration = self._cooldown_policy.get((key or "default").lower(), self._cooldown_policy["default"])
        return self._cooldown.allow(key, duration)

    def _scene_modes(self, scene: str) -> Tuple[str, str]:
        mapping = {
            "battle": ("battle", "battle"),
            "dialog": ("default", "dialog"),
            "map": ("default", "menu"),
            "menu": ("default", "menu"),
        }
        return mapping.get(scene, ("default", "dialog"))

    def _display_duration(self, style: str) -> int:
        if style == "battle":
            base = self._settings.overlay_display_ms
            return max(1200, min(2600, int(base * 0.65)))
        return self._settings.overlay_display_ms

    def _is_silenced(self) -> bool:
        if self._silence_until <= 0:
            return False
        now = time.monotonic()
        if now < self._silence_until:
            return True
        if self._silence_logged:
            logger.info("Emergency silence window expired")
            self._silence_logged = False
        self._silence_until = 0.0
        return False


def main() -> int:
    settings = CompanionSettings()
    configure_logging()
    runtime = CompanionRuntime(settings)
    hotkeys = HotkeyManager()
    stop_event = threading.Event()

    def on_toggle(active: bool) -> None:
        runtime.set_active(active)

    def on_emergency() -> None:
        runtime.trigger_emergency_silence(5.0)

    def signal_handler(_sig, _frame) -> None:
        logger.info("Shutdown requested")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    hotkeys.register_toggle(settings.hotkey_toggle, on_toggle)
    hotkeys.register_hotkey("alt+backspace", on_emergency)

    logger.info("Backend running. Press Ctrl+C to exit.")
    try:
        runtime.run(stop_event)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received; exiting runtime loop")
    finally:
        stop_event.set()
        runtime.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

