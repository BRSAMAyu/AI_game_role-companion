"""Runtime state helpers for throttling and deduplication."""
from __future__ import annotations

import time
from typing import Dict

__all__ = ["Cooldown", "RecentBuffer"]


class Cooldown:
    """Track per-key cooldown windows with optional global freezes."""

    def __init__(self) -> None:
        self._timestamps: Dict[str, float] = {}
        self._global_until: float = 0.0

    def allow(self, key: str, seconds: float) -> bool:
        """Return ``True`` if the key is outside of its cooldown window."""

        normalized = (key or "default").lower()
        window = max(0.0, float(seconds))
        now = time.monotonic()
        if now < self._global_until:
            return False

        last = self._timestamps.get(normalized)
        if last is None or now - last >= window:
            self._timestamps[normalized] = now
            return True

        return False

    def block_all(self, seconds: float) -> None:
        """Prevent *any* key from passing :meth:`allow` for ``seconds`` seconds."""

        if seconds <= 0:
            return
        self._global_until = max(self._global_until, time.monotonic() + seconds)

    def clear(self, key: str) -> None:
        """Remove an individual key from the cooldown table."""

        self._timestamps.pop((key or "default").lower(), None)

    def reset(self) -> None:
        """Reset all cooldown information."""

        self._timestamps.clear()
        self._global_until = 0.0


class RecentBuffer:
    """Remember recently emitted strings within a sliding time window."""

    def __init__(self, window_seconds: float = 30.0) -> None:
        self._window = max(0.0, float(window_seconds))
        self._entries: Dict[str, float] = {}

    def add(self, text: str) -> None:
        """Register text as recently seen."""

        normalized = self._normalize(text)
        if not normalized:
            return
        now = time.monotonic()
        self._purge(now)
        self._entries[normalized] = now

    def seen(self, text: str) -> bool:
        """Return ``True`` if the text appeared within the buffer window."""

        normalized = self._normalize(text)
        if not normalized:
            return False
        now = time.monotonic()
        self._purge(now)
        timestamp = self._entries.get(normalized)
        return timestamp is not None and now - timestamp <= self._window

    def _purge(self, now: float) -> None:
        if not self._entries:
            return
        expiry = now - self._window
        for key, ts in list(self._entries.items()):
            if ts < expiry:
                self._entries.pop(key, None)

    @staticmethod
    def _normalize(text: str) -> str:
        return (text or "").strip()

