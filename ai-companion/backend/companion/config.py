"""Configuration model for the companion backend."""
from __future__ import annotations

from pydantic import BaseSettings, Field


class CompanionSettings(BaseSettings):
    """Runtime configuration for the backend services."""

    websocket_url: str = Field(
        "ws://127.0.0.1:17865",
        description="WebSocket endpoint exposed by the Electron overlay.",
    )
    screenshot_interval_ms: int = Field(1000, ge=100, description="Polling interval for screen capture.")
    tts_voice: str | None = Field(None, description="Optional TTS voice identifier.")
    overlay_display_ms: int = Field(4000, ge=100, description="Duration for overlay text visibility.")
    hotkey_toggle: str = Field("alt+`", description="Global hotkey for enabling or disabling the companion.")

    class Config:
        env_prefix = "COMPANION_"
        env_file = ".env"
        env_file_encoding = "utf-8"
