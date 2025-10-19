from __future__ import annotations

import pytest

from companion.hotkeys import HotkeyManager
from companion.utils.state import Cooldown, RecentBuffer


def test_hotkey_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    registered = []

    def fake_add_hotkey(hotkey: str, callback):  # pragma: no cover - mimic API
        registered.append((hotkey, callback))

    monkeypatch.setattr("companion.hotkeys.keyboard.add_hotkey", fake_add_hotkey)

    manager = HotkeyManager()
    toggled: list[bool] = []

    manager.register_toggle("alt+`", lambda active: toggled.append(active))
    manager.register_hotkey("ctrl+shift+s", lambda: toggled.append(True))

    assert len(registered) == 2
    assert registered[0][0] == "alt+`"

    registered[0][1]()
    assert toggled[-1] is False
    registered[0][1]()
    assert toggled[-1] is True


def test_recent_buffer_and_cooldown_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = RecentBuffer(window_seconds=30.0)
    buffer.add("战斗准备")
    assert buffer.seen("战斗准备")

    cooldown = Cooldown()
    assert cooldown.allow("battle", 1.0)
    cooldown.block_all(5.0)
    assert not cooldown.allow("dialog", 1.0)

    cooldown.block_all(0.1)
    assert not cooldown.allow("battle", 0.5)
