from __future__ import annotations

import concurrent.futures

import pytest

from companion import tts as tts_module
from companion.tts import TextToSpeech


class FakeEngine:
    def __init__(self) -> None:
        self._rate = 200
        self.rate_history: list[int] = []
        self.playbacks: list[list[str]] = []
        self._current: list[str] = []
        self._in_run = False

    def getProperty(self, name: str):  # pragma: no cover - mirror API
        if name == "rate":
            return self._rate
        raise AttributeError(name)

    def setProperty(self, name: str, value) -> None:  # pragma: no cover - mirror API
        if name == "rate":
            self._rate = int(value)
            self.rate_history.append(int(value))
        else:
            raise AttributeError(name)

    def say(self, text: str) -> None:
        if self._in_run:
            raise RuntimeError("Concurrent say detected")
        self._current.append(text)

    def runAndWait(self) -> None:
        if self._in_run:
            raise RuntimeError("Nested runAndWait")
        self._in_run = True
        self.playbacks.append(list(self._current))
        self._current.clear()
        self._in_run = False


def test_tts_chunking_and_rates(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = FakeEngine()
    monkeypatch.setattr(tts_module.pyttsx3, "init", lambda: engine)

    speech = TextToSpeech()

    long_sentence = "元素能量充沛的时候记得及时释放元素爆发，不要错过最佳窗口，而且队友状态也要及时关注。"
    future = speech.speak(long_sentence, mode="menu")
    assert isinstance(future, concurrent.futures.Future)
    future.result(timeout=1.0)

    assert len(engine.playbacks[0]) >= 2

    future_fast = speech.speak("快闪避!", mode="battle")
    future_slow = speech.speak("打开菜单检查物资", mode="menu")
    concurrent.futures.wait([future_fast, future_slow], timeout=1.0)

    assert len(engine.playbacks) == 3

    # Rate history should show higher speed for battle than menu segments.
    battle_rate = max(engine.rate_history)
    menu_rate = min(engine.rate_history)
    assert battle_rate > menu_rate
