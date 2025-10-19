"""Shared pytest fixtures for backend tests."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def fake_clock() -> Iterator["FakeClock"]:
    class FakeClock:
        def __init__(self) -> None:
            self.value = 0.0

        def time(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    clock = FakeClock()
    yield clock
