from __future__ import annotations

import numpy as np
import pytest

from companion.scene_classifier import SceneClassifier


def _battle_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (20, 20, 20)
    frame[120:200, 220:320, :] = (40, 40, 200)
    frame[160:260, 260:360, :] = (30, 30, 220)
    return frame


def _dialog_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (12, 12, 12)
    frame[280:360, :, :] = (90, 90, 90)
    return frame


def _map_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (160, 180, 200)
    return frame


def test_battle_scene_detection() -> None:
    classifier = SceneClassifier()
    frame = _battle_frame()
    lines = ["Vaporize [y=0.3]", "蒸发", "99999"]
    label, confidence = classifier.classify(frame, lines, motion_score=0.12)
    assert label == "battle"
    assert confidence >= 0.6


def test_dialog_scene_detection() -> None:
    classifier = SceneClassifier()
    frame = _dialog_frame()
    lines = [
        "凯亚：准备了吗？[y=0.80]",
        "按F 继续[y=0.82]",
        "再聊聊吧[y=0.84]",
    ]
    label, confidence = classifier.classify(frame, lines, motion_score=0.02)
    assert label == "dialog"
    assert confidence >= 0.6


def test_map_scene_detection() -> None:
    classifier = SceneClassifier()
    frame = _map_frame()
    lines = ["地图", "传送点"]
    label, confidence = classifier.classify(frame, lines, motion_score=0.01)
    assert label == "map"
    assert confidence >= 0.55


def test_unknown_when_no_match() -> None:
    classifier = SceneClassifier()
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    lines: list[str] = []
    label, confidence = classifier.classify(frame, lines, motion_score=0.04)
    assert label == "unknown"
    assert confidence == pytest.approx(0.5, abs=0.05)
