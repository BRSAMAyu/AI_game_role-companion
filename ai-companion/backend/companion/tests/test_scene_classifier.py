"""Tests for the Genshin Impact oriented scene classifier."""
from pathlib import Path
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from companion.scene_classifier import SceneClassifier


def _make_frame(color: tuple[int, int, int], size: int = 200) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def test_scene_classifier_battle_label():
    classifier = SceneClassifier()
    frame = _make_frame((0, 0, 255))  # strong red tint
    ocr_lines = ["[y=0.20]1234", "[y=0.70]Vaporize"]
    label, confidence = classifier.classify(frame, ocr_lines, motion_score=0.2)

    assert label == "battle"
    assert 0.6 <= confidence <= 0.95


def test_scene_classifier_dialog_label():
    classifier = SceneClassifier()
    frame = _make_frame((120, 120, 120))
    ocr_lines = [
        "[y=0.65]按 F 继续",
        "[y=0.72]对话",
        "[y=0.80]更多内容",
    ]
    label, confidence = classifier.classify(frame, ocr_lines, motion_score=0.01)

    assert label == "dialog"
    assert 0.5 <= confidence <= 0.9


def test_scene_classifier_map_label():
    classifier = SceneClassifier()
    frame = _make_frame((64, 128, 128))
    ocr_lines = ["Region", "Waypoint"]
    label, confidence = classifier.classify(frame, ocr_lines, motion_score=0.005)

    assert label == "map"
    assert 0.5 <= confidence <= 0.85

