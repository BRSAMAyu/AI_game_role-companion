"""Tests for the placeholder scene classifier."""
from companion.scene_classifier import SceneClassifier


def test_scene_classifier_dialogue_label():
    classifier = SceneClassifier()
    scene = classifier.classify("Is anyone there?")
    assert scene is not None
    assert scene.label == "dialogue"


def test_scene_classifier_generic_label():
    classifier = SceneClassifier()
    scene = classifier.classify("The door is closed.")
    assert scene is not None
    assert scene.label == "generic"
