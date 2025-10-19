from __future__ import annotations

import pytest

from companion.orchestrator import DialogueOrchestrator


def always_allow(_: str) -> bool:
    return True


def always_block(_: str) -> bool:
    return False


def test_battle_short_message() -> None:
    orchestrator = DialogueOrchestrator()
    orchestrator._random.seed(123)  # type: ignore[attr-defined]
    decision = orchestrator.decide(
        "battle",
        ["危险来袭", "护盾失效"],
        "zh",
        "urgent",
        always_allow,
    )
    assert decision.action == "speak"
    assert decision.metadata["scene"] == "battle"
    assert len(decision.text or "") <= 12


def test_dialog_no_spoiler() -> None:
    orchestrator = DialogueOrchestrator()
    orchestrator._random.seed(321)  # type: ignore[attr-defined]
    lines = [
        "派蒙：这边有点危险[y=0.82]",
        "按F 继续[y=0.84]",
        "说不定有惊喜[y=0.86]",
    ]
    decision = orchestrator.decide("dialog", lines, "zh", "calm", always_allow)
    assert decision.action == "speak"
    assert "F" not in decision.text  # continue prompt moved to末尾中文提示
    assert not any(char.isdigit() for char in decision.text or "")


def test_menu_suggestion_length() -> None:
    orchestrator = DialogueOrchestrator()
    orchestrator._random.seed(99)  # type: ignore[attr-defined]
    decision = orchestrator.decide("menu", ["背包"], "zh", "guide", always_allow)
    assert decision.action == "speak"
    assert 20 <= len(decision.text or "") <= 28


def test_cooldown_blocks_with_silence_tone() -> None:
    orchestrator = DialogueOrchestrator()
    decision = orchestrator.decide("battle", ["危险来袭"], "zh", "urgent", always_block)
    assert decision.action == "silence"
    assert decision.tone == "silence"
