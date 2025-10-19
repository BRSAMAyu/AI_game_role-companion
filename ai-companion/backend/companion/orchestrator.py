"""Dialogue orchestration tuned for Genshin Impact style interactions."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence

from pydantic import BaseModel, Field

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


class Decision(BaseModel):
    """Serializable decision returned by the dialogue orchestrator."""

    action: str = Field("silence", description="What the companion should do")
    text: Optional[str] = Field(None, description="Utterance prepared for speech or overlay")
    tone: str = Field("neutral", description="Tone hint used for voice or overlay styling")
    language: str = Field("zh", description="BCP-47 language tag of the utterance")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_dump_json(self, **kwargs: Any) -> str:  # pragma: no cover - compatibility shim
        """Compatibility helper matching pydantic v2 style APIs."""

        return self.json(**kwargs)


class DialogueOrchestrator:
    """Rule-based dialogue planner with light-weight state tracking."""

    _dedup_window: float = 12.0
    _map_cooldown: float = 30.0

    def __init__(self) -> None:
        self._random = random.Random()
        self._current_scene: Optional[str] = None
        self._battle_start: float = 0.0
        self._last_battle_output: float = 0.0
        self._next_battle_cheer: float = 0.0
        self._recent_texts: MutableMapping[str, float] = {}
        self._map_category_ts: Dict[str, float] = {"explore": 0.0, "teleport": 0.0, "supply": 0.0}

        self._battle_keywords = {
            "zh": [
                ("危险", "注意防御！"),
                ("护盾", "保持护盾！"),
                ("治疗", "抓紧回血！"),
                ("元素爆发", "抓紧开大！"),
            ],
            "en": [
                ("danger", "Stay safe!"),
                ("shield", "Hold shield!"),
                ("heal", "Heal now!"),
                ("burst", "Use burst!"),
            ],
            "ja": [
                ("危険", "気を付けて！"),
                ("シールド", "盾を維持！"),
                ("回復", "今すぐ回復！"),
                ("元素爆発", "今すぐ奥義！"),
            ],
        }
        self._battle_short_messages = {
            "zh": ["小心闪避！", "再坚持一下！", "别急稳住！", "集中注意！"],
            "en": ["Hang on!", "Stay sharp!", "You got it!", "Keep moving!"],
            "ja": ["踏ん張って！", "気を付けて！", "もう少し！", "落ち着いて！"],
        }
        self._battle_long_messages = {
            "zh": ["再坚持一下！", "换个角度攻！"],
            "en": ["Keep focus!", "Stay steady!"],
            "ja": ["あと少し！", "焦らないで！"],
        }
        self._map_templates = {
            "zh": {
                "explore": [
                    "沿北方山脊慢行，留意岩洞藏宝箱刷新点位哦",
                    "东侧遗迹绕行，寻找机关触发隐藏秘道入口哦",
                ],
                "teleport": [
                    "先解锁北面高台传送锚点，再规划补给路线吧",
                    "靠近风神像同步新的传送点，返回路途更方便",
                ],
                "supply": [
                    "顺路回蒙德补充食材与锻造矿石库存更安心些",
                    "路过营地记得补给恢复药剂和紧急补品备用哦",
                ],
            },
            "en": {
                "explore": [
                    "Scan north ridge for chests",
                    "Probe east ruins for levers",
                ],
                "teleport": [
                    "Unlock cliff waypoint ahead",
                    "Sync Anemo statue for travel",
                ],
                "supply": [
                    "Restock food and ore in town",
                    "Refill meds at nearby camp",
                ],
            },
            "ja": {
                "explore": [
                    "北の尾根を進んで洞窟の宝箱を探してみよう",
                    "東の遺跡を巡って仕掛け扉のヒントを探そう",
                ],
                "teleport": [
                    "北側の高台ワープを先に解放して移動を楽にしよう",
                    "風神像でワープ更新して戻り道を確保しよう",
                ],
                "supply": [
                    "モンドに寄って食材と鉱石の備蓄を補充しよう",
                    "野営地で回復薬と緊急物資も補給しておこう",
                ],
            },
        }
        self._dialog_attitudes = {
            "zh": {
                "calm": "心态平和",
                "comfort": "感觉踏实",
                "urgent": "情绪紧张",
                "curious": "有点好奇",
            },
            "en": {
                "calm": "feels composed",
                "comfort": "sounds reassuring",
                "urgent": "feels tense",
                "curious": "sounds curious",
            },
            "ja": {
                "calm": "落ち着いた気配",
                "comfort": "安心できそう",
                "urgent": "緊張している",
                "curious": "好奇心たっぷり",
            },
        }
        self._continue_tokens = {
            "zh": ["继续", "对话", "F 键"],
            "en": ["continue", "dialog"],
            "ja": ["続け", "対話"],
        }

    # ------------------------------------------------------------------
    # LLM fallback placeholder
    def llm_generate(self, observation: Dict[str, Any]) -> Decision:  # pragma: no cover - placeholder
        """Reserved hook for future LLM integration."""

        logger.debug("LLM generate placeholder invoked with observation keys: {keys}", keys=list(observation.keys()))
        return Decision(action="silence", tone="neutral", language="zh", metadata={"source": "llm_placeholder"})

    # ------------------------------------------------------------------
    def decide(
        self,
        scene: str,
        lines: Sequence[str],
        lang: str,
        emotion_hint: Optional[str],
        cooldown_ok: Callable[[str], bool],
    ) -> Decision:
        """Choose the next action for the companion."""

        now = time.monotonic()
        scene_key = (scene or "unknown").strip().lower()
        language = self._normalize_lang(lang)
        cleaned_lines = [line.strip() for line in lines if line and line.strip()]

        if scene_key != self._current_scene:
            self._current_scene = scene_key
            logger.debug("Scene switched to {scene}", scene=scene_key)
            if scene_key == "battle":
                self._battle_start = now
                self._next_battle_cheer = now + 8.0
            else:
                self._battle_start = 0.0
                self._next_battle_cheer = 0.0

        if not cleaned_lines:
            return self._silence(scene_key, language, reason="empty_lines")

        if scene_key == "battle":
            return self._handle_battle(cleaned_lines, language, emotion_hint, cooldown_ok, now)
        if scene_key == "dialog":
            return self._handle_dialog(cleaned_lines, language, emotion_hint, cooldown_ok, now)
        if scene_key in {"map", "menu"}:
            return self._handle_map(language, emotion_hint, cooldown_ok, now)
        if scene_key in {"loading", "unknown"}:
            return self._silence(scene_key, language, reason="inactive_scene")

        return self._silence(scene_key, language, reason="unhandled_scene")

    # ------------------------------------------------------------------
    def _handle_battle(
        self,
        lines: Sequence[str],
        lang: str,
        emotion_hint: Optional[str],
        cooldown_ok: Callable[[str], bool],
        now: float,
    ) -> Decision:
        if not cooldown_ok("battle"):
            return self._silence("battle", lang, reason="battle_cooldown")

        tone = (emotion_hint or "comfort").lower()
        if not cooldown_ok(tone):
            return self._silence("battle", lang, reason="tone_cooldown")

        if now - self._last_battle_output < 5.0:
            return self._silence("battle", lang, reason="battle_throttle")

        message: Optional[str] = None
        lowered_lines = self._lowered(lines)
        keywords = self._battle_keywords.get(lang, self._battle_keywords["zh"])
        for keyword, template in keywords:
            if any(keyword in line for line in lowered_lines):
                message = template
                break

        if message is None and self._battle_start and now - self._battle_start >= 15.0:
            if now >= self._next_battle_cheer:
                pool = self._battle_long_messages.get(lang, self._battle_long_messages["zh"])
                message = self._pick_unique(pool, now)
                self._next_battle_cheer = now + self._random.uniform(8.0, 12.0)
            else:
                return self._silence("battle", lang, reason="await_cheer_window")

        if message is None:
            pool = self._battle_short_messages.get(lang, self._battle_short_messages["zh"])
            message = self._pick_unique(pool, now)

        if message is None:
            return self._silence("battle", lang, reason="no_unique_battle_text")

        if len(message) > 12:
            message = message[:12]

        if not self._register_text(message, now):
            return self._silence("battle", lang, reason="battle_duplicate")

        self._last_battle_output = now
        metadata = {"scene": "battle"}
        return Decision(action="speak", text=message, tone=tone, language=lang, metadata=metadata)

    def _handle_dialog(
        self,
        lines: Sequence[str],
        lang: str,
        emotion_hint: Optional[str],
        cooldown_ok: Callable[[str], bool],
        now: float,
    ) -> Decision:
        tone = (emotion_hint or "calm").lower()
        if not cooldown_ok(tone):
            return self._silence("dialog", lang, reason="tone_cooldown")

        summary = self._build_dialog_summary(lines, lang)
        keyword = self._extract_keyword(summary, lang)
        attitude = self._attitude_phrase(lang, tone)
        highlight = self._compose_highlight(keyword, attitude, lang)
        if lang == "zh":
            core = summary.rstrip("。！？!?")
            text = f"{core}，{highlight}"
            if not text.endswith(("。", "！", "？")):
                text += "。"
        elif lang == "ja":
            core = summary.rstrip("。！？!?")
            text = f"{core}、{highlight}"
            if not text.endswith(("。", "！", "？")):
                text += "。"
        else:
            core = summary.rstrip(".!?")
            text = f"{core}. {highlight}".strip()

        if self._needs_continue_prompt(lines, lang):
            prompt = self._continue_prompt(lang)
            text = f"{text} {prompt}" if lang != "zh" else f"{text}{prompt}"

        if not self._register_text(text, now):
            return self._silence("dialog", lang, reason="dialog_duplicate")

        metadata = {"scene": "dialog", "keyword": keyword}
        return Decision(action="speak", text=text, tone=tone, language=lang, metadata=metadata)

    def _handle_map(
        self,
        lang: str,
        emotion_hint: Optional[str],
        cooldown_ok: Callable[[str], bool],
        now: float,
    ) -> Decision:
        tone = (emotion_hint or "guide").lower()
        if not cooldown_ok(tone):
            return self._silence("map", lang, reason="tone_cooldown")

        suggestions = self._map_templates.get(lang, self._map_templates["zh"])
        categories = ["explore", "teleport", "supply"]
        eligible = [cat for cat in categories if now - self._map_category_ts.get(cat, 0.0) >= self._map_cooldown]
        if not eligible:
            eligible = sorted(categories, key=lambda cat: self._map_category_ts.get(cat, 0.0))

        text: Optional[str] = None
        chosen_category: Optional[str] = None
        for category in eligible:
            pool = suggestions.get(category, [])
            message = self._pick_unique(pool, now)
            if message:
                text = message
                chosen_category = category
                break

        if text is None:
            return self._silence("map", lang, reason="map_no_option")

        length = len(text)
        if length < 20 or length > 28:
            logger.debug("Map suggestion length adjusted", length=length, language=lang)
            text = self._pad_map_text(text, lang)

        self._map_category_ts[chosen_category or "explore"] = now

        if not self._register_text(text, now):
            return self._silence("map", lang, reason="map_duplicate")

        metadata = {"scene": "map", "category": chosen_category}
        return Decision(action="speak", text=text, tone=tone, language=lang, metadata=metadata)

    # ------------------------------------------------------------------
    def _pick_unique(self, options: Iterable[str], now: float) -> Optional[str]:
        candidates = list(options)
        self._random.shuffle(candidates)
        for candidate in candidates:
            if candidate and self._is_unique(candidate, now):
                return candidate
        return None

    def _is_unique(self, text: str, now: float) -> bool:
        self._purge_expired(now)
        last = self._recent_texts.get(text)
        return last is None or now - last >= self._dedup_window

    def _register_text(self, text: str, now: float) -> bool:
        if not self._is_unique(text, now):
            return False
        self._recent_texts[text] = now
        return True

    def _purge_expired(self, now: float) -> None:
        for key in list(self._recent_texts.keys()):
            if now - self._recent_texts[key] >= self._dedup_window:
                del self._recent_texts[key]

    def _normalize_lang(self, lang: str) -> str:
        lowered = (lang or "zh").lower()
        if lowered.startswith("en"):
            return "en"
        if lowered.startswith("ja") or lowered.startswith("jp"):
            return "ja"
        if lowered.startswith("zh"):
            return "zh"
        return "zh"

    def _build_dialog_summary(self, lines: Sequence[str], lang: str) -> str:
        focus = lines[-1].strip()
        if len(focus) > 18:
            focus = focus[:18] + "…"
        return focus

    def _extract_keyword(self, text: str, lang: str) -> str:
        cleaned = text.replace("…", "").strip().rstrip("。！？!?")
        if not cleaned:
            return ""
        if lang == "en":
            tokens = [word.strip(".,!?") for word in cleaned.split() if word]
            long_tokens = [word for word in tokens if len(word) > 3]
            chosen = long_tokens[-1] if long_tokens else (tokens[-1] if tokens else cleaned)
            return chosen
        if lang == "ja":
            return cleaned[-4:]
        return cleaned[-3:]

    def _attitude_phrase(self, lang: str, tone: str) -> str:
        table = self._dialog_attitudes.get(lang) or self._dialog_attitudes["zh"]
        return table.get(tone, table.get("calm", "心态平和"))

    def _compose_highlight(self, keyword: str, attitude: str, lang: str) -> str:
        if lang == "zh":
            return f"{keyword}·{attitude}" if keyword else attitude
        if lang == "ja":
            return f"{keyword}、{attitude}" if keyword else attitude
        return f"Key: {keyword}, {attitude}" if keyword else attitude

    def _needs_continue_prompt(self, lines: Sequence[str], lang: str) -> bool:
        tokens = self._continue_tokens.get(lang, [])
        lowered_lines = self._lowered(lines)
        return any(token.lower() in line for token in tokens for line in lowered_lines)

    def _continue_prompt(self, lang: str) -> str:
        if lang == "en":
            return "Continue?"
        if lang == "ja":
            return "続ける？"
        return "要继续吗？"

    def _pad_map_text(self, text: str, lang: str) -> str:
        target = min(28, max(20, len(text)))
        if lang == "en":
            filler = " for now"
        elif lang == "ja":
            filler = "、忘れないで"
        else:
            filler = "，别忘了"
        while len(text) < target:
            text += filler
        return text[:target]

    def _lowered(self, lines: Sequence[str]) -> List[str]:
        return [line.lower() for line in lines]

    def _silence(self, scene: str, lang: str, reason: str) -> Decision:
        logger.debug("Silence decision", scene=scene, language=lang, reason=reason)
        return Decision(action="silence", text=None, tone="neutral", language=lang, metadata={"scene": scene, "reason": reason})


# Backward-compatible orchestrator for other subsystems -----------------------
class BackendOrchestrator:  # pragma: no cover - legacy shim
    """Legacy placeholder retained for existing imports."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "BackendOrchestrator has been superseded by DialogueOrchestrator in this build."
        )
