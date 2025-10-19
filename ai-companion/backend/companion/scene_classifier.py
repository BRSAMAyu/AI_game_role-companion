"""Scene classification heuristics tailored for Genshin Impact."""
from __future__ import annotations

import math
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Utility structures


class RunningStats:
    """Online estimator for mean and standard deviation."""

    __slots__ = ("_count", "_mean", "_m2")

    def __init__(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count else 0.0

    @property
    def std(self) -> float:
        if self._count < 2:
            return 0.0
        variance = self._m2 / (self._count - 1)
        return math.sqrt(max(variance, 0.0))


@dataclass
class SceneDecision:
    label: str
    confidence: float


# ---------------------------------------------------------------------------
# Main classifier


class SceneClassifier:
    """Heuristic classifier covering common Genshin Impact scenes."""

    _BATTLE_WORDS = re.compile(
        r"(Vaporize|Melt|Overloaded|Superconduct|Swirl|Crystallize|"
        r"蒸发|融化|超载|超导|扩散|结晶)",
        re.IGNORECASE,
    )
    _INTERACT_WORDS = re.compile(
        r"\b[FＦＥ]\b|按\s*F|\bTalk\b|对话|调查|继续",
        re.IGNORECASE,
    )
    _MENU_WORDS = re.compile(
        r"背包|角色|任务|背包\(B\)|角色\(C\)",
        re.IGNORECASE,
    )
    _BIG_NUMBER = re.compile(r"(?:(?<!\d)\d{2,}(?!\d)|\d{1,3}(?:,\d{3})+)")
    _Y_COORD = re.compile(r"\[y\s*=\s*(0?\.\d+|1\.0)\]", re.IGNORECASE)

    def __init__(self) -> None:
        self._history: Deque[Tuple[float, str, float]] = deque()
        self._history_window = 1.0
        self._baseline_start = time.time()
        self._baseline_duration = 10.0
        self._baseline_stats: Dict[str, RunningStats] = {
            "red_ratio": RunningStats(),
            "motion": RunningStats(),
            "low_texture": RunningStats(),
        }
        self._baseline_ready = False
        self._thresholds: Dict[str, float] = {
            "battle_motion": 0.08,
            "battle_red": 0.05,
            "dialog_motion": 0.05,
            "map_motion": 0.03,
            "menu_motion": 0.03,
            "loading_motion": 0.01,
            "map_low_texture": 0.55,
            "menu_low_texture": 0.45,
        }
        self._adaptive_thresholds = dict(self._thresholds)
        self._last_brightness_var: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API

    def classify(
        self, frame: np.ndarray, ocr_lines: Iterable[str], motion_score: float
    ) -> Tuple[str, float]:
        """Classify the scene from a frame, OCR lines and motion metric."""

        timestamp = time.time()
        ocr_lines = [line for line in ocr_lines if line]
        features = self._compute_features(frame, ocr_lines, motion_score)

        self._update_baseline(features, timestamp)
        decision = self._apply_rules(features, ocr_lines, motion_score)
        self._update_history(decision.label, decision.confidence, timestamp)

        smoothed_label, smoothed_confidence = self._smooth_output(timestamp)

        self._last_brightness_var = features["brightness_var"]

        logger.debug(
            "Scene classification",  # structured logging friendly
            label=decision.label,
            raw_confidence=decision.confidence,
            smoothed_label=smoothed_label,
            smoothed_confidence=smoothed_confidence,
            red_ratio=features["red_ratio"],
            motion=motion_score,
            low_texture=features["low_texture_ratio"],
            circle_score=features["circle_score"],
        )

        return smoothed_label, smoothed_confidence

    # ------------------------------------------------------------------
    # Feature extraction

    def _compute_features(
        self, frame: np.ndarray, ocr_lines: List[str], motion_score: float
    ) -> Dict[str, float]:
        if frame is None or frame.size == 0:
            raise ValueError("Frame is empty or None")

        hsv = self._bgr_to_hsv(frame)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red_mask = ((h <= 15) | (h >= 345)) & (s > 0.5) & (v > 0.5)
        red_ratio = float(np.mean(red_mask))

        gray = self._bgr_to_gray(frame)
        low_texture_ratio = self._compute_low_texture_ratio(gray)
        circle_score = self._compute_circle_score(gray)

        brightness_var = float(np.var(gray))
        prev_var = self._last_brightness_var
        brightness_delta = abs(brightness_var - prev_var) if prev_var is not None else 0.0

        long_line_present = any(len(self._strip_line_meta(line)) >= 40 for line in ocr_lines)

        bottom_lines = self._count_bottom_lines(ocr_lines)

        return {
            "red_ratio": red_ratio,
            "low_texture_ratio": low_texture_ratio,
            "circle_score": circle_score,
            "brightness_var": brightness_var,
            "brightness_delta": brightness_delta,
            "long_line_present": float(long_line_present),
            "bottom_lines": float(bottom_lines),
            "line_count": float(len(ocr_lines)),
            "motion_score": float(motion_score),
        }

    def _compute_low_texture_ratio(self, gray: np.ndarray) -> float:
        sobelx = self._apply_sobel(gray, axis="x")
        sobely = self._apply_sobel(gray, axis="y")
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        tau = 30.0
        ratio = float(np.mean(magnitude < tau))
        return ratio

    def _compute_circle_score(self, gray: np.ndarray) -> float:
        h, w = gray.shape[:2]
        roi = gray[: max(1, int(0.18 * h)), : max(1, int(0.18 * w))]
        if roi.size == 0:
            return 0.0

        gx = self._apply_sobel(roi, axis="x")
        gy = self._apply_sobel(roi, axis="y")
        magnitude = np.sqrt(gx ** 2 + gy ** 2)
        threshold = max(40.0, float(np.mean(magnitude)) * 1.5)
        edges = magnitude > threshold

        if not np.any(edges):
            return 0.0

        ys, xs = np.nonzero(edges)
        coords = np.column_stack((xs, ys)).astype(np.float32)
        center = coords.mean(axis=0)
        distances = np.sqrt(((coords - center) ** 2).sum(axis=1))
        mean_r = np.mean(distances)
        if mean_r < 1e-3:
            return 0.0
        std_r = np.std(distances)
        normalized = max(0.0, 1.0 - (std_r / (mean_r + 1e-6)))
        return float(np.clip(normalized, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Image helpers

    @staticmethod
    def _bgr_to_gray(frame: np.ndarray) -> np.ndarray:
        b = frame[..., 0].astype(np.float32)
        g = frame[..., 1].astype(np.float32)
        r = frame[..., 2].astype(np.float32)
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return gray

    @staticmethod
    def _bgr_to_hsv(frame: np.ndarray) -> np.ndarray:
        bgr = frame.astype(np.float32) / 255.0
        b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
        maxc = np.max(bgr, axis=2)
        minc = np.min(bgr, axis=2)
        v = maxc
        s = np.where(maxc == 0, 0, (maxc - minc) / (maxc + 1e-8))

        h = np.zeros_like(maxc)
        mask = maxc != minc
        rc = (maxc - r) / (maxc - minc + 1e-8)
        gc = (maxc - g) / (maxc - minc + 1e-8)
        bc = (maxc - b) / (maxc - minc + 1e-8)

        h[mask & (r == maxc)] = (bc - gc)[mask & (r == maxc)]
        h[mask & (g == maxc)] = 2.0 + (rc - bc)[mask & (g == maxc)]
        h[mask & (b == maxc)] = 4.0 + (gc - rc)[mask & (b == maxc)]
        h = (h / 6.0) % 1.0
        h_deg = h * 360.0

        hsv = np.stack([h_deg, s, v], axis=2)
        return hsv

    @staticmethod
    def _apply_sobel(image: np.ndarray, axis: str) -> np.ndarray:
        kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        kernel = kernel_x if axis == "x" else kernel_y

        pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
        result = np.zeros_like(image, dtype=np.float32)

        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                result += kernel[i, j] * padded[i : i + image.shape[0], j : j + image.shape[1]]
        return result

    # ------------------------------------------------------------------
    # Baseline / calibration

    def _update_baseline(self, features: Dict[str, float], timestamp: float) -> None:
        if self._baseline_ready:
            return

        if (timestamp - self._baseline_start) <= self._baseline_duration:
            self._baseline_stats["red_ratio"].update(features["red_ratio"])
            self._baseline_stats["motion"].update(features["motion_score"])
            self._baseline_stats["low_texture"].update(features["low_texture_ratio"])
        else:
            self._finalize_baseline()

    def _finalize_baseline(self) -> None:
        if self._baseline_ready:
            return

        for key, stats in self._baseline_stats.items():
            mean = stats.mean
            std = stats.std
            if key == "red_ratio":
                base = self._thresholds["battle_red"]
                self._adaptive_thresholds["battle_red"] = self._calibrate_threshold(base, mean, std)
            elif key == "motion":
                for t_key in ("battle_motion", "dialog_motion", "map_motion", "menu_motion", "loading_motion"):
                    base = self._thresholds[t_key]
                    self._adaptive_thresholds[t_key] = self._calibrate_threshold(base, mean, std)
            elif key == "low_texture":
                for t_key in ("map_low_texture", "menu_low_texture"):
                    base = self._thresholds[t_key]
                    self._adaptive_thresholds[t_key] = self._calibrate_threshold(base, mean, std)

        self._baseline_ready = True
        logger.debug("Adaptive thresholds computed", thresholds=self._adaptive_thresholds)

    @staticmethod
    def _calibrate_threshold(base: float, mean: float, std: float) -> float:
        if base == 0:
            return base
        if std == 0.0:
            std = abs(base) * 0.05  # guard against division by zero

        upper = mean + std
        lower = max(mean - std, 0.0)

        if mean > base:
            factor = 1.1 if mean <= upper else 1.2
        elif mean < base:
            factor = 0.9 if mean >= lower else 0.8
        else:
            factor = 1.0

        adjusted = base * factor
        return float(max(0.0, adjusted))

    # ------------------------------------------------------------------
    # Rule evaluation

    def _apply_rules(
        self, features: Dict[str, float], ocr_lines: List[str], motion_score: float
    ) -> SceneDecision:
        thresholds = self._adaptive_thresholds

        cleaned_lines = [self._strip_line_meta(line) for line in ocr_lines]
        line_count = len(cleaned_lines)

        battle_words = sum(1 for line in cleaned_lines if self._BATTLE_WORDS.search(line))
        big_numbers = sum(1 for line in cleaned_lines if self._BIG_NUMBER.search(line))
        interact_hits = sum(1 for line in cleaned_lines if self._INTERACT_WORDS.search(line))
        menu_hits = sum(1 for line in cleaned_lines if self._MENU_WORDS.search(line))

        decisions: Dict[str, float] = {}

        # battle scene
        battle_conditions_met = (
            motion_score > thresholds["battle_motion"]
            and (battle_words >= 1 or big_numbers >= 2)
        )
        if battle_conditions_met:
            feature_hits = 0
            if motion_score > thresholds["battle_motion"]:
                feature_hits += 1
            if battle_words >= 1 or big_numbers >= 2:
                feature_hits += 1

            confidence = 0.6 + 0.1 * feature_hits
            if features["red_ratio"] > thresholds["battle_red"]:
                confidence += 0.1
            confidence = min(confidence, 0.95)
            decisions["battle"] = confidence

        # dialog / cutscene
        bottom_lines = int(features["bottom_lines"])
        dialog_conditions_met = (
            motion_score < thresholds["dialog_motion"]
            and line_count >= 2
            and bottom_lines >= 2
        )
        if dialog_conditions_met:
            confidence = 0.5
            confidence += min(0.2, 0.05 * max(0, line_count - 2))
            if interact_hits:
                confidence += 0.2
            confidence = min(confidence, 0.9)
            decisions["dialog"] = confidence

        # world map
        map_conditions_met = (
            motion_score < thresholds["map_motion"]
            and features["low_texture_ratio"] > thresholds["map_low_texture"]
            and features["circle_score"] < 0.15
        )
        if map_conditions_met:
            confidence = 0.5
            confidence += min(
                0.35,
                max(0.0, features["low_texture_ratio"] - thresholds["map_low_texture"]) * 0.5,
            )
            confidence = min(confidence, 0.85)
            decisions["map"] = confidence

        # menu / UI panels
        menu_conditions_met = (
            motion_score < thresholds["menu_motion"]
            and features["low_texture_ratio"] > thresholds["menu_low_texture"]
            and (features["circle_score"] >= 0.12 or menu_hits > 0)
        )
        if menu_conditions_met:
            confidence = 0.5
            confidence += min(
                0.35,
                max(0.0, features["low_texture_ratio"] - thresholds["menu_low_texture"]) * 0.4,
            )
            if menu_hits:
                confidence += 0.1
            confidence = min(confidence, 0.85)
            decisions["menu"] = confidence

        # loading screens
        loading_conditions_met = (
            motion_score < thresholds["loading_motion"]
            and (line_count <= 1 or bool(features["long_line_present"]))
            and features["brightness_var"] < 150.0
            and features["brightness_delta"] < 50.0
        )
        if loading_conditions_met:
            confidence = 0.7
            if features["long_line_present"]:
                confidence += 0.1
            if features["brightness_var"] < 80.0:
                confidence += 0.05
            confidence = min(confidence, 0.95)
            decisions["loading"] = confidence

        if not decisions:
            return SceneDecision("unknown", 0.5)

        best_label, best_conf = max(decisions.items(), key=lambda kv: kv[1])

        if best_conf < 0.55:
            return SceneDecision("unknown", 0.5)

        return SceneDecision(best_label, best_conf)

    # ------------------------------------------------------------------
    # Output smoothing

    def _update_history(self, label: str, confidence: float, timestamp: float) -> None:
        self._history.append((timestamp, label, confidence))
        cutoff = timestamp - self._history_window
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _smooth_output(self, timestamp: float) -> Tuple[str, float]:
        if not self._history:
            return "unknown", 0.5

        cutoff = timestamp - self._history_window
        label_counter: Counter[str] = Counter()
        confidence_accumulator: Dict[str, List[float]] = {}

        for ts, label, conf in self._history:
            if ts < cutoff:
                continue
            label_counter[label] += 1
            confidence_accumulator.setdefault(label, []).append(conf)

        if not label_counter:
            last_label = self._history[-1][1]
            last_conf = self._history[-1][2]
            return last_label, last_conf

        best_label = max(label_counter.items(), key=lambda kv: (kv[1], np.mean(confidence_accumulator[kv[0]])))[0]
        confidences = confidence_accumulator.get(best_label, [0.5])
        smoothed_conf = float(np.clip(np.mean(confidences), 0.0, 1.0))
        return best_label, smoothed_conf

    # ------------------------------------------------------------------
    # Helpers

    def _strip_line_meta(self, line: str) -> str:
        cleaned = self._Y_COORD.sub("", line)
        return cleaned.strip()

    def _count_bottom_lines(self, ocr_lines: Iterable[str]) -> int:
        count = 0
        for raw in ocr_lines:
            match = self._Y_COORD.search(raw)
            if not match:
                # If no explicit coordinate, assume line could belong to bottom UI once.
                continue
            try:
                y = float(match.group(1))
            except ValueError:
                continue
            if y >= 0.6:
                count += 1
        return count

