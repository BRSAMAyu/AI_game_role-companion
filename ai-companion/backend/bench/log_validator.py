"""Analyze runtime logs and produce heuristic tuning suggestions."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List

THRESHOLDS = {"battle": 0.6, "dialog": 0.6, "map": 0.55, "menu": 0.55, "loading": 0.7}
DEFAULT_LOG = Path("logs/latest.log")

LOG_PATTERN = re.compile(
    r"scene=(?P<scene>\w+)"  # scene label
    r".*?conf=(?P<conf>[0-9.]+)"
    r".*?motion=(?P<motion>[0-9.]+)"
    r".*?red_ratio=(?P<red>[0-9.]+)"
    r".*?sobel_lowtex_ratio=(?P<texture>[0-9.]+)"
    r".*?lines_count=(?P<lines>\d+)",
)


def parse_log(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    entries: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = LOG_PATTERN.search(line)
            if not match:
                continue
            data = {
                "scene": match.group("scene").lower(),
                "conf": float(match.group("conf")),
                "motion": float(match.group("motion")),
                "red_ratio": float(match.group("red")),
                "texture": float(match.group("texture")),
                "lines": float(match.group("lines")),
            }
            entries.append(data)
    return entries


def summarize(entries: List[Dict[str, float]]) -> None:
    if not entries:
        print("No frame metrics found in log.")
        return

    by_scene: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for entry in entries:
        by_scene[entry["scene"]].append(entry)

    print("Scene statistics:")
    for scene, records in sorted(by_scene.items()):
        confidences = [rec["conf"] for rec in records]
        motions = [rec["motion"] for rec in records]
        texture = [rec["texture"] for rec in records]
        print(
            f"  {scene:<8} count={len(records):>4} conf_mean={mean(confidences):.3f} conf_var={variance(confidences) if len(confidences)>1 else 0:.4f} "
            f"motion_mean={mean(motions):.4f} texture_mean={mean(texture):.4f}"
        )

    low_conf_total = 0
    for entry in entries:
        threshold = THRESHOLDS.get(entry["scene"], 0.6)
        if entry["conf"] < threshold:
            low_conf_total += 1
    print(f"Low-confidence frames: {low_conf_total}/{len(entries)} ({low_conf_total/len(entries)*100:.1f}%)")

    battle_records = by_scene.get("battle", [])
    if battle_records:
        misfires = sum(1 for rec in battle_records if rec["motion"] < 0.08)
        rate = misfires / len(battle_records)
        if rate > 0.2:
            print("Suggestion: Battle motion threshold might be too low; consider raising above 0.08.")

    dialog_records = by_scene.get("dialog", [])
    if dialog_records:
        sparse = sum(1 for rec in dialog_records if rec["lines"] < 2)
        if sparse / len(dialog_records) > 0.2:
            print("Suggestion: Dialog detection misses lines; emphasize F/继续 keywords or lower bottom-line tolerance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate runtime logs and surface tuning hints.")
    parser.add_argument("logfile", nargs="?", default=DEFAULT_LOG, type=Path, help="Path to the log file to inspect")
    args = parser.parse_args()

    try:
        entries = parse_log(args.logfile)
    except FileNotFoundError as exc:
        print(exc)
        return

    summarize(entries)


if __name__ == "__main__":  # pragma: no cover - CLI tool
    main()
