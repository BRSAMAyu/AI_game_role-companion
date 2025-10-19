"""Utility helpers for configuring logging across the backend."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def configure_logging(log_file: Optional[Path] = None, level: str = "INFO") -> None:
    """Configure loguru logger with console and optional file sinks."""
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level=level, colorize=True, format=DEFAULT_LOG_FORMAT)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=DEFAULT_LOG_FORMAT, rotation="1 week")


def get_logger(name: str):
    """Return a child logger."""
    return logger.bind(module=name)
