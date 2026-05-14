"""Structured logging helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def setup_logging(log_dir: Path, log_level: str) -> tuple[logging.Logger, Path]:
    """Create a timestamped batch log file and console logger."""

    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"batch_cellpose_{timestamp}.log"
    logger = logging.getLogger("macsima_cellpose")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path
