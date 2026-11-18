"""Troubleshooting log for Live mode."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3


def live_log_path() -> Path:
    """Return the Live log location, outside the Live folder so Clean keeps it."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "photo-workflow" / "live.log"
    return Path.home() / ".local" / "state" / "photo-workflow" / "live.log"


def configure_live_logging(path: Path | None = None) -> Path:
    """
    Send `photo_workflow.live` log records to a rotating file.

    Safe to call more than once.

    Returns:
        The log file path.
    """
    log_path = path or live_log_path()
    logger = logging.getLogger("photo_workflow.live")
    resolved = str(log_path)
    if any(getattr(handler, "baseFilename", None) == resolved for handler in logger.handlers):
        return log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return log_path
