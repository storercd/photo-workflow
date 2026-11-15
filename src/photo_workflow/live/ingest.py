"""Renaming of newly arrived Live files, reusing the memory card ingestion naming."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.stages import INPUT_SUBFOLDER
from photo_workflow.memory_card_copy import (
    build_capture_dir,
    build_target_filename,
    read_capture_timestamps,
)

LOGGER = logging.getLogger(__name__)


def read_arrival_timestamp(path: Path) -> datetime:
    """Return the capture time of an arrived file, falling back to its modified time."""
    return read_capture_timestamps([path])[path]


def rename_arrival(
    layout: LiveLayout,
    arrived: Path,
    *,
    captured_at: datetime | None = None,
) -> Path:
    """
    Move a newly arrived file to its date folder under its date-stamped name.

    This is the first step for any new file. The date folder and file name come from
    the same functions the memory card import uses. An existing target is overwritten
    as a failsafe, and a sidecar next to the arrived file moves with it.

    Returns:
        The renamed file inside the date folder's `input` subfolder.
    """
    timestamp = captured_at or read_arrival_timestamp(arrived)
    target_dir = build_capture_dir(layout.root, timestamp.date()) / INPUT_SUBFOLDER
    target = target_dir / build_target_filename(timestamp, arrived)
    target_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        LOGGER.warning("overwriting existing live file %s", target)
    sidecar = arrived.with_suffix(".xmp")
    arrived.replace(target)
    if sidecar.is_file():
        sidecar.replace(target.with_suffix(".xmp"))
    LOGGER.info("renamed arrival %s -> %s", arrived.name, target)
    return target
