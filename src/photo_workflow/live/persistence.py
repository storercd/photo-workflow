"""Saving and restoring per-image stage progress so a restart does not redo finished work."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from photo_workflow.live.stages import SPECS, STAGE_ORDER, StageId, StageStatus

LOGGER = logging.getLogger(__name__)
STATE_FILE_NAME = ".pipeline-state.json"
STATE_VERSION = 1
RESTORABLE = {StageStatus.DONE, StageStatus.SKIPPED}


def input_signature(photo: Path) -> list[int]:
    """
    Return a size and mtime fingerprint of a photo's original.

    Returns:
        `[size, mtime_ns]`, or `[-1, 0]` when the file cannot be read.
    """
    try:
        stat = photo.stat()
    except OSError:
        return [-1, 0]
    return [stat.st_size, stat.st_mtime_ns]


def dump_state(
    root: Path,
    images: dict,
    enabled: dict[str, bool],
    autocrop: list,
    revisions: dict[str, int],
) -> str:
    """
    Serialize finished stage records; `images` maps each photo to its `ImageStage` records.

    Returns:
        The JSON text to save.
    """
    saved = {}
    for photo, stages in images.items():
        records = {
            stage.value: {
                "status": record.status.value,
                "outputs": [_relative(path, root) for path in record.outputs],
                "message": record.message,
            }
            for stage, record in stages.items()
            if record.status in RESTORABLE
        }
        saved[_relative(photo, root)] = {"input": input_signature(photo), "stages": records}
    state = {
        "version": STATE_VERSION,
        "enabled": enabled,
        "autocrop": autocrop,
        "revisions": revisions,
        "images": saved,
    }
    return json.dumps(state, indent=1, sort_keys=True)


def write_state(root: Path, text: str) -> None:
    """Atomically replace the state file."""
    path = root / STATE_FILE_NAME
    temp = path.with_suffix(".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def load_state(root: Path) -> dict:
    """
    Read the saved state.

    Returns:
        The saved state, or an empty dict when there is none or it is unreadable.
    """
    try:
        state = json.loads((root / STATE_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(state, dict) or state.get("version") != STATE_VERSION:
        return {}
    return state


def restorable_records(
    root: Path,
    state: dict,
    photo: Path,
    enabled: dict[str, bool],
    autocrop: list,
    revisions: dict[str, int],
) -> dict[StageId, tuple[StageStatus, tuple[Path, ...], str]]:
    """
    Pick the saved stage records that are still valid for a photo.

    A record is dropped, along with every stage after it, when the original changed, when
    the feature's on/off state, the AutoCrop options or the stage's implementation revision
    differ from when it was saved, or when one of its output files is gone.

    Returns:
        Status, outputs and message per restorable stage.
    """
    entry = state.get("images", {}).get(_relative(photo, root))
    if not entry or entry.get("input") != input_signature(photo):
        return {}
    saved_enabled = state.get("enabled", {})
    saved_revisions = state.get("revisions", {})
    restored = {}
    for stage in STAGE_ORDER:
        record = entry.get("stages", {}).get(stage.value)
        if record is None or not _still_valid(stage, saved_enabled, enabled, state, autocrop):
            break
        if saved_revisions.get(stage.value, 1) != revisions.get(stage.value, 1):
            break
        outputs = tuple(root / name for name in record.get("outputs", []))
        if not all(path.exists() for path in outputs):
            break
        restored[stage] = (StageStatus(record["status"]), outputs, record.get("message", ""))
    return restored


def _still_valid(
    stage: StageId, saved_enabled: dict, enabled: dict[str, bool], state: dict, autocrop: list
) -> bool:
    spec = SPECS[stage]
    if stage is StageId.AUTOCROP:
        return (
            state.get("autocrop") == autocrop
            and saved_enabled.get(stage.value) == enabled.get(stage.value)
        )
    if spec.blocks_when_disabled or stage is StageId.FTP:
        return True
    return saved_enabled.get(stage.value) == enabled.get(stage.value)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
