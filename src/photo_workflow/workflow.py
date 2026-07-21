"""Full photo workflow orchestration."""

from __future__ import annotations

import logging

from photo_workflow.config import build_today_source_dir, load_config
from photo_workflow.memory_card_copy import run_memory_card_import
from photo_workflow.video_notes import run_video_notes_step


def main() -> None:
    """Run the configured workflow steps in order."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config()
    source_dir = build_today_source_dir(camera_root=config.workflow.camera_root)
    run_memory_card_import(source_dir, config=config.memory_card_copy)
    run_video_notes_step(source_dir, config=config.video_notes)
