"""Full photo workflow orchestration."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from pathlib import Path

from photo_workflow.config import build_today_source_dir, load_config
from photo_workflow.memory_card_copy import report_target_disk_space, run_memory_card_import
from photo_workflow.rejected_folders import (
    RejectedFolderAssessment,
    assess_rejected_folders,
    log_rejected_folder_assessment,
    purge_rejected_folders,
    resolve_disk_usage_path,
)
from photo_workflow.video_notes import run_video_notes_step

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments for the full photo workflow.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        prog="photo-workflow-run",
        description="Run the configured photo workflow steps.",
    )
    parser.add_argument(
        "--purge-rejected",
        action="store_true",
        help="Delete any _Rejected folders found under the configured camera root.",
    )
    return parser.parse_args(argv)


def run_rejected_folder_step(
    camera_root: Path,
    purge_rejected: bool,
) -> RejectedFolderAssessment:
    """
    Assess rejected folders and optionally purge them.

    Returns:
        The rejected-folder assessment gathered before any optional purge.
    """
    scan_start_time = time.perf_counter()
    assessment = assess_rejected_folders(camera_root)
    log_stage_elapsed("rejected-folder scan", scan_start_time)
    log_rejected_folder_assessment(assessment, purge_rejected=purge_rejected)
    if not purge_rejected or not assessment.folders:
        return assessment

    purge_start_time = time.perf_counter()
    purged_count = purge_rejected_folders(assessment)
    log_stage_elapsed("rejected-folder purge", purge_start_time)
    LOGGER.info("purged %s rejected folder(s)", purged_count)
    return assessment


def log_stage_elapsed(stage_name: str, start_time: float) -> None:
    """Log elapsed time for a named workflow stage."""
    elapsed_seconds = time.perf_counter() - start_time
    LOGGER.info("%s completed in %.2fs", stage_name, elapsed_seconds)


def open_target_folder(target_dir: Path) -> None:
    """Open the completed target folder in Finder."""
    subprocess.run(["open", str(target_dir)], check=True)


def main(argv: list[str] | None = None) -> None:
    """Run the configured workflow steps in order."""
    parsed_args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    config = load_config()
    source_dir = build_today_source_dir(camera_root=config.workflow.camera_root)
    run_memory_card_import(
        source_dir,
        config=config.memory_card_copy,
        report_disk_space=False,
    )
    video_notes_start_time = time.perf_counter()
    try:
        run_video_notes_step(source_dir, config=config.video_notes)
    finally:
        log_stage_elapsed("video notes", video_notes_start_time)

    rejected_folder_assessment = run_rejected_folder_step(
        config.workflow.camera_root,
        purge_rejected=parsed_args.purge_rejected,
    )
    report_target_disk_space(
        resolve_disk_usage_path(config.workflow.camera_root),
        config=config.memory_card_copy,
        reclaimable_percent=(
            None
            if parsed_args.purge_rejected
            else rejected_folder_assessment.total_percent_of_disk
        ),
    )
    open_target_folder(source_dir)
