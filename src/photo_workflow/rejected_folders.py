"""Assessment and purge helpers for rejected workflow folders."""

from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REJECTED_DIRNAME = "_Rejected"
BYTES_PER_GIGABYTE = 1024**3
ANSI_RESET = "\033[0m"
ANSI_SUCCESS = "\033[1;32m"

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectedFolderAssessmentItem:
    """Assessment details for one rejected folder."""

    folder_path: Path
    reclaimable_bytes: int
    percent_of_disk: float


@dataclass(frozen=True)
class RejectedFolderAssessment:
    """Aggregate assessment details for rejected folders under the camera root."""

    camera_root: Path
    disk_total_bytes: int
    folders: list[RejectedFolderAssessmentItem]
    total_reclaimable_bytes: int
    total_percent_of_disk: float


def assess_rejected_folders(camera_root: Path) -> RejectedFolderAssessment:
    """Return reclaimable-space details for rejected folders under the camera root."""
    disk_usage_path = resolve_disk_usage_path(camera_root)
    disk_total_bytes = shutil.disk_usage(disk_usage_path).total
    folders: list[RejectedFolderAssessmentItem] = []

    for folder_path in iter_rejected_folders(camera_root):
        reclaimable_bytes = calculate_directory_size(folder_path)
        folders.append(
            RejectedFolderAssessmentItem(
                folder_path=folder_path,
                reclaimable_bytes=reclaimable_bytes,
                percent_of_disk=calculate_disk_percentage(
                    reclaimable_bytes,
                    disk_total_bytes=disk_total_bytes,
                ),
            )
        )

    total_reclaimable_bytes = sum(item.reclaimable_bytes for item in folders)
    return RejectedFolderAssessment(
        camera_root=camera_root,
        disk_total_bytes=disk_total_bytes,
        folders=folders,
        total_reclaimable_bytes=total_reclaimable_bytes,
        total_percent_of_disk=calculate_disk_percentage(
            total_reclaimable_bytes,
            disk_total_bytes=disk_total_bytes,
        ),
    )


def iter_rejected_folders(camera_root: Path) -> list[Path]:
    """Return rejected folders under the configured camera root in stable order."""
    if not camera_root.exists():
        return []

    return sorted(
        path
        for path in camera_root.rglob(DEFAULT_REJECTED_DIRNAME)
        if path.is_dir() and path.name == DEFAULT_REJECTED_DIRNAME
    )


def resolve_disk_usage_path(camera_root: Path) -> Path:
    """Return an existing path suitable for disk-usage queries."""
    for candidate in (camera_root, *camera_root.parents):
        if candidate.exists():
            return candidate

    return Path.home()


def calculate_directory_size(directory_path: Path) -> int:
    """Return the total size in bytes for regular files under a directory."""
    return sum(path.stat().st_size for path in directory_path.rglob("*") if path.is_file())


def calculate_disk_percentage(reclaimable_bytes: int, *, disk_total_bytes: int) -> float:
    """Return reclaimable bytes as a percentage of the enclosing disk."""
    if disk_total_bytes == 0:
        return 0.0

    return (reclaimable_bytes / disk_total_bytes) * 100


def log_rejected_folder_assessment(
    assessment: RejectedFolderAssessment,
    *,
    purge_rejected: bool,
) -> None:
    """Log the rejected-folder assessment and optional purge intent."""
    if not assessment.folders:
        LOGGER.info(
            "no %s folders found under %s",
            DEFAULT_REJECTED_DIRNAME,
            assessment.camera_root,
        )
        return

    action_label = "purging" if purge_rejected else "assessment"
    LOGGER.info(
        "rejected folder %s under %s",
        action_label,
        assessment.camera_root,
    )
    if purge_rejected:
        LOGGER.info(format_purge_message(assessment.total_percent_of_disk))
    folder_action_label = "deleting" if purge_rejected else "would delete"
    for item in assessment.folders:
        LOGGER.info(
            "%s %s: %.1f GB reclaimable (%.2f%% of disk)",
            folder_action_label,
            item.folder_path.relative_to(assessment.camera_root),
            bytes_to_gigabytes(item.reclaimable_bytes),
            item.percent_of_disk,
        )

    LOGGER.info(
        "total rejected folders: %s, %.1f GB reclaimable (%.2f%% of disk)",
        len(assessment.folders),
        bytes_to_gigabytes(assessment.total_reclaimable_bytes),
        assessment.total_percent_of_disk,
    )


def purge_rejected_folders(assessment: RejectedFolderAssessment) -> int:
    """
    Delete all assessed rejected folders.

    Returns:
        The number of rejected folders removed.
    """
    for item in assessment.folders:
        shutil.rmtree(item.folder_path)

    return len(assessment.folders)


def format_purge_message(total_percent_of_disk: float) -> str:
    """Return a purge summary message with ANSI emphasis for interactive terminals."""
    message = f"purging {total_percent_of_disk:.2f}% disk space from rejected folders"
    if not sys.stderr.isatty():
        return message
    return f"{ANSI_SUCCESS}{message}{ANSI_RESET}"


def bytes_to_gigabytes(size_bytes: int) -> float:
    """Return a byte count converted to gigabytes."""
    return size_bytes / BYTES_PER_GIGABYTE
