"""Delete unused JPEG companion files from a photo library."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from photo_workflow.config import load_workflow_config

LOGGER = logging.getLogger(__name__)
JPEG_SUFFIX = ".jpg"


@dataclass(frozen=True)
class JpegAssessment:
    """JPEG files and disk space eligible for removal."""

    root_dir: Path
    jpeg_paths: list[Path]
    reclaimable_bytes: int


@dataclass(frozen=True)
class JpegPurgeResult:
    """The outcome of a JPEG purge attempt."""

    deleted_files: int
    reclaimed_bytes: int


def assess_jpegs(root_dir: Path) -> JpegAssessment:
    """Return JPEG files and their total size under a root directory."""
    jpeg_paths = iter_jpegs(root_dir)
    return JpegAssessment(
        root_dir=root_dir,
        jpeg_paths=jpeg_paths,
        reclaimable_bytes=sum(path.stat().st_size for path in jpeg_paths),
    )


def iter_jpegs(root_dir: Path) -> list[Path]:
    """Return regular JPEG files under a root directory in stable order."""
    if not root_dir.exists():
        return []

    return sorted(path for path in root_dir.rglob("*") if path.is_file() and is_jpeg(path))


def is_jpeg(file_path: Path) -> bool:
    """Return whether a file has the JPEG suffix used by camera companions."""
    return file_path.suffix.lower() == JPEG_SUFFIX


def purge_jpegs(assessment: JpegAssessment, *, dry_run: bool) -> JpegPurgeResult:
    """
    Delete assessed JPEG files unless dry-run mode is enabled.

    Returns:
        The number of deleted files and bytes recovered.
    """
    if dry_run:
        return JpegPurgeResult(deleted_files=0, reclaimed_bytes=0)

    for jpeg_path in assessment.jpeg_paths:
        jpeg_path.unlink()

    return JpegPurgeResult(
        deleted_files=len(assessment.jpeg_paths),
        reclaimed_bytes=assessment.reclaimable_bytes,
    )


def log_assessment(assessment: JpegAssessment, *, dry_run: bool) -> None:
    """Log the planned JPEG deletion and its reclaimable disk space."""
    action = "would delete" if dry_run else "deleting"
    LOGGER.info(
        "%s %s JPEG file(s) under %s, recovering %s",
        action,
        len(assessment.jpeg_paths),
        assessment.root_dir,
        format_bytes(assessment.reclaimable_bytes),
    )


def format_bytes(size_bytes: int) -> str:
    """
    Format a byte count using binary units.

    Returns:
        The formatted byte count.
    """
    for unit, divisor in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if size_bytes >= divisor:
            return f"{size_bytes / divisor:.1f} {unit}"
    return f"{size_bytes} bytes"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse JPEG purge command-line arguments.

    Returns:
        The parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        prog="photo-workflow-purge-jpegs",
        description="Delete .jpg companion files under a photo root.",
    )
    parser.add_argument("root_dir", nargs="?", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files and recoverable space without deleting anything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the JPEG companion-file purge command."""
    parsed_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root_dir = parsed_args.root_dir or load_workflow_config().camera_root
    assessment = assess_jpegs(root_dir)
    log_assessment(assessment, dry_run=parsed_args.dry_run)
    result = purge_jpegs(assessment, dry_run=parsed_args.dry_run)
    if not parsed_args.dry_run:
        LOGGER.info(
            "deleted %s JPEG file(s), recovered %s",
            result.deleted_files,
            format_bytes(result.reclaimed_bytes),
        )
