"""Memory-card ingest workflow."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from photo_workflow.config import (
    MemoryCardCopyConfig,
    load_memory_card_copy_config,
    load_workflow_config,
)

DEFAULT_COPY_PROGRESS_INTERVAL = 50
METADATA_BATCH_SIZE = 100
DEFAULT_CARD_MARKER_DIRNAME = "DCIM"
ANSI_RESET = "\033[0m"
ANSI_SUCCESS = "\033[1;32m"
ANSI_WARNING = "\033[1;31m"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryCardImportResult:
    """Result for a memory-card ingest run."""

    card_root: Path
    target_dirs: tuple[Path, ...]
    imported_files: int
    free_space_gb: float
    free_space_percent: float


def run_memory_card_import(
    camera_root: Path,
    *,
    config: MemoryCardCopyConfig,
    report_disk_space: bool = True,
) -> MemoryCardImportResult | None:
    """
    Import files from a mounted memory card into capture-date directories.

    Returns:
        The import result when a card is detected, otherwise `None`.
    """
    card_root = find_memory_card_mount(config.card_mount_root)
    if card_root is None:
        LOGGER.info("no memory card detected in %s", config.card_mount_root)
        return None

    ensure_memory_card_is_writable(card_root)
    source_files = iter_memory_card_files(card_root, ignored_extensions=config.ignored_extensions)
    LOGGER.info("detected memory card at %s", card_root)
    LOGGER.info("found %s importable file(s) on the memory card", len(source_files))

    camera_root.mkdir(parents=True, exist_ok=True)
    if config.halt_on_insufficient_space:
        ensure_target_has_sufficient_space(source_files, camera_root)
    staging_root = Path(tempfile.mkdtemp(prefix=".photo-workflow-staging-", dir=camera_root))
    try:
        copy_plan = stage_and_plan_import(source_files, card_root, camera_root, staging_root)
        verification_start_time = time.perf_counter()
        try:
            verify_copied_files(copy_plan, verification_method=config.copy_verification)
        finally:
            log_stage_elapsed("verification", verification_start_time)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        LOGGER.info("removed staging directory %s", staging_root)

    LOGGER.info("deleting %s file(s) from memory card", len(source_files))
    delete_start_time = time.perf_counter()
    delete_memory_card_files(source_files, card_root=card_root)
    log_stage_elapsed("memory card deletion", delete_start_time)

    eject_start_time = time.perf_counter()
    eject_memory_card(card_root)
    log_stage_elapsed("memory card eject", eject_start_time)
    LOGGER.info(format_eject_message(card_root))

    free_space_gb = 0.0
    free_space_percent = 0.0
    if report_disk_space:
        free_space_gb, free_space_percent = report_target_disk_space(camera_root, config=config)
    return MemoryCardImportResult(
        card_root=card_root,
        target_dirs=tuple(sorted({target_path.parent for _, target_path in copy_plan})),
        imported_files=len(copy_plan),
        free_space_gb=free_space_gb,
        free_space_percent=free_space_percent,
    )


def stage_and_plan_import(
    source_files: list[Path],
    card_root: Path,
    camera_root: Path,
    staging_root: Path,
) -> list[tuple[Path, Path]]:
    """
    Stage card files, resolve capture timestamps, and move them into place.

    Returns:
        Original source paths paired with their final target paths.
    """
    staging_plan = build_staging_plan(source_files, card_root, staging_root)
    LOGGER.info("copying %s file(s) to staging at %s", len(source_files), staging_root)
    stage_start_time = time.perf_counter()
    copy_files(staging_plan, progress_verb="staged")
    log_stage_elapsed("staging copy", stage_start_time)

    staged_files = [staged_path for _, staged_path in staging_plan]
    metadata_start_time = time.perf_counter()
    capture_timestamps = read_capture_timestamps(staged_files)
    log_stage_elapsed("metadata reading", metadata_start_time)

    final_plan = build_copy_plan(staged_files, camera_root, capture_timestamps)
    LOGGER.info("moving %s staged file(s) to final destinations", len(final_plan))
    finalize_start_time = time.perf_counter()
    move_files(final_plan)
    log_stage_elapsed("finalization", finalize_start_time)
    return [
        (source_path, target_path)
        for (source_path, _), (_, target_path) in zip(staging_plan, final_plan)
    ]


def build_staging_plan(
    source_files: list[Path], card_root: Path, staging_root: Path
) -> list[tuple[Path, Path]]:
    """Return source and staging paths while preserving card-relative directories."""
    return [
        (source_path, staging_root / source_path.relative_to(card_root))
        for source_path in source_files
    ]


def find_memory_card_mount(mount_root: Path) -> Path | None:
    """Return the first mounted volume that looks like a camera card."""
    if not mount_root.exists():
        return None

    for volume_path in sorted(path for path in mount_root.iterdir() if path.is_dir()):
        if (volume_path / DEFAULT_CARD_MARKER_DIRNAME).is_dir():
            return volume_path

    return None


def ensure_memory_card_is_writable(card_root: Path) -> None:
    """
    Raise an error when the mounted memory card is read-only.

    Raises:
        OSError: If the memory card's filesystem is mounted read-only.
    """
    if os.statvfs(card_root).f_flag & os.ST_RDONLY:
        raise OSError(f"memory card is read-only: {card_root}")


def iter_memory_card_files(
    card_root: Path,
    *,
    ignored_extensions: tuple[str, ...] | None = None,
) -> list[Path]:
    """Return all regular, non-hidden files under the mounted memory card."""
    active_ignored_extensions = set(
        ignored_extensions or load_memory_card_copy_config().ignored_extensions
    )
    return sorted(
        path
        for path in card_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() not in active_ignored_extensions
        and not is_hidden_card_path(path, card_root=card_root)
    )


def is_hidden_card_path(path: Path, *, card_root: Path) -> bool:
    """Return whether the card-relative path contains hidden path components."""
    relative_parts = path.relative_to(card_root).parts
    return any(part.startswith(".") for part in relative_parts)


def build_copy_plan(
    source_files: list[Path],
    camera_root: Path,
    capture_timestamps: dict[Path, datetime],
) -> list[tuple[Path, Path]]:
    """
    Return source and target pairs, failing early on unresolvable filename collisions.

    When two source files would share the same timestamped target name, the
    source's parent folder name is inserted to disambiguate them.

    Returns:
        Source and target file pairs for the import operation.

    Raises:
        FileExistsError: If a filename collision remains after disambiguation.
    """
    planned_targets: set[Path] = set()
    copy_plan: list[tuple[Path, Path]] = []
    total_files = len(source_files)
    LOGGER.info("planning import destinations for %s file(s)", total_files)

    for source_path in source_files:
        captured_at = capture_timestamps[source_path]
        capture_dir = build_capture_dir(camera_root, captured_at.date())
        target_path = capture_dir / build_target_filename(captured_at, source_path)
        if target_path in planned_targets:
            target_path = capture_dir / build_target_filename(
                captured_at, source_path, disambiguator=source_path.parent.name
            )
        if target_path in planned_targets:
            raise FileExistsError(f"Duplicate filename on memory card: {source_path.name}")
        if target_path.exists():
            raise FileExistsError(f"Target file already exists: {target_path}")
        planned_targets.add(target_path)
        copy_plan.append((source_path, target_path))

    LOGGER.info(
        "planned import destinations for %s file(s) across %s date(s)",
        total_files,
        len({captured_at.date() for captured_at in capture_timestamps.values()}),
    )
    return copy_plan


def read_capture_timestamps(source_files: list[Path]) -> dict[Path, datetime]:
    """Return EXIF capture timestamps with per-file filesystem fallbacks."""
    if not source_files:
        return {}

    exiftool_path = shutil.which("exiftool")
    if exiftool_path is None:
        LOGGER.warning("ExifTool unavailable; using filesystem timestamps for all files")
        return read_filesystem_capture_timestamps(source_files)

    total_files = len(source_files)
    capture_timestamps: dict[Path, datetime] = {}
    fallback_count = 0
    LOGGER.info("reading capture timestamps in batches of %s file(s)", METADATA_BATCH_SIZE)
    for batch_start in range(0, total_files, METADATA_BATCH_SIZE):
        source_batch = source_files[batch_start : batch_start + METADATA_BATCH_SIZE]
        batch_timestamps, batch_fallbacks = read_capture_timestamp_batch(
            exiftool_path, source_batch
        )
        capture_timestamps.update(batch_timestamps)
        fallback_count += batch_fallbacks
        processed_files = min(batch_start + len(source_batch), total_files)
        LOGGER.info("read capture timestamps for %s/%s files", processed_files, total_files)
    if fallback_count:
        LOGGER.warning(
            "used filesystem timestamps for %s/%s file(s) without usable metadata",
            fallback_count,
            total_files,
        )
    return capture_timestamps


def read_filesystem_capture_timestamps(source_files: list[Path]) -> dict[Path, datetime]:
    """Return preserved filesystem modification timestamps."""
    return {
        source_path: datetime.fromtimestamp(source_path.stat().st_mtime)
        for source_path in source_files
    }


def read_capture_timestamp_batch(
    exiftool_path: str, source_files: list[Path]
) -> tuple[dict[Path, datetime], int]:
    """Return capture timestamps and fallback count from one ExifTool batch."""
    result = subprocess.run(
        [
            exiftool_path,
            "-fast2",
            "-j",
            "-DateTimeOriginal",
            "-SubSecTimeOriginal",
            "-CreateDate",
            "-SubSecCreateDate",
            "-MediaCreateDate",
            *(str(source_path) for source_path in source_files),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    metadata_records = json.loads(result.stdout)
    metadata_by_path = {Path(record["SourceFile"]): record for record in metadata_records}
    capture_timestamps: dict[Path, datetime] = {}
    fallback_count = 0
    for source_path in source_files:
        fallback_timestamp = datetime.fromtimestamp(source_path.stat().st_mtime)
        captured_at, used_fallback = extract_capture_timestamp(
            metadata_by_path.get(source_path, {}), fallback_timestamp=fallback_timestamp
        )
        capture_timestamps[source_path] = captured_at
        fallback_count += used_fallback
    return capture_timestamps, fallback_count


def extract_capture_timestamp(
    metadata: dict[str, object],
    *,
    fallback_timestamp: datetime,
) -> tuple[datetime, bool]:
    """Return the best available capture timestamp and whether fallback was used."""
    timestamp_tags = (
        ("DateTimeOriginal", "SubSecTimeOriginal"),
        ("CreateDate", "SubSecCreateDate"),
        ("MediaCreateDate", None),
    )
    for timestamp_tag, subsecond_tag in timestamp_tags:
        captured_at = metadata.get(timestamp_tag)
        if isinstance(captured_at, str):
            try:
                timestamp = datetime.strptime(captured_at[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
            subsecond_value = metadata.get(subsecond_tag, "") if subsecond_tag else ""
            subseconds = str(subsecond_value)
            microseconds = int((subseconds + "000000")[:6]) if subseconds.isdigit() else 0
            return timestamp.replace(microsecond=microseconds), False
    return fallback_timestamp, True


def build_capture_dir(camera_root: Path, capture_date: date) -> Path:
    """Return the dated destination directory for a media capture date."""
    return (
        camera_root
        / capture_date.strftime("%Y")
        / capture_date.strftime("%m")
        / capture_date.strftime("%Y%m%d")
    )


def build_target_filename(
    captured_at: datetime, source_path: Path, *, disambiguator: str | None = None
) -> str:
    """
    Return a target filename prefixed with its capture timestamp.

    Args:
        captured_at: The best available timestamp for the captured media.
        source_path: The staged media file retaining its original filename.
        disambiguator: An extra path segment (e.g. the source's parent folder
            name) inserted before the original name to resolve a collision.

    Returns:
        The chronologically sortable destination filename.
    """
    hundredths = captured_at.microsecond // 10_000
    prefix = f"{captured_at:%Y%m%d_%H%M%S}_{hundredths:02d}"
    if disambiguator:
        prefix = f"{prefix}_{disambiguator}"
    return f"{prefix}_{source_path.name}"


def copy_files(
    copy_plan: list[tuple[Path, Path]], *, progress_verb: str = "copied"
) -> None:
    """Copy source files into the target directory and log progress."""
    total_files = len(copy_plan)
    for index, (source_path, target_path) in enumerate(copy_plan, start=1):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if should_log_copy_progress(index, total_files):
            LOGGER.info("%s %s/%s files", progress_verb, index, total_files)


def move_files(move_plan: list[tuple[Path, Path]]) -> None:
    """Move staged files to final destinations and log progress."""
    total_files = len(move_plan)
    for index, (source_path, target_path) in enumerate(move_plan, start=1):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        if should_log_copy_progress(index, total_files):
            LOGGER.info("finalized %s/%s files", index, total_files)


def ensure_target_has_sufficient_space(
    source_files: list[Path],
    target_dir: Path,
) -> None:
    """
    Raise an error when the target volume cannot hold all planned source files.

    Raises:
        OSError: If the target volume lacks space for every planned source file.
    """
    required_bytes = sum(source_path.stat().st_size for source_path in source_files)
    available_bytes = shutil.disk_usage(target_dir).free
    if required_bytes > available_bytes:
        raise OSError(
            "insufficient free space on target volume: "
            f"{format_gibibytes(required_bytes)} required, "
            f"{format_gibibytes(available_bytes)} available"
        )


def format_gibibytes(bytes_count: int) -> str:
    """Return a byte count formatted in gibibytes for user-facing output."""
    return f"{bytes_count / (1024**3):.1f} GB"


def should_log_copy_progress(index: int, total_files: int) -> bool:
    """Return whether the current copy position should emit progress logging."""
    return should_log_progress(index, total_files, interval=DEFAULT_COPY_PROGRESS_INTERVAL)


def should_log_progress(index: int, total_files: int, *, interval: int) -> bool:
    """Return whether progress should be logged for the first, periodic, or last item."""
    return index == 1 or index == total_files or index % interval == 0


def log_stage_elapsed(stage_name: str, start_time: float) -> None:
    """Log elapsed time for a named workflow stage."""
    elapsed_seconds = time.perf_counter() - start_time
    LOGGER.info("%s completed in %.2fs", stage_name, elapsed_seconds)


def verify_copied_files(
    copy_plan: list[tuple[Path, Path]],
    *,
    verification_method: str,
) -> None:
    """
    Validate imported files before the source card is modified.

    Raises:
        ValueError: If a copied file is missing, has a size mismatch, or fails checksum validation.
    """
    LOGGER.info(
        "verifying %s copied file(s) using %s verification",
        len(copy_plan),
        verification_method,
    )
    for source_path, target_path in copy_plan:
        if not target_path.exists():
            raise ValueError(f"Copied file is missing: {target_path}")
        if source_path.stat().st_size != target_path.stat().st_size:
            raise ValueError(f"Copied file size mismatch: {source_path.name}")
        if verification_method == "crc32":
            if calculate_crc32(source_path) != calculate_crc32(target_path):
                raise ValueError(f"Copied file checksum mismatch: {source_path.name}")


def calculate_crc32(file_path: Path) -> int:
    """Return the CRC32 checksum for a file."""
    checksum = 0
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum


def delete_memory_card_files(source_files: list[Path], *, card_root: Path) -> None:
    """Permanently delete imported files from the memory card."""
    for source_path in source_files:
        source_path.unlink()

    for directory_path in sorted(
        (path for path in card_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory_path.rmdir()
        except OSError:
            continue


def eject_memory_card(card_root: Path) -> None:
    """Eject the mounted memory card."""
    diskutil_path = require_tool("diskutil")
    subprocess.run(
        [diskutil_path, "eject", str(card_root)],
        capture_output=True,
        check=True,
        text=True,
    )


def report_target_disk_space(
    target_dir: Path,
    *,
    config: MemoryCardCopyConfig,
    reclaimable_percent: float | None = None,
) -> tuple[float, float]:
    """
    Log remaining disk space for the target volume and warn when low.

    Returns:
        The free space in gigabytes and percent for the target volume.
    """
    usage = shutil.disk_usage(target_dir)
    free_space_gb = usage.free / (1024**3)
    free_space_percent = (usage.free / usage.total) * 100
    LOGGER.info(
        "target volume free space: %.1f GB (%.1f%%)",
        free_space_gb,
        free_space_percent,
    )

    if (
        free_space_gb < config.low_disk_warning_gb
        or free_space_percent < config.low_disk_warning_percent
    ):
        LOGGER.warning(
            format_low_disk_warning(
                target_dir,
                free_space_gb,
                free_space_percent,
            ),
        )
        if reclaimable_percent is not None and reclaimable_percent > 0:
            LOGGER.warning(format_reclaimable_space_hint(reclaimable_percent))

    return free_space_gb, free_space_percent


def format_low_disk_warning(
    target_dir: Path,
    free_space_gb: float,
    free_space_percent: float,
) -> str:
    """Return a low-disk warning message with ANSI emphasis for interactive terminals."""
    warning_message = (
        f"low disk space on {target_dir}: {free_space_gb:.1f} GB free "
        f"({free_space_percent:.1f}%)"
    )
    if not sys.stderr.isatty():
        return warning_message
    return f"{ANSI_WARNING}{warning_message}{ANSI_RESET}"


def format_reclaimable_space_hint(reclaimable_percent: float) -> str:
    """Return a reclaimable-space hint with ANSI emphasis for interactive terminals."""
    hint_message = (
        "run with --purge-rejected to reclaim an additional "
        f"{reclaimable_percent:.2f}% disk space."
    )
    if not sys.stderr.isatty():
        return hint_message
    return f"{ANSI_WARNING}{hint_message}{ANSI_RESET}"


def format_eject_message(card_root: Path) -> str:
    """Return an ejection message with ANSI emphasis for interactive terminals."""
    message = f"ejected memory card at {card_root}"
    if not sys.stderr.isatty():
        return message
    return f"{ANSI_SUCCESS}{message}{ANSI_RESET}"


def require_tool(name: str) -> str:
    """
    Return the full path to a required external binary.

    Returns:
        The resolved executable path.

    Raises:
        FileNotFoundError: If the required executable is not available on `PATH`.
    """
    tool_path = shutil.which(name)
    if tool_path is None:
        raise FileNotFoundError(f"Required tool not found on PATH: {name}")
    return tool_path


def main() -> None:
    """Run the memory-card ingest workflow into capture-date folders."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    workflow_config = load_workflow_config()
    copy_config = load_memory_card_copy_config()
    run_memory_card_import(workflow_config.camera_root, config=copy_config)
