"""Memory-card ingest workflow."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from photo_workflow.config import (
    MemoryCardCopyConfig,
    build_today_source_dir,
    load_memory_card_copy_config,
    load_workflow_config,
)

DEFAULT_COPY_PROGRESS_INTERVAL = 50
DEFAULT_CARD_MARKER_DIRNAME = "DCIM"
ANSI_RESET = "\033[0m"
ANSI_SUCCESS = "\033[1;32m"
ANSI_WARNING = "\033[1;31m"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryCardImportResult:
    """Result for a memory-card ingest run."""

    card_root: Path
    target_dir: Path
    imported_files: int
    free_space_gb: float
    free_space_percent: float


def run_memory_card_import(
    target_dir: Path,
    *,
    config: MemoryCardCopyConfig,
    report_disk_space: bool = True,
) -> MemoryCardImportResult | None:
    """
    Import files from a mounted memory card into the target directory.

    Returns:
        The import result when a card is detected, otherwise `None`.
    """
    card_root = find_memory_card_mount(config.card_mount_root)
    if card_root is None:
        LOGGER.info("no memory card detected in %s", config.card_mount_root)
        return None

    source_files = iter_memory_card_files(card_root, ignored_extensions=config.ignored_extensions)
    LOGGER.info("detected memory card at %s", card_root)
    LOGGER.info("found %s importable file(s) on the memory card", len(source_files))

    target_dir.mkdir(parents=True, exist_ok=True)
    copy_plan = build_copy_plan(source_files, target_dir)
    if config.halt_on_insufficient_space:
        ensure_target_has_sufficient_space(copy_plan, target_dir)
    copy_start_time = time.perf_counter()
    try:
        copy_files(copy_plan)
    finally:
        log_stage_elapsed("import", copy_start_time)

    verification_start_time = time.perf_counter()
    try:
        verify_copied_files(copy_plan, verification_method=config.copy_verification)
    finally:
        log_stage_elapsed("verification", verification_start_time)

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
        free_space_gb, free_space_percent = report_target_disk_space(target_dir, config=config)
    return MemoryCardImportResult(
        card_root=card_root,
        target_dir=target_dir,
        imported_files=len(copy_plan),
        free_space_gb=free_space_gb,
        free_space_percent=free_space_percent,
    )


def find_memory_card_mount(mount_root: Path) -> Path | None:
    """Return the first mounted volume that looks like a camera card."""
    if not mount_root.exists():
        return None

    for volume_path in sorted(path for path in mount_root.iterdir() if path.is_dir()):
        if (volume_path / DEFAULT_CARD_MARKER_DIRNAME).is_dir():
            return volume_path

    return None


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


def build_copy_plan(source_files: list[Path], target_dir: Path) -> list[tuple[Path, Path]]:
    """
    Return source and target pairs, failing early on filename collisions.

    Returns:
        Source and target file pairs for the import operation.

    Raises:
        FileExistsError: If duplicate source names or existing target files would collide.
    """
    planned_targets: set[Path] = set()
    copy_plan: list[tuple[Path, Path]] = []

    for source_path in source_files:
        target_path = target_dir / build_target_filename(target_dir, source_path)
        if target_path in planned_targets:
            raise FileExistsError(f"Duplicate filename on memory card: {source_path.name}")
        if target_path.exists():
            raise FileExistsError(f"Target file already exists: {target_path}")
        planned_targets.add(target_path)
        copy_plan.append((source_path, target_path))

    return copy_plan


def build_target_filename(target_dir: Path, source_path: Path) -> str:
    """Return the target filename prefixed with the workflow date directory name."""
    return f"{target_dir.name}_{source_path.name}"


def copy_files(copy_plan: list[tuple[Path, Path]]) -> None:
    """Copy source files into the target directory and log progress."""
    total_files = len(copy_plan)
    for index, (source_path, target_path) in enumerate(copy_plan, start=1):
        shutil.copy2(source_path, target_path)
        if should_log_copy_progress(index, total_files):
            LOGGER.info("copied %s/%s files", index, total_files)


def ensure_target_has_sufficient_space(
    copy_plan: list[tuple[Path, Path]],
    target_dir: Path,
) -> None:
    """
    Raise an error when the target volume cannot hold all planned source files.

    Raises:
        OSError: If the target volume lacks space for every planned source file.
    """
    required_bytes = sum(source_path.stat().st_size for source_path, _ in copy_plan)
    available_bytes = shutil.disk_usage(target_dir).free
    if required_bytes > available_bytes:
        raise OSError(
            "insufficient free space on target volume: "
            f"{required_bytes} bytes required, {available_bytes} bytes available"
        )


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
    """Run the memory-card ingest workflow for today's target folder."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    workflow_config = load_workflow_config()
    copy_config = load_memory_card_copy_config()
    target_dir = build_today_source_dir(camera_root=workflow_config.camera_root)
    run_memory_card_import(target_dir, config=copy_config)
