"""Configuration helpers for photo workflow processes."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("photo-workflow.toml")
DEFAULT_CAMERA_ROOT = Path("/Users/christopherstorer/working/camera")
DEFAULT_CARD_MOUNT_ROOT = Path("/Volumes")
DEFAULT_LOW_DISK_WARNING_GB = 30.0
DEFAULT_LOW_DISK_WARNING_PERCENT = 5.0
DEFAULT_COPY_VERIFICATION = "basic"
DEFAULT_MAX_DURATION_SECONDS = 10.0
VALID_COPY_VERIFICATION_METHODS = {"basic", "crc32"}


@dataclass(frozen=True)
class WorkflowConfig:
    """Shared workflow settings used across processes."""

    camera_root: Path = DEFAULT_CAMERA_ROOT


@dataclass(frozen=True)
class MemoryCardCopyConfig:
    """Settings specific to memory-card ingest."""

    card_mount_root: Path = DEFAULT_CARD_MOUNT_ROOT
    low_disk_warning_gb: float = DEFAULT_LOW_DISK_WARNING_GB
    low_disk_warning_percent: float = DEFAULT_LOW_DISK_WARNING_PERCENT
    copy_verification: str = DEFAULT_COPY_VERIFICATION


@dataclass(frozen=True)
class VideoNotesConfig:
    """Settings specific to video-note generation."""

    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS


@dataclass(frozen=True)
class AppConfig:
    """Complete workflow configuration loaded from TOML."""

    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    memory_card_copy: MemoryCardCopyConfig = field(default_factory=MemoryCardCopyConfig)
    video_notes: VideoNotesConfig = field(default_factory=VideoNotesConfig)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load workflow settings from TOML, falling back to defaults."""

    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as config_file:
        config_data = tomllib.load(config_file)

    workflow_config = config_data.get("workflow", {})
    memory_card_copy_config = config_data.get("memory_card_copy", {})
    video_notes_config = config_data.get("video_notes", {})

    copy_verification = memory_card_copy_config.get(
        "copy_verification",
        DEFAULT_COPY_VERIFICATION,
    )
    if copy_verification not in VALID_COPY_VERIFICATION_METHODS:
        raise ValueError(
            "memory_card_copy.copy_verification must be one of "
            f"{sorted(VALID_COPY_VERIFICATION_METHODS)}"
        )

    return AppConfig(
        workflow=WorkflowConfig(
            camera_root=Path(
                workflow_config.get("camera_root", DEFAULT_CAMERA_ROOT)
            ).expanduser()
        ),
        memory_card_copy=MemoryCardCopyConfig(
            card_mount_root=Path(
                memory_card_copy_config.get("card_mount_root", DEFAULT_CARD_MOUNT_ROOT)
            ).expanduser(),
            low_disk_warning_gb=float(
                memory_card_copy_config.get(
                    "low_disk_warning_gb",
                    DEFAULT_LOW_DISK_WARNING_GB,
                )
            ),
            low_disk_warning_percent=float(
                memory_card_copy_config.get(
                    "low_disk_warning_percent",
                    DEFAULT_LOW_DISK_WARNING_PERCENT,
                )
            ),
            copy_verification=copy_verification,
        ),
        video_notes=VideoNotesConfig(
            max_duration_seconds=float(
                video_notes_config.get(
                    "max_duration_seconds",
                    DEFAULT_MAX_DURATION_SECONDS,
                )
            )
        ),
    )


def load_workflow_config(config_path: Path = DEFAULT_CONFIG_PATH) -> WorkflowConfig:
    """Return the shared workflow configuration."""

    return load_config(config_path).workflow


def load_memory_card_copy_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> MemoryCardCopyConfig:
    """Return the memory-card copy configuration."""

    return load_config(config_path).memory_card_copy


def load_video_notes_config(config_path: Path = DEFAULT_CONFIG_PATH) -> VideoNotesConfig:
    """Return the video-notes configuration."""

    return load_config(config_path).video_notes


def build_today_source_dir(
    *,
    today: date | None = None,
    camera_root: Path | None = None,
) -> Path:
    """Return the dated source directory for the current workflow run."""

    run_date = today or date.today()
    active_camera_root = camera_root or load_workflow_config().camera_root
    return active_camera_root / run_date.strftime("%Y%m%d")
