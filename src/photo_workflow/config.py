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
DEFAULT_HALT_ON_INSUFFICIENT_SPACE = True
DEFAULT_IGNORED_CARD_EXTENSIONS = (".ctg", ".log", ".tmp", ".to3")
DEFAULT_MAX_DURATION_SECONDS = 10.0
DEFAULT_TRANSCRIPTION_MODEL = "mlx-community/whisper-tiny-mlx"
VALID_COPY_VERIFICATION_METHODS = {"basic", "crc32"}
DEFAULT_DASHBOARD_APPS = (
    {"name": "FastRawViewer", "app_name": "FastRawViewer"},
    {"name": "Photo Workflow Crop", "command": ["photo-workflow-crop"]},
    {"name": "Lightroom Classic", "app_name": "Adobe Lightroom Classic"},
    {"name": "Photoshop", "app_name": "Adobe Photoshop 2025"},
    {"name": "Aftershoot", "app_name": "Aftershoot"},
)


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
    halt_on_insufficient_space: bool = DEFAULT_HALT_ON_INSUFFICIENT_SPACE
    ignored_extensions: tuple[str, ...] = DEFAULT_IGNORED_CARD_EXTENSIONS


@dataclass(frozen=True)
class VideoNotesConfig:
    """Settings specific to video-note generation."""

    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL


@dataclass(frozen=True)
class DashboardAppLauncher:
    """One clickable application launcher shown on the dashboard."""

    name: str
    app_name: str | None = None
    command: tuple[str, ...] | None = None


@dataclass(frozen=True)
class DashboardConfig:
    """Settings specific to the workflow dashboard."""

    apps: tuple[DashboardAppLauncher, ...] = field(
        default_factory=lambda: tuple(
            DashboardAppLauncher(
                name=entry["name"],
                app_name=entry.get("app_name"),
                command=tuple(entry["command"]) if "command" in entry else None,
            )
            for entry in DEFAULT_DASHBOARD_APPS
        )
    )


@dataclass(frozen=True)
class AppConfig:
    """Complete workflow configuration loaded from TOML."""

    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    memory_card_copy: MemoryCardCopyConfig = field(default_factory=MemoryCardCopyConfig)
    video_notes: VideoNotesConfig = field(default_factory=VideoNotesConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """
    Load workflow settings from TOML, falling back to defaults.

    Returns:
        The fully resolved application configuration.

    Raises:
        ValueError: If the configured copy verification mode or ignored extensions are invalid.
    """
    if not config_path.exists():
        return AppConfig()

    with config_path.open("rb") as config_file:
        config_data = tomllib.load(config_file)

    workflow_config = config_data.get("workflow", {})
    memory_card_copy_config = config_data.get("memory_card_copy", {})
    video_notes_config = config_data.get("video_notes", {})
    dashboard_config = config_data.get("dashboard", {})

    copy_verification = memory_card_copy_config.get(
        "copy_verification",
        DEFAULT_COPY_VERIFICATION,
    )
    if copy_verification not in VALID_COPY_VERIFICATION_METHODS:
        raise ValueError(
            "memory_card_copy.copy_verification must be one of "
            f"{sorted(VALID_COPY_VERIFICATION_METHODS)}"
        )

    ignored_extensions = normalize_ignored_extensions(
        memory_card_copy_config.get(
            "ignored_extensions",
            list(DEFAULT_IGNORED_CARD_EXTENSIONS),
        )
    )
    halt_on_insufficient_space = memory_card_copy_config.get(
        "halt_on_insufficient_space",
        DEFAULT_HALT_ON_INSUFFICIENT_SPACE,
    )
    if not isinstance(halt_on_insufficient_space, bool):
        raise ValueError("memory_card_copy.halt_on_insufficient_space must be a boolean")

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
            halt_on_insufficient_space=halt_on_insufficient_space,
            ignored_extensions=ignored_extensions,
        ),
        video_notes=VideoNotesConfig(
            max_duration_seconds=float(
                video_notes_config.get(
                    "max_duration_seconds",
                    DEFAULT_MAX_DURATION_SECONDS,
                )
            ),
            transcription_model=str(
                video_notes_config.get(
                    "transcription_model",
                    DEFAULT_TRANSCRIPTION_MODEL,
                )
            ),
        ),
        dashboard=DashboardConfig(apps=parse_dashboard_apps(dashboard_config.get("apps"))),
    )


def parse_dashboard_apps(apps_config: object) -> tuple[DashboardAppLauncher, ...]:
    """
    Return dashboard app launchers parsed from TOML, falling back to defaults.

    Returns:
        A tuple of configured dashboard app launchers in declared order.

    Raises:
        ValueError: If an entry is malformed or specifies both or neither launch target.
    """
    if apps_config is None:
        return DashboardConfig().apps
    if not isinstance(apps_config, list):
        raise ValueError("dashboard.apps must be a TOML array of tables")

    launchers: list[DashboardAppLauncher] = []
    for entry in apps_config:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("each dashboard.apps entry requires a non-empty 'name'")

        app_name = entry.get("app_name")
        command = entry.get("command")
        if bool(app_name) == bool(command):
            raise ValueError(
                f"dashboard.apps entry {entry['name']!r} must set exactly one of "
                "'app_name' or 'command'"
            )
        if command is not None and not isinstance(command, list):
            raise ValueError(f"dashboard.apps entry {entry['name']!r} 'command' must be an array")

        launchers.append(
            DashboardAppLauncher(
                name=str(entry["name"]),
                app_name=str(app_name) if app_name else None,
                command=tuple(str(part) for part in command) if command else None,
            )
        )

    return tuple(launchers)


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


def load_dashboard_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DashboardConfig:
    """Return the dashboard configuration."""
    return load_config(config_path).dashboard


def build_today_source_dir(
    *,
    today: date | None = None,
    camera_root: Path | None = None,
) -> Path:
    """Return the YYYY/MM/YYYYMMDD source directory for the current workflow run."""
    run_date = today or date.today()
    active_camera_root = camera_root or load_workflow_config().camera_root
    return (
        active_camera_root
        / run_date.strftime("%Y")
        / run_date.strftime("%m")
        / run_date.strftime("%Y%m%d")
    )


def normalize_ignored_extensions(extensions: object) -> tuple[str, ...]:
    """
    Return normalized ignored file extensions for card ingest.

    Returns:
        A normalized, de-duplicated tuple of lowercase file extensions.

    Raises:
        ValueError: If the configured extensions are not a list of non-empty strings.
    """
    if not isinstance(extensions, list):
        raise ValueError("memory_card_copy.ignored_extensions must be a TOML array")

    normalized_extensions: list[str] = []
    for extension in extensions:
        if not isinstance(extension, str):
            raise ValueError("memory_card_copy.ignored_extensions entries must be strings")

        normalized_extension = extension.strip().lower()
        if not normalized_extension:
            raise ValueError("memory_card_copy.ignored_extensions cannot contain blanks")
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"
        normalized_extensions.append(normalized_extension)

    return tuple(dict.fromkeys(normalized_extensions))
