"""Persistent Live settings; secrets are kept in the system keychain."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import keyring
from keyring.errors import KeyringError
from PySide6.QtCore import QSettings

from photo_workflow.editor.lighting import MAX_KELVIN, MIN_KELVIN
from photo_workflow.editor.model import parse_aspect_ratios
from photo_workflow.live.layout import LiveLayout, default_live_folder
from photo_workflow.live.stages import SPECS, STAGE_ORDER, StageId

LOGGER = logging.getLogger(__name__)
KEYCHAIN_SERVICE = "photo-workflow-live"
SECRET_FTP_PASSWORD = "ftp_password"  # noqa: S105
SECRET_R2_SECRET_KEY = "r2_secret_access_key"  # noqa: S105
DEFAULT_FTP_PORT = 2121
DEFAULT_AUTOCROP_RATIOS = ("1:1", "4:5", "1.91:1")
# Working name of the option that holds processing back while the user is active.
IDLE_PROCESSING_LABEL = "Idle processing"
DEFAULT_IDLE_DELAY_SECONDS = 3.0
MAX_IDLE_DELAY_SECONDS = 120.0


class SecretStore(Protocol):
    """Storage for credentials that must not be written to plain-text settings."""

    def get(self, name: str) -> str:
        """Return a stored secret, or an empty string."""

    def set(self, name: str, value: str) -> None:
        """Store a secret; an empty value deletes it."""


class KeychainSecretStore:
    """Secret store backed by the operating system keychain (macOS Keychain)."""

    def get(self, name: str) -> str:
        """Return a secret, or an empty string when absent or the keychain is unavailable."""
        try:
            return keyring.get_password(KEYCHAIN_SERVICE, name) or ""
        except KeyringError:
            LOGGER.warning("keychain unavailable; could not read %s", name)
            return ""

    def set(self, name: str, value: str) -> None:
        """Store or delete a secret; keychain failures are logged, not raised."""
        try:
            if value:
                keyring.set_password(KEYCHAIN_SERVICE, name, value)
            elif keyring.get_password(KEYCHAIN_SERVICE, name) is not None:
                keyring.delete_password(KEYCHAIN_SERVICE, name)
        except KeyringError:
            LOGGER.warning("keychain unavailable; could not store %s", name)


@dataclass(frozen=True)
class LiveSettings:
    """
    The Live configuration section.

    Every feature is off by default; `enabled` holds the stages the user switched on.
    AutoCrop is always on and never appears in `enabled`.
    """

    folder: str = ""
    enabled: frozenset[StageId] = field(default_factory=frozenset)
    ftp_port: int = DEFAULT_FTP_PORT
    ftp_username: str = ""
    ftp_password: str = ""
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    r2_min_rating: int = 0
    autocrop_ratios: tuple[str, ...] = DEFAULT_AUTOCROP_RATIOS
    auto_lighting: bool = True
    color_temperature: float | None = None
    autocrop_force: bool = False
    autocrop_ball: bool = False
    idle_processing: bool = False
    idle_delay_seconds: float = DEFAULT_IDLE_DELAY_SECONDS

    def folder_path(self, camera_root: Path) -> Path:
        """Return the configured Live folder, defaulting to the one next to the camera folder."""
        if self.folder.strip():
            return Path(self.folder.strip()).expanduser()
        return default_live_folder(camera_root)

    def layout(self, camera_root: Path) -> LiveLayout:
        """Return the layout of the Live folder."""
        return LiveLayout(self.folder_path(camera_root))

    def is_enabled(self, stage: StageId) -> bool:
        """Return whether a stage's feature is on; AutoCrop is always on."""
        return not SPECS[stage].toggleable or stage in self.enabled

    def with_enabled(self, stage: StageId, enabled: bool) -> LiveSettings:
        """Return settings with one feature toggled; AutoCrop cannot be changed."""
        if not SPECS[stage].toggleable:
            return self
        stages = self.enabled | {stage} if enabled else self.enabled - {stage}
        return replace(self, enabled=frozenset(stages))


def validate_live_settings(settings: LiveSettings) -> None:
    """
    Check Live settings are usable.

    Raises:
        ValueError: If the port, minimum rating, color temperature, idle delay or acceptable
            aspect ratios are invalid.
    """
    kelvin = settings.color_temperature
    if kelvin is not None and not MIN_KELVIN <= kelvin <= MAX_KELVIN:
        raise ValueError(
            f"Color temperature must be between {MIN_KELVIN:.0f} and {MAX_KELVIN:.0f} K"
        )
    if not 0 <= settings.r2_min_rating <= 5:
        raise ValueError("Cloud Sync minimum rating must be between 0 and 5 stars")
    if not 1 <= settings.ftp_port <= 65535:
        raise ValueError("FTP port must be between 1 and 65535")
    if not 0.5 <= settings.idle_delay_seconds <= MAX_IDLE_DELAY_SECONDS:
        raise ValueError(
            f"{IDLE_PROCESSING_LABEL} delay must be between 0.5 and "
            f"{MAX_IDLE_DELAY_SECONDS:.0f} seconds"
        )
    if not settings.autocrop_ratios:
        raise ValueError("Add at least one acceptable aspect ratio for automatic crop")
    parse_aspect_ratios(list(settings.autocrop_ratios))


def load_live_settings(
    settings: QSettings | None,
    secrets: SecretStore | None = None,
) -> LiveSettings:
    """
    Load Live settings; invalid stored values fall back to defaults.

    Returns:
        The effective settings, with credentials read from the secret store.
    """
    defaults = LiveSettings()
    if settings is None:
        return defaults
    store = secrets if secrets is not None else KeychainSecretStore()
    enabled_text = settings.value("live/enabled", "", type=str)
    enabled = frozenset(
        StageId(name)
        for name in enabled_text.split(",")
        if name in STAGE_ORDER and SPECS[StageId(name)].toggleable
    )
    ratio_text = settings.value("live/autocrop_ratios", "", type=str)
    ratios = tuple(part.strip() for part in ratio_text.split(",") if part.strip())
    loaded = LiveSettings(
        folder=settings.value("live/folder", "", type=str),
        enabled=enabled,
        ftp_port=settings.value("live/ftp_port", defaults.ftp_port, type=int),
        ftp_username=settings.value("live/ftp_username", "", type=str),
        ftp_password=store.get(SECRET_FTP_PASSWORD),
        r2_account_id=settings.value("live/r2_account_id", "", type=str),
        r2_access_key_id=settings.value("live/r2_access_key_id", "", type=str),
        r2_secret_access_key=store.get(SECRET_R2_SECRET_KEY),
        r2_bucket=settings.value("live/r2_bucket", "", type=str),
        r2_min_rating=settings.value("live/r2_min_rating", defaults.r2_min_rating, type=int),
        autocrop_ratios=ratios or defaults.autocrop_ratios,
        auto_lighting=settings.value("live/auto_lighting", True, type=bool),
        color_temperature=settings.value("live/color_temperature", 0.0, type=float) or None,
        autocrop_force=settings.value("live/autocrop_force", False, type=bool),
        autocrop_ball=settings.value("live/autocrop_ball", False, type=bool),
        idle_processing=settings.value("live/idle_processing", False, type=bool),
        idle_delay_seconds=settings.value(
            "live/idle_delay_seconds", defaults.idle_delay_seconds, type=float
        ),
    )
    try:
        validate_live_settings(loaded)
    except ValueError:
        return replace(
            loaded,
            ftp_port=defaults.ftp_port,
            r2_min_rating=defaults.r2_min_rating,
            autocrop_ratios=defaults.autocrop_ratios,
            idle_delay_seconds=defaults.idle_delay_seconds,
        )
    return loaded


def save_live_settings(
    settings: QSettings,
    live: LiveSettings,
    secrets: SecretStore | None = None,
) -> None:
    """Persist Live settings; credentials go to the secret store, never to `QSettings`."""
    store = secrets if secrets is not None else KeychainSecretStore()
    settings.setValue("live/folder", live.folder)
    settings.setValue(
        "live/enabled", ",".join(stage.value for stage in STAGE_ORDER if stage in live.enabled)
    )
    settings.setValue("live/ftp_port", live.ftp_port)
    settings.setValue("live/ftp_username", live.ftp_username)
    settings.setValue("live/r2_account_id", live.r2_account_id)
    settings.setValue("live/r2_access_key_id", live.r2_access_key_id)
    settings.setValue("live/r2_bucket", live.r2_bucket)
    settings.setValue("live/r2_min_rating", live.r2_min_rating)
    settings.setValue("live/autocrop_ratios", ", ".join(live.autocrop_ratios))
    settings.setValue("live/auto_lighting", live.auto_lighting)
    settings.setValue("live/color_temperature", live.color_temperature or 0.0)
    settings.setValue("live/autocrop_force", live.autocrop_force)
    settings.setValue("live/autocrop_ball", live.autocrop_ball)
    settings.setValue("live/idle_processing", live.idle_processing)
    settings.setValue("live/idle_delay_seconds", live.idle_delay_seconds)
    store.set(SECRET_FTP_PASSWORD, live.ftp_password)
    store.set(SECRET_R2_SECRET_KEY, live.r2_secret_access_key)
    settings.sync()
