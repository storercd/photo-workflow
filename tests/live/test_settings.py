"""Tests for Live settings, arrival renaming, and feature discovery."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from photo_workflow.live.api import StageRegistry
from photo_workflow.live.features import load_features
from photo_workflow.live.ingest import rename_arrival
from photo_workflow.live.layout import LiveLayout, default_live_folder
from photo_workflow.live.settings import (
    LiveSettings,
    load_live_settings,
    save_live_settings,
    validate_live_settings,
)
from photo_workflow.live.stages import StageId


class MemorySecrets:
    """In-memory stand-in for the system keychain."""

    def __init__(self) -> None:
        """Start empty."""
        self.values: dict[str, str] = {}

    def get(self, name: str) -> str:
        """Return a stored secret, or an empty string."""
        return self.values.get(name, "")

    def set(self, name: str, value: str) -> None:
        """Store a secret; an empty value deletes it."""
        if value:
            self.values[name] = value
        else:
            self.values.pop(name, None)


def make_settings(tmp_path: Path) -> QSettings:
    """
    Create isolated INI-backed settings.

    Returns:
        Settings stored under `tmp_path`.
    """
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_default_live_folder_replaces_only_final_camera_folder_name() -> None:
    """The Live folder sits next to the camera folder."""
    assert default_live_folder(Path("/data/work/camera")) == Path("/data/work/live")


def test_settings_default_to_everything_off_and_autocrop_on() -> None:
    """Features are disabled by default; AutoCrop cannot be switched off."""
    settings = LiveSettings()

    assert not any(settings.is_enabled(stage) for stage in StageId if stage != StageId.AUTOCROP)
    assert settings.is_enabled(StageId.AUTOCROP)
    assert not settings.with_enabled(StageId.AUTOCROP, False).enabled


def test_settings_round_trip_keeps_secrets_out_of_plain_settings(tmp_path: Path) -> None:
    """Passwords and secret keys go to the secret store, not to the settings file."""
    store = make_settings(tmp_path)
    secrets = MemorySecrets()
    live = LiveSettings(
        folder="/tmp/live",
        ftp_username="camera",
        ftp_password="ftp-secret",
        r2_access_key_id="key-id",
        r2_secret_access_key="r2-secret",
        r2_bucket="bucket",
        autocrop_ratios=("1:1", "3:2"),
    ).with_enabled(StageId.DENOISE, True)

    save_live_settings(store, live, secrets)
    loaded = load_live_settings(make_settings(tmp_path), secrets)

    assert loaded == live
    assert "secret" not in (tmp_path / "settings.ini").read_text()


def test_invalid_stored_ratios_fall_back_to_defaults(tmp_path: Path) -> None:
    """Bad stored values do not break loading."""
    store = make_settings(tmp_path)
    store.setValue("live/autocrop_ratios", "nonsense")

    loaded = load_live_settings(store, MemorySecrets())

    assert loaded.autocrop_ratios == LiveSettings().autocrop_ratios


def test_validate_rejects_bad_port_and_empty_ratios() -> None:
    """Out-of-range ports and missing ratios are rejected."""
    with pytest.raises(ValueError, match="port"):
        validate_live_settings(LiveSettings(ftp_port=0))
    with pytest.raises(ValueError, match="aspect ratio"):
        validate_live_settings(LiveSettings(autocrop_ratios=()))


def test_rename_arrival_uses_card_naming_and_overwrites(tmp_path: Path) -> None:
    """Arrivals land in the date folder under a date-stamped name, replacing old files."""
    layout = LiveLayout(tmp_path)
    layout.incoming.mkdir()
    arrived = layout.incoming / "IMG_0001.CR3"
    arrived.write_bytes(b"new")
    arrived.with_suffix(".xmp").write_text("sidecar")
    captured = datetime(2025, 10, 31, 10, 15, 0, 120000)

    target = rename_arrival(layout, arrived, captured_at=captured)
    arrived.write_bytes(b"newer")
    again = rename_arrival(layout, arrived, captured_at=captured)

    assert target == again
    assert target.parent == tmp_path / "2025" / "10" / "20251031" / "input"
    assert target.name.startswith("20251031_101500")
    assert target.read_bytes() == b"newer"
    assert target.with_suffix(".xmp").read_text() == "sidecar"
    assert not arrived.exists()


def test_feature_discovery_runs_without_features_and_keeps_stubs() -> None:
    """With no feature modules every stage stays an unregistered no-op."""
    registry = StageRegistry()

    load_features(registry)

    assert not registry.is_registered(StageId.DENOISE)
