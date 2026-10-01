"""Dashboard configuration parsing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from photo_workflow import config as app_config


def test_parse_dashboard_apps_returns_defaults_when_unset() -> None:
    """Verify default dashboard launchers are used when no apps are configured."""
    launchers = app_config.parse_dashboard_apps(None)

    assert launchers == app_config.DashboardConfig().apps
    assert launchers[0] == app_config.DashboardAppLauncher(
        name="FastRawViewer", app_name="FastRawViewer"
    )


def test_parse_dashboard_apps_parses_app_name_entries() -> None:
    """Verify an app_name launcher entry is parsed correctly."""
    launchers = app_config.parse_dashboard_apps(
        [{"name": "Aftershoot", "app_name": "Aftershoot"}]
    )

    assert launchers == (app_config.DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot"),)


def test_parse_dashboard_apps_parses_command_entries() -> None:
    """Verify a command launcher entry is parsed correctly."""
    launchers = app_config.parse_dashboard_apps(
        [{"name": "Photo Workflow Crop", "command": ["photo-workflow-crop"]}]
    )

    assert launchers == (
        app_config.DashboardAppLauncher(
            name="Photo Workflow Crop", command=("photo-workflow-crop",)
        ),
    )


def test_parse_dashboard_apps_rejects_non_list() -> None:
    """Verify a non-array dashboard.apps value raises a clear error."""
    with pytest.raises(ValueError, match="must be a TOML array"):
        app_config.parse_dashboard_apps({"name": "Aftershoot"})


def test_parse_dashboard_apps_rejects_missing_name() -> None:
    """Verify an entry without a name raises a clear error."""
    with pytest.raises(ValueError, match="requires a non-empty 'name'"):
        app_config.parse_dashboard_apps([{"app_name": "Aftershoot"}])


def test_parse_dashboard_apps_rejects_both_app_name_and_command() -> None:
    """Verify an entry specifying both launch targets raises a clear error."""
    with pytest.raises(ValueError, match="must set exactly one of"):
        app_config.parse_dashboard_apps(
            [{"name": "Aftershoot", "app_name": "Aftershoot", "command": ["aftershoot"]}]
        )


def test_parse_dashboard_apps_rejects_neither_app_name_nor_command() -> None:
    """Verify an entry specifying no launch target raises a clear error."""
    with pytest.raises(ValueError, match="must set exactly one of"):
        app_config.parse_dashboard_apps([{"name": "Aftershoot"}])


def test_parse_dashboard_apps_rejects_non_array_command() -> None:
    """Verify a non-array command value raises a clear error."""
    with pytest.raises(ValueError, match="'command' must be an array"):
        app_config.parse_dashboard_apps([{"name": "Aftershoot", "command": "aftershoot"}])


def test_load_config_parses_dashboard_apps_table(tmp_path: Path) -> None:
    """Verify load_config wires dashboard.apps entries into AppConfig."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text(
        "[[dashboard.apps]]\n"
        'name = "Aftershoot"\n'
        'app_name = "Aftershoot"\n'
    )

    config = app_config.load_config(config_path)

    assert config.dashboard.apps == (
        app_config.DashboardAppLauncher(name="Aftershoot", app_name="Aftershoot"),
    )


def test_load_dashboard_config_returns_defaults_without_config_file(tmp_path: Path) -> None:
    """Verify load_dashboard_config falls back to defaults when no file exists."""
    dashboard_config = app_config.load_dashboard_config(tmp_path / "missing.toml")

    assert dashboard_config == app_config.DashboardConfig()
