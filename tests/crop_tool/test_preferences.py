"""Tests for per-user crop tool preferences."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from photo_workflow.crop_tool.app import CropWindow
from photo_workflow.crop_tool.preferences import (
    CropPreferences,
    PreferencesDialog,
    load_preferences,
    parse_aspect_ratios,
    save_preferences,
)


def make_settings(tmp_path: Path) -> QSettings:
    """
    Create isolated INI-backed settings.

    Returns:
        Settings stored under `tmp_path`.
    """
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_parse_aspect_ratios_reads_ratio_labels() -> None:
    """Convert integer and decimal ratio strings into labeled crop ratios."""
    ratios = parse_aspect_ratios(["4:5", "16:9", "1.91:1"])

    assert [(ratio.width, ratio.height, ratio.label) for ratio in ratios] == [
        (4, 5, "4:5"),
        (16, 9, "16:9"),
        (191, 100, "1.91:1"),
    ]
    assert ratios[2].value == pytest.approx(1.91)


@pytest.mark.parametrize("value", ["bad", "1:0", "0:1", "-4:5"])
def test_parse_aspect_ratios_rejects_invalid_values(value: str) -> None:
    """Reject malformed or nonpositive aspect-ratio settings clearly."""
    with pytest.raises(ValueError, match="Invalid aspect ratio"):
        parse_aspect_ratios([value])


def test_load_preferences_without_settings_returns_defaults() -> None:
    """No settings object means defaults and no TOML dependence."""
    assert load_preferences(None) == CropPreferences()


def test_preferences_round_trip(tmp_path: Path) -> None:
    """Saved preferences load back unchanged."""
    settings = make_settings(tmp_path)
    expected = CropPreferences(
        aspect_ratios=("4:5", "3:2"),
        snap_tolerance=0.04,
        grow_on_ratio_change=False,
        preview_replaces_view=False,
        lightroom_catalog="~/catalog.lrcat",
    )
    save_preferences(settings, expected)

    assert load_preferences(settings, tmp_path / "missing.toml") == expected


def test_legacy_toml_is_imported_once(tmp_path: Path) -> None:
    """A legacy [crop_tool] section seeds settings once; later TOML edits are ignored."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text(
        '[crop_tool]\naspect_ratios = ["4:5", "3:2"]\nsnap_tolerance = 0.04\n'
        "grow_on_ratio_change = false\npreview_replaces_view = false\n"
        'lightroom_catalog = "~/x.lrcat"\n',
        encoding="utf-8",
    )
    settings = make_settings(tmp_path)

    imported = load_preferences(settings, config_path)
    config_path.write_text('[crop_tool]\naspect_ratios = ["1:1"]\n', encoding="utf-8")
    again = load_preferences(settings, config_path)

    assert imported.aspect_ratios == ("4:5", "3:2")
    assert imported.snap_tolerance == pytest.approx(0.04)
    assert imported.grow_on_ratio_change is False
    assert imported.preview_replaces_view is False
    assert imported.lightroom_catalog == "~/x.lrcat"
    assert again == imported


def test_invalid_legacy_toml_falls_back_to_defaults(tmp_path: Path) -> None:
    """Unusable legacy values are skipped rather than crashing."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text("[crop_tool]\nsnap_tolerance = 1.1\n", encoding="utf-8")

    assert load_preferences(make_settings(tmp_path), config_path) == CropPreferences()


def test_dialog_edits_and_validates(tmp_path: Path) -> None:
    """The dialog reflects preferences and returns edited values."""
    app = QApplication.instance() or QApplication([])
    dialog = PreferencesDialog(CropPreferences())
    dialog.ratios_edit.setText("4:5, 1.91:1")
    dialog.tolerance_spin.setValue(0.1)
    dialog.grow_check.setChecked(False)

    result = dialog.preferences()

    assert result.aspect_ratios == ("4:5", "1.91:1")
    assert result.snap_tolerance == pytest.approx(0.1)
    assert result.grow_on_ratio_change is False
    dialog.close()
    del app


def test_window_applies_preferences(tmp_path: Path) -> None:
    """Applying preferences updates ratios, hotkeys, and persists them."""
    app = QApplication.instance() or QApplication([])
    settings = make_settings(tmp_path)
    window = CropWindow(settings=settings)

    window._apply_preferences(CropPreferences(aspect_ratios=("4:5", "3:2"), snap_tolerance=0.2))

    assert [window.ratio_combo.itemText(i) for i in range(window.ratio_combo.count())] == [
        "Free",
        "4:5",
        "3:2",
    ]
    assert len(window._ratio_shortcuts) == 2
    assert window._snap_tolerance == pytest.approx(0.2)
    assert load_preferences(settings, tmp_path / "missing.toml").aspect_ratios == ("4:5", "3:2")
    window.close()
    del app
