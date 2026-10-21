"""Per-user crop tool preferences stored in `QSettings`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.config import DEFAULT_CONFIG_PATH
from photo_workflow.crop_tool.model import DEFAULT_ASPECT_RATIOS, AspectRatio

DEFAULT_SNAP_TOLERANCE = 0.05
LEGACY_IMPORTED_KEY = "preferences/legacy_toml_imported"


@dataclass(frozen=True)
class CropPreferences:
    """User-adjustable crop tool behavior."""

    aspect_ratios: tuple[str, ...] = tuple(ratio.label for ratio in DEFAULT_ASPECT_RATIOS)
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE
    grow_on_ratio_change: bool = True
    preview_replaces_view: bool = True
    lightroom_catalog: str = ""

    def ratios(self) -> tuple[AspectRatio, ...]:
        """Return the parsed aspect-ratio presets."""
        return parse_aspect_ratios(list(self.aspect_ratios))

    def catalog_path(self) -> Path | None:
        """Return the Lightroom catalog when it is configured and exists."""
        if not self.lightroom_catalog.strip():
            return None
        catalog = Path(self.lightroom_catalog.strip()).expanduser()
        return catalog if catalog.is_file() else None


def parse_aspect_ratios(values: list[str]) -> tuple[AspectRatio, ...]:
    """
    Parse integer or decimal ratio labels such as `4:5` and `1.91:1`.

    Returns:
        The parsed aspect ratios.

    Raises:
        ValueError: If a ratio is malformed or the list is empty.
    """
    ratios = []
    for value in values:
        try:
            width_text, height_text = value.split(":", maxsplit=1)
            ratio = Fraction(width_text) / Fraction(height_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"Invalid aspect ratio {value!r}; expected positive width:height"
            ) from error
        if ratio <= 0:
            raise ValueError(f"Invalid aspect ratio {value!r}; values must be positive")
        ratios.append(AspectRatio(ratio.numerator, ratio.denominator, value))
    if not ratios:
        raise ValueError("At least one aspect ratio is required")
    return tuple(ratios)


def split_ratio_text(text: str) -> list[str]:
    """
    Split a comma-separated ratio list, ignoring blanks.

    Returns:
        The trimmed ratio labels.
    """
    return [part.strip() for part in text.split(",") if part.strip()]


def validate_preferences(preferences: CropPreferences) -> None:
    """
    Check that preference values are usable.

    Raises:
        ValueError: If the ratios or snap tolerance are invalid.
    """
    preferences.ratios()
    if not 0 <= preferences.snap_tolerance <= 1:
        raise ValueError("Snap tolerance must be between 0 and 1")


def load_preferences(
    settings: QSettings | None,
    legacy_config_path: Path = DEFAULT_CONFIG_PATH,
) -> CropPreferences:
    """
    Load preferences, importing a legacy `[crop_tool]` TOML section once.

    Invalid stored values fall back to defaults. Without settings, defaults are
    returned and nothing is persisted.

    Returns:
        The effective preferences.
    """
    if settings is None:
        return CropPreferences()
    if not settings.value(LEGACY_IMPORTED_KEY, False, type=bool):
        legacy = _read_legacy_toml(legacy_config_path)
        if legacy is not None:
            save_preferences(settings, legacy)
        settings.setValue(LEGACY_IMPORTED_KEY, True)

    defaults = CropPreferences()
    ratio_text = settings.value("preferences/aspect_ratios", "", type=str)
    ratio_labels = tuple(split_ratio_text(ratio_text)) or defaults.aspect_ratios
    tolerance = settings.value("preferences/snap_tolerance", defaults.snap_tolerance, type=float)
    preferences = CropPreferences(
        aspect_ratios=ratio_labels,
        snap_tolerance=tolerance,
        grow_on_ratio_change=settings.value(
            "preferences/grow_on_ratio_change", defaults.grow_on_ratio_change, type=bool
        ),
        preview_replaces_view=settings.value(
            "preferences/preview_replaces_view", defaults.preview_replaces_view, type=bool
        ),
        lightroom_catalog=settings.value("preferences/lightroom_catalog", "", type=str),
    )
    try:
        validate_preferences(preferences)
    except ValueError:
        return defaults
    return preferences


def save_preferences(settings: QSettings, preferences: CropPreferences) -> None:
    """Persist preferences."""
    settings.setValue("preferences/aspect_ratios", ", ".join(preferences.aspect_ratios))
    settings.setValue("preferences/snap_tolerance", preferences.snap_tolerance)
    settings.setValue("preferences/grow_on_ratio_change", preferences.grow_on_ratio_change)
    settings.setValue("preferences/preview_replaces_view", preferences.preview_replaces_view)
    settings.setValue("preferences/lightroom_catalog", preferences.lightroom_catalog)
    settings.sync()


def _read_legacy_toml(config_path: Path) -> CropPreferences | None:
    """
    Read a legacy `[crop_tool]` section.

    Returns:
        The imported preferences, or `None` when absent or unusable.
    """
    if not config_path.is_file():
        return None
    try:
        with config_path.open("rb") as config_file:
            section = tomllib.load(config_file).get("crop_tool")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    if not section:
        return None
    defaults = CropPreferences()
    try:
        imported = CropPreferences(
            aspect_ratios=tuple(section.get("aspect_ratios", defaults.aspect_ratios)),
            snap_tolerance=float(section.get("snap_tolerance", defaults.snap_tolerance)),
            grow_on_ratio_change=bool(
                section.get("grow_on_ratio_change", defaults.grow_on_ratio_change)
            ),
            preview_replaces_view=bool(
                section.get("preview_replaces_view", defaults.preview_replaces_view)
            ),
            lightroom_catalog=str(section.get("lightroom_catalog", "")),
        )
        validate_preferences(imported)
    except (TypeError, ValueError):
        return None
    return imported


class PreferencesDialog(QDialog):
    """Modal editor for `CropPreferences`."""

    def __init__(self, preferences: CropPreferences, parent: QWidget | None = None) -> None:
        """Build the form populated from the given preferences."""
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.ratios_edit = QLineEdit()
        self.ratios_edit.setToolTip("Comma-separated width:height presets, e.g. 1:1, 4:5, 1.91:1")
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(0.0, 1.0)
        self.tolerance_spin.setDecimals(3)
        self.tolerance_spin.setSingleStep(0.01)
        self.tolerance_spin.setToolTip("Relative distance within which a crop snaps to a ratio")
        self.grow_check = QCheckBox("Grow crop (instead of shrinking) when switching ratio")
        self.replace_check = QCheckBox("Live Preview replaces the crop view")
        self.catalog_edit = QLineEdit()
        self.catalog_edit.setPlaceholderText("Optional: prefer Lightroom's cached previews")
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._browse_catalog)
        catalog_row = QHBoxLayout()
        catalog_row.addWidget(self.catalog_edit, 1)
        catalog_row.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("Aspect ratios", self.ratios_edit)
        form.addRow("Snap tolerance", self.tolerance_spin)
        form.addRow("", self.grow_check)
        form.addRow("", self.replace_check)
        form.addRow("Lightroom catalog", catalog_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.RestoreDefaults).clicked.connect(
            lambda: self._populate(CropPreferences())
        )
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        self._populate(preferences)

    def preferences(self) -> CropPreferences:
        """Return the values currently shown in the form."""
        return CropPreferences(
            aspect_ratios=tuple(split_ratio_text(self.ratios_edit.text())),
            snap_tolerance=self.tolerance_spin.value(),
            grow_on_ratio_change=self.grow_check.isChecked(),
            preview_replaces_view=self.replace_check.isChecked(),
            lightroom_catalog=self.catalog_edit.text().strip(),
        )

    def _populate(self, preferences: CropPreferences) -> None:
        """Fill the form controls."""
        self.ratios_edit.setText(", ".join(preferences.aspect_ratios))
        self.tolerance_spin.setValue(preferences.snap_tolerance)
        self.grow_check.setChecked(preferences.grow_on_ratio_change)
        self.replace_check.setChecked(preferences.preview_replaces_view)
        self.catalog_edit.setText(preferences.lightroom_catalog)

    def _browse_catalog(self) -> None:
        """Choose a Lightroom catalog file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Lightroom catalog", self.catalog_edit.text(), "Lightroom (*.lrcat)"
        )
        if path:
            self.catalog_edit.setText(path)

    def _accept_if_valid(self) -> None:
        """Accept only when the entered values validate."""
        try:
            validate_preferences(self.preferences())
        except ValueError as error:
            QMessageBox.warning(self, "Invalid preferences", str(error))
            return
        self.accept()
