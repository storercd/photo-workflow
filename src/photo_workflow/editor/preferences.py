"""Per-user editor preferences stored in `QSettings`."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent, QKeySequence, QMouseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.config import DEFAULT_CONFIG_PATH
from photo_workflow.editor.model import (
    DEFAULT_ASPECT_RATIOS,
    AspectRatio,
    parse_aspect_ratios,
)
from photo_workflow.live.settings import LiveSettings, validate_live_settings

DEFAULT_SNAP_TOLERANCE = 0.05
LEGACY_IMPORTED_KEY = "preferences/legacy_toml_imported"
MOUSE_PREFIX = "Mouse:"
MOUSE_BUTTON_NAMES: dict[Qt.MouseButton, str] = {
    Qt.MouseButton.MiddleButton: "Middle",
    Qt.MouseButton.RightButton: "Right",
    Qt.MouseButton.BackButton: "Back",
    Qt.MouseButton.ForwardButton: "Forward",
    Qt.MouseButton.TaskButton: "Task",
    Qt.MouseButton.ExtraButton4: "Extra4",
    Qt.MouseButton.ExtraButton5: "Extra5",
    Qt.MouseButton.ExtraButton6: "Extra6",
}
HOTKEY_ACTIONS: dict[str, str] = {
    "previous_image": "Previous image",
    "next_image": "Next image",
    "auto_level": "Auto-level",
    "toggle_lock": "Toggle lock",
    "toggle_snap": "Toggle snap",
    "live_preview": "Live Preview",
    "show_horizon": "Show horizon",
    "ball_detection": "Detect ball",
    "auto_light": "Auto lighting",
    "flag": "Flag photo",
    "copy_previous_crop": "Copy previous crop",
}


def mouse_binding(button: Qt.MouseButton) -> str | None:
    """Return the stored binding text for a bindable mouse button."""
    name = MOUSE_BUTTON_NAMES.get(button)
    return f"{MOUSE_PREFIX}{name}" if name else None


def is_mouse_binding(binding: str) -> bool:
    """Return whether a binding refers to a mouse button."""
    return binding.startswith(MOUSE_PREFIX)


def mouse_button_for_binding(binding: str) -> Qt.MouseButton | None:
    """Return the mouse button for a `Mouse:<name>` binding."""
    name = binding.removeprefix(MOUSE_PREFIX)
    return next((button for button, n in MOUSE_BUTTON_NAMES.items() if n == name), None)


def normalize_hotkeys(hotkeys: dict[str, str]) -> dict[str, str]:
    """
    Drop unknown actions and blank or unparseable bindings.

    Returns:
        The usable bindings keyed by action id.
    """
    result = {}
    for action, raw in hotkeys.items():
        binding = raw.strip()
        if action not in HOTKEY_ACTIONS or not binding:
            continue
        if is_mouse_binding(binding):
            if mouse_button_for_binding(binding) is None:
                continue
        elif QKeySequence(binding).isEmpty():
            continue
        result[action] = binding
    return result


@dataclass(frozen=True)
class CropPreferences:
    """User-adjustable editor behavior."""

    aspect_ratios: tuple[str, ...] = tuple(ratio.label for ratio in DEFAULT_ASPECT_RATIOS)
    snap_tolerance: float = DEFAULT_SNAP_TOLERANCE
    grow_on_ratio_change: bool = True
    preview_replaces_view: bool = True
    lightroom_catalog: str = ""
    hotkeys: dict[str, str] = field(default_factory=dict)

    def ratios(self) -> tuple[AspectRatio, ...]:
        """Return the parsed aspect-ratio presets."""
        return parse_aspect_ratios(list(self.aspect_ratios))

    def catalog_path(self) -> Path | None:
        """Return the Lightroom catalog when it is configured and exists."""
        if not self.lightroom_catalog.strip():
            return None
        catalog = Path(self.lightroom_catalog.strip()).expanduser()
        return catalog if catalog.is_file() else None


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
    seen: dict[str, str] = {}
    for action, binding in preferences.hotkeys.items():
        if binding in seen:
            raise ValueError(
                f"{binding!r} is assigned to both {HOTKEY_ACTIONS[seen[binding]]!r} "
                f"and {HOTKEY_ACTIONS[action]!r}"
            )
        seen[binding] = action


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
            "preferences/preview_replaces_view",
            defaults.preview_replaces_view,
            type=bool,
        ),
        lightroom_catalog=settings.value("preferences/lightroom_catalog", "", type=str),
        hotkeys=normalize_hotkeys({
            action: settings.value(f"preferences/hotkey/{action}", "", type=str)
            for action in HOTKEY_ACTIONS
        }),
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
    for action in HOTKEY_ACTIONS:
        settings.setValue(f"preferences/hotkey/{action}", preferences.hotkeys.get(action, ""))
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


class HotkeyEdit(QPushButton):
    """Button that records the next key press or extra mouse button as a binding."""

    def __init__(self, binding: str = "", parent: QWidget | None = None) -> None:
        """Create the editor showing an optional initial binding."""
        super().__init__(parent)
        self._binding = binding
        self._capturing = False
        self.clicked.connect(self._start_capture)
        self._refresh()

    def binding(self) -> str:
        """Return the recorded binding, or an empty string."""
        return self._binding

    def set_binding(self, binding: str) -> None:
        """Set the binding programmatically."""
        self._binding = binding
        self._capturing = False
        self._refresh()

    def _start_capture(self) -> None:
        self._capturing = True
        self.setText("Press a key or mouse button… (Esc cancels, Delete clears)")
        self.setFocus()

    def _refresh(self) -> None:
        text = self._binding
        if text and not is_mouse_binding(text):
            text = QKeySequence(text).toString(QKeySequence.SequenceFormat.NativeText)
        self.setText(text or "None")

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Record a key combination while capturing."""
        if not self._capturing:
            super().keyPressEvent(event)
            return
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._capturing = False
            self._refresh()
        elif key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.set_binding("")
        elif key not in (
            Qt.Key.Key_Shift,
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Meta,
        ):
            self.set_binding(QKeySequence(event.keyCombination()).toString())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Record a non-primary mouse button while capturing."""
        if self._capturing:
            binding = mouse_binding(event.button())
            if binding:
                self.set_binding(binding)
                event.accept()
                return
        super().mousePressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:  # noqa: N802
        """Cancel capturing when focus leaves."""
        if self._capturing:
            self._capturing = False
            self._refresh()
        super().focusOutEvent(event)


class PreferencesDialog(QDialog):
    """Modal editor for `CropPreferences`."""

    def __init__(
        self,
        preferences: CropPreferences,
        parent: QWidget | None = None,
        live: LiveSettings | None = None,
        default_live_folder: str = "",
    ) -> None:
        """Build the form populated from the given preferences (and Live settings, if given)."""
        super().__init__(parent)
        self._live = live
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

        self.hotkey_edits = {action: HotkeyEdit() for action in HOTKEY_ACTIONS}
        hotkey_form = QFormLayout()
        for action, label in HOTKEY_ACTIONS.items():
            hotkey_form.addRow(label, self.hotkey_edits[action])
        hotkey_box = QGroupBox("Alternate hotkeys (arrow keys and defaults always work)")
        hotkey_box.setLayout(hotkey_form)

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
        layout.addWidget(hotkey_box)
        if live is not None:
            layout.addWidget(self._build_live_box(live, default_live_folder))
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        self._populate(preferences)

    def _build_live_box(self, live: LiveSettings, default_folder: str) -> QGroupBox:
        """
        Build the Live section.

        Returns:
            The group box holding the Live fields.
        """
        self.live_folder_edit = QLineEdit(live.folder)
        self.live_folder_edit.setPlaceholderText(default_folder or "Next to the camera folder")
        folder_button = QPushButton("Browse…")
        folder_button.clicked.connect(self._browse_live_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.live_folder_edit, 1)
        folder_row.addWidget(folder_button)
        self.ftp_port_spin = QSpinBox()
        self.ftp_port_spin.setRange(1, 65535)
        self.ftp_port_spin.setValue(live.ftp_port)
        self.ftp_user_edit = QLineEdit(live.ftp_username)
        self.ftp_password_edit = self._secret_edit(live.ftp_password)
        self.r2_account_edit = QLineEdit(live.r2_account_id)
        self.r2_key_edit = QLineEdit(live.r2_access_key_id)
        self.r2_secret_edit = self._secret_edit(live.r2_secret_access_key)
        self.r2_bucket_edit = QLineEdit(live.r2_bucket)
        self.r2_min_rating_spin = QSpinBox()
        self.r2_min_rating_spin.setRange(0, 5)
        self.r2_min_rating_spin.setValue(live.r2_min_rating)
        self.r2_min_rating_spin.setToolTip(
            "Cloud Sync uploads images rated at least this many stars and removes the rest"
        )
        self.autocrop_ratios_edit = QLineEdit(", ".join(live.autocrop_ratios))
        self.autocrop_ratios_edit.setToolTip("Aspect ratios automatic crop may choose from")
        form = QFormLayout()
        form.addRow("Live folder", folder_row)
        form.addRow("FTP port", self.ftp_port_spin)
        form.addRow("FTP username", self.ftp_user_edit)
        form.addRow("FTP password", self.ftp_password_edit)
        form.addRow("R2 account ID", self.r2_account_edit)
        form.addRow("R2 access key ID", self.r2_key_edit)
        form.addRow("R2 secret access key", self.r2_secret_edit)
        form.addRow("R2 bucket", self.r2_bucket_edit)
        form.addRow("Cloud Sync minimum stars", self.r2_min_rating_spin)
        form.addRow("Auto-crop aspect ratios", self.autocrop_ratios_edit)
        box = QGroupBox("Live (credentials are stored in the system keychain)")
        box.setLayout(form)
        return box

    @staticmethod
    def _secret_edit(value: str) -> QLineEdit:
        """
        Create a masked line edit.

        Returns:
            The edit holding `value`.
        """
        edit = QLineEdit(value)
        edit.setEchoMode(QLineEdit.EchoMode.Password)
        return edit

    def live_settings(self) -> LiveSettings | None:
        """Return the Live values shown in the form, or `None` when Live was not offered."""
        if self._live is None:
            return None
        return replace(
            self._live,
            folder=self.live_folder_edit.text().strip(),
            ftp_port=self.ftp_port_spin.value(),
            ftp_username=self.ftp_user_edit.text().strip(),
            ftp_password=self.ftp_password_edit.text(),
            r2_account_id=self.r2_account_edit.text().strip(),
            r2_access_key_id=self.r2_key_edit.text().strip(),
            r2_secret_access_key=self.r2_secret_edit.text(),
            r2_bucket=self.r2_bucket_edit.text().strip(),
            r2_min_rating=self.r2_min_rating_spin.value(),
            autocrop_ratios=tuple(split_ratio_text(self.autocrop_ratios_edit.text())),
        )

    def _browse_live_folder(self) -> None:
        """Choose the Live folder."""
        path = QFileDialog.getExistingDirectory(
            self, "Choose Live folder", self.live_folder_edit.text()
        )
        if path:
            self.live_folder_edit.setText(path)

    def preferences(self) -> CropPreferences:
        """Return the values currently shown in the form."""
        return CropPreferences(
            aspect_ratios=tuple(split_ratio_text(self.ratios_edit.text())),
            snap_tolerance=self.tolerance_spin.value(),
            grow_on_ratio_change=self.grow_check.isChecked(),
            preview_replaces_view=self.replace_check.isChecked(),
            lightroom_catalog=self.catalog_edit.text().strip(),
            hotkeys={
                action: edit.binding()
                for action, edit in self.hotkey_edits.items()
                if edit.binding()
            },
        )

    def _populate(self, preferences: CropPreferences) -> None:
        """Fill the form controls."""
        self.ratios_edit.setText(", ".join(preferences.aspect_ratios))
        self.tolerance_spin.setValue(preferences.snap_tolerance)
        self.grow_check.setChecked(preferences.grow_on_ratio_change)
        self.replace_check.setChecked(preferences.preview_replaces_view)
        self.catalog_edit.setText(preferences.lightroom_catalog)
        for action, edit in self.hotkey_edits.items():
            edit.set_binding(preferences.hotkeys.get(action, ""))

    def _browse_catalog(self) -> None:
        """Choose a Lightroom catalog file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Lightroom catalog",
            self.catalog_edit.text(),
            "Lightroom (*.lrcat)",
        )
        if path:
            self.catalog_edit.setText(path)

    def _accept_if_valid(self) -> None:
        """Accept only when the entered values validate."""
        try:
            validate_preferences(self.preferences())
            live = self.live_settings()
            if live is not None:
                validate_live_settings(live)
        except ValueError as error:
            QMessageBox.warning(self, "Invalid preferences", str(error))
            return
        self.accept()
