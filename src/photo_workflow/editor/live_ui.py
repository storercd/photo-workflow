"""Qt side of Live mode: pipeline controller and the status panel."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.live.api import StageRegistry
from photo_workflow.live.features import load_features
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.pipeline import LivePipeline, PipelineSnapshot
from photo_workflow.live.settings import IDLE_PROCESSING_LABEL, LiveSettings
from photo_workflow.live.stages import SPECS, STAGE_ORDER, StageId

LOGGER = logging.getLogger(__name__)
POLL_INTERVAL_MS = 1000
BLINK_INTERVAL_MS = 600
MAX_WARNING_MENU_TEXT = 90
ERROR_FLAG_STYLE = "color: #ff3b30; font-weight: bold;"
BLINK_STYLE = "background: #e5bd48; color: #111; font-weight: bold; padding: 0 6px;"
COUNT_STYLE = "font-weight: bold; padding: 0 6px;"
HELD_STYLE = "color: #8c9299; font-style: italic;"
IDLE_POLL_MS = 250
ACTIVITY_EVENTS = frozenset({
    QEvent.Type.MouseButtonPress,
    QEvent.Type.MouseButtonDblClick,
    QEvent.Type.KeyPress,
    QEvent.Type.Wheel,
    QEvent.Type.TabletPress,
    QEvent.Type.TouchBegin,
})


class _CleanTask(QRunnable):
    """Wipe the Live folder off the UI thread."""

    def __init__(self, controller: LiveController) -> None:
        """Remember the controller to clean and report to."""
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        """Clean the pipeline and tell the UI when done."""
        error = ""
        try:
            self._controller.pipeline.clean()
        except Exception as exception:
            LOGGER.exception("Live clean failed")
            error = str(exception)
        self._controller.cleaned.emit(error)


class IdleMonitor(QObject):
    """
    Holds the pipeline while the user is active and releases it after a quiet spell.

    Mouse movement is found by polling the cursor position (so it is seen anywhere on
    screen); clicks, key presses and scrolling are seen by an application-wide event filter.
    """

    def __init__(
        self,
        pipeline: LivePipeline,
        parent: QObject | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        cursor: Callable[[], QPoint] = QCursor.pos,
    ) -> None:
        """Watch for activity on behalf of a pipeline; call `configure` to switch it on."""
        super().__init__(parent)
        self._pipeline = pipeline
        self._clock = clock
        self._cursor = cursor
        self._delay = 0.0
        self._enabled = False
        self._last_position = cursor()
        self._last_activity = clock()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check)

    @property
    def enabled(self) -> bool:
        """Whether idle processing is switched on."""
        return self._enabled

    def configure(self, settings: LiveSettings) -> None:
        """Apply the idle option; switching it on starts held, switching it off releases."""
        self._delay = settings.idle_delay_seconds
        if settings.idle_processing == self._enabled:
            return
        self._enabled = settings.idle_processing
        app = QApplication.instance()
        if self._enabled:
            self.note_activity()
            self._pipeline.set_held(True)
            self._timer.start(IDLE_POLL_MS)
            if app is not None:
                app.installEventFilter(self)
        else:
            self._timer.stop()
            if app is not None:
                app.removeEventFilter(self)
            self._pipeline.set_held(False)

    def note_activity(self) -> None:
        """Record that the user just did something."""
        self._last_activity = self._clock()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        """
        Treat clicks, key presses and scrolling as activity.

        Returns:
            `False`, so the event is never consumed.
        """
        if event.type() in ACTIVITY_EVENTS:
            self.note_activity()
            if not self._pipeline.held:
                self._pipeline.set_held(True)
        return False

    def check(self) -> None:
        """Compare the cursor and the clock with the last activity and hold or release."""
        position = self._cursor()
        if position != self._last_position:
            self._last_position = position
            self.note_activity()
        idle = self._clock() - self._last_activity >= self._delay
        self._pipeline.set_held(not idle)


class LiveController(QObject):
    """Owns a `LivePipeline`, forwards its changes to Qt, and polls for arrivals."""

    changed = Signal()
    cleaned = Signal(str)

    def __init__(
        self,
        layout: LiveLayout,
        settings: LiveSettings,
        parent: QObject | None = None,
        *,
        registry: StageRegistry | None = None,
    ) -> None:
        """Build the pipeline, load registered features, and start polling."""
        super().__init__(parent)
        self.registry = registry or StageRegistry()
        load_features(self.registry)
        self.pipeline = LivePipeline(layout, settings, self.registry)
        self.pipeline.add_listener(self.changed.emit)
        self.idle = IdleMonitor(self.pipeline, self)
        self.blink_on = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self.pipeline.poll_incoming)
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)

    def start(self) -> None:
        """Scan existing photos, start enabled features, and begin polling."""
        self.idle.configure(self.pipeline.settings)
        self.pipeline.scan()
        self.pipeline.start_enabled_features()
        self._poll_timer.start(POLL_INTERVAL_MS)
        self._blink_timer.start(BLINK_INTERVAL_MS)

    def set_settings(self, settings: LiveSettings) -> None:
        """Apply new settings to the pipeline and the idle option."""
        self.pipeline.set_settings(settings)
        self.idle.configure(settings)

    def stop(self) -> None:
        """Stop polling and shut the pipeline down."""
        self.idle.configure(replace(self.pipeline.settings, idle_processing=False))
        self._poll_timer.stop()
        self._blink_timer.stop()
        self._pool.waitForDone()
        self.pipeline.shutdown()

    def clean(self) -> None:
        """Wipe the Live folder in the background; `cleaned` fires when finished."""
        self._poll_timer.stop()
        self._pool.start(_CleanTask(self))

    def resume_polling(self) -> None:
        """Restart polling after a clean."""
        self._poll_timer.start(POLL_INTERVAL_MS)

    def _blink(self) -> None:
        self.blink_on = not self.blink_on
        self.changed.emit()


class LiveStatusPanel(QFrame):
    """Per-stage counts, feature toggles, warnings, and the Clean action."""

    stage_toggled = Signal(object, bool)
    idle_toggled = Signal(bool)
    follow_toggled = Signal(bool)
    warning_selected = Signal(int, str)
    warnings_cleared = Signal()
    clean_requested = Signal()
    bypass_folder_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build one row per stage."""
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFixedWidth(230)
        self._warnings: list[tuple[int, str, str]] = []
        self.warnings_button = QToolButton()
        self.warnings_button.setText("⚠ 0")
        self.warnings_button.setToolTip("Warnings")
        self.warnings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.warnings_menu = QMenu(self.warnings_button)
        self.warnings_button.setMenu(self.warnings_menu)
        self.clean_button = QPushButton("Clean…")
        self.clean_button.setToolTip("Delete everything in the Live folder, including input")
        self.clean_button.clicked.connect(self._confirm_clean)
        header = QHBoxLayout()
        title = QLabel("Live")
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.warnings_button)
        header.addWidget(self.clean_button)

        grid = QGridLayout()
        self.toggles: dict[StageId, QCheckBox] = {}
        self.count_labels: dict[StageId, QLabel] = {}
        self.error_labels: dict[StageId, QToolButton] = {}
        self._halted: dict[StageId, str] = {}
        self.info_labels: dict[StageId, QLabel] = {}
        for row, stage in enumerate(STAGE_ORDER):
            spec = SPECS[stage]
            toggle = QCheckBox(spec.label)
            toggle.setToolTip(f"Run {spec.label} on arriving photos")
            toggle.toggled.connect(lambda checked, s=stage: self.stage_toggled.emit(s, checked))
            error = QToolButton()
            error.setText("!")
            error.setAutoRaise(True)
            error.setCursor(Qt.CursorShape.PointingHandCursor)
            error.setStyleSheet(ERROR_FLAG_STYLE)
            error.setVisible(False)
            error.clicked.connect(lambda _=False, s=stage: self._show_error(s))
            count = QLabel("0")
            count.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            count.setStyleSheet(COUNT_STYLE)
            info = QLabel()
            info.setWordWrap(True)
            info.setStyleSheet("color: #8c9299; font-size: 11px; padding-left: 20px;")
            info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            info.setVisible(False)
            grid.addWidget(toggle, row * 2, 0)
            grid.addWidget(error, row * 2, 1)
            grid.addWidget(count, row * 2, 2)
            grid.addWidget(info, row * 2 + 1, 0, 1, 3)
            self.info_labels[stage] = info
            self.toggles[stage] = toggle
            self.error_labels[stage] = error
            self.count_labels[stage] = count
        self.bypass_label = QLabel("Bypass")
        self.bypass_label.setStyleSheet("font-weight: bold;")
        self.bypass_counts = QLabel("↓ 0  ↑ 0")
        self.bypass_counts.setToolTip(
            "↓ photos waiting in bypass-input   ↑ photos sent through bypass-output"
        )
        self.bypass_input_button = QToolButton()
        self.bypass_input_button.setText("In")
        self.bypass_input_button.setToolTip("Open bypass-input (select these files in Lightroom)")
        self.bypass_input_button.clicked.connect(
            lambda: self.bypass_folder_requested.emit("input")
        )
        self.bypass_output_button = QToolButton()
        self.bypass_output_button.setText("Out")
        self.bypass_output_button.setToolTip("Open bypass-output (inspect or clean exports)")
        self.bypass_output_button.clicked.connect(
            lambda: self.bypass_folder_requested.emit("output")
        )
        bypass_row = QHBoxLayout()
        bypass_row.addWidget(self.bypass_label)
        bypass_row.addWidget(self.bypass_counts)
        bypass_row.addStretch(1)
        bypass_row.addWidget(self.bypass_input_button)
        bypass_row.addWidget(self.bypass_output_button)
        self.idle_button = QPushButton(IDLE_PROCESSING_LABEL)
        self.idle_button.setCheckable(True)
        self.idle_button.setToolTip(
            "Only process while the mouse and keyboard are idle (delay is in Preferences → Live)"
        )
        self.idle_button.toggled.connect(self.idle_toggled.emit)
        self.follow_check = QCheckBox("Follow latest photo")
        self.follow_check.setToolTip("Jump to each new photo as it arrives")
        self.follow_check.toggled.connect(self.follow_toggled.emit)
        self.completed_label = QLabel("Completed: 0")
        self.held_label = QLabel("Waiting for idle")
        self.held_label.setStyleSheet(HELD_STYLE)
        self.held_label.setWordWrap(True)
        self.held_label.setToolTip("Processing resumes when the mouse and keyboard are still")
        self.held_label.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(grid)
        layout.addLayout(bypass_row)
        layout.addWidget(self.idle_button)
        layout.addWidget(self.follow_check)
        layout.addWidget(self.completed_label)
        layout.addWidget(self.held_label)
        layout.addStretch(1)

    def set_enabled_stages(self, settings: LiveSettings) -> None:
        """Reflect the feature toggles and idle option without emitting their signals."""
        self.idle_button.blockSignals(True)
        self.idle_button.setChecked(settings.idle_processing)
        self.idle_button.blockSignals(False)
        self.follow_check.blockSignals(True)
        self.follow_check.setChecked(settings.follow_latest)
        self.follow_check.blockSignals(False)
        for stage, toggle in self.toggles.items():
            toggle.blockSignals(True)
            toggle.setChecked(settings.is_enabled(stage))
            toggle.blockSignals(False)

    def update_status(
        self,
        snapshot: PipelineSnapshot,
        warnings: list[tuple[int, str, str]],
        *,
        blink_on: bool,
        info: Mapping[StageId, str] | None = None,
    ) -> None:
        """Show counts, error flags, blinking waiting counts, stage info, and the warning list."""
        for summary in snapshot.stages:
            label = self.count_labels[summary.stage]
            label.setText(str(summary.count))
            blinking = summary.waiting_disabled and summary.count > 0 and blink_on
            label.setStyleSheet(BLINK_STYLE if blinking else COUNT_STYLE)
            flag = self.error_labels[summary.stage]
            flag.setVisible(bool(summary.halted))
            flag.setToolTip(f"{summary.halted}\n(click for details)" if summary.halted else "")
            self._halted[summary.stage] = summary.halted
        for stage, label in self.info_labels.items():
            text = (info or {}).get(stage, "")
            label.setText(text)
            label.setVisible(bool(text))
        self.completed_label.setText(f"Completed: {snapshot.completed}")
        self.bypass_counts.setText(f"↓ {snapshot.bypass_waiting}  ↑ {snapshot.bypass_sent}")
        self.bypass_label.setEnabled(snapshot.bypass_enabled)
        self.bypass_counts.setEnabled(snapshot.bypass_enabled or bool(snapshot.bypass_sent))
        self.held_label.setVisible(snapshot.held)
        self._set_warnings(warnings)

    def _set_warnings(self, warnings: list[tuple[int, str, str]]) -> None:
        if warnings == self._warnings:
            return
        self._warnings = warnings
        self.warnings_button.setText(f"⚠ {len(warnings)}")
        self.warnings_button.setStyleSheet(ERROR_FLAG_STYLE if warnings else "")
        self.warnings_menu.clear()
        for warning_id, text, photo in warnings:
            action = self.warnings_menu.addAction(text[:MAX_WARNING_MENU_TEXT])
            action.setToolTip(text)
            action.triggered.connect(
                lambda _=False, i=warning_id, p=photo: self.warning_selected.emit(i, p)
            )
        if warnings:
            self.warnings_menu.addSeparator()
            details = self.warnings_menu.addAction("Show details…")
            details.triggered.connect(self._show_warnings)
            clear = self.warnings_menu.addAction("Clear all")
            clear.triggered.connect(self.warnings_cleared)

    def _show_warnings(self) -> None:
        """Open a window with the full text of every warning."""
        text = "\n\n".join(row[1] for row in self._warnings)
        QMessageBox.information(self, "Live warnings", text or "No warnings.")

    def _show_error(self, stage: StageId) -> None:
        """Open a window with why this feature halted, plus its related warnings."""
        label = SPECS[stage].label
        message = self._halted.get(stage) or "No error."
        related = [row[1] for row in self._warnings if row[1].startswith(f"{label}:")]
        text = f"{label} stopped:\n{message}"
        if related:
            text += "\n\nWarnings:\n" + "\n".join(related)
        text += "\n\nToggle the feature off and on to clear the error and retry."
        QMessageBox.warning(self, f"{label} error", text)

    def _confirm_clean(self) -> None:
        answer = QMessageBox.warning(
            self,
            "Clean Live folder",
            "Delete everything in the Live folder, including the input photos?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.clean_requested.emit()


class LiveDock(QWidget):
    """The status panel with an edge tab that hides or shows it."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the tab and panel side by side."""
        super().__init__(parent)
        self.panel = LiveStatusPanel()
        self.tab = QToolButton()
        self.tab.setAutoRaise(True)
        self.tab.setFixedWidth(16)
        self.tab.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.tab.clicked.connect(self.toggle_panel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tab)
        layout.addWidget(self.panel)
        self._sync_tab()

    def toggle_panel(self) -> None:
        """Hide the panel when shown and show it when hidden."""
        self.panel.setVisible(self.panel.isHidden())
        self._sync_tab()

    def _sync_tab(self) -> None:
        shown = not self.panel.isHidden()
        self.tab.setText("›" if shown else "‹")
        self.tab.setToolTip("Hide Live panel" if shown else "Show Live panel")


def warning_rows(controller: LiveController) -> list[tuple[int, str, str]]:
    """
    Return the pipeline warnings as `(id, text, photo)` rows for the panel.

    Returns:
        One row per warning; `photo` is empty when the warning has no photo.
    """
    rows = []
    for warning in controller.pipeline.warnings():
        photo = str(warning.photo) if warning.photo else ""
        name = f"{Path(photo).name}: " if photo else ""
        rows.append((warning.id, f"{SPECS[warning.stage].label}: {name}{warning.message}", photo))
    return rows
