"""Visual dashboard for launching and running photo workflow tools."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QFileInfo, QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileIconProvider,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.config import AppConfig, DashboardAppLauncher, load_config
from photo_workflow.dashboard.launchers import (
    LaunchError,
    launch_app,
    resolve_application_path,
    resolve_bundled_command_icon,
)
from photo_workflow.memory_card_copy import find_memory_card_mount
from photo_workflow.rejected_folders import (
    RejectedFolderAssessment,
    assess_rejected_folders,
    bytes_to_gigabytes,
    iter_rejected_folders,
    purge_rejected_folders,
    select_assessment_folders,
)
from photo_workflow.workflow import run_full_import

POLL_INTERVAL_MS = 2000
LAUNCHER_ICON_SIZE = QSize(48, 48)
IMPORT_STAGES = (
    "Assessing rejected folders",
    "Importing memory card & generating video notes",
    "Reporting disk space",
)
LOGGER = logging.getLogger(__name__)


ANSI_ESCAPE_PATTERN = re.compile(r"\033\[[0-9;]*m")


class QtLogHandler(logging.Handler, QObject):
    """A logging handler that forwards formatted records to a Qt signal."""

    message_logged = Signal(str)

    def __init__(self) -> None:
        """Initialize the handler with a timestamped single-line formatter."""
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        """Forward the formatted log line to listeners, stripping any ANSI color codes.

        Log messages may embed ANSI styling intended for a real terminal (e.g. the
        memory-card eject message). The dashboard's log widget isn't a terminal and
        can't render those escape sequences, so they must be stripped here rather
        than shown as literal garbage characters.
        """
        plain_message = ANSI_ESCAPE_PATTERN.sub("", self.format(record))
        self.message_logged.emit(plain_message)


class ImportWorkerSignals(QObject):
    """Signals emitted by a background import run."""

    stage = Signal(str)
    finished = Signal(object, object)
    error = Signal(str)


class ImportWorker(QRunnable):
    """Runs the full memory-card import off the UI thread."""

    def __init__(self, config: AppConfig) -> None:
        """Store the configuration the import run will use."""
        super().__init__()
        self.config = config
        self.signals = ImportWorkerSignals()

    def run(self) -> None:
        """Execute the import and report results or errors via signals."""
        try:
            assessment, source_dirs = run_full_import(self.config, on_stage=self.signals.stage.emit)
        except Exception as error:  # noqa: BLE001 - surface any failure to the UI
            self.signals.error.emit(str(error))
            return
        self.signals.finished.emit(assessment, source_dirs)


class PurgeWorkerSignals(QObject):
    """Signals emitted by a background rejected-folder purge."""

    finished = Signal(int)
    error = Signal(str)


class PurgeWorker(QRunnable):
    """Purges selected rejected folders off the UI thread."""

    def __init__(self, assessment: RejectedFolderAssessment) -> None:
        """Store the folders this worker will purge."""
        super().__init__()
        self.assessment = assessment
        self.signals = PurgeWorkerSignals()

    def run(self) -> None:
        """Delete the selected folders and report the count or any error."""
        try:
            purged_count = purge_rejected_folders(self.assessment)
        except Exception as error:  # noqa: BLE001 - surface any failure to the UI
            self.signals.error.emit(str(error))
            return
        self.signals.finished.emit(purged_count)


def resolve_launcher_icon(launcher: DashboardAppLauncher) -> QIcon:
    """Return the best available icon for a dashboard launcher."""
    if launcher.command:
        icon_path = resolve_bundled_command_icon(launcher.command[0])
        if icon_path is not None:
            return QIcon(str(icon_path))

    if launcher.app_name:
        app_path = resolve_application_path(launcher.app_name)
        if app_path is not None:
            return QFileIconProvider().icon(QFileInfo(str(app_path)))

    return QIcon.fromTheme("applications-other")


class LauncherButton(QToolButton):
    """A large icon button that launches one dashboard application."""

    def __init__(
        self,
        launcher: DashboardAppLauncher,
        target_folder_provider: Callable[[DashboardAppLauncher], Path | None] | None = None,
        status_reporter: Callable[[str], None] | None = None,
    ) -> None:
        """Build the button for one dashboard app launcher."""
        super().__init__()
        self.launcher = launcher
        self.target_folder_provider = target_folder_provider
        self.status_reporter = status_reporter
        self.setText(launcher.name)
        self.setIcon(resolve_launcher_icon(launcher))
        self.setIconSize(LAUNCHER_ICON_SIZE)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.setToolTip(f"Open {launcher.name}")
        self.setAutoRaise(True)
        self.clicked.connect(self._launch)

    def _launch(self) -> None:
        """Launch the configured application or command for this button."""
        target_folder = None
        if self.target_folder_provider is not None:
            target_folder = self.target_folder_provider(self.launcher)
        LOGGER.info("launching %s", self.launcher.name)
        try:
            launch_app(self.launcher, target_folder=target_folder)
        except LaunchError as error:
            LOGGER.error("%s", error)
            if self.status_reporter is not None:
                self.status_reporter(str(error))


@dataclass
class RejectedFolderRow:
    """A single rejected-folder entry shown in the purge-review dialog."""

    checkbox: QCheckBox
    folder_path: Path


class PurgeReviewDialog(QDialog):
    """Lets the user deselect specific `_Rejected` folders before purging."""

    def __init__(
        self,
        assessment: RejectedFolderAssessment,
        parent: QWidget | None = None,
    ) -> None:
        """Build the review list of rejected folders for the given assessment."""
        super().__init__(parent)
        self.assessment = assessment
        self.rows: list[RejectedFolderRow] = []
        self.setWindowTitle("Purge Rejected Folders")
        self.resize(520, 420)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(f"Found {len(assessment.folders)} _Rejected folder(s). Uncheck any to keep.")
        )

        toggle_row = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_none_button = QPushButton("Select None")
        select_all_button.clicked.connect(lambda: self._set_all_checked(True))
        select_none_button.clicked.connect(lambda: self._set_all_checked(False))
        toggle_row.addWidget(select_all_button)
        toggle_row.addWidget(select_none_button)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        for item in assessment.folders:
            label = (
                f"{item.folder_path.relative_to(assessment.camera_root)}  —  "
                f"{bytes_to_gigabytes(item.reclaimable_bytes):.2f} GB "
                f"({item.percent_of_disk:.2f}% of disk)"
            )
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self._update_total_label)
            list_layout.addWidget(checkbox)
            self.rows.append(RejectedFolderRow(checkbox=checkbox, folder_path=item.folder_path))
        list_layout.addStretch()
        scroll_area.setWidget(list_widget)
        layout.addWidget(scroll_area)

        self.total_label = QLabel()
        layout.addWidget(self.total_label)
        self._update_total_label()

        buttons = QDialogButtonBox()
        self.purge_button = buttons.addButton(
            "Purge Selected", QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.purge_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all_checked(self, checked: bool) -> None:
        """Check or uncheck every folder row."""
        for row in self.rows:
            row.checkbox.setChecked(checked)

    def _update_total_label(self) -> None:
        selected_paths = self.selected_folder_paths()
        selected_assessment = select_assessment_folders(self.assessment, selected_paths)
        self.total_label.setText(
            f"Selected: {len(selected_assessment.folders)} folder(s), "
            f"{bytes_to_gigabytes(selected_assessment.total_reclaimable_bytes):.2f} GB "
            f"({selected_assessment.total_percent_of_disk:.2f}% of disk)"
        )

    def selected_folder_paths(self) -> set[Path]:
        """Return the folder paths currently checked for purge."""
        return {row.folder_path for row in self.rows if row.checkbox.isChecked()}


class DashboardWindow(QMainWindow):
    """Main dashboard window: app launchers plus Import / Purge Rejected actions."""

    def __init__(self, config: AppConfig | None = None) -> None:
        """Build the dashboard window, loading configuration if not supplied."""
        super().__init__()
        self.config = config or load_config()
        self.thread_pool = QThreadPool.globalInstance()
        self.last_import_folder: Path | None = None
        self.setWindowTitle("Photo Workflow Dashboard")
        self.resize(720, 480)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_launcher_row())
        root_layout.addWidget(self._build_divider())
        root_layout.addLayout(self._build_action_row())

        self.status_label = QLabel("Ready.")
        root_layout.addWidget(self.status_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        root_layout.addWidget(self.log_view, stretch=1)

        self._install_log_handler()
        self._refresh_target_folder_checkbox()

        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._refresh_action_availability)
        self.poll_timer.start(POLL_INTERVAL_MS)
        self._refresh_action_availability()

    def _build_launcher_row(self) -> QGroupBox:
        group = QGroupBox("Launch")
        layout = QVBoxLayout(group)

        icon_row = QHBoxLayout()
        for launcher in self.config.dashboard.apps:
            icon_row.addWidget(
                LauncherButton(
                    launcher,
                    target_folder_provider=self._target_folder_for_launcher,
                    status_reporter=self._report_launch_failure,
                )
            )
        icon_row.addStretch()
        layout.addLayout(icon_row)

        self.target_folder_checkbox = QCheckBox("Open last imported folder in supporting apps")
        self.target_folder_checkbox.setChecked(True)
        layout.addWidget(self.target_folder_checkbox)

        return group

    def _build_divider(self) -> QFrame:
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        return divider

    def _build_action_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()

        self.import_button = QPushButton("Import Memory Card")
        self.import_button.setMinimumHeight(56)
        bold_font = QFont()
        bold_font.setBold(True)
        self.import_button.setFont(bold_font)
        self.import_button.clicked.connect(self._start_import)
        layout.addWidget(self.import_button)

        self.purge_button = QPushButton("Purge Rejected…")
        self.purge_button.setMinimumHeight(56)
        self.purge_button.setFont(bold_font)
        self.purge_button.clicked.connect(self._open_purge_review)
        layout.addWidget(self.purge_button)

        return layout

    def _install_log_handler(self) -> None:
        self.log_handler = QtLogHandler()
        self.log_handler.message_logged.connect(self.log_view.appendPlainText)
        self.log_handler.setLevel(logging.INFO)
        logging.getLogger("photo_workflow").addHandler(self.log_handler)
        logging.getLogger("photo_workflow").setLevel(logging.INFO)

    def _target_folder_for_launcher(self, launcher: DashboardAppLauncher) -> Path | None:
        """Return the folder to open a launcher with, if it opts in and one is available."""
        if not launcher.supports_target_folder:
            return None
        if not self.target_folder_checkbox.isChecked():
            return None
        return self.last_import_folder

    def _refresh_target_folder_checkbox(self) -> None:
        """Enable the target-folder checkbox once a folder is imported, checked by default."""
        has_target_folder = self.last_import_folder is not None
        self.target_folder_checkbox.setEnabled(has_target_folder)
        self.target_folder_checkbox.setChecked(has_target_folder)
        self.target_folder_checkbox.setToolTip(
            str(self.last_import_folder) if has_target_folder else "No imported folder yet"
        )

    def _report_launch_failure(self, message: str) -> None:
        self.status_label.setText(message)

    def _refresh_action_availability(self) -> None:
        card_detected = find_memory_card_mount(self.config.memory_card_copy.card_mount_root)
        self.import_button.setEnabled(card_detected is not None)
        self.import_button.setToolTip(
            "Memory card detected" if card_detected is not None else "No memory card detected"
        )

        if not self.import_button.isEnabled():
            self.import_button.setText("Import Memory Card")

        rejected_folders = iter_rejected_folders(self.config.workflow.camera_root)
        self.purge_button.setEnabled(len(rejected_folders) > 0)
        self.purge_button.setToolTip(
            f"{len(rejected_folders)} _Rejected folder(s) found"
            if rejected_folders
            else "No _Rejected folders found"
        )

    def _start_import(self) -> None:
        self.import_button.setEnabled(False)
        self.purge_button.setEnabled(False)
        self.poll_timer.stop()
        self.status_label.setText("Starting import…")

        worker = ImportWorker(self.config)
        worker.signals.stage.connect(self._on_import_stage)
        worker.signals.finished.connect(self._on_import_finished)
        worker.signals.error.connect(self._on_import_error)
        self.thread_pool.start(worker)

    def _on_import_stage(self, stage_name: str) -> None:
        stage_index = IMPORT_STAGES.index(stage_name) + 1 if stage_name in IMPORT_STAGES else "?"
        self.status_label.setText(f"({stage_index}/{len(IMPORT_STAGES)}) {stage_name}…")

    def _on_import_finished(
        self,
        assessment: RejectedFolderAssessment,
        source_dirs: tuple[Path, ...],
    ) -> None:
        folder_list = ", ".join(str(folder) for folder in source_dirs) or "none"
        self.status_label.setText(f"Import complete. Target folder(s): {folder_list}")
        self.last_import_folder = source_dirs[0] if source_dirs else None
        self._refresh_target_folder_checkbox()
        self.poll_timer.start(POLL_INTERVAL_MS)
        self._refresh_action_availability()

    def _on_import_error(self, message: str) -> None:
        self.status_label.setText(f"Import failed: {message}")
        LOGGER.error("import failed: %s", message)
        self.poll_timer.start(POLL_INTERVAL_MS)
        self._refresh_action_availability()

    def _open_purge_review(self) -> None:
        assessment = assess_rejected_folders(self.config.workflow.camera_root)
        if not assessment.folders:
            self.status_label.setText("No _Rejected folders found.")
            self._refresh_action_availability()
            return

        dialog = PurgeReviewDialog(assessment, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        selected_paths = dialog.selected_folder_paths()
        if not selected_paths:
            self.status_label.setText("No folders selected for purge.")
            return

        selected_assessment = select_assessment_folders(assessment, selected_paths)
        self.status_label.setText(f"Purging {len(selected_assessment.folders)} folder(s)…")
        self.purge_button.setEnabled(False)

        worker = PurgeWorker(selected_assessment)
        worker.signals.finished.connect(self._on_purge_finished)
        worker.signals.error.connect(self._on_purge_error)
        self.thread_pool.start(worker)

    def _on_purge_finished(self, purged_count: int) -> None:
        self.status_label.setText(f"Purged {purged_count} rejected folder(s).")
        self._refresh_action_availability()

    def _on_purge_error(self, message: str) -> None:
        self.status_label.setText(f"Purge failed: {message}")
        LOGGER.error("purge failed: %s", message)
        self._refresh_action_availability()


def main() -> int:
    """
    Start the photo workflow dashboard application.

    Returns:
        The Qt application exit code.
    """
    app = QApplication.instance() or QApplication([])
    window = DashboardWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
