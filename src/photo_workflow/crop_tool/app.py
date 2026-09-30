"""Fast folder-oriented RAW crop application."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections import OrderedDict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import qtawesome as qta
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QObject,
    QProcess,
    QRectF,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QDesktopServices,
    QIcon,
    QImage,
    QImageIOHandler,
    QImageReader,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
    QShortcut,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QSizePolicy,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.config import DEFAULT_CONFIG_PATH
from photo_workflow.crop_tool.horizon import (
    HorizonCandidate,
    detect_horizon_candidates,
    qimage_to_rgb_array,
)
from photo_workflow.crop_tool.metadata import (
    PREVIEW_TAGS,
    PhotoMetadata,
    extract_preview,
    read_photo_metadata,
    read_photo_ratings,
    write_photo_crop,
)
from photo_workflow.crop_tool.model import (
    DEFAULT_ASPECT_RATIOS,
    AspectRatio,
    CropRect,
    aspect_ratio_for_crop,
    closest_aspect_ratio,
    crop_aspect_ratio,
    fit_crop_to_ratio,
    rotated_image_size,
)
from photo_workflow.crop_tool.view import CroppedPreview, CropView

RAW_SUFFIXES = {".cr3"}
DropTarget = tuple[Path, Path | None]
PREVIEW_CACHE_SIZE = 12
MAX_IN_FLIGHT_PREVIEWS = 3
PREFETCH_FORWARD_COUNT = 8
PREFETCH_REVERSE_COUNT = 4
SAVE_DELAY_MS = 350
PREVIEW_MAX_DIMENSION = 2560
RATIO_MATCH_TOLERANCE = 0.005
AUTO_LEVEL_KEY = Qt.Key.Key_A
SHOW_HORIZON_KEY = Qt.Key.Key_H


def shortcut_modifier() -> str:
    """Return the conventional modifier label for the current platform."""
    return "⌘" if sys.platform == "darwin" else "Ctrl+"


def key_hint(key: Qt.Key) -> str:
    """Return the platform-native tooltip label for a keyboard key."""
    return QKeySequence(key).toString(QKeySequence.SequenceFormat.NativeText)


class WorkerSignals(QObject):
    """Signals emitted by background preview and save jobs."""

    metadata = Signal(str, object)
    preview = Signal(str, object, bool)
    error = Signal(str, str)
    finished = Signal(str)
    saved = Signal(str, int, object, float, str)
    horizon = Signal(str, object, str, bool)
    ratings = Signal(int, object, str)


class ClickableLabel(QLabel):
    """A label that emits a signal when clicked with the left mouse button."""

    clicked = Signal()

    def mouseReleaseEvent(self, event: object) -> None:
        """Emit `clicked` when the release occurs inside the label."""
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class RatingFilterPanel(QWidget):
    """Compact pair of rating filters with an overlay for the initial scan."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize a filter row and its scan-status veil."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.scan_overlay = QLabel("Scanning ratings", self)
        self.scan_overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scan_overlay.setStyleSheet(
            "QLabel { background: rgba(32, 36, 39, 235); color: #f4f2ec; "
            "border: 1px solid #777; padding: 2px 5px; }"
        )
        self.scan_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.scan_overlay.hide()

    def resizeEvent(self, event: object) -> None:
        """Keep the scan-status veil stretched over the filter controls."""
        super().resizeEvent(event)
        self.scan_overlay.setGeometry(self.rect())
        if self.scan_overlay.isVisible():
            self.scan_overlay.raise_()

    def set_scanning(self, scanning: bool) -> None:
        """Show or hide the scan veil over the filter row."""
        self.scan_overlay.setVisible(scanning)
        if scanning:
            self.scan_overlay.raise_()


def displayed_image_size(metadata: PhotoMetadata, rotation_angle: float) -> tuple[float, float]:
    """Return the crop-plane size after EXIF orientation and user rotation."""
    return rotated_image_size(metadata.image_width, metadata.image_height, rotation_angle)


def resolve_drop_path(path: Path) -> DropTarget | None:
    """
    Resolve a dropped folder or supported RAW file to folder and selection.

    Returns:
        The folder and optional photo to select, or `None` for unsupported paths.
    """
    if path.is_dir():
        return path, None
    if path.is_file() and path.suffix.lower() in RAW_SUFFIXES:
        return path.parent, path
    return None


def find_photo_index(photos: list[Path], selected_photo: Path | None) -> int:
    """
    Find the selected photo in a folder listing, defaulting to the first photo.

    Returns:
        The matching index or zero when no requested photo is found.
    """
    if selected_photo is None:
        return 0
    selected_path = selected_photo.resolve()
    return next(
        (index for index, photo in enumerate(photos) if photo.resolve() == selected_path),
        0,
    )


def prefetch_indices(
    photo_count: int,
    current_index: int,
    direction: int,
    *,
    forward_count: int = PREFETCH_FORWARD_COUNT,
    reverse_count: int = PREFETCH_REVERSE_COUNT,
) -> list[int]:
    """
    Return current-first indices; the forward count includes the current photo.

    Returns:
        The current index followed by bounded forward and reverse indices.

    Raises:
        ValueError: If the navigation direction is not -1 or 1.
    """
    if photo_count <= 0 or not 0 <= current_index < photo_count:
        return []
    if direction not in {-1, 1}:
        raise ValueError("Prefetch direction must be -1 or 1")

    indices = [current_index]
    indices.extend(
        candidate
        for step in range(1, forward_count)
        if 0 <= (candidate := current_index + direction * step) < photo_count
    )
    indices.extend(
        candidate
        for step in range(1, reverse_count + 1)
        if 0 <= (candidate := current_index - direction * step) < photo_count
    )
    return list(dict.fromkeys(indices))


class CropDropArea(QWidget):
    """Central app surface that accepts a folder or one RAW photo drop."""

    path_dropped = Signal(object)

    def __init__(self) -> None:
        """Enable drag-and-drop handling on the central window surface."""
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: object) -> None:
        """Accept local URLs that resolve to a folder or supported RAW photo."""
        if self._dropped_path(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: object) -> None:
        """Emit the first supported folder or RAW photo from the drop."""
        resolved_path = self._dropped_path(event)
        if resolved_path is None:
            event.ignore()
        else:
            self.path_dropped.emit(resolved_path)
            event.acceptProposedAction()

    @staticmethod
    def _dropped_path(event: object) -> DropTarget | None:
        """
        Resolve the first supported local-file URL in a drag event.

        Returns:
            The resolved folder and optional photo, or `None` if unsupported.
        """
        mime_data = event.mimeData()
        if not mime_data.hasUrls():
            return None
        for url in mime_data.urls():
            if url.isLocalFile():
                resolved = resolve_drop_path(Path(url.toLocalFile()))
                if resolved is not None:
                    return resolved
        return None


class PhotoLoadTask(QRunnable):
    """Read metadata and progressively extract preview images off the UI thread."""

    def __init__(self, raw_path: Path) -> None:
        """Initialize a task for one RAW photo."""
        super().__init__()
        self.raw_path = raw_path
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Load metadata, then emit quick and higher-quality preview images."""
        xmp_path = self.raw_path.with_suffix(".xmp")
        orientation = 1
        try:
            metadata = read_photo_metadata(self.raw_path, xmp_path)
            orientation = metadata.orientation
            self.signals.metadata.emit(str(self.raw_path), metadata)
        except Exception as error:
            self.signals.error.emit(str(self.raw_path), str(error))
        self._emit_previews(orientation)
        self.signals.finished.emit(str(self.raw_path))

    def _emit_previews(self, orientation: int) -> None:
        """Emit decoded previews in quality order, ignoring unavailable tags."""
        for preview_index, tag in enumerate(PREVIEW_TAGS):
            try:
                image = decode_preview(
                    extract_preview(self.raw_path, tag),
                    fallback_orientation=orientation,
                )
            except Exception as error:
                if preview_index == len(PREVIEW_TAGS) - 1:
                    self.signals.error.emit(str(self.raw_path), str(error))
                continue
            if not image.isNull():
                self.signals.preview.emit(str(self.raw_path), image, preview_index > 0)


class PhotoSaveTask(QRunnable):
    """Write one crop update using the application's single-threaded save pool."""

    def __init__(
        self,
        raw_path: Path,
        crop: CropRect,
        crop_angle: float,
        version: int,
    ) -> None:
        """Initialize a crop write snapshot."""
        super().__init__()
        self.raw_path = raw_path
        self.crop = crop
        self.crop_angle = crop_angle
        self.version = version
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Write the XMP update and report success or failure."""
        xmp_path = self.raw_path.with_suffix(".xmp")
        try:
            write_photo_crop(
                self.raw_path,
                xmp_path,
                self.crop,
                crop_angle=self.crop_angle,
            )
            error = ""
        except Exception as exception:
            error = str(exception)
        self.signals.saved.emit(
            str(self.raw_path), self.version, self.crop, self.crop_angle, error
        )


class HorizonAnalysisTask(QRunnable):
    """Detect plausible horizon angles without blocking the interface."""

    def __init__(self, raw_path: Path, image: QImage, apply_best: bool) -> None:
        """Initialize an analysis job with an immutable preview snapshot."""
        super().__init__()
        self.raw_path = raw_path
        self.image = image.copy()
        self.apply_best = apply_best
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Analyze the preview and report candidate angles or an error."""
        try:
            candidates = detect_horizon_candidates(qimage_to_rgb_array(self.image))
            error = ""
        except Exception as exception:
            candidates = []
            error = str(exception)
        self.signals.horizon.emit(
            str(self.raw_path), candidates, error, self.apply_best
        )


class RatingScanTask(QRunnable):
    """Read sidecar star ratings and color labels for one folder in a worker."""

    def __init__(self, generation: int, raw_paths: list[Path]) -> None:
        """Initialize a scan for a snapshot of the current folder's photo list."""
        super().__init__()
        self.generation = generation
        self.raw_paths = raw_paths
        self.signals = WorkerSignals()

    def run(self) -> None:
        """Read ratings and emit them without blocking the UI thread."""
        try:
            ratings = read_photo_ratings(self.raw_paths)
            error = ""
        except Exception as exception:
            ratings = {}
            error = str(exception)
        self.signals.ratings.emit(self.generation, ratings, error)


class CropWindow(QMainWindow):
    """Folder-based crop editor with asynchronous preview and XMP I/O."""

    def __init__(self, initial_folder: Path | None = None) -> None:
        """Create the window, controls, and background worker pools."""
        super().__init__()
        self.setWindowTitle("Photo Workflow Crop")
        self.resize(1280, 820)
        self._ratios, self._snap_tolerance = load_crop_settings()
        self._photos: list[Path] = []
        self._all_photos: list[Path] = []
        self._photo_ratings: dict[Path, tuple[int, str | None]] = {}
        self._ratings_loaded = False
        self._rating_scan_generation = 0
        self._rating_scan_error = ""
        self._star_filter: int | None = None
        self._color_filter: str | None = None
        self._pending_filter_refresh = False
        self._current_index = -1
        self._navigation_direction = 1
        self._current_metadata: PhotoMetadata | None = None
        self._current_crop: CropRect | None = None
        self._current_rotation = 0.0
        self._starting_crop: CropRect | None = None
        self._starting_rotation: float | None = None
        self._horizon_candidates: tuple[HorizonCandidate, ...] = ()
        self._selected_horizon_candidate = -1
        self._locked_ratio: AspectRatio | None = None
        self._current_path: Path | None = None
        self._dirty = False
        self._version = 0
        self._saved_version = 0
        self._save_busy = False
        self._saving_version = 0
        self._pending_navigation = 0
        self._pending_folder: Path | None = None
        self._pending_photo: Path | None = None
        self._close_requested = False
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(3)
        self._rating_pool = QThreadPool(self)
        self._rating_pool.setMaxThreadCount(1)
        self._load_failures: set[Path] = set()
        self._save_pool = QThreadPool(self)
        self._save_pool.setMaxThreadCount(1)
        self._loading: set[Path] = set()
        self._queued_loads: set[Path] = set()
        self._metadata_cache: dict[Path, PhotoMetadata] = {}
        self._preview_cache: OrderedDict[Path, QImage] = OrderedDict()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._start_save)
        self._build_ui()
        self._build_shortcuts()
        if initial_folder is not None:
            self.open_folder(initial_folder)

    def _build_ui(self) -> None:
        """Construct the compact navigation bar and crop canvas."""
        container = CropDropArea()
        layout = QVBoxLayout(container)
        toolbar = QHBoxLayout()
        self.open_button = QToolButton()
        self.open_button.setText("Open")
        self.open_button.setIcon(qta.icon("fa5s.folder-open"))
        self.open_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.open_button.setToolTip(f"Open a photo folder ({shortcut_modifier()}O)")
        self.open_button.setAccessibleName("Open photo folder")
        self.previous_button = QToolButton()
        self.previous_button.setIcon(qta.icon("fa5s.arrow-left"))
        self.previous_button.setToolTip("Previous photo (←)")
        self.previous_button.setAccessibleName("Previous photo")
        self.next_button = QToolButton()
        self.next_button.setIcon(qta.icon("fa5s.arrow-right"))
        self.next_button.setToolTip("Next photo (→)")
        self.next_button.setAccessibleName("Next photo")
        self.position_label = ClickableLabel("")
        self.position_label.setToolTip("Reveal the current photo in the file manager")
        self.position_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.position_count_label = QLabel("No folder open")
        self.position_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        navigation_panel = QWidget()
        navigation_layout = QVBoxLayout(navigation_panel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(1)
        navigation_layout.addWidget(self.position_count_label)
        navigation_buttons = QHBoxLayout()
        navigation_buttons.setContentsMargins(0, 0, 0, 0)
        navigation_buttons.setSpacing(2)
        navigation_buttons.addWidget(self.previous_button)
        navigation_buttons.addWidget(self.next_button)
        navigation_layout.addLayout(navigation_buttons)
        self.star_filter_button = QToolButton()
        self.star_filter_button.setIcon(qta.icon("fa5s.star", color="#e5bd48"))
        self.star_filter_button.setText("*")
        self.star_filter_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.star_filter_button.setFixedWidth(52)
        self.star_filter_button.setToolTip("Filter by minimum star rating")
        self.star_filter_button.setAccessibleName("Filter by star rating")
        self.star_filter_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.star_filter_button.setEnabled(False)
        star_menu = QMenu(self.star_filter_button)
        star_group = QActionGroup(star_menu)
        star_group.setExclusive(True)
        self._star_filter_actions: dict[int | None, QAction] = {}
        star_options = [
            (None, "Any stars"),
            *((value, f"{value} stars or more") for value in range(1, 6)),
        ]
        for stars, label in star_options:
            action = QAction(label, star_group)
            action.setCheckable(True)
            action.setChecked(stars is None)
            action.triggered.connect(
                lambda checked=False, selected=stars: self._set_star_filter(selected)
            )
            star_menu.addAction(action)
            self._star_filter_actions[stars] = action
        self.star_filter_button.setMenu(star_menu)
        self.color_filter_button = QToolButton()
        self.color_filter_button.setIcon(self._color_filter_icon(None))
        self.color_filter_button.setFixedSize(32, 28)
        self.color_filter_button.setAutoRaise(True)
        self.color_filter_button.setToolTip("Filter by Lightroom color label")
        self.color_filter_button.setAccessibleName("Filter by color label")
        self.color_filter_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.color_filter_button.setEnabled(False)
        color_menu = QMenu(self.color_filter_button)
        color_group = QActionGroup(color_menu)
        color_group.setExclusive(True)
        self._color_filter_actions: dict[str | None, QAction] = {}
        color_options = [
            (None, "Any color"),
            ("", "Unlabeled"),
            ("Red", "Red"),
            ("Yellow", "Yellow"),
            ("Green", "Green"),
            ("Blue", "Blue"),
            ("Purple", "Purple"),
        ]
        for label_value, label in color_options:
            action = QAction(label, color_group)
            action.setCheckable(True)
            action.setChecked(label_value is None)
            action.triggered.connect(
                lambda checked=False, selected=label_value: self._set_color_filter(selected)
            )
            color_menu.addAction(action)
            self._color_filter_actions[label_value] = action
        self.color_filter_button.setMenu(color_menu)
        self.rating_filter_panel = RatingFilterPanel()
        rating_filter_layout = QHBoxLayout(self.rating_filter_panel)
        rating_filter_layout.setContentsMargins(0, 0, 0, 0)
        rating_filter_layout.setSpacing(2)
        rating_filter_layout.addWidget(self.star_filter_button)
        rating_filter_layout.addWidget(self.color_filter_button)
        self.ratio_combo = QComboBox()
        self.ratio_combo.setFixedWidth(104)
        self.ratio_combo.setToolTip(
            "Current crop ratio; exact value shown for Free crops (1–8 to choose)"
        )
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(-450, 450)
        self.rotation_slider.setSingleStep(1)
        self.rotation_slider.setPageStep(10)
        self.rotation_slider.setFixedWidth(150)
        self.rotation_slider.setToolTip("Rotate the photo beneath the crop frame")
        self.rotation_slider.setEnabled(False)
        self.rotation_value = QLabel("0.0°")
        self.rotation_value.setFixedWidth(
            self.rotation_value.fontMetrics().horizontalAdvance("-45.0°") + 4
        )
        self.rotation_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.rotation_reset = QToolButton()
        self.rotation_reset.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.rotation_reset.setToolTip("Reset rotation")
        self.rotation_reset.setAccessibleName("Reset rotation")
        self.rotation_reset.setEnabled(False)
        self.angle_group = QGroupBox("Angle")
        angle_layout = QHBoxLayout(self.angle_group)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.setSpacing(4)
        angle_layout.addWidget(self.rotation_slider)
        angle_layout.addWidget(self.rotation_value)
        angle_layout.addWidget(self.rotation_reset)
        self.auto_level_button = QToolButton()
        self.auto_level_button.setIcon(qta.icon("fa5s.magic", color="#45d6d0"))
        self.auto_level_button.setToolTip(
            f"Auto-level to the strongest detected line ({key_hint(AUTO_LEVEL_KEY)})"
        )
        self.auto_level_button.setAccessibleName("Auto level")
        self.auto_level_button.setFixedSize(28, 28)
        self.auto_level_button.setAutoRaise(True)
        self.auto_level_button.setEnabled(False)
        self.show_horizon_button = QToolButton()
        self.show_horizon_button.setIcon(qta.icon("fa5s.eye-slash", color="#f4f2ec"))
        self.show_horizon_button.setToolTip(
            "Show detected horizon candidates without changing the angle "
            f"({key_hint(SHOW_HORIZON_KEY)})"
        )
        self.show_horizon_button.setAccessibleName("Show horizon candidates")
        self.show_horizon_button.setCheckable(True)
        self.show_horizon_button.setFixedSize(28, 28)
        self.show_horizon_button.setAutoRaise(True)
        self.show_horizon_button.setEnabled(False)
        for control in (
            self.auto_level_button,
            self.show_horizon_button,
        ):
            angle_layout.addWidget(control)
        self.side_by_side_button = QToolButton()
        self.side_by_side_button.setIcon(qta.icon("fa5s.columns", color="#f4f2ec"))
        self.side_by_side_button.setToolTip("Show side-by-side crop preview")
        self.side_by_side_button.setAccessibleName("Toggle side-by-side crop preview")
        self.side_by_side_button.setCheckable(True)
        self.side_by_side_button.setFixedSize(28, 28)
        self.side_by_side_button.setAutoRaise(True)
        self.revert_button = QToolButton()
        self.revert_button.setIcon(qta.icon("fa5s.undo", color="#f4f2ec"))
        self.revert_button.setToolTip("Revert crop and angle to when this photo was opened")
        self.revert_button.setAccessibleName("Revert this photo's crop and angle")
        self.revert_button.setFixedSize(28, 28)
        self.revert_button.setAutoRaise(True)
        self.revert_button.setEnabled(False)
        self.lock_checkbox = QToolButton()
        self.lock_checkbox.setCheckable(True)
        self.lock_checkbox.setToolTip("Lock crop ratio (L)")
        self.lock_checkbox.setAccessibleName("Lock crop ratio")
        self.lock_checkbox.setIcon(qta.icon("fa5s.lock-open", color="#f4f2ec"))
        self.snap_checkbox = QToolButton()
        self.snap_checkbox.setCheckable(True)
        self.snap_checkbox.setChecked(True)
        self.snap_checkbox.setToolTip("Aspect-ratio snapping enabled (S)")
        self.snap_checkbox.setAccessibleName("Toggle aspect-ratio snapping")
        self._update_snap_icon(True)
        self.aspect_group = QGroupBox("Aspect")
        aspect_layout = QHBoxLayout(self.aspect_group)
        aspect_layout.setContentsMargins(0, 0, 0, 0)
        aspect_layout.setSpacing(4)
        aspect_layout.addWidget(self.ratio_combo)
        aspect_layout.addWidget(self.lock_checkbox)
        aspect_layout.addWidget(self.snap_checkbox)
        self.save_label = QLabel(" ")
        self.save_label.setToolTip(f"Save current crop ({shortcut_modifier()}S)")
        status_width = self.save_label.fontMetrics().horizontalAdvance("Saving...") + 8
        self.save_label.setFixedWidth(status_width)
        self.save_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        toolbar.addWidget(self.open_button)
        toolbar.addWidget(navigation_panel)
        toolbar.addWidget(self.rating_filter_panel)
        toolbar.addWidget(self.position_label, 1)
        toolbar.addWidget(self.angle_group)
        toolbar.addWidget(self.side_by_side_button)
        toolbar.addWidget(self.revert_button)
        toolbar.addWidget(self.aspect_group)
        toolbar.addWidget(self.save_label)
        self.view = CropView()
        self.cropped_preview = CroppedPreview()
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.addWidget(QLabel("Cropped result"))
        result_layout.addWidget(self.cropped_preview, 1)
        self.preview_container = QWidget()
        preview_layout = QHBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(8)
        preview_layout.addWidget(self.view, 1)
        preview_layout.addWidget(result_panel, 1)
        result_panel.setVisible(False)
        layout.addLayout(toolbar)
        layout.addWidget(self.preview_container, 1)
        self.setCentralWidget(container)
        container.path_dropped.connect(self._open_dropped_path)
        self._populate_ratios()
        self.open_button.clicked.connect(self._choose_folder)
        self.previous_button.clicked.connect(lambda: self.navigate(-1))
        self.next_button.clicked.connect(lambda: self.navigate(1))
        self.ratio_combo.currentIndexChanged.connect(self._ratio_selected)
        self.rotation_slider.valueChanged.connect(self._rotation_slider_changed)
        self.rotation_reset.clicked.connect(lambda: self.rotation_slider.setValue(0))
        self.auto_level_button.clicked.connect(self._auto_level_clicked)
        self.show_horizon_button.clicked.connect(self._toggle_horizon_visibility)
        self.side_by_side_button.toggled.connect(self._set_side_by_side_visible)
        self.revert_button.clicked.connect(self._revert_current_photo)
        self.view.horizon_candidate_selected.connect(self._horizon_guide_selected)
        self.cropped_preview.crop_changed.connect(self._crop_edited)
        self.lock_checkbox.toggled.connect(self._update_crop_mode)
        self.snap_checkbox.toggled.connect(self._update_crop_mode)
        self.lock_checkbox.toggled.connect(self._update_lock_icon)
        self.snap_checkbox.toggled.connect(self._update_snap_icon)
        self.position_label.clicked.connect(self._reveal_current_photo)
        self._update_filter_button_labels()
        self.view.crop_changed.connect(self._crop_edited)
        self._update_lock_icon(self.lock_checkbox.isChecked())
        self._update_crop_mode()

    def _set_star_filter(self, minimum_stars: int | None) -> None:
        self._star_filter = minimum_stars
        self._update_filter_button_labels()
        self._refresh_rating_filter()

    def _set_color_filter(self, color_label: str | None) -> None:
        self._color_filter = color_label
        self._update_filter_button_labels()
        self._refresh_rating_filter()

    def _update_filter_button_labels(self) -> None:
        self.star_filter_button.setText(
            f"{self._star_filter}+" if self._star_filter is not None else "*"
        )
        self.color_filter_button.setText("")
        self.color_filter_button.setIcon(self._color_filter_icon(self._color_filter))
        self.star_filter_button.setToolTip(
            f"Star filter: {self._star_filter}+ stars"
            if self._star_filter is not None
            else "Filter by minimum star rating"
        )
        self.color_filter_button.setToolTip(
            f"Color filter: {'Unlabeled' if self._color_filter == '' else self._color_filter}"
            if self._color_filter is not None
            else "Filter by Lightroom color label"
        )

    @staticmethod
    def _color_filter_icon(color_label: str | None) -> QIcon:
        colors = {
            "Red": "#e45858",
            "Yellow": "#e5bd48",
            "Green": "#55e39f",
            "Blue": "#5297e8",
            "Purple": "#aa70d6",
        }
        if color_label == "":
            color = "#8c9299"
        elif color_label is not None:
            color = colors.get(color_label, "#8c9299")
        else:
            pixmap = QPixmap(24, 18)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            gradient = QLinearGradient(2, 2, 22, 2)
            for position, rainbow_color in enumerate(
                ("#e45858", "#e5bd48", "#55e39f", "#5297e8", "#aa70d6")
            ):
                gradient.setColorAt(position / 4, QColor(rainbow_color))
            painter.setBrush(gradient)
            painter.setPen(QColor("#f4f2ec"))
            painter.drawRoundedRect(QRectF(2, 2, 20, 14), 2, 2)
            painter.end()
            return QIcon(pixmap)
        return qta.icon("fa5s.square", options=[{"color": color}])

    def _start_rating_scan(self, raw_paths: list[Path]) -> None:
        self._rating_scan_generation += 1
        self._ratings_loaded = False
        self._rating_scan_error = ""
        self.star_filter_button.setEnabled(False)
        self.color_filter_button.setEnabled(False)
        self.rating_filter_panel.set_scanning(True)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self._update_position_label()
        task = RatingScanTask(self._rating_scan_generation, raw_paths)
        task.signals.ratings.connect(self._rating_scan_finished)
        self._rating_pool.start(task)

    def _rating_scan_finished(
        self,
        generation: int,
        ratings: dict[Path, tuple[int, str | None]],
        error: str,
    ) -> None:
        if generation != self._rating_scan_generation:
            return
        self._rating_scan_error = error
        self._ratings_loaded = not error
        if error:
            self._star_filter = None
            self._color_filter = None
            self._update_filter_button_labels()
            self._photos = list(self._all_photos)
            self.previous_button.setEnabled(self._current_index > 0)
            self.next_button.setEnabled(0 <= self._current_index < len(self._photos) - 1)
            self._update_position_label()
            self.star_filter_button.setToolTip(f"Rating filters unavailable: {error}")
            self.color_filter_button.setToolTip(f"Rating filters unavailable: {error}")
            self.rating_filter_panel.set_scanning(True, "Ratings unavailable")
            return
        self._photo_ratings = ratings
        self.rating_filter_panel.set_scanning(False)
        self.star_filter_button.setEnabled(True)
        self.color_filter_button.setEnabled(True)
        self._refresh_rating_filter()

    def _refresh_rating_filter(self) -> None:
        if not self._ratings_loaded:
            return
        filtered_photos = [
            path
            for path in self._all_photos
            if self._photo_matches_rating_filters(path)
        ]
        if filtered_photos == self._photos:
            self.previous_button.setEnabled(self._current_index > 0)
            self.next_button.setEnabled(0 <= self._current_index < len(self._photos) - 1)
            self._update_position_label()
            return
        if self._dirty or self._save_busy:
            self._pending_filter_refresh = True
            self._start_save()
            return

        previous_index = max(self._current_index, 0)
        current_path = self._current_path
        self._photos = filtered_photos
        if current_path in filtered_photos:
            self._current_index = filtered_photos.index(current_path)
            self.previous_button.setEnabled(self._current_index > 0)
            self.next_button.setEnabled(self._current_index < len(filtered_photos) - 1)
            self._update_position_label()
            self._schedule_prefetch(self._current_index)
        elif filtered_photos:
            self._show_photo(min(previous_index, len(filtered_photos) - 1))
        else:
            self._clear_current_photo_for_empty_filter()

    def _photo_matches_rating_filters(self, raw_path: Path) -> bool:
        stars, color = self._photo_ratings.get(raw_path, (0, None))
        if self._star_filter is not None and stars < self._star_filter:
            return False
        if (
            self._color_filter is not None
            and (color or "").casefold() != self._color_filter.casefold()
        ):
            return False
        return True

    def _clear_current_photo_for_empty_filter(self) -> None:
        self._current_index = -1
        self._current_path = None
        self._current_metadata = None
        self._current_crop = None
        self._starting_crop = None
        self._starting_rotation = None
        self.revert_button.setEnabled(False)
        self.rotation_slider.setEnabled(False)
        self.rotation_reset.setEnabled(False)
        self.auto_level_button.setEnabled(False)
        self.show_horizon_button.setEnabled(False)
        self._set_preview_loading(False)
        self._set_preview_image(QImage(), 1, 1)
        self._set_preview_crop(None)
        self.view.set_horizon_candidates(())
        self.position_count_label.setText("0 / 0 [Filtered]")
        self.position_label.setText("No photos match filters")

    def _update_position_label(self) -> None:
        if self._current_path is None or self._current_index < 0:
            return
        filtered_note = (
            " [Filtered]"
            if self._ratings_loaded and self._rating_filters_active()
            else " [Ratings unavailable]"
            if self._rating_scan_error
            else " [Scanning ratings]"
            if not self._ratings_loaded and self._all_photos
            else ""
        )
        self.position_count_label.setText(
            f"{self._current_index + 1} / {len(self._photos)}{filtered_note}"
        )
        self.position_label.setText(self._current_path.name)

    def _rating_filters_active(self) -> bool:
        return self._star_filter is not None or self._color_filter is not None

    def _set_side_by_side_visible(self, visible: bool) -> None:
        """Show or hide the read-only crop result pane."""
        result_panel = self.cropped_preview.parentWidget()
        result_panel.setVisible(visible)
        description = "Hide" if visible else "Show"
        self.side_by_side_button.setToolTip(f"{description} side-by-side crop preview")

    def _set_preview_image(
        self,
        image: QImage,
        image_width: int,
        image_height: int,
    ) -> None:
        """Update the editing and cropped-result panes with one source preview."""
        self.view.set_image(image, image_width, image_height)
        self.cropped_preview.set_image(image)

    def _set_preview_crop(self, crop: CropRect | None) -> CropRect | None:
        """
        Update crop geometry in both panes and return its bounded value.

        Returns:
            The bounded crop, or `None` when no crop is active.
        """
        crop = self.view.set_crop(crop)
        self.cropped_preview.set_crop(crop)
        return crop

    def _set_loaded_crop(self, path: Path, metadata: PhotoMetadata) -> PhotoMetadata:
        """
        Constrain loaded crop metadata and queue a save if bounds needed correction.

        Returns:
            Metadata containing the crop shown in the image panes.
        """
        if self._starting_crop is not None and self._current_crop is not None:
            self._set_preview_crop(self._current_crop)
            return self._current_metadata or metadata
        crop = self._set_preview_crop(metadata.crop)
        if crop is None or crop == metadata.crop:
            self._current_crop = crop
            self._capture_starting_state(crop)
            return metadata
        corrected_metadata = replace(metadata, crop=crop)
        self._metadata_cache[path] = corrected_metadata
        if path == self._current_path:
            self._current_crop = crop
            self._current_metadata = corrected_metadata
            self._capture_starting_state(crop)
            self._mark_dirty()
        return corrected_metadata

    def _capture_starting_state(self, crop: CropRect | None) -> None:
        """Cache the first displayable crop and angle for the current photo visit."""
        if self._starting_crop is not None or crop is None:
            return
        self._starting_crop = crop
        self._starting_rotation = round(self._current_rotation, 1)
        self._update_revert_button()

    def _set_preview_rotation(self, angle: float) -> None:
        """Update rotation in both image panes."""
        self.view.set_rotation(angle)
        self.cropped_preview.set_rotation(angle)

    def _set_preview_loading(self, loading: bool) -> None:
        """Set the loading state in both image panes."""
        self.view.set_loading(loading)
        self.cropped_preview.set_loading(loading)

    def _populate_ratios(self) -> None:
        """Fill the aspect-ratio selector from settings."""
        self.ratio_combo.addItem("Free", None)
        for ratio in self._ratios:
            self.ratio_combo.addItem(ratio.label, ratio)

    def _build_shortcuts(self) -> None:
        """Install navigation, crop-mode, snap, open, and save shortcuts."""
        self._add_shortcut(Qt.Key.Key_Left, lambda: self.navigate(-1))
        self._add_shortcut(Qt.Key.Key_Right, lambda: self.navigate(1))
        self._add_shortcut(AUTO_LEVEL_KEY, self._auto_level_clicked)
        self._add_shortcut(Qt.Key.Key_L, self._toggle_lock)
        self._add_shortcut(Qt.Key.Key_S, self._toggle_snap)
        self._add_shortcut(SHOW_HORIZON_KEY, self._show_horizon_clicked)
        self._add_shortcut(QKeySequence.StandardKey.Open, self._choose_folder)
        self._add_shortcut(QKeySequence.StandardKey.Save, self._start_save)
        for index in range(min(9, len(self._ratios))):
            self._add_shortcut(
                Qt.Key(int(Qt.Key.Key_1) + index),
                lambda selected_index=index: self.ratio_combo.setCurrentIndex(selected_index),
            )

    def _add_shortcut(self, key: QKeySequence | Qt.Key, callback: object) -> None:
        """Create a window-scoped keyboard shortcut."""
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(callback)

    def open_folder(self, folder: Path, selected_photo: Path | None = None) -> None:
        """Load supported RAW files and optionally start on one selected photo."""
        if self._dirty or self._save_busy:
            self._pending_folder = folder
            self._pending_photo = selected_photo
            self._pending_navigation = 0
            self._start_save()
            return
        self._all_photos = (
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in RAW_SUFFIXES
            )
            if folder.is_dir()
            else []
        )
        self._photos = list(self._all_photos)
        self._photo_ratings = {path: (0, None) for path in self._all_photos}
        self._load_failures.clear()
        self._queued_loads.clear()
        self._navigation_direction = 1
        self._current_index = -1
        self._current_path = None
        if not self._all_photos:
            self._rating_scan_generation += 1
            self._ratings_loaded = False
            self._photo_ratings = {}
            self.position_count_label.setText("No CR3 files in folder")
            self.position_label.setText("")
            self._set_preview_loading(False)
            self._set_preview_image(QImage(), 1, 1)
            self._set_preview_crop(None)
            self.star_filter_button.setEnabled(False)
            self.color_filter_button.setEnabled(False)
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return
        self._start_rating_scan(self._all_photos)
        self._show_photo(find_photo_index(self._photos, selected_photo))

    def navigate(self, offset: int) -> None:
        """Move to the adjacent photo after pending edits are safely written."""
        if not self._photos or self._current_index < 0:
            return
        target_index = min(max(self._current_index + offset, 0), len(self._photos) - 1)
        if target_index == self._current_index:
            return
        self._navigation_direction = 1 if target_index > self._current_index else -1
        if self._dirty or self._save_busy:
            self._pending_navigation = target_index - self._current_index
            self._start_save()
            return
        self._show_photo(target_index)

    def _show_photo(self, index: int) -> None:
        """Select a photo, restore cached state, and queue nearby previews."""
        self._current_index = index
        self._current_path = self._photos[index]
        self.previous_button.setEnabled(
            self._ratings_loaded and self._current_index > 0
        )
        self.next_button.setEnabled(
            self._ratings_loaded and self._current_index < len(self._photos) - 1
        )
        self._current_crop = None
        self._current_metadata = self._metadata_cache.get(self._current_path)
        self._current_rotation = 0.0
        self._starting_crop = None
        self._starting_rotation = None
        self.revert_button.setEnabled(False)
        self._set_preview_rotation(0)
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(0)
        self.rotation_value.setText("0.0°")
        self.rotation_slider.setEnabled(False)
        self.rotation_reset.setEnabled(False)
        self.rotation_slider.blockSignals(False)
        self._clear_horizon_candidates()
        self._version = 0
        self._saved_version = 0
        self._dirty = False
        self.save_label.setText(" ")
        self._update_position_label()
        self._set_preview_loading(True)
        self._set_preview_crop(None)
        cached_image = self._preview_cache.get(self._current_path)
        if cached_image is not None:
            self._show_image(cached_image)
            self.auto_level_button.setEnabled(True)
            self.show_horizon_button.setEnabled(True)
        if self._current_metadata is not None:
            self._apply_metadata(self._current_path, self._current_metadata)
        self._schedule_prefetch(index)

    def _schedule_prefetch(self, index: int) -> None:
        """Rebuild pending work in current-photo and travel-direction priority order."""
        ordered_paths = [
            self._photos[photo_index]
            for photo_index in prefetch_indices(
                len(self._photos),
                index,
                self._navigation_direction,
            )
        ]
        self._queued_loads = set(ordered_paths)
        self._pump_loads(ordered_paths)

    def _pump_loads(self, ordered_paths: list[Path] | None = None) -> None:
        """Start queued jobs in priority order without exceeding worker capacity."""
        if ordered_paths is None and self._current_index >= 0:
            ordered_paths = [
                self._photos[photo_index]
                for photo_index in prefetch_indices(
                    len(self._photos),
                    self._current_index,
                    self._navigation_direction,
                )
            ]
        if ordered_paths is None:
            return

        available_slots = MAX_IN_FLIGHT_PREVIEWS - len(self._loading)
        for raw_path in ordered_paths:
            if available_slots <= 0:
                break
            if raw_path not in self._queued_loads:
                continue
            self._queued_loads.discard(raw_path)
            if raw_path in self._loading or raw_path in self._load_failures:
                continue
            if raw_path in self._metadata_cache and raw_path in self._preview_cache:
                continue
            self._start_photo_load(raw_path)
            available_slots -= 1

    def _start_photo_load(self, raw_path: Path) -> None:
        """Submit one selected load task to the bounded thread pool."""
        task = PhotoLoadTask(raw_path)
        task.signals.metadata.connect(self._metadata_loaded)
        task.signals.preview.connect(self._preview_loaded)
        task.signals.error.connect(self._load_error)
        task.signals.finished.connect(self._load_finished)
        self._loading.add(raw_path)
        self._load_pool.start(task)

    def _metadata_loaded(self, path_text: str, metadata: PhotoMetadata) -> None:
        """Cache metadata and apply it if its photo is currently selected."""
        path = Path(path_text)
        self._metadata_cache[path] = metadata
        if path == self._current_path:
            self._apply_metadata(path, metadata)

    def _apply_metadata(self, path: Path, metadata: PhotoMetadata) -> None:
        """Set current crop bounds and source dimensions for the selected photo."""
        if path != self._current_path:
            return
        self._current_metadata = metadata
        self._current_crop = metadata.crop
        self._current_rotation = -metadata.crop_angle
        self.rotation_slider.blockSignals(True)
        self.rotation_slider.setValue(round(self._current_rotation * 10))
        self.rotation_value.setText(f"{self._current_rotation:.1f}°")
        self.rotation_slider.setEnabled(True)
        self.rotation_reset.setEnabled(True)
        self.rotation_slider.blockSignals(False)
        self._set_preview_rotation(self._current_rotation)
        display_width, display_height = displayed_image_size(metadata, self._current_rotation)
        image = self._preview_cache.get(path)
        if image is not None:
            self._set_preview_image(image, display_width, display_height)
            metadata = self._set_loaded_crop(path, metadata)
            self._set_preview_loading(False)
            self.auto_level_button.setEnabled(True)
            self.show_horizon_button.setEnabled(True)
        else:
            self._set_preview_crop(None)
            self._set_preview_loading(True)
        self._select_crop_ratio(metadata.crop)
        self._update_crop_mode()

    def _preview_loaded(self, path_text: str, image: QImage, high_quality: bool) -> None:
        """Cache previews and display progressive updates for the selected photo."""
        path = Path(path_text)
        if high_quality or path not in self._preview_cache:
            self._preview_cache[path] = image
            self._preview_cache.move_to_end(path)
            while len(self._preview_cache) > PREVIEW_CACHE_SIZE:
                self._preview_cache.popitem(last=False)
        if path == self._current_path:
            self._show_image(image)
            self.auto_level_button.setEnabled(True)
            self.show_horizon_button.setEnabled(True)
            if self._current_metadata is not None:
                self._current_metadata = self._set_loaded_crop(
                    path, self._current_metadata
                )
                self._set_preview_loading(False)

    def _show_image(self, image: QImage) -> None:
        """Set the visible preview using current source dimensions when known."""
        if self._current_metadata:
            width, height = displayed_image_size(self._current_metadata, self._current_rotation)
        else:
            width, height = image.width(), image.height()
        self._set_preview_image(image, width, height)

    def _load_error(self, path_text: str, message: str) -> None:
        """Show a load failure for the active photo without blocking navigation."""
        path = Path(path_text)
        self._load_failures.add(path)
        if path == self._current_path:
            self._set_preview_loading(False)
            self.save_label.setText(f"Load error: {message}")

    def _load_finished(self, path_text: str) -> None:
        """Release the in-flight marker for a completed photo load."""
        self._loading.discard(Path(path_text))
        if self._current_index >= 0:
            self._schedule_prefetch(self._current_index)

    def _crop_edited(self, crop: CropRect) -> None:
        """Mark the current photo dirty and debounce its XMP write."""
        if self._current_path is None or self._current_metadata is None:
            return
        constrained_crop = self._set_preview_crop(crop)
        if constrained_crop is None:
            return
        self._current_crop = constrained_crop
        self._mark_dirty()
        self._select_crop_ratio(constrained_crop)

    def _revert_current_photo(self) -> None:
        """Restore this photo's starting crop and angle and queue them for saving."""
        if (
            self._starting_crop is None
            or self._starting_rotation is None
            or self._current_path is None
            or self._current_metadata is None
        ):
            return
        self._save_timer.stop()
        self.rotation_slider.setValue(round(self._starting_rotation * 10))
        self.rotation_value.setText(f"{self._starting_rotation:.1f}°")
        self._current_rotation = self._starting_rotation
        self._set_preview_rotation(self._starting_rotation)
        crop = self._set_preview_crop(self._starting_crop)
        if crop is None:
            return
        self._current_crop = crop
        self._select_crop_ratio(crop)
        self._mark_dirty()

    def _rotation_slider_changed(self, slider_value: int) -> None:
        """Convert the slider's tenths-degree value into a rotation edit."""
        angle = slider_value / 10
        self.rotation_value.setText(f"{angle:.1f}°")
        self._rotation_changed(angle)

    def _auto_level_clicked(self) -> None:
        """Analyze the current preview and apply its strongest angle candidate."""
        self._analyze_horizon(apply_best=True)

    def _show_horizon_clicked(self) -> None:
        """Activate the Show control so shortcut and button state remain synchronized."""
        self.show_horizon_button.click()

    def _toggle_horizon_visibility(self) -> None:
        """Toggle candidate guides, analyzing only when no cached candidates exist."""
        if self.show_horizon_button.isChecked():
            if self._horizon_candidates:
                self.view.set_horizon_candidates(
                    self._horizon_candidates,
                    self._selected_horizon_candidate,
                )
                self._update_horizon_visibility_icon(True)
            else:
                self._analyze_horizon(apply_best=False)
        else:
            self.view.set_horizon_candidates(())
            self._update_horizon_visibility_icon(False)

    def _update_horizon_visibility_icon(self, visible: bool) -> None:
        """Reflect candidate visibility with an open- or closed-eye icon."""
        icon_name = "fa5s.eye" if visible else "fa5s.eye-slash"
        self.show_horizon_button.setIcon(qta.icon(icon_name, color="#f4f2ec"))
        action = "Hide" if visible else "Show"
        self.show_horizon_button.setToolTip(
            f"{action} detected horizon candidates without changing the angle "
            f"({key_hint(SHOW_HORIZON_KEY)})"
        )

    def _analyze_horizon(self, *, apply_best: bool) -> None:
        """Start asynchronous horizon analysis in apply or reveal-only mode."""
        if self._current_path is None:
            return
        image = self._preview_cache.get(self._current_path)
        if image is None:
            return
        task = HorizonAnalysisTask(self._current_path, image, apply_best)
        task.signals.horizon.connect(self._horizon_analysis_finished)
        self.auto_level_button.setEnabled(False)
        self.show_horizon_button.setEnabled(False)
        self.auto_level_button.setToolTip(
            f"Analyzing horizon candidates... ({key_hint(AUTO_LEVEL_KEY)})"
        )
        self.show_horizon_button.setToolTip(
            f"Analyzing horizon candidates... ({key_hint(SHOW_HORIZON_KEY)})"
        )
        self._load_pool.start(task)

    def _horizon_analysis_finished(
        self,
        path_text: str,
        candidates: list[HorizonCandidate],
        error: str,
        apply_best: bool,
    ) -> None:
        """Show current-photo candidates and optionally apply the strongest."""
        path = Path(path_text)
        if path != self._current_path:
            return
        self.auto_level_button.setEnabled(path in self._preview_cache)
        self.show_horizon_button.setEnabled(path in self._preview_cache)
        auto_message = error or "Auto level to the strongest candidate"
        self.auto_level_button.setToolTip(
            f"{auto_message} ({key_hint(AUTO_LEVEL_KEY)})"
        )
        self.show_horizon_button.setToolTip(
            f"{error or 'Show detected candidates without changing the angle'} "
            f"({key_hint(SHOW_HORIZON_KEY)})"
        )
        self._horizon_candidates = tuple(candidates)
        selected_index = 0 if candidates and apply_best else -1
        self._selected_horizon_candidate = selected_index
        self.view.set_horizon_candidates(tuple(candidates), selected_index)
        self.show_horizon_button.setChecked(bool(candidates))
        self._update_horizon_visibility_icon(bool(candidates))
        if candidates and apply_best:
            self.rotation_slider.setValue(round(candidates[0].angle_degrees * 10))
        elif not candidates:
            self.auto_level_button.setToolTip(
                f"{error or 'No level candidates found in this preview'} "
                f"({key_hint(AUTO_LEVEL_KEY)})"
            )

    def _horizon_guide_selected(self, index: int) -> None:
        """Apply an image-line selection through the existing angle slider."""
        if not 0 <= index < len(self._horizon_candidates):
            return
        candidate = self._horizon_candidates[index]
        self._selected_horizon_candidate = index
        self.rotation_slider.setValue(round(candidate.angle_degrees * 10))
        self.view.set_horizon_candidates(self._horizon_candidates, index)

    def _clear_horizon_candidates(self) -> None:
        """Clear suggestions when switching to another photo."""
        self._horizon_candidates = ()
        self._selected_horizon_candidate = -1
        self.view.set_horizon_candidates(())
        self.show_horizon_button.setChecked(False)
        self._update_horizon_visibility_icon(False)
        self.auto_level_button.setToolTip(
            f"Auto-level to the strongest detected line ({key_hint(AUTO_LEVEL_KEY)})"
        )
        self.auto_level_button.setEnabled(False)
        self.show_horizon_button.setToolTip(
            "Show detected horizon candidates without changing the angle "
            f"({key_hint(SHOW_HORIZON_KEY)})"
        )
        self.show_horizon_button.setEnabled(False)

    def _rotation_changed(self, angle: float) -> None:
        """Update the displayed rotation and persist it as a crop edit."""
        if self._current_path is None or self._current_metadata is None:
            return
        self._current_rotation = angle
        self._set_preview_rotation(angle)
        if self._current_metadata is not None:
            image = self._preview_cache.get(self._current_path)
            if image is not None:
                width, height = displayed_image_size(self._current_metadata, angle)
                self._set_preview_image(image, width, height)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """Mark the current photo changed and debounce its XMP write."""
        self._version += 1
        self._dirty = True
        self._update_revert_button()
        self.save_label.setText("Unsaved")
        self._save_timer.start(SAVE_DELAY_MS)

    def _update_revert_button(self) -> None:
        """Enable Revert only when crop or angle differs from its starting state."""
        changed = (
            self._starting_crop is not None
            and self._starting_rotation is not None
            and (
                self._current_crop != self._starting_crop
                or round(self._current_rotation, 1) != self._starting_rotation
            )
        )
        self.revert_button.setEnabled(changed)

    def _start_save(self) -> None:
        """Start writing the latest crop snapshot if the current photo is dirty."""
        if not self._dirty or self._save_busy or self._current_path is None:
            return
        if self._current_crop is None:
            return
        self._save_timer.stop()
        crop_angle = -self._current_rotation
        task = PhotoSaveTask(
            self._current_path,
            self._current_crop,
            crop_angle,
            self._version,
        )
        task.signals.saved.connect(self._save_finished)
        self._save_busy = True
        self._saving_version = self._version
        self.save_label.setText("Saving...")
        self._save_pool.start(task)

    def _save_finished(
        self,
        path_text: str,
        version: int,
        saved_crop: CropRect,
        saved_crop_angle: float,
        error: str,
    ) -> None:
        """Update save state and continue any deferred navigation or close."""
        self._save_busy = False
        if error:
            self._dirty = True
            self.save_label.setText(f"Save failed: {error}")
            return
        if Path(path_text) == self._current_path:
            self._saved_version = max(self._saved_version, version)
            self._dirty = self._saved_version < self._version
            self.save_label.setText("Unsaved" if self._dirty else "Saved")
        saved_path = Path(path_text)
        saved_metadata = update_saved_crop_cache(
            self._metadata_cache,
            saved_path,
            saved_crop,
            crop_angle=saved_crop_angle,
        )
        if saved_path == self._current_path and saved_metadata is not None:
            self._current_metadata = saved_metadata
        if self._dirty:
            self._start_save()
        elif self._pending_filter_refresh:
            self._pending_filter_refresh = False
            self._refresh_rating_filter()
        elif self._pending_folder is not None:
            folder = self._pending_folder
            selected_photo = self._pending_photo
            self._pending_folder = None
            self._pending_photo = None
            self.open_folder(folder, selected_photo)
        elif self._pending_navigation:
            target = self._current_index + self._pending_navigation
            self._pending_navigation = 0
            self._show_photo(min(max(target, 0), len(self._photos) - 1))
        elif self._close_requested:
            self.close()

    def _update_crop_mode(self, *args: object) -> None:
        """Apply the selected lock ratio and snap state to the crop view."""
        del args
        if self.lock_checkbox.isChecked() and self._current_crop and self._current_metadata:
            image_width, image_height = displayed_image_size(
                self._current_metadata,
                self._current_rotation,
            )
            self._locked_ratio = aspect_ratio_for_crop(
                self._current_crop,
                image_width,
                image_height,
            )
        else:
            self._locked_ratio = None
        snap_ratios = self._ratios if self.snap_checkbox.isChecked() else ()
        self.view.set_crop_mode(
            locked_ratio=self._locked_ratio,
            snap_ratios=snap_ratios,
            snap_tolerance=self._snap_tolerance,
        )

    def _ratio_selected(self, index: int) -> None:
        """Apply a chosen preset immediately or switch to freeform geometry."""
        ratio = self.ratio_combo.itemData(index)
        if ratio is not None and self._current_crop is not None and self._current_metadata:
            image_width, image_height = displayed_image_size(
                self._current_metadata,
                self._current_rotation,
            )
            crop = fit_crop_to_ratio(
                self._current_crop,
                image_width,
                image_height,
                ratio,
            )
            self._current_crop = crop
            self._set_preview_crop(crop)
            if crop != self._current_metadata.crop:
                self._crop_edited(crop)
        self._update_crop_mode()

    def _select_crop_ratio(self, crop: CropRect) -> None:
        """Show the closest preset only when the current crop actually matches it."""
        if self._current_metadata is None:
            return
        image_width, image_height = displayed_image_size(
            self._current_metadata,
            self._current_rotation,
        )
        ratio = closest_aspect_ratio(
            crop.width,
            crop.height,
            image_width,
            image_height,
            self._ratios,
            tolerance=RATIO_MATCH_TOLERANCE,
        )
        selected_index = 0
        if ratio is not None:
            selected_index = self.ratio_combo.findData(ratio)
            self.ratio_combo.setItemText(0, "Free")
            self.ratio_combo.setItemData(
                0,
                "Current crop ratio is a configured preset",
                Qt.ItemDataRole.ToolTipRole,
            )
        else:
            current_ratio = crop_aspect_ratio(
                crop,
                image_width,
                image_height,
            )
            self.ratio_combo.setItemText(0, f"Free {current_ratio:.2f}")
            self.ratio_combo.setItemData(
                0,
                f"Freeform crop ratio: {current_ratio:.6f}:1",
                Qt.ItemDataRole.ToolTipRole,
            )
        if selected_index < 0:
            selected_index = 0
        if self.ratio_combo.currentIndex() != selected_index:
            self.ratio_combo.blockSignals(True)
            self.ratio_combo.setCurrentIndex(selected_index)
            self.ratio_combo.blockSignals(False)

    def _toggle_lock(self) -> None:
        """Toggle fixed-aspect crop resizing."""
        self.lock_checkbox.toggle()

    def _toggle_snap(self) -> None:
        """Toggle freeform aspect-ratio snapping."""
        self.snap_checkbox.toggle()

    def _update_lock_icon(self, locked: bool) -> None:
        """Reflect ratio-lock state in the lock control's icon and tooltip."""
        self.lock_checkbox.setIcon(
            qta.icon("fa5s.lock" if locked else "fa5s.lock-open", color="#f4f2ec")
        )
        self.lock_checkbox.setToolTip(
            f"Crop ratio {'locked' if locked else 'unlocked'} (L)"
        )

    def _update_snap_icon(self, enabled: bool) -> None:
        """Reflect snapping state with a magnet or red disabled-state icon."""
        if enabled:
            snap_icon = qta.icon("fa5s.magnet", color="#45d6d0")
        else:
            snap_icon = qta.icon(
                "fa5s.magnet",
                "fa5s.ban",
                options=[
                    {"color": "#f4f2ec"},
                    {"color": "#e45858", "scale_factor": 0.9},
                ],
            )
        self.snap_checkbox.setIcon(snap_icon)
        self.snap_checkbox.setToolTip(
            f"Aspect-ratio snapping {'enabled' if enabled else 'disabled'} (S)"
        )

    def _reveal_current_photo(self) -> None:
        """Reveal the current RAW photo in the platform's file manager."""
        if self._current_path is None:
            return
        path = str(self._current_path.resolve())
        if sys.platform == "darwin":
            QProcess.startDetached("open", ["-R", path])
        elif sys.platform == "win32":
            QProcess.startDetached("explorer", [f"/select,{path}"])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def _choose_folder(self) -> None:
        """Prompt for a folder and open its CR3 files."""
        folder_text = QFileDialog.getExistingDirectory(self, "Open photo folder")
        if folder_text:
            self.open_folder(Path(folder_text))

    def _open_dropped_path(self, resolved_path: DropTarget) -> None:
        """Open a dropped folder or its selected RAW photo."""
        folder, selected_photo = resolved_path
        self.open_folder(folder, selected_photo)

    def closeEvent(self, event: object) -> None:
        """Wait for dirty crop data to be written before closing the window."""
        if self._dirty or self._save_busy:
            self._close_requested = True
            self._start_save()
            event.ignore()
            return
        self._queued_loads.clear()
        self._load_pool.clear()
        self._load_pool.waitForDone()
        self._rating_pool.clear()
        self._rating_pool.waitForDone()
        event.accept()


def decode_preview(image_bytes: bytes, *, fallback_orientation: int = 1) -> QImage:
    """
    Decode an embedded JPEG and fall back to the RAW orientation when needed.

    Returns:
        A Qt image, null if the bytes are not a supported image.
    """
    buffer = QBuffer()
    buffer.setData(QByteArray(image_bytes))
    buffer.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(buffer)
    reader.setAutoTransform(True)
    embedded_orientation = reader.transformation()
    source_size = reader.size()
    if max(source_size.width(), source_size.height()) > PREVIEW_MAX_DIMENSION:
        reader.setScaledSize(
            source_size.scaled(
                QSize(PREVIEW_MAX_DIMENSION, PREVIEW_MAX_DIMENSION),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        )
    image = reader.read()
    if (
        not image.isNull()
        and embedded_orientation == QImageIOHandler.Transformation.TransformationNone
    ):
        image = apply_exif_orientation(image, fallback_orientation)
    return image


def apply_exif_orientation(image: QImage, orientation: int) -> QImage:
    """
    Apply an EXIF orientation when an embedded preview lacks its own tag.

    Returns:
        The image transformed to match the RAW file's display orientation.
    """
    if orientation == 2:
        return image.mirrored(True, False)
    if orientation == 3:
        return image.transformed(QTransform().rotate(180))
    if orientation == 4:
        return image.mirrored(False, True)
    if orientation == 5:
        return image.transformed(QTransform(0, 1, 1, 0, 0, 0))
    if orientation == 6:
        return image.transformed(QTransform().rotate(90))
    if orientation == 7:
        return image.transformed(QTransform(0, 1, 1, 0, 0, 0)).mirrored(True, True)
    if orientation == 8:
        return image.transformed(QTransform().rotate(-90))
    return image


def update_saved_crop_cache(
    metadata_cache: dict[Path, PhotoMetadata],
    raw_path: Path,
    crop: CropRect,
    *,
    crop_angle: float | None = None,
) -> PhotoMetadata | None:
    """
    Update cached metadata to match the crop snapshot successfully written.

    Returns:
        The updated metadata, or `None` if the photo has not been cached.
    """
    metadata = metadata_cache.get(raw_path)
    if metadata is None:
        return None
    updated_metadata = replace(
        metadata,
        crop=crop,
        crop_angle=metadata.crop_angle if crop_angle is None else crop_angle,
    )
    metadata_cache[raw_path] = updated_metadata
    return updated_metadata


def load_crop_settings(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> tuple[tuple[AspectRatio, ...], float]:
    """
    Load configurable ratios and snap tolerance from the project TOML.

    Returns:
        The ratio list and relative snap tolerance.

    Raises:
        ValueError: If the snap tolerance or configured ratios are invalid.
    """
    if not config_path.is_file():
        return DEFAULT_ASPECT_RATIOS, 0.05
    with config_path.open("rb") as config_file:
        settings = tomllib.load(config_file).get("crop_tool", {})
    raw_ratios = settings.get("aspect_ratios")
    ratios = parse_aspect_ratios(raw_ratios) if raw_ratios is not None else DEFAULT_ASPECT_RATIOS
    tolerance = float(settings.get("snap_tolerance", 0.05))
    if tolerance < 0 or tolerance > 1:
        raise ValueError("crop_tool.snap_tolerance must be between 0 and 1")
    return ratios, tolerance


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
        raise ValueError("crop_tool.aspect_ratios must contain at least one ratio")
    return tuple(ratios)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the optional initial photo folder.

    Returns:
        The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(prog="photo-workflow-crop")
    parser.add_argument("folder", nargs="?", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Start the crop editor application.

    Returns:
        The Qt application exit code.
    """
    args = parse_args(argv)
    app = QApplication.instance() or QApplication([])
    window = CropWindow(args.folder)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
