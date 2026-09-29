"""Fast folder-oriented RAW crop application."""

from __future__ import annotations

import argparse
import tomllib
from collections import OrderedDict
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QImage, QImageIOHandler, QImageReader, QKeySequence, QShortcut, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from photo_workflow.config import DEFAULT_CONFIG_PATH
from photo_workflow.crop_tool.metadata import (
    PREVIEW_TAGS,
    PhotoMetadata,
    extract_preview,
    read_photo_metadata,
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
)
from photo_workflow.crop_tool.view import CropView

RAW_SUFFIXES = {".cr3"}
DropTarget = tuple[Path, Path | None]
PREVIEW_CACHE_SIZE = 12
MAX_IN_FLIGHT_PREVIEWS = 3
PREFETCH_FORWARD_COUNT = 8
PREFETCH_REVERSE_COUNT = 4
SAVE_DELAY_MS = 350
PREVIEW_MAX_DIMENSION = 2560
RATIO_MATCH_TOLERANCE = 0.005


class WorkerSignals(QObject):
    """Signals emitted by background preview and save jobs."""

    metadata = Signal(str, object)
    preview = Signal(str, object, bool)
    error = Signal(str, str)
    finished = Signal(str)
    saved = Signal(str, int, object, str)


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
        self.signals.saved.emit(str(self.raw_path), self.version, self.crop, error)


class CropWindow(QMainWindow):
    """Folder-based crop editor with asynchronous preview and XMP I/O."""

    def __init__(self, initial_folder: Path | None = None) -> None:
        """Create the window, controls, and background worker pools."""
        super().__init__()
        self.setWindowTitle("Photo Workflow Crop")
        self.resize(1280, 820)
        self._ratios, self._snap_tolerance = load_crop_settings()
        self._photos: list[Path] = []
        self._current_index = -1
        self._navigation_direction = 1
        self._current_metadata: PhotoMetadata | None = None
        self._current_crop: CropRect | None = None
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
        self.open_button = QPushButton("Open Folder")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.position_label = QLabel("No folder open")
        self.position_label.setMinimumWidth(260)
        self.ratio_combo = QComboBox()
        self.ratio_combo.setFixedWidth(104)
        self.ratio_combo.setToolTip("Current crop ratio; exact value shown for Free crops")
        self.lock_checkbox = QCheckBox("Lock ratio")
        self.snap_checkbox = QCheckBox("Snap")
        self.snap_checkbox.setChecked(True)
        self.save_label = QLabel(" ")
        for button in (self.open_button, self.previous_button, self.next_button):
            toolbar.addWidget(button)
        toolbar.addWidget(self.position_label, 1)
        toolbar.addWidget(self.ratio_combo)
        toolbar.addWidget(self.lock_checkbox)
        toolbar.addWidget(self.snap_checkbox)
        toolbar.addWidget(self.save_label)
        self.view = CropView()
        layout.addLayout(toolbar)
        layout.addWidget(self.view, 1)
        self.setCentralWidget(container)
        container.path_dropped.connect(self._open_dropped_path)
        self._populate_ratios()
        self.open_button.clicked.connect(self._choose_folder)
        self.previous_button.clicked.connect(lambda: self.navigate(-1))
        self.next_button.clicked.connect(lambda: self.navigate(1))
        self.ratio_combo.currentIndexChanged.connect(self._ratio_selected)
        self.lock_checkbox.toggled.connect(self._update_crop_mode)
        self.snap_checkbox.toggled.connect(self._update_crop_mode)
        self.view.crop_changed.connect(self._crop_edited)
        self._update_crop_mode()

    def _populate_ratios(self) -> None:
        """Fill the aspect-ratio selector from settings."""
        self.ratio_combo.addItem("Free", None)
        for ratio in self._ratios:
            self.ratio_combo.addItem(ratio.label, ratio)

    def _build_shortcuts(self) -> None:
        """Install navigation, crop-mode, snap, open, and save shortcuts."""
        self._add_shortcut(Qt.Key.Key_Left, lambda: self.navigate(-1))
        self._add_shortcut(Qt.Key.Key_Right, lambda: self.navigate(1))
        self._add_shortcut(Qt.Key.Key_L, self._toggle_lock)
        self._add_shortcut(Qt.Key.Key_S, self._toggle_snap)
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
        self._photos = (
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in RAW_SUFFIXES
            )
            if folder.is_dir()
            else []
        )
        self._load_failures.clear()
        self._queued_loads.clear()
        self._navigation_direction = 1
        self._current_index = -1
        self._current_path = None
        if not self._photos:
            self.position_label.setText("No CR3 files in folder")
            self.view.set_loading(False)
            self.view.set_image(QImage(), 1, 1)
            self.view.set_crop(None)
            return
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
        self._current_crop = None
        self._current_metadata = self._metadata_cache.get(self._current_path)
        self._version = 0
        self._saved_version = 0
        self._dirty = False
        self.save_label.setText(" ")
        self.position_label.setText(
            f"{index + 1} / {len(self._photos)}    {self._current_path.name}"
        )
        self.view.set_loading(True)
        self.view.set_crop(None)
        cached_image = self._preview_cache.get(self._current_path)
        if cached_image is not None:
            self._show_image(cached_image)
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
        image = self._preview_cache.get(path)
        if image is not None:
            self.view.set_image(image, metadata.image_width, metadata.image_height)
            self.view.set_crop(metadata.crop)
            self.view.set_loading(False)
        else:
            self.view.set_crop(None)
            self.view.set_loading(True)
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
            if self._current_metadata is not None:
                self.view.set_crop(self._current_metadata.crop)
                self.view.set_loading(False)

    def _show_image(self, image: QImage) -> None:
        """Set the visible preview using current source dimensions when known."""
        width = self._current_metadata.image_width if self._current_metadata else image.width()
        height = self._current_metadata.image_height if self._current_metadata else image.height()
        self.view.set_image(image, width, height)

    def _load_error(self, path_text: str, message: str) -> None:
        """Show a load failure for the active photo without blocking navigation."""
        path = Path(path_text)
        self._load_failures.add(path)
        if path == self._current_path:
            self.view.set_loading(False)
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
        self._current_crop = crop
        self._version += 1
        self._dirty = True
        self._select_crop_ratio(crop)
        self.save_label.setText("Unsaved")
        self._save_timer.start(SAVE_DELAY_MS)

    def _start_save(self) -> None:
        """Start writing the latest crop snapshot if the current photo is dirty."""
        if not self._dirty or self._save_busy or self._current_path is None:
            return
        if self._current_crop is None:
            return
        self._save_timer.stop()
        crop_angle = self._current_metadata.crop_angle if self._current_metadata else 0
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
        saved_metadata = update_saved_crop_cache(self._metadata_cache, saved_path, saved_crop)
        if saved_path == self._current_path and saved_metadata is not None:
            self._current_metadata = saved_metadata
        if self._dirty:
            self._start_save()
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
            self._locked_ratio = aspect_ratio_for_crop(
                self._current_crop,
                self._current_metadata.image_width,
                self._current_metadata.image_height,
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
            crop = fit_crop_to_ratio(
                self._current_crop,
                self._current_metadata.image_width,
                self._current_metadata.image_height,
                ratio,
            )
            self._current_crop = crop
            self.view.set_crop(crop)
            if crop != self._current_metadata.crop:
                self._crop_edited(crop)
        self._update_crop_mode()

    def _select_crop_ratio(self, crop: CropRect) -> None:
        """Show the closest preset only when the current crop actually matches it."""
        if self._current_metadata is None:
            return
        ratio = closest_aspect_ratio(
            crop.width,
            crop.height,
            self._current_metadata.image_width,
            self._current_metadata.image_height,
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
                self._current_metadata.image_width,
                self._current_metadata.image_height,
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
) -> PhotoMetadata | None:
    """
    Update cached metadata to match the crop snapshot successfully written.

    Returns:
        The updated metadata, or `None` if the photo has not been cached.
    """
    metadata = metadata_cache.get(raw_path)
    if metadata is None:
        return None
    updated_metadata = replace(metadata, crop=crop)
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
        return DEFAULT_ASPECT_RATIOS, 0.025
    with config_path.open("rb") as config_file:
        settings = tomllib.load(config_file).get("crop_tool", {})
    raw_ratios = settings.get("aspect_ratios")
    ratios = parse_aspect_ratios(raw_ratios) if raw_ratios is not None else DEFAULT_ASPECT_RATIOS
    tolerance = float(settings.get("snap_tolerance", 0.025))
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
