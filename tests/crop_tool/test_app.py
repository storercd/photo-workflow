"""Tests for crop application settings and ratio selection helpers."""

from pathlib import Path

import pytest
from PIL import Image
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QPoint,
    QPointF,
    QSettings,
    QSize,
    Qt,
    QUrl,
)
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QImage, QKeySequence, QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from photo_workflow.crop_tool import app as crop_app
from photo_workflow.crop_tool.app import (
    PREFETCH_FORWARD_COUNT,
    PREFETCH_REVERSE_COUNT,
    PREVIEW_CACHE_SIZE,
    CropDropArea,
    CropWindow,
    decode_preview,
    find_photo_index,
    load_crop_settings,
    parse_aspect_ratios,
    prefetch_indices,
    resolve_drop_path,
    update_saved_crop_cache,
)
from photo_workflow.crop_tool.horizon import HorizonCandidate
from photo_workflow.crop_tool.metadata import PhotoMetadata
from photo_workflow.crop_tool.model import CropRect


def test_resolve_drop_path_opens_folder_without_specific_selection(tmp_path: Path) -> None:
    """A dropped directory opens as a folder without choosing a photo."""
    result = resolve_drop_path(tmp_path)

    assert result == (tmp_path, None)


def test_resolve_drop_path_opens_supported_photo_parent_and_selects_photo(
    tmp_path: Path,
) -> None:
    """A dropped CR3 resolves to its parent folder and itself as the selection."""
    photo_path = tmp_path / "second.CR3"
    photo_path.touch()

    assert resolve_drop_path(photo_path) == (tmp_path, photo_path)


def test_resolve_drop_path_accepts_cr2(tmp_path: Path) -> None:
    """A dropped CR2 selects its parent folder and the selected photo."""
    photo_path = tmp_path / "backup.CR2"
    photo_path.touch()

    assert resolve_drop_path(photo_path) == (tmp_path, photo_path)


def test_cr2_preview_does_not_require_jpg_from_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CR2 PreviewImage is final and does not trigger a missing-tag error."""
    task = crop_app.PhotoLoadTask(Path("backup.CR2"))
    previews: list[tuple[str, bool]] = []
    errors: list[str] = []
    tags: list[str] = []
    task.signals.preview.connect(lambda path, image, high: previews.append((path, high)))
    task.signals.error.connect(lambda path, error: errors.append(error))

    def extract_preview(path: Path, tag: str) -> bytes:
        tags.append(tag)
        if tag != "PreviewImage":
            raise RuntimeError("missing JpgFromRaw")
        return b"preview"

    monkeypatch.setattr(crop_app, "extract_preview", extract_preview)
    monkeypatch.setattr(
        crop_app,
        "decode_preview",
        lambda data, **kwargs: QImage(8, 8, QImage.Format.Format_RGB32),
    )
    task._emit_previews(1)

    assert tags == ["PreviewImage"]
    assert previews == [("backup.CR2", True)]
    assert errors == []


def test_cr3_preview_prefers_jpg_from_raw_without_double_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR3 emits the full-resolution preview directly without a second refresh."""
    task = crop_app.PhotoLoadTask(Path("primary.CR3"))
    tags: list[str] = []
    previews: list[bool] = []
    task.signals.preview.connect(lambda path, image, final: previews.append(final))

    def extract_preview(path: Path, tag: str) -> bytes:
        tags.append(tag)
        return b"preview"

    monkeypatch.setattr(crop_app, "extract_preview", extract_preview)
    monkeypatch.setattr(
        crop_app,
        "decode_preview",
        lambda data, **kwargs: QImage(8, 8, QImage.Format.Format_RGB32),
    )

    task._emit_previews(1)

    assert tags == ["JpgFromRaw"]
    assert previews == [True]


def test_cr3_preview_falls_back_to_preview_image_when_jpg_from_raw_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CR3 falls back to PreviewImage if JpgFromRaw is unavailable."""
    task = crop_app.PhotoLoadTask(Path("primary.CR3"))
    tags: list[str] = []
    previews: list[bool] = []
    task.signals.preview.connect(lambda path, image, final: previews.append(final))

    def extract_preview(path: Path, tag: str) -> bytes:
        tags.append(tag)
        if tag == "JpgFromRaw":
            raise RuntimeError("missing tag")
        return b"preview"

    monkeypatch.setattr(crop_app, "extract_preview", extract_preview)
    monkeypatch.setattr(
        crop_app,
        "decode_preview",
        lambda data, **kwargs: QImage(8, 8, QImage.Format.Format_RGB32),
    )

    task._emit_previews(1)

    assert tags == ["JpgFromRaw", "PreviewImage"]
    assert previews == [True]


def test_resolve_drop_path_rejects_unsupported_files(tmp_path: Path) -> None:
    """Non-RAW files are not treated as photo drops."""
    unsupported_path = tmp_path / "notes.txt"
    unsupported_path.touch()

    assert resolve_drop_path(unsupported_path) is None


def test_find_photo_index_selects_requested_photo(tmp_path: Path) -> None:
    """Select the dropped photo even when it is not first in sorted order."""
    photos = [tmp_path / "first.cr3", tmp_path / "second.cr3", tmp_path / "third.cr3"]

    assert find_photo_index(photos, photos[1]) == 1
    assert find_photo_index(photos, None) == 0


def test_prefetch_window_favors_current_navigation_direction() -> None:
    """Keep current first, then bias the eight-frame window in travel direction."""
    forward = prefetch_indices(30, 10, 1)
    reverse = prefetch_indices(30, 10, -1)

    assert forward == [10, 11, 12, 13, 14, 15, 16, 17, 9, 8, 7, 6]
    assert reverse == [10, 9, 8, 7, 6, 5, 4, 3, 11, 12, 13, 14]
    assert len(forward) == PREFETCH_FORWARD_COUNT + PREFETCH_REVERSE_COUNT
    assert PREVIEW_CACHE_SIZE >= len(forward) * 2


def test_prefetch_window_clips_to_folder_edges_without_duplicates() -> None:
    """Directional windows stay within the folder and contain no repeated indices."""
    indices = prefetch_indices(5, 0, -1)

    assert indices == [0, 1, 2, 3, 4]
    assert len(indices) == len(set(indices))


def test_prefetch_scheduler_reprioritizes_when_navigation_reverses() -> None:
    """A direction change makes the new travel direction the next queued work."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    window._photos = [Path(f"{index:02}.cr3") for index in range(30)]
    window._current_index = 10
    started: list[Path] = []

    def start_load(path: Path) -> None:
        started.append(path)
        window._loading.add(path)

    window._start_photo_load = start_load
    window._schedule_prefetch(10)
    assert started == window._photos[10:13]

    window._loading.remove(window._photos[11])
    window._navigation_direction = -1
    window._schedule_prefetch(10)

    assert started[-1] == window._photos[9]
    window.close()
    app.quit()


def test_close_discards_queued_prefetches_and_waits_for_active_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Close does not destroy worker signals before active preview jobs finish."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    calls: list[str] = []

    class LoadPoolStub:
        def clear(self) -> None:
            calls.append("clear")

        def waitForDone(self) -> bool:
            calls.append("wait")
            return True

    window._load_pool = LoadPoolStub()
    window._queued_loads = {Path("queued.cr3")}
    event = QCloseEvent()

    window.closeEvent(event)

    assert calls == ["clear", "wait"]
    assert not window._queued_loads
    assert event.isAccepted()
    app.quit()


def test_open_folder_starts_on_selected_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening a folder with a dropped photo selects that file, not the first."""
    app = QApplication.instance() or QApplication([])
    first_photo = tmp_path / "first.cr3"
    selected_photo = tmp_path / "second.cr3"
    first_photo.touch()
    selected_photo.touch()
    window = CropWindow()
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    window.open_folder(tmp_path, selected_photo)

    assert window._current_index == 1
    assert window._current_path == selected_photo
    window.close()
    app.quit()


def test_open_folder_lists_cr2_and_cr3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed folder lists both Canon RAW formats without including other files."""
    app = QApplication.instance() or QApplication([])
    cr2_path = tmp_path / "first.CR2"
    cr3_path = tmp_path / "second.cr3"
    for path in (cr2_path, cr3_path, tmp_path / "notes.txt"):
        path.touch()
    window = CropWindow()
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    window.open_folder(tmp_path)

    assert window._all_photos == [cr2_path, cr3_path]
    window.close()
    app.quit()


def test_uncached_photo_keeps_previous_frame_while_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cache miss shows loading state without clearing the previous frame."""
    app = QApplication.instance() or QApplication([])
    photos = [tmp_path / "first.cr3", tmp_path / "second.cr3"]
    window = CropWindow()
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)
    previous_image = QImage(QSize(640, 480), QImage.Format.Format_RGB32)
    previous_image.fill(Qt.GlobalColor.white)
    window._photos = photos
    window.view.set_image(previous_image, 640, 480)
    image_key = window.view._image.cacheKey()

    window._show_photo(1)

    assert window.view.is_loading
    assert window.view._image.cacheKey() == image_key
    assert window.view._crop is None
    window.close()
    app.quit()


def test_crop_drop_area_accepts_and_emits_folder_drop(tmp_path: Path) -> None:
    """The Qt drop target accepts a local folder and emits its resolved path."""
    app = QApplication.instance() or QApplication([])
    drop_area = CropDropArea()
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(tmp_path))])
    emitted: list[tuple[Path, Path | None]] = []
    drop_area.path_dropped.connect(emitted.append)
    enter_event = QDragEnterEvent(
        QPoint(0, 0),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    drop_event = QDropEvent(
        QPointF(0, 0),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    drop_area.dragEnterEvent(enter_event)
    drop_area.dropEvent(drop_event)

    assert enter_event.isAccepted()
    assert drop_event.isAccepted()
    assert emitted == [(tmp_path, None)]
    drop_area.close()
    app.quit()


def test_parse_aspect_ratios_reads_ratio_labels() -> None:
    """Convert integer and decimal ratio strings into labeled crop ratios."""
    ratios = parse_aspect_ratios(["4:5", "16:9", "1.91:1"])

    assert [(ratio.width, ratio.height, ratio.label) for ratio in ratios] == [
        (4, 5, "4:5"),
        (16, 9, "16:9"),
        (191, 100, "1.91:1"),
    ]
    assert ratios[2].value == pytest.approx(1.91)


def test_save_status_keeps_a_fixed_toolbar_width() -> None:
    """Save-state text changes do not resize the toolbar status slot."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    status_width = window.save_label.width()

    for status in ("Saving...", "Saved", "Unsaved", " "):
        window.save_label.setText(status)
        window.save_label.adjustSize()
        assert window.save_label.width() == status_width

    window.close()
    app.quit()


def test_lock_and_snap_icon_controls_reflect_toggle_state() -> None:
    """Lock and snap symbols and accessible labels track their checked states."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()

    assert "←" in window.previous_button.toolTip()
    assert "→" in window.next_button.toolTip()
    assert "⌘O" in window.open_button.toolTip()
    assert "1–8" in window.ratio_combo.toolTip()
    assert "⌘S" in window.save_label.toolTip()
    assert "H" in window.show_horizon_button.toolTip()
    assert window.lock_checkbox.isCheckable()
    assert window.aspect_group.title() == "Aspect"
    assert not window.lock_checkbox.isChecked()
    assert "unlocked" in window.lock_checkbox.toolTip().lower()
    assert "L" in window.lock_checkbox.toolTip()
    window.lock_checkbox.setChecked(True)
    assert "locked" in window.lock_checkbox.toolTip().lower()

    assert window.snap_checkbox.isChecked()
    enabled_icon = window.snap_checkbox.icon().cacheKey()
    window.snap_checkbox.setChecked(False)
    assert window.snap_checkbox.icon().cacheKey() != enabled_icon
    assert "disabled" in window.snap_checkbox.toolTip().lower()
    assert "S" in window.snap_checkbox.toolTip()
    icon_image = window.snap_checkbox.icon().pixmap(32, 32).toImage()
    red_pixels = 0
    light_pixels = 0
    for y in range(icon_image.height()):
        for x in range(icon_image.width()):
            color = icon_image.pixelColor(x, y)
            red_pixels += color.red() > 160 and color.green() < 130
            light_pixels += color.red() > 180 and color.green() > 180 and color.blue() > 180
    assert red_pixels > 0
    assert light_pixels > 0

    window.close()
    app.quit()


def test_compact_navigation_and_filename_reveal_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toolbar arrows stay icon-only and clicking the filename reveals its photo."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("/tmp/selected.cr3")
    window._current_path = raw_path
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        crop_app.QProcess,
        "startDetached",
        staticmethod(lambda program, arguments: calls.append((program, arguments))),
    )

    assert window.open_button.text() == "Open"
    assert window.previous_button.text() == ""
    assert window.next_button.text() == ""
    assert not window.previous_button.icon().isNull()
    assert not window.next_button.icon().isNull()
    toolbar = window.rating_filter_panel.parentWidget().layout().itemAt(0).layout()
    assert (
        toolbar.indexOf(window.previous_button.parentWidget())
        < toolbar.indexOf(window.rating_filter_panel)
        < toolbar.indexOf(window.position_label)
        < toolbar.indexOf(window.angle_group)
    )
    window.position_label.clicked.emit()

    assert calls == [("open", ["-R", str(raw_path.resolve())])]
    window.close()
    app.quit()


def test_side_by_side_preview_is_optional_and_tracks_editing_state() -> None:
    """The result pane starts hidden and mirrors the active image, crop, and angle."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()

    assert not window.side_by_side_button.isChecked()
    assert window.cropped_preview.parentWidget().isHidden()
    assert "Live Preview" in window.side_by_side_button.toolTip()
    labels = [w.text() for w in window.cropped_preview.parentWidget().findChildren(QLabel)]
    assert "Live Preview" in labels

    p_shortcut = next(
        s for s in window.findChildren(QShortcut) if s.key() == QKeySequence(Qt.Key.Key_P)
    )
    p_shortcut.activated.emit()
    assert window.side_by_side_button.isChecked()
    assert not window.cropped_preview.parentWidget().isHidden()
    assert "Hide Live Preview" in window.side_by_side_button.toolTip()

    p_shortcut.activated.emit()
    assert not window.side_by_side_button.isChecked()
    assert window.cropped_preview.parentWidget().isHidden()

    window.side_by_side_button.click()
    assert not window.cropped_preview.parentWidget().isHidden()

    image = QImage(QSize(640, 480), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.green)
    crop = CropRect(0.2, 0.1, 0.8, 0.9)
    window._set_preview_image(image, 640, 480)
    window._current_path = Path("preview.cr3")
    window._current_metadata = PhotoMetadata(640, 480, CropRect(0, 0, 1, 1))
    window.view._set_edited_crop(crop)
    window._set_preview_rotation(6.5)

    target = window.cropped_preview._target_rect()
    assert target is not None
    start = target.center().toPoint()
    QTest.mousePress(window.cropped_preview, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(window.cropped_preview, QPoint(start.x() + 20, start.y()))
    QTest.mouseRelease(
        window.cropped_preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(start.x() + 20, start.y()),
    )

    assert window.view._image.cacheKey() == window.cropped_preview._image.cacheKey()
    assert window.view._crop == window.cropped_preview._crop
    assert window.view._crop is not None and window.view._crop.left < crop.left
    assert window.view._rotation_angle == window.cropped_preview._rotation_angle == 6.5

    window._save_timer.stop()
    window._dirty = False
    window.close()
    app.quit()


def test_auto_and_show_shortcuts_invoke_their_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A invokes Auto and H toggles candidate visibility through the button."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    shortcuts = {
        shortcut.key().toString(): shortcut
        for shortcut in window.findChildren(QShortcut)
    }
    analysis_modes: list[bool] = []
    monkeypatch.setattr(
        window,
        "_analyze_horizon",
        lambda *, apply_best: analysis_modes.append(apply_best),
    )
    window.auto_level_button.setEnabled(True)
    window.show_horizon_button.setEnabled(True)

    assert "A" in shortcuts
    assert "A" in window.auto_level_button.toolTip()

    shortcuts["H"].activated.emit()
    assert window.show_horizon_button.isChecked()
    assert analysis_modes == [False]
    shortcuts["H"].activated.emit()
    assert not window.show_horizon_button.isChecked()
    assert analysis_modes == [False]

    window.close()
    app.quit()


def test_star_and_color_filter_menus_combine_and_mark_filtered_count() -> None:
    """Menu selections combine minimum stars with a color label and mark the count."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    photos = [Path(f"photo-{index}.cr3") for index in range(3)]
    window._all_photos = photos
    window._photos = list(photos)
    window._photo_ratings = {
        photos[0]: (2, "Red"),
        photos[1]: (4, "Red"),
        photos[2]: (5, "Blue"),
    }
    window._ratings_loaded = True
    window._current_path = photos[1]
    window._current_index = 1

    window._star_filter_actions[4].trigger()

    assert window._photos == photos[1:]
    assert window._current_path == photos[1]
    assert window.position_count_label.text().startswith("1 / 2 [Filtered]")
    assert window.star_filter_button.text() == "4+"

    window._color_filter_actions["Red"].trigger()

    assert window._photos == [photos[1]]
    assert window._current_path == photos[1]
    assert window.position_count_label.text().startswith("1 / 1 [Filtered]")
    assert window.color_filter_button.text() == ""
    assert not window.color_filter_button.icon().isNull()

    window._star_filter_actions[None].trigger()
    assert window._photos == [photos[0], photos[1]]
    assert window.position_count_label.text().startswith("2 / 2 [Filtered]")
    window.close()
    app.quit()


def test_rating_filters_show_no_matches_and_clear_back_to_all_photos() -> None:
    """Unlabeled/empty filters remain visible and clearing them restores browsing."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    photos = [Path("labeled.cr3"), Path("other.cr3")]
    window._all_photos = photos
    window._photos = list(photos)
    window._photo_ratings = {
        photos[0]: (2, "Red"),
        photos[1]: (1, "Blue"),
    }
    window._ratings_loaded = True
    window._current_path = photos[0]
    window._current_index = 0
    shown: list[int] = []
    window._show_photo = shown.append

    window._set_color_filter("")

    assert window.color_filter_button.text() == ""
    assert window._photos == []
    assert window._current_path is None
    assert window.position_count_label.text().startswith("0 / 0 [Filtered]")
    assert not window.rotation_slider.isEnabled()

    window._set_color_filter(None)

    assert window._photos == photos
    assert shown == [0]
    window.close()
    app.quit()


def test_filter_scan_overlay_covers_disabled_rating_controls() -> None:
    """The startup scan message overlays both filter buttons until scan completion."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()

    assert window.star_filter_button.text() == "*"
    assert window.color_filter_button.text() == ""
    window._start_rating_scan([Path("one.cr3")])

    assert not window.star_filter_button.isEnabled()
    assert not window.color_filter_button.isEnabled()
    assert window.rating_filter_panel.scan_overlay.text() == "Scanning ratings"
    assert not window.rating_filter_panel.scan_overlay.isHidden()

    window._rating_scan_generation += 1
    window._rating_pool.clear()
    window.close()
    app.quit()


@pytest.mark.parametrize("value", ["bad", "1:0", "0:1", "-4:5"])
def test_parse_aspect_ratios_rejects_invalid_values(value: str) -> None:
    """Reject malformed or nonpositive aspect-ratio settings clearly."""
    with pytest.raises(ValueError, match="Invalid aspect ratio"):
        parse_aspect_ratios([value])


def test_load_crop_settings_reads_custom_ratios_and_tolerance(tmp_path: Path) -> None:
    """Load user-adjustable ratios and snap tolerance from TOML."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text(
        '[crop_tool]\naspect_ratios = ["4:5", "3:2"]\nsnap_tolerance = 0.04\n',
        encoding="utf-8",
    )

    ratios, tolerance = load_crop_settings(config_path)

    assert [ratio.label for ratio in ratios] == ["4:5", "3:2"]
    assert tolerance == pytest.approx(0.04)


def test_load_crop_settings_defaults_to_wider_snap_range(tmp_path: Path) -> None:
    """The default relative snap tolerance is five percent when no config exists."""
    ratios, tolerance = load_crop_settings(tmp_path / "missing.toml")

    assert ratios
    assert tolerance == pytest.approx(0.05)


def test_load_crop_settings_rejects_out_of_range_tolerance(tmp_path: Path) -> None:
    """Reject snap tolerances outside the supported normalized range."""
    config_path = tmp_path / "photo-workflow.toml"
    config_path.write_text("[crop_tool]\nsnap_tolerance = 1.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="snap_tolerance"):
        load_crop_settings(config_path)


def test_decode_preview_bounds_large_images() -> None:
    """Keep a full-resolution embedded preview within the display cache limit."""
    source = QImage(QSize(6000, 4000), QImage.Format.Format_RGB32)
    source.fill(0)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert source.save(buffer, "JPEG")

    preview = decode_preview(bytes(data))

    assert (preview.width(), preview.height()) == (2560, 1706)


@pytest.mark.parametrize("orientation", [6, 8])
def test_decode_preview_applies_raw_portrait_orientation(orientation: int) -> None:
    """Rotate an untagged embedded landscape preview using the RAW orientation."""
    source = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    source.fill(0)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert source.save(buffer, "JPEG")

    preview = decode_preview(bytes(data), fallback_orientation=orientation)

    assert (preview.width(), preview.height()) == (400, 600)


def test_decode_preview_does_not_rotate_normal_orientation() -> None:
    """Leave an untagged preview unchanged when RAW orientation is normal."""
    source = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    source.fill(0)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    assert source.save(buffer, "JPEG")

    preview = decode_preview(bytes(data), fallback_orientation=1)

    assert (preview.width(), preview.height()) == (600, 400)


def test_decode_preview_does_not_apply_raw_fallback_twice(tmp_path: Path) -> None:
    """Trust embedded JPEG orientation metadata instead of applying RAW fallback twice."""
    source = Image.new("RGB", (600, 400), "white")
    exif = Image.Exif()
    exif[274] = 6
    preview_path = tmp_path / "oriented.jpg"
    source.save(preview_path, exif=exif)

    preview = decode_preview(preview_path.read_bytes(), fallback_orientation=8)

    assert (preview.width(), preview.height()) == (400, 600)


def test_saved_crop_updates_cached_metadata_for_revisit(tmp_path: Path) -> None:
    """Revisiting a photo restores the crop that was written, not its old crop."""
    raw_path = tmp_path / "photo.cr3"
    old_crop = CropRect(0.1, 0.1, 0.9, 0.9)
    saved_crop = CropRect(0.2, 0.1, 0.8, 0.9)
    metadata_cache = {
        raw_path: PhotoMetadata(6000, 4000, old_crop, crop_angle=1.5),
    }

    updated = update_saved_crop_cache(
        metadata_cache,
        raw_path,
        saved_crop,
        crop_angle=-12.5,
    )

    assert updated is not None
    assert metadata_cache[raw_path].crop == saved_crop
    assert metadata_cache[raw_path].crop_angle == pytest.approx(-12.5)


def test_lightroom_crop_angle_is_shown_with_opposite_user_facing_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Lightroom XMP angle of -10 displays as a user-facing +10 degrees."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("portrait.cr3")
    metadata = PhotoMetadata(
        image_width=4000,
        image_height=6000,
        crop=CropRect(0, 0, 1, 1),
        crop_angle=-10,
        orientation=8,
    )
    window._current_path = raw_path
    monkeypatch.setattr(window, "_select_crop_ratio", lambda crop: None)
    monkeypatch.setattr(window, "_update_crop_mode", lambda *args: None)

    window._apply_metadata(raw_path, metadata)

    assert window.rotation_slider.value() == 100
    assert window.rotation_value.text() == "10.0°"
    assert window._current_rotation == pytest.approx(10)
    assert window.view._rotation_angle == pytest.approx(10)
    window.rotation_slider.setValue(125)
    window._save_timer.stop()
    assert window.view._rotation_angle == pytest.approx(12.5)
    assert window.rotation_value.text() == "12.5°"
    window.rotation_reset.click()
    window._save_timer.stop()
    assert window.rotation_slider.value() == 0
    assert window.rotation_value.text() == "0.0°"
    assert window.view._rotation_angle == pytest.approx(0)
    assert window._dirty
    window.close()
    app.quit()


def test_loaded_rotated_crop_is_constrained_and_marked_for_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing crop outside the rotated photo is corrected before display/save."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("rotated.cr3")
    original_crop = CropRect(0.05, 0.05, 0.95, 0.95)
    metadata = PhotoMetadata(400, 300, original_crop, crop_angle=-30)
    image = QImage(QSize(400, 300), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    window._current_path = raw_path
    window._preview_cache[raw_path] = image
    monkeypatch.setattr(window, "_select_crop_ratio", lambda crop: None)
    monkeypatch.setattr(window, "_update_crop_mode", lambda *args: None)

    window._apply_metadata(raw_path, metadata)
    window._save_timer.stop()

    assert window._current_crop is not None
    assert window._current_crop != original_crop
    assert window._current_metadata is not None
    assert window._current_metadata.crop == window._current_crop
    assert window.view._crop == window.cropped_preview._crop == window._current_crop
    assert window._dirty
    window.close()
    app.quit()


def test_revert_restores_current_photos_starting_crop_and_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revert restores the selected photo's starting crop and angle and queues a save."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("revert.cr3")
    original_crop = CropRect(0.15, 0.12, 0.82, 0.86)
    metadata = PhotoMetadata(640, 480, original_crop, crop_angle=-2.3)
    preview = QImage(QSize(640, 480), QImage.Format.Format_RGB32)
    preview.fill(Qt.GlobalColor.black)
    window._current_path = raw_path
    window._preview_cache[raw_path] = preview
    monkeypatch.setattr(window, "_select_crop_ratio", lambda crop: None)
    monkeypatch.setattr(window, "_update_crop_mode", lambda *args: None)

    window._apply_metadata(raw_path, metadata)
    window._save_timer.stop()
    assert not window.revert_button.isEnabled()
    assert window._starting_crop is not None
    assert window._starting_crop.left == pytest.approx(original_crop.left)
    assert window._starting_rotation == pytest.approx(2.3)

    edited_crop = CropRect(0.25, 0.2, 0.75, 0.8)
    window._crop_edited(edited_crop)
    window.rotation_slider.setValue(47)
    window._save_timer.stop()
    assert window.revert_button.isEnabled()

    window.revert_button.click()
    window._save_timer.stop()

    assert window._current_crop is not None
    assert window.view._crop == window.cropped_preview._crop == window._current_crop
    assert window._current_crop.left == pytest.approx(original_crop.left)
    assert window._current_crop.top == pytest.approx(original_crop.top)
    assert window._current_crop.right == pytest.approx(original_crop.right)
    assert window._current_crop.bottom == pytest.approx(original_crop.bottom)
    assert window._current_rotation == pytest.approx(2.3)
    assert window.rotation_slider.value() == 23
    assert window._dirty
    assert not window.revert_button.isEnabled()
    window._dirty = False
    window.close()
    app.quit()


def test_auto_level_applies_best_candidate_and_clicking_alternate_updates_angle() -> None:
    """Auto-level applies the strongest line suggestion and retains alternatives."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("horizon.cr3")
    preview = QImage(QSize(640, 480), QImage.Format.Format_RGB32)
    preview.fill(Qt.GlobalColor.black)
    window._current_path = raw_path
    window._current_metadata = PhotoMetadata(640, 480, CropRect(0, 0, 1, 1))
    window._preview_cache[raw_path] = preview

    window._horizon_analysis_finished(
        str(raw_path),
        [
            HorizonCandidate(-3.4, 600, (0, 0.4), (1, 0.46)),
            HorizonCandidate(2.1, 420, (0, 0.7), (1, 0.67)),
        ],
        "",
        True,
    )
    window._save_timer.stop()

    assert window.rotation_slider.value() == -34
    assert window.rotation_value.text() == "-3.4°"
    assert len(window._horizon_candidates) == 2
    assert window.view._selected_horizon_candidate == 0

    window._horizon_guide_selected(1)
    window._save_timer.stop()

    assert window.rotation_slider.value() == 21
    assert window.rotation_value.text() == "2.1°"
    assert window._current_rotation == pytest.approx(2.1)
    assert window.view._selected_horizon_candidate == 1
    window._horizon_guide_selected(0)
    window._save_timer.stop()
    assert window.rotation_slider.value() == -34
    assert window.view._selected_horizon_candidate == 0
    window.close()
    app.quit()


def test_show_horizon_candidates_does_not_change_angle_until_candidate_selected() -> None:
    """Show reveals unselected guides and preserves rotation until user chooses one."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("horizon.cr3")
    preview = QImage(QSize(640, 480), QImage.Format.Format_RGB32)
    preview.fill(Qt.GlobalColor.black)
    original_angle = 1.7
    window._current_path = raw_path
    window._current_rotation = original_angle
    window._current_metadata = PhotoMetadata(640, 480, CropRect(0, 0, 1, 1))
    window._preview_cache[raw_path] = preview
    assert window.angle_group.title() == "Angle"
    assert not window.auto_level_button.icon().isNull()
    assert window.auto_level_button.accessibleName() == "Auto level"
    auto_hint = f"({crop_app.key_hint(crop_app.AUTO_LEVEL_KEY)})"
    assert auto_hint in window.auto_level_button.toolTip()
    assert not window.show_horizon_button.icon().isNull()
    assert window.show_horizon_button.accessibleName() == "Show horizon candidates"
    window.rotation_slider.setValue(17)
    window.rotation_value.setText("1.7°")
    window._dirty = False

    window._horizon_analysis_finished(
        str(raw_path),
        [
            HorizonCandidate(-3.4, 600, (0, 0.4), (1, 0.46)),
            HorizonCandidate(2.1, 420, (0, 0.7), (1, 0.67)),
        ],
        "",
        False,
    )

    assert auto_hint in window.auto_level_button.toolTip()
    assert window.rotation_slider.value() == 17
    assert window.rotation_value.text() == "1.7°"
    assert window._current_rotation == pytest.approx(original_angle)
    assert not window._dirty
    assert len(window._horizon_candidates) == 2
    assert window.view._selected_horizon_candidate == -1
    assert window.show_horizon_button.isChecked()
    shown_icon = window.show_horizon_button.icon().cacheKey()

    window.show_horizon_button.click()
    assert not window.show_horizon_button.isChecked()
    assert window.view._horizon_candidates == ()
    assert window._current_rotation == pytest.approx(original_angle)
    hidden_icon = window.show_horizon_button.icon().cacheKey()
    assert hidden_icon != shown_icon

    window.show_horizon_button.click()
    assert window.show_horizon_button.isChecked()
    assert len(window.view._horizon_candidates) == 2
    assert window._current_rotation == pytest.approx(original_angle)

    window._horizon_guide_selected(1)
    window._save_timer.stop()
    assert window.rotation_slider.value() == 21
    assert window._current_rotation == pytest.approx(2.1)
    assert window.view._selected_horizon_candidate == 1
    window.close()
    app.quit()


def test_consecutive_ratio_selection_preserves_base_crop_without_shrinking() -> None:
    """Flipping between ratios without panning or resizing does not shrink the crop."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("sample.cr3")
    original_crop = CropRect(0.0, 0.0, 1.0, 1.0)
    metadata = PhotoMetadata(6000, 4000, original_crop)
    preview = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    preview.fill(Qt.GlobalColor.black)
    window._current_path = raw_path
    window._preview_cache[raw_path] = preview
    window._apply_metadata(raw_path, metadata)
    window._save_timer.stop()

    window._apply_ratio_index(1)
    crop_1_1_first = window._current_crop
    assert crop_1_1_first is not None

    window._apply_ratio_index(2)
    crop_4_5_first = window._current_crop
    assert crop_4_5_first is not None

    window._apply_ratio_index(1)
    assert window._current_crop == crop_1_1_first

    window._apply_ratio_index(2)
    assert window._current_crop == crop_4_5_first

    panned_crop = CropRect(0.1, 0.05, 0.9, 0.85)
    window._crop_edited(panned_crop)
    assert window._ratio_base_crop is None

    window._apply_ratio_index(1)
    assert window._ratio_base_crop == panned_crop
    assert window._current_crop != crop_1_1_first

    window.close()
    app.quit()


def test_numeric_shortcuts_select_configured_ratios() -> None:
    """Keys 1-8 select configured ratios 1 through 8, not Free."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()

    for shortcut in window.findChildren(QShortcut):
        if shortcut.key() == QKeySequence(Qt.Key.Key_1):
            shortcut.activated.emit()
            break

    assert window.ratio_combo.currentIndex() == 1
    assert window.ratio_combo.currentText() == "1:1"

    for shortcut in window.findChildren(QShortcut):
        if shortcut.key() == QKeySequence(Qt.Key.Key_2):
            shortcut.activated.emit()
            break

    assert window.ratio_combo.currentIndex() == 2
    assert window.ratio_combo.currentText() == "4:5"

    window.close()
    app.quit()


def test_angle_slider_rotation_preserves_crop_aspect_ratio() -> None:
    """Rotating the angle slider maintains the crop's physical aspect ratio on screen."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    raw_path = Path("sample.cr3")
    crop0 = CropRect(0.25, 0.125, 0.75, 0.875)
    metadata = PhotoMetadata(6000, 4000, crop0)
    preview = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    preview.fill(Qt.GlobalColor.black)
    window._current_path = raw_path
    window._preview_cache[raw_path] = preview
    window._apply_metadata(raw_path, metadata)
    window._save_timer.stop()

    assert window.view._snap_ratio is not None
    assert window.view._snap_ratio.label == "1:1"

    window.rotation_slider.setValue(150)
    window._save_timer.stop()

    image_rect = window.view._image_rect()
    crop_rect = window.view._crop_rect(image_rect)
    assert crop_rect is not None
    assert crop_rect.width() == pytest.approx(crop_rect.height(), rel=1e-3)
    assert window.view._snap_ratio is not None
    assert window.view._snap_ratio.label == "1:1"

    window.close()
    app.quit()


def test_cached_photo_navigation_does_not_blink_loading_state() -> None:
    """Navigating to a previously cached photo immediately displays without loading veil."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    photo_1 = Path("first.cr3")
    photo_2 = Path("second.cr3")
    window._photos = [photo_1, photo_2]
    window._current_index = 0
    window._current_path = photo_1

    img1 = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    img2 = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    meta1 = PhotoMetadata(6000, 4000, CropRect(0, 0, 1, 1))
    meta2 = PhotoMetadata(6000, 4000, CropRect(0.1, 0.1, 0.9, 0.9))

    window._preview_cache[photo_1] = img1
    window._preview_cache[photo_2] = img2
    window._metadata_cache[photo_1] = meta1
    window._metadata_cache[photo_2] = meta2

    window._show_photo(1)
    assert not window.view.is_loading
    assert window._current_crop == meta2.crop

    window._show_photo(0)
    assert not window.view.is_loading
    assert window._current_crop == meta1.crop

    window.close()
    app.quit()


def test_should_restore_last_session_folder_and_photo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When requested, restore the remembered folder and photo from settings."""
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "photos"
    folder.mkdir()
    photo_a = folder / "first.cr3"
    photo_b = folder / "second.cr3"
    photo_a.touch()
    photo_b.touch()

    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    settings.setValue("last_folder", str(folder.resolve()))
    settings.setValue("last_photo", str(photo_b.resolve()))
    settings.sync()

    window = CropWindow(restore_last_session=True, settings=settings)
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    assert window._current_path == photo_b
    assert window._current_index == 1
    window.close()
    app.quit()


def test_should_ignore_missing_session_folder(tmp_path: Path) -> None:
    """If the saved folder no longer exists, remain in empty initial state."""
    app = QApplication.instance() or QApplication([])
    missing_folder = tmp_path / "gone"
    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    settings.setValue("last_folder", str(missing_folder.resolve()))
    settings.setValue("last_photo", str((missing_folder / "img.cr3").resolve()))
    settings.sync()

    window = CropWindow(restore_last_session=True, settings=settings)

    assert window._current_path is None
    assert window.position_count_label.text() == "No folder open"
    window.close()
    app.quit()


def test_should_fallback_to_first_photo_when_saved_photo_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the saved photo is missing from the folder, default to the first photo."""
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "photos"
    folder.mkdir()
    photo_a = folder / "first.cr3"
    photo_a.touch()

    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    settings.setValue("last_folder", str(folder.resolve()))
    settings.setValue("last_photo", str((folder / "missing.cr3").resolve()))
    settings.sync()

    window = CropWindow(restore_last_session=True, settings=settings)
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    assert window._current_path == photo_a
    assert window._current_index == 0
    window.close()
    app.quit()


def test_should_prefer_explicit_target_over_saved_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit target overrides the saved session in settings."""
    app = QApplication.instance() or QApplication([])
    saved_folder = tmp_path / "saved"
    saved_folder.mkdir()
    (saved_folder / "saved.cr3").touch()

    explicit_folder = tmp_path / "explicit"
    explicit_folder.mkdir()
    explicit_photo = explicit_folder / "target.cr3"
    explicit_photo.touch()

    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    settings.setValue("last_folder", str(saved_folder.resolve()))
    settings.sync()

    window = CropWindow(explicit_folder, restore_last_session=True, settings=settings)
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    assert window._current_path == explicit_photo
    window.close()
    app.quit()


def test_should_record_session_state_on_navigation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Displaying photos updates last_folder and last_photo in settings."""
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "photos"
    folder.mkdir()
    photo_a = folder / "first.cr3"
    photo_b = folder / "second.cr3"
    photo_a.touch()
    photo_b.touch()

    settings_path = tmp_path / "settings.ini"
    settings = QSettings(str(settings_path), QSettings.Format.IniFormat)
    window = CropWindow(settings=settings)
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    window.open_folder(folder)
    assert settings.value("last_folder") == str(folder.resolve())
    assert settings.value("last_photo") == str(photo_a.resolve())

    window.navigate(1)
    assert settings.value("last_photo") == str(photo_b.resolve())

    window.close()
    app.quit()


def test_should_open_raw_photo_passed_as_initial_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a RAW photo file path as initial target selects that photo."""
    app = QApplication.instance() or QApplication([])
    folder = tmp_path / "photos"
    folder.mkdir()
    photo_a = folder / "first.cr3"
    photo_b = folder / "second.cr2"
    photo_a.touch()
    photo_b.touch()

    window = CropWindow(photo_b)
    monkeypatch.setattr(window, "_schedule_prefetch", lambda index: None)

    assert window._current_path == photo_b
    assert window._current_index == 1
    window.close()
    app.quit()


def test_should_enable_session_restore_in_main_when_no_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Main restores session if no path was passed, but not if one was provided."""
    launched_args: list[tuple[object, bool, bool]] = []

    class DummyCropWindow:
        def __init__(
            self,
            folder: object = None,
            *,
            restore_last_session: bool = False,
            debug: bool = False,
            **kwargs: object,
        ) -> None:
            launched_args.append((folder, restore_last_session, debug))

        def show(self) -> None:
            pass

    monkeypatch.setattr(crop_app, "CropWindow", DummyCropWindow)
    monkeypatch.setattr(crop_app.QApplication, "exec", lambda self: 0)

    crop_app.main([])
    assert launched_args == [(None, True, False)]

    launched_args.clear()
    crop_app.main(["/some/folder"])
    assert launched_args == [(Path("/some/folder"), False, False)]

    launched_args.clear()
    crop_app.main(["--debug"])
    assert launched_args == [(None, True, True)]


def test_evict_excess_previews_preserves_current_photo() -> None:
    """Cache eviction never evicts the photo currently being viewed."""
    app = QApplication.instance() or QApplication([])
    window = CropWindow()
    current = Path("current.cr3")
    window._current_path = current
    window._preview_cache[current] = QImage(10, 10, QImage.Format.Format_RGB32)

    for i in range(crop_app.PREVIEW_CACHE_SIZE + 5):
        p = Path(f"other_{i}.cr3")
        window._preview_cache[p] = QImage(10, 10, QImage.Format.Format_RGB32)
        window._evict_excess_previews()

    assert current in window._preview_cache
    assert len(window._preview_cache) <= crop_app.PREVIEW_CACHE_SIZE
    window.close()
    app.quit()


def test_status_bar_displays_diagnostics() -> None:
    """The bottom status bar shows DEBUG prefix and cache/worker status when enabled."""
    app = QApplication.instance() or QApplication([])
    window_default = CropWindow()
    assert window_default.statusBar().isHidden()
    assert window_default._status_label.text().startswith("DEBUG | ")
    assert "Cache:" in window_default._status_label.text()
    assert "In-flight:" in window_default._status_label.text()
    assert "Queued:" in window_default._status_label.text()
    assert "Ratings:" in window_default._status_label.text()
    window_default.close()

    window_debug = CropWindow(debug=True)
    assert not window_debug.statusBar().isHidden()
    assert window_debug._status_label.text().startswith("DEBUG | ")
    window_debug.close()
    app.quit()


def test_parse_args_handles_debug_flag() -> None:
    """--debug flag is parsed into args.debug."""
    args_default = crop_app.parse_args([])
    assert not args_default.debug
    assert args_default.folder is None

    args_debug = crop_app.parse_args(["--debug"])
    assert args_debug.debug
    assert args_debug.folder is None

    args_folder_debug = crop_app.parse_args(["/path/to/photos", "--debug"])
    assert args_folder_debug.debug
    assert args_folder_debug.folder == Path("/path/to/photos")
