"""Tests for crop application settings and ratio selection helpers."""

from pathlib import Path

import pytest
from PySide6.QtCore import (
    QBuffer,
    QByteArray,
    QIODevice,
    QMimeData,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QUrl,
)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
from PySide6.QtWidgets import QApplication

from photo_workflow.crop_tool.app import (
    CropDropArea,
    CropWindow,
    decode_preview,
    find_photo_index,
    load_crop_settings,
    parse_aspect_ratios,
    resolve_drop_path,
    update_saved_crop_cache,
)
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
    monkeypatch.setattr(window, "_queue_load", lambda path: None)
    monkeypatch.setattr(window, "_prefetch_neighbors", lambda index: None)

    window.open_folder(tmp_path, selected_photo)

    assert window._current_index == 1
    assert window._current_path == selected_photo
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


def test_saved_crop_updates_cached_metadata_for_revisit(tmp_path: Path) -> None:
    """Revisiting a photo restores the crop that was written, not its old crop."""
    raw_path = tmp_path / "photo.cr3"
    old_crop = CropRect(0.1, 0.1, 0.9, 0.9)
    saved_crop = CropRect(0.2, 0.1, 0.8, 0.9)
    metadata_cache = {
        raw_path: PhotoMetadata(6000, 4000, old_crop, crop_angle=1.5),
    }

    updated = update_saved_crop_cache(metadata_cache, raw_path, saved_crop)

    assert updated is not None
    assert metadata_cache[raw_path].crop == saved_crop
    assert metadata_cache[raw_path].crop_angle == pytest.approx(1.5)
