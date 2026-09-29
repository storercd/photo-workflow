"""Tests for crop application settings and ratio selection helpers."""

from pathlib import Path

import pytest
from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QSize
from PySide6.QtGui import QImage

from photo_workflow.crop_tool.app import (
    decode_preview,
    load_crop_settings,
    parse_aspect_ratios,
    update_saved_crop_cache,
)
from photo_workflow.crop_tool.metadata import PhotoMetadata
from photo_workflow.crop_tool.model import CropRect


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
