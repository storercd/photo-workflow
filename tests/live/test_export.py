"""Tests for the Live Export stage."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from photo_workflow.editor.metadata import PhotoMetadata
from photo_workflow.editor.model import CropRect
from photo_workflow.live.api import StageRegistry
from photo_workflow.live.features import export
from photo_workflow.live.layout import LiveLayout
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import StageId, StageStatus

PHOTO_NAME = "20251031_101500_00_IMG_0001.CR3"


def jpeg_bytes(size: tuple[int, int], orientation: int | None = None) -> bytes:
    """
    Build a JPEG whose left half is red and right half is blue.

    Returns:
        The encoded bytes.
    """
    image = Image.new("RGB", size, "blue")
    image.paste(Image.new("RGB", (size[0] // 2, size[1]), "red"), (0, 0))
    exif = Image.Exif()
    if orientation:
        exif[export.EXIF_ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=100, exif=exif)
    return buffer.getvalue()


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    """
    Create a fake renamed original inside a Live date folder.

    Returns:
        The path of the original.
    """
    path = tmp_path / "live" / "2025" / "10" / "20251031" / "input" / PHOTO_NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"raw")
    return path


def make_job(photo: Path, source: Path, source_stage: StageId | None = None) -> export.StageJob:
    """
    Build an Export job for a photo.

    Returns:
        The job.
    """
    return export.StageJob(
        stage=StageId.EXPORT,
        photo=photo,
        source=source,
        source_stage=source_stage,
        layout=LiveLayout(photo.parent.parent.parent.parent.parent),
        settings=LiveSettings(),
        warn=lambda message: None,
    )


def test_apply_crop_cuts_the_normalized_rectangle() -> None:
    """Apply crop cuts the normalized rectangle."""
    image = Image.new("RGB", (200, 100), "red")

    cropped = export.apply_crop(image, CropRect(0.25, 0.1, 0.75, 0.6), 0)

    assert cropped.size == (100, 50)


def test_apply_crop_without_crop_returns_the_full_image() -> None:
    """Apply crop without crop returns the full image."""
    image = Image.new("RGB", (200, 100))

    assert export.apply_crop(image, CropRect(0, 0, 1, 1), 0) is image


def test_apply_crop_straightens_inside_the_rotated_bounds() -> None:
    """Apply crop straightens inside the rotated bounds."""
    image = Image.new("RGB", (100, 100), "white")

    cropped = export.apply_crop(image, CropRect(0.25, 0.25, 0.75, 0.75), 45)

    assert all(abs(side - 71) <= 1 for side in cropped.size)  # half of the 141 px bounds


def test_load_image_source_applies_its_own_orientation(tmp_path: Path) -> None:
    """Load image source applies its own orientation."""
    source = tmp_path / "x.jpg"
    source.write_bytes(jpeg_bytes((40, 20), orientation=6))

    image, exif = export.load_source_image(source, 1)

    assert image.size == (20, 40)
    assert exif[export.EXIF_ORIENTATION_TAG] == 1


def test_load_raw_uses_the_raw_orientation_when_the_preview_has_none(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load raw uses the raw orientation when the preview has none."""
    monkeypatch.setattr(export, "extract_preview", lambda path, tag: jpeg_bytes((40, 20)))

    image, _ = export.load_source_image(photo, 6)

    assert image.size == (20, 40)


def test_load_raw_falls_back_to_the_next_preview_tag(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Load raw falls back to the next preview tag."""

    def fake_extract(path: Path, tag: str) -> bytes:
        if tag == "JpgFromRaw":
            raise RuntimeError("missing")
        return jpeg_bytes((40, 20))

    monkeypatch.setattr(export, "extract_preview", fake_extract)

    image, _ = export.load_source_image(photo, 1)

    assert image.size == (40, 20)


def test_load_raw_without_any_preview_fails(photo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Load raw without any preview fails."""

    def fail(path: Path, tag: str) -> bytes:
        raise RuntimeError("missing")

    monkeypatch.setattr(export, "extract_preview", fail)

    with pytest.raises(RuntimeError, match="No usable embedded preview"):
        export.load_source_image(photo, 1)


def test_process_exports_the_cropped_source_into_the_export_folder(
    photo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process exports the cropped source into the export folder."""
    source = photo.parent.parent / "output" / "20251031_101500_00_IMG_0001.jpg"
    source.parent.mkdir()
    source.write_bytes(jpeg_bytes((400, 200)))
    metadata = PhotoMetadata(400, 200, CropRect(0, 0, 0.5, 1), orientation=1)
    monkeypatch.setattr(export, "read_photo_metadata", lambda raw, xmp: metadata)

    result = export.Export().process(make_job(photo, source, StageId.LIGHTING))

    target = photo.parent.parent / "export" / "20251031_101500_00_IMG_0001.jpg"
    assert result.status is StageStatus.DONE
    assert result.outputs == (target,)
    with Image.open(target) as exported:
        assert exported.size == (200, 200)
        assert exported.getpixel((100, 100))[0] > 200  # only the red half remains
    assert not list(target.parent.glob(".*.part"))


def test_register_provides_the_export_stage() -> None:
    """Register provides the export stage."""
    registry = StageRegistry()
    export.register(registry)

    assert registry.is_registered(StageId.EXPORT)
