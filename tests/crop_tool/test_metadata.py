"""Tests for ExifTool-backed crop metadata and preview helpers."""

import json
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from photo_workflow.crop_tool import metadata as crop_metadata
from photo_workflow.crop_tool.model import CropRect, lightroom_crop_to_display


def test_read_photo_metadata_loads_crop_and_angle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read RAW dimensions and crop fields from their respective files."""
    outputs = [
        json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000}]),
        json.dumps(
            [
                {
                    "CropLeft": 0.233333,
                    "CropBottom": 1,
                    "CropRight": 0.766667,
                    "CropAngle": 2.25,
                }
            ]
        ),
    ]
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, outputs.pop(0)),
    )
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    metadata = crop_metadata.read_photo_metadata(tmp_path / "photo.cr3", xmp_path)

    assert metadata.image_width == 6000
    assert metadata.image_height == 4000
    expected_crop = lightroom_crop_to_display(
        CropRect(0.233333, 0, 0.766667, 1),
        6000,
        4000,
        2.25,
        1,
    )
    assert metadata.crop.left == pytest.approx(expected_crop.left)
    assert metadata.crop.top == pytest.approx(expected_crop.top)
    assert metadata.crop.right == pytest.approx(expected_crop.right)
    assert metadata.crop.bottom == pytest.approx(expected_crop.bottom)
    assert metadata.crop_angle == pytest.approx(2.25)


def test_read_photo_metadata_defaults_to_full_frame_when_crop_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An XMP sidecar without crop tags starts with the full image selected."""
    outputs = [json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000}]), "[{}]"]
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, outputs.pop(0)),
    )

    metadata = crop_metadata.read_photo_metadata(tmp_path / "photo.cr3", tmp_path / "photo.xmp")

    assert metadata.crop == CropRect(left=0, top=0, right=1, bottom=1)


def test_read_photo_ratings_batches_xmp_stars_and_color_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read ratings in one ExifTool call and default missing sidecars to unrated."""
    first_raw = tmp_path / "first.cr3"
    second_raw = tmp_path / "second.cr3"
    first_xmp = first_raw.with_suffix(".xmp")
    first_xmp.touch()
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                [
                    {
                        "SourceFile": str(first_xmp),
                        "XMP-xmp:Rating": 4,
                        "XMP-xmp:Label": "Red",
                    }
                ]
            ),
        )

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)

    ratings = crop_metadata.read_photo_ratings([first_raw, second_raw])

    assert ratings == {first_raw: (4, "Red"), second_raw: (0, None)}
    assert len(calls) == 1
    assert str(first_xmp) in calls[0]
    assert str(second_raw.with_suffix(".xmp")) not in calls[0]


def test_read_photo_metadata_swaps_dimensions_for_rotated_orientation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use display-oriented dimensions for portrait EXIF orientations."""
    outputs = [
        json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000, "Orientation": 6}]),
        json.dumps([{"CropTop": 0, "CropLeft": 0, "CropBottom": 1, "CropRight": 1}]),
    ]
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, outputs.pop(0)),
    )
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    metadata = crop_metadata.read_photo_metadata(tmp_path / "photo.cr3", xmp_path)

    assert (metadata.image_width, metadata.image_height) == (4000, 6000)
    assert metadata.orientation == 6


def test_read_photo_metadata_transforms_portrait_crop_into_display_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Orientation 8 maps Lightroom sensor-space left/right crop to display top/bottom."""
    outputs = [
        json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000, "Orientation": 8}]),
        json.dumps(
            [
                {
                    "CropTop": 0,
                    "CropLeft": 0,
                    "CropBottom": 1,
                    "CropRight": 0.744022,
                }
            ]
        ),
    ]
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, outputs.pop(0)),
    )
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    metadata = crop_metadata.read_photo_metadata(tmp_path / "photo.cr3", xmp_path)

    assert metadata.crop.left == pytest.approx(0)
    assert metadata.crop.top == pytest.approx(0.255978)
    assert metadata.crop.right == pytest.approx(1)
    assert metadata.crop.bottom == pytest.approx(1)


def test_read_photo_metadata_applies_lightroom_crop_angle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read Lightroom's negative XMP angle into the positive UI crop plane."""
    sensor_crop = CropRect(0.012828, 0.233885, 0.987172, 0.766115)
    outputs = [
        json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000, "Orientation": 1}]),
        json.dumps(
            [
                {
                    "CropTop": sensor_crop.top,
                    "CropLeft": sensor_crop.left,
                    "CropBottom": sensor_crop.bottom,
                    "CropRight": sensor_crop.right,
                    "CropAngle": -10,
                }
            ]
        ),
    ]
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, outputs.pop(0)),
    )
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    metadata = crop_metadata.read_photo_metadata(tmp_path / "photo.cr3", xmp_path)
    expected = lightroom_crop_to_display(sensor_crop, 6000, 4000, -10, 1)

    assert metadata.crop == expected
    assert metadata.crop_angle == pytest.approx(-10)


def test_write_photo_crop_transforms_display_crop_back_to_sensor_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A portrait display crop is written back in Lightroom's sensor coordinate space."""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        is_orientation_query = "-XMP-crs:CropTop=0.0000000000" not in command
        output = (
            json.dumps(
                [{"ImageWidth": 6000, "ImageHeight": 4000, "Orientation": 8}]
            )
            if is_orientation_query
            else "updated"
        )
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)
    raw_path = tmp_path / "photo.cr3"
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()
    sensor_crop = CropRect(0, 0, 0.744022, 1)
    display_crop = lightroom_crop_to_display(sensor_crop, 6000, 4000, -10, 8)

    crop_metadata.write_photo_crop(
        raw_path,
        xmp_path,
        display_crop,
        crop_angle=-10,
    )

    write_args = commands[-1]
    assert "-XMP-crs:CropTop=0.0000000000" in write_args
    assert "-XMP-crs:CropLeft=0.0000000000" in write_args
    assert "-XMP-crs:CropBottom=1.0000000000" in write_args
    assert "-XMP-crs:CropRight=0.7440220000" in write_args
    assert "-XMP-crs:CropAngle=-10.0000000000" in write_args


@pytest.mark.parametrize("raw_suffix", ["cr2", "cr3"])
def test_write_photo_crop_creates_a_minimal_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_suffix: str,
) -> None:
    """A first crop edit creates a standalone XMP sidecar without editing RAW."""
    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(
        crop_metadata.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            json.dumps([{"ImageWidth": 6000, "ImageHeight": 4000}]),
        ),
    )
    raw_path = tmp_path / f"photo.{raw_suffix}"
    xmp_path = tmp_path / "photo.xmp"
    crop = CropRect(left=0.2, top=0, right=0.8, bottom=1)

    crop_metadata.write_photo_crop(raw_path, xmp_path, crop)

    root = ElementTree.parse(xmp_path).getroot()
    description = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert description is not None
    assert description.get(f"{{{crop_metadata.CRS_NS}}}CropLeft") == "0.2000000000"
    assert description.get(f"{{{crop_metadata.CRS_NS}}}HasCrop") == "True"
    assert description.get(f"{{{crop_metadata.CRS_NS}}}RawFileName") == raw_path.name
    assert not raw_path.exists()


def test_write_photo_crop_sets_lightroom_crop_flag_on_existing_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing sidecars receive Lightroom's HasCrop flag with the crop update."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = (
            '[{"ImageWidth":6000,"ImageHeight":4000,"Orientation":1}]'
            if "-j" in command
            else "1 image files updated"
        )
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)
    raw_path = tmp_path / "photo.cr3"
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()

    crop_metadata.write_photo_crop(
        raw_path,
        xmp_path,
        CropRect(0.2, 0.1, 0.8, 0.9),
        crop_angle=-10,
    )

    assert "-XMP-crs:HasCrop=True" in calls[-1]
    assert "-XMP-crs:CropAngle=-10.0000000000" in calls[-1]


def test_extract_preview_requests_selected_embedded_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Extract the requested preview bytes without decoding on the caller side."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, b"jpeg-data")

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)

    preview = crop_metadata.extract_preview(tmp_path / "photo.cr3", "JpgFromRaw")

    assert preview == b"jpeg-data"
    assert calls[0][1:3] == ["-b", "-JpgFromRaw"]


def test_extract_preview_rejects_unknown_tag(tmp_path: Path) -> None:
    """Only known ExifTool embedded preview tags can be requested."""
    with pytest.raises(ValueError, match="Unsupported embedded preview tag"):
        crop_metadata.extract_preview(tmp_path / "photo.cr3", "Preview")
