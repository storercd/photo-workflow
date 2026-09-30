"""Tests for ExifTool-backed crop metadata and preview helpers."""

import json
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from photo_workflow.crop_tool import metadata as crop_metadata
from photo_workflow.crop_tool.model import CropRect


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
                    "CropTop": 0,
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
    assert metadata.crop.left == pytest.approx(0.233333)
    assert metadata.crop.right == pytest.approx(0.766667)
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


def test_write_photo_crop_transforms_display_crop_back_to_sensor_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A portrait display crop is written back in Lightroom's sensor coordinate space."""
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        is_orientation_query = "-XMP-crs:CropTop=0.0000000000" not in command
        output = json.dumps([{"Orientation": 8}]) if is_orientation_query else "updated"
        return subprocess.CompletedProcess(command, 0, output)

    monkeypatch.setattr(crop_metadata, "require_exiftool", lambda: "/usr/bin/exiftool")
    monkeypatch.setattr(crop_metadata.subprocess, "run", run)
    raw_path = tmp_path / "photo.cr3"
    xmp_path = tmp_path / "photo.xmp"
    xmp_path.touch()
    display_crop = CropRect(0, 0.255978, 1, 1)

    crop_metadata.write_photo_crop(raw_path, xmp_path, display_crop)

    write_args = commands[-1]
    assert "-XMP-crs:CropTop=0.0000000000" in write_args
    assert "-XMP-crs:CropLeft=0.0000000000" in write_args
    assert "-XMP-crs:CropBottom=1.0000000000" in write_args
    assert "-XMP-crs:CropRight=0.7440220000" in write_args


def test_write_photo_crop_creates_a_minimal_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    raw_path = tmp_path / "photo.cr3"
    xmp_path = tmp_path / "photo.xmp"
    crop = CropRect(left=0.2, top=0, right=0.8, bottom=1)

    crop_metadata.write_photo_crop(raw_path, xmp_path, crop)

    root = ElementTree.parse(xmp_path).getroot()
    description = root.find(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description")
    assert description is not None
    assert description.get(f"{{{crop_metadata.CRS_NS}}}CropLeft") == "0.2000000000"
    assert description.get(f"{{{crop_metadata.CRS_NS}}}HasCrop") == "True"
    assert description.get(f"{{{crop_metadata.CRS_NS}}}RawFileName") == "photo.cr3"
    assert not raw_path.exists()


def test_write_photo_crop_sets_lightroom_crop_flag_on_existing_sidecar(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing sidecars receive Lightroom's HasCrop flag with the crop update."""
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        output = '[{"Orientation":1}]' if "-j" in command else "1 image files updated"
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
    )

    assert "-XMP-crs:HasCrop=True" in calls[-1]


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
