"""ExifTool integration for RAW previews and Lightroom crop metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

from photo_workflow.crop_tool.model import (
    CropRect,
    display_crop_to_lightroom,
    lightroom_crop_to_display,
)

XMP_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
XMP_TAGS = {
    "CropTop": "top",
    "CropLeft": "left",
    "CropBottom": "bottom",
    "CropRight": "right",
    "CropAngle": "crop_angle",
}
PREVIEW_TAGS = ("PreviewImage", "JpgFromRaw")
INVERSE_ORIENTATIONS = {2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


@dataclass(frozen=True)
class PhotoMetadata:
    """Source dimensions and crop information for one RAW photo."""

    image_width: int
    image_height: int
    crop: CropRect
    crop_angle: float = 0
    orientation: int = 1


def read_photo_metadata(raw_path: Path, xmp_path: Path) -> PhotoMetadata:
    """
    Read RAW dimensions and crop fields from a matching XMP sidecar.

    Returns:
        The image dimensions and current Lightroom crop.
    """
    exiftool = require_exiftool()
    raw_data = _read_json(
        exiftool,
        raw_path,
        "-n",
        "-ImageWidth",
        "-ImageHeight",
        "-Orientation",
    )
    image_width = int(raw_data["ImageWidth"])
    image_height = int(raw_data["ImageHeight"])
    sensor_width = image_width
    sensor_height = image_height
    orientation = int(raw_data.get("Orientation", 1))
    if orientation in {5, 6, 7, 8}:
        image_width, image_height = image_height, image_width
    crop_data = (
        _read_json(exiftool, xmp_path, *(f"-XMP-crs:{tag}" for tag in XMP_TAGS))
        if xmp_path.is_file()
        else {}
    )

    sensor_crop = CropRect(
        **{
            key: float(crop_data.get(tag, default))
            for tag, key, default in (
                ("CropTop", "top", 0),
                ("CropLeft", "left", 0),
                ("CropBottom", "bottom", 1),
                ("CropRight", "right", 1),
            )
        }
    )
    crop_angle = float(crop_data.get("CropAngle", 0))
    display_crop = lightroom_crop_to_display(
        sensor_crop,
        sensor_width,
        sensor_height,
        crop_angle,
        orientation,
    )
    return PhotoMetadata(
        image_width=image_width,
        image_height=image_height,
        crop=display_crop,
        crop_angle=crop_angle,
        orientation=orientation,
    )


def read_photo_ratings(raw_paths: list[Path]) -> dict[Path, tuple[int, str | None]]:
    """
    Read Lightroom star ratings and color labels for a batch of RAW photos.

    Returns:
        A mapping from each RAW path to its integer rating and optional color label.
    """
    ratings = {path: (0, None) for path in raw_paths}
    sidecars = [path.with_suffix(".xmp") for path in raw_paths]
    existing_sidecars = [path for path in sidecars if path.is_file()]
    if not existing_sidecars:
        return ratings
    result = subprocess.run(
        [
            require_exiftool(),
            "-j",
            "-G1",
            "-XMP-xmp:Rating",
            "-XMP-xmp:Label",
            *(str(path) for path in existing_sidecars),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    sidecar_to_raw = {
        str(sidecar.resolve()): raw_path
        for raw_path, sidecar in zip(raw_paths, sidecars, strict=True)
    }
    for record in json.loads(result.stdout):
        raw_path = sidecar_to_raw.get(str(Path(record["SourceFile"]).resolve()))
        if raw_path is None:
            continue
        rating_value = _grouped_tag(record, "Rating")
        label_value = _grouped_tag(record, "Label")
        rating = min(max(int(float(rating_value or 0)), 0), 5)
        label = str(label_value).strip() if label_value else None
        ratings[raw_path] = (rating, label or None)
    return ratings


def _grouped_tag(record: dict[str, object], tag_name: str) -> object | None:
    return next(
        (value for key, value in record.items() if key.endswith(f":{tag_name}")),
        None,
    )


def extract_preview(raw_path: Path, tag: str = "PreviewImage") -> bytes:
    """
    Extract one embedded JPEG preview from a RAW file.

    Returns:
        The embedded JPEG bytes.

    Raises:
        RuntimeError: If the selected preview is missing or ExifTool is unavailable.
        ValueError: If the requested preview tag is unsupported.
    """
    if tag not in PREVIEW_TAGS:
        raise ValueError(f"Unsupported embedded preview tag: {tag}")
    result = subprocess.run(
        [require_exiftool(), "-b", f"-{tag}", str(raw_path)],
        capture_output=True,
        check=True,
    )
    if not result.stdout:
        raise RuntimeError(f"RAW file contains no {tag} preview: {raw_path}")
    return result.stdout


def write_photo_crop(
    raw_path: Path,
    xmp_path: Path,
    crop: CropRect,
    *,
    crop_angle: float = 0,
) -> None:
    """Write crop metadata without modifying the RAW; preserve existing XMP data."""
    exiftool = require_exiftool()
    raw_metadata = _read_json(
        exiftool,
        raw_path,
        "-n",
        "-ImageWidth",
        "-ImageHeight",
        "-Orientation",
    )
    orientation = int(raw_metadata.get("Orientation", 1))
    image_width = int(raw_metadata["ImageWidth"])
    image_height = int(raw_metadata["ImageHeight"])
    sensor_crop = display_crop_to_lightroom(
        crop, image_width, image_height, crop_angle, orientation
    )
    if not xmp_path.exists():
        metadata = _read_json(exiftool, raw_path, "-ImageWidth", "-ImageHeight")
        _create_sidecar(
            xmp_path,
            raw_path,
            int(metadata["ImageWidth"]),
            int(metadata["ImageHeight"]),
            sensor_crop,
            crop_angle,
        )
        return

    values = {
        "CropTop": f"{sensor_crop.top:.10f}",
        "CropLeft": f"{sensor_crop.left:.10f}",
        "CropBottom": f"{sensor_crop.bottom:.10f}",
        "CropRight": f"{sensor_crop.right:.10f}",
        "CropAngle": f"{crop_angle:.10f}",
    }
    subprocess.run(
        [
            exiftool,
            *(f"-XMP-crs:{tag}={value}" for tag, value in values.items()),
            "-XMP-crs:HasCrop=True",
            str(xmp_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )


def transform_crop(crop: CropRect, orientation: int) -> CropRect:
    """
    Transform normalized crop bounds according to an EXIF orientation.

    Args:
        crop: Crop bounds in the source coordinate space.
        orientation: EXIF orientation from 1 through 8.

    Returns:
        The same crop transformed into the oriented coordinate space.
    """
    if orientation == 1 or orientation not in range(2, 9):
        return crop

    corners = (
        (crop.left, crop.top),
        (crop.right, crop.top),
        (crop.left, crop.bottom),
        (crop.right, crop.bottom),
    )
    transformed = [_transform_point(x, y, orientation) for x, y in corners]
    x_values, y_values = zip(*transformed, strict=True)
    return CropRect(min(x_values), min(y_values), max(x_values), max(y_values))


def _transform_point(x: float, y: float, orientation: int) -> tuple[float, float]:
    """
    Map one normalized source point into EXIF-oriented coordinates.

    Returns:
        The transformed normalized x and y coordinates.
    """
    transforms = {
        2: (1 - x, y),
        3: (1 - x, 1 - y),
        4: (x, 1 - y),
        5: (y, x),
        6: (1 - y, x),
        7: (1 - y, 1 - x),
        8: (y, 1 - x),
    }
    return transforms[orientation]


def require_exiftool() -> str:
    """
    Return the ExifTool executable path or raise an actionable error.

    Returns:
        The resolved executable path.

    Raises:
        RuntimeError: If ExifTool is unavailable.
    """
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        raise RuntimeError("ExifTool is required and must be available on PATH.")
    return exiftool


def _read_json(exiftool: str, path: Path, *tags: str) -> dict[str, object]:
    """
    Read selected ExifTool tags as a JSON object.

    Returns:
        The first metadata record returned by ExifTool.

    Raises:
        ValueError: If ExifTool returns no metadata records.
    """
    result = subprocess.run(
        [exiftool, "-j", *tags, str(path)],
        capture_output=True,
        check=True,
        text=True,
    )
    records = json.loads(result.stdout)
    if not records:
        raise ValueError(f"ExifTool returned no metadata for {path}")
    return records[0]


def _create_sidecar(
    xmp_path: Path,
    raw_path: Path,
    image_width: int,
    image_height: int,
    crop: CropRect,
    crop_angle: float,
) -> None:
    """Create a minimal standalone XMP sidecar atomically for the first edit."""
    ElementTree.register_namespace("x", XMP_NS)
    ElementTree.register_namespace("rdf", RDF_NS)
    ElementTree.register_namespace("crs", CRS_NS)
    root = ElementTree.Element(f"{{{XMP_NS}}}xmpmeta")
    rdf = ElementTree.SubElement(root, f"{{{RDF_NS}}}RDF")
    description = ElementTree.SubElement(rdf, f"{{{RDF_NS}}}Description")
    description.set(f"{{{RDF_NS}}}about", "")
    for tag, value in (
        ("CropTop", crop.top),
        ("CropLeft", crop.left),
        ("CropBottom", crop.bottom),
        ("CropRight", crop.right),
        ("CropAngle", crop_angle),
        ("HasCrop", "True"),
        ("CropConstrainToUnitSquare", 1),
    ):
        formatted_value = value if isinstance(value, str) else f"{value:.10f}"
        description.set(f"{{{CRS_NS}}}{tag}", formatted_value)
    description.set(f"{{{CRS_NS}}}RawFileName", raw_path.name)
    description.set(f"{{{CRS_NS}}}ImageWidth", str(image_width))
    description.set(f"{{{CRS_NS}}}ImageHeight", str(image_height))

    xmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=xmp_path.parent, suffix=".xmp", delete=False) as temp_file:
        temporary_path = Path(temp_file.name)
    try:
        ElementTree.ElementTree(root).write(
            temporary_path,
            encoding="utf-8",
            xml_declaration=True,
        )
        temporary_path.replace(xmp_path)
    finally:
        temporary_path.unlink(missing_ok=True)
