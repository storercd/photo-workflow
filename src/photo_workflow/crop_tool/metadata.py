"""ExifTool integration for RAW previews and Lightroom crop metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path

from photo_workflow.crop_tool.model import CropRect

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


@dataclass(frozen=True)
class PhotoMetadata:
    """Source dimensions and crop information for one RAW photo."""

    image_width: int
    image_height: int
    crop: CropRect
    crop_angle: float = 0


def read_photo_metadata(raw_path: Path, xmp_path: Path) -> PhotoMetadata:
    """
    Read RAW dimensions and crop fields from a matching XMP sidecar.

    Returns:
        The image dimensions and current Lightroom crop.
    """
    exiftool = require_exiftool()
    raw_data = _read_json(exiftool, raw_path, "-n", "-ImageWidth", "-ImageHeight", "-Orientation")
    image_width = int(raw_data["ImageWidth"])
    image_height = int(raw_data["ImageHeight"])
    orientation = int(raw_data.get("Orientation", 1))
    if orientation in {5, 6, 7, 8}:
        image_width, image_height = image_height, image_width
    crop_data = (
        _read_json(exiftool, xmp_path, *(f"-XMP-crs:{tag}" for tag in XMP_TAGS))
        if xmp_path.is_file()
        else {}
    )

    coordinates = {
        key: float(crop_data.get(tag, default))
        for tag, key, default in (
            ("CropTop", "top", 0),
            ("CropLeft", "left", 0),
            ("CropBottom", "bottom", 1),
            ("CropRight", "right", 1),
        )
    }
    return PhotoMetadata(
        image_width=image_width,
        image_height=image_height,
        crop=CropRect(**coordinates),
        crop_angle=float(crop_data.get("CropAngle", 0)),
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
    if not xmp_path.exists():
        metadata = _read_json(exiftool, raw_path, "-ImageWidth", "-ImageHeight")
        _create_sidecar(
            xmp_path,
            raw_path,
            int(metadata["ImageWidth"]),
            int(metadata["ImageHeight"]),
            crop,
            crop_angle,
        )
        return

    values = {
        "CropTop": f"{crop.top:.10f}",
        "CropLeft": f"{crop.left:.10f}",
        "CropBottom": f"{crop.bottom:.10f}",
        "CropRight": f"{crop.right:.10f}",
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
