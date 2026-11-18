"""ExifTool integration for RAW previews and Lightroom crop metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from photo_workflow.editor.model import (
    CropRect,
    display_crop_to_lightroom,
    lightroom_crop_to_display,
)

XMP_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
XAP_NS = "http://ns.adobe.com/xap/1.0/"
PWF_NS = "https://github.com/storercd/photo-workflow/ns/1.0/"
PWF_CONFIG = Path(__file__).with_name("resources") / "photo_workflow_xmp.config"
CropSource = Literal["manual", "auto"]
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
    auto_crop: bool = False


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
        _read_json(
            exiftool,
            xmp_path,
            "-XMP-pwf:CropSource",
            *(f"-XMP-crs:{tag}" for tag in XMP_TAGS),
        )
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
        auto_crop=str(crop_data.get("CropSource", "")).strip().lower() == "auto",
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
    source: CropSource = "manual",
) -> None:
    """
    Write crop metadata without modifying the RAW; preserve existing XMP data.

    `source` records who made the crop in `pwf:CropSource`: Crop mode writes "manual" (the
    default) and Live's AutoCrop stage writes "auto", so a manual crop can be told apart.
    """
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
            source,
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
            "-config",
            str(PWF_CONFIG),
            *(f"-XMP-crs:{tag}={value}" for tag, value in values.items()),
            "-XMP-crs:HasCrop=True",
            f"-XMP-pwf:CropSource={source}",
            str(xmp_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )


def is_manual_crop(xmp_path: Path) -> bool:
    """
    Return whether a sidecar holds a crop made by the user.

    A crop counts as manual when `pwf:CropSource` says so, and also when it has no marker
    (it predates automatic crops, so the user made it). Only a crop marked "auto", or no
    crop at all, is not manual.

    Returns:
        `True` when automatic crop must not overwrite the sidecar's crop.
    """
    if not xmp_path.is_file():
        return False
    record = _read_json(
        require_exiftool(),
        xmp_path,
        "-config",
        str(PWF_CONFIG),
        "-XMP-pwf:CropSource",
        "-XMP-crs:HasCrop",
        *(f"-XMP-crs:{tag}" for tag in ("CropTop", "CropLeft", "CropBottom", "CropRight")),
    )
    marker = str(record.get("CropSource", "")).strip().lower()
    if marker:
        return marker != "auto"
    has_crop = str(record.get("HasCrop", "")).strip().lower() == "true"
    return has_crop or any(
        tag in record for tag in ("CropTop", "CropLeft", "CropBottom", "CropRight")
    )


def write_photo_label(xmp_path: Path, label: str | None) -> None:
    """Write or clear the Lightroom color label without touching crop fields."""
    exiftool = require_exiftool()
    if not xmp_path.exists():
        if label is None:
            return
        _create_label_sidecar(xmp_path, label)
        return
    subprocess.run(
        [exiftool, f"-XMP-xmp:Label={label or ''}", str(xmp_path)],
        capture_output=True,
        check=True,
        text=True,
    )


NOTES_PREFIX = "Notes: "


def read_photo_notes(xmp_path: Path) -> str:
    """
    Read the free-form note stored as a "Notes: ..." keyword.

    Returns:
        The note text, or an empty string when none is assigned.
    """
    if not xmp_path.is_file():
        return ""
    for keyword in _read_keywords(xmp_path):
        if keyword.startswith(NOTES_PREFIX):
            return keyword[len(NOTES_PREFIX) :]
    return ""


def write_photo_notes(xmp_path: Path, notes: str) -> None:
    """Replace the "Notes: ..." keyword, or remove it when `notes` is blank."""
    notes = " ".join(notes.split())
    if not xmp_path.exists():
        if not notes:
            return
        _create_label_sidecar(xmp_path)
    keywords = [k for k in _read_keywords(xmp_path) if not k.startswith(NOTES_PREFIX)]
    if notes:
        keywords.append(f"{NOTES_PREFIX}{notes}")
    exiftool = require_exiftool()
    for arguments in (
        ["-XMP-dc:Subject="],
        [f"-XMP-dc:Subject+={keyword}" for keyword in keywords],
    ):
        if arguments:
            subprocess.run(
                [exiftool, "-overwrite_original", *arguments, str(xmp_path)],
                capture_output=True,
                check=True,
                text=True,
            )


def read_photos_with_notes(raw_paths: list[Path]) -> set[Path]:
    """
    Find which RAW photos have a "Notes: ..." keyword in their sidecar.

    Returns:
        The subset of `raw_paths` whose sidecar carries a note.
    """
    sidecars = {path.with_suffix(".xmp").resolve(): path for path in raw_paths}
    existing = [str(sidecar) for sidecar in sidecars if sidecar.is_file()]
    if not existing:
        return set()
    result = subprocess.run(
        [require_exiftool(), "-j", "-XMP-dc:Subject", *existing],
        capture_output=True,
        check=True,
        text=True,
    )
    found = set()
    for record in json.loads(result.stdout):
        subjects = record.get("Subject", [])
        subjects = subjects if isinstance(subjects, list) else [subjects]
        raw_path = sidecars.get(Path(record["SourceFile"]).resolve())
        if raw_path is not None and any(str(s).startswith(NOTES_PREFIX) for s in subjects):
            found.add(raw_path)
    return found


def _read_keywords(xmp_path: Path) -> list[str]:
    record = _read_json(require_exiftool(), xmp_path, "-XMP-dc:Subject")
    value = record.get("Subject", [])
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


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


def read_as_shot_kelvin(raw_path: Path) -> float | None:
    """
    Read the camera's as-shot white-balance color temperature from a RAW file.

    Returns:
        The temperature in Kelvin, or `None` when the file does not record one.
    """
    record = _read_json(require_exiftool(), raw_path, "-n", "-ColorTempAsShot", "-ColorTemperature")
    for tag in ("ColorTempAsShot", "ColorTemperature"):
        value = record.get(tag)
        if isinstance(value, int | float) and value > 0:
            return float(value)
    return None


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
    source: CropSource = "manual",
) -> None:
    """Create a minimal standalone XMP sidecar atomically for the first edit."""
    ElementTree.register_namespace("x", XMP_NS)
    ElementTree.register_namespace("rdf", RDF_NS)
    ElementTree.register_namespace("crs", CRS_NS)
    ElementTree.register_namespace("pwf", PWF_NS)
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
    description.set(f"{{{PWF_NS}}}CropSource", source)
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


def write_photo_rating(xmp_path: Path, rating: int) -> None:
    """Write the Lightroom star rating (0-5) without touching crop fields."""
    rating = min(max(rating, 0), 5)
    exiftool = require_exiftool()
    if not xmp_path.exists():
        if rating == 0:
            return
        _create_label_sidecar(xmp_path, rating=rating)
        return
    subprocess.run(
        [exiftool, f"-XMP-xmp:Rating={rating}", str(xmp_path)],
        capture_output=True,
        check=True,
        text=True,
    )


def _create_label_sidecar(
    xmp_path: Path, label: str | None = None, rating: int | None = None
) -> None:
    """Create a minimal standalone XMP sidecar atomically holding a color label or rating."""
    ElementTree.register_namespace("x", XMP_NS)
    ElementTree.register_namespace("rdf", RDF_NS)
    ElementTree.register_namespace("xmp", XAP_NS)
    root = ElementTree.Element(f"{{{XMP_NS}}}xmpmeta")
    rdf = ElementTree.SubElement(root, f"{{{RDF_NS}}}RDF")
    description = ElementTree.SubElement(rdf, f"{{{RDF_NS}}}Description")
    description.set(f"{{{RDF_NS}}}about", "")
    if label is not None:
        description.set(f"{{{XAP_NS}}}Label", label)
    if rating is not None:
        description.set(f"{{{XAP_NS}}}Rating", str(rating))

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
