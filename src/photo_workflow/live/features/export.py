"""Live Export stage: render each photo with its XMP crop into a full-quality JPEG."""

from __future__ import annotations

import io
import logging
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from photo_workflow.editor.metadata import extract_preview, read_photo_metadata
from photo_workflow.editor.model import CropRect
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.layout import IMAGE_SUFFIXES
from photo_workflow.live.stages import StageId

LOGGER = logging.getLogger(__name__)
EXPORT_SUFFIX = ".jpg"
JPEG_QUALITY = 95
EXIF_ORIENTATION_TAG = 0x0112
PREVIEW_TAGS_BY_SUFFIX = {
    ".cr2": ("PreviewImage",),
    ".cr3": ("JpgFromRaw", "PreviewImage"),
}
DEFAULT_PREVIEW_TAGS = ("JpgFromRaw", "PreviewImage")
ORIENTATION_TRANSPOSITIONS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


class Export(StageImplementation):
    """Renders the best available source with the sidecar crop into the `export` folder."""

    def process(self, job: StageJob) -> StageResult:
        """
        Export one photo.

        The crop (including straightening angle) comes from the photo's XMP sidecar, so a
        manual crop and an automatic one are treated alike. The result is re-rendered
        whenever the stage is re-run, which is how crop or upstream changes propagate.

        Returns:
            A done result listing the exported JPEG.
        """
        metadata = read_photo_metadata(job.photo, job.photo.with_suffix(".xmp"))
        image, exif = load_source_image(job.source, metadata.orientation)
        rendered = apply_crop(image, metadata.crop, metadata.crop_angle)
        target = job.output_path(suffix=EXPORT_SUFFIX)
        save_jpeg(rendered, target, exif)
        LOGGER.info(
            "live export %s from %s -> %dx%d",
            job.photo.name,
            job.source.name,
            rendered.width,
            rendered.height,
        )
        return StageResult.done(target)


def load_source_image(source: Path, orientation: int) -> tuple[Image.Image, Image.Exif]:
    """
    Load the best available source as an upright image.

    Finished image outputs are opened directly. For RAW files (the original or a
    denoise intermediate) the largest embedded JPEG is used, since no RAW developer is
    bundled; `orientation` is the RAW's EXIF orientation, applied when the embedded
    JPEG carries none.

    Args:
        source: The image or RAW file to render.
        orientation: EXIF orientation of the RAW file.

    Returns:
        The upright RGB image and its EXIF data (orientation reset).

    Raises:
        RuntimeError: If no embedded preview can be read from a RAW source.
    """
    if source.suffix.lower() in IMAGE_SUFFIXES:
        with Image.open(source) as opened:
            return _upright(opened, 1)
    tags = PREVIEW_TAGS_BY_SUFFIX.get(source.suffix.lower(), DEFAULT_PREVIEW_TAGS)
    errors: list[str] = []
    for tag in tags:
        try:
            data = extract_preview(source, tag)
            with Image.open(io.BytesIO(data)) as opened:
                return _upright(opened, orientation)
        except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{tag}: {error}")
    raise RuntimeError(f"No usable embedded preview in {source.name} ({'; '.join(errors)})")


def _upright(opened: Image.Image, fallback_orientation: int) -> tuple[Image.Image, Image.Exif]:
    """
    Apply EXIF orientation, preferring the image's own tag over `fallback_orientation`.

    Returns:
        The loaded upright RGB image and its EXIF data with the orientation reset.
    """
    exif = opened.getexif()
    own_orientation = exif.get(EXIF_ORIENTATION_TAG, 1)
    image = opened.copy()
    image.load()
    if own_orientation in ORIENTATION_TRANSPOSITIONS:
        image = ImageOps.exif_transpose(image)
    elif fallback_orientation in ORIENTATION_TRANSPOSITIONS:
        image = image.transpose(ORIENTATION_TRANSPOSITIONS[fallback_orientation])
    exif[EXIF_ORIENTATION_TAG] = 1
    if "icc_profile" in opened.info:
        image.info["icc_profile"] = opened.info["icc_profile"]
    return image.convert("RGB") if image.mode != "RGB" else image, exif


def apply_crop(image: Image.Image, crop: CropRect, crop_angle: float) -> Image.Image:
    """
    Straighten an upright image and cut out the normalized display crop.

    The editor's crop is normalized to the bounding box of the image rotated by
    `-crop_angle` (clockwise positive), which is what `Image.rotate(crop_angle,
    expand=True)` produces.

    Returns:
        The cropped image (the input itself when nothing needs cutting).
    """
    if crop_angle:
        image = image.rotate(crop_angle, resample=Image.Resampling.BICUBIC, expand=True)
    width, height = image.size
    left = min(round(crop.left * width), width - 1)
    top = min(round(crop.top * height), height - 1)
    right = max(round(crop.right * width), left + 1)
    bottom = max(round(crop.bottom * height), top + 1)
    if (left, top, right, bottom) == (0, 0, width, height):
        return image
    return image.crop((left, top, right, bottom))


def save_jpeg(image: Image.Image, target: Path, exif: Image.Exif) -> None:
    """Write a full-quality JPEG atomically so readers never see a partial export."""
    temporary = target.with_name(f".{target.name}.part")
    try:
        image.save(
            temporary,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=0,
            optimize=True,
            exif=exif.tobytes(),
            icc_profile=image.info.get("icc_profile"),
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def register(registry: StageRegistry) -> None:
    """Register the Export stage."""
    registry.register(StageId.EXPORT, Export())
