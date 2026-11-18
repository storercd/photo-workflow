"""
Image loading and saving shared by the Preprocessing stages and subject detection.

Intermediates between stages are 16-bit linear-light sRGB TIFFs (white balance baked in,
no tone curve), so exposure maths such as the mask's -0.25 stop is physically correct and
no precision is lost between stages. Camera RAW files (CR2/CR3) are developed with
LibRaw through `rawpy`.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import rawpy
from numpy.typing import NDArray
from PIL import Image, ImageOps

from photo_workflow.live.layout import RAW_SUFFIXES

LOGGER = logging.getLogger(__name__)
MAX_16 = 65535.0
LOOK_FIT_WIDTH = 480
LOOK_BINS = 64
LOOK_MIN_SAMPLES = 40
LOOK_ASPECT_TOLERANCE = 0.02
SRGB_THRESHOLD = 0.0031308
# LibRaw `flip` codes -> number of counter-clockwise quarter turns (3 is a half turn).
FLIP_QUARTER_TURNS = {3: 2, 5: 1, 6: 3}
TEMP_SUFFIX = ".partial.tif"
LINEAR_SUFFIXES = frozenset({".tif", ".tiff"})


def load_linear_rgb(path: Path) -> NDArray[np.uint16]:
    """
    Load a photo as an upright, 16-bit, linear-light sRGB array.

    RAW files are developed with the camera white balance, no automatic brightness, and
    cropped to the sensor's active area; intermediates (TIFF) are read as stored.

    Returns:
        An `H x W x 3` RGB array.

    Raises:
        ValueError: If the file cannot be read as an image.
    """
    if path.suffix.lower() in RAW_SUFFIXES:
        return _develop_raw(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint16:
        raise ValueError(f"Not a 16-bit linear intermediate: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_linear_rgb(path: Path, rgb: NDArray[np.uint16]) -> None:
    """
    Write a 16-bit linear intermediate atomically, replacing any previous file.

    Raises:
        OSError: If the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + TEMP_SUFFIX)
    if not cv2.imwrite(str(partial), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {path}")
    os.replace(partial, path)


def load_srgb8(path: Path, camera_reference: Path | None = None) -> NDArray[np.uint8]:
    """
    Load any photo or stage output as an upright 8-bit display-referred RGB array.

    Args:
        path: The RAW, linear intermediate or ordinary image to load.
        camera_reference: The RAW whose embedded camera JPEG sets the look (tone curve and
            color rendering) of RAW or linear sources. Without it, or when the embedded
            preview does not match, the plain sRGB encoding is used.

    Returns:
        An `H x W x 3` RGB array: RAW and linear intermediates are developed and encoded,
        ordinary 8-bit images (such as the Lighting JPEG) are read as they are.

    Raises:
        ValueError: If the file cannot be read as an image.
    """
    if path.suffix.lower() in RAW_SUFFIXES or path.suffix.lower() in LINEAR_SUFFIXES:
        linear = load_linear_rgb(path)
        if camera_reference is not None:
            matched = match_camera_look(linear, camera_reference)
            if matched is not None:
                return matched
        return linear_to_srgb8(linear)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Not a readable image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def linear_to_srgb8(rgb: NDArray[np.uint16]) -> NDArray[np.uint8]:
    """
    Encode linear-light values with the sRGB transfer curve.

    Returns:
        An 8-bit display-referred array of the same shape.
    """
    encoded = _encode_srgb(rgb)
    return np.round(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


def match_camera_look(
    linear: NDArray[np.uint16], raw_path: Path
) -> NDArray[np.uint8] | None:
    """
    Render linear data with the tone and color curves of the camera's own JPEG.

    Linear development has no tone curve or picture style, so it looks dark, flat and
    desaturated next to the JPEG the camera embeds. Per channel, a monotonic curve from the
    linear image to the embedded JPEG is fitted on downsized copies and applied at full size.

    Returns:
        The matched 8-bit image, or `None` when the embedded preview is missing or does not
        line up with the linear image (so the caller falls back to plain sRGB).
    """
    reference = _embedded_preview(raw_path)
    if reference is None:
        return None
    height, width = linear.shape[:2]
    if abs(reference.shape[1] / reference.shape[0] - width / height) > LOOK_ASPECT_TOLERANCE:
        LOGGER.info("embedded preview of %s does not match the development", raw_path.name)
        return None
    fit_size = (LOOK_FIT_WIDTH, max(1, round(LOOK_FIT_WIDTH * height / width)))
    encoded = _encode_srgb(linear)
    source = cv2.resize(encoded, fit_size, interpolation=cv2.INTER_AREA)
    target = cv2.resize(reference, fit_size, interpolation=cv2.INTER_AREA)
    result = np.empty(linear.shape, dtype=np.uint8)
    for channel in range(3):
        grid, curve = _fit_curve(source[..., channel], target[..., channel])
        result[..., channel] = np.round(
            np.interp(encoded[..., channel], grid, curve) * 255
        ).astype(np.uint8)
    return result


def _encode_srgb(rgb: NDArray[np.uint16]) -> NDArray[np.float32]:
    """
    Encode linear 16-bit values with the sRGB curve.

    Returns:
        Float values in 0..1.
    """
    linear = rgb.astype(np.float32) / MAX_16
    return np.where(
        linear <= SRGB_THRESHOLD,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, SRGB_THRESHOLD), 1 / 2.4) - 0.055,
    ).astype(np.float32)


def _fit_curve(
    source: NDArray[np.float32], target: NDArray[np.uint8]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit a monotonic curve mapping `source` (0..1) to `target` (0..255 -> 0..1).

    Returns:
        The grid of source values and the matching target values.
    """
    x = source.ravel()
    y = target.ravel().astype(np.float32) / 255
    edges = np.linspace(0, 1, LOOK_BINS + 1)
    bin_index = np.clip(np.digitize(x, edges) - 1, 0, LOOK_BINS - 1)
    centers, medians = [], []
    for index in range(LOOK_BINS):
        selected = y[bin_index == index]
        if selected.size >= LOOK_MIN_SAMPLES:
            centers.append(float(np.median(x[bin_index == index])))
            medians.append(float(np.median(selected)))
    if not centers:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    return np.array(centers), np.maximum.accumulate(np.array(medians))


def _embedded_preview(raw_path: Path) -> NDArray[np.uint8] | None:
    """
    Read the camera JPEG embedded in a RAW file, upright.

    Returns:
        An `H x W x 3` RGB array, or `None` when no JPEG preview can be read.
    """
    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            return None
        with Image.open(io.BytesIO(thumb.data)) as opened:
            return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
    except (rawpy.LibRawError, OSError, ValueError):
        LOGGER.warning("could not read the embedded preview of %s", raw_path, exc_info=True)
        return None


def _develop_raw(path: Path) -> NDArray[np.uint16]:
    """
    Develop a RAW file to upright linear 16-bit sRGB, cropped to the active area.

    Returns:
        The developed `H x W x 3` array.
    """
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
            output_color=rawpy.ColorSpace.sRGB,
            user_flip=0,
        )
        # The crop margins are measured from the full sensor; the developed image starts
        # inside the masked border.
        top = max(0, sizes.crop_top_margin - sizes.top_margin)
        left = max(0, sizes.crop_left_margin - sizes.left_margin)
        height, width = sizes.crop_height, sizes.crop_width
        if height and width:
            rgb = rgb[top : top + height, left : left + width]
        turns = FLIP_QUARTER_TURNS.get(sizes.flip, 0)
    return np.ascontiguousarray(np.rot90(rgb, turns)) if turns else np.ascontiguousarray(rgb)
