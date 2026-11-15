"""
Image loading and saving shared by the Preprocessing stages and subject detection.

Intermediates between stages are 16-bit linear-light sRGB TIFFs (white balance baked in,
no tone curve), so exposure maths such as the mask's -0.25 stop is physically correct and
no precision is lost between stages. Camera RAW files (CR2/CR3) are developed with
LibRaw through `rawpy`.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import rawpy
from numpy.typing import NDArray

from photo_workflow.live.layout import RAW_SUFFIXES

MAX_16 = 65535.0
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


def load_srgb8(path: Path) -> NDArray[np.uint8]:
    """
    Load any photo or stage output as an upright 8-bit display-referred RGB array.

    Returns:
        An `H x W x 3` RGB array: RAW and linear intermediates are developed and encoded,
        ordinary 8-bit images (such as the Lighting JPEG) are read as they are.

    Raises:
        ValueError: If the file cannot be read as an image.
    """
    if path.suffix.lower() in RAW_SUFFIXES or path.suffix.lower() in LINEAR_SUFFIXES:
        return linear_to_srgb8(load_linear_rgb(path))
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
    linear = rgb.astype(np.float32) / MAX_16
    encoded = np.where(
        linear <= SRGB_THRESHOLD,
        linear * 12.92,
        1.055 * np.power(np.maximum(linear, SRGB_THRESHOLD), 1 / 2.4) - 0.055,
    )
    return np.round(np.clip(encoded, 0, 1) * 255).astype(np.uint8)


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
