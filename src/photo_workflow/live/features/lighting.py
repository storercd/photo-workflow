"""
Lighting/Color stage: automatic lighting plus the optional color temperature override.

Both come from the editor's Review mode (`photo_workflow.editor.lighting`): the same
estimator picks exposure, white balance, levels, shadows/highlights, contrast and
vibrance, and an explicit Kelvin value replaces the estimated white balance exactly as the
editor's white-balance slider does. The result is the final high-quality JPEG in `output/`,
which becomes the preview image across modes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageCms

from photo_workflow.editor.lighting import (
    MAX_KELVIN,
    MIN_KELVIN,
    REFERENCE_KELVIN,
    LightingAdjustment,
    apply_adjustment,
    estimate_adjustment,
    gains_from_kelvin,
)
from photo_workflow.editor.metadata import read_as_shot_kelvin
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.imaging import load_srgb8
from photo_workflow.live.settings import LiveSettings
from photo_workflow.live.stages import OUTPUT_SUBFOLDER, StageId

LOGGER = logging.getLogger(__name__)
JPEG_QUALITY = 95
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def choose_adjustment(
    rgb: NDArray[np.uint8], settings: LiveSettings, reference: float
) -> LightingAdjustment:
    """
    Decide the lighting adjustment for an image from the Live settings.

    Automatic lighting estimates everything (a color temperature override replaces only its
    white balance); with it off, an override changes white balance alone, and with neither
    the image is left as it is.

    Returns:
        The adjustment to apply, neutral when nothing is enabled.
    """
    kelvin = settings.color_temperature
    if settings.auto_lighting:
        return estimate_adjustment(rgb, kelvin, reference)
    if kelvin is None:
        return LightingAdjustment(reference_kelvin=reference)
    return LightingAdjustment(
        gains=gains_from_kelvin(kelvin, reference), reference_kelvin=reference
    )


def as_shot_reference(photo: Path) -> float:
    """
    Return the camera's as-shot color temperature, which the developed image treats as neutral.

    Returns:
        The clamped as-shot Kelvin, or the editor's default when the file records none.
    """
    try:
        kelvin = read_as_shot_kelvin(photo)
    except Exception:
        LOGGER.warning("could not read the as-shot color temperature of %s", photo, exc_info=True)
        return REFERENCE_KELVIN
    return REFERENCE_KELVIN if kelvin is None else min(max(kelvin, MIN_KELVIN), MAX_KELVIN)


def write_jpeg(path: Path, rgb: NDArray[np.uint8]) -> None:
    """Write a high-quality sRGB JPEG atomically, replacing any previous file."""
    partial = path.with_name(path.stem + ".partial.jpg")
    Image.fromarray(rgb).save(
        partial,
        "JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
        icc_profile=SRGB_PROFILE,
    )
    partial.replace(path)


class Lighting(StageImplementation):
    """Writes the final preview JPEG to `output/<name>.jpg`."""

    def process(self, job: StageJob) -> StageResult:
        """
        Apply auto lighting and the color temperature override to the best source.

        Returns:
            The result listing the final JPEG.
        """
        rgb = load_srgb8(job.source, camera_reference=job.photo)
        reference = as_shot_reference(job.photo)
        adjustment = choose_adjustment(rgb, job.settings, reference)
        output = job.output_path(OUTPUT_SUBFOLDER, ".jpg")
        write_jpeg(output, apply_adjustment(rgb, adjustment))
        return StageResult.done(output, message=adjustment.describe())


def register(registry: StageRegistry) -> None:
    """Register the Lighting/Color stage."""
    registry.register(StageId.LIGHTING, Lighting())
