"""
Mask stage: darken everything except the main subject by -0.25 stop.

The subject is found with `photo_workflow.live.detection` (Apple Vision); the mask is
inverted, feathered, and used to scale the scene-linear intermediate, which makes
the background slightly darker so the subject stands out. The detected boxes are cached
for Automatic Cropping. When no subject is found the stage is skipped, the input continues
downstream unchanged, and the viewer badge marks the photo as "mask not applied".
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.detection import analyze_image, store_detection
from photo_workflow.live.imaging import (
    MAX_16,
    linear_to_srgb8,
    load_linear_rgb,
    save_linear_rgb,
)
from photo_workflow.live.stages import StageId

BACKGROUND_EXPOSURE_STOPS = -0.25
FEATHER_FRACTION = 0.004
NO_SUBJECT_MESSAGE = "Mask could not be applied: no subject detected"


def apply_inverted_mask(
    rgb: NDArray[np.uint16],
    subject_mask: NDArray[np.float32],
    stops: float = BACKGROUND_EXPOSURE_STOPS,
) -> NDArray[np.uint16]:
    """
    Change the exposure of everything except the subject.

    Args:
        rgb: Linear 16-bit RGB image.
        subject_mask: Soft subject mask (0..1) of any size with the image's aspect ratio.
        stops: Exposure change applied to the inverted mask.

    Returns:
        The adjusted linear image.
    """
    height, width = rgb.shape[:2]
    feather = max(1.0, FEATHER_FRACTION * max(subject_mask.shape))
    soft = cv2.GaussianBlur(subject_mask, (0, 0), feather)
    background = cv2.resize(
        1 - np.clip(soft, 0, 1), (width, height), interpolation=cv2.INTER_LINEAR
    )
    gain = 1 + background * (2**stops - 1)
    scaled = rgb.astype(np.float32) * gain[..., None]
    return np.round(np.clip(scaled, 0, MAX_16)).astype(np.uint16)


class Mask(StageImplementation):
    """Writes `mask/<name>.tif` and publishes the subject and ball boxes."""

    def process(self, job: StageJob) -> StageResult:
        """
        Detect the subject and darken the background.

        Returns:
            The masked intermediate, or a skipped result when no subject was found.
        """
        linear = load_linear_rgb(job.source)
        analysis = analyze_image(linear_to_srgb8(linear))
        store_detection(job.photo, analysis.detection)
        output = job.output_path(suffix=".tif")
        if analysis.mask is None:
            output.unlink(missing_ok=True)
            return StageResult.skipped(NO_SUBJECT_MESSAGE)
        save_linear_rgb(output, apply_inverted_mask(linear, analysis.mask))
        return StageResult.done(output)


def register(registry: StageRegistry) -> None:
    """Register the Mask stage."""
    registry.register(StageId.MASK, Mask())
