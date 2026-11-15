"""
AutoCrop stage: choose a crop with no human review and write it to the XMP sidecar.

Detection (the subject and, optionally, the ball) comes from `photo_workflow.live.detection`,
shared with the Mask stage; this module only depends on the small `Detector` protocol below.
Without a subject the stage skips the photo with a warning. Scoring lives in
`photo_workflow.live.autocrop_scoring`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from photo_workflow.editor.metadata import is_manual_crop, read_photo_metadata, write_photo_crop
from photo_workflow.editor.model import CropRect, parse_aspect_ratios
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.autocrop_scoring import Box, choose_crop
from photo_workflow.live.detection import (
    BoundingBox,
    DetectionUnavailableError,
    detect_photo,
    get_detection,
)
from photo_workflow.live.stages import StageId

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """
    What the detector found in one image.

    Attributes:
        image_width: Width in pixels of the image that was analysed.
        image_height: Height in pixels of that image, in display orientation.
        subject: The whole athlete (including limbs) as `(x0, y0, x1, y1)` pixels.
        ball: The ball box in the same coordinates, when one was found.
        direction: +1 moving right, -1 moving left, 0 unknown.
    """

    image_width: int
    image_height: int
    subject: Box
    ball: Box | None = None
    direction: int = 0


class Detector(Protocol):
    """Finds the subject (and ball) of a photo."""

    def detect(
        self, photo: Path, source: Path, size: tuple[int, int], *, want_ball: bool
    ) -> Detection | None:
        """
        Analyse a photo.

        Boxes are in pixels of a `size` (width, height) image in the displayed orientation.
        Skip the ball when `want_ball` is false.

        Returns:
            The detection, or `None` when no subject was found.
        """


class SharedDetector:
    """Uses the cached `live.detection` result, running the detector when it is missing."""

    def detect(
        self, photo: Path, source: Path, size: tuple[int, int], *, want_ball: bool
    ) -> Detection | None:
        """
        Fetch the photo's normalized boxes and convert them to pixels.

        Returns:
            The detection, or `None` when no subject was found or detection is unavailable.
        """
        try:
            found = get_detection(photo) or detect_photo(photo, source)
        except DetectionUnavailableError:
            LOGGER.warning("no subject detector available; automatic crop skips %s", photo.name)
            return None
        if found.subject is None:
            return None
        width, height = size
        ball = found.ball if want_ball else None
        return Detection(
            image_width=width,
            image_height=height,
            subject=_pixels(found.subject, width, height),
            ball=_pixels(ball, width, height) if ball is not None else None,
        )


def _pixels(box: BoundingBox, width: int, height: int) -> Box:
    left, top, right, bottom = box.to_pixels(width, height)
    return (float(left), float(top), float(right), float(bottom))


class AutoCrop(StageImplementation):
    """Computes and writes an automatic crop for each photo."""

    def __init__(self, detector: Detector | None = None) -> None:
        """Create the stage; `detector` defaults to the shared `live.detection` module."""
        self._detector = detector

    def process(self, job: StageJob) -> StageResult:
        """
        Compute the crop; write it unless a manual crop is protected.

        The crop is always computed, even when a manual crop exists, so forcing a
        replacement later needs no extra step.

        Returns:
            Done when a crop was computed, skipped when there was no subject.
        """
        settings = job.settings
        xmp_path = job.photo.with_suffix(".xmp")
        metadata = read_photo_metadata(job.photo, xmp_path)
        detector = self._detector if self._detector is not None else SharedDetector()
        detection = detector.detect(
            job.photo,
            job.source,
            (metadata.image_width, metadata.image_height),
            want_ball=settings.autocrop_ball,
        )
        if detection is None:
            job.warn("Automatic crop skipped: no subject detected")
            return StageResult.skipped("no subject")

        crop, summary = compute_crop(
            detection,
            metadata.image_width,
            metadata.image_height,
            [ratio.value for ratio in parse_aspect_ratios(list(settings.autocrop_ratios))],
            use_ball=settings.autocrop_ball,
        )
        if is_manual_crop(xmp_path) and not settings.autocrop_force:
            return StageResult.done(message=f"manual crop kept; automatic would be {summary}")
        write_photo_crop(job.photo, xmp_path, crop, source="auto")
        LOGGER.info("autocrop %s: %s", job.photo.name, summary)
        return StageResult.done(message=summary)


def compute_crop(
    detection: Detection,
    display_width: int,
    display_height: int,
    ratios: list[float],
    *,
    use_ball: bool,
) -> tuple[CropRect, str]:
    """
    Choose a crop for a detection and express it in normalized display coordinates.

    The detection may come from an image of another resolution (a preview), so its boxes
    are scaled to the photo's displayed size first.

    Returns:
        The normalized crop and a one-line description for logs.
    """
    scale_x = display_width / detection.image_width
    scale_y = display_height / detection.image_height
    subject = _scaled(detection.subject, scale_x, scale_y)
    ball = _scaled(detection.ball, scale_x, scale_y) if use_ball and detection.ball else None
    choice = choose_crop(display_width, display_height, subject, ball, detection.direction, ratios)
    x0, y0, x1, y1 = choice.box
    crop = CropRect(
        left=max(0.0, x0 / display_width),
        top=max(0.0, y0 / display_height),
        right=min(1.0, x1 / display_width),
        bottom=min(1.0, y1 / display_height),
    )
    summary = (
        f"ratio {choice.ratio:.3f} scale {choice.scale} score {choice.score:.3f} "
        f"rules {choice.rules}" + (" (fallback window)" if choice.fallback else "")
    )
    return crop, summary


def _scaled(box: Box, scale_x: float, scale_y: float) -> Box:
    return (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)


def register(registry: StageRegistry) -> None:
    """Register the AutoCrop stage."""
    registry.register(StageId.AUTOCROP, AutoCrop())
