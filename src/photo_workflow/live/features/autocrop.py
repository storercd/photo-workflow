"""
AutoCrop stage: choose a crop with no human review and write it to the XMP sidecar.

Detection (the subject and, optionally, the ball) comes from the Preprocessing session's
`photo_workflow.live.detection` when it exists; this module only depends on the small
`Detector` protocol below. Without a detector the stage skips each photo with a warning,
so it stays harmless. Scoring lives in `photo_workflow.live.autocrop_scoring`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from photo_workflow.editor.metadata import is_manual_crop, read_photo_metadata, write_photo_crop
from photo_workflow.editor.model import CropRect, parse_aspect_ratios
from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.autocrop_scoring import Box, choose_crop
from photo_workflow.live.stages import StageId

LOGGER = logging.getLogger(__name__)
DETECTION_MODULE = "photo_workflow.live.detection"
DETECTION_ENTRY_POINT = "detect_scene"


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
    """Finds the subject (and ball) of an image."""

    def detect(self, image: Path, *, want_ball: bool) -> Detection | None:
        """
        Analyse an image.

        Boxes are in the displayed orientation of the image. Skip ball detection when
        `want_ball` is false.

        Returns:
            The detection, or `None` when no subject was found.
        """


class _ModuleDetector:
    """Adapts `live.detection.detect_scene(image, want_ball=...)` to `Detector`."""

    def __init__(self, module: object) -> None:
        self._entry = getattr(module, DETECTION_ENTRY_POINT)

    def detect(self, image: Path, *, want_ball: bool) -> Detection | None:
        """
        Run the shared detector and normalise its result.

        Returns:
            The detection, or `None` when no subject was found.
        """
        scene = self._entry(image, want_ball=want_ball)
        if scene is None or getattr(scene, "subject", None) is None:
            return None
        ball = getattr(scene, "ball", None)
        return Detection(
            image_width=int(scene.image_width),
            image_height=int(scene.image_height),
            subject=_box(scene.subject),
            ball=_box(ball) if ball is not None else None,
            direction=int(getattr(scene, "direction", 0) or 0),
        )


class _MissingDetector:
    """Stand-in used when no detection module exists; finds nothing."""

    def detect(self, image: Path, *, want_ball: bool) -> Detection | None:  # noqa: ARG002
        """
        Report that nothing was detected.

        Returns:
            Always `None`.
        """
        return None


def _box(value: object) -> Box:
    x0, y0, x1, y1 = value  # type: ignore[misc]
    return (float(x0), float(y0), float(x1), float(y1))


def default_detector() -> Detector:
    """
    Return the Preprocessing detector, or one that finds nothing (logged once per call).

    Returns:
        A detector for the AutoCrop stage.
    """
    try:
        module = importlib.import_module(DETECTION_MODULE)
    except ImportError:
        LOGGER.warning("%s not available; automatic crop has no detector", DETECTION_MODULE)
        return _MissingDetector()
    if not hasattr(module, DETECTION_ENTRY_POINT):
        LOGGER.warning(
            "%s has no %s(); automatic crop has no detector",
            DETECTION_MODULE,
            DETECTION_ENTRY_POINT,
        )
        return _MissingDetector()
    return _ModuleDetector(module)


class AutoCrop(StageImplementation):
    """Computes and writes an automatic crop for each photo."""

    def __init__(self, detector: Detector | None = None) -> None:
        """Create the stage; `detector` defaults to the shared Preprocessing detector."""
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
        detector = self._detector if self._detector is not None else default_detector()
        detection = detector.detect(job.source, want_ball=settings.autocrop_ball)
        if detection is None:
            job.warn("Automatic crop skipped: no subject detected")
            return StageResult.skipped("no subject")

        xmp_path = job.photo.with_suffix(".xmp")
        metadata = read_photo_metadata(job.photo, xmp_path)
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
