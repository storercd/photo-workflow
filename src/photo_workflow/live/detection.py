"""
Subject and ball detection shared by Preprocessing (Mask) and Automatic Cropping.

The Mask stage finds the main subject once and publishes its bounding box (and the ball's,
when one is found) here; Automatic Cropping reads it back instead of detecting again.

Typical use::

    from photo_workflow.live.detection import detect_photo, get_detection

    detection = get_detection(job.photo)               # cached result or None
    detection = detect_photo(job.photo, job.source)    # cached, else detects and caches
    if detection.subject is not None:
        left, top, right, bottom = detection.subject.to_pixels(width, height)

All boxes are normalized (0 to 1) in the upright, display-oriented image, the same frame
the editor's Crop mode uses, so they are independent of the resolution of whichever
pipeline output (RAW, intermediate or JPEG) the caller loaded. The cache is keyed by the
photo's path (the renamed original in `input/`), is held in memory, and is thread safe.
Subject detection uses Apple's Vision framework (macOS only); the ball uses the editor's
existing Hough-circle detector.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_workflow.editor.ball import detect_ball_candidates
from photo_workflow.live.imaging import load_srgb8

try:
    import Foundation
    import Vision
    from Quartz import CoreVideo
except ImportError:  # not macOS, or the Vision bindings are not installed
    Foundation = Vision = CoreVideo = None

WORKING_LONG_EDGE = 1600
MIN_SUBJECT_AREA_FRACTION = 0.005
MIN_RELATIVE_INSTANCE_AREA = 0.25
MASK_THRESHOLD = 0.5
CACHE_SIZE = 512
FLOAT32_PIXEL_FORMAT_BYTES = 4

# Maps an 8-bit RGB image to a float32 subject mask in 0..1 of the same size, or `None`.
SubjectDetector = Callable[[NDArray[np.uint8]], NDArray[np.float32] | None]


class DetectionUnavailableError(RuntimeError):
    """Raised when no subject-detection backend exists on this machine."""


@dataclass(frozen=True)
class BoundingBox:
    """A box as fractions of the image width and height, with `left < right`, `top < bottom`."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        """Box width as a fraction of the image width."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """Box height as a fraction of the image height."""
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        """Box center as fractions of the image width and height."""
        return (self.left + self.right) / 2, (self.top + self.bottom) / 2

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """
        Convert to pixel coordinates in an image of the given size.

        Returns:
            `(left, top, right, bottom)` in pixels.
        """
        return (
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )


@dataclass(frozen=True)
class Detection:
    """
    What was found in one photo.

    Attributes:
        subject: Box around the main subject, or `None` when none was found (the Mask
            stage could not be applied).
        ball: Box around the ball, or `None` when no ball was found.
    """

    subject: BoundingBox | None = None
    ball: BoundingBox | None = None

    @property
    def subject_found(self) -> bool:
        """Whether a main subject was detected."""
        return self.subject is not None


@dataclass(frozen=True)
class SubjectAnalysis:
    """
    Result of analyzing one image.

    Attributes:
        detection: The boxes found.
        mask: Soft subject mask (float32, 0..1) at the analysis resolution (long edge up
            to `WORKING_LONG_EDGE` pixels, same aspect as the image), `None` without a subject.
    """

    detection: Detection
    mask: NDArray[np.float32] | None = None


_CACHE: OrderedDict[str, Detection] = OrderedDict()
_LOCK = threading.Lock()


def get_detection(photo: Path) -> Detection | None:
    """
    Return the cached detection for a photo.

    Returns:
        The detection stored by the Mask stage (or `detect_photo`), else `None`.
    """
    with _LOCK:
        detection = _CACHE.get(str(photo))
        if detection is not None:
            _CACHE.move_to_end(str(photo))
        return detection


def store_detection(photo: Path, detection: Detection) -> None:
    """Cache a photo's detection, replacing any earlier one."""
    with _LOCK:
        _CACHE[str(photo)] = detection
        _CACHE.move_to_end(str(photo))
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)


def clear_detections(photo: Path | None = None) -> None:
    """Forget one photo's cached detection, or every detection when no photo is given."""
    with _LOCK:
        if photo is None:
            _CACHE.clear()
        else:
            _CACHE.pop(str(photo), None)


def detect_photo(
    photo: Path,
    source: Path | None = None,
    *,
    detector: SubjectDetector | None = None,
) -> Detection:
    """
    Return the photo's detection, running (and caching) it when it is not cached yet.

    Args:
        photo: The renamed original, used as the cache key.
        source: The image to analyze (any pipeline output); defaults to `photo`.
        detector: Overrides the subject-detection backend.

    Returns:
        The detection; `subject` is `None` when nothing was found.
    """
    cached = get_detection(photo)
    if cached is not None:
        return cached
    detection = analyze_image(load_srgb8(source or photo), detector=detector).detection
    store_detection(photo, detection)
    return detection


def analyze_image(
    rgb: NDArray[np.uint8], *, detector: SubjectDetector | None = None
) -> SubjectAnalysis:
    """
    Find the main subject and the ball in an upright 8-bit RGB image.

    Without a custom `detector`, the macOS Vision backend is used and
    `DetectionUnavailableError` propagates when it is missing.

    Returns:
        The boxes and the soft subject mask (see `SubjectAnalysis`).
    """
    small = _downscale(rgb)
    mask = (detector or detect_subject_mask)(small)
    subject = None if mask is None else _mask_box(mask)
    if subject is None:
        mask = None
    return SubjectAnalysis(Detection(subject, _find_ball(small)), mask)


def detect_subject_mask(rgb: NDArray[np.uint8]) -> NDArray[np.float32] | None:
    """
    Segment the main subject with the macOS Vision foreground-instance model.

    All instances at least a quarter the size of the largest are combined, so a pair of
    players counts as one subject.

    Returns:
        A float32 mask in 0..1 of the same size as `rgb`, or `None` when nothing
        substantial was found.

    Raises:
        DetectionUnavailableError: If the Vision framework bindings are missing.
        RuntimeError: If Vision fails to analyze the image.
    """
    if Vision is None:
        raise DetectionUnavailableError(
            "Subject detection needs macOS with pyobjc-framework-Vision installed"
        )
    encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1].tobytes()
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        Foundation.NSData.dataWithBytes_length_(encoded, len(encoded)), None
    )
    request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
    succeeded, error = handler.performRequests_error_([request], None)
    if not succeeded:
        raise RuntimeError(f"Vision subject detection failed: {error}")
    observations = request.results()
    if not observations:
        return None
    observation = observations[0]
    masks = [
        _instance_mask(observation, handler, index)
        for index in observation.allInstances()
    ]
    return _combine_instances(masks)


def _combine_instances(masks: list[NDArray[np.float32]]) -> NDArray[np.float32] | None:
    """
    Union the instance masks that are big enough to be part of the main subject.

    Returns:
        The combined mask, or `None` when there is no substantial instance.
    """
    if not masks:
        return None
    areas = [float((mask > MASK_THRESHOLD).mean()) for mask in masks]
    largest = max(areas)
    if largest < MIN_SUBJECT_AREA_FRACTION:
        return None
    kept = [
        mask
        for mask, area in zip(masks, areas, strict=True)
        if area >= largest * MIN_RELATIVE_INSTANCE_AREA
    ]
    return np.maximum.reduce(kept)


def _instance_mask(observation: object, handler: object, index: int) -> NDArray[np.float32]:
    """
    Return the full-resolution float mask of one Vision instance.

    Returns:
        A float32 mask the size of the analyzed image.

    Raises:
        RuntimeError: If Vision cannot build the mask.
    """
    indexes = Foundation.NSIndexSet.indexSetWithIndex_(index)
    buffer, error = observation.generateScaledMaskForImageForInstances_fromRequestHandler_error_(
        indexes, handler, None
    )
    if buffer is None:
        raise RuntimeError(f"Vision could not build the subject mask: {error}")
    CoreVideo.CVPixelBufferLockBaseAddress(buffer, CoreVideo.kCVPixelBufferLock_ReadOnly)
    try:
        width = CoreVideo.CVPixelBufferGetWidth(buffer)
        height = CoreVideo.CVPixelBufferGetHeight(buffer)
        stride = CoreVideo.CVPixelBufferGetBytesPerRow(buffer) // FLOAT32_PIXEL_FORMAT_BYTES
        base = CoreVideo.CVPixelBufferGetBaseAddress(buffer)
        values = np.frombuffer(
            base.as_buffer(stride * height * FLOAT32_PIXEL_FORMAT_BYTES), dtype=np.float32
        )
        return values.reshape(height, stride)[:, :width].copy()
    finally:
        CoreVideo.CVPixelBufferUnlockBaseAddress(buffer, CoreVideo.kCVPixelBufferLock_ReadOnly)


def _downscale(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Shrink an image so detection stays fast.

    Returns:
        The reduced image, or `rgb` itself when it is already small.
    """
    scale = WORKING_LONG_EDGE / max(rgb.shape[:2])
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _mask_box(mask: NDArray[np.float32]) -> BoundingBox | None:
    """Return the normalized box around the mask's foreground, or `None` when it is empty."""
    foreground = mask > MASK_THRESHOLD
    rows, columns = np.flatnonzero(foreground.any(axis=1)), np.flatnonzero(foreground.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    height, width = mask.shape
    return BoundingBox(
        float(columns[0]) / width,
        float(rows[0]) / height,
        float(columns[-1] + 1) / width,
        float(rows[-1] + 1) / height,
    )


def _find_ball(rgb: NDArray[np.uint8]) -> BoundingBox | None:
    """Return the box of the most convincing ball candidate, if any."""
    candidates = detect_ball_candidates(rgb, max_candidates=1)
    if not candidates:
        return None
    height, width = rgb.shape[:2]
    ball = candidates[0]
    radius_y = ball.radius * width / height
    return BoundingBox(
        max(0.0, ball.center_x - ball.radius),
        max(0.0, ball.center_y - radius_y),
        min(1.0, ball.center_x + ball.radius),
        min(1.0, ball.center_y + radius_y),
    )
