"""
Denoise stage: produce a denoised, full-resolution intermediate from the camera RAW.

Tool choice: LibRaw (through `rawpy`) is the practical, pip-installable decoder for Canon
CR2 and CR3 on macOS, and OpenCV's non-local-means is already a dependency, so no
paid or GUI-only tool (DxO PureRAW, Adobe DNG Converter) is needed. No open library can
write a re-encoded *mosaiced* RAW, so the "raw-format" intermediate is the scene-linear
16-bit TIFF that follows RAW development (camera white balance, no tone curve): it keeps
the RAW's linear light and full precision for the later stages.

Noise grows with brightness in linear data, so denoising runs in a square-root domain
(roughly constant noise) on a luma/chroma split with chroma smoothed more strongly. The
filter strength follows the noise measured in each image, so a clean low-ISO frame is left
almost untouched while a high-ISO frame is smoothed enough to matter.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_workflow.live.api import StageImplementation, StageJob, StageRegistry, StageResult
from photo_workflow.live.imaging import MAX_16, load_linear_rgb, save_linear_rgb
from photo_workflow.live.stages import StageId

LUMA_STRENGTH_PER_SIGMA = 1.5
CHROMA_STRENGTH_PER_SIGMA = 4.0
# Colour noise is blotchy rather than grainy, so chroma is smoothed at reduced resolution.
CHROMA_DOWNSCALE = 4
MIN_STRENGTH = 150.0
MAX_STRENGTH = 4000.0
# Immerkaer's noise kernel: a Laplacian-style difference that cancels smooth detail.
NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
TEMPLATE_WINDOW = 5
SEARCH_WINDOW = 15
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def denoise_linear(rgb: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """
    Denoise a linear 16-bit RGB image.

    Returns:
        A denoised array of the same shape and type.
    """
    encoded = np.sqrt(rgb.astype(np.float32) / MAX_16)
    luma = encoded @ LUMA_WEIGHTS
    blue_diff, red_diff = encoded[..., 2] - luma, encoded[..., 0] - luma
    luma_strength = _strength(luma, LUMA_STRENGTH_PER_SIGMA)
    luma = _smooth(luma, luma_strength)
    blue_plane, red_plane = (blue_diff + 1) / 2, (red_diff + 1) / 2
    blue_diff = _smooth_chroma(blue_plane) * 2 - 1
    red_diff = _smooth_chroma(red_plane) * 2 - 1
    green = (luma - LUMA_WEIGHTS[0] * (luma + red_diff) - LUMA_WEIGHTS[2] * (luma + blue_diff)) / (
        LUMA_WEIGHTS[1]
    )
    clean = np.stack((luma + red_diff, green, luma + blue_diff), axis=-1)
    return np.round(np.clip(clean, 0, 1) ** 2 * MAX_16).astype(np.uint16)


def _strength(plane: NDArray[np.float32], per_sigma: float) -> float:
    """
    Pick a filter strength from the noise level of a 0..1 plane.

    The noise is the robust (median-based) spread of the plane's second differences, which
    ignores smooth detail and so is only mildly inflated by texture.

    Returns:
        The non-local-means strength in 16-bit units, kept within sane limits.
    """
    differences = cv2.filter2D(plane, cv2.CV_32F, NOISE_KERNEL)
    sigma = float(np.median(np.abs(differences))) * 1.4826 / 6
    return min(max(per_sigma * sigma * MAX_16, MIN_STRENGTH), MAX_STRENGTH)


def _smooth_chroma(plane: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Smooth a 0..1 chroma plane at reduced resolution, where blotchy colour noise is grainy.

    Returns:
        The filtered plane at the original size.
    """
    height, width = plane.shape
    size = (max(width // CHROMA_DOWNSCALE, 1), max(height // CHROMA_DOWNSCALE, 1))
    small = cv2.resize(plane, size, interpolation=cv2.INTER_AREA)
    small = _smooth(small, _strength(plane, CHROMA_STRENGTH_PER_SIGMA) / 2)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def _smooth(plane: NDArray[np.float32], strength: float) -> NDArray[np.float32]:
    """
    Non-local-means filter one 0..1 plane at 16-bit precision.

    Returns:
        The filtered plane.
    """
    as_16bit = np.round(np.clip(plane, 0, 1) * MAX_16).astype(np.uint16)
    filtered = cv2.fastNlMeansDenoising(
        as_16bit, [strength], None, TEMPLATE_WINDOW, SEARCH_WINDOW, cv2.NORM_L1
    )
    return filtered.astype(np.float32) / MAX_16


class Denoise(StageImplementation):
    """Writes `denoise/<name>.tif`, the denoised linear intermediate."""

    def process(self, job: StageJob) -> StageResult:
        """
        Denoise the photo's most advanced available source.

        Returns:
            The result listing the denoised intermediate.
        """
        output = job.output_path(suffix=".tif")
        save_linear_rgb(output, denoise_linear(load_linear_rgb(job.source)))
        return StageResult.done(output)


def register(registry: StageRegistry) -> None:
    """Register the Denoise stage."""
    registry.register(StageId.DENOISE, Denoise())
