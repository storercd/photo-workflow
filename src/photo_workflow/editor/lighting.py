"""
Fast Lightroom-style "Auto" adjustment for culling previews.

Parameters (white balance, exposure, blacks/whites, shadows/highlights, contrast,
vibrance) are estimated from a small downscaled copy, then baked into one per-channel
lookup table plus a saturation lookup table so the full preview is processed in a few
milliseconds. This is a heuristic approximation meant for culling, not a RAW develop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import log

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

GAMMA = 2.2
ANALYSIS_LONG_EDGE = 256
BYTES_PER_PIXEL = 3
TARGET_MEDIAN = 0.42
MIN_EV = -1.0
MAX_EV = 2.0
MIN_EV_CORRECTION = 0.15
WB_NORM_POWER = 6
WB_STRENGTH = 0.6
WB_MAX_GAIN = 1.2
WB_MIN_SHIFT = 0.02
BLACK_PERCENTILE = 0.5
WHITE_PERCENTILE = 99.5
MAX_BLACK_POINT = 0.08
MIN_WHITE_POINT = 0.9
TARGET_WHITE = 0.98
SHADOW_PERCENTILE = 10
HIGHLIGHT_PERCENTILE = 95
SHADOW_TARGET = 0.2
HIGHLIGHT_TARGET = 0.85
MAX_SHADOW_LIFT = 0.6
MAX_HIGHLIGHT_RECOVERY = 0.5
BASE_CONTRAST = 0.2
TARGET_SATURATION = 0.4
MAX_VIBRANCE = 0.3
MIN_VIBRANCE = 0.03
NEUTRAL_GAINS = (1.0, 1.0, 1.0)
REFERENCE_KELVIN = 6500.0
MIN_KELVIN = 3000.0
MAX_KELVIN = 12000.0


@dataclass(frozen=True)
class LightingAdjustment:
    """The automatic corrections chosen for one preview."""

    gains: tuple[float, float, float] = NEUTRAL_GAINS
    exposure_ev: float = 0.0
    black_point: float = 0.0
    white_point: float = 1.0
    shadows: float = 0.0
    highlights: float = 0.0
    contrast: float = 0.0
    vibrance: float = 0.0

    @property
    def is_neutral(self) -> bool:
        """Whether applying this adjustment would leave the preview unchanged."""
        return self == LightingAdjustment()

    @property
    def temperature(self) -> float:
        """The white-balance color temperature, in Kelvin, implied by the gains."""
        red, _, blue = self.gains
        return kelvin_from_gains(red, blue)

    def describe(self) -> str:
        """
        Summarize the adjustment for the debug bar.

        Returns:
            Human-readable text, such as `exp +1.2 EV, WB R+4%/B-3%, shadows +0.3`.
        """
        if self.is_neutral:
            return "none"
        parts = [
            f"exp {self.exposure_ev:+.1f} EV",
            f"WB {self.temperature:.0f}K",
            f"blacks {self.black_point:.2f}",
            f"whites {self.white_point:.2f}",
            f"shadows {self.shadows:+.2f}",
            f"highlights {-self.highlights:+.2f}",
            f"contrast {self.contrast:+.2f}",
            f"vibrance {self.vibrance:+.2f}",
        ]
        return ", ".join(parts)


def _blackbody_rgb(kelvin: float) -> NDArray[np.float64]:
    """
    Return the linear sRGB color of a blackbody at `kelvin`, scaled so green is 1.

    Uses the Kim et al. Planckian-locus approximation, valid from 1667 K to 25000 K.

    Returns:
        Linear sRGB channel levels of the blackbody, normalized to green = 1.
    """
    t = min(max(kelvin, 1667.0), 25000.0)
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    cie_x, cie_z = x / y, (1 - x - y) / y
    rgb = np.array([
        3.2406 * cie_x - 1.5372 - 0.4986 * cie_z,
        -0.9689 * cie_x + 1.8758 + 0.0415 * cie_z,
        0.0557 * cie_x - 0.2040 + 1.0570 * cie_z,
    ])
    rgb = np.maximum(rgb, 1e-3)
    return rgb / rgb[1]


def gains_from_kelvin(kelvin: float) -> tuple[float, float, float]:
    """
    Convert a white-balance color temperature into RGB gains.

    The photo is assumed lit by a blackbody at `kelvin` and rendered as if neutral at
    6500 K, so lower values cool the image and higher values warm it.

    Returns:
        Red, green (always 1.0), and blue gains.
    """
    gains = _blackbody_rgb(REFERENCE_KELVIN) / _blackbody_rgb(kelvin)
    return float(gains[0]), 1.0, float(gains[2])


def kelvin_from_gains(red: float, blue: float) -> float:
    """
    Find the color temperature whose red/blue balance matches the given gains.

    Returns:
        The temperature in Kelvin, clamped to the supported range.
    """
    target = log(red / blue)
    low, high = MIN_KELVIN, MAX_KELVIN
    for _ in range(24):
        middle = (low + high) / 2
        gains = gains_from_kelvin(middle)
        if log(gains[0] / gains[2]) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def estimate_adjustment(rgb: NDArray[np.uint8], kelvin: float | None = None) -> LightingAdjustment:
    """
    Choose automatic corrections by analyzing a downscaled copy of the preview.

    Args:
        rgb: The preview pixels.
        kelvin: An explicit color temperature replacing the estimated white balance.

    Returns:
        The estimated adjustment, neutral when the preview already looks balanced.
    """
    sample = _downscale(rgb)
    values = sample.astype(np.float32) / 255
    gains = _estimate_gains(values) if kelvin is None else gains_from_kelvin(kelvin)
    adjustment = LightingAdjustment(gains=gains)
    adjustment = replace(adjustment, exposure_ev=_estimate_exposure(values, adjustment))
    levels = _luma(_apply_tone(values, adjustment))
    black, white = np.percentile(levels, [BLACK_PERCENTILE, WHITE_PERCENTILE])
    adjustment = replace(
        adjustment,
        black_point=float(min(black, MAX_BLACK_POINT)),
        white_point=float(max(white, MIN_WHITE_POINT)) if white < TARGET_WHITE else 1.0,
    )
    levels = _luma(_apply_tone(values, adjustment))
    shadow, highlight = np.percentile(levels, [SHADOW_PERCENTILE, HIGHLIGHT_PERCENTILE])
    dynamic_range = float(highlight - shadow)
    adjustment = replace(
        adjustment,
        shadows=float(np.clip((SHADOW_TARGET - shadow) * 2.5, 0, MAX_SHADOW_LIFT)),
        highlights=float(np.clip((highlight - HIGHLIGHT_TARGET) * 4, 0, MAX_HIGHLIGHT_RECOVERY)),
        contrast=BASE_CONTRAST * (1 - float(np.clip((dynamic_range - 0.6) / 0.4, 0, 1))),
    )
    toned = np.clip(_apply_tone(values, adjustment), 0, 1)
    saturation = float(
        cv2.cvtColor((toned * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1].mean() / 255
    )
    adjustment = replace(
        adjustment,
        vibrance=float(np.clip((TARGET_SATURATION - saturation) * 1.5, 0, MAX_VIBRANCE)),
    )
    if (
        abs(adjustment.exposure_ev) < MIN_EV_CORRECTION
        and adjustment.gains == NEUTRAL_GAINS
        and adjustment.black_point < 0.01
        and adjustment.white_point == 1.0
        and adjustment.shadows < 0.05
        and adjustment.highlights < 0.05
        and adjustment.vibrance < MIN_VIBRANCE
    ):
        return LightingAdjustment()
    return adjustment


def apply_adjustment(rgb: NDArray[np.uint8], adjustment: LightingAdjustment) -> NDArray[np.uint8]:
    """
    Apply `adjustment` to a full-size RGB preview using lookup tables.

    Returns:
        A new adjusted array, or `rgb` itself when the adjustment is neutral.
    """
    if adjustment.is_neutral:
        return rgb
    ramp = np.repeat((np.arange(256, dtype=np.float32) / 255)[:, None], 3, axis=1)
    curve = np.round(np.clip(_apply_tone(ramp, adjustment), 0, 1) * 255).astype(np.uint8)
    adjusted = cv2.LUT(rgb, curve.reshape(256, 1, 3))
    if adjustment.vibrance <= 0:
        return adjusted
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    saturation = np.arange(256, dtype=np.float32) / 255
    boost = saturation + adjustment.vibrance * saturation * (1 - saturation) * 2
    hsv[..., 1] = cv2.LUT(hsv[..., 1], np.round(np.clip(boost, 0, 1) * 255).astype(np.uint8))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def auto_light_image(
    image: QImage, kelvin: float | None = None
) -> tuple[QImage, LightingAdjustment]:
    """
    Estimate and apply the automatic adjustment to a Qt image.

    Args:
        image: The preview to adjust.
        kelvin: An explicit color temperature replacing the estimated white balance.

    Returns:
        The adjusted RGB888 image (or `image` itself when null or already balanced) and
        the adjustment that was applied.
    """
    if image.isNull():
        return image, LightingAdjustment()
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    width, height = rgb_image.width(), rgb_image.height()
    rows = np.frombuffer(rgb_image.constBits(), dtype=np.uint8).reshape(
        height, rgb_image.bytesPerLine()
    )
    rgb = np.ascontiguousarray(rows[:, : width * BYTES_PER_PIXEL].reshape(height, width, 3))
    adjustment = estimate_adjustment(rgb, kelvin)
    if adjustment.is_neutral:
        return image, adjustment
    adjusted = np.ascontiguousarray(apply_adjustment(rgb, adjustment))
    adjusted_image = QImage(
        adjusted.data, width, height, width * BYTES_PER_PIXEL, QImage.Format.Format_RGB888
    ).copy()
    return adjusted_image, adjustment


def _downscale(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Return a copy small enough for quick statistics."""
    height, width = rgb.shape[:2]
    scale = ANALYSIS_LONG_EDGE / max(height, width)
    if scale >= 1:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _luma(encoded: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return Rec. 709 luminance of encoded RGB values."""
    clipped = np.clip(encoded, 0, 1)
    return clipped @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _estimate_gains(values: NDArray[np.float32]) -> tuple[float, float, float]:
    """
    Estimate per-channel white-balance gains with the shades-of-gray illuminant guess.

    Returns:
        Damped, clamped RGB gains; neutral when the cast is negligible.
    """
    linear = values ** np.float32(GAMMA)
    peak = linear.max(axis=-1)
    usable = linear[(peak < 0.98) & (peak > 0.02)]
    if usable.size == 0:
        return NEUTRAL_GAINS
    norms = np.power(np.mean(np.power(usable, WB_NORM_POWER), axis=0), 1 / WB_NORM_POWER)
    gains = np.power(np.mean(norms) / np.maximum(norms, 1e-6), WB_STRENGTH)
    gains = np.clip(gains / gains[1], 1 / WB_MAX_GAIN, WB_MAX_GAIN)
    if max(abs(gains - 1)) < WB_MIN_SHIFT:
        return NEUTRAL_GAINS
    return float(gains[0]), 1.0, float(gains[2])


def _estimate_exposure(values: NDArray[np.float32], adjustment: LightingAdjustment) -> float:
    """Return the exposure change, in stops, that moves the median luminance to target."""
    balanced = values ** np.float32(GAMMA) * np.array(adjustment.gains, dtype=np.float32)
    median = max(float(np.median(_luma(np.clip(balanced, 0, 1) ** (1 / GAMMA)))), 1 / 255)
    exposure = float(np.clip(np.log2(TARGET_MEDIAN**GAMMA / median**GAMMA), MIN_EV, MAX_EV))
    return 0.0 if abs(exposure) < MIN_EV_CORRECTION else exposure


def _apply_tone(
    encoded: NDArray[np.float32], adjustment: LightingAdjustment
) -> NDArray[np.float32]:
    """
    Apply white balance, exposure, levels, shadows/highlights, and contrast.

    Operates per channel so the result can be baked into a lookup table.

    Returns:
        Adjusted encoded values, roughly within 0..1.
    """
    linear = encoded**GAMMA * np.array(adjustment.gains, dtype=np.float32)
    gain = 2**adjustment.exposure_ev
    linear = linear * gain
    if gain > 1:
        linear = linear * (1 + linear / gain**2) / (1 + linear)
    tone = np.clip(linear, 0, 1) ** np.float32(1 / GAMMA)
    span = max(adjustment.white_point - adjustment.black_point, 0.05)
    tone = np.clip((tone - adjustment.black_point) / span, 0, 1)
    tone = tone + adjustment.shadows * tone * (1 - tone) ** 2
    tone = tone - adjustment.highlights * tone**2 * (1 - tone)
    return tone + adjustment.contrast * tone * (1 - tone) * (2 * tone - 1)
