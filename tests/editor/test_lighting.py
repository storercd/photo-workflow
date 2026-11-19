"""Tests for automatic preview lighting adjustment."""

from dataclasses import replace

import numpy as np
from PySide6.QtGui import QImage

from photo_workflow.editor.lighting import (
    LightingAdjustment,
    apply_adjustment,
    auto_light_image,
    estimate_adjustment,
)


def scene(low: int, high: int, tint: tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
    """
    Build a gradient scene between two brightness levels.

    Returns:
        An RGB array with each channel scaled by `tint`.
    """
    ramp = np.linspace(low, high, 96, dtype=np.float32)
    plane = np.tile(ramp, (64, 1))
    return np.clip(np.stack([plane * gain for gain in tint], axis=-1), 0, 255).astype(np.uint8)


def test_dark_scene_is_brightened_without_clipping_highlights() -> None:
    """Underexposed scenes gain exposure and lifted shadows."""
    dark = scene(5, 90)
    adjustment = estimate_adjustment(dark)
    result = apply_adjustment(dark, adjustment)

    assert adjustment.exposure_ev > 0.5
    assert result.mean() > dark.mean() + 30


def test_bright_scene_is_darkened() -> None:
    """Overexposed scenes lose exposure."""
    bright = scene(150, 255)

    assert estimate_adjustment(bright).exposure_ev < 0
    assert apply_adjustment(bright, estimate_adjustment(bright)).mean() < bright.mean()


def test_color_cast_is_reduced() -> None:
    """A strong warm cast is pulled toward neutral."""
    warm = scene(40, 200, tint=(1.0, 0.85, 0.6))
    result = apply_adjustment(warm, replace(estimate_adjustment(warm), skin_tone=0.0))

    before = warm[..., 0].mean() / warm[..., 2].mean()
    after = result[..., 0].mean() / result[..., 2].mean()
    assert after < before


def test_skin_tone_enriches_skin_and_leaves_other_colors() -> None:
    """Skin-colored pixels gain chroma; a blue pixel is untouched."""
    pixels = np.array([[[200, 150, 120], [40, 80, 200]]], dtype=np.uint8)
    result = apply_adjustment(pixels, LightingAdjustment(skin_tone=1.0))

    skin_before, skin_after = pixels[0, 0].astype(int), result[0, 0].astype(int)
    assert skin_after[0] - skin_after[2] > skin_before[0] - skin_before[2]
    assert np.array_equal(result[0, 1], pixels[0, 1])


def test_neutral_adjustment_returns_same_array() -> None:
    """Applying a neutral adjustment is a no-op."""
    image = scene(0, 255)

    assert apply_adjustment(image, LightingAdjustment()) is image
    assert LightingAdjustment().describe() == "none"


def test_image_keeps_size_and_reports_adjustment() -> None:
    """Qt images keep their dimensions and report the applied adjustment."""
    image = QImage(40, 30, QImage.Format.Format_RGB888)
    image.fill(0x1E1E1E)

    result, adjustment = auto_light_image(image)

    assert (result.width(), result.height()) == (40, 30)
    assert not adjustment.is_neutral
    assert result.pixelColor(5, 5).red() > 0x1E


def test_explicit_kelvin_overrides_estimate() -> None:
    """A fixed color temperature replaces the estimated white balance."""
    warm = scene(40, 200, tint=(1.0, 0.85, 0.6))

    estimated = estimate_adjustment(warm)
    fixed = estimate_adjustment(warm, kelvin=9000)

    assert estimated.temperature < 6500
    assert abs(fixed.temperature - 9000) < 50
    assert fixed.gains[0] > 1 > fixed.gains[2]
    assert estimate_adjustment(warm, kelvin=6500).gains == (1.0, 1.0, 1.0)


def test_kelvin_is_relative_to_the_as_shot_reference() -> None:
    """The camera's as-shot temperature is neutral, and typed values shift away from it."""
    warm = scene(40, 200, tint=(1.0, 0.85, 0.6))

    neutral = estimate_adjustment(warm, kelvin=5200, reference=5200)
    shifted = estimate_adjustment(warm, kelvin=7000, reference=5200)

    assert neutral.gains == (1.0, 1.0, 1.0)
    assert abs(neutral.temperature - 5200) < 50
    assert shifted.gains[0] > 1 > shifted.gains[2]
    assert abs(shifted.temperature - 7000) < 50
