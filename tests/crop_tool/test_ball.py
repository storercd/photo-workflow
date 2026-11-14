"""Tests for volleyball candidate detection."""

import cv2
import numpy as np
import pytest

from photo_workflow.crop_tool.ball import (
    COLOR_SHARE_MINIMUMS,
    color_share_minimums,
    detect_ball_candidates,
)


def _scene(ball_colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> np.ndarray:
    image = np.full((540, 810, 3), 90, dtype=np.uint8)
    cv2.circle(image, (400, 200), 18, ball_colors[0], -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 20, 140, ball_colors[1], -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 200, 300, ball_colors[1], -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 320, 380, (20, 30, 40), -1)
    cv2.circle(image, (400, 200), 18, (30, 30, 30), 1)
    return image


def test_finds_green_and_white_ball() -> None:
    """A round white ball with green panels is returned in normalized coordinates."""
    candidates = detect_ball_candidates(_scene(((240, 240, 240), (20, 140, 60))))

    assert candidates
    assert candidates[0].center_x == pytest.approx(400 / 810, abs=0.02)
    assert candidates[0].center_y == pytest.approx(200 / 540, abs=0.02)


def test_rejects_round_object_without_panel_colors() -> None:
    """A plain white disc has no green panels, so it is not reported."""
    assert detect_ball_candidates(_scene(((240, 240, 240), (240, 240, 240)))) == []


def test_higher_sensitivity_accepts_what_lower_rejects() -> None:
    """A ball with only a sliver of green passes at high sensitivity but not low."""
    image = _scene(((240, 240, 240), (20, 140, 60)))
    cv2.circle(image, (400, 200), 18, (240, 240, 240), -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 20, 42, (20, 140, 60), -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 200, 222, (20, 30, 40), -1)
    cv2.ellipse(image, (400, 200), (18, 18), 0, 100, 160, (100, 100, 100), -1)

    assert detect_ball_candidates(image, sensitivity=0) == []
    assert detect_ball_candidates(image, sensitivity=100)


def test_color_share_minimums_interpolate_between_presets() -> None:
    """The slider passes through the presets and eases between them."""
    assert color_share_minimums(0) == COLOR_SHARE_MINIMUMS[0]
    assert color_share_minimums(50) == COLOR_SHARE_MINIMUMS[1]
    assert color_share_minimums(100) == COLOR_SHARE_MINIMUMS[2]
    assert color_share_minimums(25)[0] == pytest.approx(0.11)
