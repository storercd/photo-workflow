"""Tests for automatic horizon-angle candidate detection."""

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage

from photo_workflow.crop_tool.horizon import (
    detect_horizon_candidates,
    qimage_to_rgb_array,
)


def test_empty_preview_has_no_horizon_candidates() -> None:
    """A blank preview does not produce a fabricated leveling angle."""
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    assert detect_horizon_candidates(image) == []


def test_sloped_lines_produce_ranked_leveling_alternatives() -> None:
    """Prominent line slopes yield opposite-signed corrective angle suggestions."""
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.line(image, (50, 150), (650, 200), (255, 255, 255), 4)
    cv2.line(image, (80, 380), (620, 270), (255, 255, 255), 4)

    candidates = detect_horizon_candidates(image)
    angles = [candidate.angle_degrees for candidate in candidates]

    assert len(candidates) >= 2
    assert angles[0] == pytest.approx(-4.8, abs=1.5)
    assert any(angle == pytest.approx(11.5, abs=1.5) for angle in angles[1:])
    assert candidates[0].support >= candidates[1].support
    assert candidates[0].line_start[0] == pytest.approx(0)
    assert candidates[0].line_end[0] == pytest.approx(1)
    assert candidates[0].line_start[1] != pytest.approx(candidates[0].line_end[1])


def test_ranking_prefers_near_level_line_over_long_steep_edge() -> None:
    """A strong small-angle horizon can outrank a longer steep architectural edge."""
    image = np.zeros((500, 700, 3), dtype=np.uint8)
    cv2.line(image, (30, 120), (670, 300), (255, 255, 255), 4)
    cv2.line(image, (40, 390), (660, 355), (255, 255, 255), 4)

    candidates = detect_horizon_candidates(image)

    assert candidates[0].angle_degrees == pytest.approx(3.2, abs=1)


def test_qimage_preview_converts_to_packed_rgb_pixels() -> None:
    """The Qt preview's row padding is excluded from the OpenCV RGB array."""
    image = QImage(QSize(13, 7), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)

    pixels = qimage_to_rgb_array(image)

    assert pixels.shape == (7, 13, 3)
    assert pixels[0, 0].tolist() == [255, 0, 0]
