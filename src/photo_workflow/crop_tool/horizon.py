"""Horizon-angle suggestions derived from long, near-horizontal image lines."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, exp, hypot

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtGui import QImage

ANGLE_CLUSTER_DEGREES = 1.5
MAX_LINE_ANGLE_DEGREES = 30.0
ANGLE_PRIOR_SCALE = 12.0


@dataclass(frozen=True)
class HorizonCandidate:
    """One proposed rotation, supporting line geometry, and accumulated support."""

    angle_degrees: float
    support: float
    line_start: tuple[float, float]
    line_end: tuple[float, float]


def detect_horizon_candidates(
    image: NDArray[np.uint8],
    *,
    max_candidates: int = 5,
) -> list[HorizonCandidate]:
    """
    Find and rank corrective rotations suggested by prominent long lines.

    Returns:
        Candidate rotations ordered by accumulated supporting line length.
    """
    if image.ndim not in {2, 3} or image.size == 0:
        return []
    height, width = image.shape[:2]
    if width < 2 or height < 2 or max_candidates <= 0:
        return []
    grayscale = (
        image
        if image.ndim == 2
        else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    )
    edges = cv2.Canny(grayscale, 60, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, round(width * 0.025)),
        minLineLength=max(40, round(width * 0.12)),
        maxLineGap=max(8, round(width * 0.025)),
    )
    if lines is None:
        return []

    groups: list[dict[str, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        delta_x = float(x2 - x1)
        delta_y = float(y2 - y1)
        if delta_x < 0:
            delta_x, delta_y = -delta_x, -delta_y
        line_angle = degrees(atan2(delta_y, delta_x))
        if abs(line_angle) > MAX_LINE_ANGLE_DEGREES:
            continue
        correction = -line_angle
        length = hypot(delta_x, delta_y)
        line_mid_y = (float(y1 + y2) / 2 - height / 2) / height
        group_index = next(
            (
                index
                for index, group in enumerate(groups)
                if abs(correction - group["angle_sum"] / group["support"])
                <= ANGLE_CLUSTER_DEGREES
                and abs(line_mid_y - group["mid_y_sum"] / group["support"])
                <= 0.04
            ),
            None,
        )
        if group_index is None:
            groups.append(
                {
                    "angle_sum": correction * length,
                    "support": length,
                    "mid_y_sum": line_mid_y * length,
                }
            )
        else:
            group = groups[group_index]
            group["angle_sum"] += correction * length
            group["support"] += length
            group["mid_y_sum"] += line_mid_y * length

    candidates = [
        _candidate_from_group(group, width, height)
        for group in groups
    ]
    return sorted(
        candidates,
        key=lambda candidate: candidate.support
        * exp(-abs(candidate.angle_degrees) / ANGLE_PRIOR_SCALE),
        reverse=True,
    )[:max_candidates]


def _candidate_from_group(
    group: dict[str, float],
    width: int,
    height: int,
) -> HorizonCandidate:
    angle = group["angle_sum"] / group["support"]
    middle_y = height / 2 + group["mid_y_sum"] / group["support"] * height
    slope = -np.tan(np.radians(angle))
    half_width = width / 2
    return HorizonCandidate(
        round(angle, 1),
        group["support"],
        (0.0, (middle_y - slope * half_width) / height),
        (1.0, (middle_y + slope * half_width) / height),
    )


def qimage_to_rgb_array(image: QImage) -> NDArray[np.uint8]:
    """
    Copy a Qt preview into a tightly packed RGB array for OpenCV.

    Returns:
        A copied RGB pixel array without Qt row padding.
    """
    rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
    rows = np.frombuffer(rgb_image.constBits(), dtype=np.uint8).reshape(
        rgb_image.height(), rgb_image.bytesPerLine()
    )
    return rows[:, : rgb_image.width() * 3].reshape(
        rgb_image.height(), rgb_image.width(), 3
    ).copy()
