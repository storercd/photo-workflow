"""Volleyball candidates found with Hough circles plus a panel-colour check."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np
from numpy.typing import NDArray

WORKING_WIDTH = 810
MIN_RADIUS_FRACTION = 0.0086
MAX_RADIUS_FRACTION = 0.0247
MIN_DISTANCE_FRACTION = 0.0185
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 18
MAX_SINGLE_COLOR_SHARE = 0.75
SENSITIVITY_LOW = 0
SENSITIVITY_MEDIUM = 50
SENSITIVITY_HIGH = 100
DEFAULT_SENSITIVITY = SENSITIVITY_MEDIUM
# Minimum share of the circle that must be (white, green, dark) at the low, medium, and
# high sensitivity anchors. Dark panels are what separate a ball from pale shoes and signs.
COLOR_SHARE_MINIMUMS = ((0.12, 0.08, 0.12), (0.10, 0.05, 0.10), (0.06, 0.04, 0.05))


def color_share_minimums(sensitivity: int) -> tuple[float, float, float]:
    """
    Interpolate the colour-share minimums for a 0 to 100 sensitivity.

    Returns:
        The minimum (white, green, dark) shares between the nearest preset anchors.
    """
    position = min(max(sensitivity, SENSITIVITY_LOW), SENSITIVITY_HIGH) / SENSITIVITY_MEDIUM
    lower = min(int(position), len(COLOR_SHARE_MINIMUMS) - 2)
    fraction = position - lower
    start, end = COLOR_SHARE_MINIMUMS[lower], COLOR_SHARE_MINIMUMS[lower + 1]
    white, green, dark = (a + (b - a) * fraction for a, b in zip(start, end, strict=True))
    return white, green, dark


@dataclass(frozen=True)
class BallCandidate:
    """A possible ball as a centre and radius, normalized to the image width."""

    center_x: float
    center_y: float
    radius: float


def detect_ball_candidates(
    image: NDArray[np.uint8],
    *,
    max_candidates: int = 3,
    sensitivity: int = DEFAULT_SENSITIVITY,
) -> list[BallCandidate]:
    """
    Find round regions whose colours look like a green, white, and dark volleyball.

    Higher sensitivity (0 to 100) accepts paler or shadier balls but admits more false hits.

    Returns:
        Colour-verified circles ordered from strongest Hough support.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return []
    height, width = image.shape[:2]
    scale = min(1.0, WORKING_WIDTH / width)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 1.5)
    small_width = small.shape[1]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(1.0, MIN_DISTANCE_FRACTION * small_width),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=max(1, int(MIN_RADIUS_FRACTION * small_width)),
        maxRadius=max(2, int(MAX_RADIUS_FRACTION * small_width)),
    )
    if circles is None:
        return []
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    minimums = color_share_minimums(sensitivity)
    candidates: list[BallCandidate] = []
    for x, y, radius in circles[0]:
        if _looks_like_ball(hsv, float(x), float(y), float(radius), minimums):
            candidates.append(
                BallCandidate(
                    float(x) / small_width,
                    float(y) / small.shape[0],
                    float(radius) / small_width,
                )
            )
            if len(candidates) == max_candidates:
                break
    return candidates


def _looks_like_ball(
    hsv: NDArray[np.uint8],
    cx: float,
    cy: float,
    radius: float,
    minimums: tuple[float, float, float],
) -> bool:
    """
    Require white, green, and dark panels, and no colour dominating the disc.

    Returns:
        Whether the disc's colour mix resembles a volleyball.
    """
    img_h, img_w = hsv.shape[:2]
    reach = int(ceil(radius))
    x0, x1 = max(0, int(cx) - reach), min(img_w, int(cx) + reach + 1)
    y0, y1 = max(0, int(cy) - reach), min(img_h, int(cy) + reach + 1)
    if x1 <= x0 or y1 <= y0:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    total = int(disc.sum())
    if total == 0:
        return False
    hue, sat, val = (hsv[y0:y1, x0:x1, i] for i in range(3))
    masks = {
        "green": (hue >= 40) & (hue <= 100) & (sat >= 50) & (val >= 30),
        "white": (sat <= 70) & (val >= 130),
        "dark": val < 70,
    }
    shares = {name: float((mask & disc).sum()) / total for name, mask in masks.items()}
    min_white, min_green, min_dark = minimums
    return (
        shares["white"] >= min_white
        and shares["green"] >= min_green
        and shares["dark"] >= min_dark
        and max(shares.values()) <= MAX_SINGLE_COLOR_SHARE
    )
