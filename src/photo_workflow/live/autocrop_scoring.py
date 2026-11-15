"""
Composition scoring for automatic crop (Qt-free).

A faithful port of the fixed scoring specification (PoC v1): build the rectangle that must
stay inside the crop, generate candidate windows for the acceptable aspect ratios, score
each with six weighted composition rules and pick the best. Coordinates are image pixels
`(x0, y0, x1, y1)`; ratios are width/height values. Nothing here is user-configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WEIGHTS = {"lead": 0.25, "size": 0.20, "ball": 0.15, "thirds": 0.15, "edge": 0.15, "head": 0.10}
SCALES = (1.0, 1.15, 1.3, 1.5, 1.75)
GRID_X = 9
GRID_Y = 7
CLOSE_RATIO_FACTOR = 1.15
SUBJECT_PAD_X = 0.10
SUBJECT_PAD_Y = 0.08
BALL_PAD = 0.5
BALL_MAX_DISTANCE = 2.5
BALL_DIRECTION_MIN_OFFSET = 0.25
MIN_AREA_FRACTION = 0.20
TIE_EPSILON = 1e-9

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CropChoice:
    """
    The winning crop window.

    Attributes:
        window: `(x0, y0, width, height)` in image pixels.
        ratio: The chosen width/height ratio (always one of the allowed ratios).
        scale: How far the snapped minimum window was loosened (1.0 is tightest).
        score: Total score including the resolution factor.
        rules: Per-rule scores (and `res`) of the winner, rounded to 3 decimals.
        required: The clamped rectangle every candidate had to contain.
        fallback: Whether no candidate contained the required rectangle.
    """

    window: tuple[float, float, float, float]
    ratio: float
    scale: float
    score: float
    rules: dict[str, float]
    required: Box
    fallback: bool = False

    @property
    def box(self) -> Box:
        """Return the window as `(x0, y0, x1, y1)`."""
        x, y, width, height = self.window
        return (x, y, x + width, y + height)


def clamp(value: float, low: float, high: float) -> float:
    """
    Limit a value to a range.

    Returns:
        `value` limited to `[low, high]`.
    """
    return max(low, min(high, value))


def ramp(x: float, z0: float, p0: float, p1: float, z1: float) -> float:
    """
    Evaluate a trapezoid: 0 outside `(z0, z1)`, 1 on `[p0, p1]`, linear in between.

    Returns:
        The trapezoid value in 0..1.
    """
    if x <= z0 or x >= z1:
        return 0.0
    if x < p0:
        return (x - z0) / (p0 - z0)
    if x <= p1:
        return 1.0
    return (z1 - x) / (z1 - p1)


def _center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def resolve_direction(subject: Box, ball: Box | None, direction: int) -> int:
    """
    Fill in an unknown direction from the ball when it is clearly to one side.

    Returns:
        `direction` when known, else the ball's side as +1/-1, else 0.
    """
    if direction != 0 or ball is None:
        return direction
    subject_width = subject[2] - subject[0]
    offset = _center(ball)[0] - _center(subject)[0]
    if abs(offset) > BALL_DIRECTION_MIN_OFFSET * subject_width:
        return 1 if offset > 0 else -1
    return 0


def active_ball(subject: Box, ball: Box | None) -> Box | None:
    """
    Return the ball when it is near enough to the subject to matter.

    Returns:
        `ball`, or `None` when absent or farther than 2.5x the subject's larger side.
    """
    if ball is None:
        return None
    subject_center = _center(subject)
    ball_center = _center(ball)
    distance = math.hypot(ball_center[0] - subject_center[0], ball_center[1] - subject_center[1])
    limit = BALL_MAX_DISTANCE * max(subject[2] - subject[0], subject[3] - subject[1])
    return ball if distance <= limit else None


def required_rectangle(
    image_width: float, image_height: float, subject: Box, ball: Box | None
) -> Box:
    """
    Build the padded rectangle every candidate window must contain.

    Returns:
        The subject padded 10 % sideways and 8 % vertically, joined with the padded
        active ball, clamped to the image.
    """
    subject_width = subject[2] - subject[0]
    subject_height = subject[3] - subject[1]
    x0 = subject[0] - SUBJECT_PAD_X * subject_width
    y0 = subject[1] - SUBJECT_PAD_Y * subject_height
    x1 = subject[2] + SUBJECT_PAD_X * subject_width
    y1 = subject[3] + SUBJECT_PAD_Y * subject_height
    if ball is not None:
        pad_x = BALL_PAD * (ball[2] - ball[0])
        pad_y = BALL_PAD * (ball[3] - ball[1])
        x0 = min(x0, ball[0] - pad_x)
        y0 = min(y0, ball[1] - pad_y)
        x1 = max(x1, ball[2] + pad_x)
        y1 = max(y1, ball[3] + pad_y)
    return (max(0, x0), max(0, y0), min(image_width, x1), min(image_height, y1))


def score_window(
    window: Box,
    subject: Box,
    ball: Box | None,
    direction: int,
    image_width: float,
    image_height: float,
) -> tuple[float, dict[str, float]]:
    """
    Score one candidate window.

    Returns:
        The total score and the individual rule scores (with the resolution factor as
        `res`, rounded to three decimals).
    """
    ww = window[2] - window[0]
    wh = window[3] - window[1]
    union = subject
    if ball is not None:
        union = (
            min(subject[0], ball[0]),
            min(subject[1], ball[1]),
            max(subject[2], ball[2]),
            max(subject[3], ball[3]),
        )
    cx = ((subject[0] - window[0]) / ww + (subject[2] - window[0]) / ww) / 2
    scores: dict[str, float] = {}
    if direction != 0:
        position = cx if direction > 0 else 1 - cx
        scores["lead"] = max(0.0, 1 - abs(position - 0.38) / 0.22)
    scores["size"] = ramp(
        max((subject[3] - subject[1]) / wh, (subject[2] - subject[0]) / ww), 0.15, 0.45, 0.75, 0.95
    )
    if ball is not None:
        margin = min(
            ball[0] - window[0],
            ball[1] - window[1],
            window[2] - ball[2],
            window[3] - ball[3],
        ) / min(ww, wh)
        scores["ball"] = 1.0 if margin >= 0.03 else (0.5 if margin >= 0 else 0.0)
    scores["thirds"] = max(0.0, 1 - min(abs(cx - 1 / 3), abs(cx - 2 / 3)) / (1 / 6))
    scores["head"] = ramp((union[1] - window[1]) / wh, 0, 0.08, 0.18, 0.40)
    scores["edge"] = _edge_score(window, union, ww, wh, image_width, image_height)
    total_weight = sum(WEIGHTS[name] for name in scores)
    base = sum(WEIGHTS[name] * value for name, value in scores.items()) / total_weight
    resolution = min(1.0, math.sqrt(ww * wh / (MIN_AREA_FRACTION * image_width * image_height)))
    rounded = {name: round(value, 3) for name, value in scores.items()}
    rounded["res"] = round(resolution, 3)
    return base * resolution, rounded


def _edge_score(
    window: Box, union: Box, ww: float, wh: float, image_width: float, image_height: float
) -> float:
    sides = (
        (union[0] - window[0], window[0] <= 0, union[0] <= 0, ww),
        (window[2] - union[2], window[2] >= image_width, union[2] >= image_width, ww),
        (union[1] - window[1], window[1] <= 0, union[1] <= 0, wh),
        (window[3] - union[3], window[3] >= image_height, union[3] >= image_height, wh),
    )
    values = []
    for margin, window_on_edge, union_on_edge, dimension in sides:
        if window_on_edge and union_on_edge:
            values.append(1.0)
        else:
            values.append(clamp(margin / (0.04 * dimension), 0, 1))
    return min(values)


def choose_crop(
    image_width: float,
    image_height: float,
    subject: Box,
    ball: Box | None,
    direction: int,
    ratios: list[float] | tuple[float, ...],
) -> CropChoice:
    """
    Choose the best crop window for a subject (and optionally a ball).

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        subject: The whole athlete as `(x0, y0, x1, y1)`.
        ball: The ball box when ball influence is on and one was found.
        direction: +1 moving right, -1 moving left, 0 unknown.
        ratios: Acceptable width/height ratios; the result's ratio is always one of them.

    Returns:
        The winning window, its scores, and the required rectangle.

    Raises:
        ValueError: If no ratios are given.
    """
    if not ratios:
        raise ValueError("At least one aspect ratio is required")
    wide_direction = resolve_direction(subject, ball, direction)
    ball = active_ball(subject, ball) if ball is not None else None
    required = required_rectangle(image_width, image_height, subject, ball)
    rw = required[2] - required[0]
    rh = required[3] - required[1]
    base = {ratio: ((rw, rw / ratio) if rw / rh >= ratio else (rh * ratio, rh)) for ratio in ratios}
    smallest = min(width * height for width, height in base.values())
    kept = [ratio for ratio in ratios if base[ratio][0] * base[ratio][1] <= 1.15 * smallest]
    best: tuple[float, float, tuple[float, float, float, float], float, dict[str, float]] | None
    best = None
    for ratio in kept:
        for scale in SCALES:
            window_size = _scaled_window(base[ratio], scale, image_width, image_height)
            if window_size[0] < rw - 1e-6 or window_size[1] < rh - 1e-6:
                continue
            for candidate in _positions(window_size, required, image_width, image_height):
                value, rules = score_window(
                    candidate, subject, ball, wide_direction, image_width, image_height
                )
                if best is None or _beats(value, scale, best[0], best[1]):
                    ww, wh = window_size
                    best = (value, scale, (candidate[0], candidate[1], ww, wh), ratio, rules)
    if best is None:
        return _fallback(image_width, image_height, subject, ball, wide_direction, ratios, required)
    value, scale, window, ratio, rules = best
    return CropChoice(window, ratio, scale, value, rules, required)


def _beats(value: float, scale: float, best_value: float, best_scale: float) -> bool:
    if value > best_value + TIE_EPSILON:
        return True
    return abs(value - best_value) <= TIE_EPSILON and scale < best_scale


def _scaled_window(
    size: tuple[float, float], scale: float, image_width: float, image_height: float
) -> tuple[float, float]:
    ww, wh = size[0] * scale, size[1] * scale
    if ww > image_width or wh > image_height:
        shrink = min(image_width / ww, image_height / wh)
        ww *= shrink
        wh *= shrink
    return ww, wh


def _positions(
    size: tuple[float, float], required: Box, image_width: float, image_height: float
) -> list[Box]:
    ww, wh = size
    x_range = (max(0, required[2] - ww), min(required[0], image_width - ww))
    y_range = (max(0, required[3] - wh), min(required[1], image_height - wh))
    if x_range[0] > x_range[1] + 1e-6 or y_range[0] > y_range[1] + 1e-6:
        return []
    windows = []
    for i in range(GRID_X):
        x = x_range[0] + (x_range[1] - x_range[0]) * (i / (GRID_X - 1))
        for j in range(GRID_Y):
            y = y_range[0] + (y_range[1] - y_range[0]) * (j / (GRID_Y - 1))
            windows.append((x, y, x + ww, y + wh))
    return windows


def _fallback(
    image_width: float,
    image_height: float,
    subject: Box,
    ball: Box | None,
    direction: int,
    ratios: list[float] | tuple[float, ...],
    required: Box,
) -> CropChoice:
    """
    Pick the largest allowed-ratio window centred on the required rectangle.

    Used when the required rectangle is larger than any allowed-ratio window; of the
    allowed ratios, the window covering the most of it wins (earlier ratio on ties).

    Returns:
        The fallback choice, flagged as such.
    """
    center_x, center_y = _center(required)
    best: tuple[float, float, float, float, float] | None = None
    chosen = ratios[0]
    for ratio in ratios:
        ww = min(image_width, image_height * ratio)
        wh = ww / ratio
        x = clamp(center_x - ww / 2, 0, image_width - ww)
        y = clamp(center_y - wh / 2, 0, image_height - wh)
        overlap_w = max(0.0, min(x + ww, required[2]) - max(x, required[0]))
        overlap_h = max(0.0, min(y + wh, required[3]) - max(y, required[1]))
        overlap = overlap_w * overlap_h
        if best is None or overlap > best[0] + TIE_EPSILON:
            best = (overlap, x, y, ww, wh)
            chosen = ratio
    assert best is not None  # noqa: S101 - ratios is non-empty
    _, x, y, ww, wh = best
    value, rules = score_window(
        (x, y, x + ww, y + wh), subject, ball, direction, image_width, image_height
    )
    return CropChoice((x, y, ww, wh), chosen, 1.0, value, rules, required, fallback=True)
