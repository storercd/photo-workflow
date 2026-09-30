"""UI-independent crop geometry and aspect-ratio behavior."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import cos, isfinite, radians, sin
from typing import Iterable


@dataclass(frozen=True)
class AspectRatio:
    """A positive width-to-height aspect ratio."""

    width: int
    height: int
    label: str = ""

    def __post_init__(self) -> None:
        """
        Reject invalid ratio dimensions.

        Raises:
            ValueError: If either ratio dimension is not positive.
        """
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Aspect-ratio values must be positive.")

    @property
    def value(self) -> float:
        """The ratio as a floating-point value."""
        return self.width / self.height


@dataclass(frozen=True)
class CropRect:
    """A crop rectangle in normalized image coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        """
        Ensure the rectangle is finite, ordered, and inside the image.

        Raises:
            ValueError: If any coordinate is invalid or out of bounds.
        """
        coordinates = (self.left, self.top, self.right, self.bottom)
        if not all(isfinite(value) for value in coordinates):
            raise ValueError("Crop coordinates must be finite.")
        if not (0 <= self.left < self.right <= 1 and 0 <= self.top < self.bottom <= 1):
            raise ValueError("Crop bounds must be ordered within [0, 1].")

    @property
    def width(self) -> float:
        """The normalized crop width."""
        return self.right - self.left

    @property
    def height(self) -> float:
        """The normalized crop height."""
        return self.bottom - self.top


DEFAULT_ASPECT_RATIOS = (
    AspectRatio(1, 1, "1:1"),
    AspectRatio(4, 5, "4:5"),
    AspectRatio(5, 4, "5:4"),
    AspectRatio(3, 2, "3:2"),
    AspectRatio(2, 3, "2:3"),
    AspectRatio(4, 3, "4:3"),
    AspectRatio(3, 4, "3:4"),
    AspectRatio(16, 9, "16:9"),
)


def rotated_image_size(
    image_width: float,
    image_height: float,
    angle_degrees: float,
) -> tuple[float, float]:
    """Return the axis-aligned pixel bounds after rotating an image around center."""
    angle = radians(angle_degrees)
    cosine = abs(cos(angle))
    sine = abs(sin(angle))
    return (
        image_width * cosine + image_height * sine,
        image_width * sine + image_height * cosine,
    )


def lightroom_crop_to_display(
    crop: CropRect,
    image_width: int,
    image_height: int,
    crop_angle: float,
    orientation: int,
) -> CropRect:
    """
    Map Lightroom crop anchors to the upright crop plane shown by the editor.

    Returns:
        The crop bounds normalized to the rotated display plane.
    """
    oriented_width, oriented_height = _oriented_size(
        image_width, image_height, orientation
    )
    display_angle = -crop_angle
    rotated_width, rotated_height = rotated_image_size(
        oriented_width, oriented_height, display_angle
    )
    center_x = (crop.left + crop.right) * image_width / 2
    center_y = (crop.top + crop.bottom) * image_height / 2
    upper_left = _rotate_point(
        crop.left * image_width,
        crop.top * image_height,
        center_x,
        center_y,
        display_angle,
    )
    lower_right = _rotate_point(
        crop.right * image_width,
        crop.bottom * image_height,
        center_x,
        center_y,
        display_angle,
    )
    leveled_corners = (
        upper_left,
        (lower_right[0], upper_left[1]),
        lower_right,
        (upper_left[0], lower_right[1]),
    )
    crop_corners = _rotate_points(
        list(leveled_corners), center_x, center_y, crop_angle
    )
    visual_corners = [
        _transform_orientation_point(
            x, y, image_width, image_height, orientation
        )
        for x, y in crop_corners
    ]
    display_corners = _rotate_points(
        visual_corners, oriented_width / 2, oriented_height / 2, display_angle
    )
    normalized = [
        (
            (x - oriented_width / 2 + rotated_width / 2) / rotated_width,
            (y - oriented_height / 2 + rotated_height / 2) / rotated_height,
        )
        for x, y in display_corners
    ]
    return _bounds_rect(normalized)


def display_crop_to_lightroom(
    crop: CropRect,
    image_width: int,
    image_height: int,
    crop_angle: float,
    orientation: int,
) -> CropRect:
    """
    Map an upright editor crop back to Lightroom's unrotated crop anchors.

    Returns:
        The crop bounds normalized to the stored sensor pixel array.
    """
    oriented_width, oriented_height = _oriented_size(
        image_width, image_height, orientation
    )
    display_angle = -crop_angle
    rotated_width, rotated_height = rotated_image_size(
        oriented_width, oriented_height, display_angle
    )
    display_corners = _rect_corners(crop, rotated_width, rotated_height)
    visual_corners = [
        _rotate_point(
            x - rotated_width / 2 + oriented_width / 2,
            y - rotated_height / 2 + oriented_height / 2,
            oriented_width / 2,
            oriented_height / 2,
            -display_angle,
        )
        for x, y in display_corners
    ]
    sensor_corners = [
        _transform_orientation_point(
            x, y, oriented_width, oriented_height, INVERSE_ORIENTATIONS[orientation]
        )
        for x, y in visual_corners
    ]
    center_x = sum(x for x, _ in sensor_corners) / len(sensor_corners)
    center_y = sum(y for _, y in sensor_corners) / len(sensor_corners)
    leveled_corners = [
        _rotate_point(x, y, center_x, center_y, -crop_angle)
        for x, y in sensor_corners
    ]
    left = min(x for x, _ in leveled_corners)
    top = min(y for _, y in leveled_corners)
    right = max(x for x, _ in leveled_corners)
    bottom = max(y for _, y in leveled_corners)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    upper_left = _rotate_point(left, top, center_x, center_y, crop_angle)
    lower_right = _rotate_point(right, bottom, center_x, center_y, crop_angle)
    return CropRect(
        min(max(upper_left[0] / image_width, 0), 1),
        min(max(upper_left[1] / image_height, 0), 1),
        min(max(lower_right[0] / image_width, 0), 1),
        min(max(lower_right[1] / image_height, 0), 1),
    )


INVERSE_ORIENTATIONS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


def _oriented_size(image_width: int, image_height: int, orientation: int) -> tuple[int, int]:
    if orientation not in range(1, 9):
        orientation = 1
    return (
        (image_height, image_width)
        if orientation in {5, 6, 7, 8}
        else (image_width, image_height)
    )


def _rect_corners(
    crop: CropRect, image_width: float, image_height: float
) -> list[tuple[float, float]]:
    return [
        (crop.left * image_width, crop.top * image_height),
        (crop.right * image_width, crop.top * image_height),
        (crop.right * image_width, crop.bottom * image_height),
        (crop.left * image_width, crop.bottom * image_height),
    ]


def _rotate_points(
    points: list[tuple[float, float]],
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    return [
        _rotate_point(x, y, center_x, center_y, angle_degrees)
        for x, y in points
    ]


def _rotate_point(
    x: float,
    y: float,
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> tuple[float, float]:
    angle = radians(angle_degrees)
    offset_x = x - center_x
    offset_y = y - center_y
    return (
        center_x + offset_x * cos(angle) - offset_y * sin(angle),
        center_y + offset_x * sin(angle) + offset_y * cos(angle),
    )


def _transform_orientation_point(
    x: float,
    y: float,
    image_width: float,
    image_height: float,
    orientation: int,
) -> tuple[float, float]:
    transforms = {
        1: (x, y),
        2: (image_width - x, y),
        3: (image_width - x, image_height - y),
        4: (x, image_height - y),
        5: (y, x),
        6: (image_height - y, x),
        7: (image_height - y, image_width - x),
        8: (y, image_width - x),
    }
    return transforms.get(orientation, (x, y))


def _bounds_rect(points: list[tuple[float, float]]) -> CropRect:
    x_values, y_values = zip(*points, strict=True)
    return CropRect(
        min(max(min(x_values), 0), 1),
        min(max(min(y_values), 0), 1),
        min(max(max(x_values), 0), 1),
        min(max(max(y_values), 0), 1),
    )


def aspect_ratio_for_crop(
    crop: CropRect,
    image_width: int,
    image_height: int,
    *,
    label: str = "Free",
) -> AspectRatio:
    """
    Represent a crop's displayed aspect ratio as an aspect-ratio value.

    Returns:
        The closest rational representation of the displayed crop ratio.
    """
    value = crop_aspect_ratio(crop, image_width, image_height)
    fraction = Fraction(value).limit_denominator(100_000)
    return AspectRatio(fraction.numerator, fraction.denominator, label)


def fit_crop_to_ratio(
    crop: CropRect,
    image_width: int,
    image_height: int,
    ratio: AspectRatio,
) -> CropRect:
    """
    Fit a crop to a requested displayed ratio, centered inside its current bounds.

    Returns:
        The resized crop contained within the original crop bounds.
    """
    _validate_image_size(image_width, image_height)
    normalized_ratio = ratio.value * image_height / image_width
    height = min(crop.height, crop.width / normalized_ratio)
    width = height * normalized_ratio
    center_x = (crop.left + crop.right) / 2
    center_y = (crop.top + crop.bottom) / 2
    return CropRect(
        max(0, center_x - width / 2),
        max(0, center_y - height / 2),
        min(1, center_x + width / 2),
        min(1, center_y + height / 2),
    )


def crop_aspect_ratio(crop: CropRect, image_width: int, image_height: int) -> float:
    """
    Return the displayed width-to-height ratio of a normalized crop.

    Returns:
        The crop's displayed width-to-height ratio.
    """
    _validate_image_size(image_width, image_height)
    return crop.width * image_width / (crop.height * image_height)


def closest_aspect_ratio(
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    ratios: Iterable[AspectRatio],
    *,
    tolerance: float,
) -> AspectRatio | None:
    """
    Return the nearest ratio when its relative error is within tolerance.

    Returns:
        The closest acceptable ratio, or `None` if none is close enough.

    Raises:
        ValueError: If crop dimensions are not positive or tolerance is negative.
    """
    _validate_image_size(image_width, image_height)
    if width <= 0 or height <= 0 or tolerance < 0:
        raise ValueError("Crop dimensions must be positive and tolerance nonnegative.")

    candidate_ratio = width * image_width / (height * image_height)
    closest: AspectRatio | None = None
    closest_error = float("inf")
    for ratio in ratios:
        relative_error = abs(candidate_ratio / ratio.value - 1)
        if relative_error < closest_error:
            closest = ratio
            closest_error = relative_error
    return closest if closest_error <= tolerance else None


def resize_from_anchor(
    anchor_x: float,
    anchor_y: float,
    pointer_x: float,
    pointer_y: float,
    image_width: int,
    image_height: int,
    *,
    locked_ratio: AspectRatio | None = None,
    snap_ratios: Iterable[AspectRatio] = (),
    snap_tolerance: float = 0.025,
) -> CropRect:
    """
    Resize a crop from its fixed opposite corner, optionally locking/snapping.

    Returns:
        The bounded crop rectangle.

    Raises:
        ValueError: If the anchor is outside the image or the crop has no area.
    """
    _validate_image_size(image_width, image_height)
    if not 0 <= anchor_x <= 1 or not 0 <= anchor_y <= 1:
        raise ValueError("Anchor coordinates must be within [0, 1].")

    direction_x = 1 if pointer_x >= anchor_x else -1
    direction_y = 1 if pointer_y >= anchor_y else -1
    max_width = anchor_x if direction_x < 0 else 1 - anchor_x
    max_height = anchor_y if direction_y < 0 else 1 - anchor_y
    requested_width = min(abs(pointer_x - anchor_x), max_width)
    requested_height = min(abs(pointer_y - anchor_y), max_height)
    if requested_width <= 0 or requested_height <= 0:
        raise ValueError("Crop must have positive width and height.")

    ratio = locked_ratio
    if ratio is None:
        ratio = closest_aspect_ratio(
            requested_width,
            requested_height,
            image_width,
            image_height,
            snap_ratios,
            tolerance=snap_tolerance,
        )
    if ratio is not None:
        target_width_height = ratio.value * image_height / image_width
        scale = min(
            requested_width / target_width_height,
            requested_height,
            max_width / target_width_height,
            max_height,
        )
        requested_width = scale * target_width_height
        requested_height = scale

    moving_x = anchor_x + direction_x * requested_width
    moving_y = anchor_y + direction_y * requested_height
    return CropRect(
        left=min(anchor_x, moving_x),
        top=min(anchor_y, moving_y),
        right=max(anchor_x, moving_x),
        bottom=max(anchor_y, moving_y),
    )


def _validate_image_size(image_width: int, image_height: int) -> None:
    """
    Reject nonpositive image dimensions.

    Raises:
        ValueError: If either image dimension is not positive.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
