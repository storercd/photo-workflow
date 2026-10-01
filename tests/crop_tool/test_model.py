"""Tests for UI-independent crop geometry."""

from math import cos, radians, sin

import pytest

from photo_workflow.crop_tool.model import (
    AspectRatio,
    CropRect,
    aspect_ratio_for_crop,
    closest_aspect_ratio,
    constrain_crop_to_rotated_image,
    crop_aspect_ratio,
    display_crop_to_lightroom,
    fit_crop_to_ratio,
    lightroom_crop_to_display,
    resize_from_anchor,
    rotated_image_size,
    transform_crop_for_rotation,
)


def test_aspect_ratio_for_crop_represents_nonstandard_ratio_as_free() -> None:
    """A nonstandard crop keeps its actual ratio under the Free label."""
    crop = CropRect(left=0.1, top=0.2, right=0.8, bottom=0.9)
    ratio = aspect_ratio_for_crop(crop, 6000, 4000)

    assert ratio.label == "Free"
    assert ratio.value == pytest.approx(crop_aspect_ratio(crop, 6000, 4000))


def test_lightroom_serialized_four_by_five_crop_matches_preset_tolerance() -> None:
    """Recognize Lightroom's slightly rounded 4:5 crop while retaining its exact ratio."""
    crop = CropRect(
        left=0.3267751017,
        top=0.4389228411,
        right=0.6251296136,
        bottom=1,
    )
    actual_ratio = crop_aspect_ratio(crop, 6000, 4000)

    assert actual_ratio != pytest.approx(4 / 5, abs=1e-5)
    assert closest_aspect_ratio(
        crop.width,
        crop.height,
        6000,
        4000,
        [AspectRatio(4, 5, "4:5")],
        tolerance=0.005,
    ) == AspectRatio(4, 5, "4:5")


def test_fit_crop_to_ratio_changes_bounds_immediately_and_keeps_center() -> None:
    """A chosen preset fits within the existing crop, centered on its subject."""
    crop = CropRect(left=0.1, top=0.1, right=0.9, bottom=0.9)

    updated = fit_crop_to_ratio(crop, 6000, 4000, AspectRatio(4, 5))

    assert crop_aspect_ratio(updated, 6000, 4000) == pytest.approx(4 / 5)
    assert (updated.left + updated.right) / 2 == pytest.approx(0.5)
    assert (updated.top + updated.bottom) / 2 == pytest.approx(0.5)
    assert updated.left >= crop.left - 1e-9
    assert updated.right <= crop.right + 1e-9
    assert updated.top == pytest.approx(crop.top)
    assert updated.bottom == pytest.approx(crop.bottom)


@pytest.mark.parametrize("angle", [10, 30, -22])
def test_crop_constraint_keeps_all_corners_inside_rotated_image(angle: float) -> None:
    """Constrained crop corners remain over source pixels, not rotated AABB corners."""
    crop = CropRect(0.05, 0.05, 0.95, 0.95)
    width = 1000
    height = 800

    constrained = constrain_crop_to_rotated_image(crop, width, height, angle)

    angle_radians = radians(-angle)
    rotated_width, rotated_height = rotated_image_size(width, height, angle)
    for x, y in (
        (constrained.left, constrained.top),
        (constrained.right, constrained.top),
        (constrained.right, constrained.bottom),
        (constrained.left, constrained.bottom),
    ):
        rotated_x = (x - 0.5) * rotated_width
        rotated_y = (y - 0.5) * rotated_height
        source_x = rotated_x * cos(angle_radians) - rotated_y * sin(angle_radians)
        source_y = rotated_x * sin(angle_radians) + rotated_y * cos(angle_radians)
        assert abs(source_x) <= width / 2 + 1e-6
        assert abs(source_y) <= height / 2 + 1e-6
    assert constrained.width / constrained.height == pytest.approx(crop.width / crop.height)
    assert constrained.width < crop.width


def test_crop_constraint_leaves_zero_angle_crop_unchanged() -> None:
    """With no rotation, the actual image and its bounds are the same rectangle."""
    crop = CropRect(0.05, 0.1, 0.9, 0.8)

    assert constrain_crop_to_rotated_image(crop, 1000, 800, 0) == crop


def test_transform_crop_for_rotation_preserves_aspect_ratio_and_center() -> None:
    """Rotating an upright crop preserves its physical aspect ratio on screen."""
    crop0 = CropRect(0.25, 0.125, 0.75, 0.875)
    assert crop_aspect_ratio(crop0, 6000, 4000) == pytest.approx(1.0)

    crop15 = transform_crop_for_rotation(crop0, 6000, 4000, 0.0, 15.0)
    w15, h15 = rotated_image_size(6000, 4000, 15.0)
    assert crop_aspect_ratio(crop15, w15, h15) == pytest.approx(1.0)
    assert (crop15.left + crop15.right) / 2 == pytest.approx(0.5)
    assert (crop15.top + crop15.bottom) / 2 == pytest.approx(0.5)

    same_angle = transform_crop_for_rotation(crop0, 6000, 4000, 5.0, 5.0)
    expected = constrain_crop_to_rotated_image(crop0, 6000, 4000, 5.0)
    assert same_angle == expected


@pytest.mark.parametrize(
    ("crop", "image_width", "image_height", "angle", "orientation", "expected"),
    [
        (
            CropRect(0.012828, 0.233885, 0.987172, 0.766115),
            6000,
            4000,
            -10,
            1,
            CropRect(0.0920636288, 0.1876470319, 0.9079363712, 0.8123529681),
        ),
        (
            CropRect(0.039335, 0.39907, 0.715726, 0.661195),
            6000,
            4000,
            -10,
            8,
            CropRect(0.3238250837, 0.3239210036, 0.6726011199, 0.9015927833),
        ),
    ],
)
def test_lightroom_crop_to_display_matches_reference_corners(
    crop: CropRect,
    image_width: int,
    image_height: int,
    angle: float,
    orientation: int,
    expected: CropRect,
) -> None:
    """Adobe's crop-corner algorithm yields the reference crop bounds."""
    display_crop = lightroom_crop_to_display(
        crop, image_width, image_height, angle, orientation
    )

    assert display_crop.left == pytest.approx(expected.left)
    assert display_crop.top == pytest.approx(expected.top)
    assert display_crop.right == pytest.approx(expected.right)
    assert display_crop.bottom == pytest.approx(expected.bottom)


@pytest.mark.parametrize("orientation", [1, 6, 8])
@pytest.mark.parametrize("crop_angle", [0, -10])
def test_lightroom_crop_transform_roundtrips(
    orientation: int,
    crop_angle: float,
) -> None:
    """Saving an unchanged upright crop recovers Lightroom's original anchors."""
    crop = CropRect(0.15, 0.2, 0.78, 0.83)

    displayed = lightroom_crop_to_display(crop, 6000, 4000, crop_angle, orientation)
    restored = display_crop_to_lightroom(
        displayed, 6000, 4000, crop_angle, orientation
    )

    assert restored.left == pytest.approx(crop.left)
    assert restored.top == pytest.approx(crop.top)
    assert restored.right == pytest.approx(crop.right)
    assert restored.bottom == pytest.approx(crop.bottom)


def test_closest_aspect_ratio_respects_configured_tolerance() -> None:
    """Only a nearby configured ratio is returned as a snap target."""
    ratio = AspectRatio(4, 5, "4:5")

    assert closest_aspect_ratio(
        0.4266666667, 0.8, 6000, 4000, [ratio], tolerance=0.01
    ) == ratio
    assert closest_aspect_ratio(
        0.4, 0.8, 6000, 4000, [ratio], tolerance=0.01
    ) is None


def test_resize_locks_ratio_and_stays_inside_image() -> None:
    """A locked resize keeps its aspect ratio without crossing image bounds."""
    crop = resize_from_anchor(
        0.1,
        0.1,
        1.2,
        0.9,
        6000,
        4000,
        locked_ratio=AspectRatio(4, 5),
    )

    assert crop.left == pytest.approx(0.1)
    assert crop.top == pytest.approx(0.1)
    assert crop.right <= 1
    assert crop.bottom == pytest.approx(0.9)
    assert crop_aspect_ratio(crop, 6000, 4000) == pytest.approx(4 / 5)


def test_resize_snaps_when_close_to_configured_ratio() -> None:
    """Freeform resize adopts a configured ratio within the snap threshold."""
    crop = resize_from_anchor(
        0.1,
        0.1,
        0.52,
        0.9,
        6000,
        4000,
        snap_ratios=[AspectRatio(4, 5)],
        snap_tolerance=0.1,
    )

    assert crop_aspect_ratio(crop, 6000, 4000) == pytest.approx(4 / 5)


def test_crop_rect_rejects_out_of_bounds_coordinates() -> None:
    """Invalid rectangles cannot enter the crop model."""
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        CropRect(left=-0.1, top=0, right=1, bottom=1)
