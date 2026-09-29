"""Tests for UI-independent crop geometry."""

import pytest

from photo_workflow.crop_tool.model import (
    AspectRatio,
    CropRect,
    aspect_ratio_for_crop,
    closest_aspect_ratio,
    crop_aspect_ratio,
    fit_crop_to_ratio,
    resize_from_anchor,
)


def test_crop_aspect_ratio_accounts_for_image_dimensions() -> None:
    """Normalized crop bounds are converted using the source image geometry."""
    crop = CropRect(left=7 / 30, top=0, right=23 / 30, bottom=1)

    assert crop_aspect_ratio(crop, 6000, 4000) == pytest.approx(4 / 5)


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
