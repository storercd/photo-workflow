"""Mouse hit-testing tests for crop grips."""

import pytest
from PySide6.QtCore import QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photo_workflow.crop_tool.horizon import HorizonCandidate
from photo_workflow.crop_tool.model import AspectRatio, CropRect, crop_aspect_ratio
from photo_workflow.crop_tool.view import CroppedPreview, CropView


def test_crop_grips_are_reachable_outside_edges_and_corners() -> None:
    """Edge and corner resize grips accept points outside the crop boundary."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 400), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    view.set_image(image, 400, 400)
    view.set_crop(CropRect(0.2, 0.2, 0.8, 0.8))
    crop_rect = view._crop_rect(view._image_rect())
    assert crop_rect is not None

    assert view._hit_test(
        QPointF(crop_rect.left() - 8, crop_rect.center().y()), crop_rect
    ) == "left"
    assert view._hit_test(
        QPointF(crop_rect.left() - 12, crop_rect.top() - 12), crop_rect
    ) == "top-left"
    assert view._hit_test(
        QPointF(crop_rect.right() + 8, crop_rect.center().y()), crop_rect
    ) == "right"
    assert view._hit_test(
        QPointF(crop_rect.left() - 30, crop_rect.top()), crop_rect
    ) is None
    view.close()
    app.quit()


def test_loading_state_retains_image_and_blocks_crop_gestures() -> None:
    """Show the previous frame during loading without allowing crop edits."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 400), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    view.set_image(image, 400, 400)
    view.set_crop(CropRect(0.2, 0.2, 0.8, 0.8))
    image_key = view._image.cacheKey()
    view.set_loading(True)
    QTest.mousePress(view, Qt.MouseButton.LeftButton, pos=QPoint(250, 200))

    assert view.is_loading
    assert view._image.cacheKey() == image_key
    assert view._drag_kind is None
    view.close()
    app.quit()


def test_rotation_fit_keeps_rotated_image_inside_viewport() -> None:
    """The image fit accounts for rotated bounds while the crop frame stays upright."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    view.set_image(image, 600, 400)
    view.set_rotation(10)

    image_rect = view._image_rect()

    assert image_rect.width() < 500
    assert image_rect.height() < 400
    assert view._rotation_angle == pytest.approx(10)
    polygon = view._rotated_image_polygon(image_rect)
    bounds = polygon.boundingRect()
    assert bounds.height() > image_rect.height()
    assert bounds.width() > image_rect.width()
    crop = CropRect(0.1, 0.2, 0.9, 0.8)
    view.set_crop(crop)
    crop_rect = view._crop_rect(image_rect)
    crop_plane = view._rotated_bounds_rect(image_rect)
    assert crop_rect is not None
    assert crop_rect.width() == pytest.approx(crop_plane.width() * crop.width)
    assert crop_rect.height() == pytest.approx(crop_plane.height() * crop.height)
    view.close()
    app.quit()


def test_rotated_image_uses_neutral_gray_canvas_outside_photo() -> None:
    """The corners outside a rotated preview show the Lightroom-like gray canvas."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(600, 400), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    view.set_image(image, 600, 400)
    view.set_rotation(10)
    rendered = QImage(view.size(), QImage.Format.Format_RGB32)
    rendered.fill(Qt.GlobalColor.black)
    view.render(rendered)

    assert rendered.pixelColor(0, 0).name() == "#555555"
    view.close()
    app.quit()


def test_horizon_candidate_lines_rotate_with_the_source_image() -> None:
    """Guide endpoints stay attached to image content as the view angle changes."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 300), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.black)
    view.set_image(image, 400, 300)
    candidate = HorizonCandidate(5, 300, (0, 0.4), (1, 0.6))
    image_rect = view._image_rect()

    view.set_rotation(0)
    start, end = view._horizon_line_points(candidate, image_rect)
    initial_slope = (end.y() - start.y()) / (end.x() - start.x())

    view.set_rotation(12)
    rotated_start, rotated_end = view._horizon_line_points(candidate, image_rect)
    rotated_slope = (rotated_end.y() - rotated_start.y()) / (
        rotated_end.x() - rotated_start.x()
    )

    assert initial_slope == pytest.approx(0.15)
    assert rotated_slope != pytest.approx(initial_slope)
    view.close()
    app.quit()


def test_selected_horizon_guide_is_visible_in_rendered_preview() -> None:
    """The active candidate is drawn in amber over the source image."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 300), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    view.set_image(image, 400, 300)
    view.set_crop(CropRect(0, 0, 1, 1))
    view.set_horizon_candidates(
        (HorizonCandidate(0, 400, (0, 0.5), (1, 0.5)),),
        selected_index=0,
    )
    rendered = QImage(view.size(), QImage.Format.Format_RGB32)
    rendered.fill(Qt.GlobalColor.black)

    view.render(rendered)

    assert rendered.pixelColor(250, 200).name() == "#ffd166"
    view.close()
    app.quit()


def test_clicking_horizon_guide_emits_its_candidate_index() -> None:
    """Clicking an alternate line selects it instead of starting a crop drag."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 300), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    view.set_image(image, 400, 300)
    view.set_crop(CropRect(0, 0, 1, 1))
    view.set_horizon_candidates(
        (
            HorizonCandidate(-2, 400, (0, 0.3), (1, 0.3)),
            HorizonCandidate(4, 350, (0, 0.7), (1, 0.7)),
        ),
        selected_index=0,
    )
    selections: list[int] = []
    view.horizon_candidate_selected.connect(selections.append)

    QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=QPoint(250, 275))

    assert selections == [1]
    assert view._drag_kind is None
    view.close()
    app.quit()


def test_cropped_preview_shows_the_selected_source_region() -> None:
    """The read-only result pane maps the active crop to the source pixels."""
    app = QApplication.instance() or QApplication([])
    preview = CroppedPreview()
    preview.resize(200, 160)
    image = QImage(QSize(100, 100), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.blue)
    for y in range(100):
        for x in range(50, 100):
            image.setPixelColor(x, y, Qt.GlobalColor.red)
    preview.set_image(image)
    preview.set_crop(CropRect(0.5, 0, 1, 1))
    rendered = QImage(preview.size(), QImage.Format.Format_RGB32)
    rendered.fill(Qt.GlobalColor.black)

    preview.render(rendered)

    assert rendered.pixelColor(100, 80).name() == "#ff0000"
    preview.close()
    app.quit()


def test_dragging_cropped_preview_pans_crop_within_image_bounds() -> None:
    """Dragging the result pane shifts the crop and clamps it to source bounds."""
    app = QApplication.instance() or QApplication([])
    preview = CroppedPreview()
    preview.resize(200, 160)
    image = QImage(QSize(100, 100), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.blue)
    crop = CropRect(0.2, 0.2, 0.8, 0.8)
    preview.set_image(image)
    preview.set_crop(crop)
    target = preview._target_rect()
    assert target is not None
    changes: list[CropRect] = []
    preview.crop_changed.connect(changes.append)
    start = target.center().toPoint()

    QTest.mousePress(preview, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(preview, QPoint(start.x() + 20, start.y() + 10))
    QTest.mouseRelease(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(start.x() + 20, start.y() + 10),
    )

    assert changes
    assert preview._crop is not None
    assert preview._crop.left < crop.left
    assert preview._crop.top < crop.top
    assert preview._crop.left >= 0
    assert preview._crop.top >= 0

    edge_crop = CropRect(0.2, 0.2, 0.8, 0.8)
    preview.set_crop(edge_crop)
    target = preview._target_rect()
    assert target is not None
    start = target.center().toPoint()
    QTest.mousePress(preview, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(preview, QPoint(preview.width() + 50, start.y()))
    QTest.mouseRelease(
        preview,
        Qt.MouseButton.LeftButton,
        pos=QPoint(preview.width() + 50, start.y()),
    )
    assert preview._crop is not None
    assert preview._crop.left == pytest.approx(0)
    assert preview._crop.right == pytest.approx(0.6)
    preview.close()
    app.quit()


def test_freeform_snap_uses_wider_range_and_marks_snapped_crop() -> None:
    """A near 4:5 freeform crop snaps within 5% and displays its ratio in green."""
    app = QApplication.instance() or QApplication([])
    view = CropView()
    view.resize(500, 400)
    image = QImage(QSize(400, 400), QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    view.set_image(image, 1000, 1000)
    ratio = AspectRatio(4, 5, "4:5")
    view.set_crop_mode(
        locked_ratio=None,
        snap_ratios=(ratio,),
        snap_tolerance=0.05,
    )

    view._resize_from_anchor(QPointF(0.1, 0.1), QPointF(0.516, 0.6))

    assert view._crop is not None
    assert crop_aspect_ratio(view._crop, 1000, 1000) == pytest.approx(4 / 5)
    assert view._snap_ratio == ratio
    rendered = QImage(view.size(), QImage.Format.Format_RGB32)
    rendered.fill(Qt.GlobalColor.black)
    view.render(rendered)
    green_pixels = sum(
        rendered.pixelColor(x, y).name() == "#55e39f"
        for y in range(rendered.height())
        for x in range(rendered.width())
    )
    assert green_pixels > 0

    view.set_crop(None)
    view.set_crop_mode(
        locked_ratio=None,
        snap_ratios=(ratio,),
        snap_tolerance=0.05,
    )
    view._resize_from_anchor(QPointF(0.1, 0.1), QPointF(0.525, 0.6))
    assert view._crop is not None
    assert crop_aspect_ratio(view._crop, 1000, 1000) == pytest.approx(0.85)
    assert view._snap_ratio is None

    view.set_crop_mode(locked_ratio=None, snap_ratios=(), snap_tolerance=0.05)
    assert view._snap_ratio is None
    view.close()
    app.quit()
