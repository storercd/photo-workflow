"""Mouse hit-testing tests for crop grips."""

import pytest
from PySide6.QtCore import QPoint, QPointF, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from photo_workflow.crop_tool.model import CropRect
from photo_workflow.crop_tool.view import CropView


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
