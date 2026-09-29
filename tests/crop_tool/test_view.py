"""Mouse hit-testing tests for crop grips."""

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
    assert view._drag_kind is None
    view.close()
    app.quit()
