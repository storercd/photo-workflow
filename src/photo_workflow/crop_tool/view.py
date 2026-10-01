"""Qt image surface with mouse-driven crop interaction."""

from __future__ import annotations

from math import cos, hypot, radians, sin

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QWidget

from photo_workflow.crop_tool.horizon import HorizonCandidate
from photo_workflow.crop_tool.model import (
    AspectRatio,
    CropRect,
    closest_aspect_ratio,
    constrain_crop_to_rotated_image,
    resize_from_anchor,
)

EDGE_HIT_PIXELS = 24
HORIZON_HIT_PIXELS = 10
MIN_CROP_PIXELS = 2
CANVAS_COLOR = QColor("#555555")
SNAPPED_CROP_COLOR = QColor("#55e39f")


class CropView(QWidget):
    """Display a photo and emit normalized crop changes from mouse gestures."""

    crop_changed = Signal(object)
    horizon_candidate_selected = Signal(int)

    def __init__(self) -> None:
        """Initialize an empty image surface."""
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("background: #555555;")
        self._image = QImage()
        self._loading = False
        self._crop: CropRect | None = None
        self._image_width = 1
        self._image_height = 1
        self._rotation_angle = 0.0
        self._locked_ratio: AspectRatio | None = None
        self._snap_ratios: tuple[AspectRatio, ...] = ()
        self._snap_tolerance = 0.05
        self._snap_ratio: AspectRatio | None = None
        self._drag_kind: str | None = None
        self._drag_anchor = QPointF()
        self._drag_origin = QPointF()
        self._initial_crop: CropRect | None = None
        self._horizon_candidates: tuple[HorizonCandidate, ...] = ()
        self._selected_horizon_candidate = -1

    def set_image(self, image: QImage, image_width: int, image_height: int) -> None:
        """Set the displayed preview and source dimensions."""
        self._image = image
        self._image_width = max(1, image_width)
        self._image_height = max(1, image_height)
        self.update()

    @property
    def is_loading(self) -> bool:
        """Whether the selected photo is still loading."""
        return self._loading

    def set_loading(self, loading: bool) -> None:
        """Show a static loading veil and disable crop gestures while loading."""
        self._loading = loading
        if loading:
            self._drag_kind = None
            self._initial_crop = None
            self.unsetCursor()
        self.update()

    def set_crop(self, crop: CropRect | None) -> CropRect | None:
        """
        Set a crop constrained to the rotated image without emitting an edit.

        Returns:
            The constrained crop, or `None` when no crop is active.
        """
        self._crop = self._constrain_crop(crop) if crop is not None else None
        self._snap_ratio = (
            self._matching_snap_ratio(self._crop) if self._crop is not None else None
        )
        self.update()
        return self._crop

    def set_rotation(self, angle_degrees: float) -> None:
        """Rotate the photo beneath the upright crop frame."""
        self._rotation_angle = angle_degrees
        if self._crop is not None:
            self._snap_ratio = self._matching_snap_ratio(self._crop)
        self.update()

    def set_horizon_candidates(
        self,
        candidates: tuple[HorizonCandidate, ...],
        selected_index: int = -1,
    ) -> None:
        """Set image-anchored candidate lines and the guide to emphasize."""
        self._horizon_candidates = candidates
        self._selected_horizon_candidate = selected_index
        self.update()

    def set_crop_mode(
        self,
        *,
        locked_ratio: AspectRatio | None,
        snap_ratios: tuple[AspectRatio, ...],
        snap_tolerance: float,
    ) -> None:
        """Set the resize constraint and freeform snap configuration."""
        self._locked_ratio = locked_ratio
        self._snap_ratios = snap_ratios
        self._snap_tolerance = snap_tolerance
        self._snap_ratio = (
            self._matching_snap_ratio(self._crop)
            if self._crop is not None
            else None
        )
        self.update()

    def paintEvent(self, event: object) -> None:
        """Paint the preview, shaded crop mask, and crop guides."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), CANVAS_COLOR)
        if self._image.isNull():
            painter.setPen(QColor("#a7acb2"))
            message = "Loading preview..." if self._loading else "Open a folder to begin"
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, message)
            return

        image_rect = self._image_rect()
        painter.save()
        painter.translate(image_rect.center())
        painter.rotate(self._rotation_angle)
        painter.drawImage(
            QRectF(
                -image_rect.width() / 2,
                -image_rect.height() / 2,
                image_rect.width(),
                image_rect.height(),
            ),
            self._image,
        )
        painter.restore()
        if self._loading:
            painter.save()
            painter.translate(image_rect.center())
            painter.rotate(self._rotation_angle)
            painter.fillRect(
                QRectF(
                    -image_rect.width() / 2,
                    -image_rect.height() / 2,
                    image_rect.width(),
                    image_rect.height(),
                ),
                QColor(0, 0, 0, 120),
            )
            painter.restore()
            painter.setPen(QColor("#f4f2ec"))
            painter.drawText(image_rect, Qt.AlignmentFlag.AlignCenter, "Loading preview...")
            return
        crop_rect = self._crop_rect(image_rect)
        if crop_rect is None:
            self._draw_horizon_candidates(painter, image_rect)
            return

        mask = QPainterPath()
        mask.addPolygon(self._rotated_image_polygon(image_rect))
        cutout = QPainterPath()
        cutout.addRect(crop_rect)
        painter.fillPath(mask.subtracted(cutout), QColor(0, 0, 0, 145))
        self._draw_horizon_candidates(painter, image_rect)
        frame_color = SNAPPED_CROP_COLOR if self._snap_ratio is not None else QColor("#f4f2ec")
        painter.setPen(QPen(frame_color, 2 if self._snap_ratio is not None else 1.5))
        painter.drawRect(crop_rect)
        if self._snap_ratio is not None:
            self._draw_snap_label(painter, crop_rect, self._snap_ratio)
        painter.setPen(QPen(QColor(244, 242, 236, 110), 1))
        painter.drawLine(
            crop_rect.left() + crop_rect.width() / 3,
            crop_rect.top(),
            crop_rect.left() + crop_rect.width() / 3,
            crop_rect.bottom(),
        )
        painter.drawLine(
            crop_rect.left() + crop_rect.width() * 2 / 3,
            crop_rect.top(),
            crop_rect.left() + crop_rect.width() * 2 / 3,
            crop_rect.bottom(),
        )
        painter.drawLine(
            crop_rect.left(),
            crop_rect.top() + crop_rect.height() / 3,
            crop_rect.right(),
            crop_rect.top() + crop_rect.height() / 3,
        )
        painter.drawLine(
            crop_rect.left(),
            crop_rect.top() + crop_rect.height() * 2 / 3,
            crop_rect.right(),
            crop_rect.top() + crop_rect.height() * 2 / 3,
        )

    def _draw_horizon_candidates(self, painter: QPainter, image_rect: QRectF) -> None:
        """Draw candidate guides in the same image-local transform as the preview."""
        if not self._horizon_candidates:
            return
        painter.save()
        painter.translate(image_rect.center())
        painter.rotate(self._rotation_angle)
        local_image = QRectF(
            -image_rect.width() / 2,
            -image_rect.height() / 2,
            image_rect.width(),
            image_rect.height(),
        )
        painter.setClipRect(local_image)
        for index, candidate in enumerate(self._horizon_candidates):
            selected = index == self._selected_horizon_candidate
            color = QColor("#ffd166") if selected else QColor("#45d6d0")
            pen = QPen(color, 2.5 if selected else 1.5)
            if not selected:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            start = QPointF(
                (candidate.line_start[0] - 0.5) * image_rect.width(),
                (candidate.line_start[1] - 0.5) * image_rect.height(),
            )
            end = QPointF(
                (candidate.line_end[0] - 0.5) * image_rect.width(),
                (candidate.line_end[1] - 0.5) * image_rect.height(),
            )
            painter.drawLine(start, end)
        painter.restore()

    def _horizon_line_points(
        self,
        candidate: HorizonCandidate,
        image_rect: QRectF,
    ) -> tuple[QPointF, QPointF]:
        """
        Map a candidate's normalized source line through the current rotation.

        Returns:
            The rotated endpoints in widget coordinates.
        """
        center = image_rect.center()
        angle = radians(self._rotation_angle)
        cosine = cos(angle)
        sine = sin(angle)
        points = []
        for normalized_x, normalized_y in (
            candidate.line_start,
            candidate.line_end,
        ):
            offset_x = (normalized_x - 0.5) * image_rect.width()
            offset_y = (normalized_y - 0.5) * image_rect.height()
            points.append(
                QPointF(
                    center.x() + offset_x * cosine - offset_y * sine,
                    center.y() + offset_x * sine + offset_y * cosine,
                )
            )
        return points[0], points[1]

    def _horizon_candidate_at(
        self,
        point: QPointF,
        image_rect: QRectF,
    ) -> int | None:
        """
        Find the closest guide within its clickable screen-space hit zone.

        Returns:
            The nearest matching candidate index, or `None` outside all hit zones.
        """
        nearest_index = None
        nearest_distance = HORIZON_HIT_PIXELS
        for index, candidate in enumerate(self._horizon_candidates):
            start, end = self._horizon_line_points(candidate, image_rect)
            segment_x = end.x() - start.x()
            segment_y = end.y() - start.y()
            segment_length_squared = segment_x * segment_x + segment_y * segment_y
            if segment_length_squared == 0:
                distance = hypot(point.x() - start.x(), point.y() - start.y())
            else:
                projection = (
                    (point.x() - start.x()) * segment_x
                    + (point.y() - start.y()) * segment_y
                ) / segment_length_squared
                projection = min(max(projection, 0), 1)
                closest_x = start.x() + projection * segment_x
                closest_y = start.y() + projection * segment_y
                distance = hypot(point.x() - closest_x, point.y() - closest_y)
            if distance <= nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def mousePressEvent(self, event: object) -> None:
        """Start a new crop or begin moving/resizing the current crop."""
        if self._loading:
            event.ignore()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        image_rect = self._image_rect()
        candidate_index = self._horizon_candidate_at(event.position(), image_rect)
        if candidate_index is not None:
            self.horizon_candidate_selected.emit(candidate_index)
            event.accept()
            return
        point = event.position()
        if not self._rotated_image_polygon(image_rect).containsPoint(
            point, Qt.FillRule.WindingFill
        ):
            return
        crop_plane = self._rotated_bounds_rect(image_rect)
        normalized = self._to_normalized(point, crop_plane)
        crop_rect = self._crop_rect(image_rect)
        if self._crop is None or self._is_full_crop(self._crop):
            self._drag_kind = "draw"
            self._drag_anchor = normalized
        else:
            self._drag_kind = self._hit_test(point, crop_rect)
            if self._drag_kind is None:
                self._drag_kind = "draw"
                self._drag_anchor = normalized
        self._drag_origin = normalized
        self._initial_crop = self._crop
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: object) -> None:
        """Update the crop while dragging or show the appropriate cursor."""
        if self._loading:
            return
        image_rect = self._image_rect()
        crop_plane = self._rotated_bounds_rect(image_rect)
        point = event.position()
        if self._drag_kind is None:
            if self._horizon_candidate_at(point, image_rect) is not None:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self._update_hover_cursor(point, image_rect)
            return
        polygon = self._rotated_image_polygon(image_rect)
        if not polygon.containsPoint(point, Qt.FillRule.WindingFill):
            point = self._closest_point_on_polygon(point, polygon)
        normalized = self._to_normalized(point, crop_plane)
        if self._drag_kind == "draw":
            self._resize_from_anchor(self._drag_anchor, normalized)
        elif self._drag_kind == "move":
            self._move_crop(normalized)
        else:
            self._resize_existing(normalized)
        event.accept()

    def mouseReleaseEvent(self, event: object) -> None:
        """Finish the active crop gesture."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_kind = None
            self._initial_crop = None
            self.unsetCursor()
            event.accept()

    def _image_rect(self) -> QRectF:
        """Return the preview's aspect-preserving display rectangle."""
        if self._image.isNull():
            return QRectF()
        angle = radians(self._rotation_angle)
        rotated_width = abs(self._image.width() * cos(angle)) + abs(
            self._image.height() * sin(angle)
        )
        rotated_height = abs(self._image.width() * sin(angle)) + abs(
            self._image.height() * cos(angle)
        )
        scale = min(self.width() / rotated_width, self.height() / rotated_height)
        width = self._image.width() * scale
        height = self._image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _rotated_image_polygon(self, image_rect: QRectF) -> QPolygonF:
        """Return the photo's actual rotated boundary in widget coordinates."""
        angle = radians(self._rotation_angle)
        cosine = cos(angle)
        sine = sin(angle)
        center = image_rect.center()
        points = []
        for point in (
            image_rect.topLeft(),
            image_rect.topRight(),
            image_rect.bottomRight(),
            image_rect.bottomLeft(),
        ):
            offset_x = point.x() - center.x()
            offset_y = point.y() - center.y()
            points.append(
                QPointF(
                    center.x() + offset_x * cosine - offset_y * sine,
                    center.y() + offset_x * sine + offset_y * cosine,
                )
            )
        return QPolygonF(points)

    def _rotated_bounds_rect(self, image_rect: QRectF) -> QRectF:
        """Return the upright crop-plane bounds around the rotated image."""
        return self._rotated_image_polygon(image_rect).boundingRect()

    @staticmethod
    def _closest_point_on_polygon(point: QPointF, polygon: QPolygonF) -> QPointF:
        """
        Project an outside pointer to the nearest point on the photo boundary.

        Returns:
            The nearest point on the polygon boundary.
        """
        closest_point = polygon.first()
        closest_distance = float("inf")
        for index, start in enumerate(polygon):
            end = polygon[(index + 1) % len(polygon)]
            edge_x = end.x() - start.x()
            edge_y = end.y() - start.y()
            edge_length_squared = edge_x * edge_x + edge_y * edge_y
            if edge_length_squared == 0:
                projected = start
            else:
                amount = (
                    (point.x() - start.x()) * edge_x
                    + (point.y() - start.y()) * edge_y
                ) / edge_length_squared
                amount = min(max(amount, 0), 1)
                projected = QPointF(
                    start.x() + amount * edge_x,
                    start.y() + amount * edge_y,
                )
            distance = hypot(point.x() - projected.x(), point.y() - projected.y())
            if distance < closest_distance:
                closest_point = projected
                closest_distance = distance
        return closest_point

    def _crop_rect(self, image_rect: QRectF) -> QRectF | None:
        """
        Map normalized crop bounds into widget coordinates.

        Returns:
            The crop rectangle, or `None` when no crop is active.
        """
        if self._crop is None:
            return None
        crop_plane = self._rotated_bounds_rect(image_rect)
        return QRectF(
            crop_plane.left() + self._crop.left * crop_plane.width(),
            crop_plane.top() + self._crop.top * crop_plane.height(),
            self._crop.width * crop_plane.width(),
            self._crop.height * crop_plane.height(),
        )

    def _to_normalized(self, point: QPointF, image_rect: QRectF) -> QPointF:
        """
        Map widget coordinates to clamped normalized image coordinates.

        Returns:
            The image-relative point with both coordinates in [0, 1].
        """
        return QPointF(
            min(max((point.x() - image_rect.left()) / image_rect.width(), 0), 1),
            min(max((point.y() - image_rect.top()) / image_rect.height(), 0), 1),
        )

    def _hit_test(self, point: QPointF, crop_rect: QRectF | None) -> str | None:
        """
        Return the edge, corner, or interior drag mode under the pointer.

        Returns:
            The drag mode name, or `None` when outside the crop.
        """
        if crop_rect is None:
            return None
        hit_rect = crop_rect.adjusted(
            -EDGE_HIT_PIXELS,
            -EDGE_HIT_PIXELS,
            EDGE_HIT_PIXELS,
            EDGE_HIT_PIXELS,
        )
        if not hit_rect.contains(point):
            return None
        near_left = abs(point.x() - crop_rect.left()) <= EDGE_HIT_PIXELS
        near_right = abs(point.x() - crop_rect.right()) <= EDGE_HIT_PIXELS
        near_top = abs(point.y() - crop_rect.top()) <= EDGE_HIT_PIXELS
        near_bottom = abs(point.y() - crop_rect.bottom()) <= EDGE_HIT_PIXELS
        return self._classify_grip(near_left, near_right, near_top, near_bottom)

    @staticmethod
    def _classify_grip(
        near_left: bool,
        near_right: bool,
        near_top: bool,
        near_bottom: bool,
    ) -> str:
        """
        Choose the closest matching corner or edge grip.

        Returns:
            A grip name or `move` for the crop interior.
        """
        vertical = "top" if near_top else "bottom" if near_bottom else ""
        horizontal = "left" if near_left else "right" if near_right else ""
        if vertical and horizontal:
            return f"{vertical}-{horizontal}"
        return vertical or horizontal or "move"

    def _resize_from_anchor(self, anchor: QPointF, point: QPointF) -> None:
        """Create or resize a crop from its fixed opposite point."""
        try:
            crop = resize_from_anchor(
                anchor.x(),
                anchor.y(),
                point.x(),
                point.y(),
                self._image_width,
                self._image_height,
                locked_ratio=self._locked_ratio,
                snap_ratios=self._snap_ratios,
                snap_tolerance=self._snap_tolerance,
            )
        except ValueError:
            return
        self._set_edited_crop(crop)

    def _move_crop(self, point: QPointF) -> None:
        """Move the existing crop while preserving its size and image bounds."""
        if self._initial_crop is None:
            return
        delta_x = point.x() - self._drag_origin.x()
        delta_y = point.y() - self._drag_origin.y()
        left = min(max(self._initial_crop.left + delta_x, 0), 1 - self._initial_crop.width)
        top = min(max(self._initial_crop.top + delta_y, 0), 1 - self._initial_crop.height)
        self._set_edited_crop(
            CropRect(left, top, left + self._initial_crop.width, top + self._initial_crop.height)
        )

    def _resize_existing(self, point: QPointF) -> None:
        """Resize the selected edge or corner of the original crop."""
        if self._initial_crop is None or self._drag_kind is None:
            return
        if self._drag_kind in {"left", "right", "top", "bottom"}:
            crop = self._resize_edge(point)
        else:
            crop = self._resize_corner(point)
        if crop is not None:
            self._set_edited_crop(crop)

    def _resize_corner(self, point: QPointF) -> CropRect | None:
        """
        Resize from the opposite corner, applying lock or snap settings.

        Returns:
            The updated crop, or `None` if the new geometry is invalid.
        """
        assert self._initial_crop is not None and self._drag_kind is not None
        anchor_x = (
            self._initial_crop.left if "right" in self._drag_kind else self._initial_crop.right
        )
        anchor_y = (
            self._initial_crop.top if "bottom" in self._drag_kind else self._initial_crop.bottom
        )
        try:
            return resize_from_anchor(
                anchor_x,
                anchor_y,
                point.x(),
                point.y(),
                self._image_width,
                self._image_height,
                locked_ratio=self._locked_ratio,
                snap_ratios=self._snap_ratios,
                snap_tolerance=self._snap_tolerance,
            )
        except ValueError:
            return None

    def _resize_edge(self, point: QPointF) -> CropRect | None:
        """
        Resize one edge, centering any ratio-constrained secondary dimension.

        Returns:
            The updated crop, or `None` if the new geometry is invalid.
        """
        assert self._initial_crop is not None and self._drag_kind is not None
        crop = self._initial_crop
        if self._drag_kind in {"left", "right"}:
            left = point.x() if self._drag_kind == "left" else crop.left
            right = crop.right if self._drag_kind == "left" else point.x()
            width = abs(right - left)
            height = crop.height
        else:
            top = point.y() if self._drag_kind == "top" else crop.top
            bottom = crop.bottom if self._drag_kind == "top" else point.y()
            height = abs(bottom - top)
            width = crop.width

        if width <= 0 or height <= 0:
            return None
        ratio = self._locked_ratio
        if ratio is None:
            ratio = closest_ratio_for_edge(
                width,
                height,
                self._image_width,
                self._image_height,
                self._snap_ratios,
                self._snap_tolerance,
            )
        if ratio is not None:
            target = ratio.value * self._image_height / self._image_width
            if self._drag_kind in {"left", "right"}:
                height = width / target
            else:
                width = height * target

        if self._drag_kind in {"left", "right"}:
            center_y = (crop.top + crop.bottom) / 2
            top = min(max(center_y - height / 2, 0), 1 - height)
            bottom = top + height
            left = min(max(left, 0), 1 - width)
            right = left + width
        else:
            center_x = (crop.left + crop.right) / 2
            left = min(max(center_x - width / 2, 0), 1 - width)
            right = left + width
            top = min(max(top, 0), 1 - height)
            bottom = top + height
        try:
            return CropRect(left, top, right, bottom)
        except ValueError:
            return None

    def _update_hover_cursor(self, point: QPointF, image_rect: QRectF) -> None:
        """Set a resize or move cursor when hovering over crop controls."""
        crop_rect = self._crop_rect(image_rect)
        drag_kind = self._hit_test(point, crop_rect)
        cursors = {
            "move": Qt.CursorShape.SizeAllCursor,
            "left": Qt.CursorShape.SizeHorCursor,
            "right": Qt.CursorShape.SizeHorCursor,
            "top": Qt.CursorShape.SizeVerCursor,
            "bottom": Qt.CursorShape.SizeVerCursor,
            "top-left": Qt.CursorShape.SizeFDiagCursor,
            "bottom-right": Qt.CursorShape.SizeFDiagCursor,
            "top-right": Qt.CursorShape.SizeBDiagCursor,
            "bottom-left": Qt.CursorShape.SizeBDiagCursor,
        }
        self.setCursor(cursors.get(drag_kind, Qt.CursorShape.CrossCursor))

    @staticmethod
    def _is_full_crop(crop: CropRect) -> bool:
        """Return whether the crop still selects the complete image."""
        return crop.left == 0 and crop.top == 0 and crop.right == 1 and crop.bottom == 1

    def _set_edited_crop(self, crop: CropRect) -> None:
        """Update the crop and notify the application that it needs saving."""
        crop = self._constrain_crop(crop)
        if crop == self._crop:
            return
        self._crop = crop
        self._snap_ratio = self._matching_snap_ratio(crop)
        self.update()
        self.crop_changed.emit(crop)

    def _constrain_crop(self, crop: CropRect) -> CropRect:
        """
        Constrain a normalized crop to the rotated source image.

        Returns:
            A crop whose corners remain within the source image polygon.
        """
        image_rect = self._image_rect()
        if self._image.isNull() or not image_rect.isValid():
            return crop
        return constrain_crop_to_rotated_image(
            crop,
            image_rect.width(),
            image_rect.height(),
            self._rotation_angle,
        )

    def _matching_snap_ratio(self, crop: CropRect) -> AspectRatio | None:
        if not self._snap_ratios:
            return None
        return closest_aspect_ratio(
            crop.width,
            crop.height,
            self._image_width,
            self._image_height,
            self._snap_ratios,
            tolerance=self._snap_tolerance,
        )

    def _draw_snap_label(
        self,
        painter: QPainter,
        crop_rect: QRectF,
        ratio: AspectRatio,
    ) -> None:
        label = ratio.label or f"{ratio.width}:{ratio.height}"
        font_metrics = painter.fontMetrics()
        label_rect = QRectF(
            crop_rect.left(),
            max(2, crop_rect.top() - font_metrics.height() - 6),
            font_metrics.horizontalAdvance(label) + 12,
            font_metrics.height() + 4,
        )
        label_rect.moveLeft(min(label_rect.left(), self.width() - label_rect.width() - 2))
        painter.fillRect(label_rect, QColor(20, 45, 35, 225))
        painter.setPen(QPen(SNAPPED_CROP_COLOR, 1))
        painter.drawRect(label_rect)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)


class CroppedPreview(QWidget):
    """Show the crop result and allow panning the crop by dragging the image."""

    crop_changed = Signal(object)

    def __init__(self) -> None:
        """Initialize the result pane with no image loaded."""
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #555555;")
        self._image = QImage()
        self._crop: CropRect | None = None
        self._rotation_angle = 0.0
        self._loading = False
        self._drag_start: QPointF | None = None
        self._drag_crop: CropRect | None = None
        self._drag_target: QRectF | None = None
        self._rotated_cache = QImage()
        self._rotated_cache_key: tuple[int, float] | None = None
        self.setMouseTracking(True)

    def set_image(self, image: QImage) -> None:
        """Set the source preview used to render the cropped result."""
        self._image = image
        self._rotated_cache = QImage()
        self._rotated_cache_key = None
        self.update()

    def set_crop(self, crop: CropRect | None) -> None:
        """Set the normalized crop bounds shown in the result pane."""
        self._crop = crop
        self.update()

    def set_rotation(self, angle_degrees: float) -> None:
        """Set the source rotation used for the cropped result."""
        self._rotation_angle = angle_degrees
        self.update()

    def set_loading(self, loading: bool) -> None:
        """Show an overlay while a new source photo is loading."""
        self._loading = loading
        if loading:
            self._cancel_pan()
        self.update()

    def mousePressEvent(self, event: object) -> None:
        """Begin panning when a left drag starts inside the cropped result."""
        target_rect = self._target_rect()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._loading
            and self._crop is not None
            and target_rect is not None
            and target_rect.contains(event.position())
        ):
            self._drag_start = event.position()
            self._drag_crop = self._crop
            self._drag_target = target_rect
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: object) -> None:
        """Pan the crop in response to dragging the cropped image."""
        if self._drag_start is None or self._drag_crop is None or self._drag_target is None:
            if self._target_rect() is not None:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        delta_x = self._drag_start.x() - event.position().x()
        delta_y = self._drag_start.y() - event.position().y()
        crop_delta_x = delta_x / self._drag_target.width() * self._drag_crop.width
        crop_delta_y = delta_y / self._drag_target.height() * self._drag_crop.height
        left = min(max(self._drag_crop.left + crop_delta_x, 0), 1 - self._drag_crop.width)
        top = min(max(self._drag_crop.top + crop_delta_y, 0), 1 - self._drag_crop.height)
        crop = CropRect(
            left,
            top,
            left + self._drag_crop.width,
            top + self._drag_crop.height,
        )
        if crop != self._crop:
            self._crop = crop
            self.update()
            self.crop_changed.emit(crop)
        event.accept()

    def mouseReleaseEvent(self, event: object) -> None:
        """Finish panning and restore the hover cursor."""
        if event.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._cancel_pan()
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: object) -> None:
        """Clear the pan cursor when the pointer leaves the result pane."""
        if self._drag_start is None:
            self.unsetCursor()
        super().leaveEvent(event)

    def _cancel_pan(self) -> None:
        self._drag_start = None
        self._drag_crop = None
        self._drag_target = None
        self.unsetCursor()

    def paintEvent(self, event: object) -> None:
        """Paint the cropped image, fitting it inside the result pane."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), CANVAS_COLOR)
        if self._image.isNull():
            message = "Loading preview..." if self._loading else "Cropped preview"
            painter.setPen(QColor("#a7acb2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, message)
            return
        if self._crop is None:
            painter.setPen(QColor("#a7acb2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No crop selected")
            return

        rotated_image = self._get_rotated_image()
        source_rect = QRectF(
            self._crop.left * rotated_image.width(),
            self._crop.top * rotated_image.height(),
            self._crop.width * rotated_image.width(),
            self._crop.height * rotated_image.height(),
        )
        target_rect = self._fit_rect(source_rect.width(), source_rect.height())
        painter.drawImage(target_rect, rotated_image, source_rect)
        if self._loading:
            painter.fillRect(target_rect, QColor(0, 0, 0, 100))

    def _get_rotated_image(self) -> QImage:
        cache_key = (self._image.cacheKey(), self._rotation_angle)
        if self._rotated_cache_key != cache_key:
            self._rotated_cache = self._image.transformed(
                QTransform().rotate(self._rotation_angle),
                Qt.TransformationMode.SmoothTransformation,
            )
            self._rotated_cache_key = cache_key
        return self._rotated_cache

    def _target_rect(self) -> QRectF | None:
        if self._image.isNull() or self._crop is None:
            return None
        rotated_image = self._get_rotated_image()
        return self._fit_rect(
            self._crop.width * rotated_image.width(),
            self._crop.height * rotated_image.height(),
        )

    def _fit_rect(self, width: float, height: float) -> QRectF:
        scale = min(self.width() / width, self.height() / height)
        fitted_width = width * scale
        fitted_height = height * scale
        return QRectF(
            (self.width() - fitted_width) / 2,
            (self.height() - fitted_height) / 2,
            fitted_width,
            fitted_height,
        )


def closest_ratio_for_edge(
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    ratios: tuple[AspectRatio, ...],
    tolerance: float,
) -> AspectRatio | None:
    """
    Resolve a snap ratio for a crop edge gesture.

    Returns:
        The closest ratio within tolerance, or `None` when no ratio is close.
    """
    return closest_aspect_ratio(
        width,
        height,
        image_width,
        image_height,
        ratios,
        tolerance=tolerance,
    )
