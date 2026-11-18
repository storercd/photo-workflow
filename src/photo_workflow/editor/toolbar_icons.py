"""Vector glyphs and styling for the two-row editor toolbar."""

from __future__ import annotations

from collections.abc import Callable
from math import cos, radians, sin

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap

GLYPH_COLOR = "#d8d8dc"
ACCENT_COLOR = "#3fd8c8"
RED_COLOR = "#e0463c"
BUTTON_SIZE = 28
ICON_SIZE = 20
_CANVAS = 24
_SCALE = 4

TOOLBAR_STYLE = """
QWidget#toolbarRoot { background: #2b2b2e; }
QWidget#toolbarRoot QToolButton {
    background: #3a3a3f; border: 1px solid #4a4a50; border-radius: 5px;
    color: #d8d8dc;
}
QWidget#toolbarRoot QToolButton:hover { background: #46464c; }
QWidget#toolbarRoot QToolButton:checked { background: #4a4a50; border-color: #3fd8c8; }
QWidget#toolbarRoot QToolButton:disabled { background: #303033; border-color: #3a3a3f; }
QWidget#toolbarRoot QLabel[role="group"] {
    color: #a8a8ae; font-size: 11px; font-weight: bold;
}
QWidget#toolbarRoot QLabel[role="counter"] { color: #d8d8dc; padding: 0 4px; }
QWidget#toolbarRoot QLabel[role="filename"] {
    color: #bdbdc4; font-family: Menlo, Consolas, monospace;
}
QWidget#toolbarRoot QFrame[role="divider"] { background: #4a4a50; max-width: 1px; }
QWidget#toolbarRoot QLineEdit[role="value"] {
    background: #1e1e21; border: 1px solid #4a4a50; border-radius: 5px; color: #d8d8dc;
    padding: 0 6px; min-height: 26px; max-height: 26px;
}
QWidget#toolbarRoot QSlider::groove:horizontal { height: 3px; background: #4a4a50; }
QWidget#toolbarRoot QSlider::handle:horizontal {
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: #d8d8dc;
}
QWidget#toolbarRoot QComboBox {
    background: #1e1e21; border: 1px solid #4a4a50; border-radius: 5px; color: #d8d8dc;
    min-height: 26px; max-height: 26px; padding: 0 6px;
}
"""


def _pen(color: str, width: float = 1.4) -> QPen:
    pen = QPen(QColor(color), width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def _icon(draw: Callable[[QPainter], None]) -> QIcon:
    """
    Render `draw` on a 24x24 canvas into an icon.

    Returns:
        An icon that stays crisp when scaled to the toolbar's icon size.
    """
    pixmap = QPixmap(_CANVAS * _SCALE, _CANVAS * _SCALE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHints(
        QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
    )
    painter.scale(_SCALE, _SCALE)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    draw(painter)
    painter.end()
    return QIcon(pixmap)


def _arrow_head(painter: QPainter, tip: QPointF, angle_deg: float, color: str) -> None:
    """Draw a small filled arrowhead pointing along `angle_deg` (screen degrees)."""
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    path = QPainterPath()
    size = 4.2
    for offset, length in ((0, 0.0), (150, size), (-150, size)):
        heading = radians(angle_deg + offset)
        point = (
            tip
            if length == 0.0
            else QPointF(tip.x() + length * cos(heading), tip.y() + length * sin(heading))
        )
        if length == 0.0:
            path.moveTo(point)
        else:
            path.lineTo(point)
    path.closeSubpath()
    painter.drawPath(path)
    painter.restore()


def angle_reset_icon() -> QIcon:
    """
    Circular arrow around a literal 0° (reset the angle).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(ACCENT_COLOR, 1.8))
        rect = QRectF(3.5, 3.5, 17, 17)
        painter.drawArc(rect, 70 * 16, 290 * 16)
        _arrow_head(painter, QPointF(18.9, 18.2), -45, ACCENT_COLOR)
        font = QFont()
        font.setPixelSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(_pen(GLYPH_COLOR))
        painter.drawText(QRectF(3.5, 3.5, 17, 17), Qt.AlignmentFlag.AlignCenter, "0°")

    return _icon(draw)


def auto_level_icon() -> QIcon:
    """
    Bullseye bubble level (outer ring, inner ring, centered bubble, four ticks).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(ACCENT_COLOR, 1.5))
        painter.drawEllipse(QPointF(12, 12), 7.5, 7.5)
        painter.drawEllipse(QPointF(12, 12), 4.2, 4.2)
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            painter.drawLine(
                QPointF(12 + dx * 6.2, 12 + dy * 6.2), QPointF(12 + dx * 10.5, 12 + dy * 10.5)
            )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(ACCENT_COLOR))
        painter.drawEllipse(QPointF(12, 12), 1.9, 1.9)

    return _icon(draw)


def guides_icon() -> QIcon:
    """
    Eye with a red slash (show or hide the angle guides).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR))
        path = QPainterPath()
        path.moveTo(2.5, 12)
        path.quadTo(12, 2.5, 21.5, 12)
        path.quadTo(12, 21.5, 2.5, 12)
        painter.drawPath(path)
        painter.drawEllipse(QPointF(12, 12), 3.2, 3.2)
        painter.setPen(_pen(RED_COLOR, 1.8))
        painter.drawLine(QPointF(4.5, 19.5), QPointF(19.5, 4.5))

    return _icon(draw)


def _sparkle(painter: QPainter, center: QPointF, size: float) -> None:
    path = QPainterPath()
    path.moveTo(center.x(), center.y() - size)
    path.quadTo(center.x(), center.y(), center.x() + size, center.y())
    path.quadTo(center.x(), center.y(), center.x(), center.y() + size)
    path.quadTo(center.x(), center.y(), center.x() - size, center.y())
    path.quadTo(center.x(), center.y(), center.x(), center.y() - size)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(GLYPH_COLOR))
    painter.drawPath(path)


def wand_icon() -> QIcon:
    """
    White wand with sparkles (show lighting edits or the original).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR, 3.2))
        painter.drawLine(QPointF(6.5, 17.5), QPointF(16.5, 7.5))
        _sparkle(painter, QPointF(6.5, 6.5), 3.2)
        _sparkle(painter, QPointF(18.5, 18.5), 2.8)

    return _icon(draw)


def thermometer_icon() -> QIcon:
    """
    Old-school thermometer (vertical tube, round bulb, two ticks).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR, 1.5))
        painter.drawRoundedRect(QRectF(9.5, 3, 5, 12.5), 2.5, 2.5)
        painter.drawLine(QPointF(17, 6), QPointF(20, 6))
        painter.drawLine(QPointF(17, 9.5), QPointF(20, 9.5))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(RED_COLOR))
        painter.drawEllipse(QPointF(12, 17.5), 3.6, 3.6)
        painter.drawRoundedRect(QRectF(11, 8, 2, 9), 1, 1)

    return _icon(draw)


def ball_icon() -> QIcon:
    """
    Volleyball (circle with curved panel seams).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen("#f4f2ec", 1.5))
        painter.drawEllipse(QPointF(12, 12), 9, 9)
        painter.setClipPath(_circle_path(QPointF(12, 12), 9))
        seam = QPainterPath()
        seam.moveTo(12, 3)
        seam.cubicTo(6, 8, 6, 14, 14, 21)
        painter.drawPath(seam)
        seam = QPainterPath()
        seam.moveTo(3.5, 9)
        seam.cubicTo(11, 8, 17, 11, 20.5, 16)
        painter.drawPath(seam)
        seam = QPainterPath()
        seam.moveTo(20.5, 8)
        seam.cubicTo(16, 10, 13, 13, 12.5, 15)
        painter.drawPath(seam)

    return _icon(draw)


def _circle_path(center: QPointF, radius: float) -> QPainterPath:
    path = QPainterPath()
    path.addEllipse(center, radius, radius)
    return path


def lock_icon(locked: bool) -> QIcon:
    """
    Ultra-thin line padlock; the shackle is open when unlocked.

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR, 1.1))
        painter.drawRoundedRect(QRectF(5.5, 10.5, 13, 9.5), 2, 2)
        shackle = QPainterPath()
        if locked:
            shackle.moveTo(8.5, 10.5)
            shackle.lineTo(8.5, 8)
            shackle.arcTo(QRectF(8.5, 3.5, 7, 9), 180, -180)
            shackle.lineTo(15.5, 10.5)
        else:
            shackle.moveTo(8.5, 10.5)
            shackle.lineTo(8.5, 8)
            shackle.arcTo(QRectF(8.5, 3.5, 7, 9), 180, -180)
            shackle.lineTo(15.5, 7)
        painter.drawPath(shackle)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(GLYPH_COLOR))
        painter.drawEllipse(QPointF(12, 15.2), 1.1, 1.1)

    return _icon(draw)


def snap_icon(enabled: bool) -> QIcon:
    """
    Four corner brackets closing in on a center dot; slashed in red when snapping is off.

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        color = ACCENT_COLOR if enabled else GLYPH_COLOR
        painter.setPen(_pen(color, 1.8))
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            corner = QPointF(12 + sx * 8, 12 + sy * 8)
            painter.drawLine(corner, QPointF(corner.x() - sx * 4, corner.y()))
            painter.drawLine(corner, QPointF(corner.x(), corner.y() - sy * 4))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(GLYPH_COLOR))
        painter.drawEllipse(QPointF(12, 12), 1.5, 1.5)
        if not enabled:
            painter.setPen(_pen(RED_COLOR, 1.8))
            painter.drawLine(QPointF(4, 20), QPointF(20, 4))

    return _icon(draw)


def preview_icon() -> QIcon:
    """
    Framed photo (frame, sun dot, mountain line) for the finished-picture preview.

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR, 1.5))
        painter.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2, 2)
        mountains = QPainterPath()
        mountains.moveTo(4.5, 17)
        mountains.lineTo(9.5, 11.5)
        mountains.lineTo(13, 15)
        mountains.lineTo(15.5, 12.5)
        mountains.lineTo(19.5, 17)
        painter.drawPath(mountains)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(GLYPH_COLOR))
        painter.drawEllipse(QPointF(8, 8.5), 1.5, 1.5)

    return _icon(draw)


def refresh_icon() -> QIcon:
    """
    Counterclockwise circular arrow (revert the photo).

    Returns:
        The icon.
    """

    def draw(painter: QPainter) -> None:
        painter.setPen(_pen(GLYPH_COLOR, 1.8))
        painter.drawArc(QRectF(4, 4, 16, 16), 110 * 16, -290 * 16)
        _arrow_head(painter, QPointF(17, 18.7), 30, GLYPH_COLOR)

    return _icon(draw)
