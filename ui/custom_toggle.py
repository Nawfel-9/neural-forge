import math
from PyQt6.QtCore import Qt, QPropertyAnimation, pyqtProperty, pyqtSignal, QRectF, QEasingCurve, QPointF
from PyQt6.QtGui import (QPainter, QColor, QPainterPath, QRadialGradient, 
                         QLinearGradient, QConicalGradient, QPen, QBrush)
from PyQt6.QtWidgets import QWidget

class PremiumToggle(QWidget):
    """
    A pixel-perfect recreation of the premium futuristic toggle switch.
    Features:
    - Matte dark navy background with cyan neon border glow
    - Recessed inner track with inner shadow
    - Brushed silver metal knob using conical gradients
    - Custom drawn engraved Sun and glowing Moon + Signal icons
    """
    toggled = pyqtSignal(bool)

    def __init__(self, parent=None, is_dark_mode=True):
        super().__init__(parent)
        self.setFixedSize(120, 40)
        self.is_dark_mode = is_dark_mode
        self._checked = is_dark_mode  # True = Right (Dark), False = Left (Light)
        self._position = 1.0 if is_dark_mode else 0.0
        
        self.animation = QPropertyAnimation(self, b"position")
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.setDuration(350)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Toggle Theme")

    @pyqtProperty(float)
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._checked = not self._checked
            self.animation.stop()
            self.animation.setStartValue(self._position)
            self.animation.setEndValue(1.0 if self._checked else 0.0)
            self.animation.start()
            self.is_dark_mode = self._checked
            self.toggled.emit(self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()
        
        # ─── 1. Outer Pill ───
        margin = 3
        rect = QRectF(margin, margin, width - margin*2, height - margin*2)
        radius = rect.height() / 2

        # Background
        if self.is_dark_mode:
            bg_color = QColor("#1e2028")
            accent = QColor(14, 165, 233) # DARK_ACCENT
            track_color = QColor("#15171e")
            icon_inactive = QColor(68, 71, 85)
        else:
            bg_color = QColor("#F1F5F9")
            accent = QColor(2, 132, 199) # LIGHT_ACCENT
            track_color = QColor("#E2E8F0")
            icon_inactive = QColor(148, 163, 184)

        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, radius, radius)

        # Theme Accent Border Glow
        glow_alpha = 255
        for i in range(1, 5):
            pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), 60 // i))
            pen.setWidth(i * 2 + 1)
            painter.setPen(pen)
            painter.drawRoundedRect(rect, radius, radius)
        
        # Core solid line
        pen = QPen(QColor(accent.red(), accent.green(), accent.blue(), 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, radius, radius)

        # ─── 2. Inner Recessed Track ───
        track_w = 56
        track_h = 16
        track_x = (width - track_w) / 2
        track_y = (height - track_h) / 2
        track_rect = QRectF(track_x, track_y, track_w, track_h)
        track_radius = track_h / 2

        # Track background
        painter.setBrush(QBrush(track_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, track_radius, track_radius)

        # Inner shadow for depth
        inner_shadow_path = QPainterPath()
        inner_shadow_path.addRoundedRect(track_rect, track_radius, track_radius)
        painter.setClipPath(inner_shadow_path)
        shadow_pen = QPen(QColor(0, 0, 0, 200))
        shadow_pen.setWidth(4)
        painter.setPen(shadow_pen)
        painter.drawRoundedRect(track_rect.adjusted(-1, -1, 1, 1), track_radius, track_radius)
        painter.setClipping(False)

        # ─── 3. Icons ───
        # Sun Icon (Left)
        sun_center = QPointF(18, height / 2)
        # Active when position -> 0.0 (Left)
        sun_active = 1.0 - self._position
        sun_r = int(icon_inactive.red() + (accent.red() - icon_inactive.red()) * sun_active)
        sun_g = int(icon_inactive.green() + (accent.green() - icon_inactive.green()) * sun_active)
        sun_b = int(icon_inactive.blue() + (accent.blue() - icon_inactive.blue()) * sun_active)
        sun_color = QColor(sun_r, sun_g, sun_b)
        
        painter.setPen(QPen(sun_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(sun_center, 4, 4)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            p1 = QPointF(sun_center.x() + math.cos(rad) * 6, sun_center.y() + math.sin(rad) * 6)
            p2 = QPointF(sun_center.x() + math.cos(rad) * 9, sun_center.y() + math.sin(rad) * 9)
            painter.drawLine(p1, p2)

        # Moon Icon (Right)
        moon_center = QPointF(102, height / 2)
        # Active when position -> 1.0 (Right)
        moon_active = self._position
        moon_r = int(icon_inactive.red() + (accent.red() - icon_inactive.red()) * moon_active)
        moon_g = int(icon_inactive.green() + (accent.green() - icon_inactive.green()) * moon_active)
        moon_b = int(icon_inactive.blue() + (accent.blue() - icon_inactive.blue()) * moon_active)
        moon_color = QColor(moon_r, moon_g, moon_b)

        painter.setPen(QPen(moon_color, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        
        # Draw Moon Crescent & Signal Bars
        painter.save()
        painter.translate(moon_center.x() - 5, moon_center.y())
        painter.rotate(15) # Slight rotation for dynamic look
        
        # Mathematically perfect crescent using path subtraction
        main_circle = QPainterPath()
        main_circle.addEllipse(QRectF(-5, -5, 10, 10))
        
        cut_circle = QPainterPath()
        # Shift the cutting circle up and right
        cut_circle.addEllipse(QRectF(-1, -7, 10, 10)) 
        
        moon_path = main_circle.subtracted(cut_circle)
        painter.drawPath(moon_path)

        # Draw Signal Bars
        signal_cx = 0
        signal_cy = -2
        for r in [4, 7, 10]:
            painter.drawArc(QRectF(signal_cx - r, signal_cy - r, r*2, r*2), -10 * 16, 75 * 16)
            
        painter.restore()

        # Add Glow to active icons
        if sun_active > 0.5:
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), int(100 * sun_active)), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawEllipse(sun_center, 4, 4)
        if moon_active > 0.5:
            painter.save()
            painter.translate(moon_center.x() - 5, moon_center.y())
            painter.rotate(15)
            painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), int(100 * moon_active)), 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawPath(moon_path)
            painter.restore()

        # ─── 4. Knob ───
        knob_radius = 11
        # Sliding range: left edge of track to right edge of track
        # track_x to track_x + track_w
        # Knob center moves from track_x + knob_radius to track_x + track_w - knob_radius
        min_kx = track_x + knob_radius - 2
        max_kx = track_x + track_w - knob_radius + 2
        kx = min_kx + (max_kx - min_kx) * self._position
        ky = height / 2

        knob_rect = QRectF(kx - knob_radius, ky - knob_radius, knob_radius*2, knob_radius*2)

        # Knob Drop Shadow
        painter.setPen(Qt.PenStyle.NoPen)
        shadow_grad = QRadialGradient(kx, ky + 4, knob_radius + 4)
        shadow_grad.setColorAt(0.0, QColor(0, 0, 0, 180))
        shadow_grad.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.setBrush(QBrush(shadow_grad))
        painter.drawEllipse(QRectF(kx - knob_radius - 2, ky - knob_radius + 2, (knob_radius+2)*2, (knob_radius+4)*2))

        # Brushed Metal Material (Conical Gradient)
        # Shift angle slightly during slide for rotating effect
        rotation = self._position * 90
        metal_grad = QConicalGradient(kx, ky, rotation)
        
        # Alternating light and dark bands for brushed metal
        stops = [
            (0.00, QColor(240, 240, 240)),
            (0.15, QColor(140, 140, 140)),
            (0.30, QColor(255, 255, 255)),
            (0.45, QColor(120, 120, 120)),
            (0.60, QColor(230, 230, 230)),
            (0.75, QColor(150, 150, 150)),
            (0.90, QColor(255, 255, 255)),
            (1.00, QColor(240, 240, 240)),
        ]
        for pos, color in stops:
            metal_grad.setColorAt(pos, color)
            
        painter.setBrush(QBrush(metal_grad))
        
        # Knob Outer Bevel/Border
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawEllipse(knob_rect)

        # Knob Inner Highlight (Top light)
        highlight = QLinearGradient(knob_rect.topLeft(), knob_rect.bottomLeft())
        highlight.setColorAt(0.0, QColor(255, 255, 255, 150))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(QBrush(highlight))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(knob_rect.adjusted(1, 1, -1, -1))
