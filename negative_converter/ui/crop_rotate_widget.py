# Crop and rotate tool widget
"""
Widget for crop and rotate operations on images.
"""

from typing import Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QDoubleSpinBox, QGroupBox, QFormLayout,
    QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QPoint
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush

from ..utils.geometry import (
    CropRect, AspectRatio, ASPECT_RATIOS,
    crop_image, rotate_image, rotate_90, flip_image,
    straighten_image, detect_horizon_angle, auto_crop_borders
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class CropOverlay(QWidget):
    """
    Transparent overlay widget for drawing crop rectangle on image.
    """
    crop_changed = pyqtSignal(object)  # Emits CropRect
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        
        self._crop_rect: Optional[QRect] = None
        self._aspect_ratio: Optional[AspectRatio] = None
        self._image_rect: Optional[QRect] = None
        self._dragging = False
        self._drag_handle: Optional[str] = None
        self._drag_start: Optional[QPoint] = None
        self._drag_start_rect: Optional[QRect] = None
        
        # Handle size for resize corners
        self._handle_size = 10
    
    def set_image_rect(self, rect: QRect) -> None:
        """Set the image bounds for constraining crop."""
        self._image_rect = rect
        if self._crop_rect is None and rect is not None:
            # Initialize crop to full image
            self._crop_rect = QRect(rect)
    
    def set_aspect_ratio(self, aspect: Optional[AspectRatio]) -> None:
        """Set aspect ratio constraint."""
        self._aspect_ratio = aspect
        if self._crop_rect and aspect:
            self._apply_aspect_ratio()
            self.update()
    
    def get_crop_rect(self) -> Optional[CropRect]:
        """Get the current crop rectangle in image coordinates."""
        if self._crop_rect is None or self._image_rect is None:
            return None
        
        # Convert from widget coords to image coords
        # This is simplified - real implementation needs zoom/pan handling
        return CropRect(
            self._crop_rect.x() - self._image_rect.x(),
            self._crop_rect.y() - self._image_rect.y(),
            self._crop_rect.width(),
            self._crop_rect.height()
        )
    
    def set_crop_rect(self, rect: CropRect) -> None:
        """Set crop rectangle from image coordinates."""
        if self._image_rect is None:
            return
        
        self._crop_rect = QRect(
            self._image_rect.x() + rect.x,
            self._image_rect.y() + rect.y,
            rect.width,
            rect.height
        )
        self.update()
    
    def reset_crop(self) -> None:
        """Reset crop to full image."""
        if self._image_rect:
            self._crop_rect = QRect(self._image_rect)
            self.update()
            self._emit_crop_changed()
    
    def _apply_aspect_ratio(self) -> None:
        """Apply aspect ratio constraint to current crop."""
        if self._crop_rect is None or self._aspect_ratio is None:
            return
        
        rect = self._crop_rect
        target_ratio = self._aspect_ratio.ratio
        current_ratio = rect.width() / rect.height() if rect.height() > 0 else 1.0
        
        if abs(current_ratio - target_ratio) < 0.001:
            return
        
        # Adjust dimensions while keeping center
        center = rect.center()
        if current_ratio > target_ratio:
            new_width = int(rect.height() * target_ratio)
            new_height = rect.height()
        else:
            new_width = rect.width()
            new_height = int(rect.width() / target_ratio)
        
        self._crop_rect = QRect(
            center.x() - new_width // 2,
            center.y() - new_height // 2,
            new_width,
            new_height
        )
        self._constrain_to_image()
    
    def _constrain_to_image(self) -> None:
        """Constrain crop rect to image bounds."""
        if self._crop_rect is None or self._image_rect is None:
            return
        
        rect = self._crop_rect
        img = self._image_rect
        
        # Ensure within bounds
        if rect.left() < img.left():
            rect.moveLeft(img.left())
        if rect.top() < img.top():
            rect.moveTop(img.top())
        if rect.right() > img.right():
            rect.moveRight(img.right())
        if rect.bottom() > img.bottom():
            rect.moveBottom(img.bottom())
        
        self._crop_rect = rect
    
    def _get_handle_at(self, pos: QPoint) -> Optional[str]:
        """Get the resize handle at the given position."""
        if self._crop_rect is None:
            return None
        
        rect = self._crop_rect
        hs = self._handle_size
        
        # Check corners
        corners = {
            "top-left": QRect(rect.left() - hs, rect.top() - hs, hs * 2, hs * 2),
            "top-right": QRect(rect.right() - hs, rect.top() - hs, hs * 2, hs * 2),
            "bottom-left": QRect(rect.left() - hs, rect.bottom() - hs, hs * 2, hs * 2),
            "bottom-right": QRect(rect.right() - hs, rect.bottom() - hs, hs * 2, hs * 2),
        }
        
        for name, handle_rect in corners.items():
            if handle_rect.contains(pos):
                return name
        
        # Check edges
        edges = {
            "top": QRect(rect.left() + hs, rect.top() - hs, rect.width() - hs * 2, hs * 2),
            "bottom": QRect(rect.left() + hs, rect.bottom() - hs, rect.width() - hs * 2, hs * 2),
            "left": QRect(rect.left() - hs, rect.top() + hs, hs * 2, rect.height() - hs * 2),
            "right": QRect(rect.right() - hs, rect.top() + hs, hs * 2, rect.height() - hs * 2),
        }
        
        for name, handle_rect in edges.items():
            if handle_rect.contains(pos):
                return name
        
        # Check if inside rect (for moving)
        if rect.contains(pos):
            return "move"
        
        return None
    
    def _emit_crop_changed(self) -> None:
        """Emit crop changed signal."""
        crop = self.get_crop_rect()
        if crop:
            self.crop_changed.emit(crop)
    
    def paintEvent(self, event):
        """Draw the crop overlay."""
        if self._crop_rect is None or self._image_rect is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw darkened area outside crop
        dark_brush = QBrush(QColor(0, 0, 0, 128))
        
        # Top
        painter.fillRect(QRect(
            self._image_rect.left(), self._image_rect.top(),
            self._image_rect.width(), self._crop_rect.top() - self._image_rect.top()
        ), dark_brush)
        
        # Bottom
        painter.fillRect(QRect(
            self._image_rect.left(), self._crop_rect.bottom(),
            self._image_rect.width(), self._image_rect.bottom() - self._crop_rect.bottom()
        ), dark_brush)
        
        # Left
        painter.fillRect(QRect(
            self._image_rect.left(), self._crop_rect.top(),
            self._crop_rect.left() - self._image_rect.left(), self._crop_rect.height()
        ), dark_brush)
        
        # Right
        painter.fillRect(QRect(
            self._crop_rect.right(), self._crop_rect.top(),
            self._image_rect.right() - self._crop_rect.right(), self._crop_rect.height()
        ), dark_brush)
        
        # Draw crop rectangle border
        pen = QPen(QColor(255, 255, 255), 2)
        painter.setPen(pen)
        painter.drawRect(self._crop_rect)
        
        # Draw rule of thirds grid
        pen.setWidth(1)
        pen.setColor(QColor(255, 255, 255, 100))
        painter.setPen(pen)
        
        w = self._crop_rect.width()
        h = self._crop_rect.height()
        x = self._crop_rect.x()
        y = self._crop_rect.y()
        
        # Vertical lines
        painter.drawLine(x + w // 3, y, x + w // 3, y + h)
        painter.drawLine(x + 2 * w // 3, y, x + 2 * w // 3, y + h)
        
        # Horizontal lines
        painter.drawLine(x, y + h // 3, x + w, y + h // 3)
        painter.drawLine(x, y + 2 * h // 3, x + w, y + 2 * h // 3)
        
        # Draw resize handles
        handle_brush = QBrush(QColor(255, 255, 255))
        painter.setBrush(handle_brush)
        pen.setColor(QColor(0, 0, 0))
        painter.setPen(pen)
        
        hs = self._handle_size // 2
        corners = [
            self._crop_rect.topLeft(),
            self._crop_rect.topRight(),
            self._crop_rect.bottomLeft(),
            self._crop_rect.bottomRight(),
        ]
        
        for corner in corners:
            painter.drawRect(corner.x() - hs, corner.y() - hs, hs * 2, hs * 2)
        
        painter.end()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._get_handle_at(event.pos())
            if handle:
                self._dragging = True
                self._drag_handle = handle
                self._drag_start = event.pos()
                self._drag_start_rect = QRect(self._crop_rect) if self._crop_rect else None
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start and self._drag_start_rect:
            delta = event.pos() - self._drag_start
            
            if self._drag_handle == "move":
                new_rect = QRect(self._drag_start_rect)
                new_rect.translate(delta)
                self._crop_rect = new_rect
            else:
                self._resize_crop(delta)
            
            self._constrain_to_image()
            if self._aspect_ratio:
                self._apply_aspect_ratio()
            
            self.update()
            event.accept()
            return
        
        # Update cursor based on handle
        handle = self._get_handle_at(event.pos())
        if handle in ("top-left", "bottom-right"):
            self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        elif handle in ("top-right", "bottom-left"):
            self.setCursor(Qt.CursorShape.SizeBDiagCursor)
        elif handle in ("top", "bottom"):
            self.setCursor(Qt.CursorShape.SizeVerCursor)
        elif handle in ("left", "right"):
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif handle == "move":
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._drag_handle = None
            self._drag_start = None
            self._drag_start_rect = None
            self._emit_crop_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)
    
    def _resize_crop(self, delta: QPoint) -> None:
        """Resize crop rect based on drag handle and delta."""
        if self._crop_rect is None or self._drag_start_rect is None:
            return
        
        rect = QRect(self._drag_start_rect)
        handle = self._drag_handle
        
        if "left" in handle:
            rect.setLeft(rect.left() + delta.x())
        if "right" in handle:
            rect.setRight(rect.right() + delta.x())
        if "top" in handle:
            rect.setTop(rect.top() + delta.y())
        if "bottom" in handle:
            rect.setBottom(rect.bottom() + delta.y())
        
        # Ensure minimum size
        if rect.width() < 20:
            rect.setWidth(20)
        if rect.height() < 20:
            rect.setHeight(20)
        
        self._crop_rect = rect


class CropRotatePanel(QWidget):
    """
    Panel with crop and rotate controls.
    """
    crop_requested = pyqtSignal(object)  # Emits CropRect
    rotate_requested = pyqtSignal(float)  # Emits angle in degrees
    rotate_90_requested = pyqtSignal(bool)  # Emits clockwise (True/False)
    flip_requested = pyqtSignal(bool)  # Emits horizontal (True/False)
    straighten_requested = pyqtSignal(float)  # Emits angle
    auto_crop_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)
        
        # Aspect ratio selection
        aspect_group = QGroupBox("Aspect Ratio")
        aspect_layout = QHBoxLayout(aspect_group)
        
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECT_RATIOS.keys()))
        self.aspect_combo.setCurrentText("free")
        self.aspect_combo.currentTextChanged.connect(self._on_aspect_changed)
        aspect_layout.addWidget(self.aspect_combo)
        
        self.swap_aspect_btn = QPushButton("Swap")
        self.swap_aspect_btn.setToolTip("Swap width and height")
        self.swap_aspect_btn.clicked.connect(self._on_swap_aspect)
        aspect_layout.addWidget(self.swap_aspect_btn)
        
        layout.addWidget(aspect_group)
        
        # Rotate controls
        rotate_group = QGroupBox("Rotate")
        rotate_layout = QVBoxLayout(rotate_group)
        
        # Quick rotate buttons
        quick_rotate_layout = QHBoxLayout()
        
        self.rotate_ccw_btn = QPushButton("↺ 90°")
        self.rotate_ccw_btn.setToolTip("Rotate 90° counter-clockwise")
        self.rotate_ccw_btn.clicked.connect(lambda: self.rotate_90_requested.emit(False))
        quick_rotate_layout.addWidget(self.rotate_ccw_btn)
        
        self.rotate_cw_btn = QPushButton("↻ 90°")
        self.rotate_cw_btn.setToolTip("Rotate 90° clockwise")
        self.rotate_cw_btn.clicked.connect(lambda: self.rotate_90_requested.emit(True))
        quick_rotate_layout.addWidget(self.rotate_cw_btn)
        
        rotate_layout.addLayout(quick_rotate_layout)
        
        # Fine rotation slider
        fine_rotate_layout = QHBoxLayout()
        fine_rotate_layout.addWidget(QLabel("Angle:"))
        
        self.rotate_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotate_slider.setRange(-450, 450)  # -45.0 to 45.0 degrees * 10
        self.rotate_slider.setValue(0)
        self.rotate_slider.valueChanged.connect(self._on_rotate_slider_changed)
        fine_rotate_layout.addWidget(self.rotate_slider)
        
        self.rotate_spin = QDoubleSpinBox()
        self.rotate_spin.setRange(-45.0, 45.0)
        self.rotate_spin.setSingleStep(0.1)
        self.rotate_spin.setDecimals(1)
        self.rotate_spin.setSuffix("°")
        self.rotate_spin.valueChanged.connect(self._on_rotate_spin_changed)
        fine_rotate_layout.addWidget(self.rotate_spin)
        
        rotate_layout.addLayout(fine_rotate_layout)
        
        # Auto straighten
        self.auto_straighten_btn = QPushButton("Auto Straighten")
        self.auto_straighten_btn.setToolTip("Detect and correct horizon angle")
        self.auto_straighten_btn.clicked.connect(self._on_auto_straighten)
        rotate_layout.addWidget(self.auto_straighten_btn)
        
        layout.addWidget(rotate_group)
        
        # Flip controls
        flip_group = QGroupBox("Flip")
        flip_layout = QHBoxLayout(flip_group)
        
        self.flip_h_btn = QPushButton("↔ Horizontal")
        self.flip_h_btn.clicked.connect(lambda: self.flip_requested.emit(True))
        flip_layout.addWidget(self.flip_h_btn)
        
        self.flip_v_btn = QPushButton("↕ Vertical")
        self.flip_v_btn.clicked.connect(lambda: self.flip_requested.emit(False))
        flip_layout.addWidget(self.flip_v_btn)
        
        layout.addWidget(flip_group)
        
        # Auto crop
        self.auto_crop_btn = QPushButton("Auto Crop Borders")
        self.auto_crop_btn.setToolTip("Automatically detect and crop dark borders")
        self.auto_crop_btn.clicked.connect(self.auto_crop_requested.emit)
        layout.addWidget(self.auto_crop_btn)
        
        layout.addStretch()
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_requested.emit)
        action_layout.addWidget(self.reset_btn)
        
        action_layout.addStretch()
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_requested.emit)
        action_layout.addWidget(self.cancel_btn)
        
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        action_layout.addWidget(self.apply_btn)
        
        layout.addLayout(action_layout)
    
    def _on_aspect_changed(self, text: str):
        """Handle aspect ratio selection change."""
        # This would update the crop overlay
        pass
    
    def _on_swap_aspect(self):
        """Swap aspect ratio width/height."""
        current = self.aspect_combo.currentText()
        if current in ASPECT_RATIOS and ASPECT_RATIOS[current]:
            aspect = ASPECT_RATIOS[current]
            # Find or create swapped ratio
            swapped_name = f"{aspect.height}:{aspect.width}"
            if swapped_name in ASPECT_RATIOS:
                self.aspect_combo.setCurrentText(swapped_name)
    
    def _on_rotate_slider_changed(self, value: int):
        """Handle rotation slider change."""
        angle = value / 10.0
        self.rotate_spin.blockSignals(True)
        self.rotate_spin.setValue(angle)
        self.rotate_spin.blockSignals(False)
        self.straighten_requested.emit(angle)
    
    def _on_rotate_spin_changed(self, value: float):
        """Handle rotation spinbox change."""
        self.rotate_slider.blockSignals(True)
        self.rotate_slider.setValue(int(value * 10))
        self.rotate_slider.blockSignals(False)
        self.straighten_requested.emit(value)
    
    def _on_auto_straighten(self):
        """Request auto straighten detection."""
        # This would be connected to detect_horizon_angle
        pass
    
    def set_rotation(self, angle: float):
        """Set the rotation controls to a specific angle."""
        self.rotate_slider.blockSignals(True)
        self.rotate_spin.blockSignals(True)
        self.rotate_slider.setValue(int(angle * 10))
        self.rotate_spin.setValue(angle)
        self.rotate_slider.blockSignals(False)
        self.rotate_spin.blockSignals(False)
    
    def reset_controls(self):
        """Reset all controls to default values."""
        self.aspect_combo.setCurrentText("free")
        self.set_rotation(0)


class CropRotateDialog(QWidget):
    """
    Dialog for crop and rotate operations.
    """
    
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop & Rotate")
        self.setWindowFlags(Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(900, 700)
        
        self._original_image = image.copy()
        self._current_image = image.copy()
        self._result = None
        self._rotation_angle = 0.0
        self._accepted = False
        
        self._setup_ui()
        self._update_preview()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Preview area
        preview_layout = QVBoxLayout()
        
        self._preview_label = QLabel()
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setMinimumSize(400, 400)
        self._preview_label.setStyleSheet("background-color: #333;")
        preview_layout.addWidget(self._preview_label, stretch=1)
        
        layout.addLayout(preview_layout, stretch=3)
        
        # Control panel
        self._panel = CropRotatePanel()
        self._panel.setMaximumWidth(280)
        layout.addWidget(self._panel, stretch=1)
        
        # Connect signals
        self._panel.rotate_90_requested.connect(self._on_rotate_90)
        self._panel.flip_requested.connect(self._on_flip)
        self._panel.straighten_requested.connect(self._on_straighten)
        self._panel.auto_crop_requested.connect(self._on_auto_crop)
        self._panel.reset_requested.connect(self._on_reset)
        self._panel.apply_requested.connect(self._on_apply)
        self._panel.cancel_requested.connect(self._on_cancel)
    
    def _update_preview(self):
        """Update the preview image."""
        if self._current_image is None:
            return
        
        h, w = self._current_image.shape[:2]
        bytes_per_line = 3 * w
        
        from PyQt6.QtGui import QImage, QPixmap
        q_image = QImage(
            self._current_image.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale to fit preview area
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self._preview_label.setPixmap(scaled)
    
    def _on_rotate_90(self, clockwise: bool):
        """Handle 90 degree rotation."""
        self._current_image = rotate_90(self._current_image, clockwise)
        self._update_preview()
    
    def _on_flip(self, horizontal: bool):
        """Handle flip."""
        self._current_image = flip_image(self._current_image, horizontal)
        self._update_preview()
    
    def _on_straighten(self, angle: float):
        """Handle straighten/rotation."""
        self._rotation_angle = angle
        if abs(angle) > 0.01:
            self._current_image = straighten_image(self._original_image, angle)
        else:
            self._current_image = self._original_image.copy()
        self._update_preview()
    
    def _on_auto_crop(self):
        """Handle auto crop borders."""
        crop_rect = auto_crop_borders(self._current_image)
        if crop_rect:
            self._current_image = crop_image(self._current_image, crop_rect)
            self._update_preview()
    
    def _on_reset(self):
        """Reset to original image."""
        self._current_image = self._original_image.copy()
        self._rotation_angle = 0.0
        self._panel.reset_controls()
        self._update_preview()
    
    def _on_apply(self):
        """Apply changes and close."""
        self._result = self._current_image.copy()
        self._accepted = True
        self.close()
    
    def _on_cancel(self):
        """Cancel and close."""
        self._result = None
        self._accepted = False
        self.close()
    
    def get_result(self):
        """Get the resulting image after crop/rotate."""
        return self._result
    
    def exec(self) -> bool:
        """Show dialog modally and return True if accepted."""
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.show()
        
        # Wait for close
        from PyQt6.QtCore import QEventLoop
        loop = QEventLoop()
        self.destroyed.connect(loop.quit)
        loop.exec()
        
        return self._accepted
