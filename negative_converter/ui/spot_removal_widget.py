# Spot Removal / Healing Brush Tool
"""
Interactive spot removal tool for manually painting over dust spots
and having the algorithm fix them using inpainting.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QCheckBox, QWidget, QScrollArea,
    QSizePolicy, QGroupBox, QButtonGroup, QRadioButton,
    QApplication
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor,
    QCursor, QMouseEvent, QPaintEvent, QWheelEvent
)
from PyQt6.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal, QTimer

from ..utils.logger import get_logger
from ..processing.dust_detection import (
    remove_artifacts, detect_artifacts, DetectionParams,
    DustSpot, Scratch
)

logger = get_logger(__name__)


@dataclass
class BrushStroke:
    """A single brush stroke (list of points with radius)."""
    points: List[Tuple[int, int]]
    radius: int
    
    def get_bounds(self) -> QRect:
        """Get bounding rectangle of the stroke."""
        if not self.points:
            return QRect()
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return QRect(
            min(xs) - self.radius,
            min(ys) - self.radius,
            max(xs) - min(xs) + self.radius * 2,
            max(ys) - min(ys) + self.radius * 2
        )


class SpotRemovalCanvas(QWidget):
    """Canvas widget for painting spot removal mask."""
    
    mask_changed = pyqtSignal()  # Emitted when mask is modified
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._mask: Optional[np.ndarray] = None
        self._overlay_pixmap: Optional[QPixmap] = None
        
        # Brush settings
        self._brush_radius = 10
        self._brush_color = QColor(255, 0, 0, 128)  # Semi-transparent red
        self._eraser_mode = False
        
        # Zoom and pan
        self._zoom_factor = 1.0
        self._pan_offset = QPoint(0, 0)
        self._is_panning = False
        self._pan_start = QPoint()
        
        # Drawing state
        self._is_drawing = False
        self._current_stroke: Optional[BrushStroke] = None
        self._strokes: List[BrushStroke] = []
        self._last_point: Optional[QPoint] = None
        
        # Auto-detected spots (for accept/reject)
        self._detected_spots: List[DustSpot] = []
        self._detected_scratches: List[Scratch] = []
        self._show_detected = True
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        
    def set_image(self, image: np.ndarray):
        """Set the image to work on."""
        if image is None:
            self._image = None
            self._pixmap = None
            self._mask = None
            self.update()
            return
            
        self._image = image.copy()
        h, w = image.shape[:2]
        
        # Create QPixmap from image
        bytes_per_line = 3 * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(q_image)
        
        # Initialize empty mask
        self._mask = np.zeros((h, w), dtype=np.uint8)
        
        # Clear strokes
        self._strokes.clear()
        self._current_stroke = None
        
        # Fit to window
        self._fit_to_window()
        self.update()
        
    def set_brush_radius(self, radius: int):
        """Set the brush radius."""
        self._brush_radius = max(1, min(100, radius))
        self.update()
        
    def set_eraser_mode(self, enabled: bool):
        """Enable/disable eraser mode."""
        self._eraser_mode = enabled
        self._update_cursor()
        
    def set_show_detected(self, show: bool):
        """Show/hide auto-detected spots."""
        self._show_detected = show
        self.update()
        
    def set_detected_artifacts(self, spots: List[DustSpot], scratches: List[Scratch]):
        """Set auto-detected artifacts to display."""
        self._detected_spots = spots
        self._detected_scratches = scratches
        self.update()
        
    def add_detected_to_mask(self):
        """Add all detected artifacts to the mask."""
        if self._mask is None:
            return
            
        import cv2
        
        for spot in self._detected_spots:
            cv2.circle(self._mask, (spot.x, spot.y), spot.radius + 2, 255, -1)
            
        for scratch in self._detected_scratches:
            cv2.line(self._mask, (scratch.x1, scratch.y1), 
                    (scratch.x2, scratch.y2), 255, scratch.width + 2)
        
        self._update_overlay()
        self.mask_changed.emit()
        self.update()
        
    def clear_mask(self):
        """Clear the entire mask."""
        if self._mask is not None:
            self._mask.fill(0)
            self._strokes.clear()
            self._update_overlay()
            self.mask_changed.emit()
            self.update()
            
    def get_mask(self) -> Optional[np.ndarray]:
        """Get the current mask."""
        return self._mask.copy() if self._mask is not None else None
        
    def get_image(self) -> Optional[np.ndarray]:
        """Get the current image."""
        return self._image.copy() if self._image is not None else None
        
    def undo_stroke(self):
        """Undo the last stroke."""
        if self._strokes:
            self._strokes.pop()
            self._rebuild_mask_from_strokes()
            self.mask_changed.emit()
            self.update()
            
    def _rebuild_mask_from_strokes(self):
        """Rebuild mask from stroke history."""
        if self._mask is None:
            return
            
        import cv2
        
        self._mask.fill(0)
        
        for stroke in self._strokes:
            for i, (x, y) in enumerate(stroke.points):
                cv2.circle(self._mask, (x, y), stroke.radius, 255, -1)
                # Connect to previous point for smooth lines
                if i > 0:
                    px, py = stroke.points[i - 1]
                    cv2.line(self._mask, (px, py), (x, y), 255, stroke.radius * 2)
                    
        self._update_overlay()
        
    def _update_overlay(self):
        """Update the overlay pixmap showing the mask."""
        if self._mask is None or self._image is None:
            self._overlay_pixmap = None
            return
            
        h, w = self._mask.shape
        
        # Create RGBA overlay
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_pixels = self._mask > 0
        overlay[mask_pixels] = [255, 0, 0, 128]  # Semi-transparent red
        
        bytes_per_line = 4 * w
        q_image = QImage(overlay.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888)
        self._overlay_pixmap = QPixmap.fromImage(q_image)
        
    def _fit_to_window(self):
        """Fit image to window."""
        if self._pixmap is None:
            return
            
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        
        if pw == 0 or ph == 0:
            return
            
        scale_x = ww / pw
        scale_y = wh / ph
        self._zoom_factor = min(scale_x, scale_y) * 0.95
        self._pan_offset = QPoint(0, 0)
        
    def _update_cursor(self):
        """Update cursor based on current mode."""
        if self._eraser_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            # Create circular brush cursor
            size = max(4, int(self._brush_radius * 2 * self._zoom_factor))
            cursor_pixmap = QPixmap(size + 2, size + 2)
            cursor_pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(cursor_pixmap)
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(1, 1, size, size)
            painter.setPen(QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.DotLine))
            painter.drawEllipse(1, 1, size, size)
            painter.end()
            
            self.setCursor(QCursor(cursor_pixmap, size // 2 + 1, size // 2 + 1))
            
    def _widget_to_image(self, pos: QPoint) -> QPoint:
        """Convert widget coordinates to image coordinates."""
        if self._pixmap is None:
            return QPoint()
            
        # Calculate displayed image rect
        pw = int(self._pixmap.width() * self._zoom_factor)
        ph = int(self._pixmap.height() * self._zoom_factor)
        
        x_offset = (self.width() - pw) // 2 + self._pan_offset.x()
        y_offset = (self.height() - ph) // 2 + self._pan_offset.y()
        
        # Convert to image coordinates
        img_x = int((pos.x() - x_offset) / self._zoom_factor)
        img_y = int((pos.y() - y_offset) / self._zoom_factor)
        
        return QPoint(img_x, img_y)
        
    def _draw_at_point(self, img_pos: QPoint):
        """Draw or erase at the given image position."""
        if self._mask is None:
            return
            
        import cv2
        
        x, y = img_pos.x(), img_pos.y()
        h, w = self._mask.shape
        
        # Bounds check
        if x < 0 or x >= w or y < 0 or y >= h:
            return
            
        if self._eraser_mode:
            cv2.circle(self._mask, (x, y), self._brush_radius, 0, -1)
        else:
            cv2.circle(self._mask, (x, y), self._brush_radius, 255, -1)
            
            # Add to current stroke
            if self._current_stroke is not None:
                self._current_stroke.points.append((x, y))
                
                # Connect to last point for smooth line
                if self._last_point is not None:
                    lx, ly = self._last_point.x(), self._last_point.y()
                    cv2.line(self._mask, (lx, ly), (x, y), 255, self._brush_radius * 2)
                    
        self._last_point = img_pos
        self._update_overlay()
        
    # --- Event Handlers ---
    
    def paintEvent(self, event: QPaintEvent):
        """Paint the canvas."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self._pixmap is None:
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return
            
        # Calculate display rect
        pw = int(self._pixmap.width() * self._zoom_factor)
        ph = int(self._pixmap.height() * self._zoom_factor)
        
        x_offset = (self.width() - pw) // 2 + self._pan_offset.x()
        y_offset = (self.height() - ph) // 2 + self._pan_offset.y()
        
        target_rect = QRect(x_offset, y_offset, pw, ph)
        
        # Draw image
        painter.drawPixmap(target_rect, self._pixmap)
        
        # Draw mask overlay
        if self._overlay_pixmap is not None:
            painter.setOpacity(0.5)
            painter.drawPixmap(target_rect, self._overlay_pixmap)
            painter.setOpacity(1.0)
            
        # Draw detected artifacts if enabled
        if self._show_detected:
            painter.setPen(QPen(QColor(255, 255, 0), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            
            for spot in self._detected_spots:
                sx = int(spot.x * self._zoom_factor) + x_offset
                sy = int(spot.y * self._zoom_factor) + y_offset
                sr = int(spot.radius * self._zoom_factor)
                painter.drawEllipse(QPoint(sx, sy), sr, sr)
                
            for scratch in self._detected_scratches:
                x1 = int(scratch.x1 * self._zoom_factor) + x_offset
                y1 = int(scratch.y1 * self._zoom_factor) + y_offset
                x2 = int(scratch.x2 * self._zoom_factor) + x_offset
                y2 = int(scratch.y2 * self._zoom_factor) + y_offset
                painter.drawLine(x1, y1, x2, y2)
                
        # Draw brush cursor preview
        if self.underMouse() and not self._is_panning:
            cursor_pos = self.mapFromGlobal(QCursor.pos())
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            radius = int(self._brush_radius * self._zoom_factor)
            painter.drawEllipse(cursor_pos, radius, radius)
            
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self._is_panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            
        elif event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            # Start drawing
            self._is_drawing = True
            self._current_stroke = BrushStroke(points=[], radius=self._brush_radius)
            self._last_point = None
            
            img_pos = self._widget_to_image(event.pos())
            self._draw_at_point(img_pos)
            self.update()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self._is_panning:
            delta = event.pos() - self._pan_start
            self._pan_offset += delta
            self._pan_start = event.pos()
            self.update()
            
        elif self._is_drawing:
            img_pos = self._widget_to_image(event.pos())
            self._draw_at_point(img_pos)
            self.update()
        else:
            # Update cursor preview
            self.update()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = False
            self._update_cursor()
            
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._is_drawing and self._current_stroke is not None:
                if self._current_stroke.points:
                    self._strokes.append(self._current_stroke)
                self._current_stroke = None
                self._is_drawing = False
                self._last_point = None
                self.mask_changed.emit()
                
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        if self._pixmap is None:
            return
            
        # Get position before zoom
        old_pos = self._widget_to_image(event.position().toPoint())
        
        # Zoom
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_factor *= 1.1
        else:
            self._zoom_factor /= 1.1
            
        self._zoom_factor = max(0.1, min(10.0, self._zoom_factor))
        
        # Adjust pan to keep mouse position stable
        new_pos = self._widget_to_image(event.position().toPoint())
        diff = new_pos - old_pos
        self._pan_offset += QPoint(int(diff.x() * self._zoom_factor), 
                                   int(diff.y() * self._zoom_factor))
        
        self._update_cursor()
        self.update()
        
    def keyPressEvent(self, event):
        """Handle key press."""
        if event.key() == Qt.Key.Key_BracketLeft:
            # Decrease brush size
            self._brush_radius = max(1, self._brush_radius - 2)
            self._update_cursor()
            self.update()
        elif event.key() == Qt.Key.Key_BracketRight:
            # Increase brush size
            self._brush_radius = min(100, self._brush_radius + 2)
            self._update_cursor()
            self.update()
        elif event.key() == Qt.Key.Key_E:
            # Toggle eraser
            self._eraser_mode = not self._eraser_mode
            self._update_cursor()
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.undo_stroke()
        else:
            super().keyPressEvent(event)
            
    def resizeEvent(self, event):
        """Handle resize."""
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._fit_to_window()


class SpotRemovalDialog(QDialog):
    """Dialog for interactive spot removal."""
    
    # Signal emitted when user applies the fix
    image_fixed = pyqtSignal(np.ndarray)  # Emits the fixed image
    
    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Spot Removal Tool")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self._original_image = image.copy()
        self._result_image: Optional[np.ndarray] = None
        
        self._setup_ui()
        self._connect_signals()
        
        # Set the image
        self.canvas.set_image(image)
        
        # Auto-detect artifacts
        self._run_auto_detection()
        
    def _setup_ui(self):
        """Set up the UI."""
        layout = QHBoxLayout(self)
        
        # Left side: Canvas
        self.canvas = SpotRemovalCanvas()
        layout.addWidget(self.canvas, stretch=3)
        
        # Right side: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        
        # Brush Settings Group
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QVBoxLayout(brush_group)
        
        # Brush size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(1, 50)
        self.brush_size_slider.setValue(10)
        size_layout.addWidget(self.brush_size_slider)
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 50)
        self.brush_size_spin.setValue(10)
        size_layout.addWidget(self.brush_size_spin)
        brush_layout.addLayout(size_layout)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        self.paint_radio = QRadioButton("Paint")
        self.paint_radio.setChecked(True)
        self.erase_radio = QRadioButton("Erase")
        mode_layout.addWidget(self.paint_radio)
        mode_layout.addWidget(self.erase_radio)
        brush_layout.addLayout(mode_layout)
        
        controls_layout.addWidget(brush_group)
        
        # Auto Detection Group
        detect_group = QGroupBox("Auto Detection")
        detect_layout = QVBoxLayout(detect_group)
        
        # Sensitivity slider
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(QLabel("Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(50)
        sens_layout.addWidget(self.sensitivity_slider)
        self.sensitivity_label = QLabel("50%")
        sens_layout.addWidget(self.sensitivity_label)
        detect_layout.addLayout(sens_layout)
        
        # Show detected checkbox
        self.show_detected_check = QCheckBox("Show detected spots")
        self.show_detected_check.setChecked(True)
        detect_layout.addWidget(self.show_detected_check)
        
        # Detection buttons
        detect_btn_layout = QHBoxLayout()
        self.detect_btn = QPushButton("Re-detect")
        self.detect_btn.setToolTip("Run auto-detection with current sensitivity")
        detect_btn_layout.addWidget(self.detect_btn)
        
        self.add_detected_btn = QPushButton("Add All to Mask")
        self.add_detected_btn.setToolTip("Add all detected spots to the removal mask")
        detect_btn_layout.addWidget(self.add_detected_btn)
        detect_layout.addLayout(detect_btn_layout)
        
        controls_layout.addWidget(detect_group)
        
        # Inpainting Settings Group
        inpaint_group = QGroupBox("Inpainting Settings")
        inpaint_layout = QVBoxLayout(inpaint_group)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.method_telea_radio = QRadioButton("Telea")
        self.method_telea_radio.setChecked(True)
        self.method_ns_radio = QRadioButton("Navier-Stokes")
        self.method_patch_radio = QRadioButton("Patch Match")
        method_layout.addWidget(self.method_telea_radio)
        method_layout.addWidget(self.method_ns_radio)
        method_layout.addWidget(self.method_patch_radio)
        inpaint_layout.addLayout(method_layout)
        
        # Preserve grain checkbox
        self.preserve_grain_check = QCheckBox("Preserve film grain")
        self.preserve_grain_check.setChecked(True)
        inpaint_layout.addWidget(self.preserve_grain_check)
        
        controls_layout.addWidget(inpaint_group)
        
        # Actions Group
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.preview_btn = QPushButton("Preview Fix")
        self.preview_btn.setToolTip("Preview the result without applying")
        actions_layout.addWidget(self.preview_btn)
        
        self.clear_btn = QPushButton("Clear Mask")
        self.clear_btn.setToolTip("Clear all painted areas")
        actions_layout.addWidget(self.clear_btn)
        
        self.undo_btn = QPushButton("Undo Stroke (Ctrl+Z)")
        self.undo_btn.setToolTip("Undo the last brush stroke")
        actions_layout.addWidget(self.undo_btn)
        
        controls_layout.addWidget(actions_group)
        
        # Spacer
        controls_layout.addStretch()
        
        # Help text
        help_label = QLabel(
            "<b>Tips:</b><br>"
            "• Left-click and drag to paint<br>"
            "• Middle-click to pan<br>"
            "• Scroll to zoom<br>"
            "• [ / ] to change brush size<br>"
            "• E to toggle eraser"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 11px;")
        controls_layout.addWidget(help_label)
        
        # Dialog buttons
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply && Close")
        self.apply_btn.setDefault(True)
        self.cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.cancel_btn)
        controls_layout.addLayout(btn_layout)
        
        layout.addWidget(controls_widget, stretch=1)
        
    def _connect_signals(self):
        """Connect signals to slots."""
        # Brush size
        self.brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_spin_changed)
        
        # Mode
        self.paint_radio.toggled.connect(self._on_mode_changed)
        
        # Detection
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.show_detected_check.toggled.connect(self.canvas.set_show_detected)
        self.detect_btn.clicked.connect(self._run_auto_detection)
        self.add_detected_btn.clicked.connect(self.canvas.add_detected_to_mask)
        
        # Actions
        self.preview_btn.clicked.connect(self._preview_fix)
        self.clear_btn.clicked.connect(self.canvas.clear_mask)
        self.undo_btn.clicked.connect(self.canvas.undo_stroke)
        
        # Dialog buttons
        self.apply_btn.clicked.connect(self._apply_and_close)
        self.cancel_btn.clicked.connect(self.reject)
        
    def _on_brush_size_changed(self, value: int):
        """Handle brush size slider change."""
        self.brush_size_spin.blockSignals(True)
        self.brush_size_spin.setValue(value)
        self.brush_size_spin.blockSignals(False)
        self.canvas.set_brush_radius(value)
        
    def _on_brush_size_spin_changed(self, value: int):
        """Handle brush size spinbox change."""
        self.brush_size_slider.blockSignals(True)
        self.brush_size_slider.setValue(value)
        self.brush_size_slider.blockSignals(False)
        self.canvas.set_brush_radius(value)
        
    def _on_mode_changed(self, paint_mode: bool):
        """Handle mode change."""
        self.canvas.set_eraser_mode(not paint_mode)
        
    def _on_sensitivity_changed(self, value: int):
        """Handle sensitivity slider change."""
        self.sensitivity_label.setText(f"{value}%")
        
    def _run_auto_detection(self):
        """Run auto-detection with current sensitivity."""
        if self._original_image is None:
            return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            sensitivity = self.sensitivity_slider.value() / 100.0
            params = DetectionParams(
                dust_sensitivity=sensitivity,
                scratch_sensitivity=sensitivity,
            )
            
            result = detect_artifacts(self._original_image, params)
            self.canvas.set_detected_artifacts(result.dust_spots, result.scratches)
            
            logger.info("Auto-detected %d dust spots and %d scratches",
                       len(result.dust_spots), len(result.scratches))
        except Exception as e:
            logger.exception("Error during auto-detection")
        finally:
            QApplication.restoreOverrideCursor()
            
    def _get_inpaint_method(self) -> str:
        """Get the selected inpainting method."""
        if self.method_ns_radio.isChecked():
            return "inpaint_ns"
        elif self.method_patch_radio.isChecked():
            return "patch_match"
        return "inpaint_telea"
        
    def _preview_fix(self):
        """Preview the fix without applying."""
        mask = self.canvas.get_mask()
        if mask is None or mask.max() == 0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "No Mask", 
                "Please paint over the areas you want to fix first.")
            return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            method = self._get_inpaint_method()
            preserve_grain = self.preserve_grain_check.isChecked()
            
            result = remove_artifacts(
                self._original_image,
                mask,
                method=method,
                preserve_grain=preserve_grain
            )
            
            self._result_image = result
            self.canvas.set_image(result)
            
            logger.info("Preview generated using %s method", method)
        except Exception as e:
            logger.exception("Error during preview")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to generate preview: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            
    def _apply_and_close(self):
        """Apply the fix and close the dialog."""
        mask = self.canvas.get_mask()
        if mask is None or mask.max() == 0:
            # No mask painted - just close
            self.reject()
            return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            method = self._get_inpaint_method()
            preserve_grain = self.preserve_grain_check.isChecked()
            
            result = remove_artifacts(
                self._original_image,
                mask,
                method=method,
                preserve_grain=preserve_grain
            )
            
            self._result_image = result
            self.image_fixed.emit(result)
            self.accept()
            
            logger.info("Spot removal applied using %s method", method)
        except Exception as e:
            logger.exception("Error during spot removal")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to apply fix: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            
    def get_result(self) -> Optional[np.ndarray]:
        """Get the result image after applying fixes."""
        return self._result_image
