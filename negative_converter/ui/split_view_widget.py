# Split view comparison widget
"""
Widget for side-by-side or split comparison of different image states.
"""

from typing import Optional, Tuple, List
from enum import Enum
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSlider, QFrame, QSplitter, QScrollArea,
    QSizePolicy, QToolButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CompareMode(Enum):
    """Comparison display modes."""
    SIDE_BY_SIDE = "side_by_side"
    SPLIT_HORIZONTAL = "split_horizontal"
    SPLIT_VERTICAL = "split_vertical"
    OVERLAY_BLEND = "overlay_blend"
    DIFFERENCE = "difference"


class CompareState:
    """Represents a saved state for comparison."""
    
    def __init__(self, name: str, image: np.ndarray, adjustments: dict = None):
        self.name = name
        self.image = image.copy() if image is not None else None
        self.adjustments = adjustments.copy() if adjustments else {}
        self._pixmap: Optional[QPixmap] = None
    
    def get_pixmap(self) -> Optional[QPixmap]:
        """Get or create QPixmap from image data."""
        if self.image is None:
            return None
        
        if self._pixmap is None:
            h, w = self.image.shape[:2]
            bytes_per_line = 3 * w
            q_image = QImage(
                self.image.data, w, h, bytes_per_line,
                QImage.Format.Format_RGB888
            )
            self._pixmap = QPixmap.fromImage(q_image)
        
        return self._pixmap
    
    def invalidate_pixmap(self):
        """Invalidate cached pixmap."""
        self._pixmap = None


class SplitImageWidget(QWidget):
    """
    Widget that displays two images with a movable split divider.
    """
    split_position_changed = pyqtSignal(float)  # 0.0 to 1.0
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._left_pixmap: Optional[QPixmap] = None
        self._right_pixmap: Optional[QPixmap] = None
        self._split_position = 0.5  # 0.0 = all left, 1.0 = all right
        self._split_orientation = Qt.Orientation.Horizontal  # Vertical divider
        self._dragging = False
        self._zoom_factor = 1.0
        self._blend_opacity = 0.5
        self._mode = CompareMode.SPLIT_VERTICAL
        
        self.setMouseTracking(True)
        self.setMinimumSize(200, 200)
    
    def set_images(self, left: Optional[QPixmap], right: Optional[QPixmap]):
        """Set the two images to compare."""
        self._left_pixmap = left
        self._right_pixmap = right
        self.update()
    
    def set_split_position(self, position: float):
        """Set split position (0.0 to 1.0)."""
        self._split_position = max(0.0, min(1.0, position))
        self.update()
        self.split_position_changed.emit(self._split_position)
    
    def set_mode(self, mode: CompareMode):
        """Set comparison mode."""
        self._mode = mode
        self.update()
    
    def set_blend_opacity(self, opacity: float):
        """Set blend opacity for overlay mode."""
        self._blend_opacity = max(0.0, min(1.0, opacity))
        self.update()
    
    def set_zoom(self, factor: float):
        """Set zoom factor."""
        self._zoom_factor = max(0.1, min(10.0, factor))
        self.update()
    
    def paintEvent(self, event):
        """Paint the comparison view."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        if self._left_pixmap is None and self._right_pixmap is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No images to compare")
            return
        
        if self._mode == CompareMode.SIDE_BY_SIDE:
            self._paint_side_by_side(painter)
        elif self._mode == CompareMode.SPLIT_VERTICAL:
            self._paint_split_vertical(painter)
        elif self._mode == CompareMode.SPLIT_HORIZONTAL:
            self._paint_split_horizontal(painter)
        elif self._mode == CompareMode.OVERLAY_BLEND:
            self._paint_overlay_blend(painter)
        elif self._mode == CompareMode.DIFFERENCE:
            self._paint_difference(painter)
        
        painter.end()
    
    def _paint_side_by_side(self, painter: QPainter):
        """Paint images side by side."""
        w = self.width()
        h = self.height()
        half_w = w // 2 - 2
        
        # Left image
        if self._left_pixmap:
            scaled = self._left_pixmap.scaled(
                half_w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (half_w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        
        # Divider
        painter.setPen(QPen(QColor(128, 128, 128), 2))
        painter.drawLine(half_w + 1, 0, half_w + 1, h)
        
        # Right image
        if self._right_pixmap:
            scaled = self._right_pixmap.scaled(
                half_w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = half_w + 3 + (half_w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
    
    def _paint_split_vertical(self, painter: QPainter):
        """Paint with vertical split divider."""
        w = self.width()
        h = self.height()
        split_x = int(w * self._split_position)
        
        # Scale both images to fit
        if self._left_pixmap:
            left_scaled = self._left_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            # Center the image
            lx = (w - left_scaled.width()) // 2
            ly = (h - left_scaled.height()) // 2
            
            # Draw left portion
            painter.setClipRect(0, 0, split_x, h)
            painter.drawPixmap(lx, ly, left_scaled)
        
        if self._right_pixmap:
            right_scaled = self._right_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            rx = (w - right_scaled.width()) // 2
            ry = (h - right_scaled.height()) // 2
            
            # Draw right portion
            painter.setClipRect(split_x, 0, w - split_x, h)
            painter.drawPixmap(rx, ry, right_scaled)
        
        # Draw divider
        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(split_x, 0, split_x, h)
        
        # Draw handle
        handle_y = h // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(split_x - 8, handle_y - 8, 16, 16)
    
    def _paint_split_horizontal(self, painter: QPainter):
        """Paint with horizontal split divider."""
        w = self.width()
        h = self.height()
        split_y = int(h * self._split_position)
        
        if self._left_pixmap:
            scaled = self._left_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            
            painter.setClipRect(0, 0, w, split_y)
            painter.drawPixmap(x, y, scaled)
        
        if self._right_pixmap:
            scaled = self._right_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            
            painter.setClipRect(0, split_y, w, h - split_y)
            painter.drawPixmap(x, y, scaled)
        
        painter.setClipping(False)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawLine(0, split_y, w, split_y)
        
        handle_x = w // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(handle_x - 8, split_y - 8, 16, 16)
    
    def _paint_overlay_blend(self, painter: QPainter):
        """Paint with opacity blend."""
        w = self.width()
        h = self.height()
        
        if self._left_pixmap:
            scaled = self._left_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.setOpacity(1.0 - self._blend_opacity)
            painter.drawPixmap(x, y, scaled)
        
        if self._right_pixmap:
            scaled = self._right_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.setOpacity(self._blend_opacity)
            painter.drawPixmap(x, y, scaled)
        
        painter.setOpacity(1.0)
    
    def _paint_difference(self, painter: QPainter):
        """Paint difference between images."""
        # For difference mode, we'd need to compute pixel differences
        # This is a simplified version that just shows the right image
        # A full implementation would compute |left - right| per pixel
        w = self.width()
        h = self.height()
        
        if self._right_pixmap:
            scaled = self._right_pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (w - scaled.width()) // 2
            y = (h - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        
        # Draw "DIFF" label
        painter.setPen(QColor(255, 255, 0))
        painter.drawText(10, 20, "Difference Mode")
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._is_near_divider(event.pos()):
                self._dragging = True
                self.setCursor(Qt.CursorShape.SplitHCursor if self._mode == CompareMode.SPLIT_VERTICAL 
                              else Qt.CursorShape.SplitVCursor)
                event.accept()
                return
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self._dragging:
            if self._mode == CompareMode.SPLIT_VERTICAL:
                pos = event.pos().x() / self.width()
            else:
                pos = event.pos().y() / self.height()
            self.set_split_position(pos)
            event.accept()
            return
        
        # Update cursor
        if self._is_near_divider(event.pos()):
            self.setCursor(Qt.CursorShape.SplitHCursor if self._mode == CompareMode.SPLIT_VERTICAL 
                          else Qt.CursorShape.SplitVCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)
    
    def _is_near_divider(self, pos: QPoint) -> bool:
        """Check if position is near the split divider."""
        threshold = 10
        
        if self._mode == CompareMode.SPLIT_VERTICAL:
            split_x = int(self.width() * self._split_position)
            return abs(pos.x() - split_x) < threshold
        elif self._mode == CompareMode.SPLIT_HORIZONTAL:
            split_y = int(self.height() * self._split_position)
            return abs(pos.y() - split_y) < threshold
        
        return False


class ComparePanel(QWidget):
    """
    Panel for managing comparison states and modes.
    """
    state_selected = pyqtSignal(int, int)  # left_index, right_index
    save_state_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._states: List[CompareState] = []
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Side by Side",
            "Split Vertical",
            "Split Horizontal",
            "Overlay Blend",
            "Difference"
        ])
        self.mode_combo.setCurrentIndex(1)  # Default to split vertical
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)
        
        # Split/blend slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Position:"))
        
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 100)
        self.position_slider.setValue(50)
        slider_layout.addWidget(self.position_slider)
        
        self.position_label = QLabel("50%")
        self.position_label.setMinimumWidth(40)
        slider_layout.addWidget(self.position_label)
        layout.addLayout(slider_layout)
        
        self.position_slider.valueChanged.connect(
            lambda v: self.position_label.setText(f"{v}%")
        )
        
        # State selection
        states_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Left:"))
        self.left_combo = QComboBox()
        left_layout.addWidget(self.left_combo)
        states_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Right:"))
        self.right_combo = QComboBox()
        right_layout.addWidget(self.right_combo)
        states_layout.addLayout(right_layout)
        
        layout.addLayout(states_layout)
        
        # Swap button
        swap_btn = QPushButton("⇄ Swap")
        swap_btn.clicked.connect(self._swap_states)
        layout.addWidget(swap_btn)
        
        # Save state button
        save_btn = QPushButton("Save Current State")
        save_btn.setToolTip("Save current adjustments as a comparison state")
        save_btn.clicked.connect(self.save_state_requested.emit)
        layout.addWidget(save_btn)
        
        # States list
        self.states_label = QLabel("Saved States: 0")
        layout.addWidget(self.states_label)
        
        layout.addStretch()
        
        # Connect signals
        self.left_combo.currentIndexChanged.connect(self._on_selection_changed)
        self.right_combo.currentIndexChanged.connect(self._on_selection_changed)
    
    def add_state(self, state: CompareState):
        """Add a comparison state."""
        self._states.append(state)
        self._update_combos()
    
    def clear_states(self):
        """Clear all states."""
        self._states.clear()
        self._update_combos()
    
    def get_state(self, index: int) -> Optional[CompareState]:
        """Get state by index."""
        if 0 <= index < len(self._states):
            return self._states[index]
        return None
    
    def _update_combos(self):
        """Update combo boxes with current states."""
        self.left_combo.blockSignals(True)
        self.right_combo.blockSignals(True)
        
        self.left_combo.clear()
        self.right_combo.clear()
        
        for state in self._states:
            self.left_combo.addItem(state.name)
            self.right_combo.addItem(state.name)
        
        if len(self._states) >= 2:
            self.left_combo.setCurrentIndex(0)
            self.right_combo.setCurrentIndex(1)
        
        self.left_combo.blockSignals(False)
        self.right_combo.blockSignals(False)
        
        self.states_label.setText(f"Saved States: {len(self._states)}")
    
    def _swap_states(self):
        """Swap left and right selections."""
        left_idx = self.left_combo.currentIndex()
        right_idx = self.right_combo.currentIndex()
        
        self.left_combo.setCurrentIndex(right_idx)
        self.right_combo.setCurrentIndex(left_idx)
    
    def _on_selection_changed(self):
        """Handle state selection change."""
        left_idx = self.left_combo.currentIndex()
        right_idx = self.right_combo.currentIndex()
        self.state_selected.emit(left_idx, right_idx)
    
    def get_mode(self) -> CompareMode:
        """Get current comparison mode."""
        modes = [
            CompareMode.SIDE_BY_SIDE,
            CompareMode.SPLIT_VERTICAL,
            CompareMode.SPLIT_HORIZONTAL,
            CompareMode.OVERLAY_BLEND,
            CompareMode.DIFFERENCE,
        ]
        return modes[self.mode_combo.currentIndex()]
    
    def get_position(self) -> float:
        """Get split/blend position (0.0 to 1.0)."""
        return self.position_slider.value() / 100.0


class SplitViewWidget(QWidget):
    """
    Main dialog/widget for split view comparison.
    Combines SplitImageWidget with ComparePanel.
    """
    
    def __init__(self, states: List[CompareState] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split View Comparison")
        self.setWindowFlags(Qt.WindowType.Dialog)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(1000, 700)
        
        self._setup_ui()
        
        # Add initial states
        if states:
            for state in states:
                self._panel.add_state(state)
            self._update_comparison()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Main comparison view
        self._image_widget = SplitImageWidget()
        self._image_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._image_widget, stretch=4)
        
        # Control panel
        self._panel = ComparePanel()
        self._panel.setMaximumWidth(250)
        layout.addWidget(self._panel, stretch=1)
        
        # Connect signals
        self._panel.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._panel.position_slider.valueChanged.connect(self._on_position_changed)
        self._panel.state_selected.connect(self._on_state_selected)
    
    def _on_mode_changed(self):
        """Handle mode change."""
        mode = self._panel.get_mode()
        self._image_widget.set_mode(mode)
    
    def _on_position_changed(self, value: int):
        """Handle position slider change."""
        self._image_widget.set_split_position(value / 100.0)
        self._image_widget.set_blend_opacity(value / 100.0)
    
    def _on_state_selected(self, left_idx: int, right_idx: int):
        """Handle state selection change."""
        left_state = self._panel.get_state(left_idx)
        right_state = self._panel.get_state(right_idx)
        
        left_pixmap = left_state.get_pixmap() if left_state else None
        right_pixmap = right_state.get_pixmap() if right_state else None
        
        self._image_widget.set_images(left_pixmap, right_pixmap)
    
    def _update_comparison(self):
        """Update the comparison view with current selections."""
        left_idx = self._panel.left_combo.currentIndex()
        right_idx = self._panel.right_combo.currentIndex()
        self._on_state_selected(left_idx, right_idx)
        self._on_mode_changed()
    
    def exec(self):
        """Show the widget as a modal dialog."""
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.show()
        # For a true modal dialog, we'd need to use QDialog
        # This is a simplified version that just shows the widget
