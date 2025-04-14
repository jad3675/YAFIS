# Curves adjustment widget
import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
                             QPushButton, QSizePolicy, QApplication, QMainWindow)
from PyQt6.QtCore import pyqtSignal, QPointF, QRectF, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath

class CurveGraphWidget(QWidget):
    """Custom widget to display and interact with a curve."""
    points_changed = pyqtSignal(list) # Emits the list of points [[x,y], ...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._points = [[0, 0], [255, 255]] # Default linear curve (input range 0-255)
        self._padding = 10 # Padding around the graph area
        self._point_radius = 4
        self._selected_point_index = -1
        self._dragging = False
        self.setBackgroundRole(self.parent().backgroundRole() if self.parent() else self.backgroundRole())
        self.setAutoFillBackground(True)


    def set_points(self, points):
        """Set the control points for the curve."""
        # Basic validation and sorting
        if isinstance(points, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
            # Ensure points are within 0-255 range and sorted by x
            valid_points = []
            for p in points:
                x = max(0, min(p[0], 255))
                y = max(0, min(p[1], 255))
                valid_points.append([x, y])
            self._points = sorted(valid_points, key=lambda p: p[0])
            self.update() # Trigger repaint
        else:
            print("Warning: Invalid points format for CurveGraphWidget.")

    def get_points(self):
        """Return the current list of control points."""
        return self._points

    def _world_to_widget(self, p):
        """Convert world coordinates (0-255) to widget coordinates."""
        graph_rect = self.get_graph_rect()
        x = graph_rect.left() + (p[0] / 255.0) * graph_rect.width()
        y = graph_rect.bottom() - (p[1] / 255.0) * graph_rect.height() # Y is inverted
        return QPointF(x, y)

    def _widget_to_world(self, p):
        """Convert widget coordinates to world coordinates (0-255)."""
        graph_rect = self.get_graph_rect()
        if not graph_rect.contains(p):
             # If outside graph rect, clamp to nearest edge in world coords
             x_world = max(0, min(255, ((p.x() - graph_rect.left()) / graph_rect.width()) * 255.0))
             y_world = max(0, min(255, ((graph_rect.bottom() - p.y()) / graph_rect.height()) * 255.0))
        else:
            x_world = ((p.x() - graph_rect.left()) / graph_rect.width()) * 255.0
            y_world = ((graph_rect.bottom() - p.y()) / graph_rect.height()) * 255.0

        # Clamp results to 0-255
        return max(0, min(x_world, 255)), max(0, min(y_world, 255))


    def get_graph_rect(self):
        """Calculate the rectangle where the graph is drawn."""
        return QRectF(
            self._padding,
            self._padding,
            self.width() - 2 * self._padding,
            self.height() - 2 * self._padding
        )

    def paintEvent(self, event):
        """Draw the curve graph."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        graph_rect = self.get_graph_rect()

        # Draw background grid (optional)
        painter.setPen(QColor(200, 200, 200)) # Light grey grid
        # Draw vertical lines
        for i in range(1, 4):
            x = graph_rect.left() + i * graph_rect.width() / 4
            painter.drawLine(QPointF(x, graph_rect.top()), QPointF(x, graph_rect.bottom()))
        # Draw horizontal lines
        for i in range(1, 4):
            y = graph_rect.top() + i * graph_rect.height() / 4
            painter.drawLine(QPointF(graph_rect.left(), y), QPointF(graph_rect.right(), y))

        # Draw border
        painter.setPen(QColor(100, 100, 100))
        painter.drawRect(graph_rect)

        if not self._points:
            return

        # Draw the curve (linear interpolation between points)
        painter.setPen(QPen(QColor(0, 0, 0), 1.5)) # Black curve
        path = QPainterPath()
        widget_points = [self._world_to_widget(p) for p in self._points]
        path.moveTo(widget_points[0])
        for i in range(1, len(widget_points)):
            path.lineTo(widget_points[i])
        painter.drawPath(path)

        # Draw control points
        for i, p_widget in enumerate(widget_points):
            if i == self._selected_point_index:
                painter.setBrush(QBrush(QColor(255, 0, 0))) # Red for selected
                painter.setPen(QPen(QColor(100, 0, 0), 1))
            else:
                painter.setBrush(QBrush(QColor(50, 50, 200))) # Blue for others
                painter.setPen(QPen(QColor(0, 0, 100), 1))
            painter.drawEllipse(p_widget, self._point_radius, self._point_radius)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._selected_point_index = -1
            min_dist_sq = (self._point_radius * 2) ** 2 # Check within double the radius

            for i, p in enumerate(self._points):
                p_widget = self._world_to_widget(p)
                dist_sq = (pos.x() - p_widget.x())**2 + (pos.y() - p_widget.y())**2
                if dist_sq < min_dist_sq:
                    self._selected_point_index = i
                    self._dragging = True
                    self.update()
                    return

            # If no point clicked, add a new point if inside graph area
            graph_rect = self.get_graph_rect()
            if graph_rect.contains(pos):
                world_x, world_y = self._widget_to_world(pos)
                # Add point and keep sorted
                self._points.append([world_x, world_y])
                self._points = sorted(self._points, key=lambda p: p[0])
                # Find index of newly added point
                for i, p in enumerate(self._points):
                    if p[0] == world_x and p[1] == world_y:
                         self._selected_point_index = i
                         break
                self._dragging = True
                self.update()
                self.points_changed.emit(self._points)


    def mouseMoveEvent(self, event):
        if self._dragging and self._selected_point_index != -1:
            pos = event.position()
            world_x, world_y = self._widget_to_world(pos)

            # Prevent moving start/end points horizontally
            if self._selected_point_index == 0:
                world_x = 0
            elif self._selected_point_index == len(self._points) - 1:
                world_x = 255

            # Prevent moving points past their neighbors horizontally
            if self._selected_point_index > 0:
                world_x = max(world_x, self._points[self._selected_point_index - 1][0])
            if self._selected_point_index < len(self._points) - 1:
                world_x = min(world_x, self._points[self._selected_point_index + 1][0])

            self._points[self._selected_point_index] = [world_x, world_y]
            self.update()
            self.points_changed.emit(self._points)


    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            # Optionally deselect point on release, or keep selected
            # self._selected_point_index = -1
            # self.update()


class CurvesWidget(QWidget):
    """Widget for adjusting image curves."""
    curve_changed = pyqtSignal(str, list) # channel_name, points_list

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_channel = 'RGB'
        self._curve_points = {
            'RGB': [[0, 0], [255, 255]],
            'Red': [[0, 0], [255, 255]],
            'Green': [[0, 0], [255, 255]],
            'Blue': [[0, 0], [255, 255]],
        }
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        # --- Channel Selector ---
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(['RGB', 'Red', 'Green', 'Blue'])
        self.channel_combo.currentTextChanged.connect(self.channel_changed)
        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch(1)
        main_layout.addLayout(channel_layout)

        # --- Curve Graph Area ---
        self.graph_widget = CurveGraphWidget(self)
        self.graph_widget.points_changed.connect(self._graph_points_updated)
        main_layout.addWidget(self.graph_widget)

        # --- Control Buttons ---
        button_layout = QHBoxLayout()
        reset_button = QPushButton("Reset Curve")
        reset_button.clicked.connect(self.reset_current_curve)
        button_layout.addWidget(reset_button)
        button_layout.addStretch(1)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.channel_changed('RGB') # Initialize graph display

    def channel_changed(self, channel_name):
        """Handle channel selection change."""
        self._current_channel = channel_name
        print(f"Curves channel changed to: {self._current_channel}")
        # Update graph display to show the curve for the selected channel
        self.graph_widget.set_points(self._curve_points[self._current_channel])

    def reset_current_curve(self):
        """Reset the points for the currently selected channel."""
        print(f"Resetting curve for channel: {self._current_channel}")
        default_points = [[0, 0], [255, 255]]
        self._curve_points[self._current_channel] = default_points
        self.graph_widget.set_points(default_points) # Update graph
        # Emit signal with reset points
        self.curve_changed.emit(self._current_channel, default_points)

    def _graph_points_updated(self, points):
        """Slot to receive point changes from the graph widget."""
        self._curve_points[self._current_channel] = points
        # Emit the main signal from CurvesWidget
        self.curve_changed.emit(self._current_channel, points)

    def get_curve_points(self, channel=None):
        """Get curve points for a specific channel or all channels."""
        if channel:
            return self._curve_points.get(channel, [[0, 0], [255, 255]])
        else:
            return self._curve_points.copy()

    def set_curve_points(self, channel, points):
        """Set curve points for a specific channel (e.g., loading from settings)."""
        if channel in self._curve_points:
            # Basic validation: ensure points are list of pairs
            if isinstance(points, list) and all(isinstance(p, list) and len(p) == 2 for p in points):
                 # Sort points by x-coordinate
                self._curve_points[channel] = sorted(points, key=lambda p: p[0])
                if channel == self._current_channel:
                    self.graph_widget.set_points(self._curve_points[channel]) # Update graph
            else:
                print(f"Warning: Invalid points format for setting curve '{channel}'.")
        else:
            print(f"Warning: Invalid channel '{channel}' for setting curve points.")


# Example usage (for testing standalone)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = QMainWindow()
    mainWin.setWindowTitle("Curves Widget Test")
    curves_widget = CurvesWidget()
    mainWin.setCentralWidget(curves_widget)
    curves_widget.curve_changed.connect(lambda channel, points: print(f"Curve changed for {channel}: {points}"))
    mainWin.setGeometry(300, 300, 300, 350) # Adjusted size
    mainWin.show()
    sys.exit(app.exec())