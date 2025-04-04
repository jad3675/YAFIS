# Image display widget
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy, QScrollArea, QRubberBand
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QCursor # Added for cursor change

class ImageViewer(QWidget):
    """Widget to display an image using QLabel, with zoom, pan, and color picking."""
    color_sampled = pyqtSignal(tuple) # Signal emitting (r, g, b) tuple

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None # The original, unscaled pixmap
        self._zoom_factor = 1.0
        self._scale_increment = 1.25 # Zoom step factor
        self._rubber_band = None
        self._drag_start_pos = None
        self._picker_mode_active = False # State for WB picker
        self.setup_ui()

    def setup_ui(self):
        """Set up the UI elements."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0) # No margins for the viewer itself

        # Use QScrollArea to handle images larger than the widget
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setBackgroundRole(self.backgroundRole()) # Inherit background
        self.scroll_area.setWidgetResizable(False) # We control the label size based on zoom
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center image if smaller

        self.image_label = QLabel(self.scroll_area)
        self.image_label.setBackgroundRole(self.backgroundRole())
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setScaledContents(False) # Start without scaling
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Enable mouse tracking for the label to get move events even when button isn't pressed
        # self.image_label.setMouseTracking(True) # Not strictly needed for drag-rectangle

        self.scroll_area.setWidget(self.image_label)
        main_layout.addWidget(self.scroll_area)

        self.setLayout(main_layout)
        # Set cursor initially
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setMinimumSize(200, 200) # Set a reasonable minimum size

    def set_image(self, image_np):
        """Sets the image to be displayed from a NumPy array (RGB uint8).

        Args:
            image_np (numpy.ndarray): The image data in RGB uint8 format.
                                      If None, clears the display.
        """
        if image_np is None:
            self._pixmap = None
            self.image_label.clear()
            self.image_label.setText("No Image Loaded")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        if not isinstance(image_np, np.ndarray) or image_np.dtype != np.uint8:
            print("Error: Image must be a NumPy array of type uint8.")
            # Optionally clear or show an error message
            self._pixmap = None
            self.image_label.clear()
            self.image_label.setText("Invalid Image Data")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return

        try:
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            if q_image.isNull():
                print("Error: Failed to create QImage from NumPy array.")
                self._pixmap = None
                self.image_label.clear()
                self.image_label.setText("Image Conversion Error")
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                return

            self._pixmap = QPixmap.fromImage(q_image) # Store the original pixmap
            self.fit_to_window() # Fit new image to window by default

        except Exception as e:
            print(f"Error converting NumPy array to QPixmap: {e}")
            self._pixmap = None
            self.image_label.clear()
            self.image_label.setText("Display Error")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def _update_display(self):
        """Updates the QLabel with the pixmap scaled by the current zoom factor."""
        if self._pixmap is None or self._pixmap.isNull():
            self.image_label.clear()
            if self._pixmap is None:
                self.image_label.setText("No Image Loaded")
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Ensure label takes minimal size when cleared
            self.image_label.resize(0, 0)
            return

        # Scale the original pixmap by the zoom factor
        original_size = self._pixmap.size()
        scaled_size = QSize(int(original_size.width() * self._zoom_factor),
                            int(original_size.height() * self._zoom_factor))

        # Ensure minimum size of 1x1 pixel for the pixmap
        scaled_size.setWidth(max(1, scaled_size.width()))
        scaled_size.setHeight(max(1, scaled_size.height()))

        scaled_pixmap = self._pixmap.scaled(scaled_size,
                                            Qt.AspectRatioMode.KeepAspectRatio, # Keep aspect ratio
                                            Qt.TransformationMode.SmoothTransformation) # Use smooth scaling

        self.image_label.setPixmap(scaled_pixmap)
        # Resize the label to match the scaled pixmap exactly
        # This allows the scroll area to function correctly
        self.image_label.resize(scaled_pixmap.size())
        # Alignment is less critical now as the label fits the pixmap
        # self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    # Removed resizeEvent handler - resizing the window no longer changes zoom level.
    # The QScrollArea handles the view of the potentially larger QLabel.

    # --- Zoom Methods ---

    def zoom_in(self):
        """Zooms in on the image."""
        self._zoom_factor *= self._scale_increment
        self._update_display()

    def zoom_out(self):
        """Zooms out of the image."""
        # Set a minimum zoom level (e.g., 1% of original size)
        min_zoom = 0.01
        self._zoom_factor /= self._scale_increment
        if self._zoom_factor < min_zoom:
            self._zoom_factor = min_zoom
        self._update_display()

    def reset_zoom(self):
        """Resets the zoom to 100%."""
        self._zoom_factor = 1.0
        self._update_display()

    def fit_to_window(self):
         """Adjusts zoom factor to fit the image within the current window size."""
         if self._pixmap is None or self._pixmap.isNull():
             return

         pixmap_size = self._pixmap.size()
         viewport_size = self.scroll_area.viewport().size() # Use viewport for available space

         if pixmap_size.width() == 0 or pixmap_size.height() == 0:
             return # Avoid division by zero

         # Calculate zoom factors needed to fit width and height
         width_factor = viewport_size.width() / pixmap_size.width()
         height_factor = viewport_size.height() / pixmap_size.height()

         # Use the smaller factor to ensure the whole image fits
         self._zoom_factor = min(width_factor, height_factor)

         # Add a small margin if desired (optional)
         # self._zoom_factor *= 0.98

         # Prevent zooming smaller than a certain threshold if fitting results in tiny image
         min_fit_zoom = 0.05
         if self._zoom_factor < min_fit_zoom:
             self._zoom_factor = min_fit_zoom

         self._update_display()

    def resizeEvent(self, event):
        """Handle widget resize events."""
        # Optionally, you could call fit_to_window() here if you want
        # the image to automatically fit when the window resizes.
        # For now, we keep the zoom level constant on resize.
        super().resizeEvent(event)
        # Ensure cursor is correct after resize potentially changes context
        self._update_cursor()
        # self.fit_to_window() # Uncomment to enable auto-fit on resize

    # --- Mouse Events for Zoom Rectangle ---

    def mousePressEvent(self, event):
        """Handle mouse press events to start drawing the zoom rectangle."""
        # Handle picker mode first
        if self._picker_mode_active and event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._sample_color_at(event.pos())
            event.accept()
            return # Don't proceed with zoom/pan

        # Handle zoom rectangle start
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            self._drag_start_pos = event.pos()
            # Ensure rubber band is created relative to the image_label
            if self._rubber_band is None:
                self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.image_label) # Parent is label

            # Map start position to label coordinates
            label_pos = self.image_label.mapFromParent(event.pos())
            # Check if click is within the actual pixmap bounds on the label
            if self.image_label.pixmap() and self.image_label.pixmap().rect().contains(label_pos):
                self._rubber_band.setGeometry(QRect(label_pos, QSize()))
                self._rubber_band.show()
                event.accept()
            else:
                # Click was outside the image area on the label, don't start drag
                self._drag_start_pos = None
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event) # Pass event up if not left-click or no image

    def mouseMoveEvent(self, event):
        """Handle mouse move events to update the zoom rectangle."""
        # Don't update rubber band if in picker mode
        if self._picker_mode_active:
            event.accept() # Consume event but do nothing
            return

        if self._rubber_band is not None and self._drag_start_pos is not None:
            # Update rubber band geometry based on current mouse position relative to label
            start_label_pos = self.image_label.mapFromParent(self._drag_start_pos)
            current_label_pos = self.image_label.mapFromParent(event.pos())
            self._rubber_band.setGeometry(QRect(start_label_pos, current_label_pos).normalized())
            event.accept()
        else:
             super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events to perform the zoom."""
        # Don't perform zoom if in picker mode or if drag didn't start properly
        if self._picker_mode_active or self._drag_start_pos is None:
            self._drag_start_pos = None # Ensure drag state is reset
            if self._rubber_band: self._rubber_band.hide()
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MouseButton.LeftButton and self._rubber_band is not None:
            self._rubber_band.hide()
            end_pos = event.pos()
            selection_rect_widget = QRect(self._drag_start_pos, end_pos).normalized()

            # Map selection rectangle from widget coordinates to label coordinates
            top_left_label = self.image_label.mapFromParent(selection_rect_widget.topLeft())
            bottom_right_label = self.image_label.mapFromParent(selection_rect_widget.bottomRight())
            selection_rect_label = QRect(top_left_label, bottom_right_label).normalized()

            # Reset drag state
            self._drag_start_pos = None

            # Ignore tiny selections
            if selection_rect_label.width() < 5 or selection_rect_label.height() < 5:
                event.accept()
                return

            # --- Calculate Zoom ---
            if self._pixmap is None or self._pixmap.isNull() or self._zoom_factor == 0:
                event.accept()
                return

            # 1. Map selection rectangle (label coords) to original image coordinates
            orig_img_x = selection_rect_label.left() / self._zoom_factor
            orig_img_y = selection_rect_label.top() / self._zoom_factor
            orig_img_w = selection_rect_label.width() / self._zoom_factor
            orig_img_h = selection_rect_label.height() / self._zoom_factor

            # 2. Calculate required zoom factor to fit this original rect into viewport
            viewport_size = self.scroll_area.viewport().size()
            if orig_img_w <= 0 or orig_img_h <= 0: # Check for non-positive dimensions
                event.accept()
                return

            zoom_x = viewport_size.width() / orig_img_w
            zoom_y = viewport_size.height() / orig_img_h
            new_zoom_factor = min(zoom_x, zoom_y)

            # Limit maximum zoom if necessary (e.g., 1000%)
            # max_zoom = 10.0
            # new_zoom_factor = min(new_zoom_factor, max_zoom)

            # 3. Update zoom and display (this resizes the label)
            self._zoom_factor = new_zoom_factor
            self._update_display() # This is crucial BEFORE calculating scroll position

            # 4. Calculate scroll position to center the zoomed area
            # Center of the selected area in original image coordinates
            center_orig_x = orig_img_x + orig_img_w / 2
            center_orig_y = orig_img_y + orig_img_h / 2

            # Position of this center in the *newly zoomed* label coordinates
            center_new_label_x = center_orig_x * self._zoom_factor
            center_new_label_y = center_orig_y * self._zoom_factor

            # Calculate desired scroll position to put this point in the viewport center
            scroll_x = center_new_label_x - viewport_size.width() / 2
            scroll_y = center_new_label_y - viewport_size.height() / 2

            # 5. Apply scroll position
            self.scroll_area.horizontalScrollBar().setValue(int(scroll_x))
            self.scroll_area.verticalScrollBar().setValue(int(scroll_y))

            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # --- Picker Mode Methods ---

    def enter_picker_mode(self):
        """Activates color picker mode."""
        if self._pixmap is None:
            print("Cannot enter picker mode: No image loaded.")
            return
        self._picker_mode_active = True
        self._update_cursor()
        print("Picker mode entered.")

    def exit_picker_mode(self):
        """Deactivates color picker mode."""
        self._picker_mode_active = False
        self._update_cursor()
        print("Picker mode exited.")

    def _update_cursor(self):
        """Sets the cursor based on the current mode."""
        if self._picker_mode_active:
            self.setCursor(Qt.CursorShape.CrossCursor) # Use crosshair for picking
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor) # Default arrow

    def _sample_color_at(self, widget_pos):
        """Samples color at the given widget position and emits signal."""
        if self._pixmap is None or not self._picker_mode_active:
            return

        # 1. Map widget position to label coordinates
        label_pos = self.image_label.mapFromParent(widget_pos)

        # 2. Check if click is within the displayed pixmap bounds
        current_pixmap = self.image_label.pixmap()
        if not current_pixmap or current_pixmap.isNull() or not current_pixmap.rect().contains(label_pos):
            print("Picker click outside image bounds.")
            self.exit_picker_mode() # Exit mode if clicked outside
            return

        # 3. Map label coordinates to original image coordinates
        original_x = int(label_pos.x() / self._zoom_factor)
        original_y = int(label_pos.y() / self._zoom_factor)

        # 4. Get the QImage from the original QPixmap to sample color
        original_image = self._pixmap.toImage()

        # 5. Ensure coordinates are within the original image bounds
        if 0 <= original_x < original_image.width() and 0 <= original_y < original_image.height():
            # 6. Sample the color
            color = original_image.pixelColor(original_x, original_y)
            rgb_tuple = (color.red(), color.green(), color.blue())
            print(f"Sampled color at ({original_x}, {original_y}): {rgb_tuple}")

            # 7. Emit the signal
            self.color_sampled.emit(rgb_tuple)

            # 8. Exit picker mode after successful sample
            self.exit_picker_mode()
        else:
            print(f"Calculated original coordinates ({original_x}, {original_y}) out of bounds.")
            self.exit_picker_mode() # Exit mode if calculation is wrong

# Example usage (for testing standalone)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    class DummyMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Image Viewer Test")
            self.viewer = ImageViewer()
            self.setCentralWidget(self.viewer)

            # Create a dummy image
            dummy_image = np.zeros((300, 400, 3), dtype=np.uint8)
            dummy_image[:, 0:200] = [255, 0, 0]  # Red half
            dummy_image[:, 200:400] = [0, 255, 0] # Green half
            dummy_image[100:200, 150:250] = [0, 0, 255] # Blue square

            self.viewer.set_image(dummy_image)
            self.setGeometry(200, 200, 500, 400)

    app = QApplication(sys.argv)
    mainWin = DummyMainWindow()
    mainWin.show()
    sys.exit(app.exec())