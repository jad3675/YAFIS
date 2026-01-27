# Image display widget
import numpy as np
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy, QScrollArea, QRubberBand
from PyQt6.QtGui import QImage, QPixmap, QPainter
from PyQt6.QtCore import Qt, QSize, QPoint, QRect, pyqtSignal
from PyQt6.QtGui import QCursor # Added for cursor change

from ..utils.logger import get_logger
logger = get_logger(__name__)

class ImageViewer(QWidget):
    """Widget to display an image using QLabel, with zoom, pan, and color picking."""
    color_sampled = pyqtSignal(tuple) # Signal emitting (r, g, b) tuple
    display_mode_changed = pyqtSignal(str) # Signal emitting 'before' or 'after'

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None         # The current/after pixmap
        self._before_pixmap = None  # The stored before pixmap
        self._zoom_factor = 1.0
        self._scale_increment = 1.25
        self._rubber_band = None
        self._drag_start_pos = None
        self._picker_mode_active = False
        self._display_mode = 'after' # 'after' or 'before'

        # Wipe compare state (0..100). 0=all before, 100=all after.
        self._compare_wipe_percent = 100
        self._compare_wipe_enabled = False

        # Cache scaled pixmaps so wipe slider doesn't rescale on every tick.
        self._scaled_cache_key = None
        self._scaled_after_cache = None
        self._scaled_before_cache = None

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
            self._before_pixmap = None # Clear before state too
            self._display_mode = 'after' # Reset mode
            self.display_mode_changed.emit(self._display_mode) # Emit signal
            self.image_label.clear()
            self.image_label.setText("No Image Loaded")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._invalidate_scaled_cache()
            self._update_display() # Ensure label size is reset
            return

        if not isinstance(image_np, np.ndarray) or image_np.dtype != np.uint8:
            logger.error("Image must be a NumPy array of type uint8.")
            # Optionally clear or show an error message
            self._pixmap = None
            self._before_pixmap = None # Clear before state too
            self._display_mode = 'after' # Reset mode
            self.display_mode_changed.emit(self._display_mode) # Emit signal
            self.image_label.clear()
            self.image_label.setText("Invalid Image Data")
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._update_display()
            return

        try:
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            if q_image.isNull():
                logger.error("Failed to create QImage from NumPy array.")
                self._pixmap = None
                self._before_pixmap = None # Clear before state too
                self._display_mode = 'after' # Reset mode
                self.display_mode_changed.emit(self._display_mode) # Emit signal
                self.image_label.clear()
                self.image_label.setText("Image Conversion Error")
                self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._update_display()
                return

            self._pixmap = QPixmap.fromImage(q_image) # Store the current/after pixmap
            # self._before_pixmap = None # REMOVED: Do not clear before state when setting after image
            self._display_mode = 'after' # Ensure mode is 'after' for new image
            self.display_mode_changed.emit(self._display_mode) # Emit signal for new image
            self._invalidate_scaled_cache()
            self.fit_to_window() # Fit new image to window by default

        except Exception as e:
            logger.exception("Error converting NumPy array to QPixmap")
            self._pixmap = None
            self._before_pixmap = None # Clear before state too
            self._display_mode = 'after' # Reset mode
            self.display_mode_changed.emit(self._display_mode) # Emit signal
            self.image_label.clear()
            self.image_label.setText("Display Error")
            self._update_display()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


    def _invalidate_scaled_cache(self):
        self._scaled_cache_key = None
        self._scaled_after_cache = None
        self._scaled_before_cache = None

    def _get_scaled_pixmaps(self, *, require_before: bool):
        """Return (scaled_after, scaled_before|None) using a simple cache."""
        after_pixmap = self._pixmap
        before_pixmap = self._before_pixmap

        if after_pixmap is None or after_pixmap.isNull():
            return None, None

        if require_before and (before_pixmap is None or before_pixmap.isNull()):
            return None, None

        # Scale target is based on after pixmap size + zoom (same logic as before).
        original_size = after_pixmap.size()
        scaled_size = QSize(
            int(original_size.width() * self._zoom_factor),
            int(original_size.height() * self._zoom_factor),
        )
        scaled_size.setWidth(max(1, scaled_size.width()))
        scaled_size.setHeight(max(1, scaled_size.height()))

        cache_key = (
            int(after_pixmap.cacheKey()),
            int(before_pixmap.cacheKey()) if before_pixmap is not None else 0,
            float(self._zoom_factor),
            int(scaled_size.width()),
            int(scaled_size.height()),
            bool(require_before),
        )

        if (
            cache_key == self._scaled_cache_key
            and self._scaled_after_cache is not None
            and (not require_before or self._scaled_before_cache is not None)
        ):
            return self._scaled_after_cache, self._scaled_before_cache

        scaled_after = after_pixmap.scaled(
            scaled_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        scaled_before = None
        if require_before:
            scaled_before = before_pixmap.scaled(
                scaled_after.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        self._scaled_cache_key = cache_key
        self._scaled_after_cache = scaled_after
        self._scaled_before_cache = scaled_before
        return scaled_after, scaled_before

    def _update_display(self):
        """Updates the QLabel pixmap scaled by zoom.

        If wipe compare is enabled and a before-image exists, renders a composite:
        left side = before, right side = after (based on _compare_wipe_percent).
        """
        after_pixmap = self._pixmap
        before_pixmap = self._before_pixmap

        # Determine what we need to display (single or composite)
        use_wipe = bool(self._compare_wipe_enabled and after_pixmap is not None and before_pixmap is not None)

        if not use_wipe:
            pixmap_to_display = before_pixmap if self._display_mode == 'before' else after_pixmap
            if pixmap_to_display is None or pixmap_to_display.isNull():
                self.image_label.clear()
                if self._pixmap is None:
                    # Show appropriate message if trying to view 'before' but it's not set
                    if self._display_mode == 'before' and self._before_pixmap is None:
                        self.image_label.setText("No 'Before' Image Stored")
                    else:
                        self.image_label.setText("No Image Loaded")
                    self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                # Ensure label takes minimal size when cleared
                self.image_label.resize(0, 0)
                return

            # Scale once (cached) to avoid repeated rescale work.
            # For single-image mode, we can reuse the same cache but only need "after" scaling.
            if self._display_mode == 'before':
                # Build a scaled "before" using the same sizing as "after" (consistent zoom feel).
                # This still caches both, so repeated toggles don't rescale.
                scaled_after, scaled_before = self._get_scaled_pixmaps(require_before=True)
                scaled_pixmap = scaled_before
            else:
                scaled_after, _ = self._get_scaled_pixmaps(require_before=False)
                scaled_pixmap = scaled_after

            if scaled_pixmap is None or scaled_pixmap.isNull():
                return

            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            return

        # --- Wipe compare composite render ---
        # Use cached scaled pixmaps to avoid expensive rescale on every slider tick.
        scaled_after, scaled_before = self._get_scaled_pixmaps(require_before=True)
        if scaled_after is None or scaled_before is None:
            return

        composite = QPixmap(scaled_after.size())
        composite.fill(Qt.GlobalColor.black)

        p = QPainter(composite)
        try:
            if not p.isActive():
                # Avoid QPainter warnings / undefined rendering.
                logger.warning("QPainter not active; skipping wipe composite render.")
                return

            # Start with before everywhere
            p.drawPixmap(0, 0, scaled_before)

            # Then draw after on the right side according to the wipe percentage
            wipe_pct = float(max(0, min(100, self._compare_wipe_percent)))
            wipe_x = int((wipe_pct / 100.0) * scaled_after.width())

            if wipe_x > 0:
                after_rect = QRect(0, 0, wipe_x, scaled_after.height())
                p.setClipRect(after_rect)
                p.drawPixmap(0, 0, scaled_after)
                p.setClipping(False)

            # Optional: divider line for visibility
            divider_x = max(0, min(scaled_after.width() - 1, wipe_x))
            p.setPen(Qt.GlobalColor.white)
            p.drawLine(divider_x, 0, divider_x, scaled_after.height())
        finally:
            if p.isActive():
                p.end()

        self.image_label.setPixmap(composite)
        self.image_label.resize(composite.size())

    # --- Comparison Methods ---

    def set_display_mode(self, mode: str):
        """Explicitly set display mode to 'before' or 'after' and refresh."""
        if mode not in ('before', 'after'):
            return

        # If wipe compare is enabled, ignore explicit modes (wipe controls view).
        if self._compare_wipe_enabled:
            return

        if mode == 'before' and self._before_pixmap is None:
            logger.debug("Cannot switch to 'before': no 'before' image stored.")
            return

        if self._display_mode != mode:
            self._display_mode = mode
            self.display_mode_changed.emit(self._display_mode)

        self._update_display()

    def set_before_image(self, image_np):
        """Stores the current state as the 'before' image for comparison."""
        if image_np is None:
            self._before_pixmap = None
            self._invalidate_scaled_cache()
            logger.debug("Cleared 'before' image.")
            return

        # Convert NumPy to QPixmap (similar to set_image)
        try:
            height, width, channel = image_np.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_np.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            if not q_image.isNull():
                self._before_pixmap = QPixmap.fromImage(q_image)
                self._invalidate_scaled_cache()
                logger.debug("Stored 'before' image for comparison.")
            else:
                self._before_pixmap = None
                logger.error("Error creating QImage for 'before' state.")
        except Exception as e:
            self._before_pixmap = None
            logger.exception("Error storing 'before' image")

    def toggle_compare_mode(self):
        """Toggles the display between 'before' and 'after' states.

        Note: if wipe compare is enabled, toggling is ignored to avoid conflicting modes.
        """
        if self._pixmap is None: # No image loaded
            return
        if self._compare_wipe_enabled:
            return

        if self._display_mode == 'after':
            if self._before_pixmap is not None:
                self._display_mode = 'before'
                logger.debug("Displaying 'before' image.")
            else:
                logger.debug("Cannot switch to 'before': no 'before' image stored.")
                return # Stay in 'after' mode
        else: # Currently 'before'
            self._display_mode = 'after'
            logger.debug("Displaying 'after' image.")

        self._update_display()
        self.display_mode_changed.emit(self._display_mode) # Emit signal after mode change

    def has_before_image(self):
        """Returns True if a 'before' image is currently stored."""
        return self._before_pixmap is not None

    # --- Wipe Compare API (toolbar slider drives this) ---

    def set_compare_wipe_enabled(self, enabled: bool):
        """Enable/disable wipe compare composite rendering."""
        enabled = bool(enabled)

        if enabled == self._compare_wipe_enabled:
            return

        self._compare_wipe_enabled = enabled
        if self._compare_wipe_enabled:
            # Wipe view implies "after" is visible somewhere.
            self._display_mode = 'after'
            self.display_mode_changed.emit(self._display_mode)

        self._update_display()

    def set_compare_wipe_percent(self, percent: int):
        """Set wipe position: 0..100 (0=before, 100=after)."""
        self._compare_wipe_percent = int(max(0, min(100, percent)))
        if self._compare_wipe_enabled:
            self._update_display()

    # Removed resizeEvent handler - resizing the window no longer changes zoom level.
    # The QScrollArea handles the view of the potentially larger QLabel.

    # --- Zoom Methods ---

    def zoom_in(self):
        """Zooms in on the image."""
        self._zoom_factor *= self._scale_increment
        self._invalidate_scaled_cache()
        self._update_display()

    def zoom_out(self):
        """Zooms out of the image."""
        # Set a minimum zoom level (e.g., 1% of original size)
        min_zoom = 0.01
        self._zoom_factor /= self._scale_increment
        if self._zoom_factor < min_zoom:
            self._zoom_factor = min_zoom
        self._invalidate_scaled_cache()
        self._update_display()

    def reset_zoom(self):
        """Resets the zoom to 100%."""
        self._zoom_factor = 1.0
        self._invalidate_scaled_cache()
        self._update_display()

    def fit_to_window(self):
         """Adjusts zoom factor to fit the image within the current window size."""
         pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
         if pixmap_to_display is None or pixmap_to_display.isNull():
             return

         pixmap_size = pixmap_to_display.size()
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

         self._invalidate_scaled_cache()
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
        pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
        if self._picker_mode_active and event.button() == Qt.MouseButton.LeftButton and pixmap_to_display is not None:
            self._sample_color_at(event.pos())
            event.accept()
            return # Don't proceed with zoom/pan

        # Handle zoom rectangle start
        pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
        if event.button() == Qt.MouseButton.LeftButton and pixmap_to_display is not None:
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
            pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
            if pixmap_to_display is None or pixmap_to_display.isNull() or self._zoom_factor == 0:
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
            self._invalidate_scaled_cache()
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
        pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
        if pixmap_to_display is None:
            logger.debug("Cannot enter picker mode: no image loaded.")
            return
        self._picker_mode_active = True
        self._update_cursor()
        logger.debug("Picker mode entered.")

    def exit_picker_mode(self):
        """Deactivates color picker mode."""
        self._picker_mode_active = False
        self._update_cursor()
        logger.debug("Picker mode exited.")

    def _update_cursor(self):
        """Sets the cursor based on the current mode."""
        if self._picker_mode_active:
            self.setCursor(Qt.CursorShape.CrossCursor) # Use crosshair for picking
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor) # Default arrow

    def _sample_color_at(self, widget_pos):
        """Samples color at the given widget position and emits signal."""
        pixmap_to_display = self._before_pixmap if self._display_mode == 'before' else self._pixmap
        if pixmap_to_display is None or not self._picker_mode_active:
            return

        # 1. Map widget position to label coordinates
        label_pos = self.image_label.mapFromParent(widget_pos)

        # 2. Check if click is within the displayed pixmap bounds
        current_pixmap = self.image_label.pixmap()
        if not current_pixmap or current_pixmap.isNull() or not current_pixmap.rect().contains(label_pos):
            logger.debug("Picker click outside image bounds.")
            self.exit_picker_mode() # Exit mode if clicked outside
            return

        # 3. Map label coordinates to original image coordinates
        original_x = int(label_pos.x() / self._zoom_factor)
        original_y = int(label_pos.y() / self._zoom_factor)

        # 4. Get the QImage from the original QPixmap to sample color
        # Sample from the *currently displayed* pixmap's original QImage
        original_image = pixmap_to_display.toImage()

        # 5. Ensure coordinates are within the original image bounds
        if 0 <= original_x < original_image.width() and 0 <= original_y < original_image.height():
            # 6. Sample the color
            color = original_image.pixelColor(original_x, original_y)
            rgb_tuple = (color.red(), color.green(), color.blue())
            logger.debug("Sampled color at (%s, %s): %s", original_x, original_y, rgb_tuple)

            # 7. Emit the signal
            self.color_sampled.emit(rgb_tuple)

            # 8. Exit picker mode after successful sample
            self.exit_picker_mode()
        else:
            logger.debug("Calculated original coordinates (%s, %s) out of bounds.", original_x, original_y)
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