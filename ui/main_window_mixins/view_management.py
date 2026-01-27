from PyQt6.QtCore import pyqtSlot

from ...utils.logger import get_logger

logger = get_logger(__name__)

class ViewManagementMixin:
    """Mixin for MainWindow to handle view management operations."""

    def toggle_compare_view(self):
        """Toggle the image viewer between 'before' and 'after'."""
        if self.initial_converted_image is None:
            self.compare_action.setChecked(False)
            return

        if hasattr(self, "compare_slider"):
            self.image_viewer.set_compare_wipe_enabled(False)
            self.compare_slider.blockSignals(True)
            self.compare_slider.setValue(100)
            self.compare_slider.blockSignals(False)

        self.image_viewer.toggle_compare_mode()

        if self.image_viewer._display_mode == 'before':
            self.histogram_widget.update_histogram(self.initial_converted_image)
        else:
            current_adjusted = self._get_fully_adjusted_image(self.base_image, self.adjustment_panel.get_adjustments())
            if current_adjusted is not None:
                self.histogram_widget.update_histogram(current_adjusted)
            else:
                self.histogram_widget.update_histogram(self.base_image)

    def _on_compare_slider_pressed(self):
        self._compare_slider_is_dragging = True

    def _on_compare_slider_released(self):
        self._compare_slider_is_dragging = False
        if self._compare_slider_pending_value is not None:
            self._apply_compare_slider_value(int(self._compare_slider_pending_value), update_histogram=True)
            self._compare_slider_pending_value = None
        else:
            try:
                self._apply_compare_slider_value(int(self.compare_slider.value()), update_histogram=True)
            except Exception:
                logger.exception("Failed to update histogram on compare slider release")

    def _apply_pending_compare_slider_value(self):
        if self._compare_slider_pending_value is None:
            return

        update_hist = not self._compare_slider_is_dragging
        self._apply_compare_slider_value(
            int(self._compare_slider_pending_value),
            update_histogram=update_hist,
        )

        if not self._compare_slider_is_dragging:
            self._compare_slider_pending_value = None

    def _on_compare_slider_changed(self, value: int):
        """Throttle wipe-compare updates."""
        if self.initial_converted_image is None or self.base_image is None:
            return

        self._compare_slider_pending_value = int(value)
        if self._compare_slider_is_dragging:
            self._compare_slider_debounce_timer.start(self._compare_slider_debounce_ms)
            return

        self._apply_compare_slider_value(int(value), update_histogram=True)

    def _apply_compare_slider_value(self, value: int, *, update_histogram: bool):
        """Apply a compare slider value to the viewer."""
        v = int(value)
        if v <= 0:
            self.image_viewer.set_compare_wipe_enabled(False)
            self.image_viewer.set_display_mode('before')
        elif v >= 100:
            self.image_viewer.set_compare_wipe_enabled(False)
            self.image_viewer.set_display_mode('after')
        else:
            self.image_viewer.set_compare_wipe_enabled(True)
            self.image_viewer.set_compare_wipe_percent(v)

        if not update_histogram:
            return

        if v <= 0:
            self.histogram_widget.update_histogram(self.initial_converted_image)
        elif v >= 100:
            after_img = getattr(self.image_viewer, "_current_image", None)
            if after_img is not None:
                self.histogram_widget.update_histogram(after_img)
            else:
                self.histogram_widget.update_histogram(self.base_image)

    @pyqtSlot(str)
    def _update_status_view_mode(self, view_mode):
        """Slot to update the view mode label."""
        if view_mode == 'before':
            self.status_view_mode_label.setText(" View: Before ")
        else:
            self.status_view_mode_label.setText(" View: After ")
