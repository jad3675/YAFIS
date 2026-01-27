import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt6.QtCore import Qt, pyqtSlot, QMetaObject

from ...utils.logger import get_logger
from ...config import settings as app_settings

logger = get_logger(__name__)

class BatchProcessingMixin:
    """Mixin for MainWindow to handle batch processing operations."""

    def select_batch_output_directory(self):
        """Opens a dialog to select the output directory for batch processing."""
        start_dir = self._batch_output_dir if self._batch_output_dir else os.path.expanduser("~")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder for Batch Processing", start_dir)
        if folder_path:
            self._batch_output_dir = folder_path
            self.output_dir_label.setText(f" Output: ...{os.path.basename(folder_path)} ")
            self.output_dir_label.setToolTip(f"Output Directory: {folder_path}")
            self.update_ui_state()
        else:
            self.statusBar().showMessage("Output directory selection cancelled.", 2000)

    def start_batch_processing(self):
        """Initiates the batch processing workflow."""
        checked_files = self.filmstrip_widget.get_checked_image_paths()
        if not checked_files:
            QMessageBox.warning(self, "No Images Selected", "Please check the images in the filmstrip you want to process.")
            return

        if not self._batch_output_dir or not os.path.isdir(self._batch_output_dir):
            QMessageBox.warning(self, "Output Directory Not Set", "Please select a valid output directory using the 'Set Output Dir' button.")
            return

        current_adjustments = self.adjustment_panel.get_adjustments()
        active_preset_info = self._active_preset_details
        output_format = self.format_combo.currentText()
        
        quality_settings = {}
        if output_format == ".jpg":
            quality_settings['quality'] = self.quality_spinbox.value()
        elif output_format == ".png":
            quality_settings['png_compression'] = self.quality_spinbox.value()

        self.process_batch_action.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.statusBar().showMessage(f"Starting batch processing for {len(checked_files)} images...")

        self.batch_worker.set_inputs(
            checked_files,
            self._batch_output_dir,
            current_adjustments,
            active_preset_info,
            self.negative_converter,
            self.film_preset_manager,
            self.photo_preset_manager,
            output_format,
            quality_settings
        )

        QMetaObject.invokeMethod(self.batch_worker, "run", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(int, int)
    def _handle_batch_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.batch_progress_bar.setValue(percentage)
            self.statusBar().showMessage(f"Batch processing: {current}/{total} images...")
        else:
            self.batch_progress_bar.setValue(0)

    @pyqtSlot(list)
    def _handle_batch_finished(self, results):
        """Handle completion of the batch processing task."""
        self.batch_progress_bar.setVisible(False)
        self.update_ui_state()

        success_count = sum(1 for _, success, _ in results if success)
        fail_count = len(results) - success_count

        summary_message = f"Batch processing finished.\nSuccessfully processed: {success_count}\nFailed: {fail_count}"
        detailed_message = summary_message + "\n\nDetails:\n"
        for file_path, success, message in results:
            status = "OK" if success else "FAIL"
            detailed_message += f"- {os.path.basename(file_path)}: {status} ({message})\n"

        logger.info("Batch processing results:\n%s", detailed_message)
        QMessageBox.information(self, "Batch Processing Complete", summary_message)
        self.statusBar().showMessage("Batch processing complete.", 5000)

    @pyqtSlot(str)
    def _handle_batch_error(self, error_message):
        """Handle errors reported by the batch worker."""
        self.batch_progress_bar.setVisible(False)
        self.update_ui_state()
        QMessageBox.critical(self, "Batch Processing Error", f"An error occurred during batch processing:\n{error_message}")
        self.statusBar().showMessage(f"Batch processing error: {error_message}", 5000)

    @pyqtSlot(str)
    def _update_quality_widget_visibility(self, selected_format):
        """Show/hide and configure the quality spinbox based on selected format."""
        if selected_format == ".jpg":
            self.quality_label.setText(" Quality: ")
            self.quality_spinbox.setRange(1, 100)
            self.quality_spinbox.setToolTip("Set JPEG quality (1-100)")
            self.quality_spinbox.setValue(app_settings.UI_DEFAULTS.get("default_jpeg_quality", 95))
            self.quality_label.setVisible(True)
            self.quality_spinbox.setVisible(True)
        elif selected_format == ".png":
            self.quality_label.setText(" Compress: ")
            self.quality_spinbox.setRange(0, 9)
            self.quality_spinbox.setToolTip("Set PNG compression level (0-9)")
            self.quality_spinbox.setValue(app_settings.UI_DEFAULTS.get("default_png_compression", 6))
            self.quality_label.setVisible(True)
            self.quality_spinbox.setVisible(True)
        else:
            self.quality_label.setVisible(False)
            self.quality_spinbox.setVisible(False)
