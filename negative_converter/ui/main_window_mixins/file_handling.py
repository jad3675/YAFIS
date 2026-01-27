import os
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QApplication
from PyQt6.QtCore import Qt

from ...utils.logger import get_logger

logger = get_logger(__name__)

class FileHandlingMixin:
    """Mixin for MainWindow to handle file-related operations."""
    
    def open_image(self):
        """Open an image file using a file dialog."""
        file_filter = "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Negative Image", "", file_filter)

        if file_path:
            self.statusBar().showMessage(f"Loading image: {os.path.basename(file_path)}...")
            QApplication.processEvents() # Allow UI to update
            try:
                # Store raw loaded image first, unpack all return values
                self.raw_loaded_image, original_mode, file_size = self.conversion_service.load_image(file_path)
                if self.raw_loaded_image is None:
                    raise ValueError("Image loader returned None.")

                self.current_file_path = file_path
                self._current_file_size = file_size # Use file size returned by loader
                self._current_original_mode = original_mode
                self._current_negative_type = None # Reset detected type

                # Update status bar immediately with file info
                self._update_status_filename(self.current_file_path)
                self._update_status_size(self._current_file_size)
                self._update_status_mode(self._current_original_mode)
                self._update_status_neg_type(None) # Clear neg type initially

                # Reset UI elements before conversion
                self.adjustment_panel.reset_adjustments()
                self.image_viewer.set_image(None)
                self.previous_base_image = None
                self.compare_action.setChecked(False)
                self.histogram_widget.clear_histogram()

                # --- Trigger Initial Conversion in Worker Thread ---
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.statusBar().showMessage("Starting initial negative conversion...", 0)
                self._is_converting_initial = True
                self.update_ui_state()
                self.initial_conversion_requested.emit(self.raw_loaded_image, None)

            except Exception as e:
                logger.exception("Failed to open image")
                QMessageBox.critical(self, "Error", f"Failed to load or start conversion for image:\n{file_path}\n\nError: {e}")
                self.statusBar().showMessage(f"Error loading image: {e}", 5000)
                self.raw_loaded_image = None
                self.current_file_path = None
                self._is_converting_initial = False
                QApplication.restoreOverrideCursor()
                self.update_ui_state()

    def save_image_as(self):
        """Save the currently processed image to a new file."""
        if self.base_image is None:
            QMessageBox.warning(self, "No Image", "No image is currently loaded or processed.")
            return

        if self.current_file_path:
            base, _ = os.path.splitext(os.path.basename(self.current_file_path))
            suggested_name = f"{base}_positive.jpg"
        else:
            suggested_name = "positive_image.jpg"

        save_filter = "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png);;WebP Image (*.webp);;TIFF Image (*.tif *.tiff)"
        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Positive Image As", suggested_name, save_filter)

        if file_path:
            try:
                self.statusBar().showMessage("Applying final adjustments...")
                QApplication.processEvents()
                final_image_to_save = self._get_fully_adjusted_image(self.base_image, self.adjustment_panel.get_adjustments())

                if final_image_to_save is None:
                    raise ValueError("Failed to get final adjusted image for saving.")

                self.statusBar().showMessage(f"Saving image to {os.path.basename(file_path)}...")
                QApplication.processEvents()

                quality_params = {}
                ext = os.path.splitext(file_path)[1].lower()

                from ...config import settings as app_settings
                if ext in ['.jpg', '.jpeg']:
                    quality = app_settings.UI_DEFAULTS.get("default_jpeg_quality", 95)
                    quality_params['quality'] = quality
                elif ext == '.png':
                    compression = app_settings.UI_DEFAULTS.get("default_png_compression", 6)
                    quality_params['png_compression'] = compression
                elif ext == '.webp':
                    quality = app_settings.UI_DEFAULTS.get("default_jpeg_quality", 90)
                    quality_params['quality'] = quality
                elif ext in ['.tif', '.tiff']:
                    quality_params['compression'] = 'tiff_lzw'

                success = self.conversion_service.save_image(final_image_to_save, file_path, **quality_params)
                if success:
                    self.statusBar().showMessage(f"Image saved successfully to {file_path}", 5000)
                else:
                    raise ValueError("Failed to save image using image_saver.")

            except Exception as e:
                logger.exception("Failed to save image")
                QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")
                self.statusBar().showMessage(f"Error saving image: {e}", 5000)

    def open_batch_dialog(self):
        """Opens a dialog to select a folder for batch processing."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Negatives")
        if folder_path:
            self.statusBar().showMessage(f"Scanning folder: {folder_path}...")
            QApplication.processEvents()
            try:
                image_files = self._find_image_files(folder_path)
                if image_files:
                    self.filmstrip_widget.add_images(image_files)
                    self.filmstrip_dock.setVisible(True)
                    self.statusBar().showMessage(f"Loaded {len(image_files)} images for batch processing.", 3000)
                else:
                    self.filmstrip_dock.setVisible(False)
                    self.statusBar().showMessage(f"No supported image files found in {folder_path}.", 3000)
                    QMessageBox.information(self, "No Images Found", f"No supported image files were found in the selected folder:\n{folder_path}")
                self.update_ui_state()
            except Exception as e:
                logger.exception("Failed to scan batch folder")
                QMessageBox.critical(self, "Error Scanning Folder", f"An error occurred while scanning the folder:\n{e}")
                self.statusBar().showMessage(f"Error scanning folder: {e}", 5000)
                self.filmstrip_dock.setVisible(False)
                self.update_ui_state()

    def _find_image_files(self, folder_path):
        """Recursively find supported image files in a folder."""
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')
        image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    image_files.append(os.path.join(root, file))
        return sorted(image_files)

    def handle_filmstrip_preview(self, file_path):
        """Load and display the selected image from the filmstrip for preview."""
        if self._is_converting_initial:
            return
            
        logger.debug("Filmstrip preview requested: %s", file_path)
        self.statusBar().showMessage(f"Loading preview: {os.path.basename(file_path)}...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        
        try:
            image, original_mode, file_size = self.conversion_service.load_image(file_path)
            if image is not None:
                self.current_file_path = file_path
                self.raw_loaded_image = image
                self._current_file_size = file_size
                self._current_original_mode = original_mode
                self._current_negative_type = None
                
                self._update_status_filename(file_path)
                self._update_status_size(self._current_file_size)
                self._update_status_mode(self._current_original_mode)
                self._update_status_neg_type(None)
                
                self._run_initial_conversion(image)
            else:
                self.statusBar().showMessage(f"Failed to load preview: {os.path.basename(file_path)}", 5000)
        except Exception as e:
            logger.exception("Failed to load filmstrip preview")
            self.statusBar().showMessage(f"Error loading preview: {e}", 5000)
        finally:
            QApplication.restoreOverrideCursor()
