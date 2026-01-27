import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, Qt
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ImageProcessingWorker(QObject):
    """Worker Class for Background Processing of image adjustments."""
    finished = pyqtSignal(object) # Emits processed image (numpy array)
    error = pyqtSignal(str)       # Emits error message

    def __init__(self, processing_func):
        super().__init__()
        self.processing_func = processing_func # Store the function passed from MainWindow
        self._is_running = False

    @pyqtSlot(object, dict)
    def run(self, base_image, adjustments):
        """Calls the main processing function passed during initialization."""
        if self._is_running:
            return
        self._is_running = True

        try:
            processed_image = self.processing_func(base_image, adjustments)
            if processed_image is None:
                raise ValueError("Processing function returned None.")
            self.finished.emit(processed_image)
        except Exception as e:
            logger.exception("Error during background processing call")
            self.error.emit(f"Processing failed: {e}")
        finally:
            self._is_running = False

class InitialConversionWorker(QObject):
    """Worker for Initial Negative Conversion."""
    finished = pyqtSignal(object, str) # Emits (converted_image, mask_classification)
    progress = pyqtSignal(int, int) # Emits (current_step, total_steps)
    error = pyqtSignal(str)         # Emits error message
    conversion_requested = pyqtSignal(object, object) # raw_image, override_type (object allows None)

    def __init__(self, conversion_func):
        super().__init__()
        self.conversion_func = conversion_func # Should be NegativeConverter.convert
        self._is_running = False

    @pyqtSlot(object, object)
    def run(self, raw_image, override_type=None):
        """Calls the initial conversion function."""
        if self._is_running:
            return
        self._is_running = True
        try:
            def progress_callback(step, total):
                self.progress.emit(step, total)

            result_tuple = self.conversion_func(raw_image,
                                                progress_callback=progress_callback,
                                                override_mask_classification=override_type)
            if result_tuple is None or result_tuple[0] is None:
                 raise ValueError("Initial conversion function failed or returned None image.")
            converted_image, mask_classification = result_tuple
            self.finished.emit(converted_image, mask_classification)
        except Exception as e:
            logger.exception("Error during initial conversion call")
            self.error.emit(f"Initial conversion failed: {e}")
        finally:
            self._is_running = False

class AutoToneWorker(QObject):
    """Worker to compute Auto Tone parameters off the UI thread."""
    finished = pyqtSignal(dict)  # emits tone params dict
    error = pyqtSignal(str)

    def __init__(self, compute_func):
        super().__init__()
        self.compute_func = compute_func
        self._is_running = False

    @pyqtSlot(object)
    def run(self, base_image):
        if self._is_running:
            return
        self._is_running = True
        try:
            params = self.compute_func(base_image)
            if not isinstance(params, dict):
                raise ValueError("Auto Tone returned invalid params.")
            self.finished.emit(params)
        except Exception as e:
            logger.exception("Error during Auto Tone computation")
            self.error.emit(f"Auto Tone failed: {e}")
        finally:
            self._is_running = False

class BatchProcessingWorker(QObject):
    """Worker object to run batch processing in a separate thread."""
    finished = pyqtSignal(list) # Emits list of (file_path, success, message) tuples
    progress = pyqtSignal(int, int) # Emits (current_processed_count, total_files)
    error = pyqtSignal(str) # Emits error message if setup fails

    def __init__(self, processing_func, parent=None):
        super().__init__(parent)
        self.processing_func = processing_func
        self._is_running = False
        self._inputs = None

    def set_inputs(self, file_paths, output_dir, adjustments_dict, active_preset_info,
                   converter, film_preset_manager, photo_preset_manager,
                   output_format, quality_settings):
        """Set the inputs required for the batch run."""
        self._inputs = (file_paths, output_dir, adjustments_dict, active_preset_info,
                        converter, film_preset_manager, photo_preset_manager,
                        output_format, quality_settings)

    @pyqtSlot()
    def run(self):
        """Execute the batch processing function."""
        if self._is_running or not self._inputs:
            self.error.emit("Worker is already running or inputs not set.")
            return

        self._is_running = True
        file_paths, output_dir, adjustments_dict, active_preset_info, \
            converter, film_preset_manager, photo_preset_manager, \
            output_format, quality_settings = self._inputs

        try:
            total_files = len(file_paths)
            results = self.processing_func(
                file_paths, output_dir, adjustments_dict, active_preset_info,
                converter, film_preset_manager, photo_preset_manager,
                output_format, quality_settings
            )
            processed_count = 0
            final_results = []
            for res in results:
                 processed_count += 1
                 self.progress.emit(processed_count, total_files)
                 final_results.append(res)

            self.finished.emit(final_results)

        except Exception as e:
            logger.exception("Error during batch processing call")
            self.error.emit(f"Batch processing failed: {e}")
        finally:
            self._is_running = False
            self._inputs = None
