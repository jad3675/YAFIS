# Main application window
import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMenu,
                             QStatusBar, QFileDialog, QMessageBox, QDockWidget, QApplication, # Ensure QApplication is imported
                             QToolBar, QPushButton, QProgressBar, QLabel, QComboBox, QSpinBox, QWidgetAction,
                             QInputDialog, QSlider)
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtCore import Qt, QSize, QThread, QObject, pyqtSignal, pyqtSlot, QMetaObject, QTimer
import concurrent.futures
import math

from ..utils.logger import get_logger
logger = get_logger(__name__)

# Import UI components
from .image_viewer import ImageViewer
from .adjustment_panel import AdjustmentPanel
from .preset_panel import FilmPresetPanel
from .photo_preset_panel import PhotoPresetPanel
from .histogram_widget import HistogramWidget
from .filmstrip_widget import BatchFilmstripWidget
from .settings_dialog import SettingsDialog
from ..config import settings as app_settings # Import settings to call reload

# Import IO and Processing components
from negative_converter.processing.adjustments import apply_all_adjustments # Import the new function using absolute path
# Standard imports assuming package structure is respected
from ..io import image_loader, image_saver
from ..processing import NegativeConverter, FilmPresetManager, PhotoPresetManager, ImageAdjustments
from ..processing.adjustments import AdvancedAdjustments
from ..processing.batch import process_batch_with_adjustments
from ..services.conversion_service import ConversionService, PresetInfo


# --- Worker Class for Background Processing ---
class ImageProcessingWorker(QObject):
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

# --- Worker for Initial Negative Conversion ---
class InitialConversionWorker(QObject):
    finished = pyqtSignal(object, str) # Emits (converted_image, mask_classification)
    progress = pyqtSignal(int, int) # Emits (current_step, total_steps)
    error = pyqtSignal(str)         # Emits error message
    # Signal to request conversion with optional override - THIS WAS ALREADY ADDED, KEEPING
    conversion_requested = pyqtSignal(object, object) # raw_image, override_type (object allows None)

    def __init__(self, conversion_func):
        super().__init__()
        self.conversion_func = conversion_func # Should be MainWindow._run_initial_conversion
        self._is_running = False

    # Slot now takes raw_image and optional override_type
    # Slot signature already updated, KEEPING
    @pyqtSlot(object, object)
    def run(self, raw_image, override_type=None):
        """Calls the initial conversion function."""
        if self._is_running:
            return
        self._is_running = True
        try:
            # Define the callback function to emit the progress signal
            def progress_callback(step, total):
                self.progress.emit(step, total)

            # Pass the callback to the conversion function, which now returns (image, classification)
            # Pass override_type to the backend conversion function - THIS WAS ALREADY ADDED, KEEPING
            result_tuple = self.conversion_func(raw_image,
                                                progress_callback=progress_callback,
                                                override_mask_classification=override_type)
            if result_tuple is None or result_tuple[0] is None:
                 # Handle case where conversion returns None or (None, classification)
                 raise ValueError("Initial conversion function failed or returned None image.")
            converted_image, mask_classification = result_tuple
            self.finished.emit(converted_image, mask_classification) # Emit both results
        except Exception as e:
            logger.exception("Error during initial conversion call")
            self.error.emit(f"Initial conversion failed: {e}")
        finally:
            self._is_running = False

# --- Auto Tone Worker ---
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


# --- Batch Processing Worker ---
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

    # Updated signature to accept preset info, managers, format, and quality
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
        # Unpack updated inputs
        file_paths, output_dir, adjustments_dict, active_preset_info, \
            converter, film_preset_manager, photo_preset_manager, \
            output_format, quality_settings = self._inputs

        try:
            # For proper GUI progress, backend would need modification.
            # Current simulation based on results count remains.
            total_files = len(file_paths)
            # Pass format and quality to the backend function
            # Pass all necessary info, including preset details and managers, to the backend function
            results = self.processing_func(
                file_paths, output_dir, adjustments_dict, active_preset_info,
                converter, film_preset_manager, photo_preset_manager,
                output_format, quality_settings
            )
            # Simulate progress based on results collected
            processed_count = 0
            final_results = []
            for res in results: # Assuming results are yielded or returned progressively
                 processed_count += 1
                 self.progress.emit(processed_count, total_files)
                 final_results.append(res)

            self.finished.emit(final_results)

        except Exception as e:
            logger.exception("Error during batch processing call")
            self.error.emit(f"Batch processing failed: {e}")
        finally:
            self._is_running = False
            self._inputs = None # Clear inputs after run

# --- Main Window Class ---
class MainWindow(QMainWindow):
    """Main application window."""
    # Signal to request processing in the worker thread
    processing_requested = pyqtSignal(object, dict)

    # Signals for initial conversion worker
    initial_conversion_requested = pyqtSignal(object, object) # raw_image, override_type
    initial_conversion_finished = pyqtSignal(object, str) # Emits (converted_image, mask_classification)
    initial_conversion_error = pyqtSignal(str)

    # Signal to request Auto Tone computation in its worker thread
    _auto_tone_requested = pyqtSignal(object)


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Negative-to-Positive Converter")
        self.setGeometry(100, 100, 1200, 800)

        # --- Status Info ---
        self._current_file_size = None
        self._current_original_mode = None
        self._current_negative_type = None

        # --- Image Data ---
        self.raw_loaded_image = None # Store the original loaded image before conversion
        self.initial_converted_image = None
        self.base_image = None
        self._is_converting_initial = False # Flag for initial conversion status

        self.previous_base_image = None
        self.current_file_path = None
        self._pending_adjustments = None
        self._request_pending = False
        self._batch_output_dir = None
        self._active_preset_details = None # Stores details of the last *applied* preset for batch

        # --- Processing Engines ---
        self.negative_converter = NegativeConverter(film_profile="C41")
        self.film_preset_manager = FilmPresetManager()
        self.photo_preset_manager = PhotoPresetManager()
        self.basic_adjuster = ImageAdjustments()
        self.advanced_adjuster = AdvancedAdjustments()

        # Ensure the adjustment pipeline uses the same preset manager instances as the UI.
        try:
            from ..processing.adjustments import set_film_preset_manager, set_photo_preset_manager
            set_film_preset_manager(self.film_preset_manager)
            set_photo_preset_manager(self.photo_preset_manager)
        except Exception:
            logger.exception("Failed to inject preset managers into adjustments pipeline")

        # --- Service Facade (UI -> processing/IO) ---
        self.conversion_service = ConversionService(
            converter=self.negative_converter,
            film_preset_manager=self.film_preset_manager,
            photo_preset_manager=self.photo_preset_manager,
        )

        # --- Background Processing Thread ---
        self.processing_thread = QThread(self)
        # --- Auto Tone Thread ---
        self.auto_tone_thread = QThread(self)
        # --- Initial Conversion Thread ---
        self.initial_conversion_thread = QThread(self)
        # Pass the actual conversion method of the instance
        self.initial_conversion_worker = InitialConversionWorker(self.negative_converter.convert)
        self.initial_conversion_worker.moveToThread(self.initial_conversion_thread)
        # Connect the new signal to the worker's run slot
        # Connection already updated, KEEPING
        self.initial_conversion_requested.connect(self.initial_conversion_worker.run)
        self.initial_conversion_worker.finished.connect(self._handle_initial_conversion_finished)
        self.initial_conversion_worker.progress.connect(self._handle_initial_conversion_progress) # Connect progress signal
        self.initial_conversion_worker.error.connect(self._handle_initial_conversion_error)
        self.initial_conversion_thread.start()

        self.processing_worker = ImageProcessingWorker(self._get_fully_adjusted_image)
        self.processing_worker.moveToThread(self.processing_thread)
        self.processing_requested.connect(self.processing_worker.run)
        self.processing_worker.finished.connect(self._handle_processing_finished)
        self.processing_worker.error.connect(self._handle_processing_error)
        self.processing_thread.start()

        # Auto Tone worker setup
        self._is_auto_tone_running = False
        self.auto_tone_worker = AutoToneWorker(AdvancedAdjustments.calculate_auto_tone_params)
        self.auto_tone_worker.moveToThread(self.auto_tone_thread)
        self._auto_tone_requested.connect(self.auto_tone_worker.run, Qt.ConnectionType.QueuedConnection)
        self.auto_tone_worker.finished.connect(self._handle_auto_tone_finished)
        self.auto_tone_worker.error.connect(self._handle_auto_tone_error)
        self.auto_tone_thread.start()

        # --- Batch Processing Thread (Single instance) ---
        self.batch_thread = QThread(self)
        self.batch_worker = BatchProcessingWorker(process_batch_with_adjustments)
        self.batch_worker.moveToThread(self.batch_thread)
        self.batch_worker.finished.connect(self._handle_batch_finished)
        self.batch_worker.progress.connect(self._handle_batch_progress)
        self.batch_worker.error.connect(self._handle_batch_error)
        self.batch_thread.start()

        # --- UI Components ---
        self.setup_ui()
        self.create_actions()
        self.create_menus()
        self.create_status_bar()
        self.create_batch_toolbar() # Create the batch toolbar
        self.create_view_toolbar()  # Create the view controls toolbar
        self.create_toolbars()      # Create other standard toolbars (if any)

        # Compare slider throttling state
        self._compare_slider_debounce_ms = 50
        self._compare_slider_debounce_timer = QTimer(self)
        self._compare_slider_debounce_timer.setSingleShot(True)
        self._compare_slider_pending_value = None
        self._compare_slider_is_dragging = False
        self._compare_slider_debounce_timer.timeout.connect(self._apply_pending_compare_slider_value)

        # Connect signals from panels
        self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)
        self.adjustment_panel.awb_requested.connect(self.handle_auto_wb)
        self.adjustment_panel.auto_level_requested.connect(self.handle_auto_level)
        self.adjustment_panel.auto_color_requested.connect(self.handle_auto_color)
        self.adjustment_panel.auto_tone_requested.connect(self.handle_auto_tone)
        self.film_preset_panel.preview_requested.connect(self.handle_preset_preview)
        self.film_preset_panel.apply_requested.connect(self.handle_preset_apply)
        self.photo_preset_panel.preview_requested.connect(self.handle_preset_preview)
        self.photo_preset_panel.apply_requested.connect(self.handle_preset_apply)
        self.filmstrip_widget.preview_requested.connect(self.handle_filmstrip_preview) # For single-click preview
        self.filmstrip_widget.checked_items_changed.connect(self.update_ui_state) # For enabling batch button

        # Connect WB Picker signals
        self.adjustment_panel.wb_picker_requested.connect(self.handle_wb_picker_request)
        self.image_viewer.color_sampled.connect(self.handle_color_sampled)
        self.image_viewer.display_mode_changed.connect(self._update_status_view_mode) # Connect new signal

        self.update_ui_state()

    def setup_ui(self):
        """Set up the main UI layout and panels."""
        self.image_viewer = ImageViewer(self)
        self.setCentralWidget(self.image_viewer)
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowNestedDocks)

        # Adjustment Panel Dock
        self.adjustment_dock = QDockWidget("Adjustments", self)
        self.adjustment_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.adjustment_panel = AdjustmentPanel(self.adjustment_dock)
        self.adjustment_dock.setWidget(self.adjustment_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.adjustment_dock)

        # Film Simulation Panel Dock
        self.film_preset_dock = QDockWidget("Film Simulation", self)
        self.film_preset_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.film_preset_panel = FilmPresetPanel(main_window=self, preset_manager=self.film_preset_manager, parent=self.film_preset_dock)
        self.film_preset_dock.setWidget(self.film_preset_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.film_preset_dock)

        # Photo Preset Panel Dock
        self.photo_preset_dock = QDockWidget("Photo Styles", self)
        self.photo_preset_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.photo_preset_panel = PhotoPresetPanel(main_window=self, preset_manager=self.photo_preset_manager, parent=self.photo_preset_dock)
        self.photo_preset_dock.setWidget(self.photo_preset_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.photo_preset_dock)

        # Filmstrip Dock
        self.filmstrip_dock = QDockWidget("Batch Filmstrip", self)
        self.filmstrip_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea)
        self.filmstrip_widget = BatchFilmstripWidget(self.filmstrip_dock)
        self.filmstrip_dock.setWidget(self.filmstrip_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.filmstrip_dock)
        self.filmstrip_dock.setVisible(False) # Start hidden

        # Histogram Panel Dock
        self.histogram_dock = QDockWidget("Histogram", self)
        self.histogram_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.BottomDockWidgetArea)
        self.histogram_widget = HistogramWidget(self.histogram_dock)
        self.histogram_dock.setWidget(self.histogram_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.histogram_dock)
        self.histogram_dock.setVisible(True) # Start visible

        # Tabify right-side docks
        self.tabifyDockWidget(self.adjustment_dock, self.film_preset_dock)
        self.tabifyDockWidget(self.film_preset_dock, self.photo_preset_dock)

        # Connect auto-adjustment signals
        self.adjustment_panel.awb_requested.connect(self.handle_auto_wb)
        self.adjustment_panel.auto_level_requested.connect(self.handle_auto_level)
        self.adjustment_panel.auto_color_requested.connect(self.handle_auto_color)
        self.adjustment_panel.auto_tone_requested.connect(self.handle_auto_tone)

    @pyqtSlot(str)
    def handle_auto_wb(self, method):
        """Handle Auto White Balance request by calling the appropriate calculation method."""
        if self.base_image is None:
            self.statusBar().showMessage("Load and convert an image first", 3000)
            return

        logger.debug("handle_auto_wb called with method: %s", method)
        awb_params = {'temp': 0, 'tint': 0} # Default

        try:
            # Call the correct calculation function based on the method string
            if method == 'gray_world':
                awb_params = AdvancedAdjustments.calculate_gray_world_awb_params(self.base_image)
            elif method == 'white_patch':
                # Can add a setting for percentile later
                awb_params = AdvancedAdjustments.calculate_white_patch_awb_params(self.base_image, percentile=1.0)
            elif method == 'simple_wb':
                 awb_params = AdvancedAdjustments.calculate_simple_wb_params(self.base_image)
            elif method == 'learning_wb':
                 awb_params = AdvancedAdjustments.calculate_learning_wb_params(self.base_image)
            # Add Near-White as an option if desired, or keep it internal
            # elif method == 'near_white':
            #     awb_params = AdvancedAdjustments.calculate_near_white_awb_params(self.base_image, percentile=1.0)
            else:
                logger.warning("Unknown AWB method '%s' requested. Using default.", method)
                # Optionally default to one method, e.g., Near-White or Gray World
                # awb_params = AdvancedAdjustments.calculate_near_white_awb_params(self.base_image)

            # Update adjustment panel, temporarily disconnecting signal
            try:
                self.adjustment_panel.adjustment_changed.disconnect(self.handle_adjustment_change)
            except TypeError: # Signal already disconnected or never connected
                pass
            self.adjustment_panel.set_adjustments(awb_params) # Pass the calculated params
            self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)

            # Manually trigger the processing now
            self.handle_adjustment_change(self.adjustment_panel._current_adjustments)
            self.statusBar().showMessage(f"Auto White Balance ({method}) calculated", 3000)

        # Add the except block for the main try statement (line 338)
        except Exception as e:
            logger.exception("Auto White Balance calculation failed (method=%s)", method)
            self.statusBar().showMessage(f"Auto White Balance calculation failed: {str(e)}", 5000)
        # The 'else' block related to the initial 'if self.base_image is None:' check
        # is already handled by the return statement on line 333.
        # Remove the misplaced 'else' block entirely.

    @pyqtSlot(str, float)
    def handle_auto_level(self, colorspace_mode, midrange):
        """Handle Auto Level request."""
        # Use base_image (the initially converted image) for calculations
        if self.base_image is not None:
            try:
                # Calculate the level parameters using the base image
                level_params = AdvancedAdjustments.calculate_auto_levels_params(
                    self.base_image,
                    colorspace_mode=colorspace_mode, # Pass mode for potential future use in calculation
                    midrange=midrange
                )

                # Update the adjustment panel with the calculated parameters
                # Note: We only update the relevant level parameters
                # Update adjustment panel, temporarily disconnecting signal
                try:
                    self.adjustment_panel.adjustment_changed.disconnect(self.handle_adjustment_change)
                except TypeError:
                    pass
                self.adjustment_panel.set_adjustments(level_params)
                self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)
                # Manually trigger the processing now
                self.handle_adjustment_change(self.adjustment_panel._current_adjustments)

                self.statusBar().showMessage(f"Auto Levels calculated (mode: {colorspace_mode}, midrange: {midrange:.2f})", 3000)
            except Exception as e:
                logger.exception("Auto Levels calculation failed (mode=%s, midrange=%s)", colorspace_mode, midrange)
                self.statusBar().showMessage(f"Auto Levels calculation failed: {str(e)}", 5000)
        else:
            self.statusBar().showMessage("Load and convert an image first", 3000)

    @pyqtSlot(str)
    def handle_auto_color(self, method):
        """Handle Auto Color request."""
        # Use base_image (the initially converted image) for calculations
        if self.base_image is not None:
            try:
                # Calculate the temp/tint parameters
                color_params = AdvancedAdjustments.calculate_auto_color_params(
                    self.base_image,
                    method=method
                )

                # Update the adjustment panel with the calculated parameters
                # Update adjustment panel, temporarily disconnecting signal
                try:
                    self.adjustment_panel.adjustment_changed.disconnect(self.handle_adjustment_change)
                except TypeError:
                    pass
                self.adjustment_panel.set_adjustments(color_params)
                self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)
                # Manually trigger the processing now
                self.handle_adjustment_change(self.adjustment_panel._current_adjustments)

                self.statusBar().showMessage(f"Auto Color ({method}) calculated", 3000)
            except Exception as e:
                logger.exception("Auto Color calculation failed (method=%s)", method)
                self.statusBar().showMessage(f"Auto Color calculation failed: {str(e)}", 5000)
        else:
            self.statusBar().showMessage("Load and convert an image first", 3000)

    @pyqtSlot()
    def handle_auto_tone(self):
        """Handle Auto Tone request (runs in background)."""
        if self.base_image is None:
            self.statusBar().showMessage("Load and convert an image first", 3000)
            return
        if self._is_converting_initial or self._request_pending or self._is_auto_tone_running:
            self.statusBar().showMessage("Busy—please wait…", 2000)
            return

        self._is_auto_tone_running = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Calculating Auto Tone…", 0)
        self.update_ui_state()

        # Run in worker thread:
        # QMetaObject.invokeMethod doesn't accept arbitrary Python objects cleanly in PyQt6.
        # Emit a queued signal instead.
        self._auto_tone_requested.emit(self.base_image.copy())

    @pyqtSlot(dict)
    def _handle_auto_tone_finished(self, tone_params: dict):
        """Apply computed auto tone params on UI thread."""
        try:
            # Update adjustment panel without triggering multiple reprocesses
            try:
                self.adjustment_panel.adjustment_changed.disconnect(self.handle_adjustment_change)
            except TypeError:
                pass

            self.adjustment_panel.set_adjustments(tone_params)

            self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)

            # Trigger processing once with the updated adjustments
            self.handle_adjustment_change(self.adjustment_panel._current_adjustments)
            self.statusBar().showMessage("Auto Tone applied.", 3000)
        finally:
            self._is_auto_tone_running = False
            QApplication.restoreOverrideCursor()
            self.update_ui_state()

    @pyqtSlot(str)
    def _handle_auto_tone_error(self, error_message: str):
        self._is_auto_tone_running = False
        QApplication.restoreOverrideCursor()
        logger.error("Auto Tone error: %s", error_message)
        self.statusBar().showMessage(error_message, 5000)
        self.update_ui_state()

    def create_actions(self):
        """Create QAction objects for menus and toolbars."""
        self.open_action = QAction("&Open Negative...", self, statusTip="Open a film negative image file", shortcut=QKeySequence.StandardKey.Open, triggered=self.open_image)
        self.save_as_action = QAction("&Save Positive As...", self, statusTip="Save the processed positive image", shortcut=QKeySequence.StandardKey.SaveAs, triggered=self.save_image_as, enabled=False)
        self.open_batch_action = QAction("Open Folder for &Batch...", self, statusTip="Select a folder of images for batch processing", triggered=self.open_batch_dialog)
        self.exit_action = QAction("E&xit", self, statusTip="Exit the application", shortcut=QKeySequence.StandardKey.Quit, triggered=self.close)

        # Undo Action
        self.undo_action = QAction("&Undo", self, statusTip="Undo the last destructive operation", shortcut=QKeySequence.StandardKey.Undo, triggered=self.undo_last_destructive_op, enabled=False)
        self.compare_action = QAction("&Compare Before/After", self, statusTip="Toggle between original and adjusted view", checkable=True, triggered=self.toggle_compare_view, enabled=False)

        self.view_adjust_action = self.adjustment_dock.toggleViewAction()
        self.view_adjust_action.setText("Adjustment Panel")
        self.view_adjust_action.setStatusTip("Show/Hide the Adjustment Panel")
        self.view_film_preset_action = self.film_preset_dock.toggleViewAction()
        self.view_film_preset_action.setText("Film Simulation Panel")
        self.view_film_preset_action.setStatusTip("Show/Hide the Film Simulation Panel")
        self.view_photo_preset_action = self.photo_preset_dock.toggleViewAction()
        self.view_photo_preset_action.setText("Photo Style Panel")
        self.view_photo_preset_action.setStatusTip("Show/Hide the Photo Style Panel")
        self.view_filmstrip_action = self.filmstrip_dock.toggleViewAction()
        self.view_filmstrip_action.setText("Batch Filmstrip")
        self.view_filmstrip_action.setStatusTip("Show/Hide the Batch Filmstrip Panel")
        self.view_histogram_action = self.histogram_dock.toggleViewAction()
        self.view_histogram_action.setText("Histogram Panel")
        self.view_histogram_action.setStatusTip("Show/Hide the Histogram Panel")

        # Settings Action
        self.settings_action = QAction("&Settings...", self, statusTip="Edit application settings", triggered=self.open_settings_dialog)
    def create_menus(self):
        """Create the main menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.open_batch_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.compare_action) # Add compare action to Edit menu
        edit_menu.addSeparator()
        edit_menu.addAction(self.settings_action) # Add settings action

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.view_adjust_action)
        view_menu.addAction(self.view_film_preset_action)
        view_menu.addAction(self.view_photo_preset_action)
        view_menu.addAction(self.view_histogram_action) # Add histogram view action
        view_menu.addSeparator()
        view_menu.addAction(self.view_filmstrip_action)
        # Toolbar view actions added in their creation methods

    def create_status_bar(self):
        """Creates the status bar and adds permanent widgets for information."""
        status_bar = self.statusBar()
        status_bar.showMessage("Ready", 2000) # Show 'Ready' temporarily

        # Add permanent widgets for different info pieces
        self.status_filename_label = QLabel(" File: None ")
        self.status_filename_label.setToolTip("Current image file path")
        status_bar.addPermanentWidget(self.status_filename_label)

        self.status_size_label = QLabel(" Size: N/A ")
        self.status_size_label.setToolTip("Image file size")
        status_bar.addPermanentWidget(self.status_size_label)

        self.status_mode_label = QLabel(" Mode: N/A ")
        self.status_mode_label.setToolTip("Original image color mode")
        status_bar.addPermanentWidget(self.status_mode_label)

        self.status_neg_type_label = QLabel(" Negative Type: N/A ")
        self.status_neg_type_label.setToolTip("Detected negative base type (Click to change)")
        self.status_neg_type_label.mousePressEvent = self._handle_neg_type_label_click # Make clickable
        self.status_neg_type_label.setCursor(Qt.CursorShape.PointingHandCursor) # Indicate clickability
        status_bar.addPermanentWidget(self.status_neg_type_label)

        self.status_view_mode_label = QLabel(" View: After ")
        self.status_view_mode_label.setToolTip("Current display mode (Before/After comparison)")
        status_bar.addPermanentWidget(self.status_view_mode_label)

    # --- Status Bar Update Methods ---

    def _update_status_filename(self, file_path):
        if file_path:
            filename = os.path.basename(file_path)
            self.status_filename_label.setText(f" File: {filename} ")
            self.status_filename_label.setToolTip(f"File Path: {file_path}")
        else:
            self.status_filename_label.setText(" File: None ")
            self.status_filename_label.setToolTip("No file loaded")

    def _update_status_size(self, size_bytes):
        if size_bytes is not None:
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
            self.status_size_label.setText(f" Size: {size_str} ")
        else:
            self.status_size_label.setText(" Size: N/A ")

    def _update_status_mode(self, mode):
        if mode:
            self.status_mode_label.setText(f" Mode: {mode} ")
        else:
            self.status_mode_label.setText(" Mode: N/A ")

    def _update_status_neg_type(self, neg_type):
        if neg_type:
            self.status_neg_type_label.setText(f" Negative Type: {neg_type} ")
        else:
            self.status_neg_type_label.setText(" Negative Type: N/A ")

    @pyqtSlot(str)
    def _update_status_view_mode(self, view_mode):
        """Slot to update the view mode label based on signal from ImageViewer."""
        if view_mode == 'before':
            self.status_view_mode_label.setText(" View: Before ")
        else: # Default to 'after'
            self.status_view_mode_label.setText(" View: After ")
    def create_toolbars(self): pass # No other toolbars currently

    def create_batch_toolbar(self):
        """Create the toolbar for batch processing controls, including format and quality."""
        self.batch_toolbar = QToolBar("Batch Processing", self)
        self.batch_toolbar.setObjectName("BatchToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.batch_toolbar)
        self.batch_toolbar.setIconSize(QSize(16, 16))

        # Output Directory
        self.select_output_dir_action = QAction("Set Output Dir", self)
        self.select_output_dir_action.setStatusTip("Select the directory to save batch processed images")
        self.select_output_dir_action.triggered.connect(self.select_batch_output_directory)
        self.batch_toolbar.addAction(self.select_output_dir_action)
        self.output_dir_label = QLabel(" Output: [Not Set] ")
        self.output_dir_label.setStatusTip("Currently selected output directory for batch processing")
        self.batch_toolbar.addWidget(self.output_dir_label)

        # Format Selection (Single instance)
        self.batch_toolbar.addSeparator()
        self.batch_toolbar.addWidget(QLabel(" Format: "))
        self.format_combo = QComboBox(self)
        self.format_combo.addItems([".jpg", ".png", ".tif"]) # Supported formats
        self.format_combo.setToolTip("Select output file format")
        # Connect signal AFTER spinbox is created
        self.batch_toolbar.addWidget(self.format_combo)

        # Quality Selection (Single instance)
        self.quality_label = QLabel(" Quality: ")
        self.batch_toolbar.addWidget(self.quality_label)
        self.quality_spinbox = QSpinBox(self)
        self.quality_spinbox.setToolTip("Set output quality (JPEG: 1-100, PNG: 0-9)")
        self.quality_spinbox.setValue(95) # Set initial default value
        self.batch_toolbar.addWidget(self.quality_spinbox)

        # Now connect the signal and set initial state
        self.format_combo.currentTextChanged.connect(self._update_quality_widget_visibility)
        # Initial state will be set by the update_ui_state call at the end of __init__

        # Process Button
        self.batch_toolbar.addSeparator()
        self.process_batch_action = QAction("Process Batch", self)
        self.process_batch_action.setStatusTip("Start processing the selected images in the filmstrip")
        self.process_batch_action.triggered.connect(self.start_batch_processing)
        self.process_batch_action.setEnabled(False)
        self.batch_toolbar.addAction(self.process_batch_action)

        # Progress Bar
        self.batch_progress_bar = QProgressBar(self)
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setMaximumWidth(200)
        self.batch_toolbar.addWidget(self.batch_progress_bar)

        # Add view action for this toolbar
        view_menu = self.menuBar().findChild(QMenu, "&View")
        if view_menu:
            self.view_batch_toolbar_action = self.batch_toolbar.toggleViewAction()
            self.view_batch_toolbar_action.setText("Batch Toolbar")
            self.view_batch_toolbar_action.setStatusTip("Show/Hide the Batch Processing Toolbar")
            view_menu.addSeparator()
            view_menu.addAction(self.view_batch_toolbar_action)
        self.batch_toolbar.setVisible(False) # Start hidden

    def create_view_toolbar(self):
        """Create the toolbar for view controls (zoom, fit, etc.)."""
        self.view_toolbar = QToolBar("View Controls", self)
        self.view_toolbar.setObjectName("ViewToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.view_toolbar)
        self.view_toolbar.setIconSize(QSize(16, 16))

        # Zoom In
        zoom_in_action = QAction(QIcon.fromTheme("zoom-in", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-zoom-in-16.png")), "Zoom In", self)
        zoom_in_action.setStatusTip("Zoom in on the image")
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        self.view_toolbar.addAction(zoom_in_action)

        # Zoom Out
        zoom_out_action = QAction(QIcon.fromTheme("zoom-out", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-zoom-out-16.png")), "Zoom Out", self)
        zoom_out_action.setStatusTip("Zoom out of the image")
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        self.view_toolbar.addAction(zoom_out_action)

        # Zoom Actual Size (100%)
        zoom_actual_action = QAction(QIcon.fromTheme("zoom-original", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-zoom-actual-16.png")), "Actual Size", self)
        zoom_actual_action.setStatusTip("View image at 100% zoom")
        zoom_actual_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_0)) # Wrap in QKeySequence
        zoom_actual_action.triggered.connect(self.image_viewer.reset_zoom) # Correct method name
        self.view_toolbar.addAction(zoom_actual_action)

        # Zoom Fit
        zoom_fit_action = QAction(QIcon.fromTheme("zoom-fit-best", QIcon(":/qt-project.org/styles/commonstyle/images/standardbutton-zoom-fit-16.png")), "Zoom to Fit", self)
        zoom_fit_action.setStatusTip("Fit image to the view area")
        zoom_fit_action.setShortcut(QKeySequence(Qt.Modifier.CTRL | Qt.Key.Key_9)) # Wrap in QKeySequence
        zoom_fit_action.triggered.connect(self.image_viewer.fit_to_window) # Correct method name
        self.view_toolbar.addAction(zoom_fit_action)

        # --- Compare wipe slider (Before/After wipe) ---
        self.view_toolbar.addSeparator()
        self.view_toolbar.addWidget(QLabel(" Compare: "))

        # Clear left/right semantics: [Before] [----slider----] [After]
        self.view_toolbar.addWidget(QLabel("Before"))
        self.compare_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.compare_slider.setRange(0, 100)
        self.compare_slider.setValue(100)  # default: all "after"
        self.compare_slider.setMaximumWidth(180)
        self.compare_slider.setToolTip("Before/After wipe (0=Before, 100=After)")
        self.compare_slider.valueChanged.connect(self._on_compare_slider_changed)
        self.compare_slider.sliderPressed.connect(self._on_compare_slider_pressed)
        self.compare_slider.sliderReleased.connect(self._on_compare_slider_released)
        self.view_toolbar.addWidget(self.compare_slider)
        self.view_toolbar.addWidget(QLabel("After"))

        # Add view action for this toolbar
        view_menu = self.menuBar().findChild(QMenu, "&View")
        if view_menu:
            self.view_view_toolbar_action = self.view_toolbar.toggleViewAction()
            self.view_view_toolbar_action.setText("View Toolbar")
            self.view_view_toolbar_action.setStatusTip("Show/Hide the View Controls Toolbar")
            # view_menu.addSeparator() # Already added one for batch toolbar
            view_menu.addAction(self.view_view_toolbar_action)
        self.view_toolbar.setVisible(True) # Start visible

    def select_batch_output_directory(self):
        """Opens a dialog to select the output directory for batch processing."""
        # Suggest starting from the last used directory or user's home/pictures
        start_dir = self._batch_output_dir if self._batch_output_dir else os.path.expanduser("~")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder for Batch Processing", start_dir)
        if folder_path:
            self._batch_output_dir = folder_path
            self.output_dir_label.setText(f" Output: ...{os.path.basename(folder_path)} ")
            self.output_dir_label.setToolTip(f"Output Directory: {folder_path}")
            self.update_ui_state() # Update button states
        else:
            self.statusBar().showMessage("Output directory selection cancelled.", 2000)

    def start_batch_processing(self): # noqa C901
        """Initiates the batch processing workflow."""
        checked_files = self.filmstrip_widget.get_checked_image_paths()
        if not checked_files:
            QMessageBox.warning(self, "No Images Selected", "Please check the images in the filmstrip you want to process.")
            return

        if not self._batch_output_dir or not os.path.isdir(self._batch_output_dir):
            QMessageBox.warning(self, "Output Directory Not Set", "Please select a valid output directory using the 'Set Output Dir' button.")
            return

        # Get current adjustments from the panel
        current_adjustments = self.adjustment_panel.get_adjustments()

        # Get active preset info (if any) from the stored state
        # This is set when a preset is applied and cleared when manual adjustments are made
        active_preset_info = self._active_preset_details

        # Get output format and quality settings from the toolbar
        output_format = self.format_combo.currentText()
        quality_settings = {}
        if output_format == ".jpg":
            quality_settings['quality'] = self.quality_spinbox.value()
        elif output_format == ".png":
            quality_settings['png_compression'] = self.quality_spinbox.value()
        # Add other formats (like WebP) if supported by the saver and toolbar

        # Disable UI elements during batch processing
        self.process_batch_action.setEnabled(False)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        self.statusBar().showMessage(f"Starting batch processing for {len(checked_files)} images...")

        # Set inputs for the worker
        self.batch_worker.set_inputs(
            checked_files,
            self._batch_output_dir,
            current_adjustments,
            active_preset_info, # Pass preset info
            self.negative_converter, # Pass converter instance
            self.film_preset_manager, # Pass managers
            self.photo_preset_manager,
            output_format, # Pass format
            quality_settings # Pass quality/compression
        )

        # Start the worker using QMetaObject.invokeMethod to ensure it runs in the worker's thread
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
        self.update_ui_state() # Re-enable UI

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
        self.update_ui_state() # Re-enable UI
        QMessageBox.critical(self, "Batch Processing Error", f"An error occurred during batch processing:\n{error_message}")
        self.statusBar().showMessage(f"Batch processing error: {error_message}", 5000)


    # --- Format/Quality Toolbar Logic ---

    @pyqtSlot(str)
    def _update_quality_widget_visibility(self, selected_format):
        """Show/hide and configure the quality spinbox based on selected format."""
        if selected_format == ".jpg":
            self.quality_label.setText(" Quality: ")
            self.quality_spinbox.setRange(1, 100)
            self.quality_spinbox.setToolTip("Set JPEG quality (1-100)")
            # Fetch default from settings when format changes
            self.quality_spinbox.setValue(app_settings.UI_DEFAULTS.get("default_jpeg_quality", 95))
            self.quality_label.setVisible(True)
            self.quality_spinbox.setVisible(True)
        elif selected_format == ".png":
            self.quality_label.setText(" Compress: ")
            self.quality_spinbox.setRange(0, 9)
            self.quality_spinbox.setToolTip("Set PNG compression level (0-9)")
            # Fetch default from settings when format changes
            self.quality_spinbox.setValue(app_settings.UI_DEFAULTS.get("default_png_compression", 6))
            self.quality_label.setVisible(True)
            self.quality_spinbox.setVisible(True)
        # Add elif for WebP if needed, similar to JPG
        # elif selected_format == ".webp":
        #     self.quality_label.setText(" Quality: ")
        #     self.quality_spinbox.setRange(0, 100) # WebP quality 0-100
        #     self.quality_spinbox.setToolTip("Set WebP quality (0-100)")
        #     self.quality_spinbox.setValue(app_settings.UI_DEFAULTS.get("default_webp_quality", 90)) # Need separate setting?
        #     self.quality_label.setVisible(True)
        #     self.quality_spinbox.setVisible(True)
        else: # TIFF or other formats without quality setting here
            self.quality_label.setVisible(False)
            self.quality_spinbox.setVisible(False)

    def update_ui_state(self):
        """Enable/disable actions and controls based on application state."""
        image_loaded = self.base_image is not None
        # Check flags we explicitly manage: initial conversion, a pending adjustment request, or auto tone
        is_processing = self._is_converting_initial or self._request_pending or getattr(self, "_is_auto_tone_running", False)
        logger.debug(
            "update_ui_state: image_loaded=%s, _is_converting_initial=%s, _request_pending=%s, is_processing=%s",
            image_loaded,
            self._is_converting_initial,
            self._request_pending,
            is_processing,
        )

        self.save_as_action.setEnabled(image_loaded and not is_processing)
        self.compare_action.setEnabled(image_loaded and not is_processing)
        self.undo_action.setEnabled(self.previous_base_image is not None and not is_processing)
        self.adjustment_panel.setEnabled(image_loaded and not is_processing)
        self.film_preset_panel.setEnabled(image_loaded and not is_processing)
        self.photo_preset_panel.setEnabled(image_loaded and not is_processing)

        # Compare wipe slider: enabled only when a before-image exists and not processing.
        has_before = self.image_viewer.has_before_image()
        if hasattr(self, "compare_slider"):
            self.compare_slider.setEnabled(image_loaded and has_before and not is_processing)
            self.compare_slider.setVisible(image_loaded and has_before)

        # Photo preset panel no longer contains a "Save Preset..." button; nothing to update here.

        # Batch processing controls
        has_batch_items = self.filmstrip_widget.list_widget.count() > 0
        has_checked_items = len(self.filmstrip_widget.get_checked_image_paths()) > 0
        output_dir_set = self._batch_output_dir is not None and os.path.isdir(self._batch_output_dir)

        self.batch_toolbar.setVisible(has_batch_items) # Show toolbar only if items exist
        # Use the worker's status, not the thread's running state
        self.process_batch_action.setEnabled(has_checked_items and output_dir_set and not self.batch_worker._is_running)

        # Update quality widget visibility based on current format selection
        self._update_quality_widget_visibility(self.format_combo.currentText())


    # --- File Operations ---

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
                self.adjustment_panel.reset_adjustments() # Correct method name
                self.image_viewer.set_image(None) # Clear previous image
                self.previous_base_image = None # Clear undo state
                self.compare_action.setChecked(False) # Ensure compare is off
                # self.image_viewer.set_display_mode('after') # REMOVED: set_image handles this
                self.histogram_widget.clear_histogram() # Clear histogram

                # --- Trigger Initial Conversion in Worker Thread ---
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor
                self.statusBar().showMessage("Starting initial negative conversion...", 0) # Persistent message
                self._is_converting_initial = True
                self.update_ui_state() # Disable controls
                # Emit signal with raw image, override is None by default
                self.initial_conversion_requested.emit(self.raw_loaded_image, None)



            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load or start conversion for image:\n{file_path}\n\nError: {e}")
                self.statusBar().showMessage(f"Error loading image: {e}", 5000)
                self.raw_loaded_image = None
                self.current_file_path = None
                self._is_converting_initial = False # Ensure flag is reset on error
                QApplication.restoreOverrideCursor() # Restore cursor on error
                self.update_ui_state() # Re-enable controls if needed

    def save_image_as(self):
        """Save the currently processed image to a new file."""
        if self.base_image is None:
            QMessageBox.warning(self, "No Image", "No image is currently loaded or processed.")
            return

        # Suggest filename based on current file + suffix
        if self.current_file_path:
            base, _ = os.path.splitext(os.path.basename(self.current_file_path))
            # Default to JPG, but could potentially use the batch format combo's value?
            suggested_name = f"{base}_positive.jpg"
        else:
            suggested_name = "positive_image.jpg"

        # Define filter string for supported save formats
        save_filter = "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png);;WebP Image (*.webp);;TIFF Image (*.tif *.tiff)" # Added WebP

        file_path, selected_filter = QFileDialog.getSaveFileName(self, "Save Positive Image As", suggested_name, save_filter)

        if file_path:
            try:
                self.statusBar().showMessage("Applying final adjustments...")
                QApplication.processEvents()
                # Get the final adjusted image using the background worker for consistency
                # This ensures the saved image matches the preview exactly
                final_image_to_save = self._get_fully_adjusted_image(self.base_image, self.adjustment_panel.get_adjustments())

                if final_image_to_save is None:
                    raise ValueError("Failed to get final adjusted image for saving.")

                self.statusBar().showMessage(f"Saving image to {os.path.basename(file_path)}...")
                QApplication.processEvents()

                # Determine format and quality parameters from settings
                quality_params = {}
                ext = os.path.splitext(file_path)[1].lower()

                if ext in ['.jpg', '.jpeg']:
                    # Fetch current JPEG quality from settings
                    quality = app_settings.UI_DEFAULTS.get("default_jpeg_quality", 95)
                    quality_params['quality'] = quality
                    logger.debug("Using JPEG quality from settings: %s", quality)

                elif ext == '.png':
                    # Fetch current PNG compression from settings
                    compression = app_settings.UI_DEFAULTS.get("default_png_compression", 6) # Using 6 as a more typical default now
                    quality_params['png_compression'] = compression
                    logger.debug("Using PNG compression from settings: %s", compression)

                elif ext == '.webp':
                    # Fetch current WebP quality from settings (assuming it uses 'default_jpeg_quality' for simplicity, or add a specific setting)
                    # Let's reuse JPEG quality setting for WebP for now.
                    quality = app_settings.UI_DEFAULTS.get("default_jpeg_quality", 90) # Default 90 for WebP if not set
                    quality_params['quality'] = quality
                    logger.debug("Using WebP quality from settings: %s", quality)

                elif ext in ['.tif', '.tiff']:
                    # Add TIFF specific params if needed, e.g., compression
                    quality_params['compression'] = 'tiff_lzw' # Example: LZW compression
                    logger.debug("Using TIFF LZW compression.")

                # Other formats currently don't have specific quality/compression settings here

                # Save using the saver module
                success = self.conversion_service.save_image(final_image_to_save, file_path, **quality_params)

                if success:
                    self.statusBar().showMessage(f"Image saved successfully to {file_path}", 5000)
                else:
                    raise ValueError("Failed to save image using image_saver.")

            except Exception as e:
                 QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")
                 self.statusBar().showMessage(f"Error saving image: {e}", 5000)
            # finally: # No need for update_ui_state here usually
            #      self.update_ui_state()

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
                    # Visibility of batch toolbar handled by update_ui_state
                    self.statusBar().showMessage(f"Loaded {len(image_files)} images for batch processing.", 3000)
                else:
                    self.filmstrip_dock.setVisible(False) # Hide if no images found
                    self.statusBar().showMessage(f"No supported image files found in {folder_path}.", 3000)
                    QMessageBox.information(self, "No Images Found", f"No supported image files (e.g., JPG, PNG, TIF) were found in the selected folder:\n{folder_path}")

                self.update_ui_state() # Update button states and toolbar visibility

            except Exception as e:
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

    def closeEvent(self, event):
        """Handle application close event."""
        # Clean up threads
        self.processing_thread.quit()
        self.processing_thread.wait(1000) # Wait max 1 sec

        self.auto_tone_thread.quit()
        self.auto_tone_thread.wait(1000)

        self.initial_conversion_thread.quit()
        self.initial_conversion_thread.wait(1000)

        self.batch_thread.quit()
        self.batch_thread.wait(1000)

        logger.info("Background threads stopped.")
        event.accept()

    # --- UI Interaction Handlers ---

    @pyqtSlot(str)
    def handle_filmstrip_preview(self, file_path):
        """Load and display the selected image from the filmstrip for preview."""
        if file_path and file_path != self.current_file_path:
            logger.debug("Filmstrip preview requested for: %s", os.path.basename(file_path))
            # Essentially the same logic as open_image, but without the file dialog
            self.statusBar().showMessage(f"Loading preview: {os.path.basename(file_path)}...")
            QApplication.processEvents()
            try:
                # Unpack all return values
                self.raw_loaded_image, original_mode, file_size = self.conversion_service.load_image(file_path)
                if self.raw_loaded_image is None:
                    raise ValueError("Image loader returned None.")

                self.current_file_path = file_path
                self._current_file_size = file_size # Use file size returned by loader
                self._current_original_mode = original_mode
                self._current_negative_type = None

                self._update_status_filename(self.current_file_path)
                self._update_status_size(self._current_file_size)
                self._update_status_mode(self._current_original_mode)
                self._update_status_neg_type(None)

                self.adjustment_panel.reset_adjustments() # Correct method name
                self.image_viewer.set_image(None) # Clear previous image
                self.previous_base_image = None
                self.compare_action.setChecked(False)
                # self.image_viewer.set_display_mode('after') # REMOVED: set_image handles this
                self.histogram_widget.clear_histogram()

                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor
                self.statusBar().showMessage("Starting initial negative conversion for preview...", 0)
                self._is_converting_initial = True
                self.update_ui_state()
                self.initial_conversion_requested.emit(self.raw_loaded_image, None)

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load preview image:\n{file_path}\n\nError: {e}")
                self.statusBar().showMessage(f"Error loading preview: {e}", 5000)
                self.raw_loaded_image = None
                self.current_file_path = None
                self._is_converting_initial = False
                QApplication.restoreOverrideCursor() # Restore cursor on error
                self.update_ui_state()


    @pyqtSlot(dict)
    def handle_adjustment_change(self, adjustments_dict):
        """Request reprocessing when an adjustment slider/value changes."""
        # Clear any active preset when manual adjustments are made
        self._active_preset_details = None
        # Existing logic to request processing follows...
        self.request_processing(adjustments_dict)

    def request_processing(self, adjustments):
        """Emit signal to request processing in the worker thread."""
        # print(f"DEBUG: request_processing called. base_image is None: {self.base_image is None}, _is_converting_initial: {self._is_converting_initial}, _request_pending: {self._request_pending}") # Debug Auto Actions
        if self.base_image is None or self._is_converting_initial:
            logger.debug("Skipping processing request: no base image or initial conversion running.")
            return # Don't process if no base image or initial conversion is happening

        # Avoid queuing up too many requests if the user is rapidly changing sliders
        if self._request_pending:
            # Store the latest adjustments and return
            self._pending_adjustments = adjustments
            # print("Processing request pending, storing latest adjustments.") # Debug
            return

        # print("Requesting processing with adjustments:", adjustments) # Debug
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor for adjustments
        self.statusBar().showMessage("Applying adjustments...", 0) # Show status
        self._request_pending = True
        self._pending_adjustments = adjustments # Store the current request
        # print(f"DEBUG: Emitting processing_requested via invokeMethod. Adjustments: {self._pending_adjustments.keys()}") # Debug Auto Actions
        # Use invokeMethod to ensure the signal emission happens cleanly
        QMetaObject.invokeMethod(self, "_emit_processing_request", Qt.ConnectionType.QueuedConnection)

    @pyqtSlot()
    def _emit_processing_request(self):
        """Helper slot to emit the processing request signal."""
        if self._pending_adjustments is not None:
            # print("Emitting processing request with stored adjustments.") # Debug
            self.processing_requested.emit(self.base_image, self._pending_adjustments)
            self._pending_adjustments = None # Clear pending request
        # self._request_pending will be set to False in _handle_processing_finished/_error


    @pyqtSlot(object)
    def _handle_processing_finished(self, processed_image):
        """Update the image viewer with the processed image."""
        # print("Processing finished, updating viewer.") # Debug
        self.image_viewer.set_image(processed_image)
        self.histogram_widget.update_histogram(processed_image) # Update histogram
        self._request_pending = False # Allow new requests
        # If there was a pending request while this one finished, trigger it now
        if self._pending_adjustments is not None:
            # print("Handling pending adjustments after processing finished.") # Debug
            self.request_processing(self._pending_adjustments)
        QApplication.restoreOverrideCursor() # Restore cursor
        self.statusBar().showMessage("Adjustments applied.", 2000) # Clear status
        self.update_ui_state() # Update UI state after processing finishes


    @pyqtSlot(str)
    def _handle_processing_error(self, error_message):
        """Show processing errors."""
        logger.error("Processing error: %s", error_message)
        self.statusBar().showMessage(f"Error applying adjustments: {error_message}", 5000)
        self._request_pending = False # Allow new requests even on error
        self._pending_adjustments = None # Clear pending request on error
        QApplication.restoreOverrideCursor() # Restore cursor on error
        self.update_ui_state() # Update UI state after processing error

    @pyqtSlot(str)
    def _handle_initial_conversion_error(self, error_message):
        """Handle errors during the initial conversion."""
        QMessageBox.critical(self, "Conversion Error", f"Initial negative conversion failed:\n{error_message}")
        self.statusBar().showMessage(f"Initial conversion failed: {error_message}", 5000)
        self._is_converting_initial = False
        QApplication.restoreOverrideCursor() # Restore cursor on error
        self.update_ui_state() # Re-enable controls

    @pyqtSlot(int, int)
    def _handle_initial_conversion_progress(self, step, total_steps):
        """Update status bar with initial conversion progress."""
        if total_steps > 0:
            # Example: "Converting: Step 2/5 (Applying WB...)"
            # We need more info from the backend to show step names.
            # For now, just show step numbers.
            self.statusBar().showMessage(f"Converting negative: Step {step}/{total_steps}...", 0) # Persistent
        else:
            self.statusBar().showMessage("Converting negative...", 0)


    @pyqtSlot(object, str) # Update slot signature
    def _handle_initial_conversion_finished(self, converted_image, mask_classification):
        """Handle successful initial conversion."""
        if converted_image is None:
             # This case should ideally be caught by the worker, but double-check
             self._handle_initial_conversion_error("Conversion worker returned None image.")
             return

        logger.info("Initial conversion finished. Mask classified as: %s", mask_classification)
        self.statusBar().showMessage("Initial conversion complete.", 2000)
        self._is_converting_initial = False

        # Store the initially converted image (before any adjustments)
        self.initial_converted_image = converted_image.copy()
        # Set the base image for adjustments
        self.base_image = converted_image.copy()
        self.previous_base_image = None # Reset undo state for new image

        # Update the image viewer and histogram
        self.image_viewer.set_image(self.base_image)
        self.histogram_widget.update_histogram(self.base_image)
        # Store the initial image in the viewer for comparison
        self.image_viewer.set_before_image(self.initial_converted_image)

        # Update negative type status
        self._current_negative_type = mask_classification
        self._update_status_neg_type(self._current_negative_type)

        # Reset adjustments panel to defaults for the new image
        self.adjustment_panel.reset_adjustments() # Correct method name

        # Re-enable UI controls
        QApplication.restoreOverrideCursor() # Restore cursor
        self.update_ui_state()

        # Apply default adjustments automatically? Optional.
        # default_adjustments = self.adjustment_panel.get_adjustments()
        # self.request_processing(default_adjustments)


    def get_current_image_for_processing(self):
        """
        Provides the current image (post-initial-conversion base) for UI components
        that need to know whether saving presets is possible.
        """
        return self.base_image.copy() if self.base_image is not None else None

    # --- Image Processing Logic ---

    def _apply_tile_adjustments(self, tile_image, adjustments):
        """Applies adjustments to a single tile (placeholder)."""
        # In a real scenario, this would call adjustment functions
        return self.conversion_service.apply_adjustments(tile_image, adjustments)

    def _get_fully_adjusted_image(self, base_image, adjustments):
        """Applies all current adjustments to the base image."""
        if base_image is None:
            return None
        try:
            # For now, apply directly. Could use tiling for large images later.
            return self.conversion_service.apply_adjustments(base_image.copy(), adjustments)
        except Exception as e:
            logger.exception("Error applying adjustments in _get_fully_adjusted_image")
            # Propagate the error message back via the worker's error signal
            raise # Re-raise the exception to be caught by the worker's run method

    # --- Undo/Compare Logic ---

    def _store_previous_state(self):
        """Stores the current base image for undo."""
        if self.base_image is not None:
            # Store the image *before* the destructive operation
            # This should be called just before applying presets or other non-reversible ops
            logger.debug("Storing state for undo.")
            self.previous_base_image = self.base_image.copy()
            self.update_ui_state() # Enable undo action


    def undo_last_destructive_op(self):
        """Reverts to the previously stored base image."""
        if self.previous_base_image is not None:
            logger.debug("Performing undo.")
            self.base_image = self.previous_base_image.copy()
            self.previous_base_image = None # Can only undo once per stored state
            # Reset adjustments and update viewer
            self.adjustment_panel.reset_adjustments() # Reset sliders to default
            self.image_viewer.set_image(self.base_image)
            self.histogram_widget.update_histogram(self.base_image)
            self.statusBar().showMessage("Undo successful.", 2000)
            self.update_ui_state() # Disable undo action
        else:
            self.statusBar().showMessage("Nothing to undo.", 2000)

    def toggle_compare_view(self):
        """Toggle the image viewer between 'before' and 'after'.

        If wipe-compare is active, disable it and fall back to the original toggle behavior.
        """
        if self.initial_converted_image is None:
            self.compare_action.setChecked(False) # Cannot compare if no initial image
            return

        # If wipe is enabled, turn it off when user hits toggle-compare.
        if hasattr(self, "compare_slider"):
            self.image_viewer.set_compare_wipe_enabled(False)
            self.compare_slider.blockSignals(True)
            self.compare_slider.setValue(100)
            self.compare_slider.blockSignals(False)

        # Call the ImageViewer's internal toggle method
        self.image_viewer.toggle_compare_mode()

        # Update the histogram based on the viewer's *new* mode
        if self.image_viewer._display_mode == 'before': # Access internal state (less ideal, but necessary here)
            self.histogram_widget.update_histogram(self.initial_converted_image)
        else: # 'after' mode
            current_adjusted = self._get_fully_adjusted_image(self.base_image, self.adjustment_panel.get_adjustments())
            if current_adjusted is not None: # Handle potential error in adjustment
                self.histogram_widget.update_histogram(current_adjusted)
            else: # Fallback if adjustment fails
                self.histogram_widget.update_histogram(self.base_image)


    def _on_compare_slider_pressed(self):
        self._compare_slider_is_dragging = True

    def _on_compare_slider_released(self):
        self._compare_slider_is_dragging = False
        # Apply final value immediately on release (so the user sees the exact position).
        if self._compare_slider_pending_value is not None:
            self._apply_compare_slider_value(int(self._compare_slider_pending_value))
            self._compare_slider_pending_value = None

    def _apply_pending_compare_slider_value(self):
        if self._compare_slider_pending_value is None:
            return
        self._apply_compare_slider_value(int(self._compare_slider_pending_value))
        # Keep pending value while dragging so release can apply instantly; clear only if not dragging.
        if not self._compare_slider_is_dragging:
            self._compare_slider_pending_value = None

    def _on_compare_slider_changed(self, value: int):
        """Throttle wipe-compare updates to avoid repaint storms."""
        if self.initial_converted_image is None or self.base_image is None:
            return

        self._compare_slider_pending_value = int(value)

        # While dragging: debounce updates (every ~50ms).
        # When not dragging (e.g. keyboard step): apply immediately.
        if self._compare_slider_is_dragging:
            self._compare_slider_debounce_timer.start(self._compare_slider_debounce_ms)
            return

        self._apply_compare_slider_value(int(value))

    def _apply_compare_slider_value(self, value: int):
        """Apply a compare slider value (0..100) to the viewer."""
        # Enable wipe compare when slider is between endpoints; disable at endpoints.
        enable_wipe = (0 < int(value) < 100)
        self.image_viewer.set_compare_wipe_enabled(enable_wipe)
        self.image_viewer.set_compare_wipe_percent(int(value))

        # Histogram: update only at endpoints.
        if int(value) == 0:
            self.histogram_widget.update_histogram(self.initial_converted_image)
        elif int(value) == 100:
            # IMPORTANT: don't recompute full adjustments on the UI thread here.
            # Just show the histogram of the current displayed "after" image.
            # (Keeping UI responsive is more important than exact histogram at this moment.)
            after_img = getattr(self.image_viewer, "_current_image", None)
            if after_img is not None:
                self.histogram_widget.update_histogram(after_img)
            else:
                self.histogram_widget.update_histogram(self.base_image)

    # --- Auto Adjustment Handlers ---

    @pyqtSlot()
    def handle_wb_picker_request(self):
        """Activate the color picker mode in the image viewer."""
        if self.image_viewer.image_label and self.image_viewer.image_label.pixmap():
            self.statusBar().showMessage("White Balance Picker Active: Click on a neutral gray/white area.", 0)
            self.image_viewer.enter_picker_mode()
        else:
            self.statusBar().showMessage("Load an image before using the White Balance picker.", 3000)

    @pyqtSlot(tuple)
    def handle_color_sampled(self, rgb_tuple):
        """Receives the sampled color from the viewer and passes it to the adjustment panel."""
        if rgb_tuple:
            logger.debug("Color sampled: %s", rgb_tuple)
            r, g, b = rgb_tuple

            # Avoid issues if sampled color is pure black (unlikely neutral)
            if r == 0 and g == 0 and b == 0:
                self.statusBar().showMessage("Cannot set WB from black sample.", 3000)
                return

            # --- Calculate approximate Temp/Tint slider values ---
            # These factors (0.6, 0.3) are derived empirically from how adjust_temp_tint works
            # It maps a slider range of -100 to 100 to roughly +/- 30 units of RGB change.
            # We want to reverse this: find the slider value that would neutralize the difference.
            # Temp: Balances Blue vs Red. If B > R, need negative temp (cooler).
            # Tint: Balances Green vs Magenta (Avg(R,B)). If G > Avg(R,B), need negative tint (more magenta).

            # Calculate raw differences
            temp_diff = b - r
            tint_diff = g - (r + b) / 2.0

            # Estimate slider values needed to counteract the difference
            # Note the sign inversion: if B > R (temp_diff > 0), we need negative temp slider value
            temp_slider_val = -temp_diff / 0.6 # Divide by the approximate effect per slider unit
            tint_slider_val = -tint_diff / 0.3 # Divide by the approximate effect per slider unit

            # Clamp values to slider range [-100, 100] and round
            temp_slider_val = int(round(np.clip(temp_slider_val, -100, 100)))
            tint_slider_val = int(round(np.clip(tint_slider_val, -100, 100)))

            logger.debug("Calculated WB adjustments: Temp=%s, Tint=%s", temp_slider_val, tint_slider_val)

            # Apply the calculated adjustments to the panel
            # Apply the calculated adjustments to the panel and trigger update
            adj_dict = {'temp': temp_slider_val, 'tint': tint_slider_val}
            self.adjustment_panel.set_adjustments(adj_dict)
            self.adjustment_panel.adjustment_changed.emit(self.adjustment_panel._current_adjustments)
            self.statusBar().showMessage(f"White balance adjusted based on sampled color {rgb_tuple}.", 3000)
        else: # Sampling cancelled
            self.statusBar().showMessage("White Balance picker cancelled.", 2000)


    @pyqtSlot(str) # Slot for AWB request from AdjustmentPanel
    def handle_awb_request(self, method):
        """Handles the Auto White Balance request."""
        if self.base_image is None: return
        logger.info("AWB requested with method: %s", method)
        self.statusBar().showMessage(f"Applying Auto White Balance ({method})...")
        QApplication.processEvents()
        try:
            # Apply AWB directly for now (could be moved to worker if slow)
            # The basic_adjuster needs the current image state if methods depend on it,
            # but typically AWB works on the base image or initial conversion.
            # Let's assume it works on self.base_image for simplicity here.
            # The function should return the *change* in WB gains or the new gains.
            # We need to update the sliders in AdjustmentPanel.
            # This requires modification to ImageAdjustments.auto_white_balance
            # to return the calculated gains.
            # --- Placeholder ---
            # wb_gains = self.basic_adjuster.auto_white_balance(self.base_image, method=method)
            # if wb_gains:
            #     self.adjustment_panel.set_wb_gains(wb_gains) # Need this method in AdjustmentPanel
            #     self.statusBar().showMessage("Auto White Balance applied.", 2000)
            # else:
            #     self.statusBar().showMessage("Auto White Balance failed.", 3000)
            # --- End Placeholder ---
            QMessageBox.information(self, "Not Implemented", f"Auto White Balance ({method}) is not fully implemented yet.")
            self.statusBar().showMessage("AWB not fully implemented.", 3000) # Placeholder message
        except Exception as e:
            QMessageBox.critical(self, "AWB Error", f"Error during Auto White Balance: {e}")
            self.statusBar().showMessage(f"AWB Error: {e}", 5000)


    @pyqtSlot(str, float) # Slot for Auto Level request from AdjustmentPanel
    def handle_auto_level_request(self, mode, midrange):
        """Handles the Auto Levels request."""
        if self.base_image is None: return
        logger.info("Auto Levels requested. Mode=%s, Midrange=%s", mode, midrange)
        self.statusBar().showMessage("Applying Auto Levels...")
        QApplication.processEvents()
        try:
            # Similar to AWB, this needs coordination. Auto Levels modifies the image
            # contrast/brightness based on histogram. It might modify the base_image
            # destructively or return new black/white points to be applied via sliders.
            # Applying destructively is simpler for now.
            # --- Placeholder ---
            # self._store_previous_state() # Store state before destructive op
            # modified_image = self.basic_adjuster.auto_levels(self.base_image, mode=mode, midrange=midrange)
            # if modified_image is not None:
            #     self.base_image = modified_image
            #     self.adjustment_panel.reset_adjustments() # Reset sliders as base changed
            #     self.image_viewer.set_image(self.base_image)
            #     self.histogram_widget.update_histogram(self.base_image)
            #     self.statusBar().showMessage("Auto Levels applied.", 2000)
            # else:
            #     self.statusBar().showMessage("Auto Levels failed.", 3000)
            #     self.previous_base_image = None # Clear undo state if failed
            #     self.update_ui_state()
            # --- End Placeholder ---
            QMessageBox.information(self, "Not Implemented", "Auto Levels is not fully implemented yet.")
            self.statusBar().showMessage("Auto Levels not fully implemented.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Auto Levels Error", f"Error during Auto Levels: {e}")
            self.statusBar().showMessage(f"Auto Levels Error: {e}", 5000)
            self.previous_base_image = None # Clear undo state on error
            self.update_ui_state()


    @pyqtSlot(str) # Slot for Auto Color request from AdjustmentPanel
    def handle_auto_color_request(self, method):
        """Handles the Auto Color request."""
        if self.base_image is None: return
        logger.info("Auto Color requested with method: %s", method)
        self.statusBar().showMessage(f"Applying Auto Color ({method})...")
        QApplication.processEvents()
        try:
            # Auto Color typically adjusts saturation and color balance.
            # Needs coordination like AWB/Levels.
            # --- Placeholder ---
            # self._store_previous_state()
            # modified_image = self.basic_adjuster.auto_color(self.base_image, method=method)
            # if modified_image is not None:
            #     self.base_image = modified_image
            #     self.adjustment_panel.reset_adjustments()
            #     self.image_viewer.set_image(self.base_image)
            #     self.histogram_widget.update_histogram(self.base_image)
            #     self.statusBar().showMessage("Auto Color applied.", 2000)
            # else:
            #     self.statusBar().showMessage("Auto Color failed.", 3000)
            #     self.previous_base_image = None
            #     self.update_ui_state()
            # --- End Placeholder ---
            QMessageBox.information(self, "Not Implemented", f"Auto Color ({method}) is not fully implemented yet.")
            self.statusBar().showMessage("Auto Color not fully implemented.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Auto Color Error", f"Error during Auto Color: {e}")
            self.statusBar().showMessage(f"Auto Color Error: {e}", 5000)
            self.previous_base_image = None
            self.update_ui_state()


    @pyqtSlot() # Slot for Auto Tone request from AdjustmentPanel
    def handle_auto_tone_request(self):
        """Handles the Auto Tone request."""
        if self.base_image is None: return
        logger.info("Auto Tone requested.")
        self.statusBar().showMessage("Applying Auto Tone...")
        QApplication.processEvents()
        try:
            # Auto Tone often combines aspects of Levels and Contrast.
            # --- Placeholder ---
            # self._store_previous_state()
            # modified_image = self.basic_adjuster.auto_tone(self.base_image)
            # if modified_image is not None:
            #     self.base_image = modified_image
            #     self.adjustment_panel.reset_adjustments()
            #     self.image_viewer.set_image(self.base_image)
            #     self.histogram_widget.update_histogram(self.base_image)
            #     self.statusBar().showMessage("Auto Tone applied.", 2000)
            # else:
            #     self.statusBar().showMessage("Auto Tone failed.", 3000)
            #     self.previous_base_image = None
            #     self.update_ui_state()
            # --- End Placeholder ---
            QMessageBox.information(self, "Not Implemented", "Auto Tone is not fully implemented yet.")
            self.statusBar().showMessage("Auto Tone not fully implemented.", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Auto Tone Error", f"Error during Auto Tone: {e}")
            self.statusBar().showMessage(f"Auto Tone Error: {e}", 5000)
            self.previous_base_image = None
            self.update_ui_state()


    # --- Preset Handling ---

    # Slot to handle preview requests from either preset panel
    # Updated signature: preset_type is now a string, preset_id is the string ID
    def handle_preset_preview(self, base_image_ignored, preset_type, preset_id, intensity, grain_scale=None):
        if self.base_image is None:
            return

        # preset_type is now directly the string 'film' or None
        # preset_id is now directly the string ID or None

        logger.debug(
            "Preset preview requested: Type=%s, ID=%s, Intensity=%s, Grain=%s",
            preset_type,
            preset_id,
            intensity,
            grain_scale,
        )

        # Get current adjustments from the panel *before* applying the preset preview
        # This allows the preview to stack on existing adjustments
        current_adjustments = self.adjustment_panel.get_adjustments()

        # Create a temporary adjustments dictionary for the preview
        preview_adjustments = current_adjustments.copy()

        # Add preset info to the adjustments dict for processing
        preview_adjustments['preset_info'] = {
            'type': preset_type,
            'id': preset_id,
            'intensity': intensity,
            'grain_scale': grain_scale # Will be None if not applicable
        }

        # Request processing with the combined adjustments for preview
        # Use the *current base image* for the preview calculation
        self.request_processing(preview_adjustments)
        self.statusBar().showMessage(f"Previewing {preset_type} preset: {preset_id}...", 2000)


    # Slot to handle apply requests from either preset panel
    # Updated signature: preset_type is now a string, preset_id is the string ID
    def handle_preset_apply(self, base_image_ignored, preset_type, preset_id, intensity, grain_scale=None):
        if self.base_image is None:
            return

        # preset_type is now directly the string 'film' or None
        # preset_id is now directly the string ID or None

        logger.info(
            "Applying preset: Type=%s, ID=%s, Intensity=%s, Grain=%s",
            preset_type,
            preset_id,
            intensity,
            grain_scale,
        )
        self.statusBar().showMessage(f"Applying {preset_type} preset: {preset_id}...")
        QApplication.processEvents()

        try:
            # Store state before applying destructively
            self._store_previous_state()

            modified_image = None

            if preset_type in ("film", "photo"):
                modified_image = self.conversion_service.apply_preset(
                    self.base_image,
                    PresetInfo(
                        type=preset_type,
                        id=preset_id,
                        intensity=float(intensity),
                        grain_scale=grain_scale,
                    ),
                )
            else:
                raise ValueError(f"Unknown preset type '{preset_type}' for apply.")


            if modified_image is not None:
                # Update the base image
                self.base_image = modified_image

                # Reset adjustment sliders as the base image has fundamentally changed
                self.adjustment_panel.reset_adjustments() # Correct method name

                # Update the viewer and histogram with the new base image
                self.image_viewer.set_image(self.base_image)
                self.histogram_widget.update_histogram(self.base_image)
 
                # Store details of the applied preset for batch processing
                self._active_preset_details = {
                    'type': preset_type,
                    'id': preset_id,
                    'intensity': intensity
                }
                if preset_type == 'film':
                    self._active_preset_details['grain_scale'] = grain_scale
 
                self.statusBar().showMessage(f"{preset_type} preset '{preset_id}' applied.", 3000)
            else:
                raise ValueError(f"Applying {preset_type} preset '{preset_id}' returned None.")

        except Exception as e:
            QMessageBox.critical(self, "Preset Error", f"Error applying {preset_type} preset '{preset_id}':\n{e}")
            self.statusBar().showMessage(f"Error applying preset: {e}", 5000)
            # Don't clear undo state here, allow user to undo the failed attempt if state was stored
        finally:
            self.update_ui_state() # Update UI state (e.g., enable undo)


    # --- Negative Type Handling ---

    def _handle_neg_type_label_click(self, event):
        """Handles clicks on the negative type status label."""
        if self.raw_loaded_image is None or self._is_converting_initial:
            return # Don't allow change if no image or currently converting

        current_type = self._current_negative_type if self._current_negative_type else "Auto"
        # Define possible types (including Auto/None which triggers default detection)
        # Should match the types the backend converter understands for override
        possible_types = ["Auto", "C41", "B&W", "E6", "Other/Clear"] # Add more as supported

        # Ask user to select a new type
        new_type, ok = QInputDialog.getItem(
            self,
            "Override Negative Type",
            "Select negative base type (Auto uses detection):",
            possible_types,
            possible_types.index(current_type) if current_type in possible_types else 0,
            editable=False
        )

        if ok and new_type != current_type:
            logger.info("User selected new negative type override: %s", new_type)
            # Map UI selection to backend override value (None for Auto)
            override_value = new_type if new_type != "Auto" else None
            # Trigger re-conversion with the selected override
            self._rerun_conversion_with_override(override_value)


    def _rerun_conversion_with_override(self, override_type):
        """Internal helper to trigger re-conversion with a specific mask type."""
        if self.raw_loaded_image is None:
            self.statusBar().showMessage("No image loaded to re-convert.", 3000)
            return
        if self._is_converting_initial:
            self.statusBar().showMessage("Initial conversion already in progress.", 3000)
            return

        logger.info("Requesting re-conversion with override type: %s", override_type)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor
        self.statusBar().showMessage(f"Re-converting with type: {override_type}...", 0) # Persistent message
        self._is_converting_initial = True
        self.update_ui_state() # Disable controls during conversion
        # Emit the signal with the raw image and the override type
        self.initial_conversion_requested.emit(self.raw_loaded_image, override_type)

    # --- Settings Dialog ---
    def open_settings_dialog(self):
        """Opens the application settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            # Settings were saved successfully by the dialog's accept() method
            logger.info("Settings dialog accepted.")
            # Reload settings that can be applied dynamically (UI, Logging Level)
            try:
                app_settings.reload_settings()
                self.statusBar().showMessage("Settings saved and reloaded.", 3000)
                # Trigger updates in components that use reloaded settings
                self.filmstrip_widget.update_thumbnail_size() # Update filmstrip thumbnails
            except Exception as e:
                logger.exception("Error reloading settings")
                self.statusBar().showMessage("Settings saved, but failed to reload dynamically.", 5000)
                QMessageBox.warning(self, "Reload Warning", f"Settings were saved, but an error occurred during dynamic reload:\n{e}\n\nA restart might be needed for all changes to take effect.")
        else:
            # Settings dialog was cancelled
            logger.info("Settings dialog cancelled.")
            self.statusBar().showMessage("Settings changes cancelled.", 3000)


# --- Application Entry Point ---

def main():
    """Main function to run the application."""

    # --- Logging ---
    # Logging is configured automatically when 'utils.logger' is imported
    # by any component (e.g., MainWindow indirectly).
    # The level is determined by 'config.settings.LOGGING_LEVEL'.
    # We can optionally add a print statement here to confirm the level used.
    logger.info("Logging level set to: %s (via config/settings.py)", app_settings.LOGGING_LEVEL)

    # Create the Qt Application
    app = QApplication(sys.argv)

    # Optional: Set application details
    app.setApplicationName("Negative Converter")
    app.setOrganizationName("ExampleOrg") # Replace if desired
    # app.setWindowIcon(QIcon("path/to/icon.png")) # Set application icon

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    # This block ensures the code runs only when the script is executed directly
    main()
