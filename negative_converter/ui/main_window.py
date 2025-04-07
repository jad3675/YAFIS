# Main application window
import sys
import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QMenu,
                             QStatusBar, QFileDialog, QMessageBox, QDockWidget, QApplication,
                             QToolBar, QPushButton, QProgressBar, QLabel, QComboBox, QSpinBox, QWidgetAction,
                             QInputDialog) # Added QInputDialog
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtCore import Qt, QSize, QThread, QObject, pyqtSignal, pyqtSlot, QMetaObject
import concurrent.futures
import math
# import os # Already imported

# Import UI components
from .image_viewer import ImageViewer
from .adjustment_panel import AdjustmentPanel
from .preset_panel import FilmPresetPanel
from .photo_preset_panel import PhotoPresetPanel
from .histogram_widget import HistogramWidget # Import Histogram
from .filmstrip_widget import BatchFilmstripWidget # Import Filmstrip

# Import IO and Processing components
from negative_converter.processing.adjustments import apply_all_adjustments # Import the new function using absolute path
# Standard imports assuming package structure is respected
from ..io import image_loader, image_saver
from ..processing import NegativeConverter, FilmPresetManager, PhotoPresetManager, ImageAdjustments
from ..processing.adjustments import AdvancedAdjustments
from ..processing.batch import process_batch_with_adjustments


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
            import traceback
            print(f"Error during background processing call: {e}")
            traceback.print_exc()
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
            import traceback
            print(f"Error during initial conversion call: {e}")
            traceback.print_exc()
            self.error.emit(f"Initial conversion failed: {e}")
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
            import traceback
            print(f"Error during batch processing call: {e}")
            traceback.print_exc()
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
    # Use the new signal with override parameter
    # Signal definition already updated, KEEPING
    initial_conversion_requested = pyqtSignal(object, object) # raw_image, override_type
    initial_conversion_finished = pyqtSignal(object, str) # Emits (converted_image, mask_classification)
    initial_conversion_error = pyqtSignal(str)


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

        # --- Processing Engines ---
        self.negative_converter = NegativeConverter(film_profile="C41")
        self.film_preset_manager = FilmPresetManager()
        self.photo_preset_manager = PhotoPresetManager()
        self.basic_adjuster = ImageAdjustments()
        self.advanced_adjuster = AdvancedAdjustments()

        # --- Background Processing Thread ---
        self.processing_thread = QThread(self)
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

        # Connect signals from panels
        self.adjustment_panel.adjustment_changed.connect(self.handle_adjustment_change)
        self.adjustment_panel.awb_requested.connect(self.handle_awb_request)
        self.adjustment_panel.auto_level_requested.connect(self.handle_auto_level_request)
        self.adjustment_panel.auto_color_requested.connect(self.handle_auto_color_request)
        self.adjustment_panel.auto_tone_requested.connect(self.handle_auto_tone_request)
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
        self.tabifyDockWidget(self.photo_preset_dock, self.histogram_dock) # Add histogram to tabs
        self.adjustment_dock.raise_()

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
        # edit_menu.addAction(self.undo_action) # Remove duplicate add

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
            view_menu.addAction(self.view_batch_toolbar_action)
        else:
            print("Warning: Could not find View menu to add Batch Toolbar action.")

        self.batch_toolbar.setVisible(False) # Start hidden

    def create_view_toolbar(self):
        """Create the toolbar for image view controls (zoom, etc.)."""
        self.view_toolbar = QToolBar("View Controls", self)
        self.view_toolbar.setObjectName("ViewToolbar")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.view_toolbar)

        # Zoom Actions
        self.zoom_in_action = QAction("Zoom In (+)", self)
        self.zoom_in_action.setStatusTip("Zoom in on the image")
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_in_action.triggered.connect(self.image_viewer.zoom_in)
        self.zoom_in_action.setEnabled(False)
        self.view_toolbar.addAction(self.zoom_in_action)

        self.zoom_out_action = QAction("Zoom Out (-)", self)
        self.zoom_out_action.setStatusTip("Zoom out of the image")
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.zoom_out_action.triggered.connect(self.image_viewer.zoom_out)
        self.zoom_out_action.setEnabled(False)
        self.view_toolbar.addAction(self.zoom_out_action)

        self.reset_zoom_action = QAction("Reset Zoom (1:1)", self)
        self.reset_zoom_action.setStatusTip("Reset zoom to 100%")
        self.reset_zoom_action.triggered.connect(self.image_viewer.reset_zoom)
        self.reset_zoom_action.setEnabled(False)
        self.view_toolbar.addAction(self.reset_zoom_action)

        self.fit_window_action = QAction("Fit to Window", self)
        self.fit_window_action.setStatusTip("Zoom image to fit the window")
        self.fit_window_action.triggered.connect(self.image_viewer.fit_to_window)
        self.fit_window_action.setEnabled(False)
        self.view_toolbar.addAction(self.fit_window_action)

        # Add view action for this toolbar
        view_menu = self.menuBar().findChild(QMenu, "&View")
        if view_menu:
            self.view_view_toolbar_action = self.view_toolbar.toggleViewAction()
            self.view_view_toolbar_action.setText("View Controls Toolbar")
            self.view_view_toolbar_action.setStatusTip("Show/Hide the View Controls Toolbar")
            view_menu.addAction(self.view_view_toolbar_action)
        else:
            print("Warning: Could not find View menu to add View Toolbar action.")
        self.view_toolbar.setVisible(True)

    def select_batch_output_directory(self):
        """Opens a dialog to select the output directory for batch processing."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory for Batch")
        if dir_path:
            self._batch_output_dir = dir_path
            self.output_dir_label.setText(f" Output: ...{os.path.basename(dir_path)} ")
            self.output_dir_label.setToolTip(f"Output Directory: {dir_path}")
            self.statusBar().showMessage(f"Batch output directory set to: {dir_path}", 3000)
            self.update_ui_state()
        else:
            self.statusBar().showMessage("Batch output directory selection cancelled.", 2000)

    def start_batch_processing(self): # noqa C901
        """Initiates the batch processing operation in a separate thread."""
        if self.batch_worker._is_running:
            QMessageBox.warning(self, "Batch Busy", "Batch processing is already in progress.")
            return

        # Use checked items instead of selected items for batch processing
        checked_files = self.filmstrip_widget.get_checked_image_paths()
        if not checked_files:
            QMessageBox.warning(self, "Batch Error", "No images checked in the filmstrip for processing.")
            return
        if not self._batch_output_dir or not os.path.isdir(self._batch_output_dir):
             QMessageBox.warning(self, "Batch Error", "Please select a valid output directory first.")
             return

        # --- Get Settings for Batch ---
        # 1. Basic Adjustments
        current_adjustments = self.adjustment_panel.get_adjustments()

        # 2. Active Preset (Film or Photo)
        film_id, film_intensity, film_grain = self.film_preset_panel.get_current_selection()
        photo_id, photo_intensity = self.photo_preset_panel.get_current_selection()

        active_preset_info = None
        if photo_id:
            active_preset_info = {'type': 'photo', 'id': photo_id, 'intensity': photo_intensity}
            print(f"Batch: Using Photo Preset '{photo_id}' (Intensity: {photo_intensity:.2f})")
        elif film_id:
            active_preset_info = {'type': 'film', 'id': film_id, 'intensity': film_intensity, 'grain_scale': film_grain}
            print(f"Batch: Using Film Preset '{film_id}' (Intensity: {film_intensity:.2f}, Grain: {film_grain:.1f}x)")
        else:
            print("Batch: No active preset selected.")

        # 3. Output Format and Quality
        output_format = self.format_combo.currentText()
        quality_value = self.quality_spinbox.value()
        quality_settings = {}
        if output_format.lower() == '.jpg':
            quality_settings['jpeg_quality'] = quality_value
        elif output_format.lower() == '.png':
            quality_settings['png_compression'] = quality_value
        # --- End Get Settings ---

        total_files = len(checked_files)
        print(f"Starting batch processing of {total_files} checked files to {self._batch_output_dir} (Format: {output_format}, Quality: {quality_settings})")
        self.statusBar().showMessage(f"Starting batch processing of {total_files} images...")

        # Disable UI during processing
        self.process_batch_action.setEnabled(False)
        self.select_output_dir_action.setEnabled(False)
        self.open_batch_action.setEnabled(False)
        self.adjustment_panel.setEnabled(False)
        self.film_preset_panel.setEnabled(False)
        self.photo_preset_panel.setEnabled(False)
        self.open_action.setEnabled(False)
        self.save_as_action.setEnabled(False)
        self.format_combo.setEnabled(False) # Disable format/quality widgets
        self.quality_spinbox.setEnabled(False)

        # Setup and show progress bar
        self.batch_progress_bar.setRange(0, total_files)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)

        # Pass checked files, format and quality settings to the worker
        # Pass checked files, adjustments, PRESET INFO, format and quality settings to the worker
        # Also need to pass the preset managers themselves for the backend to use
        self.batch_worker.set_inputs(
            checked_files, self._batch_output_dir, current_adjustments, active_preset_info,
            self.negative_converter, self.film_preset_manager, self.photo_preset_manager,
            output_format, quality_settings
        )
        QMetaObject.invokeMethod(self.batch_worker, 'run', Qt.ConnectionType.QueuedConnection)


    # --- Batch Processing Slots (Single instance) ---

    @pyqtSlot(int, int)
    def _handle_batch_progress(self, current, total):
        """Update the batch progress bar."""
        self.batch_progress_bar.setValue(current)
        self.statusBar().showMessage(f"Batch processing: {current}/{total} images...")

    @pyqtSlot(list)
    def _handle_batch_finished(self, results):
        """Handle completion of the batch processing job."""
        total_processed = len(results)
        self.statusBar().showMessage(f"Batch processing finished. Processed {total_processed} files.", 5000)
        self.batch_progress_bar.setVisible(False)

        # Re-enable UI elements that were disabled during batch processing
        # Let update_ui_state handle most re-enabling based on context
        self.update_ui_state()

        # Report results
        success_count = sum(1 for _, success, _ in results if success)
        fail_count = total_processed - success_count
        summary_message = f"Batch processing complete.\n\nSuccessfully processed: {success_count}\nFailed: {fail_count}\n\n"
        if fail_count > 0:
            summary_message += "Failed files (max 10 shown):\n"
            failed_shown = 0
            for path, success, msg in results:
                if not success and failed_shown < 10:
                    summary_message += f"- {os.path.basename(path)}: {msg}\n"
                    failed_shown += 1
            if fail_count > 10:
                 summary_message += f"...and {fail_count - 10} more.\n"
            QMessageBox.warning(self, "Batch Processing Finished with Errors", summary_message)
        else:
            QMessageBox.information(self, "Batch Processing Finished", summary_message)

    @pyqtSlot(str)
    def _handle_batch_error(self, error_message):
        """Handle errors reported by the batch worker setup or execution."""
        self.statusBar().showMessage(f"Batch processing error: {error_message}", 5000)
        self.batch_progress_bar.setVisible(False)

        # Re-enable UI elements that were disabled
        # Let update_ui_state handle most re-enabling based on context
        self.update_ui_state()

        QMessageBox.critical(self, "Batch Processing Error", f"An error occurred during batch processing:\n{error_message}")

    # --- Helper for Dynamic Quality Widget ---
    @pyqtSlot(str)
    def _update_quality_widget_visibility(self, selected_format):
        """Show/hide and configure the quality spinbox based on selected format."""
        fmt = selected_format.lower()
        show_quality = False
        if fmt == '.jpg' or fmt == '.jpeg':
            show_quality = True
            self.quality_spinbox.setRange(1, 100)
            self.quality_spinbox.setValue(95)
            self.quality_spinbox.setToolTip("JPEG Quality (1-100, higher is better)")
        elif fmt == '.png':
            show_quality = True
            self.quality_spinbox.setRange(0, 9)
            self.quality_spinbox.setValue(3)
            self.quality_spinbox.setToolTip("PNG Compression Level (0-9, higher is more compressed)")
        elif fmt == '.tif' or fmt == '.tiff':
            show_quality = False # No simple quality setting
            self.quality_spinbox.setToolTip("Quality setting not applicable for TIFF")
        else:
            show_quality = False
            self.quality_spinbox.setToolTip("")

        self.quality_label.setVisible(show_quality)
        self.quality_spinbox.setVisible(show_quality)
        # Don't call update_ui_state here; it will be called after full initialization
        # or when batch mode visibility changes.

    def update_ui_state(self):
        """Update the enabled/disabled state of UI elements."""
        can_process = self.base_image is not None # Is there an image ready for adjustments?
        can_save = can_process                   # Can we save the current state?
        is_processing_batch = self.batch_worker._is_running
        is_converting = self._is_converting_initial # Are we busy with initial conversion?

        # Determine overall UI enable state (disabled if converting or batching)
        enable_general_ui = not is_converting and not is_processing_batch

        # File Menu Actions
        self.save_as_action.setEnabled(can_save and enable_general_ui)
        self.open_action.setEnabled(enable_general_ui)
        self.open_batch_action.setEnabled(enable_general_ui)

        # Edit Menu Actions
        # Enable undo if previous state exists and UI is generally enabled
        self.undo_action.setEnabled(self.previous_base_image is not None and enable_general_ui)
        # Enable compare if a 'before' image exists in the viewer and UI is generally enabled
        self.compare_action.setEnabled(self.image_viewer.has_before_image() and enable_general_ui)
        # Keep compare button checked state consistent with viewer mode
        self.compare_action.setChecked(self.image_viewer._display_mode == 'before')

        # Panels (Adjustments, Presets) - Need an image and UI enabled
        enable_panels = can_process and enable_general_ui
        self.adjustment_panel.setEnabled(enable_panels)
        self.film_preset_panel.setEnabled(enable_panels)
        self.photo_preset_panel.setEnabled(enable_panels)
        # Update Photo Preset Panel Save Button State based on panel enablement
        if hasattr(self.photo_preset_panel, '_update_save_button_state'):
            self.photo_preset_panel._update_save_button_state()

        # View Controls Toolbar - Need an image and UI enabled
        enable_view_controls = can_process and enable_general_ui
        self.zoom_in_action.setEnabled(enable_view_controls)
        self.zoom_out_action.setEnabled(enable_view_controls)
        self.reset_zoom_action.setEnabled(enable_view_controls)
        self.fit_window_action.setEnabled(enable_view_controls)

        # Batch Controls Toolbar
        is_batch_mode = self.filmstrip_dock.isVisible()
        self.batch_toolbar.setVisible(is_batch_mode) # Show/hide based on filmstrip visibility

        # Enable batch controls only if in batch mode AND general UI is enabled
        enable_batch_controls = is_batch_mode and enable_general_ui

        can_start_batch = (enable_batch_controls and # Must be in batch mode and UI enabled
                           bool(self.filmstrip_widget.get_checked_image_paths()) and # Must have items checked
                           self._batch_output_dir is not None and # Must have output dir set
                           os.path.isdir(self._batch_output_dir))

        self.process_batch_action.setEnabled(can_start_batch)
        self.select_output_dir_action.setEnabled(enable_batch_controls)
        self.format_combo.setEnabled(enable_batch_controls)
        # Enable quality spinbox only if visible and batch controls are enabled
        self.quality_spinbox.setEnabled(enable_batch_controls and self.quality_spinbox.isVisible())


    # --- Image Loading/Saving ---

    def open_image(self):
        # Clear histogram before loading new image
        self.histogram_widget.clear_histogram()
        """Open a single image file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Negative Image", "", image_loader.SUPPORTED_FORMATS_FILTER)
        if self._is_converting_initial: # Prevent opening while converting
            QMessageBox.warning(self, "Busy", "Please wait for the current image to finish converting.")
            return

        if file_path:
            try:
                self.statusBar().showMessage(f"Loading {os.path.basename(file_path)}...")
                QApplication.processEvents() # Allow status message update

                # Load using the loader module (returns image, mode, size)
                raw_image, original_mode, file_size = image_loader.load_image(file_path)

                # Update status bar with file info immediately after loading attempt
                self._update_status_filename(file_path)
                self._update_status_size(file_size)
                self._update_status_mode(original_mode)
                self._update_status_neg_type(None) # Reset neg type on new load

                if raw_image is None:
                    raise ValueError("Failed to load image.")

                # --- Start Background Conversion ---
                self.current_file_path = file_path
                self.raw_loaded_image = raw_image # Store the raw image
                # Store metadata if load was successful
                self._current_file_size = file_size
                self._current_original_mode = original_mode
                self._is_converting_initial = True
                self.statusBar().showMessage("Converting negative (this may take a moment)...")
                # Clear previous image and reset state before starting conversion
                self.initial_converted_image = None
                self.base_image = None
                self.previous_base_image = None
                self.image_viewer.set_image(None) # Clear viewer
                self.adjustment_panel.reset_adjustments() # Reset sliders
                self.update_ui_state() # Disable controls during conversion
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor

                # Emit signal to start conversion in worker thread
                # Pass the stored raw image to the conversion worker
                # Emit signal with raw image and None for override (initial load)
                # Emit signal with raw image and None for override (initial load) - THIS WAS ALREADY ADDED, KEEPING
                self.initial_conversion_requested.emit(self.raw_loaded_image, None)
                # --- End Background Conversion ---

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open image:\n{e}")
                self.statusBar().showMessage(f"Error loading image: {e}", 5000)
                # Reset state if loading failed
                self.current_file_path = None
                self.raw_loaded_image = None # Clear raw image on failure
                self._current_file_size = None
                self._current_original_mode = None
                self._current_negative_type = None
                self._is_converting_initial = False # Ensure flag is reset on error
                # Reset status bar fields on load failure
                self._update_status_filename(None)
                self._update_status_size(None)
                self._update_status_mode(None)
                self._update_status_neg_type(None)
                self.update_ui_state() # Re-enable UI
                QApplication.restoreOverrideCursor() # Restore cursor
                self.image_viewer.set_image(None) # Clear viewer
                self.setWindowTitle("Negative Converter")
            finally:
                self.update_ui_state()

    def save_image_as(self):
        """Save the currently displayed image."""
        if self.base_image is None: # Should be disabled, but check anyway
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

                # Determine format and quality based on selected filter/extension
                quality_params = {}
                ext = os.path.splitext(file_path)[1].lower()
                # Get quality/compression settings based on format
                if ext in ['.jpg', '.jpeg']:
                    quality, ok = QInputDialog.getInt(self, "JPEG Quality", "Enter JPEG quality (1-100):", 95, 1, 100, 1)
                    if ok:
                        quality_params['quality'] = quality
                    else:
                        # User cancelled, proceed with default or handle as error? Using default for now.
                        quality_params['quality'] = 95
                        self.statusBar().showMessage("JPEG quality selection cancelled, using default (95).", 2000)
                elif ext == '.png':
                    compression, ok = QInputDialog.getInt(self, "PNG Compression", "Enter PNG compression level (0-9):", 3, 0, 9, 1)
                    if ok:
                        quality_params['png_compression'] = compression
                    else:
                        quality_params['png_compression'] = 3
                        self.statusBar().showMessage("PNG compression selection cancelled, using default (3).", 2000)
                elif ext == '.webp':
                    # WebP quality is 0-100, similar to JPEG
                    quality, ok = QInputDialog.getInt(self, "WebP Quality", "Enter WebP quality (0-100):", 90, 0, 100, 1)
                    if ok:
                        quality_params['quality'] = quality
                    else:
                        quality_params['quality'] = 90
                        self.statusBar().showMessage("WebP quality selection cancelled, using default (90).", 2000)
                # TIFF and other formats currently don't have quality options here

                # Save using the saver module
                success = image_saver.save_image(final_image_to_save, file_path, **quality_params)

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
        """Finds supported image files in a given folder."""
        # Extract extensions more robustly from the filter string
        supported_ext = []
        parts = image_loader.SUPPORTED_FORMATS_FILTER.split(';;')
        for part in parts:
            # Find extensions within parentheses like (*.jpg *.jpeg)
            import re
            match = re.search(r'\((.*?)\)', part)
            if match:
                ext_list = match.group(1).split()
                supported_ext.extend([ext.replace('*', '').lower() for ext in ext_list])
        supported_ext = tuple(set(supported_ext)) # Unique extensions

        image_files = []
        try:
            for fname in os.listdir(folder_path):
                if fname.lower().endswith(supported_ext):
                    full_path = os.path.join(folder_path, fname)
                    if os.path.isfile(full_path): # Ensure it's a file
                        image_files.append(full_path)
        except OSError as e:
            print(f"Error listing directory {folder_path}: {e}")
            raise # Re-raise the exception to be caught by the caller
        return sorted(image_files)

    def closeEvent(self, event):
        """Handle closing the application."""
        # Clean up threads
        print("Stopping background threads...")
        self.processing_thread.quit()
        if not self.processing_thread.wait(1000):
             print("Warning: Processing thread did not stop gracefully.")
        self.batch_thread.quit()
        if not self.batch_thread.wait(1000):
             print("Warning: Batch thread did not stop gracefully.")
        print("Threads stopped.")
        event.accept()


    # --- Adjustment and Processing Handlers ---

    @pyqtSlot(str)
    def handle_filmstrip_preview(self, file_path):
        """Load and preview an image selected from the filmstrip."""
        if not file_path or not os.path.exists(file_path):
            print(f"Preview requested for invalid path: {file_path}")
            return

        if self.batch_worker._is_running:
            self.statusBar().showMessage("Cannot preview while batch processing is active.", 2000)
            return

        try:
            self.statusBar().showMessage(f"Loading preview for {os.path.basename(file_path)}...")
            QApplication.processEvents()

            # --- Modified Workflow: Treat preview like opening a single image ---

            # 1. Load the ORIGINAL negative image
            img_rgb = image_loader.load_image(file_path)
            if img_rgb is None:
                raise ValueError("Failed to load image for preview.")

            self.current_file_path = file_path
            self.statusBar().showMessage("Converting negative for preview...")
            QApplication.processEvents()

            # 2. Perform initial conversion
            self.initial_converted_image = self.negative_converter.convert(img_rgb)
            if self.initial_converted_image is None:
                 raise ValueError("Negative conversion failed for preview.")

            # 3. Set as the current base image (like open_image)
            self._store_previous_state() # Store previous state before overwriting base_image
            self.base_image = self.initial_converted_image.copy()

            # 4. Reset adjustments panel to defaults for this new image
            self.adjustment_panel.reset_adjustments()

            # 5. Display the base (unadjusted) image
            self.image_viewer.set_image(self.base_image)

            # 6. Update window title and status bar
            self.setWindowTitle(f"Negative Converter - {os.path.basename(file_path)}")
            self.statusBar().showMessage(f"Previewing: {os.path.basename(file_path)}. Ready to adjust.", 5000)

            # 7. Explicitly update UI state to enable controls
            self.update_ui_state()

        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not generate preview for {os.path.basename(file_path)}:\n{e}")
            self.statusBar().showMessage(f"Preview error: {e}", 4000)


    def get_current_image_for_processing(self):
        """Returns the appropriate base image for applying adjustments or presets."""
        return self.base_image # Always use the current base image

    @pyqtSlot(dict)
    def handle_adjustment_change(self, adjustments_dict):
        """Request processing when an adjustment slider changes."""
        if self.base_image is None:
            return
        self.request_processing(adjustments_dict)


    def request_processing(self, adjustments):
        """Emit signal to request processing in the background thread."""
        if self.base_image is None:
            # print("Request processing called with no base image.")
            return

        if self.processing_worker._is_running:
            self._pending_adjustments = adjustments
            self._request_pending = True
        else:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Show busy cursor
            # Ensure adjustments is a dict before emitting to prevent TypeError
            if not isinstance(adjustments, dict):
                print(f"[Error] request_processing called with non-dict adjustments: {type(adjustments)}. Using empty dict.")
                adjustments = {} # Fallback to prevent crash
            self.processing_requested.emit(self.base_image, adjustments)


    @pyqtSlot(object)
    def _handle_processing_finished(self, processed_image):
        """Update the image viewer when background processing is done."""
        self.image_viewer.set_image(processed_image)
        self.histogram_widget.update_histogram(processed_image) # Update histogram
        self.statusBar().showMessage("Adjustments applied.", 1500)
        QApplication.restoreOverrideCursor() # Restore cursor

        # Process pending adjustments if any
        if self._request_pending:
            self._request_pending = False
            self.request_processing(self._pending_adjustments)
            self._pending_adjustments = None


    @pyqtSlot(str)
    def _handle_processing_error(self, error_message):
        """Show error message when background processing fails."""
        self.statusBar().showMessage(f"Processing error: {error_message}", 5000)
        QMessageBox.warning(self, "Processing Error", f"Could not apply adjustments:\n{error_message}")
        QApplication.restoreOverrideCursor() # Restore cursor
        # Also handle pending request reset on error
        if self._request_pending:
            self._request_pending = False
            self._pending_adjustments = None

    @pyqtSlot(str)
    def _handle_initial_conversion_error(self, error_message):
        """Handle errors during initial conversion."""
        QApplication.restoreOverrideCursor() # Restore cursor
        self._is_converting_initial = False # Reset flag
        QMessageBox.critical(self, "Conversion Error", f"Failed to convert negative:\n{error_message}")
        self.statusBar().showMessage(f"Error during initial conversion: {error_message}", 5000)
        self.update_ui_state() # Re-enable UI

    @pyqtSlot(int, int)
    def _handle_initial_conversion_progress(self, step, total_steps):
        """Update status bar with initial conversion progress."""
        if total_steps > 0:
            percent = int((step / total_steps) * 100)
            # Simple status message, could use a progress bar later
            self.statusBar().showMessage(f"Converting negative... Step {step}/{total_steps} ({percent}%)")
        else:
            self.statusBar().showMessage("Converting negative...")


    # --- Slots for Initial Conversion Worker ---

    @pyqtSlot(object, str) # Update slot signature
    def _handle_initial_conversion_finished(self, converted_image, mask_classification):
        """Handle the successfully converted image from the background thread."""
        QApplication.restoreOverrideCursor() # Restore cursor
        self._is_converting_initial = False # Reset flag

        if converted_image is None or converted_image.size == 0:
            QMessageBox.critical(self, "Error", "Initial conversion failed unexpectedly.")
            self.statusBar().showMessage("Initial conversion failed.", 5000)
            self.update_ui_state() # Re-enable UI
            return

        try:
            # Store results
            self.initial_converted_image = converted_image
            self._current_negative_type = mask_classification # Store classification
            self._store_previous_state() # Store None as the very first previous state
            self.base_image = self.initial_converted_image.copy() # Start with converted image as base

            # Update UI
            self.image_viewer.set_image(self.base_image) # Display the initially converted image
            self.histogram_widget.update_histogram(self.base_image) # Update histogram
            self.histogram_widget.update_histogram(self.base_image) # Update histogram
            self.statusBar().showMessage(f"Loaded and converted: {os.path.basename(self.current_file_path)}. Type: {mask_classification}", 5000)
            self._update_status_neg_type(mask_classification) # Update status bar label
            self.setWindowTitle(f"Negative Converter - {os.path.basename(self.current_file_path)}")
            self.update_ui_state() # Re-enable controls now that image is ready

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error displaying converted image:\n{e}")
            self.statusBar().showMessage(f"Error displaying image: {e}", 5000)
            # Reset state
            self.current_file_path = None
            self.initial_converted_image = None
            self.base_image = None
            self.raw_loaded_image = None # Clear raw image on display error too
            # If base_image is None, clear previous states
            self.previous_base_image = None
            self.image_viewer.set_before_image(None) # Clear before image in viewer
            self.image_viewer.set_image(None)
            self.update_ui_state()


    @pyqtSlot(str)
    def _handle_initial_conversion_error(self, error_message):
        """Handle errors during initial background conversion."""
        QApplication.restoreOverrideCursor() # Restore cursor
        self._is_converting_initial = False # Reset flag
        QMessageBox.critical(self, "Conversion Error", f"Failed to convert negative:\n{error_message}")
        self.statusBar().showMessage(f"Conversion error: {error_message}", 5000)
        # Reset state
        self.current_file_path = None
        self.raw_loaded_image = None # Clear raw image on conversion error
        self.initial_converted_image = None
        self.base_image = None
        self.previous_base_image = None
        self.image_viewer.set_image(None)
        self.update_ui_state() # Re-enable UI


    # --- Adjustment Application Logic ---

        if self._request_pending:
            self._request_pending = False
            pending_adj = self._pending_adjustments
            self._pending_adjustments = None # Clear before potentially re-requesting
            self.request_processing(pending_adj)
        else:
            self._pending_adjustments = None
        QApplication.restoreOverrideCursor() # Restore cursor

    @pyqtSlot(str)
    def _handle_processing_error(self, error_message):
        """Show error message if background processing fails."""
        QMessageBox.warning(self, "Processing Error", error_message)
        self.statusBar().showMessage(f"Error: {error_message}", 3000)
        if self._request_pending:
             self._request_pending = False
             self._pending_adjustments = None
        QApplication.restoreOverrideCursor() # Restore cursor


    def _process_pending_request(self):
         pass # Logic integrated into _handle_processing_finished


    def _apply_tile_adjustments(self, tile_image, adjustments):
         """Applies adjustments to a single tile (placeholder/example)."""
         # Likely unused/obsolete.
         return tile_image


    def _get_fully_adjusted_image(self, base_image, adjustments):
        """
        Applies all adjustments from the dictionary to the base image.
        """
        if base_image is None:
            return None
        try:
            adjusted_image = apply_all_adjustments(base_image, adjustments)
            return adjusted_image
        except Exception as e:
            print(f"Error in _get_fully_adjusted_image: {e}")
            import traceback
            traceback.print_exc()
            raise e


    # --- Undo Functionality ---
    def _store_previous_state(self):
        """Stores the current base_image as the previous state for undo AND as the 'before' image for compare."""
        if self.base_image is not None:
            # Store for Undo
            self.previous_base_image = self.base_image.copy()
            # Store for Compare (only store if not already in 'before' mode to avoid overwriting)
            if self.image_viewer._display_mode == 'after':
                 self.image_viewer.set_before_image(self.base_image)
            self.undo_action.setEnabled(True) # Enable undo immediately
            self.compare_action.setEnabled(True) # Enable compare immediately
        else:
            # If base_image is None, clear previous states
            self.previous_base_image = None
            self.image_viewer.set_before_image(None) # Clear before image in viewer
            self.undo_action.setEnabled(False)
            self.compare_action.setEnabled(False) # Disable compare if no image
    def _clear_previous_state(self):
        """Clears the stored previous state."""
        self.previous_base_image = None

    def undo_last_destructive_op(self):
        """Restores the base_image from the stored previous state."""
        if self.previous_base_image is not None:
            print("Performing Undo...")
            self.base_image = self.previous_base_image
            self.previous_base_image = None # Single level undo

            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage("Undo successful.", 2000)
        else:
            self.statusBar().showMessage("Nothing to undo.", 2000)
            print("Undo requested but no previous state stored.")

        self.update_ui_state() # Update UI after undo

    def toggle_compare_view(self):
        """Slot to toggle the image viewer's compare mode."""
        self.image_viewer.toggle_compare_mode()
        # Update the action's checked state to match the viewer's mode
        self.compare_action.setChecked(self.image_viewer._display_mode == 'before')


    # --- Auto Adjustment Handlers ---

    @pyqtSlot()
    def handle_wb_picker_request(self):
        """Activate color sampling mode in the image viewer."""
        if self.base_image is not None:
            self.image_viewer.enter_picker_mode() # Corrected method name
            self.statusBar().showMessage("Click on a neutral gray/white area to set White Balance.")
        else:
             self.statusBar().showMessage("Load an image first to use White Balance Picker.", 3000)


    @pyqtSlot(tuple)
    def handle_color_sampled(self, rgb_tuple):
        """Receives the sampled color and applies WB adjustment."""
        if rgb_tuple:
            try:
                self.statusBar().showMessage(f"Sampled RGB: {rgb_tuple}. Applying White Balance...")
                QApplication.processEvents()
                self._store_previous_state()

                # Corrected: Calculate scales first, then apply
                scales = self.advanced_adjuster.calculate_white_balance_from_color(self.base_image, rgb_tuple)
                if scales == (1.0, 1.0, 1.0): # Check if calculation failed or was identity
                    raise ValueError("White balance scale calculation failed or resulted in no change.")

                scale_r, scale_g, scale_b = scales
                img_float = self.base_image.astype(np.float32)
                img_float[..., 0] *= scale_r
                img_float[..., 1] *= scale_g # Usually 1.0
                img_float[..., 2] *= scale_b
                wb_adjusted_image = np.clip(img_float, 0, 255).astype(np.uint8)

                if wb_adjusted_image is None: raise ValueError("White balance adjustment failed.") # Should not happen if scales are valid
                self.base_image = wb_adjusted_image

                current_adjustments = self.adjustment_panel.get_adjustments()
                self.request_processing(current_adjustments)
                self.statusBar().showMessage("White Balance applied.", 3000)

            except Exception as e:
                 QMessageBox.critical(self, "White Balance Error", f"Failed to apply White Balance:\n{e}")
                 self.statusBar().showMessage(f"White Balance error: {e}", 5000)
            finally:
                 self.update_ui_state()
        else:
            self.statusBar().showMessage("White Balance picking cancelled.", 2000)


    @pyqtSlot(str) # Slot for AWB request from AdjustmentPanel
    def handle_awb_request(self, method):
        """Handle Automatic White Balance request."""
        if self.base_image is None: return
        try:
            self.statusBar().showMessage(f"Applying Auto White Balance ({method})...")
            QApplication.processEvents()
            self._store_previous_state()

            # Corrected: Use advanced_adjuster and apply_auto_white_balance
            awb_adjusted_image = self.advanced_adjuster.apply_auto_white_balance(self.base_image, method=method)
            if awb_adjusted_image is None: raise ValueError(f"Auto White Balance ({method}) failed.")
            self.base_image = awb_adjusted_image

            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage(f"Auto White Balance ({method}) applied.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Auto White Balance Error", f"Failed to apply Auto White Balance:\n{e}")
            self.statusBar().showMessage(f"AWB error: {e}", 5000)
        finally:
            self.update_ui_state()


    @pyqtSlot(str, float) # Slot for Auto Level request from AdjustmentPanel
    def handle_auto_level_request(self, mode, midrange):
        """Handle Auto Levels request."""
        if self.base_image is None: return
        try:
            self.statusBar().showMessage(f"Applying Auto Levels ({mode}, Mid: {midrange:.2f})...")
            QApplication.processEvents()
            self._store_previous_state()

            # Corrected: Use advanced_adjuster and apply_auto_levels
            leveled_image = self.advanced_adjuster.apply_auto_levels(self.base_image, colorspace_mode=mode, midrange=midrange)
            if leveled_image is None: raise ValueError("Auto Levels failed.")
            self.base_image = leveled_image

            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage("Auto Levels applied.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Auto Levels Error", f"Failed to apply Auto Levels:\n{e}")
            self.statusBar().showMessage(f"Auto Levels error: {e}", 5000)
        finally:
            self.update_ui_state()


    @pyqtSlot(str) # Slot for Auto Color request from AdjustmentPanel
    def handle_auto_color_request(self, method):
        """Handle Auto Color request."""
        if self.base_image is None: return
        try:
            self.statusBar().showMessage(f"Applying Auto Color ({method})...")
            QApplication.processEvents()
            self._store_previous_state()

            # Corrected: Use advanced_adjuster and apply_auto_color
            colored_image = self.advanced_adjuster.apply_auto_color(self.base_image, method=method)
            if colored_image is None: raise ValueError("Auto Color failed.")
            self.base_image = colored_image

            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage("Auto Color applied.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Auto Color Error", f"Failed to apply Auto Color:\n{e}")
            self.statusBar().showMessage(f"Auto Color error: {e}", 5000)
        finally:
            self.update_ui_state()


    @pyqtSlot() # Slot for Auto Tone request from AdjustmentPanel
    def handle_auto_tone_request(self):
        """Handle Auto Tone request using AdvancedAdjustments."""
        if self.base_image is None: return
        try:
            self.statusBar().showMessage("Applying Auto Tone...")
            QApplication.processEvents()
            self._store_previous_state()

            # Corrected: Use apply_auto_tone method name
            toned_image = self.advanced_adjuster.apply_auto_tone(self.base_image)
            if toned_image is None: raise ValueError("Auto Tone failed.")
            self.base_image = toned_image

            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage("Auto Tone applied.", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Auto Tone Error", f"Failed to apply Auto Tone:\n{e}")
            self.statusBar().showMessage(f"Auto Tone error: {e}", 5000)
        finally:
            self.update_ui_state()


    # --- Preset Handlers ---

    # Removed explicit slot decorator to allow connection with varying arguments (Film vs Photo presets)
    # Made grain_scale optional with a default value
    def handle_preset_preview(self, base_image_ignored, preset_type_obj, preset_id_obj, intensity, grain_scale=None):
        """Preview a film or photo preset."""
        if self.base_image is None: return

        # Handle unselection case
        if preset_id_obj is None:
            self.statusBar().showMessage("Preset unselected. Reverting preview.", 2000)
            self.request_processing(self.adjustment_panel.get_adjustments())
            return

        try:
            # Convert received objects (likely strings) to expected types if necessary,
            # or adjust logic to use them directly. Assuming they are strings for now.
            preset_type = str(preset_type_obj)
            preset_id = str(preset_id_obj)

            self.statusBar().showMessage(f"Previewing {preset_type} preset: {preset_id}...")
            QApplication.processEvents()

            # Determine the correct manager and method based on preset_type (case-insensitive)
            if preset_type.lower() == "film":
                manager = self.film_preset_manager
                apply_method = manager.apply_preset
            elif preset_type.lower() == "photo":
                manager = self.photo_preset_manager
                apply_method = manager.apply_photo_preset # Correct method name
            else:
                # Raise error here if type is still unknown after None check
                raise ValueError(f"Unknown preset type: {preset_type}")

            current_adjustments = self.adjustment_panel.get_adjustments()
            adjusted_base = self._get_fully_adjusted_image(self.base_image, current_adjustments)
            if adjusted_base is None: raise ValueError("Failed to apply base adjustments for preview.")

            # Call the correct method with appropriate arguments (case-insensitive check)
            if preset_type.lower() == "film":
                # FilmPresetManager.apply_preset does NOT take grain_scale
                preview_image = apply_method(adjusted_base, preset_id, intensity=intensity)
            else: # Photo preset doesn't use grain_scale in its apply method
                preview_image = apply_method(adjusted_base, preset_id, intensity=intensity)

            if preview_image is None: raise ValueError("Preset application failed for preview.")

            self.image_viewer.set_image(preview_image)
            self.statusBar().showMessage(f"Previewing {preset_id} (Intensity: {intensity:.2f})", 3000)

        except Exception as e:
            QMessageBox.warning(self, "Preset Preview Error", f"Could not preview preset {preset_id}:\n{e}")
            self.statusBar().showMessage(f"Preset preview error: {e}", 4000)
            # Revert preview if error occurs
            self.request_processing(self.adjustment_panel.get_adjustments())


    # Removed explicit slot decorator to allow connection with varying arguments (Film vs Photo presets)
    # Made grain_scale optional with a default value
    def handle_preset_apply(self, base_image_ignored, preset_type_obj, preset_id_obj, intensity, grain_scale=None):
        """Apply a film or photo preset destructively."""
        if self.base_image is None: return

        # Handle unselection case (though apply shouldn't usually be triggered on unselect)
        if preset_id_obj is None:
            print("Warning: handle_preset_apply called with None preset_id.")
            return

        # Convert received objects
        preset_type = str(preset_type_obj)
        preset_id = str(preset_id_obj)

        reply = QMessageBox.question(self, "Apply Preset",
                                     f"Applying '{preset_id}' will modify the base image and reset basic adjustments.\n"
                                     "This action can be undone once.\n\nProceed?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.No:
            self.statusBar().showMessage("Preset application cancelled.", 2000)
            # Revert preview if cancelled
            self.request_processing(self.adjustment_panel.get_adjustments())
            return

        try:
            self.statusBar().showMessage(f"Applying {preset_type} preset: {preset_id}...")
            QApplication.processEvents()
            self._store_previous_state()

            # Determine the correct manager and method based on preset_type (case-insensitive)
            if preset_type.lower() == "film":
                manager = self.film_preset_manager
                apply_method = manager.apply_preset
            elif preset_type.lower() == "photo":
                manager = self.photo_preset_manager
                apply_method = manager.apply_photo_preset # Correct method name
            else:
                raise ValueError(f"Unknown preset type: {preset_type}")

            # Call the correct method with appropriate arguments (case-insensitive check)
            if preset_type.lower() == "film":
                 # FilmPresetManager.apply_preset does NOT take grain_scale
                 new_base_image = apply_method(self.base_image, preset_id, intensity=intensity)
            else: # Photo preset doesn't use grain_scale
                 new_base_image = apply_method(self.base_image, preset_id, intensity=intensity)

            if new_base_image is None: raise ValueError("Preset application failed.")
            self.base_image = new_base_image

            self.adjustment_panel.reset_adjustments()
            current_adjustments = self.adjustment_panel.get_adjustments()
            self.request_processing(current_adjustments)
            self.statusBar().showMessage(f"Applied preset: {preset_id}", 3000)

        except Exception as e:
            QMessageBox.critical(self, "Preset Apply Error", f"Failed to apply preset {preset_id}:\n{e}")
            self.statusBar().showMessage(f"Preset apply error: {e}", 5000)
            # Attempt to revert preview if apply fails
            self.request_processing(self.adjustment_panel.get_adjustments())
        finally:
            self.update_ui_state()


    # --- Manual Negative Type Selection ---

    def _handle_neg_type_label_click(self, event):
        """Handles clicks on the negative type status label."""
        # Check if we have the necessary data and are not busy
        if self.raw_loaded_image is None or self._is_converting_initial:
            self.statusBar().showMessage("Load an image first or wait for conversion to finish.", 2000)
            return

        current_type = self._current_negative_type or "Unknown/Other"
        # Define the canonical list of types the user can select
        available_types = ["Likely C-41", "Clear/Near Clear", "Unknown/Other"]

        # Ensure current_type is valid before trying to remove
        if current_type not in available_types:
             current_type = "Unknown/Other" # Default if current is somehow invalid

        # Prepare the list for the dialog, potentially removing the current type
        display_types = available_types[:] # Make a copy
        if current_type in display_types:
             display_types.remove(current_type) # Remove current to suggest alternatives first

        # Use QInputDialog to get user selection
        item, ok = QInputDialog.getItem(self, "Select Negative Type",
                                        f"Current type is '{current_type}'.\nChoose override type:",
                                        display_types, 0, False) # editable=False

        if ok and item:
            # Check if the selected type is actually different
            if item == self._current_negative_type:
                self.statusBar().showMessage(f"Selected type '{item}' is already active.", 2000)
                return

            # Proceed with re-conversion
            self.statusBar().showMessage(f"Re-converting with type: {item}...")
            QApplication.processEvents() # Update UI message
            self._rerun_conversion_with_override(item)
        else:
            self.statusBar().showMessage("Negative type selection cancelled.", 2000)


    def _rerun_conversion_with_override(self, override_type):
        """Triggers the conversion process again with a specified override type."""
        if self.raw_loaded_image is None:
            QMessageBox.warning(self, "Error", "No raw image data available for re-conversion.")
            return
        if self._is_converting_initial: # Double-check we're not already busy
             QMessageBox.warning(self, "Busy", "Already processing, please wait.")
             return

        self._is_converting_initial = True
        # Clear previous results before starting re-conversion
        self.initial_converted_image = None
        self.base_image = None
        self.previous_base_image = None # Reset undo state
        self.image_viewer.set_image(None) # Clear viewer
        self.adjustment_panel.reset_adjustments() # Reset sliders as conversion changes base image
        self.update_ui_state() # Disable UI elements
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor) # Set busy cursor
        self.statusBar().showMessage(f"Re-converting with type: {override_type}...") # Update status

        # Emit signal with stored raw image and the selected override type
        self.initial_conversion_requested.emit(self.raw_loaded_image, override_type)

# --- End of MainWindow Class ---


# --- Entry Point ---
def main():
    """Application entry point."""
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


    # --- Definitions moved inside MainWindow class ---
if __name__ == "__main__":
    main()
