# Photo Style Preset controls
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSlider, QSizePolicy, QInputDialog, QMessageBox, QScrollArea) # Added QInputDialog, QMessageBox
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon
import json # Added for dummy presets in main
import numpy as np # Added for dummy image in main

from negative_converter.utils.logger import get_logger

# Assuming PhotoPresetManager is in the processing package
try:
    from ..processing.photo_presets import PhotoPresetManager
except ImportError:
    # Fallback for running script directly or if structure differs
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from processing.photo_presets import PhotoPresetManager

logger = get_logger(__name__)


class PhotoPresetPanel(QWidget):
    """Panel for selecting and adjusting photo style presets"""
    # Signal emitted when preview needs to be updated
    # image, preset_type ('photo' or None), preset_id (or None), intensity
    preview_requested = pyqtSignal(object, object, object, float)
    # Signal emitted when changes should be applied permanently
    # image, preset_type ('photo' or None), preset_id (or None), intensity
    apply_requested = pyqtSignal(object, object, object, float)

    def __init__(self, main_window, preset_manager=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window # Store reference to main window

        # Instantiate PhotoPresetManager
        self.preset_manager = preset_manager if preset_manager else PhotoPresetManager()

        self.current_preset_id = None
        self.intensity = 1.0 # Default intensity (0.0 to 1.0)
        # No grain scale needed for photo presets

        self.setup_ui()
        self.load_presets_ui() # Populate buttons after UI setup

    def setup_ui(self):
        """Set up the UI elements"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins
        main_layout.setSpacing(10)

        # --- Photo Preset Selector Area ---
        photo_presets_group_label = QLabel("Photo Style Presets")
        photo_presets_group_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        main_layout.addWidget(photo_presets_group_label)

        # Scrollable container for many presets
        self._presets_container = QWidget()
        self.presets_layout = QGridLayout(self._presets_container)
        self.presets_layout.setSpacing(5) # Spacing between buttons
        self.preset_buttons = {} # Dictionary to store buttons by preset_id

        self._presets_scroll = QScrollArea()
        self._presets_scroll.setWidgetResizable(True)
        self._presets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._presets_scroll.setWidget(self._presets_container)

        main_layout.addWidget(self._presets_scroll, 1)

        # --- Sliders Area ---
        # Intensity slider
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity:")
        intensity_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        intensity_layout.addWidget(intensity_label)
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100) # 0% to 100%
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip("Adjust the overall strength of the photo style effect (0-100%).")
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_value_label = QLabel("100%") # Display current value
        self.intensity_value_label.setMinimumWidth(35) # Ensure space for "100%"
        intensity_layout.addWidget(self.intensity_value_label)
        main_layout.addLayout(intensity_layout)

        # --- No Grain Slider ---

        # --- Advanced Parameters (Placeholder - Might not apply to photo styles) ---
        # self.advanced_button = QPushButton("Advanced Parameters...")
        # self.advanced_button.setToolTip("Open a dialog for fine-tuning style parameters (Not Implemented).")
        # self.advanced_button.clicked.connect(self.on_advanced_clicked)
        # self.advanced_button.setEnabled(False)
        # main_layout.addWidget(self.advanced_button)

        # --- Action Buttons Area ---
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Apply the current photo style settings permanently to the image.")
        self.apply_button.clicked.connect(self.trigger_apply)
        self.apply_button.setEnabled(False) # Disabled until a preset is selected

        button_layout.addStretch(1) # Push buttons to the right
        button_layout.addWidget(self.apply_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_presets_ui(self):
        """Load photo presets and create buttons in the UI"""
        # Clear existing buttons first (if re-loading)
        for i in reversed(range(self.presets_layout.count())):
            widget = self.presets_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.preset_buttons.clear()

        all_presets = self.preset_manager.get_all_presets()
        if not all_presets:
            self.presets_layout.addWidget(QLabel("No photo presets found."), 0, 0)
            return

        # Group presets by category
        categories = {}
        for preset_id, preset_data in all_presets.items():
            category = preset_data.get("category", "Uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(preset_data)

        # Sort categories and presets within categories
        sorted_categories = sorted(categories.keys())
        row = 0
        max_cols = 3 # Number of buttons per row

        for category in sorted_categories:
            # Add category label
            label = QLabel(category)
            label.setStyleSheet("font-weight: bold; margin-top: 8px;")
            self.presets_layout.addWidget(label, row, 0, 1, max_cols) # Span across columns
            row += 1

            # Add preset buttons for this category
            col = 0
            sorted_presets = sorted(categories[category], key=lambda p: p.get("name", ""))
            for preset in sorted_presets:
                preset_id = preset["id"]
                button_text = preset.get("name", preset_id) # Fallback to ID if name missing
                button = QPushButton(button_text)
                button.setCheckable(True)
                button.setProperty("preset_id", preset_id) # Store ID for retrieval
                button.setToolTip(preset.get("description", "")) # Add description as tooltip
                button.clicked.connect(self.on_preset_selected) # Connect to handler
                self.preset_buttons[preset_id] = button

                # Add thumbnail if available (optional)
                # thumbnail_path = preset.get("thumbnail")
                # if thumbnail_path:
                #    # Construct full path
                #    # full_thumb_path = ...
                #    # if os.path.exists(full_thumb_path):
                #    #     button.setIcon(QIcon(full_thumb_path))
                #    #     button.setIconSize(QSize(60, 40))
                #    pass

                self.presets_layout.addWidget(button, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1
            # Ensure next category starts on a new row if last row wasn't full
            if col != 0:
                row += 1

        # Add vertical stretch at the end of the grid
        self.presets_layout.setRowStretch(row, 1)


    def on_preset_selected(self):
        """Handle preset button selection, allowing deselection."""
        sender = self.sender()
        if not sender: return
        selected_preset_id = sender.property("preset_id")

        if sender.isChecked():
            # Uncheck all other buttons
            for preset_id, button in self.preset_buttons.items():
                if button != sender: button.setChecked(False)

            self.current_preset_id = selected_preset_id
            self.apply_button.setEnabled(True)
            logger.debug("Photo preset selected. preset_id=%s", self.current_preset_id)
            self.trigger_preview()
        else:
            # Deselected
            if self.current_preset_id == selected_preset_id:
                self._reset_selection()

    def _reset_selection(self):
        """Resets the UI state when no preset is selected."""
        self.current_preset_id = None
        self.apply_button.setEnabled(False)
        # Uncheck all buttons
        for button in self.preset_buttons.values(): button.setChecked(False)
        logger.debug("Photo preset deselected.")
        self.trigger_preview() # Trigger preview to show original


    def on_intensity_changed(self, value):
        """Handle intensity slider change"""
        self.intensity = value / 100.0 # Convert 0-100 to 0.0-1.0
        self.intensity_value_label.setText(f"{value}%")
        # Trigger preview update if a preset is selected
        if self.current_preset_id:
            self.trigger_preview()

    # Removed on_grain_changed
    # Removed on_advanced_clicked

    def trigger_preview(self):
        """Emit signal to request a preview update, handling None preset_id."""
        # The main window slot will handle getting the current image.
        # This panel only needs to emit the selected preset details.

        if self.current_preset_id:
            logger.debug(
                "Triggering photo preview. preset_id=%s intensity=%.2f",
                self.current_preset_id,
                self.intensity,
            )
            # Emit signal with type 'photo' - image object is None here.
            self.preview_requested.emit(None, 'photo', self.current_preset_id, self.intensity)
        else:
            logger.debug("Triggering photo preview for original image (no photo preset).")
            # Emit signal with None for type and id - image object is None here.
            self.preview_requested.emit(None, None, None, 0.0)


    def trigger_apply(self):
        """Emit signal to apply changes permanently"""
        if self.current_preset_id:
            logger.debug(
                "Triggering photo apply. preset_id=%s intensity=%.2f",
                self.current_preset_id,
                self.intensity,
            )
            # The main window slot will handle getting the current image.
            # Emit signal with type 'photo' - image object is None here.
            self.apply_requested.emit(None, 'photo', self.current_preset_id, self.intensity)
            # Removed image fetching logic
        else:
            logger.debug("Apply trigger ignored: no photo preset selected.")


    def get_current_selection(self):
        """Returns the currently selected preset ID and intensity."""
        if self.current_preset_id:
            return self.current_preset_id, self.intensity
        else:
            return None, 1.0 # Return defaults if nothing selected

    # Removed get_modified_preset_params (no grain slider)

# Example usage (for testing standalone)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import cv2

    # --- Create Dummy Presets ---
    dummy_photo_presets_file = "temp_photo_presets.json"
    dummy_photo_presets_data = [
        {"id": "original", "name": "Original", "category": "Base", "parameters": {}},
        {"id": "punch", "name": "Punch", "category": "Color", "description": "Vibrant colors", "parameters": {"contrast": 30, "saturation": 35}},
        {"id": "golden", "name": "Golden", "category": "Color", "description": "Warm tones", "parameters": {"temperature": 35, "saturation": 20}},
        {"id": "bw", "name": "B&W", "category": "Monochrome", "description": "Classic B&W", "parameters": {"saturation": -100, "contrast": 15}},
    ]
    with open(dummy_photo_presets_file, 'w') as f:
        json.dump(dummy_photo_presets_data, f, indent=2)

    # --- Dummy Main Window ---
    class DummyMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Photo Preset Panel Test")
            self.image = np.zeros((200, 300, 3), dtype=np.uint8)
            self.image[:, :150] = [255, 100, 100]
            self.image[:, 150:] = [100, 100, 255]
            self.preview_image = self.image.copy()

            # Instantiate Photo Preset Manager
            self.photo_preset_mgr = PhotoPresetManager(presets_file=dummy_photo_presets_file)

            # Instantiate the panel
            self.photo_preset_panel = PhotoPresetPanel(self, preset_manager=self.photo_preset_mgr)
            self.setCentralWidget(self.photo_preset_panel)

            # Connect signals
            self.photo_preset_panel.preview_requested.connect(self.handle_preview)
            self.photo_preset_panel.apply_requested.connect(self.handle_apply)

            self.setGeometry(300, 300, 350, 400)

        def get_current_image_for_processing(self):
            print("Main window: Providing current image for processing.")
            return self.image.copy()

        # Updated handle_preview signature
        def handle_preview(self, image, preset_type, preset_id, intensity):
            print("-" * 20)
            if preset_id is None:
                print("Main window: PREVIEW request for original image.")
                self.preview_image = image.copy()
            elif preset_type == 'photo':
                print(f"Main window: PREVIEW request for photo preset '{preset_id}', intensity {intensity:.2f}")
                try:
                    self.preview_image = self.photo_preset_mgr.apply_photo_preset(image, preset_id, intensity)
                    print(f"Main window: Photo preview simulation successful.")
                except Exception as e: print(f"Main window: Error during photo preview sim: {e}")
            else:
                 print(f"Main window: Received unknown preset type '{preset_type}' in preview handler.")
                 self.preview_image = image.copy() # Fallback

            cv2.imshow("Preview", cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


        # Updated handle_apply signature
        def handle_apply(self, image, preset_type, preset_id, intensity):
            print("-" * 20)
            if preset_id is None:
                 print("Main window: APPLY request ignored (no preset selected).")
                 return

            if preset_type == 'photo':
                print(f"Main window: APPLY request for photo preset '{preset_id}', intensity {intensity:.2f}")
                try:
                    applied_img = self.photo_preset_mgr.apply_photo_preset(image, preset_id, intensity)
                    print(f"Main window: Photo apply successful. Updating main image.")
                    self.image = applied_img
                except Exception as e: print(f"Main window: Error during photo apply sim: {e}")
            else:
                 print(f"Main window: Received unknown preset type '{preset_type}' in apply handler.")


            self.preview_image = self.image.copy()
            cv2.imshow("Preview", cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


    # --- Run Application ---
    app = QApplication(sys.argv)
    mainWin = DummyMainWindow()
    mainWin.show()
    exit_code = app.exec()

    # Clean up dummy presets
    try:
        os.remove(dummy_photo_presets_file)
        print("\nCleaned up temporary preset file.")
    except Exception as e:
        print(f"\nCould not remove temp preset file: {e}")

    cv2.destroyAllWindows()
    sys.exit(exit_code)