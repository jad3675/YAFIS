# Film simulation controls
import os
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QSlider, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QIcon
import json # Added for dummy presets in main
import numpy as np # Added for dummy image in main

# Assuming FilmPresetManager is in the processing package
# Adjust the import path based on your final project structure
try:
    from ..processing.film_simulation import FilmPresetManager
except ImportError:
    # Fallback for running script directly or if structure differs
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from processing.film_simulation import FilmPresetManager


class FilmPresetPanel(QWidget): # Renamed back
    """Panel for selecting and adjusting film presets"""
    # Signal emitted when preview needs to be updated
    # image, preset_type ('film' or None), preset_id (or None), intensity, grain_scale
    preview_requested = pyqtSignal(object, object, object, float, float)
    # Signal emitted when changes should be applied permanently
    # image, preset_type ('film' or None), preset_id (or None), intensity, grain_scale
    apply_requested = pyqtSignal(object, object, object, float, float)

    def __init__(self, main_window, preset_manager=None, parent=None): # Removed photo_preset_manager
        super().__init__(parent)
        self.main_window = main_window # Store reference to main window

        # Instantiate only FilmPresetManager
        self.preset_manager = preset_manager if preset_manager else FilmPresetManager()

        self.current_preset_id = None
        # Removed self.current_preset_type
        self.intensity = 1.0 # Default intensity (0.0 to 1.0)
        self.grain_scale = 1.0 # Default grain scale (0.0 to 2.0, maps to slider 0-100)

        self.setup_ui()
        self.load_presets_ui() # Populate buttons after UI setup

    def setup_ui(self):
        """Set up the UI elements"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5) # Reduced margins
        main_layout.setSpacing(10)

        # --- Film Preset Selector Area ---
        film_presets_group_label = QLabel("Film Simulation Presets")
        film_presets_group_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        main_layout.addWidget(film_presets_group_label)

        # Using QGridLayout for film preset buttons
        self.presets_layout = QGridLayout() # Renamed back from film_presets_layout
        self.presets_layout.setSpacing(5) # Spacing between buttons
        self.preset_buttons = {} # Renamed back from film_preset_buttons
        main_layout.addLayout(self.presets_layout)

        # --- Removed Photo Preset Selector Area ---

        # --- Sliders Area ---
        # Intensity slider
        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Intensity:")
        intensity_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        intensity_layout.addWidget(intensity_label)
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100) # 0% to 100%
        self.intensity_slider.setValue(100)
        self.intensity_slider.setToolTip("Adjust the overall strength of the film simulation effect (0-100%).") # Updated tooltip
        self.intensity_slider.valueChanged.connect(self.on_intensity_changed)
        intensity_layout.addWidget(self.intensity_slider)
        self.intensity_value_label = QLabel("100%") # Display current value
        self.intensity_value_label.setMinimumWidth(35) # Ensure space for "100%"
        intensity_layout.addWidget(self.intensity_value_label)
        main_layout.addLayout(intensity_layout)

        # Grain slider
        grain_layout = QHBoxLayout()
        self.grain_label = QLabel("Grain:") # Store label reference
        self.grain_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        grain_layout.addWidget(self.grain_label)
        self.grain_slider = QSlider(Qt.Orientation.Horizontal)
        self.grain_slider.setRange(0, 100) # Maps to 0x to 2x grain intensity
        self.grain_slider.setValue(50) # Default 1x grain
        self.grain_slider.setToolTip("Adjust the intensity of the simulated film grain (0-100%, where 50% is default).") # Updated tooltip
        self.grain_slider.valueChanged.connect(self.on_grain_changed)
        grain_layout.addWidget(self.grain_slider)
        self.grain_value_label = QLabel("1.0x") # Display current scale
        self.grain_value_label.setMinimumWidth(35) # Ensure space for "2.0x"
        grain_layout.addWidget(self.grain_value_label)
        main_layout.addLayout(grain_layout)

        # --- Advanced Parameters (Placeholder) ---
        self.advanced_button = QPushButton("Advanced Parameters...")
        self.advanced_button.setToolTip("Open a dialog for fine-tuning simulation parameters (Not Implemented).")
        self.advanced_button.clicked.connect(self.on_advanced_clicked)
        self.advanced_button.setEnabled(False) # Disabled for now
        # main_layout.addWidget(self.advanced_button) # Optional: Hide if not functional

        # Add stretch to push buttons to the bottom
        main_layout.addStretch(1)

        # --- Action Buttons Area ---
        button_layout = QHBoxLayout()
        self.preview_button = QPushButton("Preview")
        self.preview_button.setToolTip("Update the main image preview with the current settings.")
        self.preview_button.clicked.connect(self.trigger_preview)
        self.preview_button.setEnabled(False) # Disabled until a preset is selected

        self.apply_button = QPushButton("Apply")
        self.apply_button.setToolTip("Apply the current film simulation settings permanently to the image.") # Updated tooltip
        self.apply_button.clicked.connect(self.trigger_apply)
        self.apply_button.setEnabled(False) # Disabled until a preset is selected

        button_layout.addStretch(1) # Push buttons to the right
        button_layout.addWidget(self.preview_button)
        button_layout.addWidget(self.apply_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def load_presets_ui(self):
        """Load film presets and create buttons in the UI""" # Updated docstring
        # Clear existing buttons first (if re-loading)
        for i in reversed(range(self.presets_layout.count())):
            widget = self.presets_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        self.preset_buttons.clear()

        all_presets = self.preset_manager.get_all_presets() # Using self.preset_manager again
        if not all_presets:
            self.presets_layout.addWidget(QLabel("No film presets found."), 0, 0) # Updated text
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
                button.clicked.connect(self.on_preset_selected) # Renamed back handler
                self.preset_buttons[preset_id] = button

                # Add thumbnail if available (optional)
                # if "thumbnail" in preset and os.path.exists(preset["thumbnail"]):
                #     button.setIcon(QIcon(preset["thumbnail"]))
                #     button.setIconSize(QSize(60, 40)) # Adjust size as needed

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

    # Removed _load_photo_presets_ui

    def on_preset_selected(self): # Renamed back from on_film_preset_selected
        """Handle preset button selection, allowing deselection.""" # Updated docstring
        sender = self.sender()
        if not sender: return
        selected_preset_id = sender.property("preset_id")

        if sender.isChecked():
            # Uncheck all other buttons
            for preset_id, button in self.preset_buttons.items():
                if button != sender: button.setChecked(False)
            # Removed unchecking of photo buttons

            self.current_preset_id = selected_preset_id
            # Removed self.current_preset_type = 'film'
            self.preview_button.setEnabled(True)
            self.apply_button.setEnabled(True)
            self.grain_slider.setEnabled(True) # Ensure grain is enabled
            self.grain_label.setEnabled(True)
            print(f"Preset selected: {self.current_preset_id}") # Updated log
            self.trigger_preview()
        else:
            # Deselected
            if self.current_preset_id == selected_preset_id:
                self._reset_selection()

    # Removed on_photo_preset_selected

    def _reset_selection(self):
        """Resets the UI state when no preset is selected."""
        self.current_preset_id = None
        # Removed self.current_preset_type = None
        self.preview_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.grain_slider.setEnabled(True) # Ensure grain slider is enabled
        self.grain_label.setEnabled(True)
        # Uncheck all buttons
        for button in self.preset_buttons.values(): button.setChecked(False)
        # Removed unchecking photo buttons
        print("Preset deselected.")
        self.trigger_preview() # Trigger preview to show original


    def on_intensity_changed(self, value):
        """Handle intensity slider change"""
        self.intensity = value / 100.0 # Convert 0-100 to 0.0-1.0
        self.intensity_value_label.setText(f"{value}%")
        # Trigger preview update if a preset is selected
        if self.current_preset_id:
            self.trigger_preview()

    def on_grain_changed(self, value):
        """Handle grain slider change"""
        # Map 0-100 slider value to 0.0x - 2.0x scale, with 50 as 1.0x
        self.grain_scale = value / 50.0
        self.grain_value_label.setText(f"{self.grain_scale:.1f}x")
         # Trigger preview update if a preset is selected (always film preset now)
        if self.current_preset_id:
            self.trigger_preview()

    def on_advanced_clicked(self):
        """Show advanced parameters dialog (Placeholder)"""
        print("Advanced parameters clicked (Not Implemented)")
        # Implementation would involve creating a QDialog
        # to show/edit parameters from self.preset_manager.get_preset(self.current_preset_id)

    def trigger_preview(self):
        """Emit signal to request a preview update, handling None preset_id."""
        # The main window slot will handle getting the current image.
        # This panel only needs to emit the selected preset details.

        if self.current_preset_id:
            # A preset is selected
            print(f"Triggering film preview for {self.current_preset_id} with intensity {self.intensity:.2f}, grain {self.grain_scale:.1f}x")
            # Emit signal with type 'film'
            # Emit signal with type 'film' - image object is None here, MainWindow will provide it.
            self.preview_requested.emit(None, 'film', self.current_preset_id, self.intensity, self.grain_scale)
        else:
            # No preset is selected (deselected state)
            print("Triggering preview for original image (no preset).")
            # Emit signal with None for type and id. Intensity/grain don't matter here.
            # Emit signal with None for type and id - image object is None here.
            self.preview_requested.emit(None, None, None, 0.0, 0.0)


    def trigger_apply(self):
        """Emit signal to apply changes permanently"""
        if self.current_preset_id:
            print(f"Triggering apply for {self.current_preset_id} with intensity {self.intensity:.2f}, grain {self.grain_scale:.1f}x")
            # The main window slot will handle getting the current image.
            # Emit signal with type 'film' - image object is None here.
            self.apply_requested.emit(None, 'film', self.current_preset_id, self.intensity, self.grain_scale)
            # Removed image fetching logic
        else:
            print("Apply trigger: No preset selected.")

    def get_modified_preset_params(self, preset_id, grain_scale): # Renamed back
        """Helper to get preset parameters modified by UI controls (like grain)"""
        preset = self.preset_manager.get_preset(preset_id) # Using self.preset_manager
        if not preset: return None

        import copy
        modified_params = copy.deepcopy(preset.get("parameters", {}))

        # Adjust grain intensity based on slider scale
        if "grainParams" in modified_params:
            original_intensity = modified_params["grainParams"].get("intensity", 0)
            modified_params["grainParams"]["intensity"] = original_intensity * grain_scale
        elif grain_scale != 1.0: # Add grain params if slider moved and none exist
             modified_params["grainParams"] = {"intensity": 10 * grain_scale, "size": 1.0, "roughness": 0.5} # Example defaults

        return modified_params

    def get_current_selection(self):
        """Returns the currently selected preset ID, intensity, and grain scale."""
        if self.current_preset_id:
            return self.current_preset_id, self.intensity, self.grain_scale
        else:
            return None, 1.0, 1.0 # Return defaults if nothing selected

    # Removed photo preset helper

# Example usage (for testing standalone) - Updated to only test FilmPresetPanel
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow
    import cv2 # Added for dummy image display

    # --- Create Dummy Presets ---
    dummy_preset_dir = "temp_film_presets"
    os.makedirs(dummy_preset_dir, exist_ok=True)
    dummy_presets_data = {
        "k25": {"id": "k25", "name": "Kodachrome 25", "category": "Slide", "description": "Classic slide film", "parameters": {"grainParams": {"intensity": 5}}},
        "v50": {"id": "v50", "name": "Velvia 50", "category": "Slide", "description": "Vivid landscape film", "parameters": {"grainParams": {"intensity": 8}}},
        "tx400": {"id": "tx400", "name": "Tri-X 400", "category": "B&W", "description": "Classic B&W", "parameters": {"grainParams": {"intensity": 15}}},
    }
    for pid, pdata in dummy_presets_data.items():
        with open(os.path.join(dummy_preset_dir, f"{pid}.json"), 'w') as f:
            json.dump(pdata, f, indent=2)

    # Removed dummy photo preset creation

    # --- Dummy Main Window ---
    class DummyMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Film Preset Panel Test") # Updated title
            # Create a dummy image
            self.image = np.zeros((200, 300, 3), dtype=np.uint8)
            self.image[:, :150] = [255, 100, 100] # Left side
            self.image[:, 150:] = [100, 100, 255] # Right side
            self.preview_image = self.image.copy() # Separate image for preview display

            # Instantiate Preset Manager with dummy paths
            self.preset_mgr = FilmPresetManager(preset_directory=dummy_preset_dir)

            # Instantiate the panel
            self.preset_panel = FilmPresetPanel(self, preset_manager=self.preset_mgr) # Use FilmPresetPanel
            self.setCentralWidget(self.preset_panel) # Add panel to window

            # Connect signals
            self.preset_panel.preview_requested.connect(self.handle_preview)
            self.preset_panel.apply_requested.connect(self.handle_apply)

            self.setGeometry(300, 300, 350, 400) # Adjusted size

        def get_current_image_for_processing(self):
            # Method the panel calls to get the image
            print("Main window: Providing current image for processing.")
            return self.image.copy() # Return a copy of the *original* image

        # Updated handle_preview signature
        def handle_preview(self, image, preset_type, preset_id, intensity, grain_scale):
            print("-" * 20)
            if preset_id is None:
                print("Main window: PREVIEW request for original image.")
                self.preview_image = image.copy() # Show original
            elif preset_type == 'film':
                # A preset IS selected, proceed with simulation
                print(f"Main window: PREVIEW request for preset '{preset_id}', intensity {intensity:.2f}, grain {grain_scale:.1f}x")
                try:
                    # Get modified params based on UI
                    modified_params = self.preset_panel.get_modified_preset_params(preset_id, grain_scale)
                    if modified_params:
                        preset_dict = {"parameters": modified_params} # Create temp dict for apply_preset
                        self.preview_image = self.preset_mgr.apply_preset(image, preset_dict, intensity)
                        print(f"Main window: Preview simulation successful.")
                    else: print(f"Main window: Could not get modified preset parameters for preview.")
                except Exception as e: print(f"Main window: Error during preview simulation: {e}")
            else:
                 print(f"Main window: Received unknown preset type '{preset_type}' in preview handler.")
                 self.preview_image = image.copy() # Fallback

            # Display the preview (simple OpenCV window for testing)
            cv2.imshow("Preview", cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1) # Allow window to update


        # Updated handle_apply signature
        def handle_apply(self, image, preset_type, preset_id, intensity, grain_scale):
            print("-" * 20)
            if preset_id is None:
                 print("Main window: APPLY request ignored (no preset selected).")
                 return # Cannot apply 'None'

            if preset_type == 'film':
                print(f"Main window: APPLY request for preset '{preset_id}', intensity {intensity:.2f}, grain {grain_scale:.1f}x")
                try:
                    modified_params = self.preset_panel.get_modified_preset_params(preset_id, grain_scale)
                    if modified_params:
                        preset_dict = {"parameters": modified_params}
                        applied_img = self.preset_mgr.apply_preset(image, preset_dict, intensity)
                        print(f"Main window: Apply simulation successful. Updating main image.")
                        self.image = applied_img # Update the main image permanently
                    else: print("Main window: Could not get modified preset parameters for apply.")
                except Exception as e: print(f"Main window: Error during apply simulation: {e}")
            else:
                 print(f"Main window: Received unknown preset type '{preset_type}' in apply handler.")


            # Update UI to show the new permanent image (just update preview window here)
            self.preview_image = self.image.copy()
            cv2.imshow("Preview", cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


    # --- Run Application ---
    app = QApplication(sys.argv)
    mainWin = DummyMainWindow()
    mainWin.show()
    exit_code = app.exec()

    # Clean up dummy presets
    import shutil
    try:
        shutil.rmtree(dummy_film_preset_dir)
        # Removed photo preset file cleanup
        print("\nCleaned up temporary preset files.")
    except Exception as e:
        print(f"\nCould not remove temp preset files: {e}")

    cv2.destroyAllWindows() # Close OpenCV window
    sys.exit(exit_code)