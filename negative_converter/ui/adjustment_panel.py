# Adjustment controls
import copy # Added for deep copying defaults
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox, QComboBox,
                             QDoubleSpinBox, QSizePolicy, QGroupBox, QFormLayout, QPushButton) # Added QPushButton
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon

# Import the placeholder curves widget
from .curves_widget import CurvesWidget

class AdjustmentPanel(QWidget):
    """Panel for basic image adjustments."""

    # Signal emitted when any adjustment value changes
    adjustment_changed = pyqtSignal(dict)
    # Signal emitted when Auto White Balance is requested
    awb_requested = pyqtSignal(str) # Emits the selected AWB method name ('gray_world' or 'white_patch')
    # Signal emitted when Auto Level is requested
    auto_level_requested = pyqtSignal(str, float) # Emits colorspace_mode, midrange
    # Signal emitted when Auto Color is requested
    auto_color_requested = pyqtSignal(str) # Emits the selected method name ('gamma', 'recolor', 'none')
    # Signal emitted when Auto Tone is requested
    auto_tone_requested = pyqtSignal()
    wb_picker_requested = pyqtSignal() # Signal to activate WB picker mode


    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Store Default Adjustments ---
        self._default_adjustments = {
            'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0, 'temp': 0, 'tint': 0,
            'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0, 'levels_out_black': 0, 'levels_out_white': 255,
            'curves_rgb': [[0, 0], [255, 255]], 'curves_red': [[0, 0], [255, 255]], 'curves_green': [[0, 0], [255, 255]], 'curves_blue': [[0, 0], [255, 255]],
            'mixer_out_channel': 'Red', # This itself isn't an adjustment value to compare
            'mixer_red_r': 100, 'mixer_red_g': 0, 'mixer_red_b': 0, 'mixer_red_const': 0,
            'mixer_green_r': 0, 'mixer_green_g': 100, 'mixer_green_b': 0, 'mixer_green_const': 0,
            'mixer_blue_r': 0, 'mixer_blue_g': 0, 'mixer_blue_b': 100, 'mixer_blue_const': 0,
            'noise_reduction_strength': 0,
            'dust_removal_enabled': False,
            'dust_removal_sensitivity': 50, # Default sensitivity (higher = more sensitive)
            'dust_removal_radius': 3,      # Default inpainting radius
            'hsl_color': 'Reds', # This itself isn't an adjustment value to compare
            'hsl_reds_h': 0, 'hsl_reds_s': 0, 'hsl_reds_l': 0,
            'hsl_yellows_h': 0, 'hsl_yellows_s': 0, 'hsl_yellows_l': 0,
            'hsl_greens_h': 0, 'hsl_greens_s': 0, 'hsl_greens_l': 0,
            'hsl_cyans_h': 0, 'hsl_cyans_s': 0, 'hsl_cyans_l': 0,
            'hsl_blues_h': 0, 'hsl_blues_s': 0, 'hsl_blues_l': 0,
            'hsl_magentas_h': 0, 'hsl_magentas_s': 0, 'hsl_magentas_l': 0,
            'sel_color': 'Reds', # This itself isn't an adjustment value to compare
            'sel_reds_c': 0, 'sel_reds_m': 0, 'sel_reds_y': 0, 'sel_reds_k': 0,
            'sel_yellows_c': 0, 'sel_yellows_m': 0, 'sel_yellows_y': 0, 'sel_yellows_k': 0,
            'sel_greens_c': 0, 'sel_greens_m': 0, 'sel_greens_y': 0, 'sel_greens_k': 0,
            'sel_cyans_c': 0, 'sel_cyans_m': 0, 'sel_cyans_y': 0, 'sel_cyans_k': 0,
            'sel_blues_c': 0, 'sel_blues_m': 0, 'sel_blues_y': 0, 'sel_blues_k': 0,
            'sel_magentas_c': 0, 'sel_magentas_m': 0, 'sel_magentas_y': 0, 'sel_magentas_k': 0,
            'sel_whites_c': 0, 'sel_whites_m': 0, 'sel_whites_y': 0, 'sel_whites_k': 0,
            'sel_neutrals_c': 0, 'sel_neutrals_m': 0, 'sel_neutrals_y': 0, 'sel_neutrals_k': 0,
            'sel_blacks_c': 0, 'sel_blacks_m': 0, 'sel_blacks_y': 0, 'sel_blacks_k': 0,
            'sel_relative': True
        }
        # Make a deep copy for current adjustments to avoid shared references (esp. for curves)
        self._current_adjustments = copy.deepcopy(self._default_adjustments)

        # --- History Stacks ---
        self._undo_stack = []
        self._redo_stack = []
        self._history_limit = 50 # Max number of undo steps

        # --- Group Box Tracking ---
        self.group_boxes = {}
        self.original_group_titles = {}
        self.group_adjustment_keys = {
            'basic': ['brightness', 'contrast', 'saturation', 'hue', 'temp', 'tint'],
            'levels': ['levels_in_black', 'levels_in_white', 'levels_gamma', 'levels_out_black', 'levels_out_white'],
            'curves': ['curves_rgb', 'curves_red', 'curves_green', 'curves_blue'],
            'mixer': [
                'mixer_red_r', 'mixer_red_g', 'mixer_red_b', 'mixer_red_const',
                'mixer_green_r', 'mixer_green_g', 'mixer_green_b', 'mixer_green_const',
                'mixer_blue_r', 'mixer_blue_g', 'mixer_blue_b', 'mixer_blue_const'
            ],
            'nr': ['noise_reduction_strength'],
            'dust_removal': ['dust_removal_enabled', 'dust_removal_sensitivity', 'dust_removal_radius'],
            'hsl': [
                'hsl_reds_h', 'hsl_reds_s', 'hsl_reds_l',
                'hsl_yellows_h', 'hsl_yellows_s', 'hsl_yellows_l',
                'hsl_greens_h', 'hsl_greens_s', 'hsl_greens_l',
                'hsl_cyans_h', 'hsl_cyans_s', 'hsl_cyans_l',
                'hsl_blues_h', 'hsl_blues_s', 'hsl_blues_l',
                'hsl_magentas_h', 'hsl_magentas_s', 'hsl_magentas_l'
            ],
            'selective_color': [
                'sel_reds_c', 'sel_reds_m', 'sel_reds_y', 'sel_reds_k',
                'sel_yellows_c', 'sel_yellows_m', 'sel_yellows_y', 'sel_yellows_k',
                'sel_greens_c', 'sel_greens_m', 'sel_greens_y', 'sel_greens_k',
                'sel_cyans_c', 'sel_cyans_m', 'sel_cyans_y', 'sel_cyans_k',
                'sel_blues_c', 'sel_blues_m', 'sel_blues_y', 'sel_blues_k',
                'sel_magentas_c', 'sel_magentas_m', 'sel_magentas_y', 'sel_magentas_k',
                'sel_whites_c', 'sel_whites_m', 'sel_whites_y', 'sel_whites_k',
                'sel_neutrals_c', 'sel_neutrals_m', 'sel_neutrals_y', 'sel_neutrals_k',
                'sel_blacks_c', 'sel_blacks_m', 'sel_blacks_y', 'sel_blacks_k',
                'sel_relative' # Include boolean flag
            ]
        }

        self.setup_ui()
        self._update_all_group_titles() # Set initial titles correctly
        self._update_history_buttons_state() # Set initial button state

    def setup_ui(self):
        """Set up the UI elements for adjustments."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(10)

        # --- Add Reset/Undo/Redo Buttons at the top ---
        top_button_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo Adj")
        self.undo_button.setToolTip("Undo last adjustment change in this panel")
        self.undo_button.clicked.connect(self.undo_adjustment)
        self.undo_button.setEnabled(False)
        top_button_layout.addWidget(self.undo_button)

        self.redo_button = QPushButton("Redo Adj")
        self.redo_button.setToolTip("Redo last undone adjustment change in this panel")
        self.redo_button.clicked.connect(self.redo_adjustment)
        self.redo_button.setEnabled(False)
        top_button_layout.addWidget(self.redo_button)

        top_button_layout.addStretch(1) # Spacer

        self.reset_all_button = QPushButton("Reset All Adjustments")
        self.reset_all_button.setToolTip("Reset all adjustments in this panel to their default values")
        self.reset_all_button.clicked.connect(self.reset_adjustments)
        top_button_layout.addWidget(self.reset_all_button)

        main_layout.addLayout(top_button_layout)
        # --- End Top Buttons ---

        # --- Basic Adjustments Group ---
        basic_group = QGroupBox("Basic Adjustments")
        self.group_boxes['basic'] = basic_group # Store reference
        self.original_group_titles['basic'] = basic_group.title() # Store original title
        basic_group.setCheckable(True); basic_group.setChecked(True)
        basic_content_widget = QWidget()
        form_layout = QFormLayout(basic_content_widget)
        form_layout.setContentsMargins(10, 5, 10, 10); form_layout.setSpacing(10)

        self.brightness_slider = self._create_slider(-100, 100, 0, self._on_brightness_change)
        self.brightness_label = QLabel("0")
        form_layout.addRow("Brightness:", self._create_slider_layout(self.brightness_slider, self.brightness_label))

        self.contrast_slider = self._create_slider(-100, 100, 0, self._on_contrast_change)
        self.contrast_label = QLabel("0")
        form_layout.addRow("Contrast:", self._create_slider_layout(self.contrast_slider, self.contrast_label))

        self.saturation_slider = self._create_slider(-100, 100, 0, self._on_saturation_change)
        self.saturation_label = QLabel("0")
        form_layout.addRow("Saturation:", self._create_slider_layout(self.saturation_slider, self.saturation_label))

        self.hue_slider = self._create_slider(-180, 180, 0, self._on_hue_change)
        self.hue_label = QLabel("0")
        form_layout.addRow("Hue:", self._create_slider_layout(self.hue_slider, self.hue_label))

        # --- White Balance Section (Temp, Tint, Auto WB, Auto Color, Auto Tone) ---
        wb_ac_at_layout = QHBoxLayout() # Layout for Temp/Tint sliders + Auto controls

        # Temp/Tint Sliders
        wb_sliders_layout = QFormLayout()
        self.temp_slider = self._create_slider(-100, 100, 0, self._on_temp_change)
        self.temp_label = QLabel("0")
        wb_sliders_layout.addRow("Temperature:", self._create_slider_layout(self.temp_slider, self.temp_label))
        self.tint_slider = self._create_slider(-100, 100, 0, self._on_tint_change)
        self.tint_label = QLabel("0")
        # Create a horizontal layout for Tint slider, label, and picker button
        tint_layout = QHBoxLayout()
        tint_layout.setContentsMargins(0,0,0,0) # Remove margins for the inner layout
        tint_layout.setSpacing(5) # Add some spacing
        tint_layout.addWidget(self.tint_slider)
        tint_layout.addWidget(self.tint_label)
        # Picker button will be added to this layout later

        # Add WB Picker Button
        self.wb_picker_button = QPushButton()
        # Attempt to find a suitable icon (requires icon file or theme)
        # Placeholder: Use text if icon not found
        # Icon loading failed or is unavailable, use text fallback
        self.wb_picker_button.setText("WB Pick")
        # Optional: Keep the warning or remove it if the text button is acceptable
        print("Warning: Icon for WB button not found or loaded. Using text fallback.")

        self.wb_picker_button.setToolTip("Pick White Balance point from image")
        # self.wb_picker_button.setFixedSize(QSize(24, 24)) # REMOVE fixed size
        # Ensure connection points to the correct signal
        self.wb_picker_button.clicked.connect(self.wb_picker_requested.emit)

        # Add the picker button to the horizontal layout next to the tint slider/label
        tint_layout.addWidget(self.wb_picker_button)
        tint_layout.addStretch(1) # Add stretch to push button towards label/slider if needed

        # Add the combined horizontal layout for Tint+Picker to the form
        wb_sliders_layout.addRow("Tint:", tint_layout)
        # Alternatively, could create a QHBoxLayout for Tint slider+label+button

        wb_ac_at_layout.addLayout(wb_sliders_layout) # Add sliders layout (now includes picker button) first

        # Auto Controls (AWB + Auto Color + Auto Tone) in a vertical layout
        auto_controls_layout = QVBoxLayout()
        auto_controls_layout.setSpacing(10) # Add some spacing between auto groups

        # Auto White Balance Controls
        awb_group_layout = QHBoxLayout()
        self.awb_method_combo = QComboBox()
        self.awb_method_combo.addItems(['Gray World', 'White Patch', 'Simple WB', 'Learning WB'])
        self.awb_method_combo.setToolTip("Select Auto White Balance algorithm")
        self.awb_apply_button = QPushButton("Apply Auto WB")
        self.awb_apply_button.setToolTip("Calculate and apply auto white balance using the selected method")
        self.awb_apply_button.clicked.connect(self._on_awb_apply_clicked)
        awb_group_layout.addWidget(self.awb_method_combo)
        awb_group_layout.addWidget(self.awb_apply_button)
        auto_controls_layout.addLayout(awb_group_layout)

        # Auto Color Controls
        ac_group_layout = QHBoxLayout()
        self.ac_method_map = {'Gamma': 'gamma', 'Recolor': 'recolor', 'None': 'none'}
        self.ac_method_combo = QComboBox()
        self.ac_method_combo.addItems(self.ac_method_map.keys())
        self.ac_method_combo.setCurrentText('Gamma') # Default
        self.ac_method_combo.setToolTip("Select Auto Color adjustment method")
        self.ac_apply_button = QPushButton("Apply Auto Color")
        self.ac_apply_button.setToolTip("Calculate and apply auto color balance and contrast stretch")
        self.ac_apply_button.clicked.connect(self._on_auto_color_apply_clicked)
        ac_group_layout.addWidget(self.ac_method_combo)
        ac_group_layout.addWidget(self.ac_apply_button)
        auto_controls_layout.addLayout(ac_group_layout)

        # Auto Tone Control (Just a button for now)
        at_group_layout = QHBoxLayout()
        self.at_apply_button = QPushButton("Apply Auto Tone")
        self.at_apply_button.setToolTip("Apply a sequence of automatic adjustments (NR, AWB, AutoLevel, Clarity)")
        self.at_apply_button.clicked.connect(self._on_auto_tone_apply_clicked)
        at_group_layout.addWidget(self.at_apply_button)
        at_group_layout.addStretch(1) # Push button left
        auto_controls_layout.addLayout(at_group_layout)

        auto_controls_layout.addStretch(1) # Push controls up
        wb_ac_at_layout.addLayout(auto_controls_layout) # Add combined auto controls next to sliders

        form_layout.addRow(wb_ac_at_layout) # Add the combined WB/AC/AT layout to the main form

        # --- End White Balance / Auto Color / Auto Tone Section ---

        basic_group_layout = QVBoxLayout(basic_group)
        basic_group_layout.setContentsMargins(0, 15, 0, 0); basic_group_layout.addWidget(basic_content_widget)
        basic_group.toggled.connect(basic_content_widget.setVisible)
        main_layout.addWidget(basic_group)

        # --- Levels Adjustments Group ---
        levels_group = QGroupBox("Levels")
        self.group_boxes['levels'] = levels_group # Store reference
        self.original_group_titles['levels'] = levels_group.title() # Store original title
        levels_group.setCheckable(True); levels_group.setChecked(False)
        levels_content_widget = QWidget(); levels_layout = QFormLayout(levels_content_widget)
        levels_layout.setContentsMargins(10, 5, 10, 10); levels_layout.setSpacing(10)
        input_levels_layout = QHBoxLayout()
        self.levels_in_black_spin = QSpinBox(); self.levels_in_black_spin.setRange(0, 254); self.levels_in_black_spin.setValue(self._current_adjustments['levels_in_black']); self.levels_in_black_spin.setToolTip("Input Black Point (0-254)"); self.levels_in_black_spin.valueChanged.connect(self._on_levels_change)
        self.levels_in_white_spin = QSpinBox(); self.levels_in_white_spin.setRange(1, 255); self.levels_in_white_spin.setValue(self._current_adjustments['levels_in_white']); self.levels_in_white_spin.setToolTip("Input White Point (1-255)"); self.levels_in_white_spin.valueChanged.connect(self._on_levels_change)
        self.levels_gamma_spin = QDoubleSpinBox(); self.levels_gamma_spin.setRange(0.1, 10.0); self.levels_gamma_spin.setSingleStep(0.1); self.levels_gamma_spin.setDecimals(2); self.levels_gamma_spin.setValue(self._current_adjustments['levels_gamma']); self.levels_gamma_spin.setToolTip("Gamma Correction (0.1-10.0)"); self.levels_gamma_spin.valueChanged.connect(self._on_levels_change)
        input_levels_layout.addWidget(QLabel("In:")); input_levels_layout.addWidget(self.levels_in_black_spin); input_levels_layout.addWidget(self.levels_gamma_spin); input_levels_layout.addWidget(self.levels_in_white_spin)
        levels_layout.addRow("Input Levels:", input_levels_layout)
        output_levels_layout = QHBoxLayout()
        self.levels_out_black_spin = QSpinBox(); self.levels_out_black_spin.setRange(0, 254); self.levels_out_black_spin.setValue(self._current_adjustments['levels_out_black']); self.levels_out_black_spin.setToolTip("Output Black Point (0-254)"); self.levels_out_black_spin.valueChanged.connect(self._on_levels_change)
        self.levels_out_white_spin = QSpinBox(); self.levels_out_white_spin.setRange(1, 255); self.levels_out_white_spin.setValue(self._current_adjustments['levels_out_white']); self.levels_out_white_spin.setToolTip("Output White Point (1-255)"); self.levels_out_white_spin.valueChanged.connect(self._on_levels_change)
        output_levels_layout.addWidget(QLabel("Out:")); output_levels_layout.addWidget(self.levels_out_black_spin); output_levels_layout.addStretch(1); output_levels_layout.addWidget(self.levels_out_white_spin)
        levels_layout.addRow("Output Levels:", output_levels_layout)

        # --- Auto Level Controls ---
        auto_level_layout = QHBoxLayout()
        auto_level_layout.setSpacing(5)
        self.al_colormode_combo = QComboBox()
        # Use title case for display, lowercase for internal use
        self.al_colormode_map = {'Luminance': 'luminance', 'Lightness': 'lightness', 'Brightness': 'brightness', 'Gray': 'gray', 'Average': 'average', 'RGB': 'rgb'}
        self.al_colormode_combo.addItems(self.al_colormode_map.keys())
        self.al_colormode_combo.setCurrentText('Luminance') # Default
        self.al_colormode_combo.setToolTip("Select colorspace/channel for Auto Level statistics")

        self.al_midrange_spin = QDoubleSpinBox()
        self.al_midrange_spin.setDecimals(2) # Set decimals first
        self.al_midrange_spin.setRange(0.01, 0.99)
        self.al_midrange_spin.setValue(0.50) # Default
        self.al_midrange_spin.setSingleStep(0.01) # Set step last
        self.al_midrange_spin.setToolTip("Target midrange for Auto Level gamma (0.01-0.99)")
        # self.al_midrange_spin.setFixedWidth(60) # Removed fixed width to allow layout sizing

        self.al_apply_button = QPushButton("Apply Auto Level")
        self.al_apply_button.setToolTip("Calculate and apply auto levels using the selected method and midrange")
        self.al_apply_button.clicked.connect(self._on_auto_level_apply_clicked)

        auto_level_layout.addWidget(QLabel("Auto:"))
        auto_level_layout.addWidget(self.al_colormode_combo)
        auto_level_layout.addWidget(QLabel("Mid:"))
        auto_level_layout.addWidget(self.al_midrange_spin)
        auto_level_layout.addWidget(self.al_apply_button)
        auto_level_layout.addStretch(1)
        levels_layout.addRow(auto_level_layout) # Add the auto level controls row
        # --- End Auto Level Controls ---

        levels_group_layout = QVBoxLayout(levels_group); levels_group_layout.setContentsMargins(0, 15, 0, 0); levels_group_layout.addWidget(levels_content_widget)
        levels_group.toggled.connect(levels_content_widget.setVisible); levels_content_widget.setVisible(False)
        main_layout.addWidget(levels_group)

        # --- Curves Adjustment Group ---
        curves_group = QGroupBox("Curves")
        self.group_boxes['curves'] = curves_group # Store reference
        self.original_group_titles['curves'] = curves_group.title() # Store original title
        curves_group.setCheckable(True); curves_group.setChecked(False)
        curves_content_widget = QWidget(); curves_content_layout = QVBoxLayout(curves_content_widget)
        curves_content_layout.setContentsMargins(10, 5, 10, 10)
        self.curves_widget = CurvesWidget(); self.curves_widget.curve_changed.connect(self._on_curve_change)
        curves_content_layout.addWidget(self.curves_widget)
        curves_group_layout_outer = QVBoxLayout(curves_group); curves_group_layout_outer.setContentsMargins(0, 15, 0, 0); curves_group_layout_outer.addWidget(curves_content_widget)
        curves_group.toggled.connect(curves_content_widget.setVisible); curves_content_widget.setVisible(False)
        main_layout.addWidget(curves_group)

        # --- Channel Mixer Group ---
        mixer_group = QGroupBox("Channel Mixer")
        self.group_boxes['mixer'] = mixer_group # Store reference
        self.original_group_titles['mixer'] = mixer_group.title() # Store original title
        mixer_group.setCheckable(True); mixer_group.setChecked(False)
        mixer_content_widget = QWidget(); mixer_layout = QVBoxLayout(mixer_content_widget)
        mixer_layout.setContentsMargins(10, 5, 10, 10); mixer_layout.setSpacing(10)
        mixer_channel_layout = QHBoxLayout(); mixer_channel_layout.addWidget(QLabel("Output Channel:"))
        self.mixer_channel_combo = QComboBox(); self.mixer_channel_combo.addItems(['Red', 'Green', 'Blue']); self.mixer_channel_combo.currentTextChanged.connect(self._on_mixer_channel_change)
        mixer_channel_layout.addWidget(self.mixer_channel_combo); mixer_channel_layout.addStretch(1); mixer_layout.addLayout(mixer_channel_layout)
        mixer_sliders_layout = QFormLayout(); mixer_sliders_layout.setSpacing(10)
        self.mixer_r_slider = self._create_slider(-200, 200, 100, self._on_mixer_value_change); self.mixer_r_label = QLabel("100"); mixer_sliders_layout.addRow("Red:", self._create_slider_layout(self.mixer_r_slider, self.mixer_r_label))
        self.mixer_g_slider = self._create_slider(-200, 200, 0, self._on_mixer_value_change); self.mixer_g_label = QLabel("0"); mixer_sliders_layout.addRow("Green:", self._create_slider_layout(self.mixer_g_slider, self.mixer_g_label))
        self.mixer_b_slider = self._create_slider(-200, 200, 0, self._on_mixer_value_change); self.mixer_b_label = QLabel("0"); mixer_sliders_layout.addRow("Blue:", self._create_slider_layout(self.mixer_b_slider, self.mixer_b_label))
        self.mixer_const_slider = self._create_slider(-100, 100, 0, self._on_mixer_value_change); self.mixer_const_label = QLabel("0"); mixer_sliders_layout.addRow("Constant:", self._create_slider_layout(self.mixer_const_slider, self.mixer_const_label))
        mixer_layout.addLayout(mixer_sliders_layout)
        mixer_group_layout_outer = QVBoxLayout(mixer_group); mixer_group_layout_outer.setContentsMargins(0, 15, 0, 0); mixer_group_layout_outer.addWidget(mixer_content_widget)
        mixer_group.toggled.connect(mixer_content_widget.setVisible); mixer_content_widget.setVisible(False)
        main_layout.addWidget(mixer_group)
        self._update_mixer_sliders()

        # --- Noise Reduction Group ---
        nr_group = QGroupBox("Noise Reduction")
        self.group_boxes['nr'] = nr_group # Store reference
        self.original_group_titles['nr'] = nr_group.title() # Store original title
        nr_group.setCheckable(True); nr_group.setChecked(False)
        nr_content_widget = QWidget(); nr_layout = QFormLayout(nr_content_widget)
        nr_layout.setContentsMargins(10, 5, 10, 10); nr_layout.setSpacing(10)
        self.noise_reduction_strength_slider = self._create_slider(0, 100, 0, self._on_nr_strength_change); self.noise_reduction_strength_label = QLabel("0")
        nr_layout.addRow("Strength:", self._create_slider_layout(self.noise_reduction_strength_slider, self.noise_reduction_strength_label))
        nr_group_layout_outer = QVBoxLayout(nr_group); nr_group_layout_outer.setContentsMargins(0, 15, 0, 0); nr_group_layout_outer.addWidget(nr_content_widget)
        nr_group.toggled.connect(nr_content_widget.setVisible); nr_content_widget.setVisible(False)
        main_layout.addWidget(nr_group)

        # --- Dust Removal Group ---
        dust_group = QGroupBox("Dust Removal")
        self.group_boxes['dust_removal'] = dust_group # Store reference
        self.original_group_titles['dust_removal'] = dust_group.title() # Store original title
        dust_group.setCheckable(True); dust_group.setChecked(False) # Default disabled
        dust_content_widget = QWidget(); dust_layout = QFormLayout(dust_content_widget)
        dust_layout.setContentsMargins(10, 5, 10, 10); dust_layout.setSpacing(10)

        # Enable Checkbox (handled by group box checkable)
        dust_group.toggled.connect(self._on_dust_removal_enabled_change)

        # Sensitivity Slider
        self.dust_sensitivity_slider = self._create_slider(1, 100, self._default_adjustments['dust_removal_sensitivity'], self._on_dust_removal_sensitivity_change)
        self.dust_sensitivity_label = QLabel(str(self._default_adjustments['dust_removal_sensitivity']))
        dust_layout.addRow("Sensitivity:", self._create_slider_layout(self.dust_sensitivity_slider, self.dust_sensitivity_label))

        # Radius Slider
        self.dust_radius_slider = self._create_slider(1, 10, self._default_adjustments['dust_removal_radius'], self._on_dust_removal_radius_change) # Radius 1-10 pixels
        self.dust_radius_label = QLabel(str(self._default_adjustments['dust_removal_radius']))
        dust_layout.addRow("Radius (px):", self._create_slider_layout(self.dust_radius_slider, self.dust_radius_label))

        dust_group_layout_outer = QVBoxLayout(dust_group); dust_group_layout_outer.setContentsMargins(0, 15, 0, 0); dust_group_layout_outer.addWidget(dust_content_widget)
        dust_group.toggled.connect(dust_content_widget.setVisible); dust_content_widget.setVisible(False) # Content hidden when group unchecked
        main_layout.addWidget(dust_group)
        # --- End Dust Removal Group ---

        # --- HSL Adjustments Group ---
        hsl_group = QGroupBox("HSL / Color")
        self.group_boxes['hsl'] = hsl_group # Store reference
        self.original_group_titles['hsl'] = hsl_group.title() # Store original title
        hsl_group.setCheckable(True); hsl_group.setChecked(False)
        hsl_content_widget = QWidget(); hsl_layout = QVBoxLayout(hsl_content_widget)
        hsl_layout.setContentsMargins(10, 5, 10, 10); hsl_layout.setSpacing(10)
        hsl_color_layout = QHBoxLayout(); hsl_color_layout.addWidget(QLabel("Color Range:"))
        self.hsl_color_combo = QComboBox(); self.hsl_color_ranges = ['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas']; self.hsl_color_combo.addItems(self.hsl_color_ranges); self.hsl_color_combo.currentTextChanged.connect(self._on_hsl_color_change)
        hsl_color_layout.addWidget(self.hsl_color_combo); hsl_color_layout.addStretch(1); hsl_layout.addLayout(hsl_color_layout)
        hsl_sliders_layout = QFormLayout(); hsl_sliders_layout.setSpacing(10)
        self.hsl_h_slider = self._create_slider(-180, 180, 0, self._on_hsl_value_change); self.hsl_h_label = QLabel("0"); hsl_sliders_layout.addRow("Hue:", self._create_slider_layout(self.hsl_h_slider, self.hsl_h_label))
        self.hsl_s_slider = self._create_slider(-100, 100, 0, self._on_hsl_value_change); self.hsl_s_label = QLabel("0"); hsl_sliders_layout.addRow("Saturation:", self._create_slider_layout(self.hsl_s_slider, self.hsl_s_label))
        self.hsl_l_slider = self._create_slider(-100, 100, 0, self._on_hsl_value_change); self.hsl_l_label = QLabel("0"); hsl_sliders_layout.addRow("Lightness:", self._create_slider_layout(self.hsl_l_slider, self.hsl_l_label))
        hsl_layout.addLayout(hsl_sliders_layout)
        hsl_group_layout_outer = QVBoxLayout(hsl_group); hsl_group_layout_outer.setContentsMargins(0, 15, 0, 0); hsl_group_layout_outer.addWidget(hsl_content_widget)
        hsl_group.toggled.connect(hsl_content_widget.setVisible); hsl_content_widget.setVisible(False)
        main_layout.addWidget(hsl_group)
        self._update_hsl_sliders()

        # --- Selective Color Group ---
        sel_color_group = QGroupBox("Selective Color")
        self.group_boxes['selective_color'] = sel_color_group # Store reference
        self.original_group_titles['selective_color'] = sel_color_group.title() # Store original title
        sel_color_group.setCheckable(True); sel_color_group.setChecked(False)
        sel_color_content_widget = QWidget(); sel_color_layout = QVBoxLayout(sel_color_content_widget)
        sel_color_layout.setContentsMargins(10, 5, 10, 10); sel_color_layout.setSpacing(10)
        sel_color_range_layout = QHBoxLayout(); sel_color_range_layout.addWidget(QLabel("Colors:"))
        self.sel_color_combo = QComboBox(); self.sel_color_ranges = ['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas', 'Whites', 'Neutrals', 'Blacks']; self.sel_color_combo.addItems(self.sel_color_ranges); self.sel_color_combo.currentTextChanged.connect(self._on_selective_color_change)
        sel_color_range_layout.addWidget(self.sel_color_combo); sel_color_range_layout.addStretch(1); sel_color_layout.addLayout(sel_color_range_layout)
        sel_sliders_layout = QFormLayout(); sel_sliders_layout.setSpacing(10)
        self.sel_c_slider = self._create_slider(-100, 100, 0, self._on_selective_color_value_change); self.sel_c_label = QLabel("0"); sel_sliders_layout.addRow("Cyan:", self._create_slider_layout(self.sel_c_slider, self.sel_c_label))
        self.sel_m_slider = self._create_slider(-100, 100, 0, self._on_selective_color_value_change); self.sel_m_label = QLabel("0"); sel_sliders_layout.addRow("Magenta:", self._create_slider_layout(self.sel_m_slider, self.sel_m_label))
        self.sel_y_slider = self._create_slider(-100, 100, 0, self._on_selective_color_value_change); self.sel_y_label = QLabel("0"); sel_sliders_layout.addRow("Yellow:", self._create_slider_layout(self.sel_y_slider, self.sel_y_label))
        self.sel_k_slider = self._create_slider(-100, 100, 0, self._on_selective_color_value_change); self.sel_k_label = QLabel("0"); sel_sliders_layout.addRow("Black:", self._create_slider_layout(self.sel_k_slider, self.sel_k_label))
        sel_color_layout.addLayout(sel_sliders_layout)
        sel_color_group_layout_outer = QVBoxLayout(sel_color_group); sel_color_group_layout_outer.setContentsMargins(0, 15, 0, 0); sel_color_group_layout_outer.addWidget(sel_color_content_widget)
        sel_color_group.toggled.connect(sel_color_content_widget.setVisible); sel_color_content_widget.setVisible(False)
        main_layout.addWidget(sel_color_group)
        self._update_selective_color_sliders()

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_slider(self, min_val, max_val, default_val, callback):
        """Helper function to create a standard adjustment slider."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val); slider.setValue(default_val)
        slider.setSingleStep(1); slider.setPageStep(10)
        slider.setTickInterval((max_val - min_val) // 10 if (max_val - min_val) > 0 else 1)
        # Connect the callback AFTER setting the initial value
        slider.valueChanged.connect(callback)
        slider.mouseDoubleClickEvent = lambda event, s=slider, v=default_val: s.setValue(v)
        slider.setToolTip(f"Adjust value ({min_val} to {max_val}). Double-click to reset.")
        return slider

    def _create_slider_layout(self, slider, value_label):
        """Helper function to create a layout containing a slider and its value label."""
        layout = QHBoxLayout()
        layout.addWidget(slider)
        value_label.setMinimumWidth(35)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(value_label)
        return layout

    # --- History Management ---
    def _push_state_to_undo_stack(self):
        """Pushes the current adjustment state onto the undo stack."""
        # print("Pushing state to undo stack") # Debug
        current_state = copy.deepcopy(self._current_adjustments)
        self._undo_stack.append(current_state)
        # Limit history size
        if len(self._undo_stack) > self._history_limit:
            del self._undo_stack[0]
        # Clear redo stack whenever a new action is taken
        if self._redo_stack:
            self._redo_stack.clear()
        self._update_history_buttons_state()

    def _update_history_buttons_state(self):
        """Updates the enabled state of Undo/Redo buttons."""
        self.undo_button.setEnabled(bool(self._undo_stack))
        self.redo_button.setEnabled(bool(self._redo_stack))

    def undo_adjustment(self):
        """Reverts to the previous state in the undo stack."""
        if not self._undo_stack:
            return
        # print("Undoing adjustment") # Debug
        # Push current state to redo stack BEFORE changing it
        self._redo_stack.append(copy.deepcopy(self._current_adjustments))
        # Pop previous state from undo stack
        previous_state = self._undo_stack.pop()
        # Set panel to previous state without adding to history again
        self.set_adjustments(previous_state, update_history=False)
        # Update button states
        self._update_history_buttons_state()
        # Emit signal so main window knows adjustments changed
        self.adjustment_changed.emit(self._current_adjustments)


    def redo_adjustment(self):
        """Reapplies the next state from the redo stack."""
        if not self._redo_stack:
            return
        # print("Redoing adjustment") # Debug
        # Push current state back to undo stack BEFORE changing it
        self._undo_stack.append(copy.deepcopy(self._current_adjustments))
         # Limit history size again (in case redo pushes it over)
        if len(self._undo_stack) > self._history_limit:
            del self._undo_stack[0]
        # Pop next state from redo stack
        next_state = self._redo_stack.pop()
        # Set panel to next state without adding to history again
        self.set_adjustments(next_state, update_history=False)
        # Update button states
        self._update_history_buttons_state()
        # Emit signal so main window knows adjustments changed
        self.adjustment_changed.emit(self._current_adjustments)

    # --- Signal Emitters for Basic Adjustments ---
    def _on_brightness_change(self, value): self._push_state_to_undo_stack(); self.brightness_label.setText(str(value)); self._current_adjustments['brightness'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')
    def _on_contrast_change(self, value): self._push_state_to_undo_stack(); self.contrast_label.setText(str(value)); self._current_adjustments['contrast'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')
    def _on_saturation_change(self, value): self._push_state_to_undo_stack(); self.saturation_label.setText(str(value)); self._current_adjustments['saturation'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')
    def _on_hue_change(self, value): self._push_state_to_undo_stack(); self.hue_label.setText(str(value)); self._current_adjustments['hue'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')
    def _on_temp_change(self, value): self._push_state_to_undo_stack(); self.temp_label.setText(str(value)); self._current_adjustments['temp'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')
    def _on_tint_change(self, value): self._push_state_to_undo_stack(); self.tint_label.setText(str(value)); self._current_adjustments['tint'] = value; self.adjustment_changed.emit(self._current_adjustments); self._check_and_update_group_title('basic')

    # --- Signal Emitter for AWB ---
    def _on_awb_apply_clicked(self):
        """Emit signal when Apply Auto WB button is clicked."""
        selected_method_text = self.awb_method_combo.currentText()
        method_key = selected_method_text.lower().replace(' ', '_')
        print(f"AWB Apply button clicked. Method: {method_key}")
        self.awb_requested.emit(method_key)
        # Auto actions clear panel history via set_adjustments call in MainWindow

    # --- Signal Emitter for Auto Level ---
    def _on_auto_level_apply_clicked(self):
        """Emit signal when Apply Auto Level button is clicked."""
        selected_mode_display = self.al_colormode_combo.currentText()
        selected_mode_key = self.al_colormode_map.get(selected_mode_display, 'luminance')
        midrange_value = self.al_midrange_spin.value()
        print(f"Auto Level Apply button clicked. Mode: {selected_mode_key}, Midrange: {midrange_value}")
        self.auto_level_requested.emit(selected_mode_key, midrange_value)
        # Auto actions clear panel history via set_adjustments call in MainWindow

    # --- Signal Emitter for Auto Color ---
    def _on_auto_color_apply_clicked(self):
        """Emit signal when Apply Auto Color button is clicked."""
        selected_method_display = self.ac_method_combo.currentText()
        selected_method_key = self.ac_method_map.get(selected_method_display, 'gamma')
        print(f"Auto Color Apply button clicked. Method: {selected_method_key}")
        self.auto_color_requested.emit(selected_method_key)
        # Auto actions clear panel history via set_adjustments call in MainWindow

    # --- Signal Emitter for Auto Tone ---
    def _on_auto_tone_apply_clicked(self):
        """Emit signal when Apply Auto Tone button is clicked."""
        print(f"Auto Tone Apply button clicked.")
        self.auto_tone_requested.emit()
        # Auto actions clear panel history via set_adjustments call in MainWindow

    # --- Handlers for Complex Adjustments ---
    def _on_levels_change(self):
        self._push_state_to_undo_stack() # Save state before change
        if self.levels_in_black_spin.value() >= self.levels_in_white_spin.value(): self.levels_in_black_spin.setValue(self.levels_in_white_spin.value() - 1)
        if self.levels_out_black_spin.value() >= self.levels_out_white_spin.value(): self.levels_out_black_spin.setValue(self.levels_out_white_spin.value() - 1)
        self._current_adjustments['levels_in_black'] = self.levels_in_black_spin.value()
        self._current_adjustments['levels_in_white'] = self.levels_in_white_spin.value()
        self._current_adjustments['levels_gamma'] = self.levels_gamma_spin.value()
        self._current_adjustments['levels_out_black'] = self.levels_out_black_spin.value()
        self._current_adjustments['levels_out_white'] = self.levels_out_white_spin.value()
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('levels')

    def _on_curve_change(self, channel, points):
        # Curve widget emits signal AFTER change, so push state before update
        self._push_state_to_undo_stack()
        channel_key_map = {'RGB': 'curves_rgb', 'Red': 'curves_red', 'Green': 'curves_green', 'Blue': 'curves_blue'}
        key = channel_key_map.get(channel)
        if key:
            self._current_adjustments[key] = points
            self.adjustment_changed.emit(self._current_adjustments)
            self._check_and_update_group_title('curves')

    def _on_mixer_channel_change(self, channel_name):
        # Changing view doesn't change adjustment values, no history push needed
        self._current_adjustments['mixer_out_channel'] = channel_name
        self._update_mixer_sliders()

    def _update_mixer_sliders(self):
        channel = self._current_adjustments['mixer_out_channel'].lower()
        self.mixer_r_slider.blockSignals(True); self.mixer_g_slider.blockSignals(True); self.mixer_b_slider.blockSignals(True); self.mixer_const_slider.blockSignals(True)
        r_val = self._current_adjustments.get(f'mixer_{channel}_r', 100 if channel == 'red' else 0)
        g_val = self._current_adjustments.get(f'mixer_{channel}_g', 100 if channel == 'green' else 0)
        b_val = self._current_adjustments.get(f'mixer_{channel}_b', 100 if channel == 'blue' else 0)
        const_val = self._current_adjustments.get(f'mixer_{channel}_const', 0)
        self.mixer_r_slider.setValue(r_val); self.mixer_r_label.setText(str(r_val))
        self.mixer_g_slider.setValue(g_val); self.mixer_g_label.setText(str(g_val))
        self.mixer_b_slider.setValue(b_val); self.mixer_b_label.setText(str(b_val))
        self.mixer_const_slider.setValue(const_val); self.mixer_const_label.setText(str(const_val))
        self.mixer_r_slider.blockSignals(False); self.mixer_g_slider.blockSignals(False); self.mixer_b_slider.blockSignals(False); self.mixer_const_slider.blockSignals(False)

    def _on_mixer_value_change(self):
        self._push_state_to_undo_stack() # Save state before change
        channel = self._current_adjustments['mixer_out_channel'].lower()
        r_val = self.mixer_r_slider.value(); self.mixer_r_label.setText(str(r_val))
        g_val = self.mixer_g_slider.value(); self.mixer_g_label.setText(str(g_val))
        b_val = self.mixer_b_slider.value(); self.mixer_b_label.setText(str(b_val))
        const_val = self.mixer_const_slider.value(); self.mixer_const_label.setText(str(const_val))
        self._current_adjustments[f'mixer_{channel}_r'] = r_val
        self._current_adjustments[f'mixer_{channel}_g'] = g_val
        self._current_adjustments[f'mixer_{channel}_b'] = b_val
        self._current_adjustments[f'mixer_{channel}_const'] = const_val
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('mixer')

    def _on_nr_strength_change(self, value):
        self._push_state_to_undo_stack() # Save state before change
        self.noise_reduction_strength_label.setText(str(value))
        self._current_adjustments['noise_reduction_strength'] = value
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('nr')

    def _on_hsl_color_change(self, color_name):
        # Changing view doesn't change adjustment values, no history push needed
        self._current_adjustments['hsl_color'] = color_name
        self._update_hsl_sliders()

    def _update_hsl_sliders(self):
        color_key = self._current_adjustments['hsl_color'].lower()
        self.hsl_h_slider.blockSignals(True); self.hsl_s_slider.blockSignals(True); self.hsl_l_slider.blockSignals(True)
        h_val = self._current_adjustments.get(f'hsl_{color_key}_h', 0)
        s_val = self._current_adjustments.get(f'hsl_{color_key}_s', 0)
        l_val = self._current_adjustments.get(f'hsl_{color_key}_l', 0)
        self.hsl_h_slider.setValue(h_val); self.hsl_h_label.setText(str(h_val))
        self.hsl_s_slider.setValue(s_val); self.hsl_s_label.setText(str(s_val))
        self.hsl_l_slider.setValue(l_val); self.hsl_l_label.setText(str(l_val))
        self.hsl_h_slider.blockSignals(False); self.hsl_s_slider.blockSignals(False); self.hsl_l_slider.blockSignals(False)

    def _on_hsl_value_change(self):
        self._push_state_to_undo_stack() # Save state before change
        color_key = self._current_adjustments['hsl_color'].lower()
        h_val = self.hsl_h_slider.value(); self.hsl_h_label.setText(str(h_val))
        s_val = self.hsl_s_slider.value(); self.hsl_s_label.setText(str(s_val))
        l_val = self.hsl_l_slider.value(); self.hsl_l_label.setText(str(l_val))
        self._current_adjustments[f'hsl_{color_key}_h'] = h_val
        self._current_adjustments[f'hsl_{color_key}_s'] = s_val
        self._current_adjustments[f'hsl_{color_key}_l'] = l_val
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('hsl')

    def _on_selective_color_change(self, color_name):
        # Changing view doesn't change adjustment values, no history push needed
        self._current_adjustments['sel_color'] = color_name
        self._update_selective_color_sliders()

    def _update_selective_color_sliders(self):
        color_key = self._current_adjustments['sel_color'].lower()
        self.sel_c_slider.blockSignals(True); self.sel_m_slider.blockSignals(True); self.sel_y_slider.blockSignals(True); self.sel_k_slider.blockSignals(True)
        c_val = self._current_adjustments.get(f'sel_{color_key}_c', 0)
        m_val = self._current_adjustments.get(f'sel_{color_key}_m', 0)
        y_val = self._current_adjustments.get(f'sel_{color_key}_y', 0)
        k_val = self._current_adjustments.get(f'sel_{color_key}_k', 0)
        self.sel_c_slider.setValue(c_val); self.sel_c_label.setText(str(c_val))
        self.sel_m_slider.setValue(m_val); self.sel_m_label.setText(str(m_val))
        self.sel_y_slider.setValue(y_val); self.sel_y_label.setText(str(y_val))
        self.sel_k_slider.setValue(k_val); self.sel_k_label.setText(str(k_val))
        self.sel_c_slider.blockSignals(False); self.sel_m_slider.blockSignals(False); self.sel_y_slider.blockSignals(False); self.sel_k_slider.blockSignals(False)

    def _on_selective_color_value_change(self):
        self._push_state_to_undo_stack() # Save state before change
        color_key = self._current_adjustments['sel_color'].lower()
        c_val = self.sel_c_slider.value(); self.sel_c_label.setText(str(c_val))
        m_val = self.sel_m_slider.value(); self.sel_m_label.setText(str(m_val))
        y_val = self.sel_y_slider.value(); self.sel_y_label.setText(str(y_val))
        k_val = self.sel_k_slider.value(); self.sel_k_label.setText(str(k_val))
        self._current_adjustments[f'sel_{color_key}_c'] = c_val
        self._current_adjustments[f'sel_{color_key}_m'] = m_val
        self._current_adjustments[f'sel_{color_key}_y'] = y_val
        self._current_adjustments[f'sel_{color_key}_k'] = k_val
        # Handle boolean 'sel_relative' if we add a checkbox for it
        # self._current_adjustments['sel_relative'] = self.sel_relative_checkbox.isChecked()
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('selective_color')
    # --- Signal Handlers for Dust Removal ---
    def _on_dust_removal_enabled_change(self, checked):
        self._push_state_to_undo_stack()
        self._current_adjustments['dust_removal_enabled'] = checked
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('dust_removal')

    def _on_dust_removal_sensitivity_change(self, value):
        self._push_state_to_undo_stack()
        self.dust_sensitivity_label.setText(str(value))
        self._current_adjustments['dust_removal_sensitivity'] = value
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('dust_removal')

    def _on_dust_removal_radius_change(self, value):
        self._push_state_to_undo_stack()
        self.dust_radius_label.setText(str(value))
        self._current_adjustments['dust_removal_radius'] = value
        self.adjustment_changed.emit(self._current_adjustments)
        self._check_and_update_group_title('dust_removal')

    # --- Group Title Update Logic ---
    def _check_and_update_group_title(self, group_key):
        """Checks if any adjustment in the group differs from default and updates title."""
        group_box = self.group_boxes.get(group_key)
        original_title = self.original_group_titles.get(group_key)
        adjustment_keys = self.group_adjustment_keys.get(group_key)

        if not group_box or not original_title or not adjustment_keys:
            # print(f"Warning: Missing data for group key '{group_key}' in title update.") # Debug
            return

        is_modified = False
        for key in adjustment_keys:
            current_value = self._current_adjustments.get(key)
            default_value = self._default_adjustments.get(key)

            if current_value is None or default_value is None:
                # print(f"Warning: Mismatch or missing key '{key}' during comparison.") # Debug
                continue

            if current_value != default_value:
                is_modified = True
                break

        if is_modified:
            if not original_title.endswith(" *"):
                group_box.setTitle(original_title + " *")
        else:
            group_box.setTitle(original_title)

    def _update_all_group_titles(self):
        """Updates titles for all tracked adjustment groups."""
        for group_key in self.group_boxes.keys():
            self._check_and_update_group_title(group_key)

    # --- Public Methods ---
    def get_adjustments(self):
        """Return the current dictionary of adjustment values."""
        return self._current_adjustments.copy()

    def reset_adjustments(self):
        """Reset all adjustment controls and internal state to defaults."""
        print("Resetting adjustments...")
        # Reset internal state using a deep copy of defaults
        self._current_adjustments = copy.deepcopy(self._default_adjustments)

        # Clear history stacks
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_history_buttons_state() # Disable buttons

        # Block signals while resetting UI elements
        # Basic
        self.brightness_slider.blockSignals(True); self.brightness_slider.setValue(self._default_adjustments['brightness']); self.brightness_label.setText(str(self._default_adjustments['brightness'])); self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(True); self.contrast_slider.setValue(self._default_adjustments['contrast']); self.contrast_label.setText(str(self._default_adjustments['contrast'])); self.contrast_slider.blockSignals(False)
        self.saturation_slider.blockSignals(True); self.saturation_slider.setValue(self._default_adjustments['saturation']); self.saturation_label.setText(str(self._default_adjustments['saturation'])); self.saturation_slider.blockSignals(False)
        self.hue_slider.blockSignals(True); self.hue_slider.setValue(self._default_adjustments['hue']); self.hue_label.setText(str(self._default_adjustments['hue'])); self.hue_slider.blockSignals(False)
        self.temp_slider.blockSignals(True); self.temp_slider.setValue(self._default_adjustments['temp']); self.temp_label.setText(str(self._default_adjustments['temp'])); self.temp_slider.blockSignals(False)
        self.tint_slider.blockSignals(True); self.tint_slider.setValue(self._default_adjustments['tint']); self.tint_label.setText(str(self._default_adjustments['tint'])); self.tint_slider.blockSignals(False)
        # Levels
        self.levels_in_black_spin.blockSignals(True); self.levels_in_black_spin.setValue(self._default_adjustments['levels_in_black']); self.levels_in_black_spin.blockSignals(False)
        self.levels_in_white_spin.blockSignals(True); self.levels_in_white_spin.setValue(self._default_adjustments['levels_in_white']); self.levels_in_white_spin.blockSignals(False)
        self.levels_gamma_spin.blockSignals(True); self.levels_gamma_spin.setValue(self._default_adjustments['levels_gamma']); self.levels_gamma_spin.blockSignals(False)
        self.levels_out_black_spin.blockSignals(True); self.levels_out_black_spin.setValue(self._default_adjustments['levels_out_black']); self.levels_out_black_spin.blockSignals(False)
        self.levels_out_white_spin.blockSignals(True); self.levels_out_white_spin.setValue(self._default_adjustments['levels_out_white']); self.levels_out_white_spin.blockSignals(False)
        # Curves
        self.curves_widget.blockSignals(True)
        default_curve = [[0, 0], [255, 255]]
        for channel in ['RGB', 'Red', 'Green', 'Blue']:
            self.curves_widget.set_curve_points(channel, default_curve)
        self.curves_widget.blockSignals(False)
        # Mixer
        self.mixer_channel_combo.blockSignals(True); self.mixer_channel_combo.setCurrentText(self._default_adjustments['mixer_out_channel']); self.mixer_channel_combo.blockSignals(False)
        self._update_mixer_sliders() # Updates sliders based on the (now default) channel and default values
        # Noise Reduction
        self.noise_reduction_strength_slider.blockSignals(True); self.noise_reduction_strength_slider.setValue(self._default_adjustments['noise_reduction_strength']); self.noise_reduction_strength_label.setText(str(self._default_adjustments['noise_reduction_strength'])); self.noise_reduction_strength_slider.blockSignals(False)
        # HSL
        self.hsl_color_combo.blockSignals(True); self.hsl_color_combo.setCurrentText(self._default_adjustments['hsl_color']); self.hsl_color_combo.blockSignals(False)
        self._update_hsl_sliders() # Updates sliders based on the (now default) color and default values
        # Selective Color
        self.sel_color_combo.blockSignals(True); self.sel_color_combo.setCurrentText(self._default_adjustments['sel_color']); self.sel_color_combo.blockSignals(False)
        self._update_selective_color_sliders() # Updates sliders based on the (now default) color and default values
        # Add reset for sel_relative if checkbox exists

        # Update all group titles to remove asterisks
        self._update_all_group_titles()

        # Emit signal with default adjustments
        self.adjustment_changed.emit(self._current_adjustments)
        print("Adjustments reset.")

    def set_adjustments(self, adjustments_dict, update_history=True):
        """
        Set the adjustment controls based on an incoming dictionary.

        Args:
            adjustments_dict (dict): Dictionary of adjustment values.
            update_history (bool): If True (default), clears the panel's undo/redo history.
                                   Set to False when called from undo/redo actions.
        """
        # print(f"Setting adjustments. Update History: {update_history}") # Debug
        # Use deepcopy to avoid modifying the input dict and ensure internal state is separate
        new_adjustments = copy.deepcopy(adjustments_dict)

        # Validate and update internal state carefully
        for key in self._default_adjustments.keys():
            if key in new_adjustments:
                self._current_adjustments[key] = new_adjustments[key]

        # Clear history ONLY if triggered externally (preset, auto, load, etc.)
        if update_history:
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._update_history_buttons_state() # Disable buttons

        # Block signals while updating UI elements
        # Basic
        self.brightness_slider.blockSignals(True); self.brightness_slider.setValue(self._current_adjustments['brightness']); self.brightness_label.setText(str(self._current_adjustments['brightness'])); self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(True); self.contrast_slider.setValue(self._current_adjustments['contrast']); self.contrast_label.setText(str(self._current_adjustments['contrast'])); self.contrast_slider.blockSignals(False)
        self.saturation_slider.blockSignals(True); self.saturation_slider.setValue(self._current_adjustments['saturation']); self.saturation_label.setText(str(self._current_adjustments['saturation'])); self.saturation_slider.blockSignals(False)
        self.hue_slider.blockSignals(True); self.hue_slider.setValue(self._current_adjustments['hue']); self.hue_label.setText(str(self._current_adjustments['hue'])); self.hue_slider.blockSignals(False)
        self.temp_slider.blockSignals(True); self.temp_slider.setValue(self._current_adjustments['temp']); self.temp_label.setText(str(self._current_adjustments['temp'])); self.temp_slider.blockSignals(False)
        self.tint_slider.blockSignals(True); self.tint_slider.setValue(self._current_adjustments['tint']); self.tint_label.setText(str(self._current_adjustments['tint'])); self.tint_slider.blockSignals(False)
        # Levels
        self.levels_in_black_spin.blockSignals(True); self.levels_in_black_spin.setValue(self._current_adjustments['levels_in_black']); self.levels_in_black_spin.blockSignals(False)
        self.levels_in_white_spin.blockSignals(True); self.levels_in_white_spin.setValue(self._current_adjustments['levels_in_white']); self.levels_in_white_spin.blockSignals(False)
        self.levels_gamma_spin.blockSignals(True); self.levels_gamma_spin.setValue(self._current_adjustments['levels_gamma']); self.levels_gamma_spin.blockSignals(False)
        self.levels_out_black_spin.blockSignals(True); self.levels_out_black_spin.setValue(self._current_adjustments['levels_out_black']); self.levels_out_black_spin.blockSignals(False)
        self.levels_out_white_spin.blockSignals(True); self.levels_out_white_spin.setValue(self._current_adjustments['levels_out_white']); self.levels_out_white_spin.blockSignals(False)
        # Curves
        self.curves_widget.blockSignals(True)
        # Ensure curve data exists before trying to set it
        default_curve = [[0, 0], [255, 255]]
        self.curves_widget.set_curve_points('RGB', self._current_adjustments.get('curves_rgb', default_curve))
        self.curves_widget.set_curve_points('Red', self._current_adjustments.get('curves_red', default_curve))
        self.curves_widget.set_curve_points('Green', self._current_adjustments.get('curves_green', default_curve))
        self.curves_widget.set_curve_points('Blue', self._current_adjustments.get('curves_blue', default_curve))
        self.curves_widget.blockSignals(False)
        # Mixer
        self.mixer_channel_combo.blockSignals(True); self.mixer_channel_combo.setCurrentText(self._current_adjustments['mixer_out_channel']); self.mixer_channel_combo.blockSignals(False)
        self._update_mixer_sliders() # Updates sliders based on the new channel and values
        # Noise Reduction
        # Noise Reduction
        nr_strength = self._current_adjustments['noise_reduction_strength']
        self.noise_reduction_strength_slider.blockSignals(True); self.noise_reduction_strength_slider.setValue(nr_strength); self.noise_reduction_strength_label.setText(str(nr_strength)); self.noise_reduction_strength_slider.blockSignals(False)
        self.group_boxes['nr'].setChecked(nr_strength != self._default_adjustments['noise_reduction_strength'])

        # Dust Removal
        dust_enabled = self._current_adjustments.get('dust_removal_enabled', self._default_adjustments['dust_removal_enabled'])
        dust_sensitivity = self._current_adjustments.get('dust_removal_sensitivity', self._default_adjustments['dust_removal_sensitivity'])
        dust_radius = self._current_adjustments.get('dust_removal_radius', self._default_adjustments['dust_removal_radius'])
        self.group_boxes['dust_removal'].blockSignals(True); self.group_boxes['dust_removal'].setChecked(dust_enabled); self.group_boxes['dust_removal'].blockSignals(False) # Block signals for group checkbox
        self.dust_sensitivity_slider.blockSignals(True); self.dust_sensitivity_slider.setValue(dust_sensitivity); self.dust_sensitivity_label.setText(str(dust_sensitivity)); self.dust_sensitivity_slider.blockSignals(False)
        self.dust_radius_slider.blockSignals(True); self.dust_radius_slider.setValue(dust_radius); self.dust_radius_label.setText(str(dust_radius)); self.dust_radius_slider.blockSignals(False)
        # HSL
        self.hsl_color_combo.blockSignals(True); self.hsl_color_combo.setCurrentText(self._current_adjustments['hsl_color']); self.hsl_color_combo.blockSignals(False)
        self._update_hsl_sliders() # Updates sliders based on the new color and values
        # Selective Color
        self.sel_color_combo.blockSignals(True); self.sel_color_combo.setCurrentText(self._current_adjustments['sel_color']); self.sel_color_combo.blockSignals(False)
        self._update_selective_color_sliders() # Updates sliders based on the new color and values
        # Add update for sel_relative if checkbox exists

        # Update all group titles based on the new state
        self._update_all_group_titles()

        # Note: We don't emit adjustment_changed here, as this method is usually
        # called *in response* to an external change or undo/redo. The caller
        # (e.g., MainWindow, undo_adjustment, redo_adjustment) is responsible
        # for emitting the signal if needed.
        # print("Adjustments set.") # Debug


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    # Dummy MainWindow to connect signals for testing
    class DummyMainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Adjustment Panel Test")
            self.panel = AdjustmentPanel()
            self.setCentralWidget(self.panel)
            self.panel.adjustment_changed.connect(self.handle_adjustments)
            self.panel.awb_requested.connect(self.handle_awb)
            self.panel.auto_level_requested.connect(self.handle_auto_level)
            self.panel.auto_color_requested.connect(self.handle_auto_color)
            self.panel.auto_tone_requested.connect(self.handle_auto_tone)
            self.panel.wb_picker_requested.connect(lambda: print("WB Picker Requested")) # Simple print for picker

        def handle_adjustments(self, adjustments):
            # print("Adjustments Changed:", adjustments)
            pass # Avoid excessive printing during slider drag

        def handle_awb(self, method):
            print(f"AWB Requested: Method={method}")
            # Simulate applying AWB by slightly changing temp/tint
            current_temp = self.panel._current_adjustments['temp']
            current_tint = self.panel._current_adjustments['tint']
            new_temp = current_temp - 5 if method == 'gray_world' else current_temp + 3
            new_tint = current_tint + 2 if method == 'gray_world' else current_tint - 4
            # Call set_adjustments to update UI AND clear panel history
            self.panel.set_adjustments({'temp': new_temp, 'tint': new_tint})
            # Manually emit signal if main window needs to reprocess image
            self.panel.adjustment_changed.emit(self.panel.get_adjustments())


        def handle_auto_level(self, mode, midrange):
            print(f"Auto Level Requested: Mode={mode}, Midrange={midrange}")
            # Simulate applying Auto Level by changing level sliders
            self.panel.set_adjustments({'levels_in_black': 5, 'levels_in_white': 250, 'levels_gamma': 1.05})
            self.panel.adjustment_changed.emit(self.panel.get_adjustments())

        def handle_auto_color(self, method):
            print(f"Auto Color Requested: Method={method}")
            # Simulate applying Auto Color by changing contrast/saturation
            self.panel.set_adjustments({'contrast': 5, 'saturation': 3})
            self.panel.adjustment_changed.emit(self.panel.get_adjustments())

        def handle_auto_tone(self):
            print(f"Auto Tone Requested")
            # Simulate applying Auto Tone by changing multiple sliders
            self.panel.set_adjustments({'brightness': 2, 'contrast': 3, 'levels_in_black': 2, 'levels_in_white': 253})
            self.panel.adjustment_changed.emit(self.panel.get_adjustments())


    app = QApplication(sys.argv)
    mainWin = DummyMainWindow()
    mainWin.show()
    sys.exit(app.exec())