# Adjustment controls - Redesigned with tabbed interface
"""
Redesigned adjustment panel with:
- Tabbed interface (Basic | Advanced | Color)
- Visual indicators for active adjustments
- Compact mode toggle
- Better tooltips
"""
import copy
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox,
    QComboBox, QDoubleSpinBox, QSizePolicy, QGroupBox, QFormLayout,
    QPushButton, QMessageBox, QInputDialog, QTabWidget, QScrollArea,
    QCheckBox, QFrame, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette

from ..utils.logger import get_logger
from ..processing.adjustment_presets import AdjustmentPresetManager
from .curves_widget import CurvesWidget

logger = get_logger(__name__)

# Tooltip descriptions for each adjustment
TOOLTIPS = {
    'brightness': "Adjust overall image brightness. Positive values lighten, negative darken.",
    'contrast': "Adjust tonal contrast. Positive increases contrast, negative decreases.",
    'saturation': "Adjust color intensity. Positive makes colors more vivid, negative more muted.",
    'hue': "Shift all colors around the color wheel. Use for creative color effects.",
    'temp': "Adjust color temperature. Positive warms (yellow/orange), negative cools (blue).",
    'tint': "Adjust green-magenta balance. Positive adds magenta, negative adds green.",
    'levels_in_black': "Set the black point. Pixels at or below this become pure black.",
    'levels_in_white': "Set the white point. Pixels at or above this become pure white.",
    'levels_gamma': "Adjust midtone brightness. <1.0 darkens midtones, >1.0 lightens.",
    'levels_out_black': "Limit the darkest output value. Raises the black level.",
    'levels_out_white': "Limit the brightest output value. Lowers the white level.",
    'curves': "Fine-tune tonal response. Click to add points, drag to adjust.",
    'mixer': "Blend color channels. Control how much of each channel contributes to output.",
    'hsl': "Adjust Hue, Saturation, and Lightness for specific color ranges.",
    'selective_color': "Adjust CMYK values within specific color ranges.",
    'noise_reduction': "Reduce image noise. Higher values = more smoothing (may lose detail).",
    'dust_removal': "Automatically detect and remove dust spots and scratches.",
}


class IndicatorLabel(QLabel):
    """A small colored indicator showing if adjustments are active."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._active = False
        self._update_style()
    
    def set_active(self, active: bool):
        self._active = active
        self._update_style()
    
    def _update_style(self):
        if self._active:
            self.setStyleSheet("""
                background-color: #4CAF50;
                border-radius: 4px;
                border: 1px solid #388E3C;
            """)
            self.setToolTip("Adjustments active in this section")
        else:
            self.setStyleSheet("""
                background-color: #9E9E9E;
                border-radius: 4px;
                border: 1px solid #757575;
            """)
            self.setToolTip("No adjustments active")


class CollapsibleSection(QWidget):
    """A collapsible section with header and content."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._expanded = True
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self._header = QFrame()
        self._header.setFrameShape(QFrame.Shape.StyledPanel)
        self._header.setStyleSheet("""
            QFrame {
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }
            QFrame:hover {
                background-color: palette(light);
            }
        """)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.mousePressEvent = self._toggle
        
        header_layout = QHBoxLayout(self._header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        
        self._arrow = QLabel("▼")
        self._arrow.setFixedWidth(16)
        header_layout.addWidget(self._arrow)
        
        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._title_label)
        
        header_layout.addStretch()
        
        self._indicator = IndicatorLabel()
        header_layout.addWidget(self._indicator)
        
        self._reset_btn = QToolButton()
        self._reset_btn.setText("↺")
        self._reset_btn.setToolTip("Reset this section to defaults")
        self._reset_btn.setFixedSize(20, 20)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._reset_btn.hide()  # Show only when active
        header_layout.addWidget(self._reset_btn)
        
        layout.addWidget(self._header)
        
        # Content container
        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(8)
        layout.addWidget(self._content)
        
        # Reset callback
        self._reset_callback = None
    
    def set_reset_callback(self, callback):
        self._reset_callback = callback
    
    def _on_reset_clicked(self):
        if self._reset_callback:
            self._reset_callback()
    
    def _toggle(self, event=None):
        self._expanded = not self._expanded
        self._content.setVisible(self._expanded)
        self._arrow.setText("▼" if self._expanded else "▶")
    
    def set_expanded(self, expanded: bool):
        self._expanded = expanded
        self._content.setVisible(expanded)
        self._arrow.setText("▼" if expanded else "▶")
    
    def content_layout(self):
        return self._content_layout
    
    def set_active(self, active: bool):
        self._indicator.set_active(active)
        self._reset_btn.setVisible(active)
        if active:
            self._title_label.setStyleSheet("font-weight: bold; color: #1976D2;")
        else:
            self._title_label.setStyleSheet("font-weight: bold;")


class AdjustmentPanel(QWidget):
    """Panel for image adjustments with tabbed interface."""

    adjustment_changed = pyqtSignal(dict)
    awb_requested = pyqtSignal(str)
    auto_level_requested = pyqtSignal(str, float)
    auto_color_requested = pyqtSignal(str)
    auto_tone_requested = pyqtSignal()
    wb_picker_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Default adjustments
        self._default_adjustments = {
            'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0, 'temp': 0, 'tint': 0,
            'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0, 
            'levels_out_black': 0, 'levels_out_white': 255,
            'curves_rgb': [[0, 0], [255, 255]], 'curves_red': [[0, 0], [255, 255]], 
            'curves_green': [[0, 0], [255, 255]], 'curves_blue': [[0, 0], [255, 255]],
            'mixer_out_channel': 'Red',
            'mixer_red_r': 100, 'mixer_red_g': 0, 'mixer_red_b': 0, 'mixer_red_const': 0,
            'mixer_green_r': 0, 'mixer_green_g': 100, 'mixer_green_b': 0, 'mixer_green_const': 0,
            'mixer_blue_r': 0, 'mixer_blue_g': 0, 'mixer_blue_b': 100, 'mixer_blue_const': 0,
            'noise_reduction_strength': 0,
            'dust_removal_enabled': False, 'dust_removal_sensitivity': 50, 'dust_removal_radius': 3,
            'hsl_color': 'Reds',
            'hsl_reds_h': 0, 'hsl_reds_s': 0, 'hsl_reds_l': 0,
            'hsl_yellows_h': 0, 'hsl_yellows_s': 0, 'hsl_yellows_l': 0,
            'hsl_greens_h': 0, 'hsl_greens_s': 0, 'hsl_greens_l': 0,
            'hsl_cyans_h': 0, 'hsl_cyans_s': 0, 'hsl_cyans_l': 0,
            'hsl_blues_h': 0, 'hsl_blues_s': 0, 'hsl_blues_l': 0,
            'hsl_magentas_h': 0, 'hsl_magentas_s': 0, 'hsl_magentas_l': 0,
            'sel_color': 'Reds', 'sel_relative': True,
            'sel_reds_c': 0, 'sel_reds_m': 0, 'sel_reds_y': 0, 'sel_reds_k': 0,
            'sel_yellows_c': 0, 'sel_yellows_m': 0, 'sel_yellows_y': 0, 'sel_yellows_k': 0,
            'sel_greens_c': 0, 'sel_greens_m': 0, 'sel_greens_y': 0, 'sel_greens_k': 0,
            'sel_cyans_c': 0, 'sel_cyans_m': 0, 'sel_cyans_y': 0, 'sel_cyans_k': 0,
            'sel_blues_c': 0, 'sel_blues_m': 0, 'sel_blues_y': 0, 'sel_blues_k': 0,
            'sel_magentas_c': 0, 'sel_magentas_m': 0, 'sel_magentas_y': 0, 'sel_magentas_k': 0,
            'sel_whites_c': 0, 'sel_whites_m': 0, 'sel_whites_y': 0, 'sel_whites_k': 0,
            'sel_neutrals_c': 0, 'sel_neutrals_m': 0, 'sel_neutrals_y': 0, 'sel_neutrals_k': 0,
            'sel_blacks_c': 0, 'sel_blacks_m': 0, 'sel_blacks_y': 0, 'sel_blacks_k': 0,
        }
        self._current_adjustments = copy.deepcopy(self._default_adjustments)
        
        # History
        self._undo_stack = []
        self._redo_stack = []
        self._history_limit = 50
        
        # Compact mode
        self._compact_mode = False
        
        # Section tracking for indicators
        self._sections = {}
        
        # Preset manager
        self._preset_manager = AdjustmentPresetManager()
        self._preset_id_by_name = {}
        
        self._setup_ui()
        self._update_all_indicators()
        self._refresh_preset_list()

    def _setup_ui(self):
        """Set up the tabbed UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(4)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 4)
        
        self.undo_btn = QPushButton("↶")
        self.undo_btn.setToolTip("Undo (Ctrl+Z)")
        self.undo_btn.setFixedWidth(28)
        self.undo_btn.clicked.connect(self.undo_adjustment)
        self.undo_btn.setEnabled(False)
        toolbar.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("↷")
        self.redo_btn.setToolTip("Redo (Ctrl+Y)")
        self.redo_btn.setFixedWidth(28)
        self.redo_btn.clicked.connect(self.redo_adjustment)
        self.redo_btn.setEnabled(False)
        toolbar.addWidget(self.redo_btn)
        
        toolbar.addStretch()
        
        self.compact_btn = QCheckBox("Compact")
        self.compact_btn.setToolTip("Show only essential controls")
        self.compact_btn.toggled.connect(self._toggle_compact_mode)
        toolbar.addWidget(self.compact_btn)
        
        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.setToolTip("Reset all adjustments to defaults")
        self.reset_btn.clicked.connect(self.reset_adjustments)
        toolbar.addWidget(self.reset_btn)
        
        main_layout.addLayout(toolbar)
        
        # Preset bar
        preset_bar = QHBoxLayout()
        preset_bar.setContentsMargins(4, 0, 4, 4)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Select an adjustment preset")
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        preset_bar.addWidget(self.preset_combo)
        
        self.apply_preset_btn = QPushButton("Apply")
        self.apply_preset_btn.setToolTip("Apply selected preset")
        self.apply_preset_btn.clicked.connect(self._apply_preset)
        preset_bar.addWidget(self.apply_preset_btn)
        
        self.save_preset_btn = QPushButton("Save")
        self.save_preset_btn.setToolTip("Save current settings as preset")
        self.save_preset_btn.clicked.connect(self._save_preset)
        preset_bar.addWidget(self.save_preset_btn)
        
        main_layout.addLayout(preset_bar)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_basic_tab()
        self._create_advanced_tab()
        self._create_color_tab()

    def _create_basic_tab(self):
        """Create the Basic adjustments tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Light section (Brightness, Contrast)
        light_section = CollapsibleSection("Light")
        light_section.set_reset_callback(lambda: self._reset_section(['brightness', 'contrast']))
        self._sections['light'] = light_section
        
        form = QFormLayout()
        form.setSpacing(6)
        
        self.brightness_slider, self.brightness_label = self._create_slider_row(
            -100, 100, 0, "brightness", TOOLTIPS['brightness'])
        form.addRow("Brightness:", self._slider_with_label(self.brightness_slider, self.brightness_label))
        
        self.contrast_slider, self.contrast_label = self._create_slider_row(
            -100, 100, 0, "contrast", TOOLTIPS['contrast'])
        form.addRow("Contrast:", self._slider_with_label(self.contrast_slider, self.contrast_label))
        
        light_section.content_layout().addLayout(form)
        layout.addWidget(light_section)
        
        # Color section (Saturation, Hue)
        color_section = CollapsibleSection("Color")
        color_section.set_reset_callback(lambda: self._reset_section(['saturation', 'hue']))
        self._sections['color_basic'] = color_section
        
        form = QFormLayout()
        form.setSpacing(6)
        
        self.saturation_slider, self.saturation_label = self._create_slider_row(
            -100, 100, 0, "saturation", TOOLTIPS['saturation'])
        form.addRow("Saturation:", self._slider_with_label(self.saturation_slider, self.saturation_label))
        
        self.hue_slider, self.hue_label = self._create_slider_row(
            -180, 180, 0, "hue", TOOLTIPS['hue'])
        form.addRow("Hue:", self._slider_with_label(self.hue_slider, self.hue_label))
        
        color_section.content_layout().addLayout(form)
        layout.addWidget(color_section)
        
        # White Balance section
        wb_section = CollapsibleSection("White Balance")
        wb_section.set_reset_callback(lambda: self._reset_section(['temp', 'tint']))
        self._sections['wb'] = wb_section
        
        form = QFormLayout()
        form.setSpacing(6)
        
        self.temp_slider, self.temp_label = self._create_slider_row(
            -100, 100, 0, "temp", TOOLTIPS['temp'])
        form.addRow("Temperature:", self._slider_with_label(self.temp_slider, self.temp_label))
        
        self.tint_slider, self.tint_label = self._create_slider_row(
            -100, 100, 0, "tint", TOOLTIPS['tint'])
        
        tint_row = self._slider_with_label(self.tint_slider, self.tint_label)
        self.wb_picker_btn = QPushButton("Pick")
        self.wb_picker_btn.setToolTip("Pick a neutral color from the image to set white balance")
        self.wb_picker_btn.setFixedWidth(40)
        self.wb_picker_btn.clicked.connect(self.wb_picker_requested.emit)
        tint_row.addWidget(self.wb_picker_btn)
        form.addRow("Tint:", tint_row)
        
        wb_section.content_layout().addLayout(form)
        
        # Auto buttons
        auto_layout = QHBoxLayout()
        
        self.awb_combo = QComboBox()
        self.awb_combo.addItems(['Gray World', 'White Patch', 'Simple WB', 'Learning WB'])
        self.awb_combo.setToolTip("Select auto white balance algorithm")
        auto_layout.addWidget(self.awb_combo)
        
        self.awb_btn = QPushButton("Auto WB")
        self.awb_btn.setToolTip("Apply automatic white balance")
        self.awb_btn.clicked.connect(self._on_awb_clicked)
        auto_layout.addWidget(self.awb_btn)
        
        wb_section.content_layout().addLayout(auto_layout)
        layout.addWidget(wb_section)
        
        # Auto Color section
        auto_color_layout = QHBoxLayout()
        
        self.ac_combo = QComboBox()
        self.ac_combo.addItems(['Gamma', 'Recolor', 'None'])
        self.ac_combo.setToolTip("Select auto color correction method")
        auto_color_layout.addWidget(self.ac_combo)
        
        self.ac_btn = QPushButton("Auto Color")
        self.ac_btn.setToolTip("Apply automatic color correction")
        self.ac_btn.clicked.connect(self._on_auto_color_clicked)
        auto_color_layout.addWidget(self.ac_btn)
        
        auto_color_layout.addStretch()
        layout.addLayout(auto_color_layout)
        
        # Auto Tone button
        auto_tone_layout = QHBoxLayout()
        self.auto_tone_btn = QPushButton("🪄 Auto Tone")
        self.auto_tone_btn.setToolTip("Automatically adjust brightness, contrast, and color")
        self.auto_tone_btn.clicked.connect(self.auto_tone_requested.emit)
        auto_tone_layout.addWidget(self.auto_tone_btn)
        auto_tone_layout.addStretch()
        layout.addLayout(auto_tone_layout)
        
        layout.addStretch()
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Basic")


    def _create_advanced_tab(self):
        """Create the Advanced adjustments tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # Levels section
        levels_section = CollapsibleSection("Levels")
        levels_section.set_reset_callback(lambda: self._reset_section([
            'levels_in_black', 'levels_in_white', 'levels_gamma',
            'levels_out_black', 'levels_out_white']))
        self._sections['levels'] = levels_section
        
        form = QFormLayout()
        form.setSpacing(6)
        
        # Input levels
        input_layout = QHBoxLayout()
        self.levels_in_black = QSpinBox()
        self.levels_in_black.setRange(0, 254)
        self.levels_in_black.setValue(0)
        self.levels_in_black.setToolTip(TOOLTIPS['levels_in_black'])
        self.levels_in_black.valueChanged.connect(self._on_levels_changed)
        input_layout.addWidget(QLabel("Black:"))
        input_layout.addWidget(self.levels_in_black)
        
        self.levels_gamma = QDoubleSpinBox()
        self.levels_gamma.setRange(0.1, 10.0)
        self.levels_gamma.setSingleStep(0.1)
        self.levels_gamma.setValue(1.0)
        self.levels_gamma.setToolTip(TOOLTIPS['levels_gamma'])
        self.levels_gamma.valueChanged.connect(self._on_levels_changed)
        input_layout.addWidget(QLabel("γ:"))
        input_layout.addWidget(self.levels_gamma)
        
        self.levels_in_white = QSpinBox()
        self.levels_in_white.setRange(1, 255)
        self.levels_in_white.setValue(255)
        self.levels_in_white.setToolTip(TOOLTIPS['levels_in_white'])
        self.levels_in_white.valueChanged.connect(self._on_levels_changed)
        input_layout.addWidget(QLabel("White:"))
        input_layout.addWidget(self.levels_in_white)
        form.addRow("Input:", input_layout)
        
        # Output levels
        output_layout = QHBoxLayout()
        self.levels_out_black = QSpinBox()
        self.levels_out_black.setRange(0, 254)
        self.levels_out_black.setValue(0)
        self.levels_out_black.setToolTip(TOOLTIPS['levels_out_black'])
        self.levels_out_black.valueChanged.connect(self._on_levels_changed)
        output_layout.addWidget(QLabel("Black:"))
        output_layout.addWidget(self.levels_out_black)
        output_layout.addStretch()
        
        self.levels_out_white = QSpinBox()
        self.levels_out_white.setRange(1, 255)
        self.levels_out_white.setValue(255)
        self.levels_out_white.setToolTip(TOOLTIPS['levels_out_white'])
        self.levels_out_white.valueChanged.connect(self._on_levels_changed)
        output_layout.addWidget(QLabel("White:"))
        output_layout.addWidget(self.levels_out_white)
        form.addRow("Output:", output_layout)
        
        # Auto level
        auto_layout = QHBoxLayout()
        self.auto_level_combo = QComboBox()
        self.auto_level_combo.addItems(['Luminance', 'Lightness', 'Brightness', 'Gray', 'Average', 'RGB'])
        self.auto_level_combo.setToolTip("Select colorspace for auto levels calculation")
        auto_layout.addWidget(self.auto_level_combo)
        
        self.auto_level_mid = QDoubleSpinBox()
        self.auto_level_mid.setRange(0.01, 0.99)
        self.auto_level_mid.setValue(0.50)
        self.auto_level_mid.setSingleStep(0.05)
        self.auto_level_mid.setToolTip("Target midpoint for auto levels")
        auto_layout.addWidget(self.auto_level_mid)
        
        self.auto_level_btn = QPushButton("Auto")
        self.auto_level_btn.setToolTip("Automatically set levels")
        self.auto_level_btn.clicked.connect(self._on_auto_level_clicked)
        auto_layout.addWidget(self.auto_level_btn)
        form.addRow("", auto_layout)
        
        levels_section.content_layout().addLayout(form)
        levels_section.set_expanded(False)
        layout.addWidget(levels_section)
        
        # Curves section
        curves_section = CollapsibleSection("Curves")
        curves_section.set_reset_callback(lambda: self._reset_section([
            'curves_rgb', 'curves_red', 'curves_green', 'curves_blue']))
        self._sections['curves'] = curves_section
        
        self.curves_widget = CurvesWidget()
        self.curves_widget.curve_changed.connect(self._on_curve_changed)
        self.curves_widget.setMinimumHeight(180)
        curves_section.content_layout().addWidget(self.curves_widget)
        curves_section.set_expanded(False)
        layout.addWidget(curves_section)
        
        # Channel Mixer section
        mixer_section = CollapsibleSection("Channel Mixer")
        mixer_section.set_reset_callback(lambda: self._reset_section([
            'mixer_red_r', 'mixer_red_g', 'mixer_red_b', 'mixer_red_const',
            'mixer_green_r', 'mixer_green_g', 'mixer_green_b', 'mixer_green_const',
            'mixer_blue_r', 'mixer_blue_g', 'mixer_blue_b', 'mixer_blue_const']))
        self._sections['mixer'] = mixer_section
        
        mixer_form = QFormLayout()
        mixer_form.setSpacing(6)
        
        self.mixer_channel = QComboBox()
        self.mixer_channel.addItems(['Red', 'Green', 'Blue'])
        self.mixer_channel.setToolTip("Select output channel to adjust")
        self.mixer_channel.currentTextChanged.connect(self._update_mixer_display)
        mixer_form.addRow("Output:", self.mixer_channel)
        
        self.mixer_r_slider, self.mixer_r_label = self._create_slider_row(
            -200, 200, 100, None, "Red channel contribution")
        self.mixer_r_slider.valueChanged.connect(self._on_mixer_changed)
        mixer_form.addRow("Red:", self._slider_with_label(self.mixer_r_slider, self.mixer_r_label))
        
        self.mixer_g_slider, self.mixer_g_label = self._create_slider_row(
            -200, 200, 0, None, "Green channel contribution")
        self.mixer_g_slider.valueChanged.connect(self._on_mixer_changed)
        mixer_form.addRow("Green:", self._slider_with_label(self.mixer_g_slider, self.mixer_g_label))
        
        self.mixer_b_slider, self.mixer_b_label = self._create_slider_row(
            -200, 200, 0, None, "Blue channel contribution")
        self.mixer_b_slider.valueChanged.connect(self._on_mixer_changed)
        mixer_form.addRow("Blue:", self._slider_with_label(self.mixer_b_slider, self.mixer_b_label))
        
        self.mixer_const_slider, self.mixer_const_label = self._create_slider_row(
            -100, 100, 0, None, "Constant offset")
        self.mixer_const_slider.valueChanged.connect(self._on_mixer_changed)
        mixer_form.addRow("Constant:", self._slider_with_label(self.mixer_const_slider, self.mixer_const_label))
        
        mixer_section.content_layout().addLayout(mixer_form)
        mixer_section.set_expanded(False)
        layout.addWidget(mixer_section)
        
        # Noise Reduction section
        nr_section = CollapsibleSection("Noise Reduction")
        nr_section.set_reset_callback(lambda: self._reset_section(['noise_reduction_strength']))
        self._sections['nr'] = nr_section
        
        nr_form = QFormLayout()
        self.nr_slider, self.nr_label = self._create_slider_row(
            0, 100, 0, "noise_reduction_strength", TOOLTIPS['noise_reduction'])
        nr_form.addRow("Strength:", self._slider_with_label(self.nr_slider, self.nr_label))
        
        nr_section.content_layout().addLayout(nr_form)
        nr_section.set_expanded(False)
        layout.addWidget(nr_section)
        
        # Dust Removal section
        dust_section = CollapsibleSection("Dust Removal")
        dust_section.set_reset_callback(lambda: self._reset_section([
            'dust_removal_enabled', 'dust_removal_sensitivity', 'dust_removal_radius']))
        self._sections['dust'] = dust_section
        
        dust_form = QFormLayout()
        
        self.dust_enabled = QCheckBox("Enable")
        self.dust_enabled.setToolTip(TOOLTIPS['dust_removal'])
        self.dust_enabled.toggled.connect(self._on_dust_changed)
        dust_form.addRow("", self.dust_enabled)
        
        self.dust_sensitivity_slider, self.dust_sensitivity_label = self._create_slider_row(
            1, 100, 50, None, "Detection sensitivity (higher = more sensitive)")
        self.dust_sensitivity_slider.valueChanged.connect(self._on_dust_changed)
        dust_form.addRow("Sensitivity:", self._slider_with_label(self.dust_sensitivity_slider, self.dust_sensitivity_label))
        
        self.dust_radius_slider, self.dust_radius_label = self._create_slider_row(
            1, 10, 3, None, "Inpainting radius in pixels")
        self.dust_radius_slider.valueChanged.connect(self._on_dust_changed)
        dust_form.addRow("Radius:", self._slider_with_label(self.dust_radius_slider, self.dust_radius_label))
        
        dust_section.content_layout().addLayout(dust_form)
        dust_section.set_expanded(False)
        layout.addWidget(dust_section)
        
        layout.addStretch()
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Advanced")

    def _create_color_tab(self):
        """Create the Color adjustments tab."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        
        # HSL section
        hsl_section = CollapsibleSection("HSL / Color")
        hsl_section.set_reset_callback(lambda: self._reset_hsl())
        self._sections['hsl'] = hsl_section
        
        hsl_form = QFormLayout()
        hsl_form.setSpacing(6)
        
        self.hsl_color = QComboBox()
        self.hsl_color.addItems(['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas'])
        self.hsl_color.setToolTip("Select color range to adjust")
        self.hsl_color.currentTextChanged.connect(self._update_hsl_display)
        hsl_form.addRow("Color:", self.hsl_color)
        
        self.hsl_h_slider, self.hsl_h_label = self._create_slider_row(
            -180, 180, 0, None, "Shift hue for this color range")
        self.hsl_h_slider.valueChanged.connect(self._on_hsl_changed)
        hsl_form.addRow("Hue:", self._slider_with_label(self.hsl_h_slider, self.hsl_h_label))
        
        self.hsl_s_slider, self.hsl_s_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust saturation for this color range")
        self.hsl_s_slider.valueChanged.connect(self._on_hsl_changed)
        hsl_form.addRow("Saturation:", self._slider_with_label(self.hsl_s_slider, self.hsl_s_label))
        
        self.hsl_l_slider, self.hsl_l_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust lightness for this color range")
        self.hsl_l_slider.valueChanged.connect(self._on_hsl_changed)
        hsl_form.addRow("Lightness:", self._slider_with_label(self.hsl_l_slider, self.hsl_l_label))
        
        hsl_section.content_layout().addLayout(hsl_form)
        layout.addWidget(hsl_section)
        
        # Selective Color section
        sel_section = CollapsibleSection("Selective Color")
        sel_section.set_reset_callback(lambda: self._reset_selective_color())
        self._sections['selective'] = sel_section
        
        sel_form = QFormLayout()
        sel_form.setSpacing(6)
        
        self.sel_color = QComboBox()
        self.sel_color.addItems(['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas', 
                                 'Whites', 'Neutrals', 'Blacks'])
        self.sel_color.setToolTip("Select color range to adjust")
        self.sel_color.currentTextChanged.connect(self._update_sel_display)
        sel_form.addRow("Color:", self.sel_color)
        
        self.sel_c_slider, self.sel_c_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust cyan component")
        self.sel_c_slider.valueChanged.connect(self._on_sel_changed)
        sel_form.addRow("Cyan:", self._slider_with_label(self.sel_c_slider, self.sel_c_label))
        
        self.sel_m_slider, self.sel_m_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust magenta component")
        self.sel_m_slider.valueChanged.connect(self._on_sel_changed)
        sel_form.addRow("Magenta:", self._slider_with_label(self.sel_m_slider, self.sel_m_label))
        
        self.sel_y_slider, self.sel_y_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust yellow component")
        self.sel_y_slider.valueChanged.connect(self._on_sel_changed)
        sel_form.addRow("Yellow:", self._slider_with_label(self.sel_y_slider, self.sel_y_label))
        
        self.sel_k_slider, self.sel_k_label = self._create_slider_row(
            -100, 100, 0, None, "Adjust black component")
        self.sel_k_slider.valueChanged.connect(self._on_sel_changed)
        sel_form.addRow("Black:", self._slider_with_label(self.sel_k_slider, self.sel_k_label))
        
        self.sel_relative = QCheckBox("Relative")
        self.sel_relative.setChecked(True)
        self.sel_relative.setToolTip("Use relative adjustments (recommended)")
        self.sel_relative.toggled.connect(self._on_sel_changed)
        sel_form.addRow("", self.sel_relative)
        
        sel_section.content_layout().addLayout(sel_form)
        sel_section.set_expanded(False)
        layout.addWidget(sel_section)
        
        layout.addStretch()
        scroll.setWidget(content)
        self.tabs.addTab(scroll, "Color")


    # --- Helper Methods ---
    
    def _create_slider_row(self, min_val, max_val, default, key, tooltip):
        """Create a slider with value label."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setToolTip(tooltip)
        
        # Double-click to reset
        def reset_on_double_click(event, s=slider, v=default):
            s.setValue(v)
        slider.mouseDoubleClickEvent = reset_on_double_click
        
        label = QLabel(str(default))
        label.setFixedWidth(35)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        # Connect to update label and emit change
        if key:
            slider.valueChanged.connect(lambda v, l=label, k=key: self._on_slider_changed(v, l, k))
        else:
            slider.valueChanged.connect(lambda v, l=label: l.setText(str(v)))
        
        return slider, label
    
    def _slider_with_label(self, slider, label):
        """Create a horizontal layout with slider and label."""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(slider)
        layout.addWidget(label)
        return layout
    
    def _on_slider_changed(self, value, label, key):
        """Handle slider value change."""
        label.setText(str(value))
        self._push_undo()
        self._current_adjustments[key] = value
        self._emit_change()
        self._update_all_indicators()
    
    def _push_undo(self):
        """Push current state to undo stack."""
        self._undo_stack.append(copy.deepcopy(self._current_adjustments))
        if len(self._undo_stack) > self._history_limit:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_history_buttons()
    
    def _emit_change(self):
        """Emit adjustment changed signal."""
        self.adjustment_changed.emit(self._current_adjustments)
    
    def _update_history_buttons(self):
        """Update undo/redo button states."""
        self.undo_btn.setEnabled(len(self._undo_stack) > 0)
        self.redo_btn.setEnabled(len(self._redo_stack) > 0)
    
    def _update_all_indicators(self):
        """Update all section indicators based on current adjustments."""
        # Light section
        light_active = (self._current_adjustments['brightness'] != 0 or 
                       self._current_adjustments['contrast'] != 0)
        if 'light' in self._sections:
            self._sections['light'].set_active(light_active)
        
        # Color basic section
        color_active = (self._current_adjustments['saturation'] != 0 or 
                       self._current_adjustments['hue'] != 0)
        if 'color_basic' in self._sections:
            self._sections['color_basic'].set_active(color_active)
        
        # WB section
        wb_active = (self._current_adjustments['temp'] != 0 or 
                    self._current_adjustments['tint'] != 0)
        if 'wb' in self._sections:
            self._sections['wb'].set_active(wb_active)
        
        # Levels section
        levels_active = (self._current_adjustments['levels_in_black'] != 0 or
                        self._current_adjustments['levels_in_white'] != 255 or
                        self._current_adjustments['levels_gamma'] != 1.0 or
                        self._current_adjustments['levels_out_black'] != 0 or
                        self._current_adjustments['levels_out_white'] != 255)
        if 'levels' in self._sections:
            self._sections['levels'].set_active(levels_active)
        
        # Curves section
        identity = [[0, 0], [255, 255]]
        curves_active = (self._current_adjustments['curves_rgb'] != identity or
                        self._current_adjustments['curves_red'] != identity or
                        self._current_adjustments['curves_green'] != identity or
                        self._current_adjustments['curves_blue'] != identity)
        if 'curves' in self._sections:
            self._sections['curves'].set_active(curves_active)
        
        # Mixer section
        mixer_active = False
        for ch in ['red', 'green', 'blue']:
            defaults = {'r': 100 if ch == 'red' else 0, 'g': 100 if ch == 'green' else 0, 
                       'b': 100 if ch == 'blue' else 0, 'const': 0}
            for comp, default in defaults.items():
                if self._current_adjustments.get(f'mixer_{ch}_{comp}', default) != default:
                    mixer_active = True
                    break
        if 'mixer' in self._sections:
            self._sections['mixer'].set_active(mixer_active)
        
        # NR section
        nr_active = self._current_adjustments['noise_reduction_strength'] > 0
        if 'nr' in self._sections:
            self._sections['nr'].set_active(nr_active)
        
        # Dust section
        dust_active = self._current_adjustments.get('dust_removal_enabled', False)
        if 'dust' in self._sections:
            self._sections['dust'].set_active(dust_active)
        
        # HSL section
        hsl_active = False
        for color in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas']:
            for comp in ['h', 's', 'l']:
                if self._current_adjustments.get(f'hsl_{color}_{comp}', 0) != 0:
                    hsl_active = True
                    break
        if 'hsl' in self._sections:
            self._sections['hsl'].set_active(hsl_active)
        
        # Selective color section
        sel_active = False
        for color in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas', 
                     'whites', 'neutrals', 'blacks']:
            for comp in ['c', 'm', 'y', 'k']:
                if self._current_adjustments.get(f'sel_{color}_{comp}', 0) != 0:
                    sel_active = True
                    break
        if 'selective' in self._sections:
            self._sections['selective'].set_active(sel_active)
    
    def _toggle_compact_mode(self, enabled):
        """Toggle compact mode."""
        self._compact_mode = enabled
        # In compact mode, collapse all sections except the first in each tab
        for name, section in self._sections.items():
            if enabled:
                # Keep only essential sections expanded
                essential = ['light', 'wb']
                section.set_expanded(name in essential)
            # When disabling compact mode, don't auto-expand everything
    
    def _reset_section(self, keys):
        """Reset specific adjustment keys to defaults."""
        self._push_undo()
        for key in keys:
            if key in self._default_adjustments:
                self._current_adjustments[key] = copy.deepcopy(self._default_adjustments[key])
        self._update_ui_from_adjustments()
        self._emit_change()
        self._update_all_indicators()
    
    def _reset_hsl(self):
        """Reset all HSL adjustments."""
        keys = []
        for color in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas']:
            for comp in ['h', 's', 'l']:
                keys.append(f'hsl_{color}_{comp}')
        self._reset_section(keys)
    
    def _reset_selective_color(self):
        """Reset all selective color adjustments."""
        keys = ['sel_relative']
        for color in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas',
                     'whites', 'neutrals', 'blacks']:
            for comp in ['c', 'm', 'y', 'k']:
                keys.append(f'sel_{color}_{comp}')
        self._reset_section(keys)

    # --- Event Handlers ---
    
    def _on_awb_clicked(self):
        """Handle Auto WB button click."""
        method_map = {'Gray World': 'gray_world', 'White Patch': 'white_patch',
                     'Simple WB': 'simple_wb', 'Learning WB': 'learning_wb'}
        method = method_map.get(self.awb_combo.currentText(), 'gray_world')
        self.awb_requested.emit(method)
    
    def _on_auto_color_clicked(self):
        """Handle Auto Color button click."""
        method_map = {'Gamma': 'gamma', 'Recolor': 'recolor', 'None': 'none'}
        method = method_map.get(self.ac_combo.currentText(), 'gamma')
        self.auto_color_requested.emit(method)
    
    def _on_auto_level_clicked(self):
        """Handle Auto Level button click."""
        mode_map = {'Luminance': 'luminance', 'Lightness': 'lightness',
                   'Brightness': 'brightness', 'Gray': 'gray', 
                   'Average': 'average', 'RGB': 'rgb'}
        mode = mode_map.get(self.auto_level_combo.currentText(), 'luminance')
        midrange = self.auto_level_mid.value()
        self.auto_level_requested.emit(mode, midrange)
    
    def _on_levels_changed(self):
        """Handle levels value change."""
        self._push_undo()
        self._current_adjustments['levels_in_black'] = self.levels_in_black.value()
        self._current_adjustments['levels_in_white'] = self.levels_in_white.value()
        self._current_adjustments['levels_gamma'] = self.levels_gamma.value()
        self._current_adjustments['levels_out_black'] = self.levels_out_black.value()
        self._current_adjustments['levels_out_white'] = self.levels_out_white.value()
        self._emit_change()
        self._update_all_indicators()
    
    def _on_curve_changed(self, channel, points):
        """Handle curve change."""
        self._push_undo()
        key_map = {'RGB': 'curves_rgb', 'Red': 'curves_red', 
                  'Green': 'curves_green', 'Blue': 'curves_blue'}
        key = key_map.get(channel)
        if key:
            self._current_adjustments[key] = points
            self._emit_change()
            self._update_all_indicators()
    
    def _on_mixer_changed(self):
        """Handle mixer value change."""
        self._push_undo()
        channel = self.mixer_channel.currentText().lower()
        self._current_adjustments[f'mixer_{channel}_r'] = self.mixer_r_slider.value()
        self._current_adjustments[f'mixer_{channel}_g'] = self.mixer_g_slider.value()
        self._current_adjustments[f'mixer_{channel}_b'] = self.mixer_b_slider.value()
        self._current_adjustments[f'mixer_{channel}_const'] = self.mixer_const_slider.value()
        self._emit_change()
        self._update_all_indicators()
    
    def _update_mixer_display(self):
        """Update mixer sliders for selected channel."""
        channel = self.mixer_channel.currentText().lower()
        
        # Block signals while updating
        for slider in [self.mixer_r_slider, self.mixer_g_slider, 
                      self.mixer_b_slider, self.mixer_const_slider]:
            slider.blockSignals(True)
        
        # Set values
        defaults = {'red': {'r': 100, 'g': 0, 'b': 0},
                   'green': {'r': 0, 'g': 100, 'b': 0},
                   'blue': {'r': 0, 'g': 0, 'b': 100}}
        
        self.mixer_r_slider.setValue(self._current_adjustments.get(
            f'mixer_{channel}_r', defaults[channel]['r']))
        self.mixer_g_slider.setValue(self._current_adjustments.get(
            f'mixer_{channel}_g', defaults[channel]['g']))
        self.mixer_b_slider.setValue(self._current_adjustments.get(
            f'mixer_{channel}_b', defaults[channel]['b']))
        self.mixer_const_slider.setValue(self._current_adjustments.get(
            f'mixer_{channel}_const', 0))
        
        # Update labels
        self.mixer_r_label.setText(str(self.mixer_r_slider.value()))
        self.mixer_g_label.setText(str(self.mixer_g_slider.value()))
        self.mixer_b_label.setText(str(self.mixer_b_slider.value()))
        self.mixer_const_label.setText(str(self.mixer_const_slider.value()))
        
        # Unblock signals
        for slider in [self.mixer_r_slider, self.mixer_g_slider,
                      self.mixer_b_slider, self.mixer_const_slider]:
            slider.blockSignals(False)
    
    def _on_dust_changed(self):
        """Handle dust removal settings change."""
        self._push_undo()
        self._current_adjustments['dust_removal_enabled'] = self.dust_enabled.isChecked()
        self._current_adjustments['dust_removal_sensitivity'] = self.dust_sensitivity_slider.value()
        self._current_adjustments['dust_removal_radius'] = self.dust_radius_slider.value()
        self._emit_change()
        self._update_all_indicators()
    
    def _on_hsl_changed(self):
        """Handle HSL value change."""
        self._push_undo()
        color = self.hsl_color.currentText().lower()
        self._current_adjustments[f'hsl_{color}_h'] = self.hsl_h_slider.value()
        self._current_adjustments[f'hsl_{color}_s'] = self.hsl_s_slider.value()
        self._current_adjustments[f'hsl_{color}_l'] = self.hsl_l_slider.value()
        self._emit_change()
        self._update_all_indicators()
    
    def _update_hsl_display(self):
        """Update HSL sliders for selected color."""
        color = self.hsl_color.currentText().lower()
        
        for slider in [self.hsl_h_slider, self.hsl_s_slider, self.hsl_l_slider]:
            slider.blockSignals(True)
        
        self.hsl_h_slider.setValue(self._current_adjustments.get(f'hsl_{color}_h', 0))
        self.hsl_s_slider.setValue(self._current_adjustments.get(f'hsl_{color}_s', 0))
        self.hsl_l_slider.setValue(self._current_adjustments.get(f'hsl_{color}_l', 0))
        
        self.hsl_h_label.setText(str(self.hsl_h_slider.value()))
        self.hsl_s_label.setText(str(self.hsl_s_slider.value()))
        self.hsl_l_label.setText(str(self.hsl_l_slider.value()))
        
        for slider in [self.hsl_h_slider, self.hsl_s_slider, self.hsl_l_slider]:
            slider.blockSignals(False)
    
    def _on_sel_changed(self):
        """Handle selective color value change."""
        self._push_undo()
        color = self.sel_color.currentText().lower()
        self._current_adjustments[f'sel_{color}_c'] = self.sel_c_slider.value()
        self._current_adjustments[f'sel_{color}_m'] = self.sel_m_slider.value()
        self._current_adjustments[f'sel_{color}_y'] = self.sel_y_slider.value()
        self._current_adjustments[f'sel_{color}_k'] = self.sel_k_slider.value()
        self._current_adjustments['sel_relative'] = self.sel_relative.isChecked()
        self._emit_change()
        self._update_all_indicators()
    
    def _update_sel_display(self):
        """Update selective color sliders for selected color."""
        color = self.sel_color.currentText().lower()
        
        for slider in [self.sel_c_slider, self.sel_m_slider, 
                      self.sel_y_slider, self.sel_k_slider]:
            slider.blockSignals(True)
        
        self.sel_c_slider.setValue(self._current_adjustments.get(f'sel_{color}_c', 0))
        self.sel_m_slider.setValue(self._current_adjustments.get(f'sel_{color}_m', 0))
        self.sel_y_slider.setValue(self._current_adjustments.get(f'sel_{color}_y', 0))
        self.sel_k_slider.setValue(self._current_adjustments.get(f'sel_{color}_k', 0))
        
        self.sel_c_label.setText(str(self.sel_c_slider.value()))
        self.sel_m_label.setText(str(self.sel_m_slider.value()))
        self.sel_y_label.setText(str(self.sel_y_slider.value()))
        self.sel_k_label.setText(str(self.sel_k_slider.value()))
        
        for slider in [self.sel_c_slider, self.sel_m_slider,
                      self.sel_y_slider, self.sel_k_slider]:
            slider.blockSignals(False)


    # --- Public API ---
    
    def get_adjustments(self):
        """Get current adjustment values."""
        return copy.deepcopy(self._current_adjustments)
    
    def set_adjustments(self, adjustments: dict):
        """Set adjustment values from a dictionary."""
        for key, value in adjustments.items():
            if key in self._current_adjustments:
                self._current_adjustments[key] = copy.deepcopy(value)
        self._update_ui_from_adjustments()
        self._update_all_indicators()
    
    def reset_adjustments(self):
        """Reset all adjustments to defaults."""
        self._push_undo()
        self._current_adjustments = copy.deepcopy(self._default_adjustments)
        self._update_ui_from_adjustments()
        self._emit_change()
        self._update_all_indicators()
    
    def undo_adjustment(self):
        """Undo last adjustment change."""
        if self._undo_stack:
            self._redo_stack.append(copy.deepcopy(self._current_adjustments))
            self._current_adjustments = self._undo_stack.pop()
            self._update_ui_from_adjustments()
            self._emit_change()
            self._update_all_indicators()
            self._update_history_buttons()
    
    def redo_adjustment(self):
        """Redo last undone adjustment change."""
        if self._redo_stack:
            self._undo_stack.append(copy.deepcopy(self._current_adjustments))
            self._current_adjustments = self._redo_stack.pop()
            self._update_ui_from_adjustments()
            self._emit_change()
            self._update_all_indicators()
            self._update_history_buttons()
    
    def _update_ui_from_adjustments(self):
        """Update all UI elements from current adjustments."""
        # Block all signals during update
        self._block_all_signals(True)
        
        try:
            # Basic tab
            self.brightness_slider.setValue(self._current_adjustments['brightness'])
            self.brightness_label.setText(str(self._current_adjustments['brightness']))
            self.contrast_slider.setValue(self._current_adjustments['contrast'])
            self.contrast_label.setText(str(self._current_adjustments['contrast']))
            self.saturation_slider.setValue(self._current_adjustments['saturation'])
            self.saturation_label.setText(str(self._current_adjustments['saturation']))
            self.hue_slider.setValue(self._current_adjustments['hue'])
            self.hue_label.setText(str(self._current_adjustments['hue']))
            self.temp_slider.setValue(self._current_adjustments['temp'])
            self.temp_label.setText(str(self._current_adjustments['temp']))
            self.tint_slider.setValue(self._current_adjustments['tint'])
            self.tint_label.setText(str(self._current_adjustments['tint']))
            
            # Advanced tab - Levels
            self.levels_in_black.setValue(self._current_adjustments['levels_in_black'])
            self.levels_in_white.setValue(self._current_adjustments['levels_in_white'])
            self.levels_gamma.setValue(self._current_adjustments['levels_gamma'])
            self.levels_out_black.setValue(self._current_adjustments['levels_out_black'])
            self.levels_out_white.setValue(self._current_adjustments['levels_out_white'])
            
            # Curves
            self.curves_widget.set_curve_points('RGB', self._current_adjustments['curves_rgb'])
            self.curves_widget.set_curve_points('Red', self._current_adjustments['curves_red'])
            self.curves_widget.set_curve_points('Green', self._current_adjustments['curves_green'])
            self.curves_widget.set_curve_points('Blue', self._current_adjustments['curves_blue'])
            
            # Mixer
            self._update_mixer_display()
            
            # Noise reduction
            self.nr_slider.setValue(self._current_adjustments['noise_reduction_strength'])
            self.nr_label.setText(str(self._current_adjustments['noise_reduction_strength']))
            
            # Dust removal
            self.dust_enabled.setChecked(self._current_adjustments.get('dust_removal_enabled', False))
            self.dust_sensitivity_slider.setValue(self._current_adjustments.get('dust_removal_sensitivity', 50))
            self.dust_sensitivity_label.setText(str(self.dust_sensitivity_slider.value()))
            self.dust_radius_slider.setValue(self._current_adjustments.get('dust_removal_radius', 3))
            self.dust_radius_label.setText(str(self.dust_radius_slider.value()))
            
            # Color tab - HSL
            self._update_hsl_display()
            
            # Selective color
            self._update_sel_display()
            self.sel_relative.setChecked(self._current_adjustments.get('sel_relative', True))
            
        finally:
            self._block_all_signals(False)
    
    def _block_all_signals(self, block):
        """Block or unblock signals from all input widgets."""
        widgets = [
            self.brightness_slider, self.contrast_slider, self.saturation_slider,
            self.hue_slider, self.temp_slider, self.tint_slider,
            self.levels_in_black, self.levels_in_white, self.levels_gamma,
            self.levels_out_black, self.levels_out_white,
            self.mixer_r_slider, self.mixer_g_slider, self.mixer_b_slider, self.mixer_const_slider,
            self.nr_slider, self.dust_enabled, self.dust_sensitivity_slider, self.dust_radius_slider,
            self.hsl_h_slider, self.hsl_s_slider, self.hsl_l_slider,
            self.sel_c_slider, self.sel_m_slider, self.sel_y_slider, self.sel_k_slider,
            self.sel_relative, self.curves_widget
        ]
        for widget in widgets:
            widget.blockSignals(block)
    
    # --- Preset Management ---
    
    def _refresh_preset_list(self):
        """Refresh the preset combo box."""
        self.preset_combo.clear()
        self._preset_id_by_name = {}
        
        presets = self._preset_manager.list_presets()
        for preset in presets:
            name = preset.get('name', preset.get('id', 'Unknown'))
            preset_id = preset.get('id', name)
            self.preset_combo.addItem(name)
            self._preset_id_by_name[name] = preset_id
    
    def _apply_preset(self):
        """Apply selected preset."""
        name = self.preset_combo.currentText()
        preset_id = self._preset_id_by_name.get(name)
        if preset_id:
            preset_data = self._preset_manager.get_preset(preset_id)
            if preset_data and 'parameters' in preset_data:
                self._push_undo()
                self.set_adjustments(preset_data['parameters'])
                self._emit_change()
    
    def _save_preset(self):
        """Save current adjustments as a preset."""
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name:
            success, result = self._preset_manager.add_preset(
                name=name,
                parameters=self.get_adjustments(),
                overwrite=True
            )
            if success:
                self._refresh_preset_list()
                QMessageBox.information(self, "Saved", f"Preset '{name}' saved.")
            else:
                QMessageBox.warning(self, "Error", f"Failed to save preset: {result}")


# For backwards compatibility, keep the old class name working
# by importing from this module
