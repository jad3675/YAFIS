"""
Settings dialog with collapsible sections for better organization.
"""

import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QDialogButtonBox,
    QLabel, QMessageBox, QCheckBox, QScrollArea, QFrame, QHBoxLayout,
    QGroupBox, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..utils.logger import get_logger
from ..config import settings as app_settings

logger = get_logger(__name__)


class CollapsibleSection(QWidget):
    """A collapsible section widget with header and content area."""
    
    def __init__(self, title: str, parent=None, initially_collapsed: bool = True):
        super().__init__(parent)
        self._is_collapsed = initially_collapsed
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header button
        self._header = QPushButton(self._get_arrow() + " " + title)
        self._header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 10px;
                font-weight: bold;
                background-color: palette(button);
                border: 1px solid palette(mid);
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: palette(light);
            }
        """)
        self._header.clicked.connect(self._toggle)
        self._title = title
        layout.addWidget(self._header)
        
        # Content frame
        self._content = QFrame()
        self._content.setFrameShape(QFrame.Shape.StyledPanel)
        self._content.setStyleSheet("QFrame { border: 1px solid palette(mid); border-top: none; }")
        self._content_layout = QFormLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._content)
        
        # Set initial state
        self._content.setVisible(not initially_collapsed)
    
    def _get_arrow(self) -> str:
        return "▶" if self._is_collapsed else "▼"
    
    def _toggle(self):
        self._is_collapsed = not self._is_collapsed
        self._content.setVisible(not self._is_collapsed)
        self._header.setText(self._get_arrow() + " " + self._title)
    
    def add_row(self, label, widget):
        """Add a row to the content area."""
        self._content_layout.addRow(label, widget)
    
    def add_widget(self, widget):
        """Add a widget spanning the full width."""
        self._content_layout.addRow(widget)
    
    def expand(self):
        """Expand the section."""
        if self._is_collapsed:
            self._toggle()
    
    def collapse(self):
        """Collapse the section."""
        if not self._is_collapsed:
            self._toggle()


class SettingsDialog(QDialog):
    """Settings dialog with organized collapsible sections."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Settings")
        self.setMinimumWidth(550)
        self.setMinimumHeight(600)

        layout = QVBoxLayout(self)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self._create_conversion_tab()
        self._create_ui_output_tab()
        self._create_system_tab()

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.load_settings()

    def _create_conversion_tab(self):
        """Create the conversion settings tab with collapsible sections."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(8)
        
        # Info label
        info = QLabel("<i>Note: Changes require application restart to take effect.</i>")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # === General Mask Detection ===
        self._general_section = CollapsibleSection("General Detection Settings", initially_collapsed=True)
        
        self.mask_sample_size = QSpinBox()
        self.mask_sample_size.setRange(1, 100)
        self.mask_sample_size.setToolTip("Size of corner samples for mask detection")
        self._general_section.add_row("Sample Size:", self.mask_sample_size)
        
        self.mask_clear_sat_max = QSpinBox()
        self.mask_clear_sat_max.setRange(0, 255)
        self.mask_clear_sat_max.setToolTip("Maximum saturation to classify as clear/neutral base")
        self._general_section.add_row("Clear Base Max Saturation:", self.mask_clear_sat_max)
        
        layout.addWidget(self._general_section)
        
        # === C-41 Detection ===
        self._c41_section = CollapsibleSection("C-41 (Color Negative) Detection", initially_collapsed=True)
        
        self.mask_c41_hue_min = QSpinBox()
        self.mask_c41_hue_min.setRange(0, 180)
        self._c41_section.add_row("Hue Min:", self.mask_c41_hue_min)
        
        self.mask_c41_hue_max = QSpinBox()
        self.mask_c41_hue_max.setRange(0, 180)
        self._c41_section.add_row("Hue Max:", self.mask_c41_hue_max)
        
        self.mask_c41_sat_min = QSpinBox()
        self.mask_c41_sat_min.setRange(0, 255)
        self._c41_section.add_row("Saturation Min:", self.mask_c41_sat_min)
        
        self.mask_c41_val_min = QSpinBox()
        self.mask_c41_val_min.setRange(0, 255)
        self._c41_section.add_row("Value Min:", self.mask_c41_val_min)
        
        self.mask_c41_val_max = QSpinBox()
        self.mask_c41_val_max.setRange(0, 255)
        self._c41_section.add_row("Value Max:", self.mask_c41_val_max)
        
        layout.addWidget(self._c41_section)
        
        # === ECN-2 Detection ===
        self._ecn2_section = CollapsibleSection("ECN-2 (Motion Picture) Detection", initially_collapsed=True)
        
        self.mask_ecn2_hue_min = QSpinBox()
        self.mask_ecn2_hue_min.setRange(0, 180)
        self._ecn2_section.add_row("Hue Min:", self.mask_ecn2_hue_min)
        
        self.mask_ecn2_hue_max = QSpinBox()
        self.mask_ecn2_hue_max.setRange(0, 180)
        self._ecn2_section.add_row("Hue Max:", self.mask_ecn2_hue_max)
        
        self.mask_ecn2_sat_min = QSpinBox()
        self.mask_ecn2_sat_min.setRange(0, 255)
        self._ecn2_section.add_row("Saturation Min:", self.mask_ecn2_sat_min)
        
        self.mask_ecn2_val_min = QSpinBox()
        self.mask_ecn2_val_min.setRange(0, 255)
        self._ecn2_section.add_row("Value Min:", self.mask_ecn2_val_min)
        
        self.mask_ecn2_val_max = QSpinBox()
        self.mask_ecn2_val_max.setRange(0, 255)
        self._ecn2_section.add_row("Value Max:", self.mask_ecn2_val_max)
        
        layout.addWidget(self._ecn2_section)
        
        # === E-6 Detection ===
        self._e6_section = CollapsibleSection("E-6 (Slide/Reversal) Detection", initially_collapsed=True)
        
        self.mask_e6_sat_max = QSpinBox()
        self.mask_e6_sat_max.setRange(0, 255)
        self._e6_section.add_row("Saturation Max:", self.mask_e6_sat_max)
        
        self.mask_e6_val_min = QSpinBox()
        self.mask_e6_val_min.setRange(0, 255)
        self._e6_section.add_row("Value Min:", self.mask_e6_val_min)
        
        layout.addWidget(self._e6_section)
        
        # === B&W Detection ===
        self._bw_section = CollapsibleSection("B&W (Black & White) Detection", initially_collapsed=True)
        
        self.mask_bw_sat_max = QSpinBox()
        self.mask_bw_sat_max.setRange(0, 255)
        self._bw_section.add_row("Saturation Max:", self.mask_bw_sat_max)
        
        self.mask_bw_val_min = QSpinBox()
        self.mask_bw_val_min.setRange(0, 255)
        self._bw_section.add_row("Value Min:", self.mask_bw_val_min)
        
        self.mask_bw_val_max = QSpinBox()
        self.mask_bw_val_max.setRange(0, 255)
        self._bw_section.add_row("Value Max:", self.mask_bw_val_max)
        
        layout.addWidget(self._bw_section)
        
        # === White Balance ===
        self._wb_section = CollapsibleSection("White Balance", initially_collapsed=True)
        
        self._wb_section.add_widget(QLabel("<b>Standard (C-41)</b>"))
        
        self.wb_target_gray = QDoubleSpinBox()
        self.wb_target_gray.setRange(0.0, 255.0)
        self.wb_target_gray.setDecimals(1)
        self._wb_section.add_row("Target Gray:", self.wb_target_gray)
        
        self.wb_clamp_min = QDoubleSpinBox()
        self.wb_clamp_min.setRange(0.1, 2.0)
        self.wb_clamp_min.setDecimals(2)
        self.wb_clamp_min.setSingleStep(0.05)
        self._wb_section.add_row("Clamp Min:", self.wb_clamp_min)
        
        self.wb_clamp_max = QDoubleSpinBox()
        self.wb_clamp_max.setRange(0.1, 5.0)
        self.wb_clamp_max.setDecimals(2)
        self.wb_clamp_max.setSingleStep(0.05)
        self._wb_section.add_row("Clamp Max:", self.wb_clamp_max)
        
        self._wb_section.add_widget(QLabel("<b>ECN-2 (Motion Picture)</b>"))
        
        self.wb_target_gray_ecn2 = QDoubleSpinBox()
        self.wb_target_gray_ecn2.setRange(0.0, 255.0)
        self.wb_target_gray_ecn2.setDecimals(1)
        self._wb_section.add_row("Target Gray:", self.wb_target_gray_ecn2)
        
        self.wb_ecn2_clamp_min = QDoubleSpinBox()
        self.wb_ecn2_clamp_min.setRange(0.1, 2.0)
        self.wb_ecn2_clamp_min.setDecimals(2)
        self._wb_section.add_row("Clamp Min:", self.wb_ecn2_clamp_min)
        
        self.wb_ecn2_clamp_max = QDoubleSpinBox()
        self.wb_ecn2_clamp_max.setRange(0.1, 5.0)
        self.wb_ecn2_clamp_max.setDecimals(2)
        self._wb_section.add_row("Clamp Max:", self.wb_ecn2_clamp_max)
        
        self._wb_section.add_widget(QLabel("<b>E-6 (Slide Film - Gentle)</b>"))
        
        self.wb_e6_clamp_min = QDoubleSpinBox()
        self.wb_e6_clamp_min.setRange(0.5, 1.0)
        self.wb_e6_clamp_min.setDecimals(2)
        self._wb_section.add_row("Clamp Min:", self.wb_e6_clamp_min)
        
        self.wb_e6_clamp_max = QDoubleSpinBox()
        self.wb_e6_clamp_max.setRange(1.0, 1.5)
        self.wb_e6_clamp_max.setDecimals(2)
        self._wb_section.add_row("Clamp Max:", self.wb_e6_clamp_max)
        
        layout.addWidget(self._wb_section)
        
        # === Channel Curves ===
        self._curves_section = CollapsibleSection("Channel Curves", initially_collapsed=True)
        
        self.curve_clip_percent = QDoubleSpinBox()
        self.curve_clip_percent.setRange(0.0, 50.0)
        self.curve_clip_percent.setDecimals(2)
        self.curve_clip_percent.setSuffix(" %")
        self._curves_section.add_row("Clip Percent:", self.curve_clip_percent)
        
        self.curve_gamma_red = QDoubleSpinBox()
        self.curve_gamma_red.setRange(0.1, 3.0)
        self.curve_gamma_red.setDecimals(2)
        self.curve_gamma_red.setSingleStep(0.05)
        self._curves_section.add_row("Gamma (Red):", self.curve_gamma_red)
        
        self.curve_gamma_green = QDoubleSpinBox()
        self.curve_gamma_green.setRange(0.1, 3.0)
        self.curve_gamma_green.setDecimals(2)
        self.curve_gamma_green.setSingleStep(0.05)
        self._curves_section.add_row("Gamma (Green):", self.curve_gamma_green)
        
        self.curve_gamma_blue = QDoubleSpinBox()
        self.curve_gamma_blue.setRange(0.1, 3.0)
        self.curve_gamma_blue.setDecimals(2)
        self.curve_gamma_blue.setSingleStep(0.05)
        self._curves_section.add_row("Gamma (Blue):", self.curve_gamma_blue)
        
        self.curve_num_intermediate_points = QSpinBox()
        self.curve_num_intermediate_points.setRange(0, 20)
        self._curves_section.add_row("Intermediate Points:", self.curve_num_intermediate_points)
        
        layout.addWidget(self._curves_section)
        
        # === Color Grading ===
        self._grading_section = CollapsibleSection("Color Grading (LAB/HSV)", initially_collapsed=True)
        
        self._grading_section.add_widget(QLabel("<b>LAB A Channel (Green-Magenta)</b>"))
        
        self.lab_a_target = QDoubleSpinBox()
        self.lab_a_target.setRange(0.0, 255.0)
        self.lab_a_target.setDecimals(1)
        self._grading_section.add_row("Target:", self.lab_a_target)
        
        self.lab_a_correction_factor = QDoubleSpinBox()
        self.lab_a_correction_factor.setRange(0.0, 5.0)
        self.lab_a_correction_factor.setDecimals(2)
        self._grading_section.add_row("Correction Factor:", self.lab_a_correction_factor)
        
        self.lab_a_correction_max = QDoubleSpinBox()
        self.lab_a_correction_max.setRange(0.0, 20.0)
        self.lab_a_correction_max.setDecimals(1)
        self._grading_section.add_row("Correction Max:", self.lab_a_correction_max)
        
        self._grading_section.add_widget(QLabel("<b>LAB B Channel (Blue-Yellow)</b>"))
        
        self.lab_b_target = QDoubleSpinBox()
        self.lab_b_target.setRange(0.0, 255.0)
        self.lab_b_target.setDecimals(1)
        self._grading_section.add_row("Target:", self.lab_b_target)
        
        self.lab_b_correction_factor = QDoubleSpinBox()
        self.lab_b_correction_factor.setRange(0.0, 5.0)
        self.lab_b_correction_factor.setDecimals(2)
        self._grading_section.add_row("Correction Factor:", self.lab_b_correction_factor)
        
        self.lab_b_correction_max = QDoubleSpinBox()
        self.lab_b_correction_max.setRange(0.0, 20.0)
        self.lab_b_correction_max.setDecimals(1)
        self._grading_section.add_row("Correction Max:", self.lab_b_correction_max)
        
        self._grading_section.add_widget(QLabel("<b>HSV Saturation</b>"))
        
        self.hsv_saturation_boost = QDoubleSpinBox()
        self.hsv_saturation_boost.setRange(0.1, 3.0)
        self.hsv_saturation_boost.setDecimals(2)
        self._grading_section.add_row("Saturation Boost:", self.hsv_saturation_boost)
        
        layout.addWidget(self._grading_section)
        
        # Add stretch at bottom
        layout.addStretch()
        
        scroll.setWidget(container)
        self.tab_widget.addTab(scroll, "Conversion")

    def _create_ui_output_tab(self):
        """Create the UI & Output settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        layout.addRow(QLabel("<b>Output Defaults</b>"))
        
        self.default_jpeg_quality = QSpinBox()
        self.default_jpeg_quality.setRange(1, 100)
        self.default_jpeg_quality.setSuffix(" %")
        layout.addRow("JPEG Quality:", self.default_jpeg_quality)
        
        self.default_png_compression = QSpinBox()
        self.default_png_compression.setRange(0, 9)
        layout.addRow("PNG Compression:", self.default_png_compression)
        
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>UI Settings</b>"))
        
        self.filmstrip_thumb_size = QSpinBox()
        self.filmstrip_thumb_size.setRange(32, 512)
        self.filmstrip_thumb_size.setSuffix(" px")
        layout.addRow("Filmstrip Thumbnail Size:", self.filmstrip_thumb_size)
        
        self.apply_embedded_icc_profile = QCheckBox()
        self.apply_embedded_icc_profile.setToolTip(
            "Convert images with embedded ICC profiles to sRGB when loading"
        )
        layout.addRow("Apply Embedded ICC Profile:", self.apply_embedded_icc_profile)
        
        self.tab_widget.addTab(widget, "UI & Output")

    def _create_system_tab(self):
        """Create the System settings tab."""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        layout.addRow(QLabel("<b>GPU Acceleration</b>"))
        
        self.gpu_enabled = QCheckBox()
        self.gpu_enabled.setToolTip(
            "Enable GPU acceleration for image processing.\n"
            "Disable if GPU causes issues."
        )
        layout.addRow("Enable GPU:", self.gpu_enabled)
        
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow("", self.gpu_status_label)
        self._update_gpu_status_label()
        self.gpu_enabled.stateChanged.connect(self._update_gpu_status_label)
        
        layout.addRow(QLabel(""))  # Spacer
        layout.addRow(QLabel("<b>Logging</b>"))
        
        self.logging_level = QComboBox()
        self.logging_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        layout.addRow("Log Level:", self.logging_level)
        
        layout.addRow(QLabel("<i>Note: Log level change requires restart.</i>"))
        
        self.tab_widget.addTab(widget, "System")

    def _update_gpu_status_label(self):
        """Update GPU status label."""
        try:
            from ..utils.gpu import get_gpu_info
            gpu_info = get_gpu_info()
            
            if not gpu_info.get("available", False):
                self.gpu_status_label.setText("No GPU detected")
                self.gpu_enabled.setEnabled(False)
                self.gpu_enabled.setChecked(False)
            else:
                device_name = gpu_info.get("device_name", "Unknown GPU")
                if self.gpu_enabled.isChecked():
                    self.gpu_status_label.setText(f"Will use: {device_name}")
                else:
                    self.gpu_status_label.setText(f"Available: {device_name} (disabled)")
        except Exception as e:
            self.gpu_status_label.setText(f"GPU status unknown: {e}")

    def _get_setting(self, category: str, key: str, default):
        """Get a setting value with fallback to default."""
        try:
            if category == "CONVERSION":
                return app_settings.CONVERSION_DEFAULTS.get(key, default)
            elif category == "UI":
                return app_settings.UI_DEFAULTS.get(key, default)
            return default
        except (KeyError, AttributeError):
            return default

    def load_settings(self):
        """Load current settings into UI widgets."""
        # General detection
        self.mask_sample_size.setValue(self._get_setting("CONVERSION", "mask_sample_size", 10))
        self.mask_clear_sat_max.setValue(self._get_setting("CONVERSION", "mask_clear_sat_max", 40))
        
        # C-41
        self.mask_c41_hue_min.setValue(self._get_setting("CONVERSION", "mask_c41_hue_min", 8))
        self.mask_c41_hue_max.setValue(self._get_setting("CONVERSION", "mask_c41_hue_max", 22))
        self.mask_c41_sat_min.setValue(self._get_setting("CONVERSION", "mask_c41_sat_min", 70))
        self.mask_c41_val_min.setValue(self._get_setting("CONVERSION", "mask_c41_val_min", 60))
        self.mask_c41_val_max.setValue(self._get_setting("CONVERSION", "mask_c41_val_max", 210))
        
        # ECN-2
        self.mask_ecn2_hue_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_hue_min", 5))
        self.mask_ecn2_hue_max.setValue(self._get_setting("CONVERSION", "mask_ecn2_hue_max", 25))
        self.mask_ecn2_sat_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_sat_min", 50))
        self.mask_ecn2_val_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_val_min", 30))
        self.mask_ecn2_val_max.setValue(self._get_setting("CONVERSION", "mask_ecn2_val_max", 80))
        
        # E-6
        self.mask_e6_sat_max.setValue(self._get_setting("CONVERSION", "mask_e6_sat_max", 25))
        self.mask_e6_val_min.setValue(self._get_setting("CONVERSION", "mask_e6_val_min", 200))
        
        # B&W
        self.mask_bw_sat_max.setValue(self._get_setting("CONVERSION", "mask_bw_sat_max", 20))
        self.mask_bw_val_min.setValue(self._get_setting("CONVERSION", "mask_bw_val_min", 100))
        self.mask_bw_val_max.setValue(self._get_setting("CONVERSION", "mask_bw_val_max", 255))
        
        # White Balance
        self.wb_target_gray.setValue(self._get_setting("CONVERSION", "wb_target_gray", 128.0))
        self.wb_clamp_min.setValue(self._get_setting("CONVERSION", "wb_clamp_min", 0.8))
        self.wb_clamp_max.setValue(self._get_setting("CONVERSION", "wb_clamp_max", 1.3))
        self.wb_target_gray_ecn2.setValue(self._get_setting("CONVERSION", "wb_target_gray_ecn2", 140.0))
        self.wb_ecn2_clamp_min.setValue(self._get_setting("CONVERSION", "wb_ecn2_clamp_min", 0.7))
        self.wb_ecn2_clamp_max.setValue(self._get_setting("CONVERSION", "wb_ecn2_clamp_max", 1.5))
        self.wb_e6_clamp_min.setValue(self._get_setting("CONVERSION", "wb_e6_clamp_min", 0.95))
        self.wb_e6_clamp_max.setValue(self._get_setting("CONVERSION", "wb_e6_clamp_max", 1.05))
        
        # Curves
        self.curve_clip_percent.setValue(self._get_setting("CONVERSION", "curve_clip_percent", 0.5))
        self.curve_gamma_red.setValue(self._get_setting("CONVERSION", "curve_gamma_red", 0.95))
        self.curve_gamma_green.setValue(self._get_setting("CONVERSION", "curve_gamma_green", 1.0))
        self.curve_gamma_blue.setValue(self._get_setting("CONVERSION", "curve_gamma_blue", 1.1))
        self.curve_num_intermediate_points.setValue(self._get_setting("CONVERSION", "curve_num_intermediate_points", 5))
        
        # Color Grading
        self.lab_a_target.setValue(self._get_setting("CONVERSION", "lab_a_target", 128.0))
        self.lab_a_correction_factor.setValue(self._get_setting("CONVERSION", "lab_a_correction_factor", 0.5))
        self.lab_a_correction_max.setValue(self._get_setting("CONVERSION", "lab_a_correction_max", 5.0))
        self.lab_b_target.setValue(self._get_setting("CONVERSION", "lab_b_target", 128.0))
        self.lab_b_correction_factor.setValue(self._get_setting("CONVERSION", "lab_b_correction_factor", 0.7))
        self.lab_b_correction_max.setValue(self._get_setting("CONVERSION", "lab_b_correction_max", 10.0))
        self.hsv_saturation_boost.setValue(self._get_setting("CONVERSION", "hsv_saturation_boost", 1.15))
        
        # UI/Output
        self.default_jpeg_quality.setValue(self._get_setting("UI", "default_jpeg_quality", 95))
        self.default_png_compression.setValue(self._get_setting("UI", "default_png_compression", 6))
        self.filmstrip_thumb_size.setValue(self._get_setting("UI", "filmstrip_thumb_size", 120))
        self.apply_embedded_icc_profile.setChecked(bool(self._get_setting("UI", "apply_embedded_icc_profile", False)))
        
        # System
        self.gpu_enabled.setChecked(app_settings.UI_DEFAULTS.get("gpu_acceleration_enabled", True))
        self._update_gpu_status_label()
        
        level = app_settings.LOGGING_LEVEL
        if level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.logging_level.setCurrentText(level)
        else:
            self.logging_level.setCurrentText("INFO")

    def accept(self):
        """Save settings when OK is clicked."""
        try:
            settings_to_save = {
                "CONVERSION_DEFAULTS": {
                    # General Detection
                    "mask_sample_size": self.mask_sample_size.value(),
                    "mask_clear_sat_max": self.mask_clear_sat_max.value(),
                    # C-41
                    "mask_c41_hue_min": self.mask_c41_hue_min.value(),
                    "mask_c41_hue_max": self.mask_c41_hue_max.value(),
                    "mask_c41_sat_min": self.mask_c41_sat_min.value(),
                    "mask_c41_val_min": self.mask_c41_val_min.value(),
                    "mask_c41_val_max": self.mask_c41_val_max.value(),
                    # ECN-2
                    "mask_ecn2_hue_min": self.mask_ecn2_hue_min.value(),
                    "mask_ecn2_hue_max": self.mask_ecn2_hue_max.value(),
                    "mask_ecn2_sat_min": self.mask_ecn2_sat_min.value(),
                    "mask_ecn2_val_min": self.mask_ecn2_val_min.value(),
                    "mask_ecn2_val_max": self.mask_ecn2_val_max.value(),
                    # E-6
                    "mask_e6_sat_max": self.mask_e6_sat_max.value(),
                    "mask_e6_val_min": self.mask_e6_val_min.value(),
                    # B&W
                    "mask_bw_sat_max": self.mask_bw_sat_max.value(),
                    "mask_bw_val_min": self.mask_bw_val_min.value(),
                    "mask_bw_val_max": self.mask_bw_val_max.value(),
                    # White Balance
                    "wb_target_gray": self.wb_target_gray.value(),
                    "wb_clamp_min": self.wb_clamp_min.value(),
                    "wb_clamp_max": self.wb_clamp_max.value(),
                    "wb_target_gray_ecn2": self.wb_target_gray_ecn2.value(),
                    "wb_ecn2_clamp_min": self.wb_ecn2_clamp_min.value(),
                    "wb_ecn2_clamp_max": self.wb_ecn2_clamp_max.value(),
                    "wb_e6_clamp_min": self.wb_e6_clamp_min.value(),
                    "wb_e6_clamp_max": self.wb_e6_clamp_max.value(),
                    # Curves
                    "curve_clip_percent": self.curve_clip_percent.value(),
                    "curve_gamma_red": self.curve_gamma_red.value(),
                    "curve_gamma_green": self.curve_gamma_green.value(),
                    "curve_gamma_blue": self.curve_gamma_blue.value(),
                    "curve_num_intermediate_points": self.curve_num_intermediate_points.value(),
                    # Color Grading
                    "lab_a_target": self.lab_a_target.value(),
                    "lab_a_correction_factor": self.lab_a_correction_factor.value(),
                    "lab_a_correction_max": self.lab_a_correction_max.value(),
                    "lab_b_target": self.lab_b_target.value(),
                    "lab_b_correction_factor": self.lab_b_correction_factor.value(),
                    "lab_b_correction_max": self.lab_b_correction_max.value(),
                    "hsv_saturation_boost": self.hsv_saturation_boost.value(),
                },
                "UI_DEFAULTS": {
                    "default_jpeg_quality": self.default_jpeg_quality.value(),
                    "default_png_compression": self.default_png_compression.value(),
                    "filmstrip_thumb_size": self.filmstrip_thumb_size.value(),
                    "apply_embedded_icc_profile": self.apply_embedded_icc_profile.isChecked(),
                    "gpu_acceleration_enabled": self.gpu_enabled.isChecked(),
                },
                "LOGGING_LEVEL": self.logging_level.currentText()
            }

            # Merge with existing settings
            existing = app_settings.user_settings.get("CONVERSION_DEFAULTS", {})
            existing.update(settings_to_save["CONVERSION_DEFAULTS"])
            settings_to_save["CONVERSION_DEFAULTS"] = existing

            if app_settings.save_user_settings(settings_to_save):
                # Apply GPU setting immediately
                try:
                    from ..utils.gpu import set_gpu_enabled, is_gpu_available
                    if is_gpu_available():
                        set_gpu_enabled(self.gpu_enabled.isChecked())
                except Exception as e:
                    logger.warning("Could not apply GPU setting: %s", e)
                
                QMessageBox.information(
                    self, "Settings Saved",
                    "Settings saved. Some changes require restart."
                )
                super().accept()
            else:
                QMessageBox.warning(self, "Error", "Could not save settings.")
        except Exception:
            logger.exception("Error saving settings")
            QMessageBox.critical(self, "Error", "Unexpected error saving settings.")


if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = SettingsDialog()
    dialog.exec()
