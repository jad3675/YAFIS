import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout,
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPushButton, QDialogButtonBox,
    QLabel, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt

from ..utils.logger import get_logger

# Import the settings and the save function using relative import
# Go up one level from 'ui' to 'negative_converter', then down to 'config'
from ..config import settings as app_settings

logger = get_logger(__name__)

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Application Settings")
        self.setMinimumWidth(500)

        # Main layout
        layout = QVBoxLayout(self)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.conversion_tab = QWidget()
        self.ui_output_tab = QWidget()
        self.system_tab = QWidget()

        self.tab_widget.addTab(self.conversion_tab, "Conversion")
        self.tab_widget.addTab(self.ui_output_tab, "UI & Output")
        self.tab_widget.addTab(self.system_tab, "System")

        # Populate tabs
        self._create_conversion_tab()
        self._create_ui_output_tab()
        self._create_system_tab()

        # Dialog buttons (OK, Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.accept) # Connect Ok to accept
        self.button_box.rejected.connect(self.reject) # Connect Cancel to reject
        layout.addWidget(self.button_box)

        self.load_settings()

    def _create_conversion_tab(self):
        layout = QFormLayout(self.conversion_tab)
        layout.addRow(QLabel("<i>Note: Complex settings like the correction matrix are not editable here.</i>"))
        layout.addRow(QLabel("<i><b>Note:</b> Changes to Conversion parameters require an <b>application restart</b>.</i>"))

        # --- Mask Detection ---
        layout.addRow(QLabel("<b>Mask Detection</b>"))
        self.mask_sample_size = QSpinBox()
        self.mask_sample_size.setRange(1, 100)
        layout.addRow("Mask Sample Size:", self.mask_sample_size)

        self.mask_clear_sat_max = QSpinBox()
        self.mask_clear_sat_max.setRange(0, 255)
        layout.addRow("Mask Clear Max Saturation:", self.mask_clear_sat_max)

        self.mask_c41_hue_min = QSpinBox()
        self.mask_c41_hue_min.setRange(0, 360)
        layout.addRow("C41 Mask Hue Min:", self.mask_c41_hue_min)

        self.mask_c41_hue_max = QSpinBox()
        self.mask_c41_hue_max.setRange(0, 360)
        layout.addRow("C41 Mask Hue Max:", self.mask_c41_hue_max)

        self.mask_c41_sat_min = QSpinBox()
        self.mask_c41_sat_min.setRange(0, 255)
        layout.addRow("C41 Mask Saturation Min:", self.mask_c41_sat_min)

        self.mask_c41_val_min = QSpinBox()
        self.mask_c41_val_min.setRange(0, 255)
        layout.addRow("C41 Mask Value Min:", self.mask_c41_val_min)

        self.mask_c41_val_max = QSpinBox()
        self.mask_c41_val_max.setRange(0, 255)
        layout.addRow("C41 Mask Value Max:", self.mask_c41_val_max)

        # --- ECN-2 (Motion Picture) Detection ---
        layout.addRow(QLabel("<b>ECN-2 Detection</b>"))
        self.mask_ecn2_hue_min = QSpinBox()
        self.mask_ecn2_hue_min.setRange(0, 360)
        layout.addRow("ECN-2 Hue Min:", self.mask_ecn2_hue_min)

        self.mask_ecn2_hue_max = QSpinBox()
        self.mask_ecn2_hue_max.setRange(0, 360)
        layout.addRow("ECN-2 Hue Max:", self.mask_ecn2_hue_max)

        self.mask_ecn2_sat_min = QSpinBox()
        self.mask_ecn2_sat_min.setRange(0, 255)
        layout.addRow("ECN-2 Saturation Min:", self.mask_ecn2_sat_min)

        self.mask_ecn2_val_min = QSpinBox()
        self.mask_ecn2_val_min.setRange(0, 255)
        layout.addRow("ECN-2 Value Min:", self.mask_ecn2_val_min)

        self.mask_ecn2_val_max = QSpinBox()
        self.mask_ecn2_val_max.setRange(0, 255)
        layout.addRow("ECN-2 Value Max:", self.mask_ecn2_val_max)

        # --- E-6 (Slide Film) Detection ---
        layout.addRow(QLabel("<b>E-6 Detection</b>"))
        self.mask_e6_sat_max = QSpinBox()
        self.mask_e6_sat_max.setRange(0, 255)
        layout.addRow("E-6 Saturation Max:", self.mask_e6_sat_max)

        self.mask_e6_val_min = QSpinBox()
        self.mask_e6_val_min.setRange(0, 255)
        layout.addRow("E-6 Value Min:", self.mask_e6_val_min)

        # --- B&W Detection ---
        layout.addRow(QLabel("<b>B&W Detection</b>"))
        self.mask_bw_sat_max = QSpinBox()
        self.mask_bw_sat_max.setRange(0, 255)
        layout.addRow("B&W Saturation Max:", self.mask_bw_sat_max)

        self.mask_bw_val_min = QSpinBox()
        self.mask_bw_val_min.setRange(0, 255)
        layout.addRow("B&W Value Min:", self.mask_bw_val_min)

        self.mask_bw_val_max = QSpinBox()
        self.mask_bw_val_max.setRange(0, 255)
        layout.addRow("B&W Value Max:", self.mask_bw_val_max)

        # --- White Balance ---
        layout.addRow(QLabel("<b>White Balance</b>"))
        self.wb_target_gray = QDoubleSpinBox()
        self.wb_target_gray.setRange(0.0, 255.0)
        self.wb_target_gray.setDecimals(1)
        layout.addRow("WB Target Gray:", self.wb_target_gray)

        self.wb_clamp_min = QDoubleSpinBox()
        self.wb_clamp_min.setRange(0.1, 2.0)
        self.wb_clamp_min.setDecimals(2)
        self.wb_clamp_min.setSingleStep(0.05)
        layout.addRow("WB Clamp Min:", self.wb_clamp_min)

        self.wb_clamp_max = QDoubleSpinBox()
        self.wb_clamp_max.setRange(0.1, 5.0) # Increased max range slightly
        self.wb_clamp_max.setDecimals(2)
        self.wb_clamp_max.setSingleStep(0.05)
        layout.addRow("WB Clamp Max:", self.wb_clamp_max)

        # --- Channel Curve ---
        layout.addRow(QLabel("<b>Channel Curves</b>"))
        self.curve_clip_percent = QDoubleSpinBox()
        self.curve_clip_percent.setRange(0.0, 50.0)
        self.curve_clip_percent.setDecimals(2)
        self.curve_clip_percent.setSuffix(" %")
        layout.addRow("Curve Clip Percent:", self.curve_clip_percent)

        self.curve_gamma_red = QDoubleSpinBox()
        self.curve_gamma_red.setRange(0.1, 3.0)
        self.curve_gamma_red.setDecimals(2)
        self.curve_gamma_red.setSingleStep(0.05)
        layout.addRow("Curve Gamma (Red):", self.curve_gamma_red)

        self.curve_gamma_green = QDoubleSpinBox()
        self.curve_gamma_green.setRange(0.1, 3.0)
        self.curve_gamma_green.setDecimals(2)
        self.curve_gamma_green.setSingleStep(0.05)
        layout.addRow("Curve Gamma (Green):", self.curve_gamma_green)

        self.curve_gamma_blue = QDoubleSpinBox()
        self.curve_gamma_blue.setRange(0.1, 3.0)
        self.curve_gamma_blue.setDecimals(2)
        self.curve_gamma_blue.setSingleStep(0.05)
        layout.addRow("Curve Gamma (Blue):", self.curve_gamma_blue)

        self.curve_num_intermediate_points = QSpinBox()
        self.curve_num_intermediate_points.setRange(0, 20)
        layout.addRow("Curve Intermediate Points:", self.curve_num_intermediate_points)

        # --- Final Color Grading ---
        layout.addRow(QLabel("<b>Color Grading (LAB/HSV)</b>"))

        self.lab_a_target = QDoubleSpinBox()
        self.lab_a_target.setRange(0.0, 255.0)
        self.lab_a_target.setDecimals(1)
        layout.addRow("LAB A Target:", self.lab_a_target)

        self.lab_a_correction_factor = QDoubleSpinBox()
        self.lab_a_correction_factor.setRange(0.0, 5.0)
        self.lab_a_correction_factor.setDecimals(2)
        self.lab_a_correction_factor.setSingleStep(0.1)
        layout.addRow("LAB A Correction Factor:", self.lab_a_correction_factor)

        self.lab_a_correction_max = QDoubleSpinBox()
        self.lab_a_correction_max.setRange(0.0, 20.0)
        self.lab_a_correction_max.setDecimals(1)
        layout.addRow("LAB A Correction Max:", self.lab_a_correction_max)

        self.lab_b_target = QDoubleSpinBox()
        self.lab_b_target.setRange(0.0, 255.0)
        self.lab_b_target.setDecimals(1)
        layout.addRow("LAB B Target:", self.lab_b_target)

        self.lab_b_correction_factor = QDoubleSpinBox()
        self.lab_b_correction_factor.setRange(0.0, 5.0)
        self.lab_b_correction_factor.setDecimals(2)
        self.lab_b_correction_factor.setSingleStep(0.1)
        layout.addRow("LAB B Correction Factor:", self.lab_b_correction_factor)

        self.lab_b_correction_max = QDoubleSpinBox()
        self.lab_b_correction_max.setRange(0.0, 20.0)
        self.lab_b_correction_max.setDecimals(1)
        layout.addRow("LAB B Correction Max:", self.lab_b_correction_max)

        self.hsv_saturation_boost = QDoubleSpinBox()
        self.hsv_saturation_boost.setRange(0.1, 3.0)
        self.hsv_saturation_boost.setDecimals(2)
        self.hsv_saturation_boost.setSingleStep(0.05)
        layout.addRow("HSV Saturation Boost:", self.hsv_saturation_boost)


    def _create_ui_output_tab(self):
        layout = QFormLayout(self.ui_output_tab)

        self.default_jpeg_quality = QSpinBox()
        self.default_jpeg_quality.setRange(1, 100)
        layout.addRow("Default JPEG Quality:", self.default_jpeg_quality)

        self.default_png_compression = QSpinBox()
        self.default_png_compression.setRange(0, 9)
        layout.addRow("Default PNG Compression:", self.default_png_compression)

        self.filmstrip_thumb_size = QSpinBox()
        self.filmstrip_thumb_size.setRange(32, 512)
        self.filmstrip_thumb_size.setSuffix(" px")
        layout.addRow("Filmstrip Thumbnail Size:", self.filmstrip_thumb_size)

        self.apply_embedded_icc_profile = QCheckBox()
        self.apply_embedded_icc_profile.setToolTip("If enabled, convert images with embedded ICC profiles into sRGB when loading.")
        layout.addRow("Apply embedded ICC profile (convert to sRGB):", self.apply_embedded_icc_profile)

    def _create_system_tab(self):
        layout = QFormLayout(self.system_tab)

        # GPU Acceleration toggle
        layout.addRow(QLabel("<b>GPU Acceleration</b>"))
        self.gpu_enabled = QCheckBox()
        self.gpu_enabled.setToolTip(
            "Enable or disable GPU acceleration for image processing.\n"
            "Disabling may be useful for debugging or if GPU causes issues."
        )
        layout.addRow("Enable GPU Acceleration:", self.gpu_enabled)
        
        # GPU status label
        self.gpu_status_label = QLabel()
        self.gpu_status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addRow("", self.gpu_status_label)
        self._update_gpu_status_label()
        
        # Connect checkbox to update label
        self.gpu_enabled.stateChanged.connect(self._update_gpu_status_label)
        
        layout.addRow(QLabel(""))  # Spacer

        self.logging_level = QComboBox()
        self.logging_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        layout.addRow("Logging Level:", self.logging_level)
        layout.addRow(QLabel("<i><b>Note:</b> Logging level change requires an <b>application restart</b>.</i>"))

    def _update_gpu_status_label(self):
        """Update the GPU status label based on current state."""
        try:
            from ..utils.gpu import get_gpu_info, is_gpu_available
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


    def _get_setting(self, category, key, default_value):
        """Helper to get setting value, handling potential KeyError."""
        try:
            if category == "CONVERSION":
                return app_settings.CONVERSION_DEFAULTS.get(key, default_value)
            elif category == "UI":
                return app_settings.UI_DEFAULTS.get(key, default_value)
            elif category == "LOGGING":
                return app_settings.LOGGING_LEVEL  # Direct access for logging level
            else:
                return default_value
        except KeyError:
            logger.warning("Setting key '%s' not found in '%s' defaults.", key, category)
            return default_value
        except AttributeError:
            logger.warning("Settings category '%s' not found.", category)
            return default_value

    def load_settings(self):
        """Load current settings into the UI widgets."""
        # Conversion Tab
        self.mask_sample_size.setValue(self._get_setting("CONVERSION", "mask_sample_size", 10))
        self.mask_clear_sat_max.setValue(self._get_setting("CONVERSION", "mask_clear_sat_max", 40))
        self.mask_c41_hue_min.setValue(self._get_setting("CONVERSION", "mask_c41_hue_min", 8))
        self.mask_c41_hue_max.setValue(self._get_setting("CONVERSION", "mask_c41_hue_max", 22))
        self.mask_c41_sat_min.setValue(self._get_setting("CONVERSION", "mask_c41_sat_min", 70))
        self.mask_c41_val_min.setValue(self._get_setting("CONVERSION", "mask_c41_val_min", 60))
        self.mask_c41_val_max.setValue(self._get_setting("CONVERSION", "mask_c41_val_max", 210))

        # ECN-2 settings
        self.mask_ecn2_hue_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_hue_min", 5))
        self.mask_ecn2_hue_max.setValue(self._get_setting("CONVERSION", "mask_ecn2_hue_max", 25))
        self.mask_ecn2_sat_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_sat_min", 50))
        self.mask_ecn2_val_min.setValue(self._get_setting("CONVERSION", "mask_ecn2_val_min", 30))
        self.mask_ecn2_val_max.setValue(self._get_setting("CONVERSION", "mask_ecn2_val_max", 80))

        # E-6 settings
        self.mask_e6_sat_max.setValue(self._get_setting("CONVERSION", "mask_e6_sat_max", 25))
        self.mask_e6_val_min.setValue(self._get_setting("CONVERSION", "mask_e6_val_min", 200))

        # B&W settings
        self.mask_bw_sat_max.setValue(self._get_setting("CONVERSION", "mask_bw_sat_max", 20))
        self.mask_bw_val_min.setValue(self._get_setting("CONVERSION", "mask_bw_val_min", 100))
        self.mask_bw_val_max.setValue(self._get_setting("CONVERSION", "mask_bw_val_max", 255))

        self.wb_target_gray.setValue(self._get_setting("CONVERSION", "wb_target_gray", 128.0))
        self.wb_clamp_min.setValue(self._get_setting("CONVERSION", "wb_clamp_min", 0.8))
        self.wb_clamp_max.setValue(self._get_setting("CONVERSION", "wb_clamp_max", 1.3))

        self.curve_clip_percent.setValue(self._get_setting("CONVERSION", "curve_clip_percent", 0.5))
        self.curve_gamma_red.setValue(self._get_setting("CONVERSION", "curve_gamma_red", 0.95))
        self.curve_gamma_green.setValue(self._get_setting("CONVERSION", "curve_gamma_green", 1.0))
        self.curve_gamma_blue.setValue(self._get_setting("CONVERSION", "curve_gamma_blue", 1.1))
        self.curve_num_intermediate_points.setValue(self._get_setting("CONVERSION", "curve_num_intermediate_points", 5))

        self.lab_a_target.setValue(self._get_setting("CONVERSION", "lab_a_target", 128.0))
        self.lab_a_correction_factor.setValue(self._get_setting("CONVERSION", "lab_a_correction_factor", 0.5))
        self.lab_a_correction_max.setValue(self._get_setting("CONVERSION", "lab_a_correction_max", 5.0))
        self.lab_b_target.setValue(self._get_setting("CONVERSION", "lab_b_target", 128.0))
        self.lab_b_correction_factor.setValue(self._get_setting("CONVERSION", "lab_b_correction_factor", 0.7))
        self.lab_b_correction_max.setValue(self._get_setting("CONVERSION", "lab_b_correction_max", 10.0))
        self.hsv_saturation_boost.setValue(self._get_setting("CONVERSION", "hsv_saturation_boost", 1.15))

        # UI/Output Tab
        self.default_jpeg_quality.setValue(self._get_setting("UI", "default_jpeg_quality", 95))
        self.default_png_compression.setValue(self._get_setting("UI", "default_png_compression", 6))
        self.filmstrip_thumb_size.setValue(self._get_setting("UI", "filmstrip_thumb_size", 120))
        self.apply_embedded_icc_profile.setChecked(bool(self._get_setting("UI", "apply_embedded_icc_profile", False)))

        # System Tab
        current_level = self._get_setting("LOGGING", "LOGGING_LEVEL", "INFO")

        # UI/Output Tab
        self.default_jpeg_quality.setValue(app_settings.UI_DEFAULTS.get("default_jpeg_quality", 95))
        self.default_png_compression.setValue(app_settings.UI_DEFAULTS.get("default_png_compression", 6))
        self.filmstrip_thumb_size.setValue(app_settings.UI_DEFAULTS.get("filmstrip_thumb_size", 120))

        # System Tab - GPU and Logging
        gpu_enabled = app_settings.UI_DEFAULTS.get("gpu_acceleration_enabled", True)
        self.gpu_enabled.setChecked(gpu_enabled)
        self._update_gpu_status_label()
        
        current_level = app_settings.LOGGING_LEVEL
        if current_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            self.logging_level.setCurrentText(current_level)
        else:
            self.logging_level.setCurrentText("INFO") # Default fallback


    def accept(self):
        """Save settings when OK is clicked."""
        try:
            settings_to_save = {
                "CONVERSION_DEFAULTS": {
                    # Mask Detection
                    "mask_sample_size": self.mask_sample_size.value(),
                    "mask_clear_sat_max": self.mask_clear_sat_max.value(),
                    "mask_c41_hue_min": self.mask_c41_hue_min.value(),
                    "mask_c41_hue_max": self.mask_c41_hue_max.value(),
                    "mask_c41_sat_min": self.mask_c41_sat_min.value(),
                    "mask_c41_val_min": self.mask_c41_val_min.value(),
                    "mask_c41_val_max": self.mask_c41_val_max.value(),
                    # ECN-2 Detection
                    "mask_ecn2_hue_min": self.mask_ecn2_hue_min.value(),
                    "mask_ecn2_hue_max": self.mask_ecn2_hue_max.value(),
                    "mask_ecn2_sat_min": self.mask_ecn2_sat_min.value(),
                    "mask_ecn2_val_min": self.mask_ecn2_val_min.value(),
                    "mask_ecn2_val_max": self.mask_ecn2_val_max.value(),
                    # E-6 Detection
                    "mask_e6_sat_max": self.mask_e6_sat_max.value(),
                    "mask_e6_val_min": self.mask_e6_val_min.value(),
                    # B&W Detection
                    "mask_bw_sat_max": self.mask_bw_sat_max.value(),
                    "mask_bw_val_min": self.mask_bw_val_min.value(),
                    "mask_bw_val_max": self.mask_bw_val_max.value(),
                    # White Balance
                    "wb_target_gray": self.wb_target_gray.value(),
                    "wb_clamp_min": self.wb_clamp_min.value(),
                    "wb_clamp_max": self.wb_clamp_max.value(),
                    # Channel Curve
                    "curve_clip_percent": self.curve_clip_percent.value(),
                    "curve_gamma_red": self.curve_gamma_red.value(),
                    "curve_gamma_green": self.curve_gamma_green.value(),
                    "curve_gamma_blue": self.curve_gamma_blue.value(),
                    "curve_num_intermediate_points": self.curve_num_intermediate_points.value(),
                    # Final Color Grading
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

            # Merge existing non-editable conversion defaults back in
            # to avoid losing them when saving.
            # A bit simplistic, assumes flat structure for CONVERSION_DEFAULTS in JSON
            existing_conversion = app_settings.user_settings.get("CONVERSION_DEFAULTS", {})
            existing_conversion.update(settings_to_save["CONVERSION_DEFAULTS"])
            settings_to_save["CONVERSION_DEFAULTS"] = existing_conversion


            if app_settings.save_user_settings(settings_to_save):
                # Apply GPU setting immediately
                try:
                    from ..utils.gpu import set_gpu_enabled, is_gpu_available
                    if is_gpu_available():
                        set_gpu_enabled(self.gpu_enabled.isChecked())
                except Exception as e:
                    logger.warning("Could not apply GPU setting: %s", e)
                
                QMessageBox.information(
                    self,
                    "Settings Saved",
                    "Settings have been saved successfully.\nSome changes may require an application restart.",
                )
                super().accept()  # Close the dialog
            else:
                QMessageBox.warning(self, "Save Error", "Could not save settings to the file.")
                # Keep dialog open

        except Exception:
            logger.exception("Unexpected error while saving settings.")
            QMessageBox.critical(self, "Error Saving Settings", "An unexpected error occurred while saving settings.")
            # Keep dialog open


# Example usage (for testing)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dialog = SettingsDialog()
    if dialog.exec():
        logger.info("Settings dialog accepted (saved).")
    else:
        logger.info("Settings dialog cancelled.")
    sys.exit()