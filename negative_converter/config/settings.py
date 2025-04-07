# Application settings
import numpy as np

# --- Conversion Parameters ---
CONVERSION_DEFAULTS = {
    # Default correction matrix (used if no profile-specific one is found)
    "correction_matrix": np.array([
        [1.6, -0.2, -0.1],
        [-0.1, 1.5, -0.1],
        [-0.1, -0.3, 1.4]
    ], dtype=np.float32),

    # Mask Detection & Classification Thresholds
    "mask_sample_size": 10, # Size of corner samples
    "mask_clear_sat_max": 40, # Max saturation for clear/neutral base
    "mask_c41_hue_min": 8,
    "mask_c41_hue_max": 22,
    "mask_c41_sat_min": 70,
    "mask_c41_val_min": 60,
    "mask_c41_val_max": 210,

    # White Balance Parameters
    "wb_target_gray": 128.0,
    "wb_clamp_min": 0.8,
    "wb_clamp_max": 1.3,

    # Channel Curve Parameters
    "curve_clip_percent": 0.5,
    "curve_gamma_red": 0.95,
    "curve_gamma_green": 1.0, # Default gamma
    "curve_gamma_blue": 1.1,
    "curve_num_intermediate_points": 5,

    # Final Color Grading Parameters (LAB/HSV)
    "lab_a_target": 128.0,
    "lab_a_correction_factor": 0.5,
    "lab_a_correction_max": 5.0,
    "lab_b_target": 128.0,
    "lab_b_correction_factor": 0.7,
    "lab_b_correction_max": 10.0,
    "hsv_saturation_boost": 1.15,
}

# --- GPU Settings ---
# (GPU detection itself is in utils.gpu, but related settings could go here)
# e.g., force_cpu = False

# --- UI Defaults ---
UI_DEFAULTS = {
    "default_jpeg_quality": 95,
    "default_png_compression": 6, # Typical default
    "filmstrip_thumb_size": 120,
}

# --- Logging ---
LOGGING_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR