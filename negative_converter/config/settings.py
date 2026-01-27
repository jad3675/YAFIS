# Application settings
import logging
import numpy as np
import json
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)

# --- Configuration File ---
CONFIG_DIR = os.path.dirname(__file__)
USER_SETTINGS_PATH = os.path.join(CONFIG_DIR, "user_settings.json")

# --- Helper Function to Load Settings ---
def load_user_settings(path):
    """Loads settings from a JSON file, returning an empty dict if not found or invalid."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.exception("Could not load user settings from %s", path)
        return {}

# --- Load User Settings ---
user_settings = load_user_settings(USER_SETTINGS_PATH)

# --- Conversion Parameters (Defaults) ---
# These remain the primary source, user settings override specific keys
_CONVERSION_DEFAULTS_BASE = {
    # Default correction matrix (used if no profile-specific one is found)
    "correction_matrix": np.array([
        [1.6, -0.2, -0.1],
        [-0.1, 1.5, -0.1],
        [-0.1, -0.3, 1.4]
    ], dtype=np.float32),

    # Mask Detection & Classification Thresholds
    # These HSV values are used to classify film base types from corner samples.
    # OpenCV uses H: 0-180, S: 0-255, V: 0-255
    
    "mask_sample_size": 10,  # Size of corner samples in pixels
    "mask_clear_sat_max": 40,  # Max saturation for clear/neutral base
    
    # C-41 (Standard Color Negative) - Orange mask
    # The orange mask in C-41 film typically has:
    # - Hue: 8-22° (orange range in OpenCV's 0-180 scale, ~15-45° in 0-360)
    # - Saturation: High (≥70) due to strong orange color
    # - Value: Medium-high (60-210) - not too dark, not blown out
    "mask_c41_hue_min": 8,   # Lower bound of orange hue
    "mask_c41_hue_max": 22,  # Upper bound of orange hue
    "mask_c41_sat_min": 70,  # Minimum saturation (orange is quite saturated)
    "mask_c41_val_min": 60,  # Minimum brightness (not underexposed)
    "mask_c41_val_max": 210, # Maximum brightness (not overexposed)
    
    # ECN-2 (Motion Picture Negative) - Darker orange/brown mask
    # ECN-2 film has a denser, darker mask than consumer C-41:
    # - Hue: 5-25° (slightly wider range, can be more brown)
    # - Saturation: Medium-high (≥50) - less saturated than C-41
    # - Value: Low (30-80) - distinctly darker than C-41
    "mask_ecn2_hue_min": 5,   # Can be more red/brown
    "mask_ecn2_hue_max": 25,  # Can extend into yellow-orange
    "mask_ecn2_sat_min": 50,  # Lower saturation threshold
    "mask_ecn2_val_min": 30,  # Much darker than C-41
    "mask_ecn2_val_max": 80,  # Upper limit still quite dark
    
    # E-6 (Slide/Reversal Film) - Clear base, high value
    # Slide film has no orange mask, just clear film base:
    # - Saturation: Very low (≤25) - nearly colorless
    # - Value: Very high (≥200) - clear/transparent base appears bright
    "mask_e6_sat_max": 25,   # Nearly colorless
    "mask_e6_val_min": 200,  # Very bright (clear base)
    
    # B&W Negative - Clear or slightly tinted, low saturation
    # B&W film base is typically clear or has slight tint:
    # - Saturation: Very low (≤20) - essentially grayscale
    # - Value: Medium to high (100-255) - varies by film and exposure
    "mask_bw_sat_max": 20,   # Essentially no color
    "mask_bw_val_min": 100,  # Not too dark
    "mask_bw_val_max": 255,  # Can be very bright

    # White Balance Parameters
    # These control how white balance correction is applied based on film type.
    
    # Standard C-41 white balance
    "wb_target_gray": 128.0,  # Target neutral gray value after WB
    "wb_target_gray_ecn2": 140.0,  # ECN-2 needs higher target due to darker base
    "wb_clamp_min": 0.8,  # Minimum WB scale factor (prevents over-correction)
    "wb_clamp_max": 1.3,  # Maximum WB scale factor
    
    # ECN-2 allows wider WB range due to darker, more variable mask
    "wb_ecn2_clamp_min": 0.7,
    "wb_ecn2_clamp_max": 1.5,
    
    # E-6 slide film uses very gentle correction (already color-balanced)
    "wb_e6_clamp_min": 0.95,  # Very conservative - slide film is already balanced
    "wb_e6_clamp_max": 1.05,
    
    "gray_world_clamp_enabled": True,  # Allow disabling clamps for gray world WB

    # Mask Detection & Classification Robustness
    "variance_threshold": 25.0,  # For mask detection warning

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
    "hsv_saturation_boost": 1.15, # Example: User might override this
}

# --- GPU Settings ---
# (GPU detection itself is in utils.gpu, but related settings could go here)
# e.g., force_cpu = False

# --- UI Defaults (Base) ---
_UI_DEFAULTS_BASE = {
    "default_jpeg_quality": 95,
    "default_png_compression": 6, # Typical default
    "filmstrip_thumb_size": 120, # Example: User might override this
    "apply_embedded_icc_profile": False,
    "gpu_acceleration_enabled": True,  # GPU enabled by default
}

# --- Logging (Base) ---
_LOGGING_LEVEL_BASE = "INFO" # Options: DEBUG, INFO, WARNING, ERROR

# --- Apply User Overrides ---
# We merge the base defaults with user settings. User settings take precedence.
# Note: Deep merging is not handled here for simplicity. If nested dicts need merging,
# a more complex merge function would be required. The correction_matrix is handled separately.

CONVERSION_DEFAULTS = _CONVERSION_DEFAULTS_BASE.copy()
CONVERSION_DEFAULTS.update(user_settings.get("CONVERSION_DEFAULTS", {}))

# Allow users to override `correction_matrix` via JSON as a list-of-lists.
# If provided but invalid, fall back to base default.
_user_matrix = user_settings.get("CONVERSION_DEFAULTS", {}).get("correction_matrix")
if _user_matrix is not None:
    try:
        CONVERSION_DEFAULTS["correction_matrix"] = np.asarray(_user_matrix, dtype=np.float32)
    except Exception:
        logger.exception("Invalid user correction_matrix; falling back to base default")
        CONVERSION_DEFAULTS["correction_matrix"] = _CONVERSION_DEFAULTS_BASE["correction_matrix"]
else:
    CONVERSION_DEFAULTS["correction_matrix"] = _CONVERSION_DEFAULTS_BASE["correction_matrix"]


UI_DEFAULTS = _UI_DEFAULTS_BASE.copy()
UI_DEFAULTS.update(user_settings.get("UI_DEFAULTS", {}))

LOGGING_LEVEL = user_settings.get("LOGGING_LEVEL", _LOGGING_LEVEL_BASE)

# --- Logger Setup ---
# The logger should be configured elsewhere (e.g., in main.py)
# after settings are loaded, using the LOGGING_LEVEL defined here.

# --- Functions to Save and Reload Settings ---

# Store base defaults separately to allow resetting/reloading
_BASE_CONVERSION_DEFAULTS = _CONVERSION_DEFAULTS_BASE.copy()
_BASE_UI_DEFAULTS = _UI_DEFAULTS_BASE.copy()
_BASE_LOGGING_LEVEL = _LOGGING_LEVEL_BASE

def reload_settings():
    """Reload settings from disk and update in-memory variables.

    Note: CONVERSION_DEFAULTS *are* reloaded here, but callers must ensure they
    rebuild any long-lived objects that captured old values.
    """
    global user_settings, UI_DEFAULTS, LOGGING_LEVEL, CONVERSION_DEFAULTS

    logger.info("Reloading user settings from %s", USER_SETTINGS_PATH)
    user_settings = load_user_settings(USER_SETTINGS_PATH)

    # Reload conversion defaults
    new_conv = _BASE_CONVERSION_DEFAULTS.copy()
    new_conv.update(user_settings.get("CONVERSION_DEFAULTS", {}))

    user_matrix = user_settings.get("CONVERSION_DEFAULTS", {}).get("correction_matrix")
    if user_matrix is not None:
        try:
            new_conv["correction_matrix"] = np.asarray(user_matrix, dtype=np.float32)
        except Exception:
            logger.exception("Invalid user correction_matrix; falling back to base default")
            new_conv["correction_matrix"] = _BASE_CONVERSION_DEFAULTS["correction_matrix"]
    else:
        new_conv["correction_matrix"] = _BASE_CONVERSION_DEFAULTS["correction_matrix"]

    CONVERSION_DEFAULTS.clear()
    CONVERSION_DEFAULTS.update(new_conv)

    # Reload UI Defaults
    new_ui_defaults = _BASE_UI_DEFAULTS.copy()
    new_ui_defaults.update(user_settings.get("UI_DEFAULTS", {}))
    UI_DEFAULTS.clear()
    UI_DEFAULTS.update(new_ui_defaults)

    # Reload logging level
    LOGGING_LEVEL = user_settings.get("LOGGING_LEVEL", _BASE_LOGGING_LEVEL)

    # Apply new logging level immediately
    try:
        from ..utils.logger import set_log_level

        set_log_level(LOGGING_LEVEL)
    except Exception:
        logger.exception("Failed to apply updated log level")

    logger.info("Settings reloaded. Logging level=%s", LOGGING_LEVEL)

def save_user_settings(settings_dict):
    """Saves the provided dictionary to the user settings JSON file."""
    save_data = {}
    # Only save sections that were actually provided by the settings dialog
    if "CONVERSION_DEFAULTS" in settings_dict:
        # Normalize numpy arrays into JSON-safe lists.
        normalized = {}
        for k, v in settings_dict["CONVERSION_DEFAULTS"].items():
            if isinstance(v, np.ndarray):
                normalized[k] = v.tolist()
            else:
                normalized[k] = v
        save_data["CONVERSION_DEFAULTS"] = normalized
    if "UI_DEFAULTS" in settings_dict:
        save_data["UI_DEFAULTS"] = settings_dict["UI_DEFAULTS"]
    if "LOGGING_LEVEL" in settings_dict:
        save_data["LOGGING_LEVEL"] = settings_dict["LOGGING_LEVEL"]

    try:
        with open(USER_SETTINGS_PATH, 'w') as f:
            json.dump(save_data, f, indent=4)
        logger.info("User settings saved to %s", USER_SETTINGS_PATH)
        return True
    except IOError:
        logger.exception("Could not save user settings to %s", USER_SETTINGS_PATH)
        return False