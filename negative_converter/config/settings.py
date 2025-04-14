# Application settings
import numpy as np
import json
import os
# Removed: from utils.logger import setup_logger

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
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load user settings from {path}. Error: {e}")
        # Consider logging this properly
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
}

# --- Logging (Base) ---
_LOGGING_LEVEL_BASE = "INFO" # Options: DEBUG, INFO, WARNING, ERROR

# --- Apply User Overrides ---
# We merge the base defaults with user settings. User settings take precedence.
# Note: Deep merging is not handled here for simplicity. If nested dicts need merging,
# a more complex merge function would be required. The correction_matrix is handled separately.

CONVERSION_DEFAULTS = _CONVERSION_DEFAULTS_BASE.copy()
CONVERSION_DEFAULTS.update(user_settings.get("CONVERSION_DEFAULTS", {}))
# Ensure the matrix remains a numpy array from the base defaults
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
    """Reloads settings from user_settings.json and updates in-memory variables."""
    global user_settings, UI_DEFAULTS, LOGGING_LEVEL
    # CONVERSION_DEFAULTS are intentionally NOT reloaded dynamically here
    # as the core converter likely uses them at initialization.

    print("Reloading user settings...") # Log this
    user_settings = load_user_settings(USER_SETTINGS_PATH)

    # Reload UI Defaults
    new_ui_defaults = _BASE_UI_DEFAULTS.copy()
    new_ui_defaults.update(user_settings.get("UI_DEFAULTS", {}))
    # Update the global dictionary directly
    UI_DEFAULTS.clear()
    UI_DEFAULTS.update(new_ui_defaults)

    # Reload Logging Level (Note: This won't reconfigure the logger itself)
    LOGGING_LEVEL = user_settings.get("LOGGING_LEVEL", _BASE_LOGGING_LEVEL)
    print(f"Settings reloaded. Current UI Defaults: {UI_DEFAULTS}, Logging Level: {LOGGING_LEVEL}") # Log

def save_user_settings(settings_dict):
    """Saves the provided dictionary to the user settings JSON file."""
    save_data = {}
    # Only save sections that were actually provided by the settings dialog
    if "CONVERSION_DEFAULTS" in settings_dict:
        save_data["CONVERSION_DEFAULTS"] = {
            k: v for k, v in settings_dict["CONVERSION_DEFAULTS"].items()
            if not isinstance(v, np.ndarray) # Exclude numpy arrays from JSON
        }
    if "UI_DEFAULTS" in settings_dict:
        save_data["UI_DEFAULTS"] = settings_dict["UI_DEFAULTS"]
    if "LOGGING_LEVEL" in settings_dict:
        save_data["LOGGING_LEVEL"] = settings_dict["LOGGING_LEVEL"]

    try:
        with open(USER_SETTINGS_PATH, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"User settings saved to {USER_SETTINGS_PATH}") # Log this
        return True
    except IOError as e:
        print(f"Error: Could not save user settings to {USER_SETTINGS_PATH}. Error: {e}") # Log this
        return False