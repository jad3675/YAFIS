# Film simulation implementation
import numpy as np
import cv2
import os
import glob
import json
import concurrent.futures
import math
import os # For cpu_count
# Use centralized GPU detection and logger
from negative_converter.utils.gpu import GPU_ENABLED, xp, cp_module # Import cp_module as well
from negative_converter.utils.logger import get_logger

logger = get_logger(__name__)
cp = cp_module # Assign cp_module to cp for local use if needed, otherwise use xp

# Import the centralized adjustments - MOVED inside _apply_full_preset to break circular import
# from .adjustments import AdvancedAdjustments

# Import the new utility function
import sys
import os
# Use centralized imaging utility
from negative_converter.utils.imaging import apply_curve

# --- Helper Functions (from Section 5.3) ---

def convert_to_linear_rgb(image):
    """Convert from sRGB to linear RGB (GPU/CPU)"""
    if image is None or image.size == 0: return image
    if GPU_ENABLED:
        xp = cp; img_arr = xp.asarray(image, dtype=xp.float32) / 255.0
    else:
        xp = np; img_arr = image.astype(xp.float32) / 255.0
    try:
        linear_arr = xp.power(img_arr, 2.2) * 255.0
        # Return CuPy array if GPU, NumPy if CPU
        return linear_arr
    except Exception as e:
        logger.error(f"Linear Conversion failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
        if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32) / 255.0; linear_arr = xp.power(img_arr, 2.2) * 255.0; return linear_arr # Fallback logic remains complex
        else: return image.astype(np.float32) # Return float CPU if CPU failed? Or original?

def convert_to_srgb(linear_image):
    """Convert from linear RGB float back to sRGB uint8 for display (GPU/CPU)"""
    if linear_image is None or linear_image.size == 0: return linear_image
    # Determine if input is CuPy or NumPy
    is_cupy_input = 'cupy' in str(type(linear_image))
    xp = cp if is_cupy_input and GPU_ENABLED else np
    linear_image_backend = linear_image if (is_cupy_input and GPU_ENABLED) else (cp.asnumpy(linear_image) if is_cupy_input else linear_image)

    try:
        # Ensure input is float32
        linear_float = xp.asarray(linear_image_backend, dtype=xp.float32) / 255.0
        srgb_float = xp.power(linear_float, 1.0/2.2)
        srgb_float_scaled = xp.clip(srgb_float * 255.0, 0, 255)
        # Convert final result to uint8 NumPy
        return (xp.asnumpy(srgb_float_scaled) if xp == cp else srgb_float_scaled).astype(np.uint8)
    except Exception as e:
        logger.error(f"sRGB Conversion to uint8 failed ({'GPU' if xp == cp else 'CPU'}): {e}. Trying CPU fallback.")
        # Fallback to CPU if GPU failed or if CPU failed initially (though less likely)
        xp_fallback = np
        linear_image_np = cp.asnumpy(linear_image_backend) if is_cupy_input else linear_image_backend
        try:
            linear_float_fb = linear_image_np.astype(xp_fallback.float32) / 255.0
            srgb_fb = xp_fallback.power(linear_float_fb, 1.0/2.2)
            return xp_fallback.clip(srgb_fb * 255.0, 0, 255).astype(xp_fallback.uint8)
        except Exception as e_cpu:
             logger.error(f"sRGB Conversion CPU fallback also failed: {e_cpu}. Returning None.")
             # Return None or raise error? Returning None might be safer downstream.
             return None # Indicate failure

def convert_to_srgb_float(linear_image_float):
    """Convert from linear RGB float back to sRGB float (GPU/CPU, returns float32 0-255)"""
    if linear_image_float is None or linear_image_float.size == 0: return linear_image_float
    # Determine if input is CuPy or NumPy
    is_cupy_input = 'cupy' in str(type(linear_image_float))
    xp = cp if is_cupy_input and GPU_ENABLED else np
    linear_image_backend = linear_image_float # Assume input is already correct backend type

    try:
        # Ensure input is float32
        linear_float_norm = xp.asarray(linear_image_backend, dtype=xp.float32) / 255.0
        srgb_float_norm = xp.power(linear_float_norm, 1.0/2.2)
        srgb_float_scaled = srgb_float_norm * 255.0
        # Return float result (CuPy or NumPy)
        return srgb_float_scaled
    except Exception as e:
        logger.error(f"Linear to sRGB Float Conversion failed ({'GPU' if xp == cp else 'CPU'}): {e}. Trying CPU fallback.")
        # Fallback to CPU if GPU failed or if CPU failed initially
        xp_fallback = np
        linear_image_np = cp.asnumpy(linear_image_backend) if is_cupy_input else linear_image_backend
        try:
            linear_float_fb_norm = linear_image_np.astype(xp_fallback.float32) / 255.0
            srgb_fb_norm = xp_fallback.power(linear_float_fb_norm, 1.0/2.2)
            return srgb_fb_norm * 255.0 # Return NumPy float
        except Exception as e_cpu:
             logger.error(f"Linear to sRGB Float CPU fallback also failed: {e_cpu}. Returning input.")
             return linear_image_float.copy() # Return original float input


def apply_color_matrix(rgb_image, matrix):
    """Apply color transformation matrix to RGB image (GPU/CPU, expects float 0-255)"""
    if rgb_image is None or rgb_image.size == 0: return rgb_image
    matrix_np = np.asarray(matrix, dtype=np.float32)
    is_cupy_input = 'cupy' in str(type(rgb_image))

    if GPU_ENABLED:
        xp = cp
        try:
            img_gpu = xp.asarray(rgb_image, dtype=xp.float32) # Ensure CuPy float
            matrix_gpu = xp.asarray(matrix_np)
            h, w, c = img_gpu.shape
            if c != 3: raise ValueError("Input image must be 3-channel RGB")
            pixels_gpu = img_gpu.reshape(-1, 3)
            transformed_gpu = xp.dot(pixels_gpu, matrix_gpu.T)
            return transformed_gpu.reshape(h, w, 3) # Return CuPy float
        except Exception as e:
            logger.error(f"Color Matrix failed (GPU): {e}. Falling back to CPU.")
            # Fallback needs NumPy input
            rgb_image_np = cp.asnumpy(rgb_image) if is_cupy_input else rgb_image
            xp = np # Switch to NumPy for fallback
            img_float = rgb_image_np.astype(xp.float32)
            h, w, c = img_float.shape
            if c != 3: raise ValueError("Input image must be 3-channel RGB")
            pixels = img_float.reshape(-1, 3)
            transformed = xp.dot(pixels, matrix_np.T) # Use NumPy matrix
            return transformed.reshape(h, w, 3) # Return NumPy float
    else: # CPU Path
        xp = np
        # Ensure input is NumPy array
        rgb_image_np = cp.asnumpy(rgb_image) if is_cupy_input else rgb_image
        img_float = rgb_image_np.astype(xp.float32)
        h, w, c = img_float.shape
        if c != 3: raise ValueError("Input image must be 3-channel RGB")
        pixels = img_float.reshape(-1, 3)
        transformed = xp.dot(pixels, matrix_np.T)
        return transformed.reshape(h, w, 3) # Return NumPy float


# Removed redundant helper functions:
# _apply_tone_curve_channel_cpu_lut
# _apply_tone_curve_channel_float_cpu
# _apply_tone_curve_channel_gpu
# Use utils.imaging.apply_curve instead.

# apply_film_grain function moved to processing/adjustments.py (AdvancedAdjustments class)

# Removed local apply_color_balance function (now in AdvancedAdjustments)


def apply_dynamic_range(image_float, compression, shadow_preservation, highlight_rolloff):
    """Apply dynamic range compression (GPU/CPU, expects float32 0-255, returns float32 0-255)"""
    if image_float is None or image_float.size == 0: return image_float
    if compression == 1.0 and shadow_preservation == 0.0 and highlight_rolloff == 0.0: return image_float.copy()

    # Determine backend based on input type
    is_cupy_input = 'cupy' in str(type(image_float))
    xp = cp if is_cupy_input else np

    # Need LAB conversion, which requires uint8 input for cv2.cvtColor
    # Convert float input to temporary uint8 on CPU for LAB conversion
    temp_image_uint8_np = np.clip(cp.asnumpy(image_float) if is_cupy_input else image_float, 0, 255).astype(np.uint8)
    lab_np = cv2.cvtColor(temp_image_uint8_np, cv2.COLOR_RGB2LAB).astype(np.float32) # Keep LAB as float

    # Perform calculations using the appropriate backend
    try:
        lab = xp.asarray(lab_np) # Transfer LAB to GPU if needed
        l_channel = lab[..., 0] / 255.0 # Normalize L channel (0-1)

        # Apply compression logic
        v_compressed = xp.power(l_channel, compression)
        shadow_mask = 1.0 - xp.power(l_channel, 0.5)
        v_compressed += shadow_mask * shadow_preservation * (l_channel - v_compressed)
        highlight_mask = xp.power(l_channel, 2.0)
        v_compressed += highlight_mask * highlight_rolloff * (l_channel - v_compressed)

        # Update L channel in LAB array (still float)
        lab[..., 0] = v_compressed * 255.0 # Scale back to 0-255 range

        # Convert LAB back to RGB (needs uint8 input for cv2.cvtColor)
        # Clip LAB result, convert to uint8 NumPy on CPU
        lab_result_uint8_np = np.clip(cp.asnumpy(lab) if is_cupy_input else lab, 0, 255).astype(np.uint8)
        rgb_result_uint8_np = cv2.cvtColor(lab_result_uint8_np, cv2.COLOR_LAB2RGB)

        # Convert final RGB result back to float (matching input type)
        return xp.asarray(rgb_result_uint8_np, dtype=xp.float32)

    except Exception as e:
        logger.error(f"Dynamic Range application failed ({'GPU' if is_cupy_input else 'CPU'}): {e}.")
        # Fallback: If GPU failed, try CPU if input was GPU
        if is_cupy_input:
            logger.info("Dynamic Range: Falling back to CPU.")
            xp_fallback = np
            try:
                 l_channel_fb = lab_np[..., 0] / 255.0
                 v_compressed_fb = xp_fallback.power(l_channel_fb, compression)
                 shadow_mask_fb = 1.0 - xp_fallback.power(l_channel_fb, 0.5)
                 v_compressed_fb += shadow_mask_fb * shadow_preservation * (l_channel_fb - v_compressed_fb)
                 highlight_mask_fb = xp_fallback.power(l_channel_fb, 2.0)
                 v_compressed_fb += highlight_mask_fb * highlight_rolloff * (l_channel_fb - v_compressed_fb)
                 lab_np[..., 0] = v_compressed_fb * 255.0
                 lab_result_uint8_np_fb = np.clip(lab_np, 0, 255).astype(np.uint8)
                 rgb_result_uint8_np_fb = cv2.cvtColor(lab_result_uint8_np_fb, cv2.COLOR_LAB2RGB)
                 return rgb_result_uint8_np_fb.astype(np.float32) # Return NumPy float
            except Exception as e_cpu:
                 logger.error(f"Dynamic Range CPU fallback also failed: {e_cpu}. Returning input.")
                 return image_float.copy() # Return original float input
        else:
            # If CPU failed initially
            logger.error("Dynamic Range CPU application failed. Returning input.")
            return image_float.copy() # Return original float input


# --- Film Type-Specific Implementations ---
# Note: These are examples and likely need adjustment based on actual preset data
# They also currently use the CPU tone curve function for simplicity after conversion

def apply_kodachrome_25(image):
    """Apply Kodachrome 25 film simulation (expects uint8 sRGB)"""
    # This is a placeholder - real application uses _apply_full_preset
    logger.warning("apply_kodachrome_25 is a placeholder, use apply_preset.")
    return image

def apply_velvia_50(image):
    """Apply Velvia 50 film simulation (expects uint8 sRGB)"""
    # This is a placeholder - real application uses _apply_full_preset
    logger.warning("apply_velvia_50 is a placeholder, use apply_preset.")
    return image

# --- Film Preset Manager ---

class FilmPresetManager:
    """Manages film simulation presets"""
    def __init__(self, preset_directory=None):
        if preset_directory is None:
            script_dir = os.path.dirname(__file__)
            self.preset_directory = os.path.abspath(os.path.join(script_dir,"..","config","presets", "film")) # Point to film subdirectory
        else:
            self.preset_directory = preset_directory
        self.presets = {}
        self.load_presets()

    def load_presets(self):
        """Load all presets from preset directory"""
        self.presets = {}
        if not os.path.isdir(self.preset_directory):
            logger.warning(f"Preset directory not found: {self.preset_directory}")
            return
        preset_files = glob.glob(os.path.join(self.preset_directory, "*.json"))
        for preset_file in preset_files:
            try:
                with open(preset_file, 'r') as f:
                    preset_data = json.load(f)
                    preset_id = preset_data.get("id", os.path.splitext(os.path.basename(preset_file))[0])
                    preset_data["id"] = preset_id
                    self.presets[preset_id] = preset_data
            except Exception as e:
                logger.error(f"Error loading preset {preset_file}: {e}")
        logger.info(f"Loaded {len(self.presets)} film presets.")


    def get_preset(self, preset_id):
        """Retrieve a specific preset by its ID."""
        return self.presets.get(preset_id)

    def get_all_presets(self):
        """Return a dictionary of all loaded presets."""
        return self.presets.copy()

    # Modified signature to accept grain_scale from UI
    def apply_preset(self, image, preset, intensity=1.0, grain_scale=None):
        """
        Applies a loaded film preset to an image sequentially (no tiling).

        Args:
            image (np.ndarray): Input image (expects uint8 RGB, already adjusted).
            preset (dict or str): Preset data dictionary or preset ID string.
            intensity (float): The intensity of the preset effect (0.0 to 1.0).
            grain_scale (float, optional): UI override for grain scale (e.g., 0.0 to 2.0). Defaults to None.


        Returns:
            np.ndarray: The processed image (uint8 RGB).
        """
        if image is None or image.size == 0: return image

        preset_data = None
        preset_id_str = "N/A" # For logging
        if isinstance(preset, str):
            preset_id_str = preset
            preset_data = self.get_preset(preset)
            if not preset_data:
                logger.warning(f"Film preset '{preset_id_str}' not found.")
                return image.copy()
        elif isinstance(preset, dict) and "parameters" in preset:
             preset_data = preset
             preset_id_str = preset_data.get("id", "N/A")
        else:
             logger.warning("Invalid preset format passed to apply_preset.")
             return image.copy()

        if "parameters" not in preset_data:
             logger.warning(f"Preset '{preset_id_str}' has no 'parameters' key.")
             return image.copy()

        # Make a deep copy to avoid modifying the original preset dict
        import copy
        params = copy.deepcopy(preset_data["parameters"])

        # Add preset ID to params for logging within _apply_full_preset if not already there
        if 'id' not in params:
             params['id'] = preset_id_str

        # Inject the UI grain_scale into the params if provided
        # This modifies the 'intensity' within grainParams based on the scale
        if grain_scale is not None:
             if "grainParams" not in params:
                 # If preset has no grain but slider is moved, add default grain structure
                 # Use a sensible default base intensity if adding grain structure
                 params["grainParams"] = {"intensity": 10.0, "size": 1.0, "roughness": 0.5}
                 logger.debug(f"  Adding default grainParams structure for preset {preset_id_str}")

             # Calculate new intensity based on scale. Assume preset intensity is the base.
             base_intensity = params["grainParams"].get("intensity", 10.0) # Default base if somehow missing after check
             params["grainParams"]["intensity"] = base_intensity * grain_scale
             logger.debug(f"  Injected grain_scale {grain_scale}, base intensity {base_intensity}, resulting intensity {params['grainParams']['intensity']}")
        # --- Apply preset sequentially to the whole image ---
        logger.info(f"Applying film preset '{preset_id_str}' sequentially...")
        try:
            processed_image = self._apply_full_preset(image, params)
            if processed_image is None:
                 logger.error(f"_apply_full_preset returned None for '{preset_id_str}'.")
                 return image.copy() # Return original on error
        except Exception as e:
             logger.exception(f"Error applying film preset '{preset_id_str}': {e}") # Use exception to log traceback
             # import traceback # No longer needed
             # traceback.print_exc() # No longer needed
             return image.copy() # Return original on error

        logger.info(f"Sequential processing finished for film preset '{preset_id_str}'.")

        # Blend with original based on intensity
        if 0.0 <= intensity < 0.99:
            original_image_float = image.astype(np.float32)
            processed_image_float = processed_image.astype(np.float32)
            blended = cv2.addWeighted(original_image_float, 1.0 - intensity,
                                      processed_image_float, intensity, 0)
            return np.clip(blended, 0, 255).astype(np.uint8)
        else:
            return processed_image # Return the fully processed uint8 image


    def _apply_full_preset(self, image, params):
        # Import moved here to break circular dependency
        from .adjustments import AdvancedAdjustments, ImageAdjustments
        """Internal method to apply all adjustments defined in a preset's parameters using a float pipeline."""
        if image is None or image.size == 0: return image
        preset_id_str = params.get('id', 'N/A')
        logger.debug(f"  _apply_full_preset Start (Float Pipeline): {preset_id_str}")

        # Determine backend
        xp = cp if GPU_ENABLED else np
        logger.debug(f"    Preset Backend: {'CuPy (GPU)' if GPU_ENABLED else 'NumPy (CPU)'}")

        # --- Processing Pipeline (float32) ---
        try:
            # 1. Convert to Linear RGB (float)
            logger.debug("    Preset Step 1: Converting to Linear Float...")
            working_float = convert_to_linear_rgb(image) # Returns float CuPy or NumPy
            if working_float is None: raise ValueError("Linear conversion failed")
            logger.debug("    Preset Step 1: Done.")

            # 2. Apply Color Matrix (in linear float space)
            if "colorMatrix" in params:
                logger.debug("    Preset Step 2: Applying Color Matrix...")
                working_float = apply_color_matrix(working_float, params["colorMatrix"])
                if working_float is None: raise ValueError("Color matrix application failed")
                logger.debug("    Preset Step 2: Done.")

            # 3. Apply Tone Curves (in linear float space)
            if "toneCurves" in params:
                logger.debug("    Preset Step 3: Applying Tone Curves...")
                curves = params["toneCurves"]
                # Use the imported apply_curve utility directly
                # It handles both NumPy and CuPy float32 inputs

                r_ch = working_float[..., 0]
                g_ch = working_float[..., 1]
                b_ch = working_float[..., 2]

                # Apply curves using the utility function
                # Use 'if curves.get("channel"):' to handle missing keys or empty lists gracefully
                if curves.get("r"): r_ch = apply_curve(r_ch, curves["r"])
                if curves.get("g"): g_ch = apply_curve(g_ch, curves["g"])
                if curves.get("b"): b_ch = apply_curve(b_ch, curves["b"])
                if curves.get("rgb"): # Apply RGB curve if individual channels not specified or missing
                    # Check again using .get() for safety, in case key exists but value is None/empty
                    if not curves.get("r"): r_ch = apply_curve(r_ch, curves["rgb"])
                    if not curves.get("g"): g_ch = apply_curve(g_ch, curves["rgb"])
                    if not curves.get("b"): b_ch = apply_curve(b_ch, curves["rgb"])

                working_float = xp.stack([r_ch, g_ch, b_ch], axis=-1)
                logger.debug("    Preset Step 3: Done.")

            # 4. Convert back to sRGB Float
            logger.debug("    Preset Step 4: Converting to sRGB Float...")
            working_float = convert_to_srgb_float(working_float)
            if working_float is None: raise ValueError("sRGB float conversion failed")
            logger.debug("    Preset Step 4: Done.")

            # 5. Apply Color Balance (in sRGB float space)
            if "colorBalance" in params:
                logger.debug("    Preset Step 5: Applying Color Balance...")
                cb = params["colorBalance"]
                red_shift=cb.get("redShift", 0)
                green_shift=cb.get("greenShift", 0)
                blue_shift=cb.get("blueShift", 0)
                red_balance=cb.get("redBalance", 1.0)
                green_balance=cb.get("greenBalance", 1.0)
                blue_balance=cb.get("blueBalance", 1.0)

                # Apply directly to float array
                if red_shift != 0 or green_shift != 0 or blue_shift != 0:
                    working_float += xp.array([red_shift, green_shift, blue_shift], dtype=xp.float32)
                if red_balance != 1.0 or green_balance != 1.0 or blue_balance != 1.0:
                    working_float *= xp.array([red_balance, green_balance, blue_balance], dtype=xp.float32)
                logger.debug("    Preset Step 5: Done.")

            # 6. Apply Dynamic Range Compression (in sRGB float space)
            if "dynamicRange" in params:
                logger.debug("    Preset Step 6: Applying Dynamic Range...")
                dr = params["dynamicRange"]
                working_float = apply_dynamic_range( # Now expects/returns float
                    working_float,
                    compression=dr.get("compression", 1.0),
                    shadow_preservation=dr.get("shadowPreservation", 0.0),
                    highlight_rolloff=dr.get("highlightRolloff", 0.0)
                )
                if working_float is None: raise ValueError("Dynamic range application failed")
                logger.debug("    Preset Step 6: Done.")

            # 7. Apply Film Grain (in sRGB float space) - Now uses AdvancedAdjustments
            if "grainParams" in params:
                logger.debug("    Preset Step 7: Applying Grain...")
                gp = params["grainParams"]
                # Normalize image to 0.0-1.0 for grain function
                working_float_norm = working_float / 255.0
                # Call the static method from AdvancedAdjustments
                grained_float_norm = AdvancedAdjustments.apply_film_grain(
                    working_float_norm,
                    intensity=gp.get("intensity", 0),
                    size=gp.get("size", 0.5),
                    roughness=gp.get("roughness", 0.5)
                )
                if grained_float_norm is None: raise ValueError("Grain application failed")
                # Scale back to 0-255 range
                working_float = grained_float_norm * 255.0
                logger.debug("    Preset Step 7: Done.")

            # 8. Final Clip and Convert to uint8 NumPy
            logger.debug("    Preset Step 8: Clipping and Converting to uint8...")
            final_float_clipped = xp.clip(working_float, 0, 255)
            final_uint8_np = (xp.asnumpy(final_float_clipped) if GPU_ENABLED else final_float_clipped).astype(np.uint8)
            logger.debug(f"  _apply_full_preset End: {preset_id_str}")
            return final_uint8_np

        except Exception as e:
            logger.exception(f"Error during preset '{preset_id_str}' application: {e}")
            # import traceback # No longer needed
            # traceback.print_exc() # No longer needed
            # Attempt to return original image on failure
            return image.copy()


    def save_preset(self, preset_id, preset_data):
        """Save preset data to a JSON file."""
        if not preset_id or not isinstance(preset_data, dict):
            logger.error("Invalid preset ID or data for saving.")
            return False
        # Ensure the preset has an ID matching the filename intention
        preset_data["id"] = preset_id
        file_path = os.path.join(self.preset_directory, f"{preset_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(preset_data, f, indent=2)
            logger.info(f"Preset saved successfully to {file_path}")
            # Reload presets to include the new one? Or just add it?
            self.presets[preset_id] = preset_data
            return True
        except Exception as e:
            logger.error(f"Error saving preset {preset_id}: {e}")
            return False
