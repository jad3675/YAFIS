# Photo preset management and application
import os
import json
import glob
import numpy as np
import cv2
import concurrent.futures
import math
# import os # Already imported below
try:
    import appdirs
except ImportError:
    appdirs = None # Handle case where appdirs is not installed

# Import necessary adjustment functions
from .adjustments import ImageAdjustments, AdvancedAdjustments
# Import specific functions needed from film_simulation (or utils if refactored)
# from .film_simulation import apply_film_grain # Removed apply_color_balance, now in AdvancedAdjustments - Grain moved to AdvancedAdjustments

class PhotoPresetManager:
    """Manages photo presets defined in JSON files."""
    def __init__(self, presets_file=None):
        if presets_file:
            # Allow overriding for testing or specific cases
            self.presets_file = presets_file
        elif appdirs:
            # Use appdirs to find the standard user data directory
            app_name = "NegativeConverter"
            app_author = "NegativeConverter" # Can be same as app_name or your name/org
            data_dir = appdirs.user_data_dir(app_name, app_author)
            # Ensure the directory exists
            os.makedirs(data_dir, exist_ok=True)
            self.presets_file = os.path.join(data_dir, "photo_presets.json")
            print(f"Using user preset file location: {self.presets_file}")
        else:
            # Fallback if appdirs is not installed: use project root (original behavior)
            print("Warning: 'appdirs' library not found. Falling back to project root for presets.")
            script_dir = os.path.dirname(__file__)
            self.presets_file = os.path.abspath(os.path.join(
                script_dir, "..", "..", "photo_presets.json"
            ))

        self.presets = {}
        self.default_presets = {} # Store defaults separately
        self.load_presets()

    def load_presets(self):
        """Load default presets from config/presets and user presets from AppData."""
        default_presets = {}
        user_presets = {}

        # 1. Load Default Presets (Assuming they are in config/presets like film sims)
        #    We might need a way to distinguish them, e.g., a "type": "photo" key.
        script_dir = os.path.dirname(__file__)
        default_presets_dir = os.path.abspath(os.path.join(script_dir, "..", "config", "presets", "photo")) # Point to photo subdirectory
        if os.path.isdir(default_presets_dir):
            json_files = glob.glob(os.path.join(default_presets_dir, "*.json"))
            for file_path in json_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        preset_data = json.load(f)
                        # Load any JSON file in this directory as a default photo preset
                        # No need to check "type" key anymore as they are in the dedicated folder
                        if "id" in preset_data:
                             default_presets[preset_data["id"]] = preset_data
                except Exception as e:
                    print(f"Error loading default preset from {os.path.basename(file_path)}: {e}")
            print(f"Loaded {len(default_presets)} default photo presets from {default_presets_dir}.")
        else:
             print(f"Warning: Default presets directory not found: {default_presets_dir}")


        # 2. Load User Presets (from the location determined by __init__)
        if os.path.isfile(self.presets_file):
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    user_preset_list = json.load(f)
                    # Ensure it's a list and presets have IDs
                    if isinstance(user_preset_list, list):
                        user_presets = {preset["id"]: preset for preset in user_preset_list if isinstance(preset, dict) and "id" in preset}
                        print(f"Loaded {len(user_presets)} user photo presets from {self.presets_file}.")
                    else:
                        print(f"Warning: User presets file ({self.presets_file}) does not contain a valid JSON list.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from user presets file {self.presets_file}: {e}")
            except Exception as e:
                print(f"Error loading user presets from {self.presets_file}: {e}")
        else:
             print(f"No user presets file found at {self.presets_file}. Only default presets will be available initially.")

        # 3. Store defaults and create merged list
        self.default_presets = default_presets # Keep defaults separate
        self.presets = self.default_presets.copy() # Start with defaults
        self.presets.update(user_presets) # Update with user presets (overwriting defaults with same ID)
        print(f"Total photo presets available: {len(self.presets)}")

    def get_preset(self, preset_id):
        """Retrieve a specific preset by its ID."""
        return self.presets.get(preset_id)

    def get_all_presets(self):
        """Return a dictionary of all loaded presets."""
        return self.presets.copy()

    def _save_presets_to_file(self):
        """Save the current state of self.presets back to the JSON file."""
        try:
            # Identify user-specific presets (new or modified compared to defaults)
            user_presets_to_save = []
            for preset_id, preset_data in self.presets.items():
                # Check if it's a new preset (not in defaults) or if it's different from the default
                if preset_id not in self.default_presets or preset_data != self.default_presets[preset_id]:
                    user_presets_to_save.append(preset_data)

            # Save only the user-specific presets to the user file
            preset_list = user_presets_to_save
            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(preset_list, f, indent=2) # Use indent for readability
            print(f"Successfully saved {len(preset_list)} presets to {self.presets_file}")
            return True
        except Exception as e:
            print(f"Error saving presets to {self.presets_file}: {e}")
            return False

    def add_preset(self, preset_data):
        """Adds or updates a preset and saves the changes to the file."""
        if not isinstance(preset_data, dict) or "id" not in preset_data:
            print("Error: Invalid preset data provided to add_preset.")
            return False

        preset_id = preset_data["id"]
        self.presets[preset_id] = preset_data # Add or overwrite
        print(f"Preset '{preset_id}' added/updated in manager.")

        # Save the updated presets list back to the file
        return self._save_presets_to_file()

    def apply_photo_preset(self, image, preset_id, intensity=1.0):
        """Applies a loaded photo preset to an image sequentially (no tiling)."""
        if image is None or image.size == 0:
            print("Warning: Cannot apply preset to empty image.")
            return image

        preset_data = self.get_preset(preset_id)
        if not preset_data:
            print(f"Warning: Photo preset '{preset_id}' not found.")
            return image.copy()

        if "parameters" not in preset_data:
            print(f"Warning: Preset '{preset_id}' has no 'parameters' key.")
            return image.copy()

        params = preset_data["parameters"]

        # --- Apply preset sequentially to the whole image ---
        print(f"PhotoPreset Apply: Applying preset '{preset_id}' sequentially...")
        try:
            processed_image = self._apply_full_photo_preset(image, params)
            if processed_image is None:
                 print(f"PhotoPreset Error: _apply_full_photo_preset returned None for '{preset_id}'.")
                 return image.copy() # Return original on error
        except Exception as e:
             print(f"PhotoPreset Error applying preset '{preset_id}': {e}")
             import traceback
             traceback.print_exc()
             return image.copy() # Return original on error

        print(f"PhotoPreset Apply: Sequential processing finished for preset '{preset_id}'.")

        # Blend with original based on intensity
        if 0.0 <= intensity < 0.99:
            original_image_float = image.astype(np.float32)
            processed_image_float = processed_image.astype(np.float32)
            blended = cv2.addWeighted(original_image_float, 1.0 - intensity,
                                      processed_image_float, intensity, 0)
            return np.clip(blended, 0, 255).astype(np.uint8)
        else:
            return processed_image # Return the fully processed uint8 image

    def _apply_full_photo_preset(self, image, params):
        """Internal method to apply all adjustments defined in a preset's parameters."""
        if image is None or image.size == 0: return image
        current_image = image.copy()
        preset_id_str = params.get('id', 'N/A') # For logging

        def check_image(step_name):
            if current_image is None:
                print(f"PhotoPreset Error ({preset_id_str}): Image became None after step '{step_name}'. Aborting preset application.")
                return False
            return True

        # --- Apply adjustments in a logical order ---
        # Order based on common image editing workflows

        # 1. Basic Tonal Adjustments (Exposure, Contrast, Highlights, Shadows)
        if "brightness" in params:
            current_image = ImageAdjustments.adjust_brightness(current_image, params["brightness"])
            if not check_image("Brightness"): return None
        if "contrast" in params:
            current_image = ImageAdjustments.adjust_contrast(current_image, params["contrast"])
            if not check_image("Contrast"): return None
        if "shadows" in params or "highlights" in params:
            shadows_val = params.get("shadows", 0)
            highlights_val = params.get("highlights", 0)
            current_image = AdvancedAdjustments.adjust_shadows_highlights(current_image, shadows_val, highlights_val)
            if not check_image("Shadows/Highlights"): return None

        # 2. Color Adjustments (Temp, Tint, Vibrance, Saturation)
        if "temperature" in params or "tint" in params:
            temp_val = params.get("temperature", 0)
            tint_val = params.get("tint", 0)
            current_image = ImageAdjustments.adjust_temp_tint(current_image, temp_val, tint_val)
            if not check_image("Temp/Tint"): return None
        if "vibrance" in params: # NEW
            current_image = AdvancedAdjustments.adjust_vibrance(current_image, params["vibrance"])
            if not check_image("Vibrance"): return None
        if "saturation" in params:
            # Apply saturation *after* vibrance if both exist
            current_image = ImageAdjustments.adjust_saturation(current_image, params["saturation"])
            if not check_image("Saturation"): return None

        # 3. Advanced Color Adjustments (Color Balance, Color Grading)
        if "colorBalance" in params:
             cb = params["colorBalance"]
             current_image = AdvancedAdjustments.apply_color_balance(current_image, # Use AdvancedAdjustments
                                                 red_shift=0, green_shift=0, blue_shift=0, # Photo presets only use balance part
                                                 red_balance=cb.get("redBalance", 1.0),
                                                 green_balance=cb.get("greenBalance", 1.0),
                                                 blue_balance=cb.get("blueBalance", 1.0))
             if not check_image("Color Balance"): return None
        if "colorGrading" in params: # NEW
            cg = params["colorGrading"]
            current_image = AdvancedAdjustments.apply_color_grading(
                current_image,
                shadows_rgb=cg.get("shadows", [0.0, 0.0, 0.0]),
                midtones_rgb=cg.get("midtones", [0.0, 0.0, 0.0]),
                highlights_rgb=cg.get("highlights", [0.0, 0.0, 0.0])
            )
            if not check_image("Color Grading"): return None

        # 4. Tone Curve
        if "toneCurve" in params and "rgb" in params["toneCurve"]:
             rgb_curve = params["toneCurve"]["rgb"]
             # Apply the same curve to all channels using AdvancedAdjustments.apply_curves
             current_image = AdvancedAdjustments.apply_curves(current_image, rgb_curve, rgb_curve, rgb_curve)
             if not check_image("Tone Curve"): return None

        # 5. Effects (Clarity, Grain, Vignette)
        if "clarity" in params: # NEW
            current_image = AdvancedAdjustments.adjust_clarity(current_image, params["clarity"])
            if not check_image("Clarity"): return None
        if "grainParams" in params:
            gp = params["grainParams"]
            # Call the static method from AdvancedAdjustments
            # Note: apply_film_grain expects float32 input, but photo preset pipeline works on uint8.
            # We need to convert to float, apply grain, then convert back.
            current_image_float = current_image.astype(np.float32) # Error occurred here if current_image was None
            processed_float = AdvancedAdjustments.apply_film_grain(current_image_float,
                                             intensity=gp.get("intensity", 0),
                                             size=gp.get("size", 0.5),
                                             roughness=gp.get("roughness", 0.5))
            if processed_float is None: # Check if grain application itself failed
                 print(f"PhotoPreset Error ({preset_id_str}): Grain application returned None.")
                 return None
            # Convert back to uint8 after applying grain
            current_image = np.clip(processed_float, 0, 255).astype(np.uint8)
            if not check_image("Grain"): return None # Should not be None here, but check anyway
        if "specialEffects" in params and "vignette" in params["specialEffects"]: # NEW
            vig = params["specialEffects"]["vignette"]
            current_image = AdvancedAdjustments.apply_vignette(
                current_image,
                amount=vig.get("amount", 0) * 100, # Scale 0-1 amount to -100..100 expected by function
                center_x=vig.get("centerX", 0.5),
                center_y=vig.get("centerY", 0.5),
                radius=vig.get("radius", 0.7),
                feather=vig.get("feather", 0.3),
                color=vig.get("color") # Pass color if present
            )
            if not check_image("Vignette"): return None

        # 6. Monochrome Conversion (Apply last if needed)
        # Check if saturation is -100, indicating B&W intent
        is_bw = params.get("saturation", 0) <= -100
        if is_bw:
            if "bwMix" in params: # NEW
                bw = params["bwMix"]
                current_image = AdvancedAdjustments.apply_bw_mix(
                    current_image,
                    red_weight=bw.get("red", 30), # Default weights if missing
                    green_weight=bw.get("green", 59),
                    blue_weight=bw.get("blue", 11)
                    # Note: Yellow, Cyan, Magenta weights are ignored by this implementation
                )
                if not check_image("B&W Mix"): return None
            elif len(current_image.shape) == 3: # Apply standard grayscale if saturation is -100 but no bwMix
                 gray = cv2.cvtColor(current_image, cv2.COLOR_RGB2GRAY)
                 current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                 if not check_image("Grayscale Conversion"): return None


        # Ensure final image is uint8
        return np.clip(current_image, 0, 255).astype(np.uint8)

# Example Usage (for testing)
if __name__ == '__main__':
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[:, :, 0] = np.tile(np.linspace(0, 255, 100), (100, 1))
    dummy_image[:, :, 1] = np.tile(np.linspace(0, 255, 100).reshape(-1, 1), (1, 100))

    preset_manager = PhotoPresetManager() # Assumes photo_presets.json exists

    if preset_manager.presets:
        print("\nAvailable Photo Presets:")
        for pid, pdata in preset_manager.get_all_presets().items():
            print(f"- {pid}: {pdata.get('name', 'N/A')}")

        # Test a preset that uses some of the new params
        # preset_id_to_test = 'radiate' # Has clarity
        # preset_id_to_test = 'bw' # Has bwMix (implicitly via saturation)
        preset_id_to_test = 'film' # Has grain and color grading
        # preset_id_to_test = 'burn' # Has vignette

        print(f"\nApplying preset: {preset_id_to_test}")
        processed = preset_manager.apply_photo_preset(dummy_image, preset_id_to_test, intensity=1.0)

        if processed is not None:
            print(f"Successfully applied preset '{preset_id_to_test}'. Output shape: {processed.shape}")
            # Display side-by-side if possible
            try:
                combined = np.hstack((dummy_image, processed))
                cv2.imshow(f"Original vs {preset_id_to_test}", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as display_e:
                print(f"Could not display image: {display_e}")
        else:
            print(f"Failed to apply preset '{preset_id_to_test}'.")
    else:
        print("\nNo presets loaded, cannot run example.")