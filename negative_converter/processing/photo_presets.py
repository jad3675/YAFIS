# Photo preset management and application
import os
import json
import glob
import numpy as np
import cv2
import concurrent.futures
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import appdirs
except ImportError:
    appdirs = None  # Handle case where appdirs is not installed


def _validate_photo_preset(preset_data, source="<unknown>"):
    """
    Validate a photo preset payload.

    Required:
      - id: str
      - parameters: dict

    Returns:
      (ok: bool, preset: dict|None)
    """
    if not isinstance(preset_data, dict):
        logger.warning("Invalid photo preset from %s: expected dict, got %s", source, type(preset_data))
        return False, None

    preset_id = preset_data.get("id")
    if not isinstance(preset_id, str) or not preset_id.strip():
        logger.warning("Invalid photo preset from %s: missing/invalid 'id'", source)
        return False, None

    params = preset_data.get("parameters")
    if not isinstance(params, dict):
        logger.warning("Invalid photo preset '%s' from %s: missing/invalid 'parameters'", preset_id, source)
        return False, None

    # Optional fields
    name = preset_data.get("name")
    if name is not None and not isinstance(name, str):
        logger.warning("Photo preset '%s' from %s: 'name' should be a string; ignoring invalid value", preset_id, source)
        preset_data = dict(preset_data)
        preset_data.pop("name", None)

    return True, preset_data


class PhotoPresetManager:
    """Manages photo presets defined in JSON files."""

    def __init__(self, presets_file=None):
        if presets_file:
            self.presets_file = presets_file
        elif appdirs:
            app_name = "NegativeConverter"
            app_author = "NegativeConverter"
            data_dir = appdirs.user_data_dir(app_name, app_author)
            os.makedirs(data_dir, exist_ok=True)
            self.presets_file = os.path.join(data_dir, "photo_presets.json")
            logger.info("Using user preset file location: %s", self.presets_file)
        else:
            # Fallback if appdirs is not installed: use project root (original behavior)
            logger.warning("'appdirs' library not found. Falling back to project root for presets.")
            script_dir = os.path.dirname(__file__)
            self.presets_file = os.path.abspath(os.path.join(script_dir, "..", "..", "photo_presets.json"))

        self.presets = {}
        self.default_presets = {}  # Store defaults separately
        self.load_presets()

    def load_presets(self):
        """Load default presets from config/presets and user presets from AppData."""
        default_presets = {}
        user_presets = {}

        # 1. Load Default Presets
        script_dir = os.path.dirname(__file__)
        default_presets_dir = os.path.abspath(os.path.join(script_dir, "..", "config", "presets", "photo"))
        if os.path.isdir(default_presets_dir):
            json_files = glob.glob(os.path.join(default_presets_dir, "*.json"))
            for file_path in json_files:
                source = os.path.basename(file_path)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        preset_data = json.load(f)

                    ok, validated = _validate_photo_preset(preset_data, source=source)
                    if ok:
                        default_presets[validated["id"]] = validated
                except json.JSONDecodeError:
                    logger.exception("Error decoding JSON for default photo preset: %s", source)
                except Exception:
                    logger.exception("Error loading default photo preset: %s", source)

            logger.info("Loaded %s default photo presets from %s.", len(default_presets), default_presets_dir)
        else:
            logger.warning("Default photo presets directory not found: %s", default_presets_dir)

        # 2. Load User Presets
        if os.path.isfile(self.presets_file):
            try:
                with open(self.presets_file, "r", encoding="utf-8") as f:
                    user_preset_list = json.load(f)

                if isinstance(user_preset_list, list):
                    for idx, preset in enumerate(user_preset_list):
                        source = f"{self.presets_file}#{idx}"
                        ok, validated = _validate_photo_preset(preset, source=source)
                        if ok:
                            user_presets[validated["id"]] = validated
                    logger.info("Loaded %s user photo presets from %s.", len(user_presets), self.presets_file)
                else:
                    logger.warning("User presets file (%s) does not contain a JSON list.", self.presets_file)
            except json.JSONDecodeError:
                logger.exception("Error decoding JSON from user presets file %s", self.presets_file)
            except Exception:
                logger.exception("Error loading user presets from %s", self.presets_file)
        else:
            logger.info(
                "No user presets file found at %s. Only default presets will be available initially.",
                self.presets_file,
            )

        # 3. Store defaults and create merged list
        self.default_presets = default_presets
        self.presets = self.default_presets.copy()
        self.presets.update(user_presets)  # user presets overwrite defaults with same ID
        logger.info("Total photo presets available: %s", len(self.presets))

    def get_preset(self, preset_id):
        """Retrieve a specific preset by its ID."""
        return self.presets.get(preset_id)

    def get_all_presets(self):
        """Return a dictionary of all loaded presets."""
        return self.presets.copy()

    def _save_presets_to_file(self):
        """Save user-specific presets back to the JSON file."""
        try:
            user_presets_to_save = []
            for preset_id, preset_data in self.presets.items():
                if preset_id not in self.default_presets or preset_data != self.default_presets[preset_id]:
                    ok, validated = _validate_photo_preset(preset_data, source=f"in-memory:{preset_id}")
                    if ok:
                        user_presets_to_save.append(validated)
                    else:
                        logger.warning("Skipping invalid preset during save: %s", preset_id)

            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(user_presets_to_save, f, indent=2)

            logger.info("Saved %s user presets to %s", len(user_presets_to_save), self.presets_file)
            return True
        except Exception:
            logger.exception("Error saving presets to %s", self.presets_file)
            return False

    def add_preset(self, preset_data):
        """Adds or updates a preset and saves the changes to the file."""
        ok, validated = _validate_photo_preset(preset_data, source="add_preset")
        if not ok:
            logger.error("Invalid preset data provided to add_preset.")
            return False

        preset_id = validated["id"]
        self.presets[preset_id] = validated
        logger.info("Preset '%s' added/updated in manager.", preset_id)

        return self._save_presets_to_file()

    def apply_photo_preset(self, image, preset_id, intensity=1.0, preview_mode=False):
        """Applies a loaded photo preset to an image sequentially (no tiling)."""
        if image is None or image.size == 0:
            logger.warning("Cannot apply preset to empty image.")
            return image

        # For preview mode, process at lower resolution
        original_size = None
        if preview_mode:
            h, w = image.shape[:2]
            # Target ~1MP for preview (roughly 1000x1000)
            max_preview_pixels = 1_000_000
            current_pixels = h * w
            if current_pixels > max_preview_pixels:
                scale = np.sqrt(max_preview_pixels / current_pixels)
                new_h, new_w = int(h * scale), int(w * scale)
                original_size = (w, h)  # Store for upscaling later
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"Preview mode: downscaled from {w}x{h} to {new_w}x{new_h}")

        preset_data = self.get_preset(preset_id)
        if not preset_data:
            logger.warning("Photo preset '%s' not found.", preset_id)
            return image.copy()

        ok, validated = _validate_photo_preset(preset_data, source=f"manager:{preset_id}")
        if not ok:
            logger.warning("Photo preset '%s' is invalid. Skipping.", preset_id)
            return image.copy()

        params = validated["parameters"]

        logger.debug("Applying photo preset '%s' %s...", preset_id, "(preview)" if preview_mode else "sequentially")
        try:
            processed_image = self._apply_full_photo_preset(image, params)
            if processed_image is None:
                logger.error("_apply_full_photo_preset returned None for '%s'.", preset_id)
                if original_size:
                    return cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)
                return image.copy()
        except Exception:
            logger.exception("Error applying photo preset '%s'.", preset_id)
            if original_size:
                return cv2.resize(image, original_size, interpolation=cv2.INTER_LINEAR)
            return image.copy()

        # Blend with original based on intensity
        if 0.0 <= intensity < 0.99:
            original_image_float = image.astype(np.float32)
            processed_image_float = processed_image.astype(np.float32)
            blended = cv2.addWeighted(
                original_image_float, 1.0 - intensity, processed_image_float, intensity, 0
            )
            result = np.clip(blended, 0, 255).astype(np.uint8)
        else:
            result = processed_image
        
        # Upscale back to original size if preview mode
        if original_size:
            result = cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)
            logger.debug(f"Preview mode: upscaled back to {original_size[0]}x{original_size[1]}")
        
        return result

    def _apply_full_photo_preset(self, image, params):
        # Import moved here to break circular dependency
        from .adjustments import ImageAdjustments, AdvancedAdjustments
        """Internal method to apply all adjustments defined in a preset's parameters."""
        if image is None or image.size == 0: return image
        current_image = image.copy()
        preset_id_str = params.get('id', 'N/A') # For logging

        def check_image(step_name):
            if current_image is None:
                logger.error(
                    "PhotoPreset Error (%s): Image became None after step '%s'. Aborting preset application.",
                    preset_id_str,
                    step_name,
                )
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
            # apply_film_grain expects float in 0..1. Convert to float, apply grain, convert back to uint8.
            current_image_float01 = current_image.astype(np.float32) / 255.0
            processed_float01 = AdvancedAdjustments.apply_film_grain(
                current_image_float01,
                intensity=gp.get("intensity", 0),
                size=gp.get("size", 0.5),
                roughness=gp.get("roughness", 0.5),
            )
            if processed_float01 is None:
                logger.error("PhotoPreset Error (%s): Grain application returned None.", preset_id_str)
                return None
            current_image = (np.clip(processed_float01, 0.0, 1.0) * 255.0).astype(np.uint8)
            if not check_image("Grain"):
                return None
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
            except Exception:
                logger.exception("Could not display image.")
        else:
            print(f"Failed to apply preset '{preset_id_to_test}'.")
    else:
        print("\nNo presets loaded, cannot run example.")