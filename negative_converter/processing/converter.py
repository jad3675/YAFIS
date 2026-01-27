# Negative to positive conversion algorithm
"""
Main converter module for film negative to positive conversion.

This module orchestrates the conversion pipeline using:
- mask_detection.py for film base detection and classification
- processing_strategy.py for GPU/CPU processing abstraction
"""

import numpy as np
import cv2
from typing import Optional, Tuple, Callable

from ..utils.gpu import xp, is_cupy_backend
from ..utils.imaging import apply_curve
from ..config.settings import CONVERSION_DEFAULTS
from ..utils.logger import get_logger

from .mask_detection import FilmBaseDetector, detect_orange_mask
from .processing_strategy import (
    ProcessingContext,
    WhiteBalanceCalculator,
)

logger = get_logger(__name__)


# Re-export for backward compatibility
__all__ = ['NegativeConverter', 'detect_orange_mask', 'apply_color_correction']


def apply_color_correction(image: np.ndarray, correction_matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply color correction matrix to normalize colors.
    
    Args:
        image: Input uint8 RGB image.
        correction_matrix: 3x3 color correction matrix. Uses default if None.
        
    Returns:
        Color-corrected uint8 RGB image.
    """
    if image is None or image.size == 0:
        logger.warning("apply_color_correction received empty image.")
        return image
    if len(image.shape) != 3 or image.shape[2] != 3:
        logger.warning(f"apply_color_correction expects 3-channel image, got shape {image.shape}.")
        return image
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if correction_matrix is None:
        correction_matrix = np.array([
            [1.50, -0.20, -0.30],
            [-0.30, 1.60, -0.30],
            [-0.20, -0.20, 1.40]
        ], dtype=np.float32)
    else:
        correction_matrix = np.asarray(correction_matrix, dtype=np.float32)

    # Use processing context for GPU/CPU abstraction
    ctx = ProcessingContext()
    result, _ = ctx.process_core_pipeline(
        image,
        invert=False,
        wb_scales=None,
        color_matrix=correction_matrix
    )
    
    return np.clip(result, 0, 255).astype(np.uint8)


class NegativeConverter:
    """
    Converts film negatives to positive images.
    
    Supports multiple film types (C-41, ECN-2, E-6, B&W) with automatic
    detection or manual override. Uses GPU acceleration when available.
    """
    
    def __init__(self, film_profile: str = "C41"):
        """
        Initialize converter with a film profile.
        
        Args:
            film_profile: Profile ID ("C41", "BW", "E6", "ECN2").
        """
        self.film_profile = film_profile
        self.profile_data = self._load_film_profile(film_profile)
        self.params = self._build_params()
        
        # Initialize components
        self._detector = FilmBaseDetector(self.params)
        self._processing_ctx = ProcessingContext()
        self._wb_calculator = WhiteBalanceCalculator(self.params)
        
        logger.debug("NegativeConverter initialized with profile: %s", film_profile)

    def _build_params(self) -> dict:
        """Build parameters by merging defaults with profile data."""
        params = CONVERSION_DEFAULTS.copy()
        params.update({
            "correction_matrix": self.profile_data.get("correction_matrix", params["correction_matrix"]),
            "curve_gamma_red": self.profile_data["gamma"].get("red", params.get("curve_gamma_red")),
            "curve_gamma_green": self.profile_data["gamma"].get("green", params.get("curve_gamma_green")),
            "curve_gamma_blue": self.profile_data["gamma"].get("blue", params.get("curve_gamma_blue")),
            "hsv_saturation_boost": self.profile_data.get("saturation_boost", params.get("hsv_saturation_boost")),
            "lab_a_target": self.profile_data["lab_correction"].get("a_target", params.get("lab_a_target")),
            "lab_a_correction_factor": self.profile_data["lab_correction"].get("a_factor", params.get("lab_a_correction_factor")),
            "lab_b_target": self.profile_data["lab_correction"].get("b_target", params.get("lab_b_target")),
            "lab_b_correction_factor": self.profile_data["lab_correction"].get("b_factor", params.get("lab_b_correction_factor")),
        })
        return params

    def _load_film_profile(self, profile_id: str) -> dict:
        """Load film profile parameters from JSON file."""
        import json
        import os
        
        base_data = {
            "correction_matrix": CONVERSION_DEFAULTS["correction_matrix"],
            "gamma": {
                "red": CONVERSION_DEFAULTS.get("curve_gamma_red", 0.95),
                "green": CONVERSION_DEFAULTS.get("curve_gamma_green", 1.0),
                "blue": CONVERSION_DEFAULTS.get("curve_gamma_blue", 1.1)
            },
            "saturation_boost": CONVERSION_DEFAULTS.get("hsv_saturation_boost", 1.15),
            "lab_correction": {
                "a_target": CONVERSION_DEFAULTS.get("lab_a_target", 128.0),
                "a_factor": CONVERSION_DEFAULTS.get("lab_a_correction_factor", 0.5),
                "b_target": CONVERSION_DEFAULTS.get("lab_b_target", 128.0),
                "b_factor": CONVERSION_DEFAULTS.get("lab_b_correction_factor", 0.7)
            }
        }

        profile_map = {
            "C41": "c41_generic.json",
            "BW": "bw_generic.json",
            "E6": "e6_generic.json",
            "ECN2": "ecn2_generic.json"
        }
        
        profile_filename = profile_map.get(profile_id, f"{profile_id.lower()}.json")
        config_dir = os.path.dirname(os.path.dirname(__file__))
        profile_path = os.path.join(config_dir, "config", "film_profiles", profile_filename)
        
        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r') as f:
                    data = json.load(f)
                    logger.debug("Loaded film profile from %s", profile_path)
                    
                    if "correction_matrix" in data:
                        base_data["correction_matrix"] = np.array(data["correction_matrix"], dtype=np.float32)
                    if "gamma" in data:
                        base_data["gamma"].update(data["gamma"])
                    if "saturation_boost" in data:
                        base_data["saturation_boost"] = data["saturation_boost"]
                    if "lab_correction" in data:
                        base_data["lab_correction"].update(data["lab_correction"])
            except Exception as e:
                logger.error("Failed to load film profile %s: %s", profile_path, e)
                
        return base_data

    def convert(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        override_mask_classification: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Convert negative to positive.

        Args:
            image: Input negative image (NumPy array, uint8 RGB).
            progress_callback: Optional callback(current_step, total_steps).
            override_mask_classification: Override auto-detection with specific type.
                Valid values: "Likely C-41", "Likely ECN-2", "Likely E-6", 
                "Likely B&W", "Clear/Near Clear", "Unknown/Other".

        Returns:
            Tuple of (converted uint8 image, mask classification string).
            Returns (None, "Error") on failure.
        """
        total_steps = 6
        
        def report_progress(step: int):
            if progress_callback:
                try:
                    progress_callback(step, total_steps)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        report_progress(0)
        
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (RGB) image")

        try:
            # Step 0: Detect mask and classify
            mask_classification, mask_color = self._detect_and_classify(
                image, override_mask_classification
            )
            
            # Calculate white balance scales
            wb_scales = self._calculate_wb_scales(image, mask_classification, mask_color)
            
            # Steps 1-3: Core pipeline (invert, WB, color matrix)
            corrected_float = self._process_core_pipeline(
                image, wb_scales, report_progress
            )
            
            # Step 4: Channel curves
            curves_result = self._apply_channel_curves(corrected_float)
            report_progress(4)
            
            # Step 5: Final color grading
            final_image = self._apply_color_grading(curves_result)
            report_progress(5)
            
            logger.info(f"Conversion completed. Mask: {mask_classification}")
            report_progress(total_steps)
            
            return final_image, mask_classification
            
        except Exception as e:
            logger.exception(f"Error during conversion: {e}")
            return None, "Error"

    def _detect_and_classify(
        self,
        image: np.ndarray,
        override: Optional[str]
    ) -> Tuple[str, np.ndarray]:
        """Detect mask and classify film type."""
        if override:
            logger.debug(f"Using override classification: {override}")
            detection = self._detector.detect(image)
            return override, detection.color
        
        detection = self._detector.detect(image)
        logger.debug(f"Auto-detected: {detection.classification} (HSV: {detection.hsv})")
        return detection.classification, detection.color

    def _calculate_wb_scales(
        self,
        image: np.ndarray,
        classification: str,
        mask_color: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate white balance scales based on classification."""
        scales = self._wb_calculator.get_scales_for_classification(
            classification, mask_color, image
        )
        
        # Handle deferred gray world calculation
        if isinstance(scales, str):
            inverted = 255.0 - image.astype(np.float32)
            gentle = scales == "gray_world_gentle"
            scales = self._wb_calculator.calculate_gray_world(inverted, gentle)
            logger.debug(f"Gray World scales ({'gentle' if gentle else 'normal'}): {scales}")
        
        if scales is not None:
            return tuple(scales)
        return None

    def _process_core_pipeline(
        self,
        image: np.ndarray,
        wb_scales: Optional[Tuple[float, float, float]],
        report_progress: Callable[[int], None]
    ) -> np.ndarray:
        """Process core pipeline (invert, WB, color matrix)."""
        correction_matrix = self.params['correction_matrix']
        
        result, strategy_used = self._processing_ctx.process_core_pipeline(
            image,
            invert=True,
            wb_scales=wb_scales,
            color_matrix=correction_matrix
        )
        
        logger.debug(f"Core pipeline completed using {strategy_used}")
        report_progress(1)
        report_progress(2)
        report_progress(3)
        
        return result

    def _apply_channel_curves(self, image_float: np.ndarray) -> np.ndarray:
        """Apply per-channel curves for contrast and gamma."""
        result = image_float.copy()
        temp_u8 = np.clip(image_float, 0, 255).astype(np.uint8)
        
        gamma_values = [
            self.params['curve_gamma_red'],
            self.params['curve_gamma_green'],
            self.params['curve_gamma_blue']
        ]
        
        for c in range(3):
            channel_float = result[:, :, c]
            channel_u8 = temp_u8[:, :, c]
            
            # Calculate histogram and find black/white points
            hist, _ = np.histogram(channel_u8.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            total_pixels = cdf[-1]
            
            if total_pixels == 0:
                continue
            
            clip_percent = self.params['curve_clip_percent']
            clip_low = total_pixels * clip_percent / 100.0
            clip_high = total_pixels * (100.0 - clip_percent) / 100.0
            
            black_point = np.clip(np.searchsorted(cdf, clip_low, side='right'), 0, 254)
            white_point = np.clip(np.searchsorted(cdf, clip_high, side='left'), black_point + 1, 255)
            
            # Build curve points
            curve_points = self._build_curve_points(
                black_point, white_point, gamma_values[c]
            )
            
            result[:, :, c] = apply_curve(channel_float, curve_points)
        
        return result

    def _build_curve_points(
        self,
        black_point: int,
        white_point: int,
        gamma: float
    ) -> list:
        """Build curve points for a channel."""
        if white_point <= black_point:
            return [[0, 0], [black_point, 0], [white_point, 255], [255, 255]]
        
        num_points = self.params['curve_num_intermediate_points']
        in_points = np.linspace(black_point, white_point, num_points)
        normalized = (in_points - black_point) / (white_point - black_point)
        out_points = np.power(normalized, 1.0 / gamma) * 255.0
        
        curve_points = [[0, 0]]
        curve_points.extend([list(p) for p in zip(in_points, out_points)])
        curve_points.append([255, 255])
        
        return sorted(curve_points, key=lambda p: p[0])

    def _apply_color_grading(self, image_float: np.ndarray) -> np.ndarray:
        """Apply final LAB and HSV color grading."""
        # Convert to uint8 for cv2
        image_u8 = np.clip(image_float, 0, 255).astype(np.uint8)
        
        # LAB adjustments
        lab = cv2.cvtColor(image_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # A channel (green-magenta)
        a_avg = np.mean(lab[:, :, 1])
        a_target = self.params['lab_a_target']
        a_factor = self.params['lab_a_correction_factor']
        a_max = self.params['lab_a_correction_max']
        
        if a_avg > a_target:
            lab[:, :, 1] -= min((a_avg - a_target) * a_factor, a_max)
        elif a_avg < a_target:
            lab[:, :, 1] += min((a_target - a_avg) * a_factor, a_max)
        
        # B channel (blue-yellow)
        b_avg = np.mean(lab[:, :, 2])
        b_target = self.params['lab_b_target']
        b_factor = self.params['lab_b_correction_factor']
        b_max = self.params['lab_b_correction_max']
        
        if b_avg < b_target:
            lab[:, :, 2] += min((b_target - b_avg) * b_factor, b_max)
        
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        
        graded = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        # HSV saturation boost
        hsv = cv2.cvtColor(graded, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.params['hsv_saturation_boost']
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    def apply_tone_curve(self, image: np.ndarray, curve_points: list) -> np.ndarray:
        """Apply a tone curve to an image."""
        if image is None or image.size == 0 or not curve_points:
            return image if image is not None else np.array([])
        
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        result = image.copy()
        for c in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
            result[..., c] = apply_curve(result[..., c], curve_points)
        
        return result

    def auto_levels(self, image: np.ndarray, clip_percent: float = 1.0) -> np.ndarray:
        """Apply auto-levels with optional clipping."""
        if image is None or image.size == 0:
            return image
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image

        result = image.copy()
        
        for c in range(3):
            channel = image[:, :, c]
            hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            total = cdf[-1]
            
            if total == 0:
                continue
            
            clip_low = total * clip_percent / 100.0
            clip_high = total * (100.0 - clip_percent) / 100.0
            
            black = np.clip(np.searchsorted(cdf, clip_low, side='right'), 0, 255)
            white = np.clip(max(black, np.searchsorted(cdf, clip_high, side='left')), 0, 255)
            
            # Build LUT
            lut = np.arange(256, dtype=np.float32)
            if white > black:
                lut = (lut - black) * 255.0 / (white - black)
                lut = np.clip(lut, 0, 255)
            else:
                lut.fill(0 if black == 0 else 255)
            
            result[:, :, c] = cv2.LUT(channel, lut.astype(np.uint8))
        
        return result
