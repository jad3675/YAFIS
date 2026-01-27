"""
Film base mask detection and classification.

This module handles detection of film base colors from scanned negatives
and classifies them into known film types (C-41, ECN-2, E-6, B&W, etc.)
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..config.settings import CONVERSION_DEFAULTS
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MaskDetectionResult:
    """Result of film base mask detection."""
    color: np.ndarray  # RGB color of detected mask
    classification: str  # Film type classification
    hsv: Tuple[int, int, int]  # HSV values of mask
    confidence: str  # "high", "medium", "low"


class FilmBaseDetector:
    """
    Detects and classifies film base masks from scanned negatives.
    
    Samples corners and edge midpoints of the image to determine
    the film base color, then classifies it based on HSV thresholds.
    """
    
    # Classification constants
    CLASSIFICATION_C41 = "Likely C-41"
    CLASSIFICATION_ECN2 = "Likely ECN-2"
    CLASSIFICATION_E6 = "Likely E-6"
    CLASSIFICATION_BW = "Likely B&W"
    CLASSIFICATION_CLEAR = "Clear/Near Clear"
    CLASSIFICATION_UNKNOWN = "Unknown/Other"
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize detector with optional custom parameters.
        
        Args:
            params: Dictionary of detection parameters. If None, uses defaults.
        """
        self.params = params if params is not None else CONVERSION_DEFAULTS.copy()
    
    def detect(self, image: np.ndarray) -> MaskDetectionResult:
        """
        Detect film base mask color and classify the film type.
        
        Args:
            image: Input image as NumPy array (uint8 or float, RGB).
            
        Returns:
            MaskDetectionResult with color, classification, and metadata.
        """
        mask_color = self._sample_base_color(image)
        hsv = self._rgb_to_hsv(mask_color)
        classification = self._classify_mask(hsv)
        confidence = self._assess_confidence(image, mask_color)
        
        return MaskDetectionResult(
            color=mask_color,
            classification=classification,
            hsv=hsv,
            confidence=confidence
        )
    
    def _sample_base_color(self, image: np.ndarray) -> np.ndarray:
        """
        Sample the film base color from corners and edge midpoints.
        
        Args:
            image: Input image as NumPy array.
            
        Returns:
            Average RGB color of sampled areas as float32 array.
        """
        if image is None or image.size == 0:
            logger.warning("Received empty image for mask detection.")
            return np.array([0, 0, 0], dtype=np.float32)
        
        if len(image.shape) < 2 or image.shape[0] < 10 or image.shape[1] < 10:
            logger.warning(f"Image too small for corner sampling ({image.shape}).")
            return np.array([0, 0, 0], dtype=np.float32)
        
        h, w = image.shape[:2]
        s = self.params.get('mask_sample_size', 10)
        s_half = s // 2
        
        # Adjust sample size for small images
        if h < s or w < s:
            logger.warning(f"Image dimensions ({h}x{w}) too small for sample size {s}.")
            s = min(h, w)
            s_half = s // 2
            if s == 0:
                logger.error("Image too small for any sampling.")
                return np.array([0, 0, 0], dtype=np.float32)
        
        # Define sample areas (corners and edge midpoints)
        sample_coords = [
            (0, s, 0, s),                                      # Top-left
            (0, s, w - s, w),                                  # Top-right
            (h - s, h, 0, s),                                  # Bottom-left
            (h - s, h, w - s, w),                              # Bottom-right
            (0, s, w // 2 - s_half, w // 2 + s_half),          # Top-mid
            (h - s, h, w // 2 - s_half, w // 2 + s_half),      # Bottom-mid
            (h // 2 - s_half, h // 2 + s_half, 0, s),          # Left-mid
            (h // 2 - s_half, h // 2 + s_half, w - s, w)       # Right-mid
        ]
        
        sample_means = []
        for r1, r2, c1, c2 in sample_coords:
            r1, r2 = max(0, r1), min(h, r2)
            c1, c2 = max(0, c1), min(w, c2)
            
            if r1 < r2 and c1 < c2:
                sample = image[r1:r2, c1:c2]
                if sample.size > 0:
                    if len(sample.shape) == 3:
                        mean_color = np.mean(sample, axis=(0, 1))
                    else:
                        mean_val = np.mean(sample)
                        mean_color = np.array([mean_val, mean_val, mean_val])
                    sample_means.append(mean_color)
        
        if not sample_means:
            logger.error("No valid sample areas found for mask detection.")
            return np.array([0, 0, 0], dtype=np.float32)
        
        return np.mean(sample_means, axis=0).astype(np.float32)
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> Tuple[int, int, int]:
        """Convert RGB color to HSV tuple."""
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)[0, 0]
        return (int(hsv[0]), int(hsv[1]), int(hsv[2]))
    
    def _classify_mask(self, hsv: Tuple[int, int, int]) -> str:
        """
        Classify film type based on HSV values.
        
        Classification order matters - most specific first.
        """
        hue, sat, val = hsv
        
        # Get thresholds from params
        clear_sat_max = self.params.get('mask_clear_sat_max', 40)
        
        # C-41 thresholds
        c41_hue_min = self.params.get('mask_c41_hue_min', 8)
        c41_hue_max = self.params.get('mask_c41_hue_max', 22)
        c41_sat_min = self.params.get('mask_c41_sat_min', 70)
        c41_val_min = self.params.get('mask_c41_val_min', 60)
        c41_val_max = self.params.get('mask_c41_val_max', 210)
        
        # ECN-2 thresholds (darker orange/brown mask)
        ecn2_hue_min = self.params.get('mask_ecn2_hue_min', 5)
        ecn2_hue_max = self.params.get('mask_ecn2_hue_max', 25)
        ecn2_sat_min = self.params.get('mask_ecn2_sat_min', 50)
        ecn2_val_min = self.params.get('mask_ecn2_val_min', 30)
        ecn2_val_max = self.params.get('mask_ecn2_val_max', 80)
        
        # E-6 thresholds (clear, bright base)
        e6_sat_max = self.params.get('mask_e6_sat_max', 25)
        e6_val_min = self.params.get('mask_e6_val_min', 200)
        
        # B&W thresholds
        bw_sat_max = self.params.get('mask_bw_sat_max', 20)
        bw_val_min = self.params.get('mask_bw_val_min', 100)
        bw_val_max = self.params.get('mask_bw_val_max', 255)
        
        # Classification logic - order matters (most specific first)
        
        # ECN-2: darker orange/brown mask (motion picture negative)
        if (ecn2_hue_min <= hue <= ecn2_hue_max and
            sat >= ecn2_sat_min and
            ecn2_val_min <= val <= ecn2_val_max):
            return self.CLASSIFICATION_ECN2
        
        # C-41: standard orange mask (color negative)
        if (c41_hue_min <= hue <= c41_hue_max and
            sat >= c41_sat_min and
            c41_val_min <= val <= c41_val_max):
            return self.CLASSIFICATION_C41
        
        # E-6: clear bright base (slide/reversal film)
        if sat <= e6_sat_max and val >= e6_val_min:
            return self.CLASSIFICATION_E6
        
        # B&W: low saturation, moderate to high value
        if sat <= bw_sat_max and bw_val_min <= val <= bw_val_max:
            return self.CLASSIFICATION_BW
        
        # Clear/Near Clear: generic low saturation
        if sat < clear_sat_max:
            return self.CLASSIFICATION_CLEAR
        
        return self.CLASSIFICATION_UNKNOWN
    
    def _assess_confidence(self, image: np.ndarray, mask_color: np.ndarray) -> str:
        """Assess confidence of detection based on sample variance."""
        if image is None or image.size == 0 or len(image.shape) < 2:
            return "low"
        
        h, w = image.shape[:2]
        s = self.params.get('mask_sample_size', 10)
        
        if h < s or w < s:
            return "low"
        
        # Quick variance check on corners only
        corners = [
            image[:s, :s],
            image[:s, -s:],
            image[-s:, :s],
            image[-s:, -s:]
        ]
        
        means = []
        for corner in corners:
            if corner.size > 0 and len(corner.shape) == 3:
                means.append(np.mean(corner, axis=(0, 1)))
        
        if len(means) < 2:
            return "low"
        
        std_dev = np.mean(np.std(means, axis=0))
        variance_threshold = self.params.get('variance_threshold', 25.0)
        
        if std_dev > variance_threshold * 1.5:
            return "low"
        elif std_dev > variance_threshold:
            return "medium"
        return "high"


# Convenience function for backward compatibility
def detect_orange_mask(negative_image: np.ndarray) -> np.ndarray:
    """
    Detect the film base mask color.
    
    This is a convenience function that maintains backward compatibility.
    For full detection results, use FilmBaseDetector directly.
    
    Args:
        negative_image: Input image as NumPy array.
        
    Returns:
        RGB color of detected mask as float32 array.
    """
    detector = FilmBaseDetector()
    result = detector.detect(negative_image)
    return result.color


def classify_film_type(image: np.ndarray, params: Optional[Dict] = None) -> Tuple[str, np.ndarray]:
    """
    Classify film type from image.
    
    Args:
        image: Input image as NumPy array.
        params: Optional detection parameters.
        
    Returns:
        Tuple of (classification string, mask color array).
    """
    detector = FilmBaseDetector(params)
    result = detector.detect(image)
    return result.classification, result.color
