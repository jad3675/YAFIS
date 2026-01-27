"""Tests for the mask detection module."""

import numpy as np
import pytest
from negative_converter.processing.mask_detection import (
    FilmBaseDetector,
    MaskDetectionResult,
    detect_orange_mask,
    classify_film_type,
)


class TestFilmBaseDetector:
    """Tests for FilmBaseDetector class."""
    
    def test_detector_initialization(self):
        """Detector should initialize with default params."""
        detector = FilmBaseDetector()
        assert detector.params is not None
    
    def test_detector_with_custom_params(self):
        """Detector should accept custom parameters."""
        custom_params = {'mask_sample_size': 20}
        detector = FilmBaseDetector(custom_params)
        assert detector.params['mask_sample_size'] == 20
    
    def test_detect_returns_result(self, sample_image_uint8):
        """detect() should return MaskDetectionResult."""
        detector = FilmBaseDetector()
        result = detector.detect(sample_image_uint8)
        
        assert isinstance(result, MaskDetectionResult)
        assert isinstance(result.color, np.ndarray)
        assert result.color.shape == (3,)
        assert result.classification in [
            FilmBaseDetector.CLASSIFICATION_C41,
            FilmBaseDetector.CLASSIFICATION_ECN2,
            FilmBaseDetector.CLASSIFICATION_E6,
            FilmBaseDetector.CLASSIFICATION_BW,
            FilmBaseDetector.CLASSIFICATION_CLEAR,
            FilmBaseDetector.CLASSIFICATION_UNKNOWN,
        ]
        assert len(result.hsv) == 3
        assert result.confidence in ["high", "medium", "low"]
    
    def test_detect_c41_orange_mask(self):
        """Should detect C-41 from orange corners."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Orange color typical of C-41 mask
        orange = [200, 120, 80]
        img[:10, :10] = orange
        img[:10, -10:] = orange
        img[-10:, :10] = orange
        img[-10:, -10:] = orange
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        assert result.classification == FilmBaseDetector.CLASSIFICATION_C41
    
    def test_detect_clear_base(self):
        """Should detect clear/neutral base."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # High value, low saturation should be E-6 or clear
        assert result.classification in [
            FilmBaseDetector.CLASSIFICATION_E6,
            FilmBaseDetector.CLASSIFICATION_CLEAR,
            FilmBaseDetector.CLASSIFICATION_BW,
        ]
    
    def test_detect_empty_image(self):
        """Should handle empty image gracefully."""
        detector = FilmBaseDetector()
        result = detector.detect(np.array([]))
        
        assert result.color.shape == (3,)
        assert np.allclose(result.color, [0, 0, 0])
    
    def test_detect_small_image(self):
        """Should handle very small images."""
        img = np.full((5, 5, 3), 128, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        assert result.color.shape == (3,)


class TestBackwardCompatibility:
    """Tests for backward-compatible functions."""
    
    def test_detect_orange_mask_function(self, sample_image_uint8):
        """detect_orange_mask() should return color array."""
        color = detect_orange_mask(sample_image_uint8)
        
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)
        assert color.dtype == np.float32
    
    def test_classify_film_type_function(self, sample_image_uint8):
        """classify_film_type() should return tuple."""
        classification, color = classify_film_type(sample_image_uint8)
        
        assert isinstance(classification, str)
        assert isinstance(color, np.ndarray)
        assert color.shape == (3,)


class TestConfidenceAssessment:
    """Tests for detection confidence assessment."""
    
    def test_high_confidence_uniform_corners(self):
        """Uniform corners should give high confidence."""
        img = np.full((100, 100, 3), 150, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        assert result.confidence == "high"
    
    def test_low_confidence_varied_corners(self):
        """Highly varied corners should give lower confidence."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Very different colors in each corner
        img[:10, :10] = [255, 0, 0]      # Red
        img[:10, -10:] = [0, 255, 0]     # Green
        img[-10:, :10] = [0, 0, 255]     # Blue
        img[-10:, -10:] = [255, 255, 0]  # Yellow
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # Should have lower confidence due to variance
        assert result.confidence in ["low", "medium"]


class TestEdgeCases:
    """Tests for edge cases in mask detection."""
    
    def test_very_dark_scan(self):
        """Very dark scans should still be classified (likely Unknown)."""
        # Simulate underexposed/dark scan
        img = np.full((100, 100, 3), 20, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # Should not crash, should return valid result
        assert result.classification is not None
        assert result.color.shape == (3,)
        # Very dark, low saturation - likely Unknown or Clear
        assert result.hsv[2] < 50  # Low value
    
    def test_overexposed_frame(self):
        """Overexposed/bright frames should be handled."""
        # Simulate overexposed scan - very bright
        img = np.full((100, 100, 3), 250, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        assert result.classification is not None
        # Very bright, low saturation - likely E-6 or Clear
        assert result.classification in [
            FilmBaseDetector.CLASSIFICATION_E6,
            FilmBaseDetector.CLASSIFICATION_CLEAR,
            FilmBaseDetector.CLASSIFICATION_BW,
        ]
    
    def test_mixed_lighting_borders(self):
        """Mixed lighting in borders should give medium/low confidence."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Very different lighting in each corner to trigger variance warning
        img[:10, :10] = [255, 150, 50]      # Bright warm
        img[:10, -10:] = [100, 60, 30]      # Dark warm
        img[-10:, :10] = [200, 200, 200]    # Neutral bright
        img[-10:, -10:] = [50, 30, 20]      # Very dark
        # Fill center with something different
        img[30:70, 30:70] = [100, 100, 100]
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # With such varied corners, confidence should not be high
        # The actual classification may vary, but confidence should reflect uncertainty
        assert result.confidence in ["low", "medium"]
    
    def test_grayscale_input(self):
        """Grayscale images should be handled gracefully."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # Should handle 2D input
        assert result.color.shape == (3,)
        # Grayscale represented as equal RGB
        assert np.allclose(result.color[0], result.color[1], atol=1)
        assert np.allclose(result.color[1], result.color[2], atol=1)
    
    def test_single_pixel_image(self):
        """Single pixel images should return low confidence."""
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        assert result.confidence == "low"
    
    def test_extreme_saturation_orange(self):
        """Highly saturated orange should be detected as C-41."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Very saturated orange in corners
        saturated_orange = [255, 100, 0]
        img[:10, :10] = saturated_orange
        img[:10, -10:] = saturated_orange
        img[-10:, :10] = saturated_orange
        img[-10:, -10:] = saturated_orange
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # High saturation orange should be C-41
        assert result.classification == FilmBaseDetector.CLASSIFICATION_C41
    
    def test_purple_tinted_base(self):
        """Purple/magenta tinted base should be Unknown."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Purple tint in corners
        purple = [180, 100, 180]
        img[:10, :10] = purple
        img[:10, -10:] = purple
        img[-10:, :10] = purple
        img[-10:, -10:] = purple
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # Purple is not a standard film base color
        assert result.classification == FilmBaseDetector.CLASSIFICATION_UNKNOWN
    
    def test_ecn2_dark_orange(self):
        """Dark orange/brown should be detected as ECN-2."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Dark brownish-orange typical of ECN-2
        dark_orange = [70, 45, 25]
        img[:10, :10] = dark_orange
        img[:10, -10:] = dark_orange
        img[-10:, :10] = dark_orange
        img[-10:, -10:] = dark_orange
        
        detector = FilmBaseDetector()
        result = detector.detect(img)
        
        # Dark orange in ECN-2 value range
        assert result.classification == FilmBaseDetector.CLASSIFICATION_ECN2


class TestWhiteBalanceCalculation:
    """Tests for white balance calculation in different scenarios."""
    
    def test_wb_calculator_c41(self):
        """WB calculator should produce valid scales for C-41."""
        from negative_converter.processing.processing_strategy import WhiteBalanceCalculator
        from negative_converter.config.settings import CONVERSION_DEFAULTS
        
        calc = WhiteBalanceCalculator(CONVERSION_DEFAULTS)
        mask_color = np.array([200, 120, 80], dtype=np.float32)
        
        scales = calc.calculate_for_c41(mask_color)
        
        assert scales.shape == (3,)
        assert all(0.8 <= s <= 1.3 for s in scales)  # Within clamp range
    
    def test_wb_calculator_ecn2(self):
        """WB calculator should handle ECN-2's darker mask."""
        from negative_converter.processing.processing_strategy import WhiteBalanceCalculator
        from negative_converter.config.settings import CONVERSION_DEFAULTS
        
        calc = WhiteBalanceCalculator(CONVERSION_DEFAULTS)
        mask_color = np.array([70, 45, 25], dtype=np.float32)
        
        scales = calc.calculate_for_ecn2(mask_color)
        
        assert scales.shape == (3,)
        # ECN-2 has wider clamp range
        assert all(0.7 <= s <= 1.5 for s in scales)
    
    def test_wb_gray_world_gentle(self):
        """Gentle gray world should have very conservative scales."""
        from negative_converter.processing.processing_strategy import WhiteBalanceCalculator
        from negative_converter.config.settings import CONVERSION_DEFAULTS
        
        calc = WhiteBalanceCalculator(CONVERSION_DEFAULTS)
        # Slightly unbalanced inverted image
        inverted = np.zeros((50, 50, 3), dtype=np.float32)
        inverted[:, :, 0] = 130  # R
        inverted[:, :, 1] = 128  # G
        inverted[:, :, 2] = 126  # B
        
        scales = calc.calculate_gray_world(inverted, gentle=True)
        
        # Gentle mode should have very tight clamps
        assert all(0.95 <= s <= 1.05 for s in scales)
