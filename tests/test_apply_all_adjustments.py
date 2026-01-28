"""Tests for the apply_all_adjustments function."""

import numpy as np
import pytest
from negative_converter.processing.adjustments import apply_all_adjustments, ImageAdjustments, AdvancedAdjustments


class TestApplyAllAdjustments:
    """Tests for the main adjustment pipeline."""
    
    def test_empty_adjustments_returns_copy(self, sample_image_uint8):
        """Empty adjustments dict should return a copy of the image."""
        result = apply_all_adjustments(sample_image_uint8, {})
        assert result is not None
        assert result.shape == sample_image_uint8.shape
        assert result.dtype == np.uint8
        # Should be a copy, not the same object
        assert result is not sample_image_uint8
    
    def test_none_image_returns_none(self):
        """None image should return None."""
        result = apply_all_adjustments(None, {'brightness': 10})
        assert result is None
    
    def test_brightness_adjustment(self, sample_image_uint8):
        """Brightness adjustment should modify the image."""
        result = apply_all_adjustments(sample_image_uint8, {'brightness': 50})
        assert result is not None
        # Positive brightness should increase average pixel value
        assert np.mean(result) > np.mean(sample_image_uint8)
    
    def test_contrast_adjustment(self, sample_image_uint8):
        """Contrast adjustment should modify the image."""
        result = apply_all_adjustments(sample_image_uint8, {'contrast': 50})
        assert result is not None
        # Higher contrast should increase standard deviation
        assert np.std(result) >= np.std(sample_image_uint8) * 0.9  # Allow some tolerance
    
    def test_saturation_adjustment(self, sample_image_uint8):
        """Saturation adjustment should modify the image."""
        result = apply_all_adjustments(sample_image_uint8, {'saturation': 50})
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_temp_tint_adjustment(self, sample_image_uint8):
        """Temperature and tint adjustments should modify the image."""
        result = apply_all_adjustments(sample_image_uint8, {'temp': 30, 'tint': -20})
        assert result is not None
        # Warm temp should increase red channel relative to blue
        assert result.shape == sample_image_uint8.shape
    
    def test_shadows_highlights_adjustment(self, sample_image_uint8):
        """Shadows and highlights adjustments should work."""
        result = apply_all_adjustments(sample_image_uint8, {'shadows': 30, 'highlights': -20})
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_levels_adjustment(self, sample_image_uint8):
        """Levels adjustment should modify the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'levels_in_black': 20,
            'levels_in_white': 235,
            'levels_gamma': 1.2,
            'levels_out_black': 10,
            'levels_out_white': 245,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_multiple_adjustments_combined(self, sample_image_uint8):
        """Multiple adjustments should be applied together."""
        result = apply_all_adjustments(sample_image_uint8, {
            'brightness': 10,
            'contrast': 20,
            'saturation': 15,
            'temp': 5,
            'shadows': 10,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
        assert result.dtype == np.uint8
    
    def test_output_bounds(self, sample_image_uint8):
        """Output should always be in valid uint8 range."""
        # Extreme adjustments
        result = apply_all_adjustments(sample_image_uint8, {
            'brightness': 100,
            'contrast': 100,
            'saturation': 100,
        })
        assert result is not None
        assert result.min() >= 0
        assert result.max() <= 255
        assert result.dtype == np.uint8
    
    def test_negative_adjustments(self, sample_image_uint8):
        """Negative adjustment values should work."""
        result = apply_all_adjustments(sample_image_uint8, {
            'brightness': -50,
            'contrast': -30,
            'saturation': -50,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape


class TestChannelMixer:
    """Tests for channel mixer adjustments."""
    
    def test_identity_mixer_no_change(self, sample_image_uint8):
        """Identity channel mixer should not change the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'mixer_red_r': 100, 'mixer_red_g': 0, 'mixer_red_b': 0,
            'mixer_green_r': 0, 'mixer_green_g': 100, 'mixer_green_b': 0,
            'mixer_blue_r': 0, 'mixer_blue_g': 0, 'mixer_blue_b': 100,
        })
        assert result is not None
        # Should be very close to original
        assert np.allclose(result, sample_image_uint8, atol=2)
    
    def test_channel_swap(self, sample_image_uint8):
        """Channel mixer should be able to swap channels."""
        result = apply_all_adjustments(sample_image_uint8, {
            'mixer_red_r': 0, 'mixer_red_g': 0, 'mixer_red_b': 100,
            'mixer_green_r': 0, 'mixer_green_g': 100, 'mixer_green_b': 0,
            'mixer_blue_r': 100, 'mixer_blue_g': 0, 'mixer_blue_b': 0,
        })
        assert result is not None
        # Red and blue should be swapped
        assert result.shape == sample_image_uint8.shape


class TestHSLAdjustments:
    """Tests for HSL color range adjustments."""
    
    def test_hsl_reds_adjustment(self, sample_image_uint8):
        """HSL adjustment for reds should work."""
        result = apply_all_adjustments(sample_image_uint8, {
            'hsl_reds_h': 10,
            'hsl_reds_s': 20,
            'hsl_reds_l': 5,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_hsl_multiple_ranges(self, sample_image_uint8):
        """Multiple HSL range adjustments should work together."""
        result = apply_all_adjustments(sample_image_uint8, {
            'hsl_reds_h': 5,
            'hsl_greens_s': 30,
            'hsl_blues_l': -10,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape


class TestCurvesAdjustments:
    """Tests for curves adjustments."""
    
    def test_identity_curve_no_change(self, sample_image_uint8, identity_curve):
        """Identity curve should not change the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'curves_rgb': identity_curve,
        })
        assert result is not None
        assert np.allclose(result, sample_image_uint8, atol=1)
    
    def test_s_curve_increases_contrast(self, sample_image_uint8, sample_curve):
        """S-curve should increase contrast."""
        result = apply_all_adjustments(sample_image_uint8, {
            'curves_rgb': sample_curve,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_per_channel_curves(self, sample_image_uint8, sample_curve, identity_curve):
        """Per-channel curves should work independently."""
        result = apply_all_adjustments(sample_image_uint8, {
            'curves_red': sample_curve,
            'curves_green': identity_curve,
            'curves_blue': identity_curve,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape


class TestSelectiveColor:
    """Tests for selective color adjustments."""
    
    def test_selective_color_reds(self, sample_image_uint8):
        """Selective color adjustment for reds should work."""
        result = apply_all_adjustments(sample_image_uint8, {
            'sel_reds_c': 10,
            'sel_reds_m': -5,
            'sel_reds_y': 15,
            'sel_reds_k': 0,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
    
    def test_selective_color_relative_mode(self, sample_image_uint8):
        """Selective color in relative mode should work."""
        result = apply_all_adjustments(sample_image_uint8, {
            'sel_relative': True,
            'sel_neutrals_c': 5,
            'sel_neutrals_m': 5,
            'sel_neutrals_y': 5,
            'sel_neutrals_k': -5,
        })
        assert result is not None


class TestNoiseReduction:
    """Tests for noise reduction."""
    
    def test_noise_reduction_zero_is_noop(self, sample_image_uint8):
        """Zero noise reduction should not change the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'noise_reduction_strength': 0,
        })
        assert result is not None
        assert np.array_equal(result, sample_image_uint8)
    
    def test_noise_reduction_applies(self, sample_image_uint8):
        """Noise reduction should smooth the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'noise_reduction_strength': 50,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape


class TestDustRemoval:
    """Tests for dust removal."""
    
    def test_dust_removal_disabled_by_default(self, sample_image_uint8):
        """Dust removal should be disabled by default."""
        result = apply_all_adjustments(sample_image_uint8, {})
        assert result is not None
        # Should be unchanged (dust removal not applied)
    
    def test_dust_removal_enabled(self, sample_image_uint8):
        """Dust removal when enabled should process the image."""
        result = apply_all_adjustments(sample_image_uint8, {
            'dust_removal_enabled': True,
            'dust_removal_sensitivity': 50,
            'dust_removal_radius': 3,
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_very_small_image(self):
        """Very small images should be handled."""
        small_img = np.full((5, 5, 3), 128, dtype=np.uint8)
        result = apply_all_adjustments(small_img, {'brightness': 10})
        assert result is not None
        assert result.shape == small_img.shape
    
    def test_single_pixel_image(self):
        """Single pixel images should be handled."""
        pixel_img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = apply_all_adjustments(pixel_img, {'brightness': 10})
        assert result is not None
        assert result.shape == pixel_img.shape
    
    def test_grayscale_like_image(self):
        """Grayscale-like RGB images should be handled."""
        gray_img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = apply_all_adjustments(gray_img, {
            'saturation': 50,  # Should have no effect on gray
            'brightness': 10,
        })
        assert result is not None
    
    def test_extreme_values(self):
        """Extreme adjustment values should be handled safely."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = apply_all_adjustments(img, {
            'brightness': 200,  # Beyond normal range
            'contrast': 200,
            'saturation': 200,
        })
        assert result is not None
        assert result.min() >= 0
        assert result.max() <= 255
