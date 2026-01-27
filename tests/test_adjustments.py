import numpy as np
import pytest
from negative_converter.processing.adjustments import ImageAdjustments

def test_brightness_zero_is_noop(sample_image_uint8):
    """Brightness of 0 should be a no-op."""
    adj = ImageAdjustments()
    result = adj.adjust_brightness(sample_image_uint8, 0)
    # Brightness might have tiny rounding diffs depending on implementation, 
    # but for uint8 and 0 it should be exact.
    assert np.array_equal(sample_image_uint8, result)

def test_contrast_zero_is_noop(sample_image_uint8):
    """Contrast of 0 should be a no-op."""
    adj = ImageAdjustments()
    result = adj.adjust_contrast(sample_image_uint8, 0)
    assert np.array_equal(sample_image_uint8, result)

def test_saturation_zero_is_noop(sample_image_uint8):
    """Saturation of 0 should be a no-op."""
    adj = ImageAdjustments()
    result = adj.adjust_saturation(sample_image_uint8, 0)
    assert np.allclose(sample_image_uint8, result, atol=1) # LAB/HSV conversion might have tiny diffs

def test_adjustments_output_bounds(sample_image_uint8):
    """Adjustments should keep output in [0, 255] range."""
    adj = ImageAdjustments()
    # High brightness/contrast
    res1 = adj.adjust_brightness(sample_image_uint8, 100)
    res2 = adj.adjust_contrast(sample_image_uint8, 100)
    
    assert res1.min() >= 0 and res1.max() <= 255
    assert res2.min() >= 0 and res2.max() <= 255
    assert res1.dtype == np.uint8
    assert res2.dtype == np.uint8
