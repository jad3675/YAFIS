import numpy as np
import pytest
from negative_converter.utils.imaging import apply_curve

def test_apply_curve_identity(sample_image_uint8, identity_curve):
    """Applying an identity curve should not change the image."""
    result = apply_curve(sample_image_uint8, identity_curve)
    assert np.array_equal(sample_image_uint8, result)

def test_apply_curve_shape(sample_image_uint8, sample_curve):
    """Applying a curve should maintain the image shape and dtype."""
    result = apply_curve(sample_image_uint8, sample_curve)
    assert result.shape == sample_image_uint8.shape
    assert result.dtype == sample_image_uint8.dtype

def test_apply_curve_black_white_preserved(identity_curve):
    """Black (0) and White (255) should map correctly in identity curve."""
    img = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = apply_curve(img, identity_curve)
    assert result[0, 0, 0] == 0
    assert result[0, 0, 2] == 255
