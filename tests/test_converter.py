import numpy as np
import pytest
from negative_converter.processing.converter import NegativeConverter, detect_orange_mask

def test_detect_orange_mask_returns_array(sample_image_uint8):
    """detect_orange_mask should return a 3-element numpy array."""
    mask = detect_orange_mask(sample_image_uint8)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (3,)
    assert mask.dtype == np.float32

def test_convert_returns_valid_image(sample_image_uint8):
    """NegativeConverter.convert should return a valid uint8 image."""
    converter = NegativeConverter()
    # Mocking progress callback
    steps = []
    def callback(curr, total):
        steps.append(curr)
        
    positive, mask_type = converter.convert(sample_image_uint8, progress_callback=callback)
    
    assert positive is not None
    assert positive.shape == sample_image_uint8.shape
    assert positive.dtype == np.uint8
    assert mask_type in ["Likely C-41", "Clear/Near Clear", "Unknown/Other", "Error"]
    assert len(steps) > 0

def test_convert_input_validation():
    """Converter should raise ValueError for invalid inputs."""
    converter = NegativeConverter()
    
    # Empty image
    with pytest.raises(ValueError, match="Input image is empty"):
        converter.convert(np.array([]))
        
    # Wrong channels
    with pytest.raises(ValueError, match="Input image must be a 3-channel"):
        converter.convert(np.zeros((10, 10, 1), dtype=np.uint8))

def test_convert_consistency(sample_image_uint8):
    """Converting the same image twice should yield the same result."""
    converter = NegativeConverter()
    res1, _ = converter.convert(sample_image_uint8)
    res2, _ = converter.convert(sample_image_uint8)
    assert np.array_equal(res1, res2)


def test_mask_classification_c41():
    """Test C-41 mask detection with orange-tinted image."""
    # Create an image with orange corners (typical C-41 mask)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill corners with orange color (R=200, G=120, B=80)
    img[:10, :10] = [200, 120, 80]
    img[:10, -10:] = [200, 120, 80]
    img[-10:, :10] = [200, 120, 80]
    img[-10:, -10:] = [200, 120, 80]
    
    converter = NegativeConverter()
    _, mask_type = converter.convert(img)
    assert mask_type == "Likely C-41"

def test_mask_classification_clear():
    """Test clear/neutral mask detection."""
    # Create a grayscale-ish image (low saturation)
    img = np.full((100, 100, 3), 180, dtype=np.uint8)
    
    converter = NegativeConverter()
    _, mask_type = converter.convert(img)
    # Should be detected as clear, B&W, or E-6 (all have low saturation)
    assert mask_type in ["Clear/Near Clear", "Likely B&W", "Likely E-6"]

def test_mask_classification_bw():
    """Test B&W mask detection with neutral gray image."""
    # Create a neutral gray image (very low saturation, moderate value)
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    
    converter = NegativeConverter()
    _, mask_type = converter.convert(img)
    # Should be detected as B&W or clear
    assert mask_type in ["Likely B&W", "Clear/Near Clear"]

def test_new_film_profiles_exist():
    """Test that new film profiles can be loaded."""
    # Test E-6 profile
    converter_e6 = NegativeConverter(film_profile="E6")
    assert converter_e6.profile_data is not None
    
    # Test ECN-2 profile
    converter_ecn2 = NegativeConverter(film_profile="ECN2")
    assert converter_ecn2.profile_data is not None
