import pytest
import numpy as np

@pytest.fixture
def sample_image_uint8():
    """Returns a simple 100x100 uint8 RGB image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:50, :50] = [255, 0, 0]    # Red quadrant
    img[:50, 50:] = [0, 255, 0]    # Green quadrant
    img[50:, :50] = [0, 0, 255]    # Blue quadrant
    img[50:, 50:] = [255, 255, 0]  # Yellow quadrant
    return img

@pytest.fixture
def identity_curve():
    """Returns identity curve points."""
    return [[0, 0], [255, 255]]

@pytest.fixture
def sample_curve():
    """Returns a simple S-curve."""
    return [[0, 0], [64, 48], [128, 128], [192, 207], [255, 255]]
