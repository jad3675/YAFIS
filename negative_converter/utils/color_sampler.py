# Color sampler tool for reading pixel values
"""
Color sampler utility for reading and displaying pixel color values.
"""

from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
import numpy as np
import colorsys

from .logger import get_logger

logger = get_logger(__name__)


class ColorSample(NamedTuple):
    """A color sample with position and values."""
    x: int
    y: int
    rgb: Tuple[int, int, int]
    
    @property
    def r(self) -> int:
        return self.rgb[0]
    
    @property
    def g(self) -> int:
        return self.rgb[1]
    
    @property
    def b(self) -> int:
        return self.rgb[2]
    
    @property
    def hex(self) -> str:
        """Get hex color string."""
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"
    
    @property
    def hsv(self) -> Tuple[int, int, int]:
        """Get HSV values (H: 0-360, S: 0-100, V: 0-100)."""
        r, g, b = self.rgb
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        return (int(h * 360), int(s * 100), int(v * 100))
    
    @property
    def hsl(self) -> Tuple[int, int, int]:
        """Get HSL values (H: 0-360, S: 0-100, L: 0-100)."""
        r, g, b = self.rgb
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        return (int(h * 360), int(s * 100), int(l * 100))
    
    @property
    def lab(self) -> Tuple[float, float, float]:
        """Get CIE LAB values (approximate)."""
        return rgb_to_lab(self.rgb)
    
    @property
    def luminance(self) -> float:
        """Get relative luminance (0-1)."""
        r, g, b = self.rgb
        return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)
    
    def format_rgb(self) -> str:
        """Format as RGB string."""
        return f"R:{self.r} G:{self.g} B:{self.b}"
    
    def format_hsv(self) -> str:
        """Format as HSV string."""
        h, s, v = self.hsv
        return f"H:{h}° S:{s}% V:{v}%"
    
    def format_hsl(self) -> str:
        """Format as HSL string."""
        h, s, l = self.hsl
        return f"H:{h}° S:{s}% L:{l}%"
    
    def format_lab(self) -> str:
        """Format as LAB string."""
        l, a, b = self.lab
        return f"L:{l:.1f} a:{a:.1f} b:{b:.1f}"


def rgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """
    Convert RGB to CIE LAB color space.
    
    Args:
        rgb: RGB tuple (0-255 each).
        
    Returns:
        LAB tuple (L: 0-100, a: -128 to 127, b: -128 to 127).
    """
    # Normalize RGB to 0-1
    r, g, b = [x / 255.0 for x in rgb]
    
    # Apply gamma correction (sRGB to linear)
    def gamma_correct(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        return c / 12.92
    
    r, g, b = [gamma_correct(c) for c in (r, g, b)]
    
    # Convert to XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize for D65 white point
    x /= 0.95047
    y /= 1.00000
    z /= 1.08883
    
    # Convert to LAB
    def f(t):
        if t > 0.008856:
            return t ** (1/3)
        return (903.3 * t + 16) / 116
    
    fx, fy, fz = f(x), f(y), f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return (L, a, b)


def sample_color(image: np.ndarray, x: int, y: int) -> Optional[ColorSample]:
    """
    Sample color at a specific pixel location.
    
    Args:
        image: Image array (H, W, C) in RGB format.
        x: X coordinate.
        y: Y coordinate.
        
    Returns:
        ColorSample or None if coordinates are out of bounds.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return None
    
    pixel = image[y, x]
    if len(image.shape) == 2:
        # Grayscale
        rgb = (int(pixel), int(pixel), int(pixel))
    else:
        rgb = (int(pixel[0]), int(pixel[1]), int(pixel[2]))
    
    return ColorSample(x, y, rgb)


def sample_color_averaged(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int = 2
) -> Optional[ColorSample]:
    """
    Sample color averaged over a small region.
    
    Args:
        image: Image array (H, W, C) in RGB format.
        x: Center X coordinate.
        y: Center Y coordinate.
        radius: Radius of sampling region.
        
    Returns:
        ColorSample with averaged color, or None if out of bounds.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Calculate bounds
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius + 1)
    y2 = min(h, y + radius + 1)
    
    if x1 >= x2 or y1 >= y2:
        return None
    
    # Extract region and average
    region = image[y1:y2, x1:x2]
    
    if len(image.shape) == 2:
        avg = np.mean(region)
        rgb = (int(avg), int(avg), int(avg))
    else:
        avg = np.mean(region, axis=(0, 1))
        rgb = (int(avg[0]), int(avg[1]), int(avg[2]))
    
    return ColorSample(x, y, rgb)


@dataclass
class ColorSamplerState:
    """State for the color sampler tool."""
    samples: List[ColorSample]
    max_samples: int = 5
    
    def __init__(self, max_samples: int = 5):
        self.samples = []
        self.max_samples = max_samples
    
    def add_sample(self, sample: ColorSample) -> None:
        """Add a sample, removing oldest if at max."""
        self.samples.append(sample)
        while len(self.samples) > self.max_samples:
            self.samples.pop(0)
    
    def clear(self) -> None:
        """Clear all samples."""
        self.samples.clear()
    
    def remove_sample(self, index: int) -> None:
        """Remove sample at index."""
        if 0 <= index < len(self.samples):
            self.samples.pop(index)
    
    def get_sample(self, index: int) -> Optional[ColorSample]:
        """Get sample at index."""
        if 0 <= index < len(self.samples):
            return self.samples[index]
        return None


def analyze_skin_tone(sample: ColorSample) -> dict:
    """
    Analyze if a color sample is likely a skin tone.
    
    Returns analysis with suggestions for correction.
    """
    r, g, b = sample.rgb
    h, s, v = sample.hsv
    l, a, lab_b = sample.lab
    
    result = {
        "is_skin_tone": False,
        "quality": "unknown",
        "suggestions": []
    }
    
    # Basic skin tone detection (hue in orange-red range, moderate saturation)
    if 0 <= h <= 50 and 15 <= s <= 60 and 40 <= v <= 95:
        result["is_skin_tone"] = True
        
        # Check for common issues
        if a > 20:
            result["suggestions"].append("Skin appears too red/magenta")
        elif a < 5:
            result["suggestions"].append("Skin appears too green")
        
        if lab_b > 30:
            result["suggestions"].append("Skin appears too yellow")
        elif lab_b < 10:
            result["suggestions"].append("Skin appears too blue")
        
        if s > 45:
            result["suggestions"].append("Skin saturation may be too high")
        
        if not result["suggestions"]:
            result["quality"] = "good"
        else:
            result["quality"] = "needs_adjustment"
    
    return result


def check_neutral(sample: ColorSample, tolerance: int = 10) -> dict:
    """
    Check if a sample is neutral (gray) and analyze white balance.
    
    Args:
        sample: Color sample to analyze.
        tolerance: RGB channel difference tolerance.
        
    Returns:
        Analysis dict with neutrality info and WB suggestions.
    """
    r, g, b = sample.rgb
    
    max_diff = max(abs(r - g), abs(g - b), abs(r - b))
    
    result = {
        "is_neutral": max_diff <= tolerance,
        "max_channel_diff": max_diff,
        "dominant_channel": None,
        "wb_suggestion": None
    }
    
    if not result["is_neutral"]:
        # Determine dominant channel
        if r > g and r > b:
            result["dominant_channel"] = "red"
            result["wb_suggestion"] = "Reduce temperature (cooler)"
        elif b > r and b > g:
            result["dominant_channel"] = "blue"
            result["wb_suggestion"] = "Increase temperature (warmer)"
        elif g > r and g > b:
            result["dominant_channel"] = "green"
            result["wb_suggestion"] = "Adjust tint toward magenta"
        
        # Check for magenta/green tint
        if abs(g - (r + b) / 2) > tolerance:
            if g < (r + b) / 2:
                result["wb_suggestion"] = "Adjust tint toward green"
            else:
                result["wb_suggestion"] = "Adjust tint toward magenta"
    
    return result
