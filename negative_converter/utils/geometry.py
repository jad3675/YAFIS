# Geometry tools for crop and rotate operations
"""
Image geometry operations: crop, rotate, straighten, flip.
"""

from typing import Tuple, Optional, NamedTuple
from dataclasses import dataclass
import numpy as np
import math

from .logger import get_logger

logger = get_logger(__name__)


class CropRect(NamedTuple):
    """Rectangle for cropping (x, y, width, height)."""
    x: int
    y: int
    width: int
    height: int
    
    def to_slice(self) -> Tuple[slice, slice]:
        """Convert to numpy array slices (y_slice, x_slice)."""
        return (
            slice(self.y, self.y + self.height),
            slice(self.x, self.x + self.width)
        )
    
    def is_valid(self, image_width: int, image_height: int) -> bool:
        """Check if crop rect is valid for given image dimensions."""
        return (
            self.x >= 0 and self.y >= 0 and
            self.width > 0 and self.height > 0 and
            self.x + self.width <= image_width and
            self.y + self.height <= image_height
        )
    
    def clamp(self, image_width: int, image_height: int) -> 'CropRect':
        """Clamp crop rect to image bounds."""
        x = max(0, min(self.x, image_width - 1))
        y = max(0, min(self.y, image_height - 1))
        w = max(1, min(self.width, image_width - x))
        h = max(1, min(self.height, image_height - y))
        return CropRect(x, y, w, h)


@dataclass
class AspectRatio:
    """Aspect ratio constraint for cropping."""
    width: int
    height: int
    name: str = ""
    
    @property
    def ratio(self) -> float:
        """Get the ratio as a float (width/height)."""
        return self.width / self.height if self.height > 0 else 1.0
    
    def constrain(self, rect: CropRect, anchor: str = "center") -> CropRect:
        """
        Constrain a crop rect to this aspect ratio.
        
        Args:
            rect: The crop rectangle to constrain.
            anchor: Where to anchor ("center", "top-left", etc.)
            
        Returns:
            New CropRect with the aspect ratio applied.
        """
        current_ratio = rect.width / rect.height if rect.height > 0 else 1.0
        target_ratio = self.ratio
        
        if abs(current_ratio - target_ratio) < 0.001:
            return rect
        
        if current_ratio > target_ratio:
            # Too wide, reduce width
            new_width = int(rect.height * target_ratio)
            new_height = rect.height
        else:
            # Too tall, reduce height
            new_width = rect.width
            new_height = int(rect.width / target_ratio)
        
        # Adjust position based on anchor
        if anchor == "center":
            new_x = rect.x + (rect.width - new_width) // 2
            new_y = rect.y + (rect.height - new_height) // 2
        elif anchor == "top-left":
            new_x, new_y = rect.x, rect.y
        else:
            new_x, new_y = rect.x, rect.y
        
        return CropRect(new_x, new_y, new_width, new_height)


# Common aspect ratios
ASPECT_RATIOS = {
    "free": None,
    "1:1": AspectRatio(1, 1, "Square"),
    "4:3": AspectRatio(4, 3, "4:3"),
    "3:2": AspectRatio(3, 2, "3:2 (35mm)"),
    "16:9": AspectRatio(16, 9, "16:9 (Widescreen)"),
    "5:4": AspectRatio(5, 4, "5:4"),
    "7:5": AspectRatio(7, 5, "7:5"),
    "6:7": AspectRatio(6, 7, "6:7 (Medium Format)"),
    "4:5": AspectRatio(4, 5, "4:5 (Large Format)"),
    "original": AspectRatio(0, 0, "Original"),  # Special: use image's original ratio
}


def crop_image(image: np.ndarray, rect: CropRect) -> np.ndarray:
    """
    Crop an image to the specified rectangle.
    
    Args:
        image: Input image array (H, W, C).
        rect: Crop rectangle.
        
    Returns:
        Cropped image array.
    """
    if image is None:
        return None
    
    h, w = image.shape[:2]
    rect = rect.clamp(w, h)
    
    y_slice, x_slice = rect.to_slice()
    return image[y_slice, x_slice].copy()


def rotate_image(
    image: np.ndarray,
    angle: float,
    expand: bool = True,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Rotate an image by the specified angle.
    
    Args:
        image: Input image array (H, W, C).
        angle: Rotation angle in degrees (positive = counter-clockwise).
        expand: If True, expand output to fit rotated image.
        fill_color: Color to fill empty areas.
        
    Returns:
        Rotated image array.
    """
    if image is None:
        return None
    
    import cv2
    
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if expand:
        # Calculate new image size
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new center
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        output_size = (new_w, new_h)
    else:
        output_size = (w, h)
    
    # Perform rotation
    rotated = cv2.warpAffine(
        image, M, output_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill_color
    )
    
    return rotated


def rotate_90(image: np.ndarray, clockwise: bool = True) -> np.ndarray:
    """
    Rotate image by 90 degrees.
    
    Args:
        image: Input image array.
        clockwise: If True, rotate clockwise; otherwise counter-clockwise.
        
    Returns:
        Rotated image array.
    """
    if image is None:
        return None
    
    if clockwise:
        return np.rot90(image, k=-1)  # k=-1 is clockwise
    else:
        return np.rot90(image, k=1)   # k=1 is counter-clockwise


def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: Input image array.
        horizontal: If True, flip horizontally; otherwise vertically.
        
    Returns:
        Flipped image array.
    """
    if image is None:
        return None
    
    if horizontal:
        return np.fliplr(image).copy()
    else:
        return np.flipud(image).copy()


def straighten_image(
    image: np.ndarray,
    angle: float,
    auto_crop: bool = True
) -> np.ndarray:
    """
    Straighten an image by rotating and optionally auto-cropping.
    
    Args:
        image: Input image array.
        angle: Straighten angle in degrees.
        auto_crop: If True, crop to remove empty corners.
        
    Returns:
        Straightened image array.
    """
    if image is None or abs(angle) < 0.01:
        return image
    
    # Rotate with expansion
    rotated = rotate_image(image, angle, expand=True)
    
    if auto_crop:
        # Calculate the largest inscribed rectangle
        h, w = image.shape[:2]
        rh, rw = rotated.shape[:2]
        
        # Calculate crop bounds
        angle_rad = abs(math.radians(angle))
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Maximum inscribed rectangle dimensions
        if w * sin_a + h * cos_a > 0:
            crop_w = int((w * cos_a - h * sin_a) / (cos_a * cos_a - sin_a * sin_a + 0.001))
            crop_h = int((h * cos_a - w * sin_a) / (cos_a * cos_a - sin_a * sin_a + 0.001))
        else:
            crop_w, crop_h = w, h
        
        # Ensure positive dimensions
        crop_w = max(1, min(crop_w, rw))
        crop_h = max(1, min(crop_h, rh))
        
        # Center crop
        x = (rw - crop_w) // 2
        y = (rh - crop_h) // 2
        
        rect = CropRect(x, y, crop_w, crop_h)
        return crop_image(rotated, rect)
    
    return rotated


def detect_horizon_angle(image: np.ndarray) -> float:
    """
    Attempt to detect the horizon angle in an image using edge detection.
    
    Args:
        image: Input image array.
        
    Returns:
        Detected angle in degrees, or 0 if detection fails.
    """
    if image is None:
        return 0.0
    
    import cv2
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Collect angles near horizontal (within 45 degrees)
        angles = []
        for line in lines[:50]:  # Limit to first 50 lines
            rho, theta = line[0]
            angle_deg = math.degrees(theta) - 90
            
            # Only consider near-horizontal lines
            if abs(angle_deg) < 45:
                angles.append(angle_deg)
        
        if not angles:
            return 0.0
        
        # Return median angle
        return float(np.median(angles))
        
    except Exception:
        logger.exception("Horizon detection failed")
        return 0.0


def auto_crop_borders(
    image: np.ndarray,
    threshold: int = 10,
    margin: int = 0
) -> CropRect:
    """
    Detect and return crop rect to remove dark/uniform borders.
    
    Useful for removing film rebate or scanner borders.
    
    Args:
        image: Input image array.
        threshold: Pixel value threshold for border detection.
        margin: Additional margin to remove.
        
    Returns:
        CropRect for the detected content area.
    """
    if image is None:
        return CropRect(0, 0, 1, 1)
    
    import cv2
    
    h, w = image.shape[:2]
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Find non-border pixels
    mask = gray > threshold
    
    # Find bounding box of non-border region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return CropRect(0, 0, w, h)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Apply margin
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(w - 1, x_max + margin)
    y_max = min(h - 1, y_max + margin)
    
    return CropRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
