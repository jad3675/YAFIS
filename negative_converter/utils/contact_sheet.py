# Contact Sheet Generator
"""
Generate contact sheets (thumbnail grids) from a collection of images.
"""
import math
from typing import List, Optional, Tuple
import numpy as np
import cv2

from .logger import get_logger

logger = get_logger(__name__)


def create_contact_sheet(
    images: List[np.ndarray],
    filenames: Optional[List[str]] = None,
    columns: int = 4,
    thumb_size: Tuple[int, int] = (300, 200),
    padding: int = 10,
    background_color: Tuple[int, int, int] = (40, 40, 40),
    border_color: Tuple[int, int, int] = (80, 80, 80),
    border_width: int = 2,
    show_filenames: bool = True,
    font_color: Tuple[int, int, int] = (200, 200, 200),
    font_scale: float = 0.4,
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Create a contact sheet from a list of images.
    
    Args:
        images: List of RGB uint8 numpy arrays
        filenames: Optional list of filenames to display under each thumbnail
        columns: Number of columns in the grid
        thumb_size: (width, height) of each thumbnail
        padding: Padding between thumbnails in pixels
        background_color: RGB background color
        border_color: RGB border color around thumbnails
        border_width: Width of border around thumbnails
        show_filenames: Whether to show filenames under thumbnails
        font_color: RGB color for filename text
        font_scale: Font scale for filename text
        title: Optional title to display at the top
        
    Returns:
        Contact sheet as RGB uint8 numpy array
    """
    if not images:
        logger.warning("No images provided for contact sheet")
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    num_images = len(images)
    rows = math.ceil(num_images / columns)
    
    thumb_w, thumb_h = thumb_size
    
    # Calculate text height if showing filenames
    text_height = 20 if show_filenames else 0
    
    # Calculate title height
    title_height = 40 if title else 0
    
    # Calculate total dimensions
    cell_w = thumb_w + 2 * border_width
    cell_h = thumb_h + 2 * border_width + text_height
    
    total_w = columns * cell_w + (columns + 1) * padding
    total_h = rows * cell_h + (rows + 1) * padding + title_height
    
    # Create background
    sheet = np.full((total_h, total_w, 3), background_color, dtype=np.uint8)
    
    # Add title if provided
    if title:
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, 0.8, 2)[0]
        title_x = (total_w - title_size[0]) // 2
        title_y = 30
        cv2.putText(sheet, title, (title_x, title_y), font, 0.8, font_color, 2, cv2.LINE_AA)
    
    # Place thumbnails
    for idx, img in enumerate(images):
        row = idx // columns
        col = idx % columns
        
        # Calculate position
        x = padding + col * (cell_w + padding)
        y = title_height + padding + row * (cell_h + padding)
        
        # Create thumbnail
        thumb = create_thumbnail(img, thumb_w, thumb_h)
        
        # Draw border
        cv2.rectangle(
            sheet,
            (x, y),
            (x + cell_w - 1, y + thumb_h + 2 * border_width - 1),
            border_color,
            border_width
        )
        
        # Place thumbnail
        thumb_x = x + border_width
        thumb_y = y + border_width
        sheet[thumb_y:thumb_y + thumb_h, thumb_x:thumb_x + thumb_w] = thumb
        
        # Add filename if provided
        if show_filenames and filenames and idx < len(filenames):
            filename = filenames[idx]
            # Truncate long filenames
            max_chars = thumb_w // 7  # Approximate chars that fit
            if len(filename) > max_chars:
                filename = filename[:max_chars-3] + "..."
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_y = y + thumb_h + 2 * border_width + 15
            cv2.putText(sheet, filename, (x + 5, text_y), font, font_scale, font_color, 1, cv2.LINE_AA)
    
    return sheet


def create_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Create a thumbnail that fits within the given dimensions while preserving aspect ratio.
    Centers the image on a black background if aspect ratios don't match.
    
    Args:
        image: RGB uint8 numpy array
        width: Target width
        height: Target height
        
    Returns:
        Thumbnail as RGB uint8 numpy array
    """
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    img_h, img_w = image.shape[:2]
    
    # Calculate scale to fit within bounds
    scale_w = width / img_w
    scale_h = height / img_h
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create black background
    thumb = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Center the resized image
    x_offset = (width - new_w) // 2
    y_offset = (height - new_h) // 2
    
    thumb[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return thumb


def save_contact_sheet(
    sheet: np.ndarray,
    output_path: str,
    quality: int = 95
) -> bool:
    """
    Save a contact sheet to file.
    
    Args:
        sheet: Contact sheet as RGB uint8 numpy array
        output_path: Output file path
        quality: JPEG quality (1-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
        
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            cv2.imwrite(output_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.lower().endswith('.png'):
            cv2.imwrite(output_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        else:
            cv2.imwrite(output_path, bgr)
        
        logger.info("Contact sheet saved to %s", output_path)
        return True
        
    except Exception:
        logger.exception("Failed to save contact sheet to %s", output_path)
        return False
