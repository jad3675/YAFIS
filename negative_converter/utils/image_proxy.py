# Image proxy/preview resolution management
"""
Provides proxy image generation for efficient preview editing of large images.

This module handles:
- Automatic downscaling for preview operations
- Memory-efficient processing of large images
- Upscaling results back to full resolution when needed
"""

import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


# Default thresholds
DEFAULT_PREVIEW_MAX_PIXELS = 4_000_000  # 4 megapixels for preview
DEFAULT_PROCESSING_MAX_PIXELS = 16_000_000  # 16 megapixels for processing
DEFAULT_WARNING_THRESHOLD_PIXELS = 50_000_000  # 50 megapixels triggers warning


@dataclass
class ImageProxyInfo:
    """Information about a proxy image."""
    original_shape: Tuple[int, int, int]
    proxy_shape: Tuple[int, int, int]
    scale_factor: float
    is_proxy: bool
    
    @property
    def original_megapixels(self) -> float:
        """Original image size in megapixels."""
        return (self.original_shape[0] * self.original_shape[1]) / 1_000_000
    
    @property
    def proxy_megapixels(self) -> float:
        """Proxy image size in megapixels."""
        return (self.proxy_shape[0] * self.proxy_shape[1]) / 1_000_000
    
    @property
    def memory_saved_mb(self) -> float:
        """Approximate memory saved in MB (assuming 3 channels, uint8)."""
        original_bytes = self.original_shape[0] * self.original_shape[1] * 3
        proxy_bytes = self.proxy_shape[0] * self.proxy_shape[1] * 3
        return (original_bytes - proxy_bytes) / (1024 * 1024)


def estimate_memory_usage(image: np.ndarray) -> float:
    """
    Estimate memory usage of an image in MB.
    
    Args:
        image: NumPy array image.
        
    Returns:
        Estimated memory usage in megabytes.
    """
    if image is None:
        return 0.0
    return image.nbytes / (1024 * 1024)


def get_pixel_count(image: np.ndarray) -> int:
    """Get total pixel count of an image."""
    if image is None or len(image.shape) < 2:
        return 0
    return image.shape[0] * image.shape[1]


def should_use_proxy(
    image: np.ndarray,
    max_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
) -> bool:
    """
    Determine if a proxy should be used for this image.
    
    Args:
        image: Input image.
        max_pixels: Maximum pixels before proxy is recommended.
        
    Returns:
        True if proxy should be used.
    """
    return get_pixel_count(image) > max_pixels


def is_large_image(
    image: np.ndarray,
    threshold: int = DEFAULT_WARNING_THRESHOLD_PIXELS,
) -> bool:
    """
    Check if image is large enough to warrant a warning.
    
    Args:
        image: Input image.
        threshold: Pixel count threshold.
        
    Returns:
        True if image is considered large.
    """
    return get_pixel_count(image) > threshold


def calculate_scale_factor(
    image: np.ndarray,
    max_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
) -> float:
    """
    Calculate the scale factor needed to fit within max_pixels.
    
    Args:
        image: Input image.
        max_pixels: Target maximum pixels.
        
    Returns:
        Scale factor (1.0 if no scaling needed, <1.0 for downscaling).
    """
    current_pixels = get_pixel_count(image)
    if current_pixels <= max_pixels:
        return 1.0
    return np.sqrt(max_pixels / current_pixels)


def create_proxy(
    image: np.ndarray,
    max_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
    interpolation: int = cv2.INTER_AREA,
) -> Tuple[np.ndarray, ImageProxyInfo]:
    """
    Create a proxy (downscaled) version of an image.
    
    Args:
        image: Input image (uint8 RGB).
        max_pixels: Maximum pixels for the proxy.
        interpolation: OpenCV interpolation method.
        
    Returns:
        Tuple of (proxy_image, proxy_info).
    """
    if image is None or image.size == 0:
        info = ImageProxyInfo(
            original_shape=(0, 0, 0),
            proxy_shape=(0, 0, 0),
            scale_factor=1.0,
            is_proxy=False,
        )
        return image, info
    
    original_shape = image.shape
    scale = calculate_scale_factor(image, max_pixels)
    
    if scale >= 1.0:
        # No scaling needed
        info = ImageProxyInfo(
            original_shape=original_shape,
            proxy_shape=original_shape,
            scale_factor=1.0,
            is_proxy=False,
        )
        return image, info
    
    # Calculate new dimensions
    new_height = int(image.shape[0] * scale)
    new_width = int(image.shape[1] * scale)
    
    # Ensure minimum dimensions
    new_height = max(1, new_height)
    new_width = max(1, new_width)
    
    # Resize
    proxy = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    info = ImageProxyInfo(
        original_shape=original_shape,
        proxy_shape=proxy.shape,
        scale_factor=scale,
        is_proxy=True,
    )
    
    logger.debug(
        "Created proxy: %.1f MP -> %.1f MP (scale=%.3f, saved %.1f MB)",
        info.original_megapixels,
        info.proxy_megapixels,
        scale,
        info.memory_saved_mb,
    )
    
    return proxy, info


def upscale_to_original(
    proxy_image: np.ndarray,
    proxy_info: ImageProxyInfo,
    interpolation: int = cv2.INTER_LANCZOS4,
) -> np.ndarray:
    """
    Upscale a proxy image back to original dimensions.
    
    Args:
        proxy_image: The proxy image to upscale.
        proxy_info: Information about the original image.
        interpolation: OpenCV interpolation method.
        
    Returns:
        Upscaled image matching original dimensions.
    """
    if not proxy_info.is_proxy:
        return proxy_image
    
    if proxy_image is None or proxy_image.size == 0:
        return proxy_image
    
    original_h, original_w = proxy_info.original_shape[:2]
    
    upscaled = cv2.resize(
        proxy_image,
        (original_w, original_h),
        interpolation=interpolation,
    )
    
    logger.debug(
        "Upscaled proxy: %.1f MP -> %.1f MP",
        proxy_info.proxy_megapixels,
        proxy_info.original_megapixels,
    )
    
    return upscaled


def process_with_proxy(
    image: np.ndarray,
    process_func,
    max_preview_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
    preview_mode: bool = True,
    **process_kwargs,
) -> np.ndarray:
    """
    Process an image using proxy for preview, full resolution for final.
    
    Args:
        image: Input image.
        process_func: Processing function that takes image as first argument.
        max_preview_pixels: Maximum pixels for preview mode.
        preview_mode: If True, use proxy; if False, process at full resolution.
        **process_kwargs: Additional arguments to pass to process_func.
        
    Returns:
        Processed image.
    """
    if image is None:
        return None
    
    if preview_mode and should_use_proxy(image, max_preview_pixels):
        # Create proxy, process, upscale
        proxy, info = create_proxy(image, max_preview_pixels)
        processed_proxy = process_func(proxy, **process_kwargs)
        return upscale_to_original(processed_proxy, info)
    else:
        # Process at full resolution
        return process_func(image, **process_kwargs)


class TiledProcessor:
    """
    Process very large images in tiles to manage memory.
    
    Useful for images that are too large to fit in memory even as proxies.
    """
    
    def __init__(
        self,
        tile_size: int = 1024,
        overlap: int = 64,
    ):
        """
        Initialize tiled processor.
        
        Args:
            tile_size: Size of each tile (square).
            overlap: Overlap between tiles to avoid seams.
        """
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process(
        self,
        image: np.ndarray,
        process_func,
        **process_kwargs,
    ) -> np.ndarray:
        """
        Process image in tiles.
        
        Args:
            image: Input image.
            process_func: Function to apply to each tile.
            **process_kwargs: Additional arguments for process_func.
            
        Returns:
            Processed image.
        """
        if image is None or image.size == 0:
            return image
        
        h, w = image.shape[:2]
        # Use float32 for accumulation to avoid casting errors
        result = np.zeros(image.shape, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        
        # Calculate tile positions
        step = self.tile_size - self.overlap
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Extract tile with overlap
                y1 = y
                y2 = min(y + self.tile_size, h)
                x1 = x
                x2 = min(x + self.tile_size, w)
                
                tile = image[y1:y2, x1:x2].copy()
                
                # Process tile
                processed_tile = process_func(tile, **process_kwargs)
                
                if processed_tile is None:
                    processed_tile = tile
                
                # Create blending weight (feather edges)
                tile_h, tile_w = processed_tile.shape[:2]
                tile_weight = self._create_blend_weight(tile_h, tile_w)
                
                # Accumulate result
                if len(processed_tile.shape) == 3:
                    for c in range(processed_tile.shape[2]):
                        result[y1:y2, x1:x2, c] += (
                            processed_tile[:, :, c].astype(np.float32) * tile_weight
                        )
                else:
                    result[y1:y2, x1:x2] += processed_tile.astype(np.float32) * tile_weight
                
                weight[y1:y2, x1:x2] += tile_weight
        
        # Normalize by weight
        weight = np.maximum(weight, 1e-6)  # Avoid division by zero
        if len(result.shape) == 3:
            for c in range(result.shape[2]):
                result[:, :, c] = result[:, :, c] / weight
        else:
            result = result / weight
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _create_blend_weight(self, h: int, w: int) -> np.ndarray:
        """Create a blending weight mask with feathered edges."""
        # Simple linear ramp for edges
        weight = np.ones((h, w), dtype=np.float32)
        
        ramp_size = min(self.overlap, h // 4, w // 4)
        if ramp_size > 0:
            ramp = np.linspace(0, 1, ramp_size)
            
            # Top edge
            weight[:ramp_size, :] *= ramp[:, np.newaxis]
            # Bottom edge
            weight[-ramp_size:, :] *= ramp[::-1, np.newaxis]
            # Left edge
            weight[:, :ramp_size] *= ramp[np.newaxis, :]
            # Right edge
            weight[:, -ramp_size:] *= ramp[::-1][np.newaxis, :]
        
        return weight


def get_memory_warning_message(image: np.ndarray) -> Optional[str]:
    """
    Get a warning message if image is very large.
    
    Args:
        image: Input image.
        
    Returns:
        Warning message string, or None if no warning needed.
    """
    if image is None:
        return None
    
    pixels = get_pixel_count(image)
    memory_mb = estimate_memory_usage(image)
    
    if pixels > DEFAULT_WARNING_THRESHOLD_PIXELS:
        megapixels = pixels / 1_000_000
        return (
            f"Large image detected ({megapixels:.1f} MP, ~{memory_mb:.0f} MB). "
            f"Processing may be slow. Consider using preview mode for adjustments."
        )
    
    return None
