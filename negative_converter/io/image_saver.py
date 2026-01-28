# Export functionality using Pillow
import os
import numpy as np
from typing import Optional, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, UnidentifiedImageError
    from PIL.ExifTags import TAGS

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.error("Pillow library not found. Image saving will not work. Install with 'pip install Pillow'.")

# Import ImageMetadata for type hints
try:
    from .image_loader import ImageMetadata
except ImportError:
    ImageMetadata = None  # type: ignore

# Placeholder for sRGB ICC profile bytes
# NOTE: Keeping this as None for now (YAGNI). If profile embedding is required,
# add a small bundled ICC file and load it here.
SRGB_PROFILE_BYTES = None

def save_image(
    image_rgb: np.ndarray,
    file_path: str,
    quality: int = 95,
    png_compression: int = 3,
    metadata: Optional["ImageMetadata"] = None,
    preserve_exif: bool = True,
) -> bool:
    """Saves the given RGB image to the specified file path using Pillow.

    Embeds an sRGB profile if available. Handles JPEG quality and PNG compression.
    Optionally preserves EXIF metadata from the original image.

    Args:
        image_rgb (numpy.ndarray): The image to save (uint8 RGB format).
        file_path (str): The full path where the image should be saved,
                         including the desired file extension (e.g., .jpg, .png).
        quality (int): The quality setting for JPEG (1-100, higher is better).
        png_compression (int): Compression level for PNG (0-9, higher is more compressed).
        metadata (ImageMetadata): Optional metadata object from image loading.
        preserve_exif (bool): If True and metadata is provided, preserve EXIF data.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not PILLOW_AVAILABLE:
        logger.error("Cannot save image: Pillow library is not available.")
        return False

    if image_rgb is None or image_rgb.size == 0:
        logger.error("Cannot save an empty image.")
        return False

    if not isinstance(file_path, str) or not file_path:
        logger.error("Invalid file path provided for saving.")
        return False

    if image_rgb.dtype != np.uint8:
        logger.warning("Image data type is not uint8. Clipping and converting.")
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        logger.error("Image must be in RGB format (3 channels) to save.")
        return False

    # Validate quality and compression types
    if not isinstance(quality, int):
        logger.error(
            "Invalid type for quality parameter: expected int, got %s. Value=%s",
            type(quality),
            quality,
        )
        return False
    if not isinstance(png_compression, int):
        logger.error(
            "Invalid type for png_compression parameter: expected int, got %s. Value=%s",
            type(png_compression),
            png_compression,
        )
        return False

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.info("Created output directory: %s", output_dir)
        except OSError:
            logger.exception("Could not create directory '%s'", output_dir)
            return False

    try:
        # Convert NumPy array (RGB) to Pillow Image object
        img = Image.fromarray(image_rgb, 'RGB')

        # Prepare save options
        save_kwargs = {}
        ext = os.path.splitext(file_path)[1].lower()

        # Prepare EXIF data if available and preservation is requested
        exif_bytes = None
        if preserve_exif and metadata is not None and metadata.has_exif():
            try:
                exif_bytes = metadata.exif_data
                logger.debug("EXIF data available for preservation (%d bytes)", len(exif_bytes))
            except Exception:
                logger.debug("Could not prepare EXIF data for preservation")
                exif_bytes = None

        if ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = max(1, min(100, quality))  # Clamp quality 1-100 for Pillow JPEG
            save_kwargs['optimize'] = True  # Try to optimize JPEG size
            save_kwargs['progressive'] = True  # Use progressive JPEG for better web display
            # Embed sRGB profile if available (JPEG supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # Preserve EXIF data
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes
                logger.debug("Preserving EXIF data in JPEG output")
        elif ext == '.png':
            save_kwargs['compress_level'] = max(0, min(9, png_compression))  # Clamp 0-9
            # Embed sRGB profile if available (PNG supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # PNG doesn't natively support EXIF in the same way, but Pillow can embed it
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes
                logger.debug("Preserving EXIF data in PNG output")
        elif ext in ['.tif', '.tiff']:
            # Embed sRGB profile if available (TIFF supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # TIFF supports EXIF
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes
                logger.debug("Preserving EXIF data in TIFF output")
            # Add LZW compression for smaller file sizes
            save_kwargs['compression'] = 'tiff_lzw'
        elif ext == '.webp':
            save_kwargs['quality'] = max(0, min(100, quality))  # WebP quality 0-100
            # Embed sRGB profile if available (WebP supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # WebP supports EXIF
            if exif_bytes:
                save_kwargs['exif'] = exif_bytes
                logger.debug("Preserving EXIF data in WebP output")
        # BMP and other formats might not support profiles/quality settings

        # Attempt to save the image
        img.save(file_path, **save_kwargs)
        logger.info("Successfully saved image to: '%s'", file_path)
        return True

    except FileNotFoundError:
        logger.error("File path not found during save: '%s'", file_path)
        return False
    except UnidentifiedImageError:
        logger.error("Pillow could not determine save format for: '%s'", file_path)
        return False
    except OSError:
        logger.exception("OS error saving image '%s'", file_path)
        return False
    except Exception:
        logger.exception("Unexpected error saving image '%s' with Pillow.", file_path)
        return False
    finally:
        try:
            if "img" in locals() and hasattr(img, "close"):
                img.close()
        except Exception:
            logger.exception("Error closing image object during save.")


def copy_metadata(source_path: str, dest_path: str) -> bool:
    """
    Copy metadata from source image to destination image.
    
    Useful when you want to preserve metadata after processing.
    
    Args:
        source_path: Path to the source image with metadata.
        dest_path: Path to the destination image to update.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    if not PILLOW_AVAILABLE:
        logger.error("Cannot copy metadata: Pillow library is not available.")
        return False
    
    try:
        # Load source image to get metadata
        with Image.open(source_path) as src_img:
            exif = src_img.getexif()
            if not exif:
                logger.debug("No EXIF data in source image: %s", source_path)
                return True  # Not an error, just no metadata to copy
            
            exif_bytes = exif.tobytes()
        
        # Load destination image and save with metadata
        with Image.open(dest_path) as dest_img:
            # Determine format from extension
            ext = os.path.splitext(dest_path)[1].lower()
            save_kwargs = {'exif': exif_bytes}
            
            if ext in ['.jpg', '.jpeg']:
                save_kwargs['quality'] = 95
            
            dest_img.save(dest_path, **save_kwargs)
        
        logger.info("Copied metadata from %s to %s", source_path, dest_path)
        return True
        
    except Exception:
        logger.exception("Failed to copy metadata from %s to %s", source_path, dest_path)
        return False