# Image import functionality using Pillow
import os
import io
import numpy as np
from typing import Optional, Tuple, Dict, Any

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, ImageOps, UnidentifiedImageError
    from PIL.ExifTags import TAGS, GPSTAGS, IFD

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.error(
        "Pillow library not found. Image loading will not work. Install with 'pip install Pillow'."
    )

# Attempt to import rawpy for RAW file support
try:
    import rawpy
    RAWPY_AVAILABLE = True
except ImportError:
    RAWPY_AVAILABLE = False
    logger.debug("rawpy not installed. RAW file support disabled. Install with 'pip install rawpy'.")

# Define the filter string for supported image formats for QFileDialog
# Pillow supports a wider range, but keep this consistent for now unless expanded
_STANDARD_FORMATS = "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"
_RAW_FORMATS = "*.dng *.cr2 *.cr3 *.nef *.arw *.orf *.rw2 *.pef *.raf *.srw"

if RAWPY_AVAILABLE:
    SUPPORTED_FORMATS_FILTER = (
        f"All Supported ({_STANDARD_FORMATS} {_RAW_FORMATS});;"
        f"Images ({_STANDARD_FORMATS});;"
        f"RAW Files ({_RAW_FORMATS});;"
        "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;All Files (*)"
    )
else:
    SUPPORTED_FORMATS_FILTER = (
        f"Images ({_STANDARD_FORMATS});;"
        "JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;Bitmap (*.bmp);;WebP (*.webp);;All Files (*)"
    )

# RAW file extensions for detection
RAW_EXTENSIONS = {'.dng', '.cr2', '.cr3', '.nef', '.arw', '.orf', '.rw2', '.pef', '.raf', '.srw'}


class ImageMetadata:
    """Container for image metadata (EXIF, ICC profile, etc.)."""
    
    def __init__(self):
        self.exif_data: Optional[bytes] = None
        self.icc_profile: Optional[bytes] = None
        self.original_mode: Optional[str] = None
        self.file_size: int = 0
        self.exif_dict: Dict[str, Any] = {}
    
    def has_exif(self) -> bool:
        """Check if EXIF data is available."""
        return self.exif_data is not None and len(self.exif_data) > 0
    
    def has_icc_profile(self) -> bool:
        """Check if ICC profile is available."""
        return self.icc_profile is not None and len(self.icc_profile) > 0


def extract_metadata(img: "Image.Image", file_path: str) -> ImageMetadata:
    """
    Extract metadata from a Pillow image.
    
    Args:
        img: Pillow Image object.
        file_path: Path to the image file.
        
    Returns:
        ImageMetadata object containing extracted metadata.
    """
    metadata = ImageMetadata()
    
    try:
        metadata.file_size = os.path.getsize(file_path)
    except OSError:
        metadata.file_size = 0
    
    metadata.original_mode = img.mode
    
    # Extract ICC profile
    metadata.icc_profile = img.info.get("icc_profile")
    
    # Extract EXIF data
    try:
        exif = img.getexif()
        if exif:
            # Store raw EXIF bytes for preservation
            metadata.exif_data = exif.tobytes()
            
            # Also parse into dict for easy access
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                # Handle nested IFD data (like GPS)
                if tag_id in IFD.__members__.values():
                    try:
                        ifd_data = exif.get_ifd(tag_id)
                        if ifd_data:
                            metadata.exif_dict[tag_name] = dict(ifd_data)
                    except Exception:
                        pass
                else:
                    # Convert bytes to string for JSON serialization
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore')
                        except Exception:
                            value = str(value)
                    metadata.exif_dict[tag_name] = value
    except Exception:
        logger.debug("Could not extract EXIF data from %s", file_path)
    
    return metadata


def load_raw_image(file_path: str) -> Tuple[Optional[np.ndarray], Optional[str], Optional[int], Optional[ImageMetadata]]:
    """
    Load a RAW image file using rawpy.
    
    Args:
        file_path: Path to the RAW file.
        
    Returns:
        Tuple of (image_array, original_mode, file_size, metadata).
        Returns (None, None, None, None) on failure.
    """
    if not RAWPY_AVAILABLE:
        logger.error("Cannot load RAW file: rawpy library is not available.")
        return None, None, None, None
    
    try:
        file_size = os.path.getsize(file_path)
        
        with rawpy.imread(file_path) as raw:
            # Postprocess with sensible defaults for film scanning
            rgb = raw.postprocess(
                use_camera_wb=True,      # Use camera white balance
                output_bps=8,            # 8-bit output
                no_auto_bright=False,    # Allow auto brightness
                output_color=rawpy.ColorSpace.sRGB,  # sRGB output
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # High quality demosaic
            )
        
        metadata = ImageMetadata()
        metadata.file_size = file_size
        metadata.original_mode = "RAW"
        
        # Try to extract EXIF from RAW file using Pillow (works for some formats)
        try:
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    metadata.exif_data = exif.tobytes()
        except Exception:
            pass
        
        logger.info("Successfully loaded RAW image: '%s'", file_path)
        return rgb, "RAW", file_size, metadata
        
    except Exception:
        logger.exception("Error loading RAW image '%s'", file_path)
        return None, None, None, None

def load_image(file_path: str, return_metadata: bool = False):
    """Loads an image from the specified file path using Pillow (or rawpy for RAW files).

    Handles EXIF orientation automatically and attempts to read ICC profile info.

    Args:
        file_path (str): The path to the image file.
        return_metadata (bool): If True, returns ImageMetadata object instead of tuple.

    Returns:
        If return_metadata is False (default):
            tuple: (numpy.ndarray, str, int) - image in RGB format, original mode, file size.
        If return_metadata is True:
            tuple: (numpy.ndarray, ImageMetadata) - image in RGB format, metadata object.
        Returns (None, None, None) or (None, None) on failure.
    """
    if not PILLOW_AVAILABLE:
        logger.error("Cannot load image: Pillow library is not available.")
        return (None, None) if return_metadata else (None, None, None)

    if not isinstance(file_path, str) or not file_path:
        logger.error("Invalid file path provided.")
        return (None, None) if return_metadata else (None, None, None)

    if not os.path.isfile(file_path):
        logger.error("File not found at '%s'", file_path)
        return (None, None) if return_metadata else (None, None, None)

    # Check if this is a RAW file
    ext = os.path.splitext(file_path)[1].lower()
    if ext in RAW_EXTENSIONS:
        if RAWPY_AVAILABLE:
            image_np, original_mode, file_size, metadata = load_raw_image(file_path)
            if image_np is not None:
                if return_metadata:
                    return image_np, metadata
                return image_np, original_mode, file_size
        else:
            logger.error("RAW file detected but rawpy is not installed. Install with 'pip install rawpy'.")
        return (None, None) if return_metadata else (None, None, None)

    try:
        # Get file size first
        file_size = os.path.getsize(file_path)

        # Open image using Pillow
        img = Image.open(file_path)

        # Extract metadata before any transformations
        metadata = extract_metadata(img, file_path)

        # Store original mode *before* potential conversion
        original_mode = img.mode

        # --- Orientation Handling ---
        # Apply EXIF orientation using ImageOps.exif_transpose
        # This handles all orientation tags correctly in place (returns a new image object)
        img_oriented = ImageOps.exif_transpose(img)
        # Note: Pillow might issue warnings if EXIF data is corrupt, but usually proceeds.

        # --- Color Profile Handling ---
        icc_profile = img_oriented.info.get("icc_profile")
        apply_icc = False
        try:
            from ..config import settings as app_settings
            apply_icc = bool(app_settings.UI_DEFAULTS.get("apply_embedded_icc_profile", False))
        except Exception:
            # Don't fail image loading due to settings import issues.
            logger.exception("Failed to read ICC apply setting; defaulting to disabled.")

        if icc_profile:
            if apply_icc:
                try:
                    # Convert from embedded profile to sRGB for consistent processing/display.
                    from PIL import ImageCms
                    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
                    dst_profile = ImageCms.createProfile("sRGB")
                    img_oriented = ImageCms.profileToProfile(
                        img_oriented,
                        src_profile,
                        dst_profile,
                        outputMode="RGB",
                    )
                    logger.info(
                        "Applied embedded ICC profile (%s bytes) and converted to sRGB.",
                        len(icc_profile),
                    )
                except Exception:
                    logger.exception(
                        "Failed to apply embedded ICC profile (%s bytes); continuing without ICC conversion.",
                        len(icc_profile),
                    )
            else:
                logger.info(
                    "Found embedded ICC profile (%s bytes). Profile is currently ignored.",
                    len(icc_profile),
                )

        # --- Ensure RGB Format ---
        if img_oriented.mode != "RGB":
            logger.info("Converting image from mode '%s' to 'RGB'.", img_oriented.mode)
            img_rgb = img_oriented.convert("RGB")
        else:
            img_rgb = img_oriented

        # --- Convert to NumPy array ---
        # Convert the Pillow image object to a NumPy array (uint8, RGB order)
        image_np = np.array(img_rgb)

        if image_np.size == 0:
            logger.error("Loaded image is empty after processing: '%s'", file_path)
            img.close()
            img_oriented.close()
            if img_rgb is not img_oriented:
                img_rgb.close()
            return (None, None) if return_metadata else (None, None, None)

        logger.info("Successfully loaded image: '%s'", file_path)

        # Close the image file handle opened by Pillow
        img.close()
        # Check if oriented/rgb created new objects that need closing (usually not, but safe)
        if img_oriented is not img: img_oriented.close()
        if img_rgb is not img_oriented: img_rgb.close()

        if return_metadata:
            return image_np, metadata
        return image_np, original_mode, file_size

    except UnidentifiedImageError:
        logger.error(
            "Pillow could not identify image file format or file is corrupted: '%s'",
            file_path,
        )
        return (None, None) if return_metadata else (None, None, None)
    except FileNotFoundError:
        logger.error("File not found (exception): '%s'", file_path)
        return (None, None) if return_metadata else (None, None, None)
    except Exception:
        logger.exception("Error loading image '%s' with Pillow.", file_path)
        # Ensure image file handle is closed in case of error during processing
        try:
            if "img" in locals() and hasattr(img, "close"):
                img.close()
            if "img_oriented" in locals() and hasattr(img_oriented, "close") and img_oriented is not img:
                img_oriented.close()
            if "img_rgb" in locals() and hasattr(img_rgb, "close") and img_rgb is not img_oriented:
                img_rgb.close()
        except Exception:
            logger.exception("Error closing image file during exception handling.")
        return (None, None) if return_metadata else (None, None, None)

def is_raw_file(file_path: str) -> bool:
    """Check if a file is a RAW image format."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in RAW_EXTENSIONS


def is_raw_supported() -> bool:
    """Check if RAW file support is available."""
    return RAWPY_AVAILABLE