# Image import functionality using Pillow
import os
import io
import numpy as np

from negative_converter.utils.logger import get_logger

logger = get_logger(__name__)

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, ImageOps, UnidentifiedImageError

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    logger.error(
        "Pillow library not found. Image loading will not work. Install with 'pip install Pillow'."
    )

# Define the filter string for supported image formats for QFileDialog
# Pillow supports a wider range, but keep this consistent for now unless expanded
SUPPORTED_FORMATS_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;Bitmap (*.bmp);;WebP (*.webp);;All Files (*)"

def load_image(file_path):
    """Loads an image from the specified file path using Pillow.

    Handles EXIF orientation automatically and attempts to read ICC profile info.

    Args:
        file_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing:
               - numpy.ndarray: The loaded image in RGB format (uint8), correctly oriented.
               - str: The original image mode (e.g., 'RGB', 'L', 'RGBA').
               - int: The file size in bytes.
               Returns (None, None, None) if loading fails or file not found.
    """
    if not PILLOW_AVAILABLE:
        logger.error("Cannot load image: Pillow library is not available.")
        return None, None, None

    if not isinstance(file_path, str) or not file_path:
        logger.error("Invalid file path provided.")
        return None, None, None

    if not os.path.isfile(file_path):
        logger.error("File not found at '%s'", file_path)
        return None, None, None

    try:
        # Get file size first
        file_size = os.path.getsize(file_path)

        # Open image using Pillow
        img = Image.open(file_path)

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
            return None, None, None

        logger.info("Successfully loaded image: '%s'", file_path)

        # Close the image file handle opened by Pillow
        img.close()
        # Check if oriented/rgb created new objects that need closing (usually not, but safe)
        if img_oriented is not img: img_oriented.close()
        if img_rgb is not img_oriented: img_rgb.close()


        return image_np, original_mode, file_size

    except UnidentifiedImageError:
        logger.error(
            "Pillow could not identify image file format or file is corrupted: '%s'",
            file_path,
        )
        return None, None, None
    except FileNotFoundError:
        logger.error("File not found (exception): '%s'", file_path)
        return None, None, None
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
        return None, None, None

# TODO: Add support for RAW files using rawpy if needed (requires rawpy library)
# Example structure:
# import rawpy
# def load_raw_image(file_path):
#     try:
#         with rawpy.imread(file_path) as raw:
#             # Postprocess converts to RGB, applies demosaicing, etc.
#             # Options like gamma, no_auto_bright can be set
#             rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
#         return rgb # Returns uint8 RGB numpy array
#     except Exception as e:
#         print(f"Error loading RAW image {file_path}: {e}")
#         return None

# Note: Pillow can read some EXIF data itself, potentially removing the need
# for a separate exifread library if only basic tags are needed.
# However, exifread might provide more comprehensive tag access.