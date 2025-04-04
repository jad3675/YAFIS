# Image import functionality using Pillow
import os
import numpy as np

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, ImageOps, UnidentifiedImageError
    # ImageOps is needed for exif_transpose
    # UnidentifiedImageError is a specific Pillow error for bad formats
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    # Fallback message if Pillow isn't installed
    print("[Image Loader Error] Pillow library not found. Image loading will not work. Install with 'pip install Pillow'")
    # Define a dummy function if Pillow is absolutely required downstream
    # but in this case, the application likely won't start without it.

# Define the filter string for supported image formats for QFileDialog
# Pillow supports a wider range, but keep this consistent for now unless expanded
SUPPORTED_FORMATS_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp);;JPEG (*.jpg *.jpeg);;PNG (*.png);;TIFF (*.tif *.tiff);;Bitmap (*.bmp);;WebP (*.webp);;All Files (*)"

def load_image(file_path):
    """Loads an image from the specified file path using Pillow.

    Handles EXIF orientation automatically and attempts to read ICC profile info.

    Args:
        file_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image in RGB format (uint8), correctly oriented,
                       or None if loading fails or file not found.
    """
    if not PILLOW_AVAILABLE:
        print("[Image Loader Error] Cannot load image: Pillow library is not available.")
        return None

    if not isinstance(file_path, str) or not file_path:
        print("[Image Loader Error] Invalid file path provided.")
        return None

    if not os.path.isfile(file_path):
        print(f"[Image Loader Error] File not found at '{file_path}'")
        return None

    try:
        # Open image using Pillow
        img = Image.open(file_path)

        # --- Orientation Handling ---
        # Apply EXIF orientation using ImageOps.exif_transpose
        # This handles all orientation tags correctly in place (returns a new image object)
        img_oriented = ImageOps.exif_transpose(img)
        # Note: Pillow might issue warnings if EXIF data is corrupt, but usually proceeds.

        # --- Color Profile Info (Extraction only for now) ---
        icc_profile = img_oriented.info.get('icc_profile')
        if icc_profile:
            # TODO: Implement actual color management using the profile if needed.
            # For now, just log its presence. Requires libraries like littlecms (lcms2)
            # via Pillow's ImageCms module or external libraries.
            print(f"[Image Loader Info] Found embedded ICC profile ({len(icc_profile)} bytes). Profile is currently ignored.")
            pass
        else:
            # Assume sRGB if no profile - this is standard behavior for many apps
            # print("[Image Loader Info] No ICC profile found. Assuming sRGB.")
            pass

        # --- Ensure RGB Format ---
        # Convert to RGB if it's not already (e.g., grayscale, RGBA, palette)
        if img_oriented.mode != 'RGB':
            print(f"[Image Loader Info] Converting image from mode '{img_oriented.mode}' to 'RGB'.")
            img_rgb = img_oriented.convert('RGB')
        else:
            img_rgb = img_oriented

        # --- Convert to NumPy array ---
        # Convert the Pillow image object to a NumPy array (uint8, RGB order)
        image_np = np.array(img_rgb)

        if image_np.size == 0:
             print(f"Error: Loaded image is empty after processing: '{file_path}'")
             # Close the image file handle if Pillow keeps it open
             img.close()
             img_oriented.close()
             if img_rgb is not img_oriented: img_rgb.close()
             return None

        print(f"Successfully loaded and oriented image: '{file_path}'")

        # Close the image file handle opened by Pillow
        img.close()
        # Check if oriented/rgb created new objects that need closing (usually not, but safe)
        if img_oriented is not img: img_oriented.close()
        if img_rgb is not img_oriented: img_rgb.close()


        return image_np

    except UnidentifiedImageError:
        print(f"Error: Pillow could not identify image file format or file is corrupted: '{file_path}'")
        return None
    except FileNotFoundError:
        # Should be caught by os.path.isfile, but handle defensively
        print(f"Error: File not found (exception): '{file_path}'")
        return None
    except Exception as e:
        # Catch any other unexpected errors during loading or conversion
        print(f"Error loading image '{file_path}' with Pillow: {e}")
        # Ensure image file handle is closed in case of error during processing
        try:
            if 'img' in locals() and hasattr(img, 'close'): img.close()
            if 'img_oriented' in locals() and hasattr(img_oriented, 'close') and img_oriented is not img: img_oriented.close()
            if 'img_rgb' in locals() and hasattr(img_rgb, 'close') and img_rgb is not img_oriented: img_rgb.close()
        except Exception as close_e:
            print(f"Error closing image file during exception handling: {close_e}")
        return None

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