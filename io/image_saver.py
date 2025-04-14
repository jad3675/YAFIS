# Export functionality using Pillow
import os
import numpy as np

# Attempt to import Pillow (PIL)
try:
    from PIL import Image, UnidentifiedImageError
    # Import ImageCms for potential future profile handling if needed
    # from PIL import ImageCms
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("[Image Saver Error] Pillow library not found. Image saving will not work. Install with 'pip install Pillow'")

# Placeholder for sRGB ICC profile bytes
# TODO: Load a standard sRGB ICC profile from a file or generate one.
# Example:
# try:
#     with open("path/to/sRGB_IEC61966-2-1_black_scaled.icc", "rb") as f:
#         SRGB_PROFILE_BYTES = f.read()
# except Exception as e:
#     print(f"[Image Saver Warning] Could not load sRGB profile: {e}. Images will be saved without an embedded profile.")
#     SRGB_PROFILE_BYTES = None
SRGB_PROFILE_BYTES = None # Set to None until profile is available

def save_image(image_rgb, file_path, quality=95, png_compression=3):
    """Saves the given RGB image to the specified file path using Pillow.

    Embeds an sRGB profile if available. Handles JPEG quality and PNG compression.

    Args:
        image_rgb (numpy.ndarray): The image to save (uint8 RGB format).
        file_path (str): The full path where the image should be saved,
                         including the desired file extension (e.g., .jpg, .png).
        quality (int): The quality setting for JPEG (1-100, higher is better).
        png_compression (int): Compression level for PNG (0-9, higher is more compressed).

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not PILLOW_AVAILABLE:
        print("[Image Saver Error] Cannot save image: Pillow library is not available.")
        return False

    if image_rgb is None or image_rgb.size == 0:
        print("[Image Saver Error] Cannot save an empty image.")
        return False

    if not isinstance(file_path, str) or not file_path:
        print("[Image Saver Error] Invalid file path provided for saving.")
        return False

    if image_rgb.dtype != np.uint8:
         print("[Image Saver Warning] Image data type is not uint8. Clipping and converting.")
         image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if len(image_rgb.shape) != 3 or image_rgb.shape[2] != 3:
        print("[Image Saver Error] Image must be in RGB format (3 channels) to save.")
        return False

    # Validate quality and compression types
    if not isinstance(quality, int):
        print(f"[Image Saver Error] Invalid type for quality parameter: expected int, got {type(quality)}. Value: {quality}")
        return False
    if not isinstance(png_compression, int):
        print(f"[Image Saver Error] Invalid type for png_compression parameter: expected int, got {type(png_compression)}. Value: {png_compression}")
        return False

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"[Image Saver Info] Created output directory: {output_dir}")
        except OSError as e:
            print(f"[Image Saver Error] Could not create directory '{output_dir}': {e}")
            return False

    try:
        # Convert NumPy array (RGB) to Pillow Image object
        img = Image.fromarray(image_rgb, 'RGB')

        # Prepare save options
        save_kwargs = {}
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.jpg', '.jpeg']:
            save_kwargs['quality'] = max(1, min(100, quality)) # Clamp quality 1-100 for Pillow JPEG
            save_kwargs['optimize'] = True # Try to optimize JPEG size
            save_kwargs['progressive'] = True # Use progressive JPEG for better web display
            # Embed sRGB profile if available (JPEG supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # TODO: Add EXIF preservation here if needed, e.g., save_kwargs['exif'] = exif_bytes
        elif ext == '.png':
            save_kwargs['compress_level'] = max(0, min(9, png_compression)) # Clamp 0-9
            # Embed sRGB profile if available (PNG supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # TODO: Add EXIF preservation (less common for PNG, might need specific chunks)
        elif ext in ['.tif', '.tiff']:
            # Embed sRGB profile if available (TIFF supports icc_profile)
            if SRGB_PROFILE_BYTES:
                save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
            # TODO: Add EXIF preservation
            # TODO: Add TIFF compression options (e.g., save_kwargs['compression'] = 'tiff_lzw')
            pass # Use Pillow defaults for TIFF for now
        elif ext == '.webp':
             save_kwargs['quality'] = max(0, min(100, quality)) # WebP quality 0-100
             # Embed sRGB profile if available (WebP supports icc_profile)
             if SRGB_PROFILE_BYTES:
                 save_kwargs['icc_profile'] = SRGB_PROFILE_BYTES
             # TODO: Add EXIF preservation
        # BMP and other formats might not support profiles/quality settings

        # Attempt to save the image
        img.save(file_path, **save_kwargs)
        print(f"Successfully saved image to: '{file_path}'")
        return True

    except FileNotFoundError:
        # Should be caught by directory check, but handle defensively
        print(f"Error: File path not found during save: '{file_path}'")
        return False
    except UnidentifiedImageError: # Should not happen on save, but include
         print(f"Error: Pillow could not determine save format for: '{file_path}'")
         return False
    except OSError as e:
        # Catch disk full, permissions errors, etc.
        print(f"Error: OS error saving image '{file_path}': {e}")
        return False
    except Exception as e:
        # Catch any other unexpected errors during saving
        print(f"Error saving image '{file_path}' with Pillow: {e}")
        return False
    finally:
        # Ensure image object is closed if necessary (though usually handled by save)
        try:
            if 'img' in locals() and hasattr(img, 'close'):
                img.close()
        except Exception as close_e:
             print(f"Error closing image object during save: {close_e}")


# TODO: Add metadata preservation if needed (requires external library like piexif or Pillow's EXIF handling)
# Example structure using Pillow's EXIF handling:
# from PIL.ExifTags import TAGS
# def get_exif_dict(pil_image):
#     exif_data = pil_image.getexif()
#     exif_dict = {}
#     if exif_data:
#         for k, v in exif_data.items():
#             if k in TAGS:
#                 exif_dict[TAGS[k]] = v
#     return exif_dict
#
# # In save_image:
# # exif_dict_to_save = get_exif_dict(original_pil_image) # Need original image's exif
# # if exif_dict_to_save:
# #    try:
# #        exif_bytes = piexif.dump(exif_dict_to_save) # Or use Pillow's internal mechanism if sufficient
# #        save_kwargs['exif'] = exif_bytes
# #    except Exception as e: print(f"Warning: Could not dump EXIF: {e}")