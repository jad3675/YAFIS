import numpy as np
import cv2

# Use centralized GPU detection and logger
# Imports from within the same package can be relative
from .gpu import GPU_ENABLED, xp, cp_module, is_cupy_backend
from .logger import get_logger
# No changes needed here, imports are already correct relative to utils

logger = get_logger(__name__)
cp = cp_module # Assign cp_module to cp for compatibility if needed locally, or just use xp

def apply_curve(image_channel, curve_points):
    """
    Applies a tone curve defined by points to a single image channel.

    Handles both uint8 (using cv2.LUT or interpolation) and float32 (using interpolation).
    Note: If input is uint8 CuPy array, it's converted to NumPy for cv2.LUT and a NumPy array is returned.

    Args:
        image_channel: NumPy or CuPy array representing a single image channel.
                       Expected range: 0-255 for both uint8 and float32.
        curve_points: List or NumPy array of [x, y] points defining the curve.
                      Points should be within the 0-255 range.

    Returns:
        NumPy or CuPy array (matching input type) with the curve applied.
        Returns the original channel if curve_points are invalid or empty,
        if the input channel is empty, or if the dtype is unsupported.
    """
    if image_channel is None or image_channel.size == 0:
        logger.warning("Input channel is empty.")
        return image_channel

    # Determine if input is CuPy array (only possible if CuPy backend is active)
    is_cupy_input = False
    if is_cupy_backend() and cp is not None:
        try:
            is_cupy_input = cp.get_array_module(image_channel) == cp
        except Exception:
            is_cupy_input = False

    # Validate and prepare curve points (using NumPy for simplicity as points array is small)
    if curve_points is None:
        logger.warning("curve_points is None. Returning original channel.")
        return image_channel
    try:
        points_np = np.array(sorted(curve_points))
        if points_np.ndim != 2 or points_np.shape[1] != 2 or points_np.shape[0] == 0:
            raise ValueError("Invalid shape or empty.")
    except Exception as e:
        logger.warning(f"Invalid curve_points: {e}. Returning original channel.")
        return image_channel

    # Ensure curve spans 0-255
    if points_np[0, 0] > 0:
        # Use the y-value of the first point for the start to avoid sudden jump if y!=0
        points_np = np.vstack(([0, points_np[0, 1]], points_np))
    if points_np[-1, 0] < 255:
         # Use the y-value of the last point for the end
        points_np = np.vstack((points_np, [255, points_np[-1, 1]]))

    # Use the appropriate array module based on input
    arr_module = cp if is_cupy_input else np

    # Generate base x-values for LUT (0-255)
    lut_x = arr_module.arange(256, dtype=arr_module.float32)

    # Generate float LUT y-values using interpolation
    xp_curve = arr_module.asarray(points_np[:, 0], dtype=arr_module.float32)
    fp_curve = arr_module.asarray(points_np[:, 1], dtype=arr_module.float32)
    lut_y_float = arr_module.interp(lut_x, xp_curve, fp_curve)
    # Clip LUT values to ensure they are within the valid 0-255 range
    lut_y_float = arr_module.clip(lut_y_float, 0, 255)

    # Apply based on input dtype
    if image_channel.dtype == np.uint8:
        if is_cupy_input:
            # cv2.LUT requires NumPy array. Convert CuPy uint8 input to NumPy.
            logger.warning("uint8 input was CuPy array, converting to NumPy for cv2.LUT.")
            image_channel_np = cp.asnumpy(image_channel)
            # Convert float LUT (which is CuPy array) to NumPy uint8
            lut_uint8 = cp.asnumpy(lut_y_float).astype(np.uint8)
            result = cv2.LUT(image_channel_np, lut_uint8)
            return result
        else:
            # Input is NumPy uint8
            lut_uint8 = lut_y_float.astype(np.uint8)
            return cv2.LUT(image_channel, lut_uint8)

    elif image_channel.dtype == np.float32:
        # Apply LUT using interpolation for float32 input (works for NumPy/CuPy)
        result_float = arr_module.interp(image_channel, lut_x, lut_y_float)
        return result_float

    else:
        logger.warning(f"Unsupported input dtype: {image_channel.dtype}. Returning original channel.")
        return image_channel