import numpy as np
import cv2

# Try importing CuPy, but don't fail if it's not there
try:
    import cupy as cp
    # Check if CUDA is available and a device is accessible
    try:
        cp.cuda.runtime.getDeviceCount()
        _GPU_ENABLED_UTIL = True
    except cp.cuda.runtime.CUDARuntimeError:
        _GPU_ENABLED_UTIL = False
except ImportError:
    cp = None # Define cp as None if import fails
    _GPU_ENABLED_UTIL = False

def apply_curve(image_channel, curve_points):
    """
    Applies a tone curve defined by points to a single image channel.

    Handles both uint8 (NumPy only, using cv2.LUT) and float32 (NumPy/CuPy, using interpolation).
    Assumes input uint8 arrays are NumPy arrays.

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
        print("[apply_curve Warning] Input channel is empty.")
        return image_channel

    # Determine backend (xp) and if input is CuPy array
    is_cupy_input = cp and isinstance(image_channel, cp.ndarray)
    xp = cp if is_cupy_input else np

    # Validate and prepare curve points (using NumPy for simplicity as points array is small)
    if curve_points is None:
        print("[apply_curve Warning] curve_points is None. Returning original channel.")
        return image_channel
    try:
        points_np = np.array(sorted(curve_points))
        if points_np.ndim != 2 or points_np.shape[1] != 2 or points_np.shape[0] == 0:
            raise ValueError("Invalid shape or empty.")
    except Exception as e:
        print(f"[apply_curve Warning] Invalid curve_points: {e}. Returning original channel.")
        return image_channel

    # Ensure curve spans 0-255
    if points_np[0, 0] > 0:
        # Use the y-value of the first point for the start to avoid sudden jump if y!=0
        points_np = np.vstack(([0, points_np[0, 1]], points_np))
    if points_np[-1, 0] < 255:
         # Use the y-value of the last point for the end
        points_np = np.vstack((points_np, [255, points_np[-1, 1]]))

    # Generate base x-values for LUT (0-255) using the determined backend
    lut_x = xp.arange(256, dtype=xp.float32)

    # Generate float LUT y-values using interpolation on the determined backend
    # Transfer curve points to GPU if needed for interpolation
    xp_curve = xp.asarray(points_np[:, 0], dtype=xp.float32)
    fp_curve = xp.asarray(points_np[:, 1], dtype=xp.float32)
    lut_y_float = xp.interp(lut_x, xp_curve, fp_curve)
    # Clip LUT values to ensure they are within the valid 0-255 range
    lut_y_float = xp.clip(lut_y_float, 0, 255)

    # Apply based on input dtype
    if image_channel.dtype == np.uint8:
        if is_cupy_input:
            # cv2.LUT requires NumPy array. Convert CuPy uint8 input to NumPy.
            print("[apply_curve Warning] uint8 input was CuPy array, converting to NumPy for cv2.LUT.")
            image_channel_np = cp.asnumpy(image_channel)
            # Convert float LUT (which might be CuPy array) to NumPy uint8
            lut_uint8 = xp.asnumpy(lut_y_float).astype(np.uint8) if is_cupy_input else lut_y_float.astype(np.uint8)
            result = cv2.LUT(image_channel_np, lut_uint8)
            # Convert result back to CuPy? Or decide utility always returns NumPy for uint8?
            # For now, let's return NumPy for uint8 input, consistent with cv2.LUT output.
            return result
        else:
            # Input is NumPy uint8
            lut_uint8 = lut_y_float.astype(np.uint8) # lut_y_float is already NumPy
            return cv2.LUT(image_channel, lut_uint8)

    elif image_channel.dtype == np.float32:
        # Apply LUT using interpolation for float32 input (works for NumPy/CuPy)
        # We interpolate the input channel's values using the LUT definition
        # lut_x defines the "indices" (0-255), lut_y_float defines the "values" at those indices
        result_float = xp.interp(image_channel, lut_x, lut_y_float)
        # Clipping float result might be desired depending on context, but leave it unclamped for now
        # return xp.clip(result_float, 0, 255)
        return result_float

    else:
        print(f"[apply_curve Warning] Unsupported input dtype: {image_channel.dtype}. Returning original channel.")
        return image_channel