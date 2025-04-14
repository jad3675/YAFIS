# Negative to positive algorithm
import numpy as np
import cv2
# Import GPU utility which handles detection and provides xp (array module)
from negative_converter.utils.gpu import GPU_ENABLED, xp

# Conditionally import cupy for specific exception handling if available
try:
    import cupy as cp_module
except ImportError:
    cp_module = None # cupy not available

# Import the new utility function
import sys
import os
# Standard relative import assuming package structure is respected
from negative_converter.utils.imaging import apply_curve
from negative_converter.config.settings import CONVERSION_DEFAULTS
from negative_converter.utils.logger import get_logger

logger = get_logger(__name__)
def detect_orange_mask(negative_image):
    """
    Detect the orange mask in a C-41 negative.
    Assumes input is a NumPy array (uint8 or float).
    """
    if negative_image is None or negative_image.size == 0:
        logger.warning("detect_orange_mask received empty image.")
        return np.array([0, 0, 0], dtype=np.float32) # Return default black
    if len(negative_image.shape) < 2 or negative_image.shape[0] < 10 or negative_image.shape[1] < 10:
         logger.warning(f"detect_orange_mask received image too small for corner sampling ({negative_image.shape}). Returning default.")
         return np.array([0, 0, 0], dtype=np.float32) # Return default black
    if len(negative_image.shape) == 3 and negative_image.shape[2] != 3:
         logger.warning(f"detect_orange_mask expects 3-channel image if color, got shape {negative_image.shape}. Proceeding anyway.")
         # Allow grayscale or other channel counts for now, mean will work

    h, w = negative_image.shape[:2]
    s = CONVERSION_DEFAULTS.get('mask_sample_size', 10) # Get sample size from config
    s_half = s // 2

    # Ensure sample size isn't too large for image dimensions
    if h < s or w < s:
        logger.warning(f"Image dimensions ({h}x{w}) too small for sample size {s}. Using smaller samples.")
        s = min(h, w)
        s_half = s // 2
        if s == 0: # Handle extremely small images
             logger.error("Image too small for any sampling.")
             return np.array([0, 0, 0], dtype=np.float32)

    # Define sample areas (corners and edge midpoints)
    sample_coords = [
        (0, s, 0, s),                     # Top-left corner
        (0, s, w - s, w),                 # Top-right corner
        (h - s, h, 0, s),                 # Bottom-left corner
        (h - s, h, w - s, w),             # Bottom-right corner
        (0, s, w // 2 - s_half, w // 2 + s_half), # Top-mid
        (h - s, h, w // 2 - s_half, w // 2 + s_half), # Bottom-mid
        (h // 2 - s_half, h // 2 + s_half, 0, s), # Left-mid
        (h // 2 - s_half, h // 2 + s_half, w - s, w)  # Right-mid
    ]

    samples = []
    for r1, r2, c1, c2 in sample_coords:
        # Ensure coordinates are valid after potential adjustments
        r1, r2 = max(0, r1), min(h, r2)
        c1, c2 = max(0, c1), min(w, c2)
        if r1 < r2 and c1 < c2: # Check if sample area is valid
             samples.append(negative_image[r1:r2, c1:c2])

    # Calculate average color of samples
    # Calculate average color of each sample area
    sample_means = []
    for sample in samples:
        if sample.size > 0:
            # Calculate mean color for this sample (handle grayscale if needed)
            if len(sample.shape) == 3:
                mean_color = np.mean(sample, axis=(0, 1))
            else: # Grayscale or other
                mean_val = np.mean(sample)
                mean_color = np.array([mean_val, mean_val, mean_val]) # Represent as gray
            sample_means.append(mean_color)

    if not sample_means:
        logger.error("No valid sample areas found for mask detection.")
        return np.array([0, 0, 0], dtype=np.float32)

    # Calculate overall average mask color
    mask_color = np.mean(sample_means, axis=0).astype(np.float32)

    # Check variance between sample means (optional robustness check)
    if len(sample_means) > 1:
        sample_means_np = np.array(sample_means)
        # Calculate std dev across samples for each channel, then average the std devs
        std_dev = np.mean(np.std(sample_means_np, axis=0))
        # Define a threshold (could be configurable)
        VARIANCE_THRESHOLD = 25.0 # Example threshold for std dev in 0-255 range
        if std_dev > VARIANCE_THRESHOLD:
            logger.warning(f"High variance ({std_dev:.2f}) detected between mask sample areas. Mask color might be inaccurate.")
        # Old averaging logic removed. The correct mask_color is calculated
        # using np.mean(sample_means, axis=0) above.

    return mask_color

# remove_orange_mask function removed as it's superseded by the logic within convert()


def apply_color_correction(image, correction_matrix=None):
    """
    Apply color correction matrix to normalize colors (GPU or CPU).
    Expects image to be uint8 RGB.
    """
    if image is None or image.size == 0:
        logger.warning("apply_color_correction received empty image.")
        return image
    if len(image.shape) != 3 or image.shape[2] != 3:
        logger.warning(f"apply_color_correction expects 3-channel image, got shape {image.shape}. Returning original.")
        return image
    if image.dtype != np.uint8:
         logger.warning(f"apply_color_correction expects uint8 image, got {image.dtype}. Attempting conversion.")
         image = np.clip(image, 0, 255).astype(np.uint8)

    # Define default matrix (as NumPy initially)
    if correction_matrix is None:
        correction_matrix_np = np.array([
            [1.50, -0.20, -0.30],
            [-0.30, 1.60, -0.30],
            [-0.20, -0.20, 1.40]
        ], dtype=np.float32)
    else:
        # Ensure input is a NumPy array if provided
        correction_matrix_np = np.asarray(correction_matrix, dtype=np.float32)

    # Use xp imported from gpu util
    if GPU_ENABLED:
        try:
            # Transfer data to GPU
            img_gpu = xp.asarray(image, dtype=xp.float32)
            matrix_gpu = xp.asarray(correction_matrix_np) # Transfer the NumPy matrix

            # Reshape image for matrix multiplication (h*w, 3)
            h, w, c = img_gpu.shape
            flat_image_gpu = img_gpu.reshape(-1, 3)

            # Apply matrix: result = flat_image @ correction_matrix.T
            corrected_flat_gpu = xp.dot(flat_image_gpu, matrix_gpu.T)

            # Clip and reshape back
            corrected_gpu = xp.clip(corrected_flat_gpu, 0, 255)
            corrected_gpu = corrected_gpu.reshape(h, w, 3)

            # Transfer result back to CPU
            return xp.asnumpy(corrected_gpu).astype(np.uint8)

        # Catch specific CUDA error only if cupy was successfully imported
        except cp_module.cuda.runtime.CUDARuntimeError as e:
             logger.error(f"CUDA runtime error during GPU color correction: {e}")
             logger.info("Falling back to CPU for color correction.")
             # Fall through to CPU implementation if GPU fails
        except Exception as e: # Catch other potential GPU errors
             logger.error(f"Unexpected error during GPU color correction: {e}")
             logger.info("Falling back to CPU for color correction.")
             # Fall through to CPU implementation

    # CPU Implementation (Original logic or fallback)
    # CPU Implementation (Original logic or fallback)
    # xp is already numpy if GPU_ENABLED is False
    # Ensure image is float32 for matrix multiplication
    img_float = image.astype(xp.float32)
    # Use the prepared NumPy matrix
    correction_matrix = correction_matrix_np

    # Reshape image for matrix multiplication (h*w, 3)
    h, w, c = img_float.shape
    flat_image = img_float.reshape(-1, 3)

    # Apply matrix: result = flat_image @ correction_matrix.T
    corrected_flat = xp.dot(flat_image, correction_matrix.T)

    # Clip and reshape back
    corrected = xp.clip(corrected_flat, 0, 255)
    return corrected.reshape(h, w, 3).astype(xp.uint8)


class NegativeConverter:
    def __init__(self, film_profile="C41"):
        self.film_profile = film_profile
        # TODO: Load appropriate conversion parameters based on film type if needed
        # Use defaults from config, could be overridden by film_profile later
        self.params = CONVERSION_DEFAULTS.copy()

    def convert(self, image, progress_callback=None, override_mask_classification=None):
        """
        Convert negative to positive using professional film processing techniques.

        Args:
            image: The input negative image (NumPy array).
            progress_callback (callable, optional): A function to call with (current_step, total_steps). Defaults to None.
            override_mask_classification (str, optional): If provided, skips auto-detection and uses this classification.
                                                        Valid values: "Likely C-41", "Clear/Near Clear", "Unknown/Other". Defaults to None.

        Returns:
            tuple: A tuple containing:
                   - numpy.ndarray: The converted positive image (uint8).
                   - str: The detected or overridden mask classification.
                   Returns (None, "Error") on failure.
        """
        total_steps = 6 # Define the total number of major steps for progress reporting
        def report_progress(step):
            if progress_callback:
                try:
                    progress_callback(step, total_steps)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

        report_progress(0) # Step 0: Start
        logger.debug(f"Starting conversion for image shape: {image.shape if image is not None else 'None'}, dtype: {image.dtype if image is not None else 'None'}")
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3-channel (RGB) image")
        logger.debug("Input validation passed.")

        # Backend (xp) is determined by the imported utils.gpu module

        # --- STEP 0: Initial Conversion to Float32 (CPU/GPU) ---
        # Convert the input uint8 image to float32 ONCE.
        # If GPU is enabled, this will transfer to GPU memory.
        try:
            img_float = xp.asarray(image, dtype=xp.float32)
            logger.debug(f"Initial conversion to {('CuPy' if GPU_ENABLED else 'NumPy')} float32 complete.")
        except Exception as e:
            logger.error(f"Failed initial conversion/transfer to {('GPU' if GPU_ENABLED else 'CPU')} float32: {e}")
            if GPU_ENABLED:
                logger.info("Falling back to CPU for the entire conversion.")
                # xp is already numpy if GPU_ENABLED is False, this case shouldn't be reachable
                # If GPU failed during asarray, xp should still be cp, but we might want to force CPU
                # However, the gpu util aims to provide a stable xp, so we trust it.
                # If asarray fails, it likely throws before assignment, so xp remains cp.
                # The exception handler should ideally signal to retry with CPU if possible.
                # For now, assume the initial xp determination is robust.
                pass # Let the exception propagate or be handled differently if needed
                img_float = xp.asarray(image, dtype=xp.float32) # Try again with NumPy
            else:
                raise # Re-raise if CPU conversion failed

        try:
            # --- STEP 1: INVERT (Float32) ---
            logger.debug("Step 1: Inverting...")
            inverted_float = 255.0 - img_float
            logger.debug("Inversion complete (float32)")
            report_progress(1) # Step 1: After Inversion

            # --- STEP 2: POST-INVERSION MASK REMOVAL / WHITE BALANCE (Float32) ---
            logger.debug("Step 2: Mask Removal / WB...")

            if override_mask_classification:
                mask_classification = override_mask_classification
                logger.debug(f"Using overridden Mask Classification: {mask_classification}")
                # We still need a mask_color_np for C-41 neutralization, detect it even if overridden
                # This ensures the neutralization logic has the color it needs if C-41 is forced
                mask_color_np = detect_orange_mask(image)
                logger.debug(f"Detected mask color (for potential C-41 override): {mask_color_np}")
            else:
                # --- Auto-Detect and Classify Mask ---
                logger.debug("Auto-detecting mask color and classifying...")
                mask_color_np = detect_orange_mask(image) # Pass original uint8 image
                logger.debug(f"Detected mask color (from corners): {mask_color_np}")

                # --- Classify detected mask color ---
                mask_classification = "Unknown/Other" # Default
                # Convert RGB [0-255] to HSV [0-179, 0-255, 0-255] for analysis
                # Reshape to 1x1x3 for cvtColor
                mask_rgb_u8 = np.clip(mask_color_np, 0, 255).astype(np.uint8).reshape(1, 1, 3)
                mask_hsv = cv2.cvtColor(mask_rgb_u8, cv2.COLOR_RGB2HSV)[0, 0]
                hue, sat, val = mask_hsv[0], mask_hsv[1], mask_hsv[2]
                logger.debug(f"Mask HSV: H={hue}, S={sat}, V={val}")

                # --- Classify Base Type ---
                # Use thresholds from config
                CLEAR_SAT_MAX = self.params['mask_clear_sat_max']
                C41_HUE_MIN = self.params['mask_c41_hue_min']
                C41_HUE_MAX = self.params['mask_c41_hue_max']
                C41_SAT_MIN = self.params['mask_c41_sat_min']
                C41_VAL_MIN = self.params['mask_c41_val_min']
                C41_VAL_MAX = self.params['mask_c41_val_max']

                if sat < CLEAR_SAT_MAX:
                    mask_classification = "Clear/Near Clear"
                elif (C41_HUE_MIN <= hue <= C41_HUE_MAX and
                      sat >= C41_SAT_MIN and
                      C41_VAL_MIN <= val <= C41_VAL_MAX):
                    mask_classification = "Likely C-41"
                # else: remains "Unknown/Other"
                logger.debug(f"Auto-Detected Mask Classification: {mask_classification}")

            # --- Apply Neutralization Based on Classification ---
            if mask_classification == "Likely C-41":
                logger.debug("Applying C-41 mask neutralization...")
                # Invert the mask color (target cast color in the inverted image)
                inverted_mask_color_np = 255.0 - mask_color_np
                inverted_mask_color_np = np.maximum(inverted_mask_color_np, 1e-6) # Avoid division by zero

                # Calculate scaling factors to make the inverted mask color neutral gray
                # Use fixed mid-gray (128.0) as target gray
                target_gray = self.params['wb_target_gray']
                # TODO: Consider alternative target_gray calculations if needed
                scale_factors_np = target_gray / inverted_mask_color_np
                # Clamp scale factors to prevent extreme shifts (Values might need tuning)
                # TODO: Make clamp range configurable?
                # Use clamp values from config
                scale_factors_np = np.clip(scale_factors_np,
                                           self.params['wb_clamp_min'],
                                           self.params['wb_clamp_max'])

                logger.debug(f"Inverted Mask Color: {inverted_mask_color_np}")
                logger.debug(f"Target Gray: {target_gray}")
                logger.debug(f"WB Scale Factors: {scale_factors_np}")

                # Transfer scale factors to GPU if needed
                scale_factors = xp.asarray(scale_factors_np, dtype=xp.float32)

                # Apply scaling factors to the inverted float image
                neutralized_float = inverted_float * scale_factors
                logger.debug("C-41 mask removal / WB scaling applied (float32)")

            elif mask_classification == "Clear/Near Clear":
                logger.debug("Skipping neutralization (Clear/Near Clear base detected).")
                neutralized_float = inverted_float # Use the simply inverted image
            else: # Unknown/Other (Includes Vision 250D for now)
                logger.debug("Applying Gray World AWB for Unknown/Other base...")
                # Calculate average R, G, B of the inverted image
                # Need to handle potential GPU array here
                if GPU_ENABLED:
                    avg_r = float(xp.mean(inverted_float[:,:,0]))
                    avg_g = float(xp.mean(inverted_float[:,:,1]))
                    avg_b = float(xp.mean(inverted_float[:,:,2]))
                else: # NumPy array
                    avg_r = np.mean(inverted_float[:,:,0])
                    avg_g = np.mean(inverted_float[:,:,1])
                    avg_b = np.mean(inverted_float[:,:,2])

                # Avoid division by zero if channel average is zero
                avg_r = max(avg_r, 1e-6)
                avg_g = max(avg_g, 1e-6)
                avg_b = max(avg_b, 1e-6)

                # Calculate overall average brightness
                overall_avg = (avg_r + avg_g + avg_b) / 3.0

                # Calculate Gray World scaling factors
                scale_factors_np = np.array([overall_avg / avg_r,
                                             overall_avg / avg_g,
                                             overall_avg / avg_b], dtype=np.float32)

                # Clamp scale factors (using same clamps as C-41 for now)
                # TODO: Consider different/no clamps for Gray World?
                # Use clamp values from config
                scale_factors_np = np.clip(scale_factors_np,
                                           self.params['wb_clamp_min'],
                                           self.params['wb_clamp_max'])

                logger.debug(f"Gray World Averages: R={avg_r:.2f}, G={avg_g:.2f}, B={avg_b:.2f}")
                logger.debug(f"Gray World Overall Avg: {overall_avg:.2f}")
                logger.debug(f"Gray World Scale Factors (Clamped): {scale_factors_np}")

                # Transfer scale factors to GPU if needed
                scale_factors = xp.asarray(scale_factors_np, dtype=xp.float32)

                # Apply scaling factors
                neutralized_float = inverted_float * scale_factors
                logger.debug("Gray World AWB applied (float32)")

            # Result of this step is 'neutralized_float'
            report_progress(2) # Step 2: After Mask Removal / WB


            # --- STEP 3: CHANNEL-SPECIFIC CORRECTIONS (using float32) ---
            logger.debug("Step 3: Channel-specific adjustments...")
            # Use default correction matrix from config
            # TODO: Allow overriding based on film_profile
            correction_matrix_np = self.params['correction_matrix']
            correction_matrix = xp.asarray(correction_matrix_np) # Transfer matrix if needed

            # Apply matrix multiplication (on float32) using the neutralized image
            h, w, c = neutralized_float.shape
            flat_image_float = neutralized_float.reshape(-1, 3)
            # Use xp.dot which works for both NumPy and CuPy
            corrected_flat_float = xp.dot(flat_image_float, correction_matrix.T)
            corrected_float = corrected_flat_float.reshape(h, w, 3)
            # No clipping or uint8 conversion yet
            logger.debug("Color correction matrix applied (float32)")
            report_progress(3) # Step 3: After Color Correction

            # --- STEP 4: AUTO WHITE BALANCE (REMOVED - Handled by Step 2) ---
            # print("[Converter Debug] Step 4: White balance...")
            # # For WB calculations, we might need LAB conversion.
            # # If using GPU, cvtColor might not be available directly in CuPy.
            # # Perform LAB conversion and analysis on CPU temporarily for simplicity.
            # # Transfer corrected_float back to CPU ONLY for this calculation if on GPU.
            # if GPU_ENABLED:
            #     corrected_np_temp = xp.asnumpy(corrected_float)
            # else:
            #     corrected_np_temp = corrected_float # It's already NumPy
            #
            # # Clip to uint8 range *before* LAB conversion to avoid issues
            # corrected_np_temp_u8 = np.clip(corrected_np_temp, 0, 255).astype(np.uint8)
            # l_channel_np = cv2.cvtColor(corrected_np_temp_u8, cv2.COLOR_RGB2LAB)[:,:,0]
            # lower = np.percentile(l_channel_np, 60)
            # upper = np.percentile(l_channel_np, 90)
            # wb_mask_np = (l_channel_np >= lower) & (l_channel_np <= upper)
            #
            # r_factor, g_factor, b_factor = 1.0, 1.0, 1.0 # Default factors
            # if np.sum(wb_mask_np) > 100:
            #     wb_pixels_np = corrected_np_temp[wb_mask_np] # Sample from float array
            #     r_avg = np.mean(wb_pixels_np[:,0])
            #     g_avg = np.mean(wb_pixels_np[:,1])
            #     b_avg = np.mean(wb_pixels_np[:,2])
            #
            #     if b_avg > r_avg and b_avg > g_avg: # Custom WB for blue cast
            #         target = (r_avg + g_avg) / 2
            #         r_factor = min(max(target / max(r_avg, 1e-6), 0.8), 1.2) # Avoid division by zero
            #         g_factor = min(max(target / max(g_avg, 1e-6), 0.8), 1.2)
            #         b_factor = min(max(target / max(b_avg, 1e-6), 0.5), 0.9)
            #         print(f"[Converter Debug] Custom WB factors: R:{r_factor:.2f}, G:{g_factor:.2f}, B:{b_factor:.2f}")
            #     else: # Standard gray-world WB
            #         avg = (r_avg + g_avg + b_avg) / 3.0
            #         r_factor = avg / max(r_avg, 1e-6)
            #         g_factor = avg / max(g_avg, 1e-6)
            #         b_factor = avg / max(b_avg, 1e-6)
            #         r_factor = min(max(r_factor, 0.8), 1.2)
            #         g_factor = min(max(g_factor, 0.8), 1.2)
            #         b_factor = min(max(b_factor, 0.8), 1.2)
            #         print(f"[Converter Debug] Gray-world WB factors: R:{r_factor:.2f}, G:{g_factor:.2f}, B:{b_factor:.2f}")
            #
            #     # Apply WB factors to the main float32 array (CPU or GPU)
            #     corrected_float[:,:,0] *= r_factor
            #     corrected_float[:,:,1] *= g_factor
            #     corrected_float[:,:,2] *= b_factor
            #     print("[Converter Debug] White balance applied (float32)")
            # else:
            #     print("[Converter Debug] Skipped WB due to insufficient reference pixels")
            # # No clipping or uint8 conversion yet

            # --- STEP 4: CHANNEL-SPECIFIC CURVES (using apply_curve on float32) ---
            logger.debug("Step 4: Channel-specific curves (float32)...")
            # Work directly on the float32 array from the previous step
            curves_result_float = corrected_float.copy()

            # Histogram calculation still needs uint8 data for np.histogram(..., range=[0, 256])
            # Create a temporary uint8 copy *on the CPU* for histogramming
            temp_u8_for_hist = xp.clip(corrected_float, 0, 255)
            if GPU_ENABLED:
                temp_u8_for_hist = xp.asnumpy(temp_u8_for_hist).astype(np.uint8)
            else:
                temp_u8_for_hist = temp_u8_for_hist.astype(np.uint8)

            for c in range(3):
                channel_float = curves_result_float[:,:,c] # Get the float channel to apply curve to
                channel_u8_hist = temp_u8_for_hist[:,:,c] # Get the uint8 channel for histogram

                hist, bins = np.histogram(channel_u8_hist.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                total_pixels = cdf[-1]

                curve_points = [[0, 0], [255, 255]] # Default to identity curve

                if total_pixels > 0:
                    clip_percent = self.params['curve_clip_percent']
                    clip_val_low = total_pixels * clip_percent / 100.0
                    clip_val_high = total_pixels * (100.0 - clip_percent) / 100.0
                    black_point = np.searchsorted(cdf, clip_val_low, side='right')
                    white_point = np.searchsorted(cdf, clip_val_high, side='left')
                    black_point = np.clip(black_point, 0, 254)
                    white_point = np.clip(white_point, black_point + 1, 255)

                    # Get gamma from config based on channel index
                    if c == 0: gamma = self.params['curve_gamma_red']
                    elif c == 1: gamma = self.params['curve_gamma_green']
                    else: gamma = self.params['curve_gamma_blue']

                    # Generate the curve points based on black/white point and gamma
                    # This defines the mapping: input -> output
                    if white_point > black_point:
                        # Create key points for the curve: 0, black_point, white_point, 255
                        # Map 0 to 0, map values >= white_point to 255
                        # Apply gamma correction between black_point and white_point
                        # Need intermediate points for gamma curve shape
                        num_intermediate = self.params['curve_num_intermediate_points']
                        in_points = np.linspace(black_point, white_point, num_intermediate)
                        normalized = (in_points - black_point) / (white_point - black_point)
                        gamma_corrected = np.power(normalized, 1.0 / gamma)
                        out_points = gamma_corrected * 255.0

                        curve_points = [[0, 0]] # Start at 0,0
                        # Convert zipped tuples to lists before extending
                        curve_points.extend([list(p) for p in zip(in_points, out_points)])
                        curve_points.append([255, 255]) # End at 255,255
                        # Ensure points are sorted and unique on x-axis (should be by linspace)
                        curve_points = sorted(curve_points, key=lambda p: p[0])
                    else:
                        # If black == white, create a step function
                        curve_points = [[0, 0], [black_point, 0], [white_point, 255], [255, 255]]


                # Apply the curve using the utility function on the float channel
                curves_result_float[:,:,c] = apply_curve(channel_float, curve_points)

            logger.debug("Channel-specific curves applied (float32)")
            report_progress(4) # Step 4: After Curves
            # curves_result_float now holds the image after curves (still float32)

            # --- STEP 5: FINAL COLOR GRADING (LAB/HSV on float32) ---
            # Convert the result of curves back to float32 for grading
            logger.debug("Step 5: Final color grading...")
            grading_input_float = curves_result_float # Already float32 (CPU or GPU) from Step 5

            # Perform LAB/HSV adjustments. Requires CPU for cv2.cvtColor.
            if GPU_ENABLED:
                grading_input_np = xp.asnumpy(grading_input_float)
            else:
                grading_input_np = grading_input_float # Already NumPy

            # LAB adjustments
            lab_np = cv2.cvtColor(grading_input_np.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
            a_channel = lab_np[:,:,1]
            a_avg = np.mean(a_channel)
            a_target = self.params['lab_a_target']
            a_factor = self.params['lab_a_correction_factor']
            a_max_adjust = self.params['lab_a_correction_max']
            if a_avg > a_target: a_channel -= min((a_avg - a_target) * a_factor, a_max_adjust)
            elif a_avg < a_target: a_channel += min((a_target - a_avg) * a_factor, a_max_adjust)

            b_channel = lab_np[:,:,2]
            b_avg = np.mean(b_channel)
            b_target = self.params['lab_b_target']
            b_factor = self.params['lab_b_correction_factor']
            b_max_adjust = self.params['lab_b_correction_max']
            # Typically adjust B channel towards target if below (less blue/more yellow)
            if b_avg < b_target: b_channel += min((b_target - b_avg) * b_factor, b_max_adjust)

            lab_np[:,:,1] = np.clip(a_channel, 0, 255)
            lab_np[:,:,2] = np.clip(b_channel, 0, 255)
            graded_rgb_np = cv2.cvtColor(lab_np.astype(np.uint8), cv2.COLOR_LAB2RGB)

            # HSV Saturation adjustment
            hsv_np = cv2.cvtColor(graded_rgb_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv_np[:,:,1] *= self.params['hsv_saturation_boost']
            hsv_np[:,:,1] = np.clip(hsv_np[:,:,1], 0, 255)
            final_graded_np = cv2.cvtColor(hsv_np.astype(np.uint8), cv2.COLOR_HSV2RGB)
            logger.debug("Final color grading complete (CPU)")
            report_progress(5) # Step 5: After Grading

            # --- STEP 6 (FINAL): Convert back to uint8 ---
            # The final result is final_graded_np on the CPU.
            final_image = final_graded_np # Already uint8 from last cvtColor

            logger.info(f"Conversion completed successfully. Mask: {mask_classification}") # Add classification to log
            report_progress(total_steps) # Step 6: Finished
            return final_image, mask_classification # Return image and classification

        except Exception as e:
            import traceback
            # Use logger.exception to include traceback information automatically
            logger.exception(f"Unexpected error during conversion pipeline (Steps 1-5): {e}")
            # traceback.print_exc() # Can be removed as logger.exception handles it
            # Return None and error status to indicate failure to the caller
            return None, "Error" # Return None for image, "Error" for classification

    def apply_tone_curve(self, image, curve_points):
        """
        Apply a tone curve to an image using the centralized utility function.
        Expects a uint8 RGB image.
        """
        if image is None or image.size == 0:
            logger.warning("apply_tone_curve received empty image.")
            return image
        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("apply_tone_curve expects RGB image.")
            return image # Expects RGB
        # Ensure input is uint8 NumPy array for curve application
        if GPU_ENABLED and xp.get_array_module(image) == xp: # Check if it's a CuPy array
             logger.warning("apply_tone_curve expects NumPy image, got CuPy. Converting.")
             image = xp.asnumpy(image) # Convert from GPU to CPU

        # Ensure uint8 type after potential conversion
        if image.dtype != np.uint8:
             logger.warning(f"apply_tone_curve expects uint8 image, got {image.dtype}. Attempting conversion.")
             image = np.clip(image, 0, 255).astype(np.uint8)

        if not curve_points:
            # logger.debug("apply_tone_curve received no curve points.") # Keep commented out or remove
            return image.copy() # No curve to apply

        result = image.copy()
        try:
            # Apply the curve to each channel using the utility function
            # The utility function handles uint8 NumPy input correctly
            result[..., 0] = apply_curve(result[..., 0], curve_points)
            result[..., 1] = apply_curve(result[..., 1], curve_points)
            result[..., 2] = apply_curve(result[..., 2], curve_points)
        except Exception as e:
            logger.error(f"Error applying curve via utility in apply_tone_curve: {e}")
            # Return original image on error
            return image.copy()

        return result


    def auto_levels(self, image, clip_percent=1):
        """Auto-levels implementation with optional clipping"""
        if image is None or image.size == 0: return image
        if len(image.shape) != 3 or image.shape[2] != 3: return image # Expects RGB

        result = image.copy()
        for c in range(3):
            channel = image[:,:,c]
            if channel.size == 0: continue # Skip empty channels

            # Calculate histogram and CDF
            hist, bins = np.histogram(channel.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            total_pixels = cdf[-1]

            if total_pixels == 0: continue # Skip if channel is empty/uniform

            # Find black and white points based on clipping percentage
            clip_val_low = total_pixels * clip_percent / 100.0
            clip_val_high = total_pixels * (100.0 - clip_percent) / 100.0

            # Use searchsorted to find the first index where CDF >= clip_val
            black_point = np.searchsorted(cdf, clip_val_low, side='right')
            white_point = np.searchsorted(cdf, clip_val_high, side='left') # Find first index >= clip_val_high
            # Ensure white_point is at least black_point and points are within [0, 255]
            white_point = np.clip(max(black_point, white_point), 0, 255)
            black_point = np.clip(black_point, 0, 255)


            # Create lookup table for this channel
            # Vectorized LUT creation
            lut = np.arange(256, dtype=np.float32) # Create base range as float for calculation
            if white_point > black_point:
                scale = 255.0 / (white_point - black_point)
                lut = (lut - black_point) * scale
                lut = np.clip(lut, 0, 255) # Clip the results
            elif black_point == white_point: # Handle case of flat histogram after clipping
                lut.fill(0 if black_point == 0 else 255)

            lut = lut.astype(np.uint8) # Convert final LUT to uint8


            # Apply lookup table
            result[:,:,c] = cv2.LUT(channel, lut)

        return result