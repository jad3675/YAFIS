# Image adjustment operations
import numpy as np
import cv2
import cv2.xphoto as xphoto # Import the extended photo module
try:
    import cupy as cp
    # Check if CUDA is available and a device is accessible
    try:
        cp.cuda.runtime.getDeviceCount()
        GPU_ENABLED = True
        print("[Adjustments Info] CuPy found and GPU detected. GPU acceleration enabled for applicable functions.")
    except cp.cuda.runtime.CUDARuntimeError:
        GPU_ENABLED = False
        print("[Adjustments Warning] CuPy found but no compatible CUDA GPU detected or driver issue. Using CPU.")
except ImportError:
    print("[Adjustments Warning] CuPy not found. Install CuPy (e.g., 'pip install cupy-cudaXXX' where XXX is your CUDA version) for GPU acceleration. Using CPU.")
    GPU_ENABLED = False

# Import the new utility function
# Use relative import assuming utils is at the same level as processing parent
import sys
import os
# Add parent directory to sys.path if running directly or structure requires it
# This might be needed depending on how the project is run.
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    # Try relative import first (common case when run as part of the package)
    from ..utils.imaging import apply_curve
except ImportError:
    # Fallback if running script directly or structure differs
    # This assumes 'utils' is in the parent directory of 'processing'
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from utils.imaging import apply_curve
    except ImportError as e:
        print(f"Error importing apply_curve: {e}. Ensure utils/imaging.py exists and parent directory is accessible.")
        # Define a dummy function to avoid NameError later if import fails catastrophically
        def apply_curve(image_channel, curve_points):
            print("ERROR: apply_curve utility function could not be imported!")
            return image_channel

# --- Helper Function (Now uses the utility) ---
def _apply_tone_curve_channel(image_channel, curve_points):
    """
    Apply tone curve to a single image channel (expects uint8 NumPy array).
    Uses the centralized apply_curve utility.
    """
    if image_channel is None or image_channel.size == 0: return image_channel
    if len(image_channel.shape) != 2: raise ValueError("Input must be a single channel")
    # Ensure input is NumPy uint8 for this specific helper's contract
    if image_channel.dtype != np.uint8:
        raise TypeError(f"Input channel must be uint8 NumPy array, got {image_channel.dtype}")
    if 'cupy' in str(type(image_channel)):
        raise TypeError("Input channel must be NumPy array, not CuPy")

    # Call the utility function
    # The utility handles None/empty curve_points and returns uint8 NumPy array for uint8 input
    return apply_curve(image_channel, curve_points)

# --- Basic Adjustments ---
class ImageAdjustments:
    """Basic image adjustment operations"""
    @staticmethod
    def adjust_brightness(image, value):
        """Adjust image brightness (GPU/CPU)"""
        if image is None or image.size == 0: return image
        if value == 0: return image.copy()
        brightness_offset = (value / 100.0) * 127.0
        if GPU_ENABLED:
            xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else:
            xp = np; img_arr = image.astype(xp.float32)
        try:
            result_arr = img_arr + brightness_offset
            result_arr = xp.clip(result_arr, 0, 255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Brightness failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); result_arr = img_arr + brightness_offset; return xp.clip(result_arr, 0, 255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def adjust_contrast(image, value):
        """Adjust image contrast (GPU/CPU)"""
        if image is None or image.size == 0: return image
        if value == 0: return image.copy()
        factor = (259.0 * (value + 255.0)) / (255.0 * (259.0 - value)) if value != 259 else 259.0
        mean_gray = 128.0
        if GPU_ENABLED:
            xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else:
            xp = np; img_arr = image.astype(xp.float32)
        try:
            result_arr = factor * (img_arr - mean_gray) + mean_gray
            result_arr = xp.clip(result_arr, 0, 255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Contrast failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); result_arr = factor * (img_arr - mean_gray) + mean_gray; return xp.clip(result_arr, 0, 255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def adjust_saturation(image, value):
        """Adjust image saturation (expects uint8 RGB)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        factor = 1.0 + (value / 100.0)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def adjust_hue(image, value):
        """Adjust image hue (expects uint8 RGB)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + value) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def adjust_temp_tint(image, temp, tint):
        """Adjust image color temperature and tint (GPU/CPU)"""
        if image is None or image.size == 0 or (temp == 0 and tint == 0): return image.copy()
        temp_factor = temp * 0.3; tint_factor = tint * 0.3
        if GPU_ENABLED: xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else: xp = np; img_arr = image.astype(xp.float32)
        try:
            result_arr = img_arr.copy()
            result_arr[..., 0] += temp_factor; result_arr[..., 2] -= temp_factor; result_arr[..., 1] -= tint_factor
            result_arr = xp.clip(result_arr, 0, 255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Temp/Tint failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); result_arr = img_arr.copy(); result_arr[..., 0] += temp_factor; result_arr[..., 2] -= temp_factor; result_arr[..., 1] -= tint_factor; return xp.clip(result_arr, 0, 255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def adjust_levels(image, in_black, in_white, gamma, out_black, out_white):
        """Adjust image levels (expects uint8 RGB or single channel)"""
        if image is None or image.size == 0: return image
        in_black=max(0,min(int(in_black),254)); in_white=max(in_black+1,min(int(in_white),255))
        gamma=max(0.1,min(float(gamma),10.0)); out_black=max(0,min(int(out_black),254))
        out_white=max(out_black+1,min(int(out_white),255))
        if (in_black==0 and in_white==255 and gamma==1.0 and out_black==0 and out_white==255): return image.copy()

        # Ensure in_white > in_black and out_white > out_black
        if in_white <= in_black: in_white = in_black + 1
        if out_white <= out_black: out_white = out_black + 1

        lut=np.arange(256,dtype=np.float32)
        # Avoid division by zero if in_black == in_white (shouldn't happen with checks above)
        scale = (in_white - in_black)
        if scale < 1e-6: scale = 1e-6
        lut=(lut-in_black)/ scale
        lut=np.clip(lut,0,1)
        if gamma!=1.0: lut=np.power(lut,1.0/gamma)
        lut=lut*(out_white-out_black)+out_black; lut=np.clip(lut,0,255)
        return cv2.LUT(image, lut.astype(np.uint8))

# --- Advanced Adjustments ---
class AdvancedAdjustments:
    """Advanced image adjustment operations"""
    @staticmethod
    def apply_curves(image, curve_points_r=None, curve_points_g=None, curve_points_b=None, curve_points_rgb=None):
        """
        Apply curves to image channels (expects uint8 RGB).
        Applies RGB curve if provided, unless overridden by a specific channel curve.
        """
        if image is None or image.size == 0: return image
        result = image.copy()
        # Ensure input is uint8 NumPy for the helper _apply_tone_curve_channel
        if result.dtype != np.uint8:
            print("[Adjustments Warning] apply_curves expects uint8 input. Converting.")
            result = np.clip(result, 0, 255).astype(np.uint8)
        # Check for CuPy *after* ensuring uint8
        if 'cupy' in str(type(result)):
             print("[Adjustments Warning] apply_curves expects NumPy input. Converting.")
             # Need cp defined if GPU_ENABLED is true, rely on global scope
             if GPU_ENABLED and 'cp' in globals() and cp: # Check cp exists
                 result = cp.asnumpy(result)
             else:
                 # Should not happen if dtype is uint8, but handle defensively
                 raise TypeError("Cannot convert non-GPU array using cp.asnumpy")


        # Define the default identity curve for comparison
        identity_curve = [[0, 0], [255, 255]]

        # Determine which curve to apply for each channel
        # Prioritize specific channel curve only if it's not None AND not the identity curve.
        # Otherwise, use the RGB curve if it's not None AND not the identity curve.
        r_curve_to_apply = None
        if curve_points_r and curve_points_r != identity_curve:
            r_curve_to_apply = curve_points_r
            r_curve_source = "Specific"
        elif curve_points_rgb and curve_points_rgb != identity_curve:
            r_curve_to_apply = curve_points_rgb
            r_curve_source = "RGB"
        else:
            r_curve_source = "None" # No curve applied (or identity)

        g_curve_to_apply = None
        if curve_points_g and curve_points_g != identity_curve:
            g_curve_to_apply = curve_points_g
            g_curve_source = "Specific"
        elif curve_points_rgb and curve_points_rgb != identity_curve:
            g_curve_to_apply = curve_points_rgb
            g_curve_source = "RGB"
        else:
            g_curve_source = "None"

        b_curve_to_apply = None
        if curve_points_b and curve_points_b != identity_curve:
            b_curve_to_apply = curve_points_b
            b_curve_source = "Specific"
        elif curve_points_rgb and curve_points_rgb != identity_curve:
            b_curve_to_apply = curve_points_rgb
            b_curve_source = "RGB"
        else:
            b_curve_source = "None"

        # Apply the determined curve for each channel, with error handling
        try:
            if r_curve_to_apply:
                result[..., 0] = _apply_tone_curve_channel(result[..., 0], r_curve_to_apply)
        except Exception as e:
            import traceback
            print(f"[Adjustments Error] Failed applying RED curve: {e}")
            traceback.print_exc()
            # Continue processing other channels

        try:
            if g_curve_to_apply:
                result[..., 1] = _apply_tone_curve_channel(result[..., 1], g_curve_to_apply)
        except Exception as e:
            import traceback
            print(f"[Adjustments Error] Failed applying GREEN curve: {e}")
            traceback.print_exc()
            # Continue processing other channels

        try:
            if b_curve_to_apply:
                result[..., 2] = _apply_tone_curve_channel(result[..., 2], b_curve_to_apply)
        except Exception as e:
            import traceback
            print(f"[Adjustments Error] Failed applying BLUE curve: {e}")
            traceback.print_exc()
            # Continue processing other channels

        return result

    @staticmethod
    def adjust_shadows_highlights(image, shadows, highlights):
        """Adjust shadows and highlights (expects uint8 RGB)"""
        if image is None or image.size == 0 or (shadows == 0 and highlights == 0): return image.copy()
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[..., 0]; l_norm = l_channel / 255.0
        shadow_mask = np.clip(1.0 - l_norm, 0, 1)**1.5
        highlight_mask = np.clip(l_norm, 0, 1)**1.5
        shadow_adjust = (shadows / 100.0) * 100; highlight_adjust = (highlights / 100.0) * 100
        l_adjusted = l_channel + shadow_mask * shadow_adjust + highlight_mask * highlight_adjust
        lab[..., 0] = np.clip(l_adjusted, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

    @staticmethod
    def adjust_color_balance_additive(image, cyan_red, magenta_green, yellow_blue):
        """Adjust color balance using CMY/RGB opposites (additive) (GPU/CPU)"""
        if image is None or image.size == 0 or (cyan_red == 0 and magenta_green == 0 and yellow_blue == 0): return image.copy()
        red_adj=(cyan_red/100.0)*127.0; green_adj=(magenta_green/100.0)*127.0; blue_adj=(yellow_blue/100.0)*127.0
        if GPU_ENABLED: xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else: xp = np; img_arr = image.astype(xp.float32)
        try:
            result_arr = img_arr.copy()
            result_arr[..., 0] += red_adj; result_arr[..., 1] -= green_adj; result_arr[..., 2] -= blue_adj
            result_arr = xp.clip(result_arr, 0, 255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Additive Color Balance failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); result_arr = img_arr.copy(); result_arr[..., 0] += red_adj; result_arr[..., 1] -= green_adj; result_arr[..., 2] -= blue_adj; return xp.clip(result_arr, 0, 255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def apply_color_balance(image, red_shift, green_shift, blue_shift, red_balance, green_balance, blue_balance):
        """Apply color balance adjustments (additive shifts + multiplicative balance) (GPU/CPU)"""
        if image is None or image.size == 0: return image
        is_identity = (red_shift == 0 and green_shift == 0 and blue_shift == 0 and red_balance == 1.0 and green_balance == 1.0 and blue_balance == 1.0)
        if is_identity: return image.copy()
        if GPU_ENABLED: xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else: xp = np; img_arr = image.astype(xp.float32)
        try:
            result_arr = img_arr.copy()
            if red_shift!=0 or green_shift!=0 or blue_shift!=0: result_arr += xp.array([red_shift, green_shift, blue_shift], dtype=xp.float32)
            if red_balance!=1.0 or green_balance!=1.0 or blue_balance!=1.0: result_arr *= xp.array([red_balance, green_balance, blue_balance], dtype=xp.float32)
            result_arr = xp.clip(result_arr, 0, 255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Color Balance failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); result_arr = img_arr.copy(); result_arr += xp.array([red_shift, green_shift, blue_shift]); result_arr *= xp.array([red_balance, green_balance, blue_balance]); return xp.clip(result_arr, 0, 255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def adjust_channel_mixer(image, output_channel, r_mix, g_mix, b_mix, constant=0):
        """Adjusts a single output channel based on input channel mix (GPU/CPU)"""
        if image is None or image.size == 0: return image
        channel_map={'Red':0,'Green':1,'Blue':2}; out_idx=channel_map.get(output_channel)
        if out_idx is None: raise ValueError("output_channel must be 'Red', 'Green', or 'Blue'")
        is_id=(out_idx==0 and r_mix==100 and g_mix==0 and b_mix==0 and constant==0) or \
              (out_idx==1 and r_mix==0 and g_mix==100 and b_mix==0 and constant==0) or \
              (out_idx==2 and r_mix==0 and g_mix==0 and b_mix==100 and constant==0)
        if is_id: return image.copy()
        r_f=r_mix/100.0; g_f=g_mix/100.0; b_f=b_mix/100.0; const_v=(constant/100.0)*127.5
        if GPU_ENABLED: xp = cp; img_arr = xp.asarray(image, dtype=xp.float32)
        else: xp = np; img_arr = image.astype(xp.float32)
        try:
            r_in,g_in,b_in=img_arr[...,0],img_arr[...,1],img_arr[...,2]
            new_ch=(r_in*r_f+g_in*g_f+b_in*b_f+const_v)
            result_arr=img_arr.copy(); result_arr[...,out_idx]=new_ch
            result_arr=xp.clip(result_arr,0,255)
            return xp.asnumpy(result_arr).astype(np.uint8) if GPU_ENABLED else result_arr.astype(np.uint8)
        except Exception as e:
            print(f"[Adjustments Error] Channel Mixer failed ({'GPU' if GPU_ENABLED else 'CPU'}): {e}. Falling back.")
            if GPU_ENABLED: xp = np; img_arr = image.astype(xp.float32); r_in,g_in,b_in=img_arr[...,0],img_arr[...,1],img_arr[...,2]; new_ch=(r_in*r_f+g_in*g_f+b_in*b_f+const_v); result_arr=img_arr.copy(); result_arr[...,out_idx]=new_ch; return xp.clip(result_arr,0,255).astype(np.uint8)
            else: return image.copy()

    @staticmethod
    def apply_noise_reduction(image, strength=10, template_window=7, search_window=21):
        """Applies Non-local Means Denoising for color images."""
        if image is None or image.size == 0 or strength <= 0: return image.copy()

        # Check if xphoto module is available
        if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'createSimpleWB'): # Check for a known function
            print("[Adjustments Error] Noise Reduction requires the 'opencv-contrib-python' package. Please install it (`pip install opencv-contrib-python`). Skipping.")
            return image.copy()

        # Ensure window sizes are odd and valid
        if template_window % 2 == 0: template_window += 1
        if search_window % 2 == 0: search_window += 1
        template_window = max(3, template_window)
        search_window = max(template_window, search_window) # Search window must be >= template window

        try:
            # Convert to BGR as fastNlMeansDenoisingColored often expects BGR
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Apply denoising
            denoised_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=float(strength), hColor=float(strength), templateWindowSize=template_window, searchWindowSize=search_window)
            # Convert back to RGB
            return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"[Adjustments Error] Noise Reduction (fastNlMeansDenoisingColored) failed: {e}. Is 'opencv-contrib-python' installed correctly?")
            return image.copy()
        except Exception as e:
            print(f"[Adjustments Error] Noise Reduction unexpected error: {e}")
            return image.copy()

    @staticmethod
    def adjust_hsl_by_range(image, color_range, hue_shift, saturation_shift, lightness_shift):
        """Adjusts Hue, Saturation, and Lightness for a specific color range."""
        if image is None or image.size == 0 or (hue_shift == 0 and saturation_shift == 0 and lightness_shift == 0): return image.copy()
        hue_ranges={'Reds':[(0,10),(170,179)],'Yellows':[(11,30)],'Greens':[(31,70)],'Cyans':[(71,100)],'Blues':[(101,130)],'Magentas':[(131,169)]}
        if color_range not in hue_ranges: raise ValueError(f"Invalid color_range: {color_range}")
        hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS).astype(np.float32); h,l,s=hls[...,0],hls[...,1],hls[...,2]
        mask=np.zeros_like(h,dtype=bool)
        for h_min,h_max in hue_ranges[color_range]: mask|=((h>=h_min)&(h<=h_max))
        h[mask]=(h[mask]+hue_shift)%180
        s[mask]*=(1.0+(saturation_shift/100.0))
        l[mask]+=(lightness_shift/100.0)*127.5
        hls[...,0]=h; hls[...,1]=np.clip(l,0,255); hls[...,2]=np.clip(s,0,255)
        return cv2.cvtColor(hls.astype(np.uint8),cv2.COLOR_HLS2RGB)

    @staticmethod
    def adjust_selective_color(image, color_range, cyan, magenta, yellow, black, relative=True):
        """
        Adjusts CMYK components within a selected color range (simulated in RGB).
        Expects uint8 RGB image.
        Adjustments are percentages (-100 to 100).
        """
        if image is None or image.size == 0: return image
        if cyan == 0 and magenta == 0 and yellow == 0 and black == 0: return image.copy()

        # Define approximate hue ranges (0-179 for OpenCV HSV/HLS)
        # These might need tuning based on visual results
        hue_ranges = {
            'Reds': [(0, 15), (165, 179)],
            'Yellows': [(16, 40)],
            'Greens': [(41, 75)],
            'Cyans': [(76, 105)],
            'Blues': [(106, 135)],
            'Magentas': [(136, 164)],
            # Grayscale adjustments need a different approach (saturation/lightness based)
            'Whites': [], # Special handling based on lightness/saturation
            'Neutrals': [], # Special handling based on lightness/saturation
            'Blacks': []  # Special handling based on lightness/saturation
        }

        if color_range not in hue_ranges:
            print(f"[Adjustments Warning] Selective Color for '{color_range}' not implemented or invalid range.")
            return image.copy()

        img_float = image.astype(np.float32) / 255.0 # Work with 0.0-1.0 range

        # --- Mask Creation ---
        mask = np.zeros(img_float.shape[:2], dtype=bool)
        if color_range in ['Whites', 'Neutrals', 'Blacks']:
            # Grayscale handling (simplified: based on lightness and low saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            sat_thresh = 30 # Low saturation threshold
            if color_range == 'Whites':
                mask = (v > 220) & (s < sat_thresh)
            elif color_range == 'Neutrals':
                mask = (v > 40) & (v < 220) & (s < sat_thresh)
            elif color_range == 'Blacks':
                mask = (v < 40) & (s < sat_thresh)
        else:
            # Color range handling
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h = hsv[..., 0]
            for h_min, h_max in hue_ranges[color_range]:
                mask |= ((h >= h_min) & (h <= h_max))

        if not np.any(mask):
            return image.copy() # No pixels in range

        # --- Apply Adjustments directly using the mask ---
        result_float = img_float.copy()

        # Convert percentages to factors (-1.0 to 1.0)
        c_adj = cyan / 100.0
        m_adj = magenta / 100.0
        y_adj = yellow / 100.0
        k_adj = black / 100.0

        # Select the pixels to modify using the mask
        r_masked = result_float[..., 0][mask]
        g_masked = result_float[..., 1][mask]
        b_masked = result_float[..., 2][mask]

        # Calculate CMY equivalents for masked pixels
        c_orig_masked = 1.0 - r_masked
        m_orig_masked = 1.0 - g_masked
        y_orig_masked = 1.0 - b_masked # Yellow is complement of Blue

        # --- Relative Adjustment Logic ---
        if relative:
            # Adjust C, M, Y relative to their original amount in the masked pixels
            delta_c = c_adj * c_orig_masked
            delta_m = m_adj * m_orig_masked
            delta_y = y_adj * y_orig_masked

            # Calculate new RGB values for masked pixels
            new_r_masked = r_masked + delta_c - delta_y - delta_m
            new_g_masked = g_masked + delta_m - delta_c - delta_y
            new_b_masked = b_masked + delta_y - delta_m - delta_c

            # Black adjustment affects lightness
            black_effect = k_adj * (r_masked + g_masked + b_masked) / 3.0
            new_r_masked -= black_effect
            new_g_masked -= black_effect
            new_b_masked -= black_effect

        # --- Absolute Adjustment Logic (Placeholder) ---
        else:
            print("[Adjustments Warning] Selective Color 'Absolute' mode not fully implemented. Using Relative logic.")
            # Fallback to relative logic for now (same as above)
            delta_c = c_adj * c_orig_masked
            delta_m = m_adj * m_orig_masked
            delta_y = y_adj * y_orig_masked
            new_r_masked = r_masked + delta_c - delta_y - delta_m
            new_g_masked = g_masked + delta_m - delta_c - delta_y
            new_b_masked = b_masked + delta_y - delta_m - delta_c
            black_effect = k_adj * (r_masked + g_masked + b_masked) / 3.0
            new_r_masked -= black_effect
            new_g_masked -= black_effect
            new_b_masked -= black_effect

        # Apply the calculated changes back to the result image using the mask
        result_float[..., 0][mask] = new_r_masked
        result_float[..., 1][mask] = new_g_masked
        result_float[..., 2][mask] = new_b_masked

        # Clip and convert back to uint8
        result_uint8 = (np.clip(result_float, 0.0, 1.0) * 255.0).astype(np.uint8)
        return result_uint8

    @staticmethod
    def adjust_vibrance(image, value):
        """Adjust image vibrance (expects uint8 RGB)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV).astype(np.float32); s=hsv[...,1]
        factor=value/100.0; sat_weight=1.0-(s/255.0); adjustment=factor*sat_weight*128
        hsv[...,1]=np.clip(s+adjustment,0,255)
        return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2RGB)

    @staticmethod
    def adjust_clarity(image, value):
        """Adjust image clarity using unsharp masking (expects uint8 RGB)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        lab=cv2.cvtColor(image,cv2.COLOR_RGB2LAB); l_channel=lab[...,0]
        amount=value/150.0; kernel_size=(5,5)
        blurred_l=cv2.GaussianBlur(l_channel,kernel_size,0)
        sharpened_l=cv2.addWeighted(l_channel,1.0+amount,blurred_l,-amount,0)
        lab[...,0]=sharpened_l
        return cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)

    @staticmethod
    def apply_vignette(image, amount, center_x=0.5, center_y=0.5, radius=0.7, feather=0.3, color=None):
        """Apply vignette effect (expects uint8 RGB)"""
        if image is None or image.size == 0 or amount == 0: return image.copy()
        h, w = image.shape[:2]
        center_x_px, center_y_px = int(w * center_x), int(h * center_y)
        max_dist = np.sqrt(max(center_x_px, w - center_x_px)**2 + max(center_y_px, h - center_y_px)**2)
        radius_px = max_dist * radius
        feather_px = max_dist * feather
        if radius_px <= 0: return image.copy() # Avoid division by zero if radius is too small

        Y, X = np.ogrid[:h, :w]
        dist_sq = (X - center_x_px)**2 + (Y - center_y_px)**2
        dist = np.sqrt(dist_sq)

        # Calculate vignette mask (0 = full effect, 1 = no effect)
        mask = np.ones((h, w), dtype=np.float32)
        inner_radius = radius_px - feather_px / 2.0
        outer_radius = radius_px + feather_px / 2.0

        if feather_px > 1e-3: # Apply smooth feathering if feather > 0
            mask[dist >= inner_radius] = (dist[dist >= inner_radius] - inner_radius) / feather_px
            mask = 1.0 - np.clip(mask, 0, 1) # Invert and clip
        else: # Hard edge if no feather
            mask[dist >= radius_px] = 0.0

        # Apply vignette effect
        img_float = image.astype(np.float32)
        if amount < 0: # Darken
            factor = 1.0 + (amount / 100.0) # 0.0 to 1.0
            result_float = img_float * (factor + (1.0 - factor) * mask[..., np.newaxis])
        else: # Lighten
            factor = amount / 100.0 # 0.0 to 1.0
            result_float = img_float + (255.0 - img_float) * factor * (1.0 - mask[..., np.newaxis])

        return np.clip(result_float, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_bw_mix(image, red_weight, green_weight, blue_weight):
        """Convert to B&W using channel weights (expects uint8 RGB)"""
        if image is None or image.size == 0: return image
        total_weight=red_weight+green_weight+blue_weight
        if total_weight<=0: print("[Adjustments Warning] B&W Mix weights sum <= 0. Using standard grayscale."); gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        else: img_float=image.astype(np.float32); r,g,b=img_float[...,0],img_float[...,1],img_float[...,2]; wr,wg,wb=red_weight/100.0,green_weight/100.0,blue_weight/100.0; gray_float=r*wr+g*wg+b*wb; gray=np.clip(gray_float,0,255).astype(np.uint8)
        return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

    @staticmethod
    def apply_color_grading(image, shadows_rgb, midtones_rgb, highlights_rgb):
        """Applies color grading to shadows, midtones, and highlights (expects uint8 RGB)"""
        if image is None or image.size == 0: return image
        is_identity = np.all(np.array(shadows_rgb)==0) and np.all(np.array(midtones_rgb)==0) and np.all(np.array(highlights_rgb)==0)
        if is_identity: return image.copy()
        img_float=image.astype(np.float32); luminance=np.mean(img_float,axis=2)/255.0
        def sigmoid(x,center,width): return 1/(1+np.exp(-(x-center)/(width+1e-6)))
        # Correctly indented color grading logic
        shadow_width=0.15; highlight_width=0.15; shadow_center=0.25; highlight_center=0.75
        shadow_mask=1.0-sigmoid(luminance,shadow_center,shadow_width)
        highlight_mask=sigmoid(luminance,highlight_center,highlight_width)
        midtones_mask=np.clip(1.0-shadow_mask-highlight_mask,0,1)
        shift_scale=30.0; s_shift=np.array(shadows_rgb)*shift_scale; m_shift=np.array(midtones_rgb)*shift_scale; h_shift=np.array(highlights_rgb)*shift_scale
        result=img_float.copy()
        result+=shadow_mask[...,np.newaxis]*s_shift; result+=midtones_mask[...,np.newaxis]*m_shift; result+=highlight_mask[...,np.newaxis]*h_shift
        return np.clip(result,0,255).astype(np.uint8)

    # Moved apply_film_grain out of apply_color_grading
    @staticmethod
    def apply_film_grain(image_float, intensity, size, roughness):
        """Apply film grain to image (expects float32 0-255, returns float32 0-255)"""
        if image_float is None or image_float.size == 0 or intensity <= 0: return image_float.copy()
        height, width, channels = image_float.shape
        if channels != 3: raise ValueError("Input image must be 3-channel RGB")

        # Determine backend based on input type
        # Use the global GPU_ENABLED and cp defined in this module
        is_cupy_input = GPU_ENABLED and 'cp' in globals() and cp and isinstance(image_float, cp.ndarray) # Added check for cp existence
        xp = cp if is_cupy_input else np

        # Generate noise on CPU (usually fast enough)
        noise_np = np.random.normal(0, 1, (height, width)).astype(np.float64)
        if size > 0:
            blur_size = max(3, int(size * 20) | 1)
            noise_np = cv2.GaussianBlur(noise_np, (blur_size, blur_size), 0)
        std_dev = np.std(noise_np)
        if std_dev > 1e-5: noise_np = (noise_np / std_dev) * intensity
        else: noise_np.fill(0)

        # Calculate luminance mask (needs uint8 input for cvtColor)
        # Convert float input to temporary uint8 on CPU for this step
        temp_image_uint8_np = np.clip(cp.asnumpy(image_float) if is_cupy_input else image_float, 0, 255).astype(np.uint8)
        luminance_np = cv2.cvtColor(temp_image_uint8_np, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grain_mask_np = np.power(1.0 - luminance_np, roughness)

        # Apply noise using the appropriate backend
        try:
            noise = xp.asarray(noise_np)
            grain_mask = xp.asarray(grain_mask_np)
            # Add noise weighted by mask to the input float image
            result_float = image_float + (noise * grain_mask)[..., xp.newaxis]
            # No clipping here, return float result
            return result_float
        except Exception as e:
            print(f"[Adjustments Error] Grain application failed ({'GPU' if is_cupy_input else 'CPU'}): {e}.") # Changed print context
            # Fallback: If GPU failed, try CPU if input was GPU
            if is_cupy_input:
                print("[Adjustments Info] Grain: Falling back to CPU.") # Changed print context
                xp_fallback = np
                image_float_np = cp.asnumpy(image_float)
                try:
                     result_float_fallback = image_float_np + (noise_np * grain_mask_np)[..., xp_fallback.newaxis]
                     return result_float_fallback # Return NumPy float
                except Exception as e_cpu:
                     print(f"[Adjustments Error] Grain CPU fallback also failed: {e_cpu}. Returning input.") # Changed print context
                     return image_float.copy() # Return original float input
            else:
                # If CPU failed initially
                print("[Adjustments Error] Grain CPU application failed. Returning input.") # Changed print context
                return image_float.copy() # Return original float input
        return np.clip(result,0,255).astype(np.uint8) # This return was incorrectly placed

    # --- Auto White Balance ---

    # --- Internal Scale Calculation Methods (for existing AWB) ---
    @staticmethod
    def _calculate_gray_world_scales(image):
        """Calculates RGB scales based on Gray World assumption."""
        if image is None or image.size == 0: return (1.0, 1.0, 1.0)
        # Calculate average for each channel
        avg_r = np.mean(image[..., 0])
        avg_g = np.mean(image[..., 1])
        avg_b = np.mean(image[..., 2])
        # Avoid division by zero
        if avg_g < 1e-5: avg_g = 1e-5
        # Calculate scales relative to green
        scale_r = avg_g / (avg_r + 1e-6)
        scale_b = avg_g / (avg_b + 1e-6)
        return (scale_r, 1.0, scale_b)

    @staticmethod
    def _calculate_white_patch_scales(image, percentile=99):
        """Calculates RGB scales based on White Patch assumption (brightest pixels)."""
        if image is None or image.size == 0: return (1.0, 1.0, 1.0)
        # Find the 'percentile'-th brightest pixel value in each channel
        max_r = np.percentile(image[..., 0], percentile)
        max_g = np.percentile(image[..., 1], percentile)
        max_b = np.percentile(image[..., 2], percentile)
        # Avoid division by zero
        if max_g < 1e-5: max_g = 1e-5
        # Calculate scales relative to green
        scale_r = max_g / (max_r + 1e-6)
        scale_b = max_g / (max_b + 1e-6)
        return (scale_r, 1.0, scale_b)

    @staticmethod
    def calculate_white_balance_from_color(image, picked_color_rgb):
        """Calculates RGB scales needed to make the picked color neutral gray."""
        if image is None or image.size == 0: return (1.0, 1.0, 1.0)
        if not isinstance(picked_color_rgb, (list, tuple)) or len(picked_color_rgb) != 3:
            print("[Adjustments Error] Invalid picked_color_rgb format.")
            return (1.0, 1.0, 1.0)

        r_pick, g_pick, b_pick = picked_color_rgb
        # Target is gray, so target R=G=B. We scale relative to Green.
        # Avoid division by zero
        if g_pick < 1e-5: g_pick = 1e-5
        scale_r = g_pick / (r_pick + 1e-6)
        scale_b = g_pick / (b_pick + 1e-6)
        return (scale_r, 1.0, scale_b)

    # --- OpenCV xphoto White Balance Methods ---
    @staticmethod
    def _apply_simple_wb(image):
        """Applies SimpleWB from cv2.xphoto"""
        if not hasattr(cv2, 'xphoto'):
            print("[Adjustments Error] SimpleWB requires 'opencv-contrib-python'. Skipping.")
            return image.copy()
        try:
            wb = xphoto.createSimpleWB()
            # SimpleWB might expect BGR input
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            balanced_bgr = wb.balanceWhite(img_bgr)
            return cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"[Adjustments Error] SimpleWB failed: {e}. Is 'opencv-contrib-python' installed?")
            return image.copy()
        except Exception as e:
            print(f"[Adjustments Error] SimpleWB unexpected error: {e}")
            return image.copy()

    @staticmethod
    def _apply_learning_based_wb(image):
        """Applies LearningBasedWB from cv2.xphoto"""
        if not hasattr(cv2, 'xphoto'):
            print("[Adjustments Error] LearningBasedWB requires 'opencv-contrib-python'. Skipping.")
            return image.copy()
        try:
            wb = xphoto.createLearningBasedWB()
            # LearningBasedWB might expect BGR input
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            balanced_bgr = wb.balanceWhite(img_bgr)
            return cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"[Adjustments Error] LearningBasedWB failed: {e}. Is 'opencv-contrib-python' installed?")
            return image.copy()
        except Exception as e:
            print(f"[Adjustments Error] LearningBasedWB unexpected error: {e}")
            return image.copy()

    # --- Main Auto White Balance Function ---
    @staticmethod
    def apply_auto_white_balance(image, method='gray_world', percentile=99):
        """Applies automatic white balance using the specified method."""
        if image is None or image.size == 0: return image

        if method == 'gray_world':
            scales = AdvancedAdjustments._calculate_gray_world_scales(image)
        elif method == 'white_patch':
            scales = AdvancedAdjustments._calculate_white_patch_scales(image, percentile)
        elif method == 'simple_wb':
            return AdvancedAdjustments._apply_simple_wb(image)
        elif method == 'learning_wb':
            return AdvancedAdjustments._apply_learning_based_wb(image)
        else:
            print(f"[Adjustments Warning] Unknown AWB method: {method}. Using Gray World.")
            scales = AdvancedAdjustments._calculate_gray_world_scales(image)

        # Apply scales if calculated (not for xphoto methods)
        if method in ['gray_world', 'white_patch']:
            scale_r, scale_g, scale_b = scales
            # Apply scaling using float32 for precision
            img_float = image.astype(np.float32)
            img_float[..., 0] *= scale_r
            img_float[..., 1] *= scale_g # Usually 1.0
            img_float[..., 2] *= scale_b
            # Clip and convert back to uint8
            return np.clip(img_float, 0, 255).astype(np.uint8)
        else:
            # For xphoto methods, the result is already returned
            return image # Should have been returned by the specific method call

    # --- Auto Levels ---
    @staticmethod
    def apply_auto_levels(image, colorspace_mode='luminance', midrange=0.5):
        """
        Applies auto levels adjustment based on histogram stretching.
        Args:
            image (np.ndarray): Input uint8 RGB image.
            colorspace_mode (str): Channel/colorspace to use for statistics ('luminance', 'lightness', 'brightness', 'gray', 'average', 'rgb').
            midrange (float): Target midpoint for gamma correction (0.01 to 0.99).
        Returns:
            np.ndarray: Adjusted uint8 RGB image.
        """
        if image is None or image.size == 0: return image
        midrange = np.clip(midrange, 0.01, 0.99)

        # --- Calculate Histogram Source ---
        if colorspace_mode == 'luminance':
            # Calculate Luminance (Y) from RGB using standard coefficients
            source = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif colorspace_mode == 'lightness':
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            source = lab[..., 0] # L channel (0-255)
        elif colorspace_mode == 'brightness':
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            source = hsv[..., 2] # V channel (0-255)
        elif colorspace_mode == 'gray':
            source = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif colorspace_mode == 'average':
            source = np.mean(image, axis=2)
        elif colorspace_mode == 'rgb':
            source = image # Process each channel independently
        else:
            print(f"[Adjustments Warning] Unknown Auto Levels colorspace_mode: {colorspace_mode}. Using Luminance.")
            source = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]

        # --- Calculate Levels and Gamma ---
        result = image.copy()
        if colorspace_mode == 'rgb':
            # Process each channel separately
            for i in range(3):
                channel = image[..., i]
                # Find min/max ignoring pure black/white unless they are the only values
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                min_val = 0; max_val = 255
                for j in range(1, 255): # Find first non-zero bin > 0
                    if hist[j][0] > 0: min_val = j; break
                for j in range(254, 0, -1): # Find first non-zero bin < 255
                    if hist[j][0] > 0: max_val = j; break
                if min_val >= max_val: continue # Skip if channel is flat

                # Calculate gamma
                mean_val = np.mean(channel[channel > min_val]) if np.any(channel > min_val) else min_val
                target_mean = min_val + (max_val - min_val) * midrange
                gamma = np.log( (mean_val - min_val) / (max_val - min_val + 1e-6) + 1e-6 ) / \
                        np.log( (target_mean - min_val) / (max_val - min_val + 1e-6) + 1e-6 )
                gamma = np.clip(gamma, 0.1, 10.0) # Clamp gamma

                # Apply levels and gamma using ImageAdjustments.adjust_levels
                result[..., i] = ImageAdjustments.adjust_levels(channel, min_val, max_val, gamma, 0, 255)
        else:
            # Process based on single source channel statistics
            source_uint8 = np.clip(source, 0, 255).astype(np.uint8)
            hist = cv2.calcHist([source_uint8], [0], None, [256], [0, 256])
            min_val = 0; max_val = 255
            for j in range(1, 255): # Find first non-zero bin > 0
                if hist[j][0] > 0: min_val = j; break
            for j in range(254, 0, -1): # Find first non-zero bin < 255
                if hist[j][0] > 0: max_val = j; break
            if min_val >= max_val: return image.copy() # Skip if source is flat

            # Calculate gamma
            mean_val = np.mean(source_uint8[source_uint8 > min_val]) if np.any(source_uint8 > min_val) else min_val
            target_mean = min_val + (max_val - min_val) * midrange
            gamma = np.log( (mean_val - min_val) / (max_val - min_val + 1e-6) + 1e-6 ) / \
                    np.log( (target_mean - min_val) / (max_val - min_val + 1e-6) + 1e-6 )
            gamma = np.clip(gamma, 0.1, 10.0) # Clamp gamma

            # Apply levels and gamma to all RGB channels using the same parameters
            for i in range(3):
                result[..., i] = ImageAdjustments.adjust_levels(image[..., i], min_val, max_val, gamma, 0, 255)

        return result

    # --- Auto Color ---
    @staticmethod
    def apply_auto_color(image, method='gamma', clipmode='together', cliplow=0.1, cliphigh=None, neutralgray_percent=None):
        """
        Applies automatic color correction based on histogram analysis.
        Args:
            image (np.ndarray): Input uint8 RGB image.
            method (str): 'gamma' (default), 'recolor', 'none'.
            clipmode (str): 'together' (default) or 'separate'. How clipping limits are determined.
            cliplow (float): Percentage of pixels to clip at the low end (0.0 to 10.0). Default 0.1.
            cliphigh (float): Percentage of pixels to clip at the high end (0.0 to 10.0). Default is cliplow.
            neutralgray_percent (float): Percentage of pixels around the median to force neutral gray (0.0 to 10.0). Default None (disabled).
        Returns:
            np.ndarray: Adjusted uint8 RGB image.
        """
        if image is None or image.size == 0: return image
        if method == 'none': return image.copy()

        img_float = image.astype(np.float32)
        h, w, c = img_float.shape
        if c != 3: raise ValueError("Input must be 3-channel RGB")

        cliplow_pct = np.clip(cliplow, 0.0, 10.0) / 100.0
        cliphigh_pct = np.clip(cliphigh if cliphigh is not None else cliplow, 0.0, 10.0) / 100.0
        neutral_pct = np.clip(neutralgray_percent, 0.0, 10.0) / 100.0 if neutralgray_percent is not None else None

        result_float = img_float.copy()

        # --- Calculate Clipping Limits ---
        if clipmode == 'together':
            # Use average channel for clipping limits
            avg_channel = np.mean(img_float, axis=2)
            hist, _ = np.histogram(avg_channel.ravel(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max() # Normalize for visualization? Not needed here.
            total_pixels = h * w
            low_thresh_val = 0; high_thresh_val = 255
            # Find low threshold
            for i in range(256):
                if cdf[i] >= total_pixels * cliplow_pct:
                    low_thresh_val = i; break
            # Find high threshold
            for i in range(255, -1, -1):
                if cdf[i] <= total_pixels * (1.0 - cliphigh_pct):
                    high_thresh_val = i; break
            if low_thresh_val >= high_thresh_val: high_thresh_val = low_thresh_val + 1 # Ensure high > low
            low_thresh_r, low_thresh_g, low_thresh_b = low_thresh_val, low_thresh_val, low_thresh_val
            high_thresh_r, high_thresh_g, high_thresh_b = high_thresh_val, high_thresh_val, high_thresh_val
        else: # 'separate'
            low_thresh_r, low_thresh_g, low_thresh_b = 0, 0, 0
            high_thresh_r, high_thresh_g, high_thresh_b = 255, 255, 255
            for i in range(3):
                channel = img_float[..., i]
                hist, _ = np.histogram(channel.ravel(), 256, [0, 256])
                cdf = hist.cumsum(); total_pixels = h * w
                low_val = 0; high_val = 255
                for j in range(256):
                    if cdf[j] >= total_pixels * cliplow_pct: low_val = j; break
                for j in range(255, -1, -1):
                    if cdf[j] <= total_pixels * (1.0 - cliphigh_pct): high_val = j; break
                if low_val >= high_val: high_val = low_val + 1
                if i == 0: low_thresh_r, high_thresh_r = low_val, high_val
                elif i == 1: low_thresh_g, high_thresh_g = low_val, high_val
                else: low_thresh_b, high_thresh_b = low_val, high_val

        # --- Apply Clipping and Scaling ---
        scale_r = 255.0 / (high_thresh_r - low_thresh_r + 1e-6)
        scale_g = 255.0 / (high_thresh_g - low_thresh_g + 1e-6)
        scale_b = 255.0 / (high_thresh_b - low_thresh_b + 1e-6)

        result_float[..., 0] = (result_float[..., 0] - low_thresh_r) * scale_r
        result_float[..., 1] = (result_float[..., 1] - low_thresh_g) * scale_g
        result_float[..., 2] = (result_float[..., 2] - low_thresh_b) * scale_b
        result_float = np.clip(result_float, 0, 255)

        # --- Neutral Gray Correction ---
        if neutral_pct is not None and neutral_pct > 0:
            median_r = np.median(result_float[..., 0])
            median_g = np.median(result_float[..., 1])
            median_b = np.median(result_float[..., 2])
            avg_median = (median_r + median_g + median_b) / 3.0
            # Calculate scaling factors to push medians towards average median
            scale_r_ng = avg_median / (median_r + 1e-6)
            scale_g_ng = avg_median / (median_g + 1e-6)
            scale_b_ng = avg_median / (median_b + 1e-6)
            # Apply scaling only to pixels within the neutral percentage range around the median
            # This part is complex to implement correctly without LAB space.
            # Simplified approach: Apply globally for now (might affect colors undesirably)
            # A better approach would involve masking based on saturation/luminance.
            print("[Adjustments Warning] Neutral Gray correction in Auto Color is simplified and applied globally.")
            result_float[..., 0] *= scale_r_ng
            result_float[..., 1] *= scale_g_ng
            result_float[..., 2] *= scale_b_ng
            result_float = np.clip(result_float, 0, 255)


        # --- Gamma Correction (if method='gamma') ---
        if method == 'gamma':
            # Calculate average intensity after clipping/scaling
            avg_intensity = np.mean(result_float)
            target_intensity = 128.0 # Target mid-gray
            # Calculate gamma to shift average intensity towards target
            gamma = np.log(avg_intensity / 255.0 + 1e-6) / np.log(target_intensity / 255.0 + 1e-6)
            gamma = np.clip(gamma, 0.1, 10.0) # Clamp gamma
            # Apply gamma correction
            result_float = np.power(result_float / 255.0, gamma) * 255.0
            result_float = np.clip(result_float, 0, 255)

        # --- Recolor (if method='recolor') ---
        elif method == 'recolor':
            # This is a placeholder for a more sophisticated recoloring algorithm
            # Could involve histogram matching to a reference, etc.
            # For now, just return the clipped/scaled result without gamma.
            print("[Adjustments Warning] Auto Color 'recolor' method is not fully implemented. Applying clipping/scaling only.")
            pass # Result already clipped/scaled

        return result_float.astype(np.uint8)

    # --- Auto Tone ---
    @staticmethod
    def apply_auto_tone(image, nr_strength=5, awb_method='gray_world', al_mode='luminance', al_midrange=0.5, clarity_strength=10):
        """
        Applies a sequence of automatic adjustments: NR, AWB, Auto Levels, Clarity.
        Args:
            image (np.ndarray): Input uint8 RGB image.
            nr_strength (int): Strength for noise reduction (0-100).
            awb_method (str): Method for Auto White Balance.
            al_mode (str): Colorspace mode for Auto Levels.
            al_midrange (float): Midrange target for Auto Levels.
            clarity_strength (int): Strength for clarity adjustment (-100 to 100).
        Returns:
            np.ndarray: Adjusted uint8 RGB image.
        """
        if image is None or image.size == 0: return image
        print("[Adjustments Info] Applying Auto Tone Sequence...")
        current_image = image.copy()

        # 1. Noise Reduction
        if nr_strength > 0:
            print(f"  Auto Tone Step 1: Noise Reduction (Strength: {nr_strength})")
            current_image = AdvancedAdjustments.apply_noise_reduction(current_image, strength=nr_strength)
            if current_image is None: print("    NR failed."); return image.copy() # Revert on failure

        # 2. Auto White Balance
        print(f"  Auto Tone Step 2: Auto White Balance (Method: {awb_method})")
        current_image = AdvancedAdjustments.apply_auto_white_balance(current_image, method=awb_method)
        if current_image is None: print("    AWB failed."); return image.copy() # Revert on failure

        # 3. Auto Levels
        print(f"  Auto Tone Step 3: Auto Levels (Mode: {al_mode}, Mid: {al_midrange})")
        current_image = AdvancedAdjustments.apply_auto_levels(current_image, colorspace_mode=al_mode, midrange=al_midrange)
        if current_image is None: print("    Auto Levels failed."); return image.copy() # Revert on failure

        # 4. Clarity
        if clarity_strength != 0:
            print(f"  Auto Tone Step 4: Clarity (Strength: {clarity_strength})")
            current_image = AdvancedAdjustments.adjust_clarity(current_image, value=clarity_strength)
            if current_image is None: print("    Clarity failed."); return image.copy() # Revert on failure

        print("[Adjustments Info] Auto Tone Sequence Finished.")
        return current_image


# --- Global Adjustment Function ---

def apply_all_adjustments(image, adjustments_dict):
    """
    Applies all adjustments specified in the dictionary to the image.
    Expects uint8 RGB image input. Returns uint8 RGB image.
    """
    if image is None or image.size == 0: return image
    if not adjustments_dict: return image.copy()

    # Create instances (consider making these class members if performance is critical)
    basic_adjuster = ImageAdjustments()
    advanced_adjuster = AdvancedAdjustments()

    # Start with a copy
    current_image = image.copy()
    print("[Adjustments Pipeline] Starting adjustment application...")

    # Determine backend for potential GPU use in some functions
    xp = cp if GPU_ENABLED else np
    print(f"[Adjustments Pipeline] Using backend: {'CuPy (GPU)' if GPU_ENABLED else 'NumPy (CPU)'}")

    # --- Pipeline Order ---
    # Apply adjustments in a reasonable order, checking if image is valid after each step

    def check_img(step_name):
        if current_image is None:
            print(f"[Adjustments Pipeline Error] Image became None after step '{step_name}'. Aborting.")
            return False
        return True

    # 1. Basic Adjustments (Brightness, Contrast, Saturation, Hue)
    # These often work better in sRGB space before more complex transforms
    print("  Pipeline Step: Basic Adjustments...")
    current_image = basic_adjuster.adjust_brightness(current_image, adjustments_dict.get('brightness', 0))
    if not check_img("Brightness"): return None
    current_image = basic_adjuster.adjust_contrast(current_image, adjustments_dict.get('contrast', 0))
    if not check_img("Contrast"): return None
    current_image = basic_adjuster.adjust_saturation(current_image, adjustments_dict.get('saturation', 0))
    if not check_img("Saturation"): return None
    current_image = basic_adjuster.adjust_hue(current_image, adjustments_dict.get('hue', 0))
    if not check_img("Hue"): return None
    print("  Pipeline Step: Basic Adjustments Done.")

    # 2. White Balance (Temp/Tint)
    print("  Pipeline Step: White Balance (Temp/Tint)...")
    current_image = basic_adjuster.adjust_temp_tint(current_image, adjustments_dict.get('temp', 0), adjustments_dict.get('tint', 0))
    if not check_img("Temp/Tint"): return None
    print("  Pipeline Step: White Balance Done.")

    # 3. Levels
    print("  Pipeline Step: Levels...")
    current_image = basic_adjuster.adjust_levels(
        current_image,
        adjustments_dict.get('levels_in_black', 0),
        adjustments_dict.get('levels_in_white', 255),
        adjustments_dict.get('levels_gamma', 1.0),
        adjustments_dict.get('levels_out_black', 0),
        adjustments_dict.get('levels_out_white', 255)
    )
    if not check_img("Levels"): return None
    print("  Pipeline Step: Levels Done.")

    # 4. Curves
    print("  Pipeline Step: Curves...")
    current_image = advanced_adjuster.apply_curves(
        current_image,
        adjustments_dict.get('curves_red'),
        adjustments_dict.get('curves_green'),
        adjustments_dict.get('curves_blue'),
        adjustments_dict.get('curves_rgb')
    )
    if not check_img("Curves"): return None
    print("  Pipeline Step: Curves Done.")

    # 5. Channel Mixer (Apply per output channel)
    print("  Pipeline Step: Channel Mixer...")
    for out_ch_name in ['Red', 'Green', 'Blue']:
        r_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_r', 100 if out_ch_name == 'Red' else 0)
        g_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_g', 100 if out_ch_name == 'Green' else 0)
        b_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_b', 100 if out_ch_name == 'Blue' else 0)
        const = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_const', 0)
        # Only apply if not identity transform for that channel
        is_id = (out_ch_name == 'Red' and r_mix == 100 and g_mix == 0 and b_mix == 0 and const == 0) or \
                (out_ch_name == 'Green' and r_mix == 0 and g_mix == 100 and b_mix == 0 and const == 0) or \
                (out_ch_name == 'Blue' and r_mix == 0 and g_mix == 0 and b_mix == 100 and const == 0)
        if not is_id:
            current_image = advanced_adjuster.adjust_channel_mixer(current_image, out_ch_name, r_mix, g_mix, b_mix, const)
            if not check_img(f"Channel Mixer ({out_ch_name})"): return None
    print("  Pipeline Step: Channel Mixer Done.")

    # 6. HSL Adjustments (Apply per color range)
    print("  Pipeline Step: HSL...")
    for color_range in ['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas']:
        color_key = color_range.lower()
        h_shift = adjustments_dict.get(f'hsl_{color_key}_h', 0)
        s_shift = adjustments_dict.get(f'hsl_{color_key}_s', 0)
        l_shift = adjustments_dict.get(f'hsl_{color_key}_l', 0)
        if h_shift != 0 or s_shift != 0 or l_shift != 0:
            current_image = advanced_adjuster.adjust_hsl_by_range(current_image, color_range, h_shift, s_shift, l_shift)
            if not check_img(f"HSL ({color_range})"): return None
    print("  Pipeline Step: HSL Done.")

    # 7. Selective Color (Apply per color range)
    print("  Pipeline Step: Selective Color...")
    relative_mode = adjustments_dict.get('sel_relative', True)
    for color_range in ['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas', 'Whites', 'Neutrals', 'Blacks']:
         color_key = color_range.lower()
         c_adj = adjustments_dict.get(f'sel_{color_key}_c', 0)
         m_adj = adjustments_dict.get(f'sel_{color_key}_m', 0)
         y_adj = adjustments_dict.get(f'sel_{color_key}_y', 0)
         k_adj = adjustments_dict.get(f'sel_{color_key}_k', 0)
         if c_adj != 0 or m_adj != 0 or y_adj != 0 or k_adj != 0:
             current_image = advanced_adjuster.adjust_selective_color(current_image, color_range, c_adj, m_adj, y_adj, k_adj, relative_mode)
             if not check_img(f"Selective Color ({color_range})"): return None
    print("  Pipeline Step: Selective Color Done.")

    # --- Convert to Float for remaining adjustments ---
    print("  Pipeline Step: Converting to Float32...")
    current_image_float = current_image.astype(np.float32)
    print("  Pipeline Step: Float Conversion Done.")

    # --- Float Adjustments ---

    # 8. Clarity (on float)
    print("  Pipeline Step: Clarity...")
    clarity_val = adjustments_dict.get('clarity', 0) # Assuming 'clarity' key exists if needed
    if clarity_val != 0:
        # adjust_clarity expects uint8, so convert, adjust, convert back
        temp_uint8 = np.clip(current_image_float, 0, 255).astype(np.uint8)
        clarity_result_uint8 = advanced_adjuster.adjust_clarity(temp_uint8, clarity_val)
        if clarity_result_uint8 is None: print("    Clarity failed."); return None
        current_image_float = clarity_result_uint8.astype(np.float32)
        if not check_img("Clarity"): return None # Check after conversion back
    print("  Pipeline Step: Clarity Done.")


    # 9. Vibrance (on float)
    print("  Pipeline Step: Vibrance...")
    vibrance_val = adjustments_dict.get('vibrance', 0) # Assuming 'vibrance' key
    if vibrance_val != 0:
        # adjust_vibrance expects uint8, convert, adjust, convert back
        temp_uint8 = np.clip(current_image_float, 0, 255).astype(np.uint8)
        vibrance_result_uint8 = advanced_adjuster.adjust_vibrance(temp_uint8, vibrance_val)
        if vibrance_result_uint8 is None: print("    Vibrance failed."); return None
        current_image_float = vibrance_result_uint8.astype(np.float32)
        if not check_img("Vibrance"): return None
    print("  Pipeline Step: Vibrance Done.")


    # 10. Noise Reduction (on float)
    print("  Pipeline Step: Noise Reduction...")
    nr_strength = adjustments_dict.get('noise_reduction_strength', 0)
    if nr_strength > 0:
        # apply_noise_reduction expects uint8, convert, adjust, convert back
        temp_uint8 = np.clip(current_image_float, 0, 255).astype(np.uint8)
        nr_result_uint8 = advanced_adjuster.apply_noise_reduction(temp_uint8, strength=nr_strength)
        if nr_result_uint8 is None: print("    Noise Reduction failed."); return None
        current_image_float = nr_result_uint8.astype(np.float32)
        if not check_img("Noise Reduction"): return None
    print("  Pipeline Step: Noise Reduction Done.")


    # --- Final Conversion and Return ---
    print("[Adjustments Pipeline] Clipping final float result.")
    final_float_clipped = np.clip(current_image_float, 0, 255)

    print("[Adjustments Pipeline] Converting final result to uint8.")
    final_uint8 = final_float_clipped.astype(np.uint8)

    print("[Adjustments Pipeline] Finished.")
    return final_uint8

# Helper conversion functions (if needed elsewhere, move to utils)
def to_uint8_for_cv(float_arr):
    """Safely convert float (0-255) to uint8 for OpenCV functions."""
    if float_arr is None: return None
    # Check if CuPy array and convert if necessary
    if GPU_ENABLED and 'cp' in globals() and cp and isinstance(float_arr, cp.ndarray):
        float_arr = cp.asnumpy(float_arr)
    return np.clip(float_arr, 0, 255).astype(np.uint8)

def update_float_from_cv(uint8_result):
    """Convert uint8 result from OpenCV back to float32."""
    if uint8_result is None: return None
    return uint8_result.astype(np.float32)