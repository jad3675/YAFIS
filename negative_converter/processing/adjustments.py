# Image adjustment operations
import numpy as np
import cv2
import cv2.xphoto as xphoto  # Import the extended photo module

from ..utils.gpu import GPU_ENABLED, xp, cp_module as cp, is_cupy_backend
from ..utils.logger import get_logger
from ..utils.imaging import apply_curve

logger = get_logger(__name__)

# Import Preset Managers (needed for apply_all_adjustments)
try:
    from .film_simulation import FilmPresetManager
    from .photo_presets import PhotoPresetManager

    PRESET_MANAGERS_AVAILABLE = True
except ImportError:
    logger.warning("Could not import Preset Managers. Preset previews will not work.")
    PRESET_MANAGERS_AVAILABLE = False

    # Define dummy classes if needed downstream, though apply_all_adjustments will check the flag
    class FilmPresetManager:  # noqa: D401
        pass

    class PhotoPresetManager:  # noqa: D401
        pass

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
        use_gpu = is_cupy_backend()
        if use_gpu:
            arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else:
            arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            result_arr = img_arr + brightness_offset
            result_arr = arr_mod.clip(result_arr, 0, 255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Brightness failed (%s). Falling back to CPU.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr_cpu = image.astype(np.float32)
                result_arr_cpu = img_arr_cpu + brightness_offset
                return np.clip(result_arr_cpu, 0, 255).astype(np.uint8)
            return image.copy()

    @staticmethod
    def adjust_contrast(image, value):
        """Adjust image contrast (GPU/CPU)"""
        if image is None or image.size == 0: return image
        if value == 0: return image.copy()
        factor = (259.0 * (value + 255.0)) / (255.0 * (259.0 - value)) if value != 259 else 259.0
        mean_gray = 128.0
        use_gpu = is_cupy_backend()
        if use_gpu:
            arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else:
            arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            result_arr = factor * (img_arr - mean_gray) + mean_gray
            result_arr = arr_mod.clip(result_arr, 0, 255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Contrast failed (%s). Falling back to CPU.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr_cpu = image.astype(np.float32)
                result_arr_cpu = factor * (img_arr_cpu - mean_gray) + mean_gray
                return np.clip(result_arr_cpu, 0, 255).astype(np.uint8)
            return image.copy()

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
        use_gpu = is_cupy_backend()
        if use_gpu: arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else: arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            result_arr = img_arr.copy()
            result_arr[..., 0] += temp_factor
            result_arr[..., 2] -= temp_factor
            result_arr[..., 1] -= tint_factor
            result_arr = arr_mod.clip(result_arr, 0, 255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Temp/Tint failed (%s). Falling back to CPU.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr_cpu = image.astype(np.float32)
                result_arr_cpu = img_arr_cpu.copy()
                result_arr_cpu[..., 0] += temp_factor
                result_arr_cpu[..., 2] -= temp_factor
                result_arr_cpu[..., 1] -= tint_factor
                return np.clip(result_arr_cpu, 0, 255).astype(np.uint8)
            return image.copy()

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
            logger.warning("apply_curves expects uint8 input. Converting.")
            result = np.clip(result, 0, 255).astype(np.uint8)
        # Check for CuPy *after* ensuring uint8
        if "cupy" in str(type(result)):
            logger.warning("apply_curves expects NumPy input. Converting.")
            if is_cupy_backend() and cp is not None:
                result = cp.asnumpy(result)
            else:
                raise TypeError("Cannot convert non-GPU array using cp.asnumpy")

        def is_identity_curve(curve):
            """Check if a curve is effectively the identity curve (no change)."""
            if curve is None:
                return True
            if not isinstance(curve, list) or len(curve) < 2:
                return True
            # Check if it's exactly [[0, 0], [255, 255]]
            if len(curve) == 2:
                p0, p1 = curve[0], curve[1]
                if (p0[0] == 0 and p0[1] == 0 and p1[0] == 255 and p1[1] == 255):
                    return True
            # For curves with more points, check if all points lie on the identity line (y = x)
            for p in curve:
                if abs(p[0] - p[1]) > 1:  # Allow small tolerance
                    return False
            return True

        # Determine which curve to apply for each channel
        # Prioritize specific channel curve only if it's not None AND not the identity curve.
        # Otherwise, use the RGB curve if it's not None AND not the identity curve.
        r_curve_to_apply = None
        if not is_identity_curve(curve_points_r):
            r_curve_to_apply = curve_points_r
        elif not is_identity_curve(curve_points_rgb):
            r_curve_to_apply = curve_points_rgb

        g_curve_to_apply = None
        if not is_identity_curve(curve_points_g):
            g_curve_to_apply = curve_points_g
        elif not is_identity_curve(curve_points_rgb):
            g_curve_to_apply = curve_points_rgb

        b_curve_to_apply = None
        if not is_identity_curve(curve_points_b):
            b_curve_to_apply = curve_points_b
        elif not is_identity_curve(curve_points_rgb):
            b_curve_to_apply = curve_points_rgb

        # Apply the determined curve for each channel, with error handling
        try:
            if r_curve_to_apply:
                result[..., 0] = _apply_tone_curve_channel(result[..., 0], r_curve_to_apply)
        except Exception:
            logger.exception("Failed applying RED curve. Continuing.")

        try:
            if g_curve_to_apply:
                result[..., 1] = _apply_tone_curve_channel(result[..., 1], g_curve_to_apply)
        except Exception:
            logger.exception("Failed applying GREEN curve. Continuing.")

        try:
            if b_curve_to_apply:
                result[..., 2] = _apply_tone_curve_channel(result[..., 2], b_curve_to_apply)
        except Exception:
            logger.exception("Failed applying BLUE curve. Continuing.")

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
        use_gpu = is_cupy_backend()
        if use_gpu: arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else: arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            result_arr = img_arr.copy()
            result_arr[..., 0] += red_adj; result_arr[..., 1] -= green_adj; result_arr[..., 2] -= blue_adj
            result_arr = arr_mod.clip(result_arr, 0, 255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Additive Color Balance failed (%s). Falling back.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr = image.astype(np.float32)
                result_arr = img_arr.copy()
                result_arr[..., 0] += red_adj
                result_arr[..., 1] -= green_adj
                result_arr[..., 2] -= blue_adj
                return np.clip(result_arr, 0, 255).astype(np.uint8)
            return image.copy()

    @staticmethod
    def apply_color_balance(image, red_shift, green_shift, blue_shift, red_balance, green_balance, blue_balance):
        """Apply color balance adjustments (additive shifts + multiplicative balance) (GPU/CPU)"""
        if image is None or image.size == 0: return image
        is_identity = (red_shift == 0 and green_shift == 0 and blue_shift == 0 and red_balance == 1.0 and green_balance == 1.0 and blue_balance == 1.0)
        if is_identity: return image.copy()
        use_gpu = is_cupy_backend()
        if use_gpu: arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else: arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            result_arr = img_arr.copy()
            if red_shift!=0 or green_shift!=0 or blue_shift!=0: result_arr += arr_mod.array([red_shift, green_shift, blue_shift], dtype=arr_mod.float32)
            if red_balance!=1.0 or green_balance!=1.0 or blue_balance!=1.0: result_arr *= arr_mod.array([red_balance, green_balance, blue_balance], dtype=arr_mod.float32)
            result_arr = arr_mod.clip(result_arr, 0, 255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Color Balance failed (%s). Falling back.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr = image.astype(np.float32)
                result_arr = img_arr.copy()
                result_arr += np.array([red_shift, green_shift, blue_shift])
                result_arr *= np.array([red_balance, green_balance, blue_balance])
                return np.clip(result_arr, 0, 255).astype(np.uint8)
            return image.copy()

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
        use_gpu = is_cupy_backend()
        if use_gpu: arr_mod = cp; img_arr = arr_mod.asarray(image, dtype=arr_mod.float32)
        else: arr_mod = np; img_arr = image.astype(arr_mod.float32)
        try:
            r_in,g_in,b_in=img_arr[...,0],img_arr[...,1],img_arr[...,2]
            new_ch=(r_in*r_f+g_in*g_f+b_in*b_f+const_v)
            result_arr=img_arr.copy(); result_arr[...,out_idx]=new_ch
            result_arr=arr_mod.clip(result_arr,0,255)
            return arr_mod.asnumpy(result_arr).astype(np.uint8) if use_gpu else result_arr.astype(np.uint8)
        except Exception:
            logger.exception(
                "Channel Mixer failed (%s). Falling back.",
                "GPU" if use_gpu else "CPU",
            )
            if use_gpu:
                img_arr = image.astype(np.float32)
                r_in, g_in, b_in = img_arr[..., 0], img_arr[..., 1], img_arr[..., 2]
                new_ch = (r_in * r_f + g_in * g_f + b_in * b_f + const_v)
                result_arr = img_arr.copy()
                result_arr[..., out_idx] = new_ch
                return np.clip(result_arr, 0, 255).astype(np.uint8)
            return image.copy()

    @staticmethod
    def apply_noise_reduction(image, strength=10, template_window=7, search_window=21, fast_mode=True):
        """
        Applies noise reduction to color images.
        
        Args:
            image: Input RGB uint8 image
            strength: Denoising strength (1-100)
            template_window: Template window size for NLM (odd number)
            search_window: Search window size for NLM (odd number)
            fast_mode: If True, uses faster bilateral filter instead of NLM
            
        Note: Non-local Means (NLM) provides better quality but is very slow.
              Bilateral filter is much faster with good results for most cases.
        """
        if image is None or image.size == 0 or strength <= 0: 
            return image.copy()

        try:
            if fast_mode:
                # Fast mode: Use bilateral filter (much faster, good quality)
                # Map strength (1-100) to bilateral filter parameters
                # d: diameter of pixel neighborhood (higher = slower but better)
                # sigmaColor: filter sigma in color space
                # sigmaSpace: filter sigma in coordinate space
                d = min(9, max(3, int(strength / 10)))  # 3-9 based on strength
                sigma_color = strength * 2  # 2-200
                sigma_space = strength * 2  # 2-200
                
                img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                denoised_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
                return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
            else:
                # Quality mode: Use Non-local Means (slow but high quality)
                # For large images, process at reduced resolution
                h, w = image.shape[:2]
                max_pixels = 2_000_000  # 2 MP threshold
                
                if h * w > max_pixels:
                    # Downscale for processing
                    scale = np.sqrt(max_pixels / (h * w))
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    img_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)
                    denoised_bgr = cv2.fastNlMeansDenoisingColored(
                        img_bgr, None, 
                        h=float(strength), hColor=float(strength),
                        templateWindowSize=template_window, 
                        searchWindowSize=search_window
                    )
                    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
                    
                    # Upscale back
                    return cv2.resize(denoised_rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
                else:
                    # Process at full resolution
                    # Ensure window sizes are odd and valid
                    if template_window % 2 == 0: template_window += 1
                    if search_window % 2 == 0: search_window += 1
                    template_window = max(3, template_window)
                    search_window = max(template_window, search_window)
                    
                    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    denoised_bgr = cv2.fastNlMeansDenoisingColored(
                        img_bgr, None, 
                        h=float(strength), hColor=float(strength),
                        templateWindowSize=template_window, 
                        searchWindowSize=search_window
                    )
                    return cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
                    
        except cv2.error:
            logger.exception("Noise Reduction failed.")
            return image.copy()
        except Exception:
            logger.exception("Noise Reduction unexpected error.")
            return image.copy()

    @staticmethod
    def apply_dust_removal(image, sensitivity=50, radius=3):
        """
        Attempts to remove small dark spots (dust) using morphological operations and inpainting.
        Expects uint8 RGB image.
        Sensitivity controls the threshold for detecting spots (higher = more sensitive).
        Radius controls the inpainting neighborhood size.
        """
        if image is None or image.size == 0 or sensitivity <= 0:
            return image.copy()
        # Ensure radius is positive
        radius = max(1, int(radius))

        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # --- Use Median Filter Difference for Detection ---
            # Apply median filter. Kernel size should be odd and related to radius.
            median_ksize = max(3, int(radius) * 2 + 1) # Ensure odd, minimum 3
            median_filtered = cv2.medianBlur(gray, median_ksize)

            # Calculate absolute difference
            diff_image = cv2.absdiff(gray, median_filtered)

            # Threshold the difference image.
            # Higher sensitivity means a lower threshold, detecting smaller differences.
            # Map sensitivity (0-100) to threshold (e.g., 30 down to 1). Needs tuning.
            threshold_value = max(1, int(30 - (sensitivity * 29 / 100)))
            _, mask = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)

            # --- Contour Filtering (Re-enabled) ---
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_mask = np.zeros_like(mask) # Start with an empty mask

            # Define min/max area thresholds (WIDENED RANGE FOR DIAGNOSTICS)
            min_area = 1  # Allow single pixels
            max_area = 1000 # Allow much larger areas (adjust as needed)

            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    valid_contours.append(contour)

            # Draw the valid contours onto the filtered mask
            cv2.drawContours(filtered_mask, valid_contours, -1, (255), thickness=cv2.FILLED)

            # Optional: Dilate the filtered mask slightly
            # kernel_size = max(3, int(radius) * 2 + 1) # Need kernel if dilating
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            # filtered_mask = cv2.dilate(filtered_mask, kernel, iterations=1)

            # Inpaint using the original image and the *filtered* mask
            # cv2.INPAINT_TELEA is generally faster, cv2.INPAINT_NS might be higher quality
            inpainted_image = cv2.inpaint(image, filtered_mask, int(radius), cv2.INPAINT_NS) # Use NS algorithm

            return inpainted_image

        except cv2.error:
            logger.exception("Dust Removal (cv2) failed.")
            return image.copy()
        except Exception:
            logger.exception("Dust Removal unexpected error.")
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
            logger.warning(
                "Selective Color for '%s' not implemented or invalid range.",
                color_range,
            )
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

        # --- CMYK Simulation (Approximate) ---
        # Convert RGB to CMY (subtractive)
        c_orig = 1.0 - r_masked
        m_orig = 1.0 - g_masked
        y_orig = 1.0 - b_masked

        # Calculate K (black component) - simplest method
        k_orig = np.minimum.reduce([c_orig, m_orig, y_orig])

        # Remove K from C, M, Y
        c_pure = c_orig - k_orig
        m_pure = m_orig - k_orig
        y_pure = y_orig - k_orig

        # --- Apply Adjustments ---
        if relative:
            # Adjust C, M, Y relative to their current amount
            c_new = c_pure * (1 + c_adj)
            m_new = m_pure * (1 + m_adj)
            y_new = y_pure * (1 + y_adj)
        else:
            # Adjust C, M, Y absolutely (add percentage of total possible color)
            c_new = c_pure + c_adj
            m_new = m_pure + m_adj
            y_new = y_pure + y_adj

        # Adjust K
        k_new = k_orig * (1 - k_adj) # Black adjustment reduces black

        # Clip CMY values (0 to 1-K) - Ensure they don't exceed max possible after K removal
        max_color = 1.0 - k_new
        c_new = np.clip(c_new, 0, max_color)
        m_new = np.clip(m_new, 0, max_color)
        y_new = np.clip(y_new, 0, max_color)
        k_new = np.clip(k_new, 0, 1) # Clip K between 0 and 1

        # --- Convert back to RGB ---
        # Add K back to C, M, Y
        c_final = c_new + k_new
        m_final = m_new + k_new
        y_final = y_new + k_new

        # Convert CMY back to RGB
        r_new = 1.0 - c_final
        g_new = 1.0 - m_final
        b_new = 1.0 - y_final

        # Apply the modified RGB values back to the result image using the mask
        result_float[..., 0][mask] = r_new
        result_float[..., 1][mask] = g_new
        result_float[..., 2][mask] = b_new

        # Clip final RGB result and convert back to uint8
        result_uint8 = (np.clip(result_float, 0, 1) * 255).astype(np.uint8)

        return result_uint8

    @staticmethod
    def adjust_vibrance(image, value):
        """Adjust image vibrance (boosts less saturated colors more)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        s = hsv[..., 1]; factor = value / 100.0
        boost = factor * (1.0 - (s / 255.0)) # More boost for lower saturation
        hsv[..., 1] = np.clip(s * (1.0 + boost), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    @staticmethod
    def adjust_clarity(image, value):
        """Adjust image clarity (local contrast enhancement)"""
        if image is None or image.size == 0 or value == 0: return image.copy()
        # Simple clarity: Unsharp masking with a large radius
        # More advanced methods exist (e.g., using bilateral filter differences)
        factor = value / 100.0
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(1, int(10 * abs(factor))))
        if value > 0: # Increase clarity
            result = cv2.addWeighted(image, 1.0 + factor * 0.5, blurred, -factor * 0.5, 0)
        else: # Decrease clarity (blur) - use a smaller factor for blur
            result = cv2.addWeighted(image, 1.0 + factor * 0.2, blurred, -factor * 0.2, 0)
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_vignette(image, amount, center_x=0.5, center_y=0.5, radius=0.7, feather=0.3, color=None):
        """Applies a vignette effect (darkening or lightening corners)."""
        if image is None or image.size == 0 or amount == 0: return image.copy()
        rows, cols = image.shape[:2]
        center_x_px = int(center_x * cols); center_y_px = int(center_y * rows)
        # Ensure radius and feather are within reasonable bounds
        radius = max(0.01, min(radius, 1.5)) # Allow radius slightly > 1 for full coverage
        feather = max(0.01, min(feather, 1.0))
        # Calculate max distance based on corners for normalization
        max_dist = np.sqrt(max(center_x_px**2 + center_y_px**2,
                           (cols - center_x_px)**2 + center_y_px**2,
                           center_x_px**2 + (rows - center_y_px)**2,
                           (cols - center_x_px)**2 + (rows - center_y_px)**2))
        # Create coordinate grids
        x = np.arange(cols); y = np.arange(rows)
        xx, yy = np.meshgrid(x, y)
        # Calculate distance from center, normalize by max distance
        dist = np.sqrt((xx - center_x_px)**2 + (yy - center_y_px)**2) / max_dist
        # Calculate vignette mask based on radius and feather
        inner_radius = radius * (1.0 - feather)
        mask = np.clip((dist - inner_radius) / (radius - inner_radius), 0.0, 1.0)
        # Apply amount (positive darkens, negative lightens)
        vignette_factor = 1.0 - mask * (amount / 100.0)
        # Ensure factor doesn't go below zero for darkening
        if amount > 0: vignette_factor = np.maximum(0, vignette_factor)
        # Apply to image channels
        result = image.astype(np.float32)
        for i in range(3): result[..., i] *= vignette_factor
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_bw_mix(image, red_weight, green_weight, blue_weight):
        """Converts image to B&W using specified channel weights."""
        if image is None or image.size == 0: return image.copy()
        total = red_weight + green_weight + blue_weight
        if total == 0: total = 1 # Avoid division by zero, default to equal mix
        r_w = red_weight / total; g_w = green_weight / total; b_w = blue_weight / total
        gray = cv2.transform(image, np.array([[r_w, g_w, b_w]], dtype=np.float32))
        # Convert single channel gray back to 3-channel gray for consistency
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def apply_color_grading(image, shadows_rgb, midtones_rgb, highlights_rgb):
        """
        Applies simple color grading to shadows, midtones, and highlights.

        Expects:
          - image: uint8 RGB
          - shadows_rgb / midtones_rgb / highlights_rgb: list/tuple of 3 floats
            representing additive RGB offsets in 0..1 space (e.g., 0.05 == +5%).
        """
        if image is None or image.size == 0:
            return image.copy() if image is not None else None

        # Normalize and validate inputs
        def _vec3(v):
            if v is None:
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
            logger.warning("apply_color_grading: invalid vector %r; using zeros", v)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        sh = _vec3(shadows_rgb)
        mid = _vec3(midtones_rgb)
        hi = _vec3(highlights_rgb)

        if np.allclose(sh, 0) and np.allclose(mid, 0) and np.allclose(hi, 0):
            return image.copy()

        # Work in float 0..1
        img = image.astype(np.float32) / 255.0

        # Simple luminance proxy (Rec.709)
        lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)

        # Build smooth masks for tonal ranges.
        # shadows: 1 at 0.0 -> 0 at ~0.5
        # highlights: 0 at ~0.5 -> 1 at 1.0
        # midtones: what's left (clipped)
        shadows_w = np.clip((0.5 - lum) / 0.5, 0.0, 1.0)
        highlights_w = np.clip((lum - 0.5) / 0.5, 0.0, 1.0)
        midtones_w = np.clip(1.0 - shadows_w - highlights_w, 0.0, 1.0)

        # Apply additive grading per range
        graded = img.copy()
        graded += shadows_w[..., None] * sh[None, None, :]
        graded += midtones_w[..., None] * mid[None, None, :]
        graded += highlights_w[..., None] * hi[None, None, :]

        return (np.clip(graded, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def apply_film_grain(image_float, intensity, size, roughness):
        """
        Adds simulated film grain to a float image (0.0-1.0 range).
        Intensity: Controls the visibility/contrast of the grain (e.g., 0-50).
        Size: Controls the scale of the grain noise pattern (e.g., 1.0-3.0).
        Roughness: Controls the uniformity of the grain (lower is more uniform).
        """
        if image_float is None or image_float.size == 0 or intensity <= 0:
            return image_float

        height, width = image_float.shape[:2]
        scale = max(1.0, size) # Ensure scale is at least 1

        # Generate Perlin-like noise for a more natural grain look
        # Create low-res noise and upscale it
        low_res_w, low_res_h = int(width / scale), int(height / scale)
        # Ensure low-res dimensions are at least 1
        low_res_w = max(1, low_res_w)
        low_res_h = max(1, low_res_h)

        # Generate base noise (adjust range based on roughness)
        noise_range = 0.5 + (1.0 - max(0.1, roughness)) * 0.5 # Roughness 1->0.5 range, 0.1->1.0 range
        noise = np.random.uniform(0.5 - noise_range / 2, 0.5 + noise_range / 2,
                                  (low_res_h, low_res_w, image_float.shape[2] if len(image_float.shape) > 2 else 1))

        # Upscale noise using linear interpolation
        grain = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        # Ensure grain has the same number of channels as the image if it's color
        if len(image_float.shape) == 3 and len(grain.shape) == 2:
            grain = cv2.cvtColor(grain, cv2.COLOR_GRAY2RGB)
        elif len(image_float.shape) == 2 and len(grain.shape) == 3:
             grain = cv2.cvtColor(grain, cv2.COLOR_RGB2GRAY) # Should not happen with current noise gen

        # Ensure grain shape matches image shape if channels were added/removed
        if grain.shape != image_float.shape:
             # Attempt to reshape or broadcast if channel mismatch occurred during resize/conversion
             if len(image_float.shape) == 3 and len(grain.shape) == 2:
                 grain = grain[..., np.newaxis] # Add channel dim
             elif len(image_float.shape) == 2 and len(grain.shape) == 3:
                 grain = np.mean(grain, axis=2) # Average channels if image is grayscale

             # Final check, if still mismatch, log error and return original
             if grain.shape != image_float.shape:
                  logger.error(
                      "Grain shape mismatch after resize/conversion: Img=%s Grain=%s",
                      image_float.shape,
                      grain.shape,
                  )
                  return image_float


        # Apply grain: Add noise centered around 0 to the image
        # Intensity scales the noise range. Map intensity (0-50) to a factor (e.g., 0-0.5)
        intensity_factor = intensity / 100.0
        grain_applied = image_float + (grain - 0.5) * intensity_factor

        return np.clip(grain_applied, 0.0, 1.0)


    # --- Auto Adjustment Methods ---
    # These often require the image data itself

    @staticmethod
    def _calculate_gray_world_scales(image):
        """Calculates scaling factors based on Gray World assumption."""
        if image is None or image.size == 0: return None
        # Calculate average for each channel, avoiding pure black/white pixels slightly
        mask = (image > 10) & (image < 245)
        masked_image = image * mask
        avg_r = np.mean(masked_image[..., 0][mask[..., 0]])
        avg_g = np.mean(masked_image[..., 1][mask[..., 1]])
        avg_b = np.mean(masked_image[..., 2][mask[..., 2]])
        if avg_r == 0 or avg_g == 0 or avg_b == 0: return None # Avoid division by zero
        avg_gray = (avg_r + avg_g + avg_b) / 3.0
        # Scale factors to make average gray
        scale_r = avg_gray / avg_r
        scale_g = avg_gray / avg_g
        scale_b = avg_gray / avg_b
        return scale_r, scale_g, scale_b

    @staticmethod
    def _calculate_white_patch_scales(image, percentile=99):
        """Calculates scaling factors based on White Patch (brightest pixel) assumption."""
        if image is None or image.size == 0: return None
        # Find near-white point for each channel
        max_r = np.percentile(image[..., 0], percentile)
        max_g = np.percentile(image[..., 1], percentile)
        max_b = np.percentile(image[..., 2], percentile)
        if max_r == 0 or max_g == 0 or max_b == 0: return None # Avoid division by zero
        # Target white (usually 255)
        target_white = 255.0
        # Scale factors to make the percentile white
        scale_r = target_white / max_r
        scale_g = target_white / max_g
        scale_b = target_white / max_b
        return scale_r, scale_g, scale_b

    @staticmethod
    def calculate_white_balance_from_color(image, picked_color_rgb):
        """Calculates WB scales needed to make the picked color neutral gray."""
        if image is None or picked_color_rgb is None: return None
        r, g, b = picked_color_rgb
        if r == 0 or g == 0 or b == 0: return None # Avoid division by zero
        avg_gray = (r + g + b) / 3.0
        scale_r = avg_gray / r
        scale_g = avg_gray / g
        scale_b = avg_gray / b
        return scale_r, scale_g, scale_b


    # --- OpenCV xphoto White Balance Methods ---
    @staticmethod
    def _apply_simple_wb(image):
        """Applies OpenCV's SimpleWB."""
        if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'createSimpleWB'): return None
        try:
            wb = cv2.xphoto.createSimpleWB()
            # SimpleWB might work better on linear data, but try on sRGB first
            return wb.balanceWhite(image)
        except cv2.error:
            logger.exception("SimpleWB failed.")
            return None

    @staticmethod
    def _apply_learning_based_wb(image):
        """Applies OpenCV's LearningBasedWB."""
        if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'createLearningBasedWB'): return None
        try:
            wb = cv2.xphoto.createLearningBasedWB()
            # May need a path to a model file if not using default
            # wb.loadModel("path/to/model")
            return wb.balanceWhite(image)
        except cv2.error:
            logger.exception("LearningBasedWB failed.")
            return None

    # --- Combined Auto White Balance ---
    @staticmethod
    def apply_auto_white_balance(image, method='gray_world', percentile=99):
        """Applies automatic white balance using various methods."""
        if image is None or image.size == 0: return image.copy()
        original_image = image.copy() # Keep original in case of failure

        scales = None
        if method == 'gray_world':
            scales = AdvancedAdjustments._calculate_gray_world_scales(image)
        elif method == 'white_patch':
            scales = AdvancedAdjustments._calculate_white_patch_scales(image, percentile)
        elif method == 'simple_wb':
            return AdvancedAdjustments._apply_simple_wb(image) or original_image # Return original on failure
        elif method == 'learning_wb':
            return AdvancedAdjustments._apply_learning_based_wb(image) or original_image # Return original on failure
        else:
            logger.warning("Unknown AWB method: %s", method)
            return original_image

        if scales is None:
            logger.warning("AWB method '%s' failed to calculate scales.", method)
            return original_image

        scale_r, scale_g, scale_b = scales
        # Apply scales (similar to color balance multiplication)
        # Use float32 for calculation to avoid clipping issues before final clip
        img_float = image.astype(np.float32)
        img_float[..., 0] *= scale_r
        img_float[..., 1] *= scale_g
        img_float[..., 2] *= scale_b
        return np.clip(img_float, 0, 255).astype(np.uint8)

    # --- Auto Levels ---
    @staticmethod
    def calculate_auto_levels_params(image, colorspace_mode='luminance', midrange=0.5):
        """
        Calculates automatic levels adjustment parameters based on histogram stretching.
        Returns a dictionary: {'levels_in_black': int, 'levels_in_white': int, 'levels_gamma': float}
        Calculation is based on the Luminance channel (LAB) for consistency with UI sliders.
        midrange: Target midpoint for gamma adjustment (0.01 to 0.99)
        """
        if image is None or image.size == 0:
            return {'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0}

        midrange = np.clip(midrange, 0.01, 0.99)

        try:
            # --- Determine Channel for Histogram based on colorspace_mode ---
            img_float = image.astype(np.float32) # Use float for calculations

            if colorspace_mode == 'luminance' or colorspace_mode == 'magnitude' or colorspace_mode == 'rgb':
                # Calculate Rec709Luma: 0.2126*R + 0.7152*G + 0.0722*B
                # Note: 'magnitude' and 'rgb' modes are proxied using Luminance for single-channel analysis
                if colorspace_mode != 'luminance':
                     logger.info("Auto Levels mode '%s' proxied using Rec709Luma.", colorspace_mode)
                channel = (0.2126 * img_float[:,:,0] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,2]).astype(np.uint8)
            elif colorspace_mode == 'lightness':
                # Convert to HLS and use the L channel (index 1)
                hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                channel = hls[:, :, 1]
            elif colorspace_mode == 'brightness':
                # Convert to HSV and use the V channel (index 2)
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                channel = hsv[:, :, 2]
            elif colorspace_mode == 'gray':
                # Convert to Grayscale
                channel = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif colorspace_mode == 'average':
                # Calculate the average of R, G, B channels (proxy for OHTA)
                channel = np.mean(img_float, axis=2).astype(np.uint8)
            else:
                logger.warning("Unknown colorspace_mode '%s'. Defaulting to Rec709Luma.", colorspace_mode)
                channel = (0.2126 * img_float[:,:,0] + 0.7152 * img_float[:,:,1] + 0.0722 * img_float[:,:,2]).astype(np.uint8)
            # --- End Channel Determination ---

            # Calculate histogram and CDF
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            cdf = hist.cumsum()

            # Find black and white points using CDF
            cdf_m = np.ma.masked_equal(cdf, 0)

            # Handle flat CDF (avoid division by zero or errors in min/max)
            min_cdf = cdf_m.min()
            max_cdf = cdf_m.max()
            if max_cdf is np.ma.masked or min_cdf is np.ma.masked or (max_cdf - min_cdf) == 0:
                in_black, in_white = 0, 255
            else:
                cdf_normalized = (cdf_m - min_cdf) * 255 / (max_cdf - min_cdf)
                cdf = np.ma.filled(cdf_normalized, 0).astype('uint8')
                try:
                    # Find first pixel value > 1 and last pixel value < 254 in normalized CDF
                    in_black = np.min(np.where(cdf > 1))
                    in_white = np.max(np.where(cdf < 254))
                except ValueError: # Handle cases where channel is flat or nearly flat after normalization
                    in_black, in_white = 0, 255

            # Adjust and ensure validity
            in_black = max(0, in_black - 1)
            in_white = min(255, in_white + 1)
            if in_white <= in_black:
                in_white = in_black + 1

            # Calculate gamma based on midrange target
            median_val = np.median(channel)
            gamma = 1.0 # Default gamma
            if median_val > in_black and median_val < in_white and in_white > in_black:
                normalized_median = (median_val - in_black) / (in_white - in_black)
                # Ensure arguments to log are valid
                if normalized_median > 0 and normalized_median < 1 and midrange > 0 and midrange < 1:
                    try:
                        # Use log base 10 or natural log, doesn't matter as it cancels
                        gamma = np.log(midrange) / np.log(normalized_median)
                        gamma = np.clip(gamma, 0.1, 10.0) # Clamp gamma to reasonable range
                    except (ValueError, ZeroDivisionError):
                        gamma = 1.0 # Fallback if log calculation fails
            
            # Return the calculated parameters
            return {'levels_in_black': int(round(in_black)),
                    'levels_in_white': int(round(in_white)),
                    'levels_gamma': round(gamma, 2)} # Round gamma for UI

        except cv2.error:
            logger.exception("OpenCV error during Auto Levels calculation.")
            return {'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0}
        except Exception:
            logger.exception("Unexpected error during Auto Levels calculation.")
            return {'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0}

    # --- Auto Color ---
    @staticmethod
    def calculate_auto_color_params(image, method='gamma'):
        """
        Calculates adjustment parameters based on the selected auto color method.
        - 'gamma': Calculates per-channel gamma to neutralize color cast and returns
                   approximate 3-point curves for R, G, B channels.
                   {'curves_red': [[0,0],[128,mid_r],[255,255]], ...}
        - 'recolor': Calculates overall color cast shift (like Grayworld) and returns
                     Temp/Tint parameters to approximate it.
                     {'temp': val, 'tint': val}
        - 'none': Returns default parameters (identity curves, zero temp/tint).
        """
        # Define default identity curve and params
        identity_curve = [[0, 0], [255, 255]]
        default_params = {
            'curves_red': identity_curve, 'curves_green': identity_curve, 'curves_blue': identity_curve,
            'temp': 0, 'tint': 0
        }

        if image is None or image.size == 0:
            return default_params

        try:
            img_float = image.astype(np.float32)
            means = np.mean(img_float, axis=(0, 1))
            mean_r, mean_g, mean_b = means[0], means[1], means[2]

            # Calculate neutral gray (mean luminance - simple average for this)
            neutral_gray = np.mean(means)
            if neutral_gray < 1e-5: neutral_gray = 1e-5 # Avoid division by zero

            if method == 'gamma':
                params = {}
                # Calculate per-channel gamma
                for i, mean_val in enumerate([mean_r, mean_g, mean_b]):
                    channel_name = ['red', 'green', 'blue'][i]
                    curve_key = f'curves_{channel_name}'
                    gamma = 1.0 # Default
                    if mean_val > 1e-5: # Avoid division by zero and log(0)
                        try:
                            # Formula: target_mean = mean ^ gamma => neutral_gray = mean ^ gamma
                            # log(neutral_gray) = gamma * log(mean) => gamma = log(neutral_gray) / log(mean)
                            # Normalize to 0-1 range for pow calculation later
                            norm_mean = mean_val / 255.0
                            norm_gray = neutral_gray / 255.0
                            if norm_mean > 0 and norm_gray > 0: # Ensure log args are positive
                                gamma = np.log(norm_gray) / np.log(norm_mean)
                                gamma = np.clip(gamma, 0.1, 10.0) # Clamp gamma
                        except (ValueError, ZeroDivisionError, RuntimeWarning):
                            gamma = 1.0 # Fallback

                    # Calculate midpoint for 3-point curve approximation
                    # mid_val = round(pow(0.5, 1.0/gamma) * 255.0) # Inverse gamma applied to midpoint
                    # Simpler approximation: apply gamma to midpoint 128
                    mid_val = round(pow(128.0 / 255.0, gamma) * 255.0)
                    mid_val = np.clip(mid_val, 0, 255) # Ensure valid range

                    params[curve_key] = [[0, 0], [128, int(mid_val)], [255, 255]]
                    logger.debug(
                        "Auto Color (Gamma): Channel=%s Mean=%.1f TargetGray=%.1f Gamma=%.2f CurveMid=%s",
                        channel_name,
                        mean_val,
                        neutral_gray,
                        gamma,
                        mid_val,
                    )

                # Ensure temp/tint are reset when applying curves
                params['temp'] = 0
                params['tint'] = 0
                return params

            elif method == 'recolor':
                # Calculate scaling factors needed to push means towards neutral gray
                scale_r = neutral_gray / mean_r if mean_r > 1e-5 else 1.0
                scale_g = neutral_gray / mean_g if mean_g > 1e-5 else 1.0
                scale_b = neutral_gray / mean_b if mean_b > 1e-5 else 1.0

                # Estimate the resulting color shift (difference from original means)
                # This is complex to map perfectly to Temp/Tint. We approximate based on
                # the *intended* shift towards gray.
                # If scale_b > scale_r, it means blue was lower and needs boosting more -> warmer temp needed -> negative temp slider
                # If scale_g > avg(scale_r, scale_b), green needs boosting more -> more magenta needed -> negative tint slider
                temp_diff_effect = (scale_b - scale_r) * neutral_gray # Approximate effect difference
                tint_diff_effect = (scale_g - (scale_r + scale_b)/2.0) * neutral_gray

                # Estimate slider values (inverse relationship, scaled differently)
                temp_adj = int(round(np.clip(-temp_diff_effect / 0.6, -100, 100))) # Heuristic scaling
                tint_adj = int(round(np.clip(-tint_diff_effect / 0.3, -100, 100))) # Heuristic scaling
                logger.debug(
                    "Auto Color (Recolor): Scales=(%.2f,%.2f,%.2f) Est. Temp=%s Est. Tint=%s",
                    scale_r,
                    scale_g,
                    scale_b,
                    temp_adj,
                    tint_adj,
                )

                # Return temp/tint and reset curves
                return {
                    'temp': temp_adj, 'tint': tint_adj,
                    'curves_red': identity_curve, 'curves_green': identity_curve, 'curves_blue': identity_curve
                }

            elif method == 'none':
                 return default_params # Return defaults to reset everything
            else:
                 logger.warning("Unknown Auto Color method: %s. Returning defaults.", method)
                 return default_params

        except Exception:
            logger.exception("Auto Color calculation unexpected error.")
            return default_params

    @staticmethod
    def calculate_gray_world_awb_params(image):
        """
        Calculates Temp/Tint adjustments based on the Grayworld assumption
        (average color of the scene is neutral gray).
        Returns a dictionary: {'temp': int, 'tint': int}
        """
        if image is None or image.size == 0:
            return {'temp': 0, 'tint': 0}

        try:
            img_float = image.astype(np.float32)
            means = np.mean(img_float, axis=(0, 1))
            mean_r, mean_g, mean_b = means[0], means[1], means[2]

            # Neutral gray is the average of the channel means
            neutral_gray = np.mean(means)
            if neutral_gray < 1e-5: neutral_gray = 1e-5 # Avoid division by zero

            # Calculate scaling factors needed
            scale_r = neutral_gray / mean_r if mean_r > 1e-5 else 1.0
            scale_g = neutral_gray / mean_g if mean_g > 1e-5 else 1.0
            scale_b = neutral_gray / mean_b if mean_b > 1e-5 else 1.0

            # Estimate Temp/Tint adjustments
            temp_diff_effect = (scale_b - scale_r) * neutral_gray
            tint_diff_effect = (scale_g - (scale_r + scale_b)/2.0) * neutral_gray
            temp_adj = int(round(np.clip(-temp_diff_effect / 0.6, -100, 100)))
            tint_adj = int(round(np.clip(-tint_diff_effect / 0.3, -100, 100)))

            logger.debug(
                "Gray World AWB: TargetGray=%.1f Scales=(%.2f,%.2f,%.2f) Est. Temp=%s Est. Tint=%s",
                neutral_gray,
                scale_r,
                scale_g,
                scale_b,
                temp_adj,
                tint_adj,
            )
            return {'temp': temp_adj, 'tint': tint_adj}

        except Exception:
            logger.exception("Gray World AWB calculation unexpected error.")
            return {'temp': 0, 'tint': 0}

    @staticmethod
    def calculate_white_patch_awb_params(image, percentile=1.0):
        """
        Calculates Temp/Tint adjustments based on the White Patch assumption
        (brightest pixels should be white).
        Percentile determines the percentage of brightest pixels to exclude (0.0 to 10.0).
        Returns a dictionary: {'temp': int, 'tint': int}
        """
        if image is None or image.size == 0:
            return {'temp': 0, 'tint': 0}

        percentile = np.clip(percentile, 0.0, 10.0) # Percentile to exclude

        try:
            img_float = image.astype(np.float32)
            # Exclude top percentile brightest pixels if needed
            if percentile > 0:
                # Calculate overall brightness (e.g., max channel value per pixel)
                brightness = np.max(img_float, axis=2)
                num_pixels = image.shape[0] * image.shape[1]
                k = int(num_pixels * (percentile / 100.0))
                k = max(1, min(k, num_pixels -1)) # Ensure k is valid and leaves at least one pixel
                # Find brightness threshold to exclude top k pixels
                brightness_threshold = np.partition(brightness.flatten(), -k)[-k]
                # Create mask for pixels *below* the threshold
                mask = brightness < brightness_threshold
                if not np.any(mask): # If all pixels are above threshold, use all pixels
                     mask = np.ones(brightness.shape, dtype=bool)
            else:
                mask = np.ones((image.shape[0], image.shape[1]), dtype=bool) # Use all pixels

            # Find max R, G, B within the masked area
            max_r = np.max(img_float[:,:,0][mask])
            max_g = np.max(img_float[:,:,1][mask])
            max_b = np.max(img_float[:,:,2][mask])

            # Target is 255 (or max possible value)
            target_white = 255.0

            # Calculate scaling factors
            scale_r = target_white / max_r if max_r > 1e-5 else 1.0
            scale_g = target_white / max_g if max_g > 1e-5 else 1.0
            scale_b = target_white / max_b if max_b > 1e-5 else 1.0

            # Estimate Temp/Tint adjustments
            # Use the *target* white as the reference for calculating effect difference
            temp_diff_effect = (scale_b - scale_r) * target_white
            tint_diff_effect = (scale_g - (scale_r + scale_b)/2.0) * target_white
            temp_adj = int(round(np.clip(-temp_diff_effect / 0.6, -100, 100)))
            tint_adj = int(round(np.clip(-tint_diff_effect / 0.3, -100, 100)))

            logger.debug(
                "White Patch AWB (%s%% excluded): Target=255 Scales=(%.2f,%.2f,%.2f) Est. Temp=%s Est. Tint=%s",
                percentile,
                scale_r,
                scale_g,
                scale_b,
                temp_adj,
                tint_adj,
            )
            return {'temp': temp_adj, 'tint': tint_adj}

        except Exception:
            logger.exception("White Patch AWB calculation unexpected error.")
            return {'temp': 0, 'tint': 0}

    @staticmethod
    def calculate_simple_wb_params(image):
        """
        Calculates Temp/Tint adjustments based on OpenCV's SimpleWB algorithm.
        Returns a dictionary: {'temp': int, 'tint': int}
        """
        if image is None or image.size == 0: return {'temp': 0, 'tint': 0}
        if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'createSimpleWB'):
            logger.error("SimpleWB requires 'opencv-contrib-python'.")
            return {'temp': 0, 'tint': 0}

        try:
            original_image = image.copy()
            img_float = image.astype(np.float32) # SimpleWB might prefer float

            wb = cv2.xphoto.createSimpleWB()
            # Set parameters if needed, e.g., input/output max, p
            # wb.setInputMax(255.0)
            # wb.setOutputMax(255.0)
            # wb.setP(0.2) # Example p value
            result_float = wb.balanceWhite(img_float)

            # Calculate the change in RGB means
            orig_means = np.mean(img_float, axis=(0, 1))
            new_means = np.mean(result_float, axis=(0, 1))

            # Estimate Temp/Tint adjustments
            temp_diff = (new_means[2] - new_means[0]) - (orig_means[2] - orig_means[0])
            tint_diff = (new_means[1] - (new_means[0] + new_means[2])/2.0) - (orig_means[1] - (orig_means[0] + orig_means[2])/2.0)
            temp_adj = int(round(np.clip(-temp_diff / 0.6, -100, 100)))
            tint_adj = int(round(np.clip(-tint_diff / 0.3, -100, 100)))

            logger.debug("SimpleWB AWB: Est. Temp=%s Est. Tint=%s", temp_adj, tint_adj)
            return {'temp': temp_adj, 'tint': tint_adj}

        except cv2.error:
            logger.exception("SimpleWB calculation (cv2) failed.")
            return {'temp': 0, 'tint': 0}
        except Exception:
            logger.exception("SimpleWB calculation unexpected error.")
            return {'temp': 0, 'tint': 0}

    @staticmethod
    def calculate_learning_wb_params(image):
        """
        Calculates Temp/Tint adjustments based on OpenCV's LearningBasedWB algorithm.
        NOTE: May require a pre-trained model file. Uses default if possible.
        Returns a dictionary: {'temp': int, 'tint': int}
        """
        if image is None or image.size == 0: return {'temp': 0, 'tint': 0}
        if not hasattr(cv2, 'xphoto') or not hasattr(cv2.xphoto, 'createLearningBasedWB'):
            logger.error("LearningBasedWB requires 'opencv-contrib-python'.")
            return {'temp': 0, 'tint': 0}

        try:
            original_image = image.copy()
            # LearningBasedWB typically expects BGR uint8 input
            img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

            wb = cv2.xphoto.createLearningBasedWB()
            # Optional: Load a specific model: wb.loadModel("path/to/model.yml")
            # Optional: Set parameters like range_max_val, saturation_thresh, hist_bin_num
            # wb.setRangeMaxVal(255)
            result_bgr = wb.balanceWhite(img_bgr)
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

            # Calculate the change in RGB means
            orig_means = np.mean(original_image.astype(np.float32), axis=(0, 1))
            new_means = np.mean(result_rgb.astype(np.float32), axis=(0, 1))

            # Estimate Temp/Tint adjustments
            temp_diff = (new_means[2] - new_means[0]) - (orig_means[2] - orig_means[0])
            tint_diff = (new_means[1] - (new_means[0] + new_means[2])/2.0) - (orig_means[1] - (orig_means[0] + orig_means[2])/2.0)
            temp_adj = int(round(np.clip(-temp_diff / 0.6, -100, 100)))
            tint_adj = int(round(np.clip(-tint_diff / 0.3, -100, 100)))

            logger.debug("LearningBasedWB AWB: Est. Temp=%s Est. Tint=%s", temp_adj, tint_adj)
            return {'temp': temp_adj, 'tint': tint_adj}

        except cv2.error:
            logger.exception("LearningBasedWB calculation (cv2) failed. Might need model file?")
            return {'temp': 0, 'tint': 0}
        except Exception:
            logger.exception("LearningBasedWB calculation unexpected error.")
            return {'temp': 0, 'tint': 0}


    @staticmethod
    def calculate_near_white_awb_params(image, percentile=1.0):
        """
        Calculates Temp/Tint adjustments based on pixels closest to white
        (high value, low saturation).
        Percentile determines the percentage of near-white pixels to consider (0.1 to 10.0).
        Returns a dictionary: {'temp': int, 'tint': int}
        """
        if image is None or image.size == 0:
            return {'temp': 0, 'tint': 0}

        percentile = np.clip(percentile, 0.1, 10.0) # Ensure percentile is reasonable

        try:
            img_float = image.astype(np.float32)

            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

            # Create weight map: (1 - Saturation/255) * (Value/255)
            # Higher weight means brighter and less saturated
            # Add small epsilon to avoid issues with pure black/white
            epsilon = 1e-6
            weight_map = (1.0 - s / (255.0 + epsilon)) * (v / (255.0 + epsilon))

            # Find the threshold for the top percentile
            num_pixels = image.shape[0] * image.shape[1]
            k = int(num_pixels * (percentile / 100.0))
            k = max(1, min(k, num_pixels)) # Ensure k is valid

            flat_weights = weight_map.flatten()
            # Use partition which is generally faster than full sort for finding k-th element
            threshold = np.partition(flat_weights, -k)[-k]

            # Create mask for pixels above or equal to the threshold
            mask = weight_map >= threshold

            if not np.any(mask):
                logger.warning(
                    "Near-White AWB: no pixels found in the top percentile. Using Grayworld fallback."
                )
                # Fallback to Grayworld-like calculation if mask is empty
                means = np.mean(img_float, axis=(0, 1))
                mean_r, mean_g, mean_b = means[0], means[1], means[2]
                neutral_gray = np.mean(means)
            else:
                # Calculate mean R, G, B *within the masked area*
                # Use cv2.mean for efficiency with mask
                mean_r = cv2.mean(img_float[:,:,0], mask=mask.astype(np.uint8))[0]
                mean_g = cv2.mean(img_float[:,:,1], mask=mask.astype(np.uint8))[0]
                mean_b = cv2.mean(img_float[:,:,2], mask=mask.astype(np.uint8))[0]

                # Target gray is the max of the masked means (White Patch adaptation)
                neutral_gray = max(mean_r, mean_g, mean_b)

            # Ensure means and gray are valid for division
            if neutral_gray < 1e-5: neutral_gray = 1e-5

            # Calculate scaling factors needed
            scale_r = neutral_gray / mean_r if mean_r > 1e-5 else 1.0
            scale_g = neutral_gray / mean_g if mean_g > 1e-5 else 1.0
            scale_b = neutral_gray / mean_b if mean_b > 1e-5 else 1.0

            # Estimate Temp/Tint adjustments
            # Heuristic mapping from scale factors to Temp/Tint slider values
            temp_diff_effect = (scale_b - scale_r) * neutral_gray
            tint_diff_effect = (scale_g - (scale_r + scale_b)/2.0) * neutral_gray

            # These scaling factors (0.6, 0.3) are empirical based on adjust_temp_tint's effect
            temp_adj = int(round(np.clip(-temp_diff_effect / 0.6, -100, 100)))
            tint_adj = int(round(np.clip(-tint_diff_effect / 0.3, -100, 100)))

            logger.debug(
                "Near-White AWB (%s%%): TargetGray=%.1f Scales=(%.2f,%.2f,%.2f) Est. Temp=%s Est. Tint=%s",
                percentile,
                neutral_gray,
                scale_r,
                scale_g,
                scale_b,
                temp_adj,
                tint_adj,
            )
            return {'temp': temp_adj, 'tint': tint_adj}

        except cv2.error:
            logger.exception("Near-White AWB calculation (cv2) failed.")
            return {'temp': 0, 'tint': 0}
        except Exception:
            logger.exception("Near-White AWB calculation unexpected error.")
            return {'temp': 0, 'tint': 0}


    # --- Auto Tone ---
    @staticmethod
    def calculate_auto_tone_params(image, nr_strength=5, awb_method='gray_world', al_mode='luminance', al_midrange=0.5, clarity_strength=10):
        """
        Calculates parameters for several adjustments typically used in Auto Tone.
        Returns a dictionary of parameters for NR, AWB (Temp/Tint), and Auto Levels.
        Note: Clarity is not directly mapped to sliders here.
        """
        if image is None or image.size == 0:
            return {
                'noise_reduction_strength': 0, 'temp': 0, 'tint': 0,
                'levels_in_black': 0, 'levels_in_white': 255, 'levels_gamma': 1.0
            }

        logger.debug("--- Calculating Auto Tone Parameters ---")
        params = {}

        # 1. Noise Reduction Strength (Fixed value from defaults)
        params['noise_reduction_strength'] = nr_strength
        logger.debug("Auto Tone Param: Noise Reduction Strength=%s", nr_strength)

        # 2. Auto White Balance (Calculate Temp/Tint)
        logger.debug("Auto Tone Param: Calculating AWB (Method=%s)", awb_method)
        # Use the existing calculate_auto_white_balance_params (or similar logic)
        # Assuming calculate_auto_white_balance_params exists and returns {'temp': val, 'tint': val}
        # If not, we need to implement it based on apply_auto_white_balance logic
        try:
            # Re-use logic similar to calculate_auto_color_params
            awb_result = AdvancedAdjustments.apply_auto_white_balance(image, method=awb_method)
            if awb_result is not None:
                orig_means = np.mean(image.astype(np.float32), axis=(0, 1))
                new_means = np.mean(awb_result.astype(np.float32), axis=(0, 1))
                temp_diff = (new_means[2] - new_means[0]) - (orig_means[2] - orig_means[0])
                tint_diff = (new_means[1] - (new_means[0] + new_means[2])/2.0) - (orig_means[1] - (orig_means[0] + orig_means[2])/2.0)
                params['temp'] = int(round(np.clip(-temp_diff / 0.6, -100, 100)))
                params['tint'] = int(round(np.clip(-tint_diff / 0.3, -100, 100)))
                logger.debug("Auto Tone Param: Calculated Temp=%s Tint=%s", params['temp'], params['tint'])
            else:
                logger.debug("Auto Tone Param: AWB calculation failed, using defaults.")
                params['temp'] = 0
                params['tint'] = 0
        except Exception:
            logger.exception("Auto Tone Param: error calculating AWB params.")
            params['temp'] = 0
            params['tint'] = 0


        # 3. Auto Levels (Calculate Levels Params)
        logger.debug("Auto Tone Param: Calculating Auto Levels (Mode=%s Midrange=%s)", al_mode, al_midrange)
        try:
            level_params = AdvancedAdjustments.calculate_auto_levels_params(image, colorspace_mode=al_mode, midrange=al_midrange)
            params.update(level_params)
            logger.debug("Auto Tone Param: Calculated Levels=%s", level_params)
        except Exception:
            logger.exception("Auto Tone Param: error calculating Auto Levels params.")
            params['levels_in_black'] = 0
            params['levels_in_white'] = 255
            params['levels_gamma'] = 1.0

        # 4. Clarity (Not directly calculated for sliders)
        # The clarity adjustment will be applied based on its own slider value during the main processing.
        # We don't set a specific clarity value here.
        logger.debug(
            "Auto Tone Param: Clarity Strength (%s) will be applied by main pipeline if slider > 0.",
            clarity_strength,
        )

        logger.debug("--- Auto Tone Parameter Calculation Finished ---")
        return params


# --- Main Adjustment Application Function ---

# Preset managers used for preview inside apply_all_adjustments().
# Default behavior keeps a module-level instance, but the UI/service can inject the
# live manager(s) so newly saved presets are immediately available.
_film_preset_manager = FilmPresetManager() if PRESET_MANAGERS_AVAILABLE else None
_photo_preset_manager = PhotoPresetManager() if PRESET_MANAGERS_AVAILABLE else None


def set_film_preset_manager(manager):
    """Inject the film preset manager used for preview in apply_all_adjustments()."""
    global _film_preset_manager
    _film_preset_manager = manager


def set_photo_preset_manager(manager):
    """Inject the photo preset manager used for preview in apply_all_adjustments()."""
    global _photo_preset_manager
    _photo_preset_manager = manager


def _curve_points_to_lut(curve_points):
    """
    Convert curve control points to a 256-element LUT for GPU processing.
    
    Args:
        curve_points: List of (x, y) tuples where x,y are in 0-255 range
        
    Returns:
        numpy array of 256 float32 values representing the curve LUT
    """
    if not curve_points or len(curve_points) < 2:
        # Identity curve
        return np.arange(256, dtype=np.float32)
    
    # Sort points by x coordinate
    points = sorted(curve_points, key=lambda p: p[0])
    
    # Extract x and y coordinates
    x_points = np.array([p[0] for p in points], dtype=np.float32)
    y_points = np.array([p[1] for p in points], dtype=np.float32)
    
    # Ensure we have endpoints
    if x_points[0] > 0:
        x_points = np.insert(x_points, 0, 0)
        y_points = np.insert(y_points, 0, 0)
    if x_points[-1] < 255:
        x_points = np.append(x_points, 255)
        y_points = np.append(y_points, 255)
    
    # Interpolate to create 256-element LUT
    lut = np.interp(np.arange(256), x_points, y_points)
    return np.clip(lut, 0, 255).astype(np.float32)


def apply_all_adjustments(image, adjustments_dict):
    """
    Applies a dictionary of adjustments to an image in a specific order.
    Handles both basic adjustments and advanced ones like curves, mixer, HSL, etc.
    Also handles preset previews if 'preset_info' is in the dictionary.
    Expects image as uint8 NumPy array (RGB). Returns uint8 NumPy array (RGB).
    
    Uses GPU acceleration when available for most adjustments.
    """
    if image is None:
        logger.error("apply_all_adjustments received None image.")
        return None
    if not adjustments_dict:
        logger.warning("apply_all_adjustments received empty adjustments dict.")
        return image.copy()  # Return copy if no adjustments

    logger.debug("Starting adjustment pipeline. adjustments_dict=%s", adjustments_dict)

    current_image = image.copy() # Work on a copy

    # Helper instances
    basic_adjuster = ImageAdjustments()
    advanced_adjuster = AdvancedAdjustments()

    # --- Debug: Check image state ---
    def check_img(step_name):
        if current_image is None:
            logger.error("Image became None after step: %s", step_name)
            return False
        return True

    # --- Gather all adjustment parameters ---
    brightness = adjustments_dict.get('brightness', 0)
    contrast = adjustments_dict.get('contrast', 0)
    saturation = adjustments_dict.get('saturation', 0)
    hue = adjustments_dict.get('hue', 0)
    temp = adjustments_dict.get('temp', 0)
    tint = adjustments_dict.get('tint', 0)
    shadows = adjustments_dict.get('shadows', 0)
    highlights = adjustments_dict.get('highlights', 0)
    vibrance = adjustments_dict.get('vibrance', 0)
    
    # Levels parameters
    levels_in_black = adjustments_dict.get('levels_in_black', 0)
    levels_in_white = adjustments_dict.get('levels_in_white', 255)
    levels_gamma = adjustments_dict.get('levels_gamma', 1.0)
    levels_out_black = adjustments_dict.get('levels_out_black', 0)
    levels_out_white = adjustments_dict.get('levels_out_white', 255)
    
    # Check what adjustments we have
    has_basic_adjustments = any([
        brightness != 0, contrast != 0, saturation != 0, hue != 0,
        temp != 0, tint != 0, shadows != 0, highlights != 0, vibrance != 0
    ])
    has_levels = (levels_in_black != 0 or levels_in_white != 255 or 
                  levels_gamma != 1.0 or levels_out_black != 0 or levels_out_white != 255)
    
    # Build channel mixer config
    channel_mixer = None
    mixer_identity = True
    for out_ch_name in ['red', 'green', 'blue']:
        r_mix = adjustments_dict.get(f'mixer_{out_ch_name}_r', 100 if out_ch_name == 'red' else 0)
        g_mix = adjustments_dict.get(f'mixer_{out_ch_name}_g', 100 if out_ch_name == 'green' else 0)
        b_mix = adjustments_dict.get(f'mixer_{out_ch_name}_b', 100 if out_ch_name == 'blue' else 0)
        const = adjustments_dict.get(f'mixer_{out_ch_name}_const', 0)
        
        is_id = ((out_ch_name == 'red' and r_mix == 100 and g_mix == 0 and b_mix == 0 and const == 0) or
                 (out_ch_name == 'green' and r_mix == 0 and g_mix == 100 and b_mix == 0 and const == 0) or
                 (out_ch_name == 'blue' and r_mix == 0 and g_mix == 0 and b_mix == 100 and const == 0))
        
        if not is_id:
            mixer_identity = False
            if channel_mixer is None:
                channel_mixer = {}
            channel_mixer[out_ch_name] = {'r': r_mix, 'g': g_mix, 'b': b_mix, 'constant': const}
    
    # Build HSL config
    hsl_config = None
    for color_range in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas']:
        h_shift = adjustments_dict.get(f'hsl_{color_range}_h', 0)
        s_shift = adjustments_dict.get(f'hsl_{color_range}_s', 0)
        l_shift = adjustments_dict.get(f'hsl_{color_range}_l', 0)
        if h_shift != 0 or s_shift != 0 or l_shift != 0:
            if hsl_config is None:
                hsl_config = {}
            hsl_config[color_range] = {'h': h_shift, 's': s_shift, 'l': l_shift}
    
    # Build curves config (convert curve points to LUTs)
    curves_config = None
    curves_red = adjustments_dict.get('curves_red')
    curves_green = adjustments_dict.get('curves_green')
    curves_blue = adjustments_dict.get('curves_blue')
    curves_rgb = adjustments_dict.get('curves_rgb')
    
    if curves_red or curves_green or curves_blue or curves_rgb:
        curves_config = {}
        if curves_rgb:
            curves_config['rgb'] = _curve_points_to_lut(curves_rgb)
        if curves_red:
            curves_config['r'] = _curve_points_to_lut(curves_red)
        if curves_green:
            curves_config['g'] = _curve_points_to_lut(curves_green)
        if curves_blue:
            curves_config['b'] = _curve_points_to_lut(curves_blue)
    
    # Build selective color config
    selective_color_config = None
    selective_color_relative = adjustments_dict.get('sel_relative', True)
    for color_range in ['reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas', 'whites', 'neutrals', 'blacks']:
        c_adj = adjustments_dict.get(f'sel_{color_range}_c', 0)
        m_adj = adjustments_dict.get(f'sel_{color_range}_m', 0)
        y_adj = adjustments_dict.get(f'sel_{color_range}_y', 0)
        k_adj = adjustments_dict.get(f'sel_{color_range}_k', 0)
        if c_adj != 0 or m_adj != 0 or y_adj != 0 or k_adj != 0:
            if selective_color_config is None:
                selective_color_config = {}
            selective_color_config[color_range] = {'c': c_adj, 'm': m_adj, 'y': y_adj, 'k': k_adj}
    
    # Build noise reduction / smoothing config
    smoothing_config = None
    nr_strength = adjustments_dict.get('noise_reduction_strength', 0)
    if nr_strength > 0:
        # Map noise reduction strength to bilateral filter parameters
        smoothing_config = {
            'radius': min(3, 1 + nr_strength // 30),  # 1-3 based on strength
            'sigma_s': 10 + nr_strength * 0.5,  # Spatial sigma
            'sigma_r': 20 + nr_strength * 0.8,  # Range sigma
        }
    
    # Check if we have any GPU-acceleratable adjustments
    has_any_adjustments = (has_basic_adjustments or has_levels or channel_mixer or 
                           hsl_config or curves_config or selective_color_config or smoothing_config)
    
    # --- Try GPU acceleration for all supported operations ---
    logger.debug("Pipeline Step: GPU-accelerated adjustments...")
    gpu_used = False
    
    try:
        from ..utils.gpu import get_gpu_engine, has_gpu_engine
        
        if has_gpu_engine() and has_any_adjustments:
            engine = get_gpu_engine()
            
            # Build levels dict if needed
            levels_dict = None
            if has_levels:
                levels_dict = {
                    'in_black': levels_in_black,
                    'in_white': levels_in_white,
                    'gamma': levels_gamma,
                    'out_black': levels_out_black,
                    'out_white': levels_out_white,
                }
            
            # Process ALL adjustments in one GPU pass
            current_image = engine.process_adjustments(
                current_image,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                temp=temp,
                tint=tint,
                shadows=shadows,
                highlights=highlights,
                vibrance=vibrance,
                hue=hue,
                levels=levels_dict,
                channel_mixer=channel_mixer,
                curves=curves_config,
                hsl=hsl_config,
                selective_color=selective_color_config,
                selective_color_relative=selective_color_relative,
                smoothing=smoothing_config,
            )
            gpu_used = True
            logger.debug("All adjustments applied via GPU (single pass)")
    except Exception as e:
        logger.warning("GPU acceleration failed, falling back to CPU: %s", e)
        gpu_used = False
    
    # CPU fallback for basic adjustments
    if not gpu_used:
        if brightness != 0 or contrast != 0 or saturation != 0:
            current_image = basic_adjuster.adjust_brightness(current_image, brightness)
            current_image = basic_adjuster.adjust_contrast(current_image, contrast)
            current_image = basic_adjuster.adjust_saturation(current_image, saturation)
        
        if temp != 0 or tint != 0:
            current_image = basic_adjuster.adjust_temp_tint(current_image, temp, tint)
        
        if has_levels:
            current_image = basic_adjuster.adjust_levels(
                current_image, levels_in_black, levels_in_white,
                levels_gamma, levels_out_black, levels_out_white
            )
        
        if shadows != 0 or highlights != 0:
            current_image = advanced_adjuster.adjust_shadows_highlights(current_image, shadows, highlights)
        
        if vibrance != 0:
            current_image = advanced_adjuster.adjust_vibrance(current_image, vibrance)
        
        if hue != 0:
            current_image = basic_adjuster.adjust_hue(current_image, hue)
        
        # Channel Mixer (CPU fallback)
        if channel_mixer:
            for out_ch_name, cfg in channel_mixer.items():
                current_image = advanced_adjuster.adjust_channel_mixer(
                    current_image, out_ch_name.capitalize(),
                    cfg.get('r', 0), cfg.get('g', 0), cfg.get('b', 0), cfg.get('constant', 0)
                )
        
        # HSL (CPU fallback)
        if hsl_config:
            for color_range, adj in hsl_config.items():
                current_image = advanced_adjuster.adjust_hsl_by_range(
                    current_image, color_range.capitalize(),
                    adj.get('h', 0), adj.get('s', 0), adj.get('l', 0)
                )
    
    if not check_img("Basic Adjustments"): return None
    logger.debug("Pipeline Step: Basic Adjustments Done.")

    # --- Apply Preset (if specified for preview) ---
    preset_info = adjustments_dict.get('preset_info')
    logger.debug(
        "Checking preset preview block. preset_info=%s PRESET_MANAGERS_AVAILABLE=%s",
        preset_info,
        PRESET_MANAGERS_AVAILABLE,
    )
    if preset_info and PRESET_MANAGERS_AVAILABLE:
        preset_type = preset_info.get('type')
        preset_id = preset_info.get('id')
        intensity = preset_info.get('intensity', 1.0)
        grain_scale = preset_info.get('grain_scale') # Might be None

        if preset_type == 'film' and preset_id and _film_preset_manager:
            logger.debug(
                "Applying film preset preview: %s (intensity=%s grain_scale=%s)",
                preset_id,
                intensity,
                grain_scale,
            )
            try:
                # We need the preset definition to apply grain correctly if needed
                preset_data = _film_preset_manager.get_preset(preset_id)
                if preset_data:
                     # Use preview_mode=True for faster preview at lower resolution
                     current_image = _film_preset_manager.apply_preset(
                         current_image, preset_data, intensity, grain_scale, preview_mode=True
                     )
                     if not check_img(f"Film Preset: {preset_id}"): return None # Check result after applying
                else:
                    logger.warning("Film preset '%s' not found for preview.", preset_id)
            except Exception:
                logger.exception("Failed applying film preset preview '%s'.", preset_id)

        elif preset_type == 'photo' and preset_id and _photo_preset_manager:
            logger.debug(
                "Applying photo preset preview: %s (intensity=%s)",
                preset_id,
                intensity,
            )
            try:
                # Use preview_mode=True for faster preview at lower resolution
                current_image = _photo_preset_manager.apply_photo_preset(
                    current_image, preset_id, intensity, preview_mode=True
                )
                if not check_img(f"Photo Preset: {preset_id}"):
                    return None  # Check result after applying
            except Exception:
                logger.exception("Failed applying photo preset preview '%s'.", preset_id)

    # 4. Curves - Skip if already done via GPU
    if not gpu_used:
        logger.debug("Pipeline Step: Curves (CPU fallback)...")
        current_image = advanced_adjuster.apply_curves(
            current_image,
            adjustments_dict.get('curves_red'),
            adjustments_dict.get('curves_green'),
            adjustments_dict.get('curves_blue'),
            adjustments_dict.get('curves_rgb')
        )
        if not check_img("Curves"): return None
        logger.debug("Pipeline Step: Curves Done.")

    # --- Dust Removal (always CPU - specialized inpainting) ---
    if adjustments_dict.get('dust_removal_enabled', False):
        logger.debug("Pipeline Step: Dust Removal...")
        current_image = AdvancedAdjustments.apply_dust_removal(
            current_image,
            sensitivity=adjustments_dict.get('dust_removal_sensitivity', 50), # Default sensitivity
            radius=adjustments_dict.get('dust_removal_radius', 3) # Default radius
        )
        if not check_img("Dust Removal"): return None
        logger.debug("Pipeline Step: Dust Removal Done.")

    # 5. Channel Mixer - Skip if already done via GPU
    if not gpu_used:
        logger.debug("Pipeline Step: Channel Mixer (CPU fallback)...")
        for out_ch_name in ['Red', 'Green', 'Blue']:
            r_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_r', 100 if out_ch_name == 'Red' else 0)
            g_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_g', 100 if out_ch_name == 'Green' else 0)
            b_mix = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_b', 100 if out_ch_name == 'Blue' else 0)
            const = adjustments_dict.get(f'mixer_{out_ch_name.lower()}_const', 0)
            is_id = (out_ch_name == 'Red' and r_mix == 100 and g_mix == 0 and b_mix == 0 and const == 0) or \
                    (out_ch_name == 'Green' and r_mix == 0 and g_mix == 100 and b_mix == 0 and const == 0) or \
                    (out_ch_name == 'Blue' and r_mix == 0 and g_mix == 0 and b_mix == 100 and const == 0)
            if not is_id:
                current_image = advanced_adjuster.adjust_channel_mixer(current_image, out_ch_name, r_mix, g_mix, b_mix, const)
                if not check_img(f"Channel Mixer ({out_ch_name})"): return None
        logger.debug("Pipeline Step: Channel Mixer Done.")

    # 6. HSL Adjustments - Skip if already done via GPU
    if not gpu_used:
        logger.debug("Pipeline Step: HSL (CPU fallback)...")
        for color_range in ['Reds', 'Yellows', 'Greens', 'Cyans', 'Blues', 'Magentas']:
            color_key = color_range.lower()
            h_shift = adjustments_dict.get(f'hsl_{color_key}_h', 0)
            s_shift = adjustments_dict.get(f'hsl_{color_key}_s', 0)
            l_shift = adjustments_dict.get(f'hsl_{color_key}_l', 0)
            if h_shift != 0 or s_shift != 0 or l_shift != 0:
                current_image = advanced_adjuster.adjust_hsl_by_range(current_image, color_range, h_shift, s_shift, l_shift)
                if not check_img(f"HSL ({color_range})"): return None
        logger.debug("Pipeline Step: HSL Done.")

    # 7. Selective Color - Skip if already done via GPU
    if not gpu_used:
        logger.debug("Pipeline Step: Selective Color (CPU fallback)...")
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
        logger.debug("Pipeline Step: Selective Color Done.")

    # 8. Noise Reduction - Skip if already done via GPU
    if not gpu_used:
        nr_strength = adjustments_dict.get('noise_reduction_strength', 0)
        if nr_strength > 0:
            logger.debug("Pipeline Step: Noise Reduction (CPU fallback)...")
            current_image = advanced_adjuster.apply_noise_reduction(current_image, strength=nr_strength)
            if not check_img("Noise Reduction"): return None
            logger.debug("Pipeline Step: Noise Reduction Done.")

    # Add other adjustments here in the desired order...
    # e.g., Vignette, Grain (Grain might be better applied last or handled by preset logic)

    logger.debug("Adjustment pipeline finished.")
    return current_image


# --- Utility Functions ---

def to_uint8_for_cv(float_arr):
    """Converts float (0-1) to uint8 (0-255) for OpenCV functions."""
    if float_arr is None: return None
    return (np.clip(float_arr, 0.0, 1.0) * 255).astype(np.uint8)

def update_float_from_cv(uint8_result):
    """Converts uint8 (0-255) result back to float (0-1)."""
    if uint8_result is None: return None
    return uint8_result.astype(np.float32) / 255.0