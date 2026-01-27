"""
GPU Engine - Multi-stage compute pipeline for image processing.

This engine provides efficient GPU acceleration by:
1. Using a single unified shader that performs all operations in one dispatch
2. Only one texture upload at the start and one readback at the end
3. Supporting both wgpu and CuPy backends with a unified interface

The key insight is that GPU overhead comes from data transfers, not computation.
By doing all processing in a single shader dispatch, we minimize transfers.
"""

import struct
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

# Constants
WORKGROUP_SIZE = 8
UNIFORM_SIZE = 1024  # Increased for full extended pipeline


class GPUEngine:
    """
    Core GPU processing engine supporting wgpu and CuPy backends.
    
    For wgpu: Uses a unified compute shader that performs all operations
    For CuPy: Uses fused array operations on GPU memory
    """
    
    def __init__(self) -> None:
        from .gpu_device import GPUDevice
        
        self.gpu = GPUDevice.get()
        self._initialized = False
        
        # wgpu-specific state
        self._pipeline: Optional[Any] = None
        self._uniform_buffer: Optional[Any] = None
        self._texture_pool: Optional[Any] = None
        self._curves_lut_texture: Optional[Any] = None
    
    def is_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu.is_available
    
    def get_backend_name(self) -> str:
        """Get the name of the active backend."""
        if self.gpu.is_cupy:
            return "CuPy"
        elif self.gpu.is_wgpu:
            return "wgpu"
        return "CPU"

    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_wgpu_resources(self) -> None:
        """Initialize wgpu pipeline and buffers."""
        if self._initialized or not self.gpu.is_wgpu:
            return
        
        import wgpu
        from .gpu_resources import TexturePool
        from .gpu_shaders import ShaderLoader
        
        device = self.gpu.wgpu_device
        if not device:
            return
        
        # Create texture pool
        self._texture_pool = TexturePool()
        
        # Create the unified pipeline shader
        try:
            module = ShaderLoader.load("full_pipeline")
            self._pipeline = device.create_compute_pipeline(
                layout="auto",
                compute={"module": module, "entry_point": "main"}
            )
        except Exception as e:
            logger.error(f"Failed to create unified pipeline: {e}")
            return
        
        # Create uniform buffer (large enough for all parameters)
        self._uniform_buffer = device.create_buffer(
            size=UNIFORM_SIZE,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        )
        
        # Create default curves LUT (identity)
        self._create_default_curves_lut()
        
        self._initialized = True
        logger.info("GPU Engine: wgpu unified pipeline initialized")
    
    def _create_default_curves_lut(self) -> None:
        """Create default identity curves LUT texture."""
        import wgpu
        device = self.gpu.wgpu_device
        
        # 256x4 texture: rows for R, G, B, RGB curves
        lut_data = np.zeros((4, 256), dtype=np.float32)
        for i in range(256):
            lut_data[0, i] = float(i)  # R
            lut_data[1, i] = float(i)  # G
            lut_data[2, i] = float(i)  # B
            lut_data[3, i] = float(i)  # RGB
        
        self._curves_lut_texture = device.create_texture(
            size=(256, 4, 1),
            format=wgpu.TextureFormat.r32float,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )
        
        device.queue.write_texture(
            {"texture": self._curves_lut_texture},
            lut_data.tobytes(),
            {"bytes_per_row": 256 * 4, "rows_per_image": 4},
            (256, 4, 1),
        )
    
    def _get_texture(self, width: int, height: int, label: str) -> Any:
        """Get a texture from the pool."""
        import wgpu
        
        usage = (
            wgpu.TextureUsage.TEXTURE_BINDING |
            wgpu.TextureUsage.STORAGE_BINDING |
            wgpu.TextureUsage.COPY_DST |
            wgpu.TextureUsage.COPY_SRC
        )
        return self._texture_pool.get(width, height, usage, label)

    # =========================================================================
    # Unified Processing API
    # =========================================================================
    
    def process_full_pipeline(
        self,
        image: np.ndarray,
        # Conversion parameters
        invert: bool = False,
        wb_scales: Optional[Tuple[float, float, float]] = None,
        color_matrix: Optional[np.ndarray] = None,
        # Adjustment parameters
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        temp: float = 0,
        tint: float = 0,
        gamma: float = 1.0,
        shadows: float = 0,
        highlights: float = 0,
        vibrance: float = 0,
        hue: float = 0,
        # Levels parameters
        levels: Optional[Dict[str, float]] = None,
        # Channel mixer (dict with red/green/blue output channel configs)
        channel_mixer: Optional[Dict[str, Dict[str, float]]] = None,
        # Curves (dict with r/g/b/rgb LUT arrays)
        curves: Optional[Dict[str, np.ndarray]] = None,
        # HSL per-range adjustments
        hsl: Optional[Dict[str, Dict[str, float]]] = None,
        # Selective Color (CMYK adjustments per color range)
        selective_color: Optional[Dict[str, Dict[str, float]]] = None,
        selective_color_relative: bool = True,
        # LAB color grading
        lab_grading: Optional[Dict[str, float]] = None,
        # Edge-aware smoothing
        smoothing: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Process image through the full pipeline in a single GPU dispatch.
        
        Args:
            image: Input uint8 RGB image
            invert: Whether to invert (negative to positive)
            wb_scales: White balance scale factors (R, G, B)
            color_matrix: 3x3 color correction matrix
            brightness: -100 to 100
            contrast: -100 to 100
            saturation: -100 to 100
            temp: Temperature adjustment
            tint: Tint adjustment
            gamma: Gamma correction (1.0 = no change)
            shadows: Shadow adjustment (-100 to 100)
            highlights: Highlight adjustment (-100 to 100)
            vibrance: Vibrance adjustment (-100 to 100)
            hue: Hue shift in degrees (-180 to 180)
            levels: Dict with in_black, in_white, gamma, out_black, out_white
            channel_mixer: Dict with 'red', 'green', 'blue' keys, each containing
                          'r', 'g', 'b', 'constant' values (percentages)
            curves: Dict with 'r', 'g', 'b', 'rgb' keys containing 256-element LUTs
            hsl: Dict with color range keys ('reds', 'yellows', etc.) containing
                 'h', 's', 'l' adjustment values
            selective_color: Dict with color range keys containing 'c', 'm', 'y', 'k' values
            selective_color_relative: Whether to use relative mode for selective color
            lab_grading: Dict with 'l_shift', 'a_shift', 'b_shift', 'a_target', 'a_factor',
                        'b_target', 'b_factor' values
            smoothing: Dict with 'radius', 'sigma_s', 'sigma_r' for bilateral filter
            
        Returns:
            Processed uint8 RGB image
        """
        if self.gpu.is_cupy:
            return self._process_full_pipeline_cupy(
                image, invert, wb_scales, color_matrix,
                brightness, contrast, saturation, temp, tint, gamma,
                shadows, highlights, vibrance, hue, levels,
                channel_mixer, curves, hsl,
                selective_color, selective_color_relative, lab_grading, smoothing
            )
        elif self.gpu.is_wgpu:
            return self._process_full_pipeline_wgpu(
                image, invert, wb_scales, color_matrix,
                brightness, contrast, saturation, temp, tint, gamma,
                shadows, highlights, vibrance, hue, levels,
                channel_mixer, curves, hsl,
                selective_color, selective_color_relative, lab_grading, smoothing
            )
        else:
            raise RuntimeError("No GPU backend available")

    def _process_full_pipeline_cupy(
        self,
        image: np.ndarray,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray],
        brightness: float,
        contrast: float,
        saturation: float,
        temp: float,
        tint: float,
        gamma: float,
        shadows: float,
        highlights: float,
        vibrance: float,
        hue: float,
        levels: Optional[Dict[str, float]],
        channel_mixer: Optional[Dict[str, Dict[str, float]]],
        curves: Optional[Dict[str, np.ndarray]],
        hsl: Optional[Dict[str, Dict[str, float]]],
        selective_color: Optional[Dict[str, Dict[str, float]]],
        selective_color_relative: bool,
        lab_grading: Optional[Dict[str, float]],
        smoothing: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """CuPy implementation - all operations fused on GPU."""
        cp = self.gpu.cupy
        
        # Single transfer to GPU
        img = cp.asarray(image, dtype=cp.float32)
        
        # Stage 1: Invert
        if invert:
            img = 255.0 - img
        
        # Stage 2: White balance
        if wb_scales:
            scales = cp.array(wb_scales, dtype=cp.float32)
            img = img * scales
        
        # Stage 3: Color matrix
        if color_matrix is not None:
            matrix = cp.asarray(color_matrix, dtype=cp.float32)
            h, w, c = img.shape
            flat = img.reshape(-1, 3)
            img = cp.dot(flat, matrix.T).reshape(h, w, 3)
        
        # Stage 4: Basic Adjustments
        if brightness != 0:
            offset = (brightness / 100.0) * 127.0
            img = img + offset
        
        if contrast != 0:
            factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
            img = factor * (img - 128.0) + 128.0
        
        if temp != 0 or tint != 0:
            temp_factor = temp * 0.3
            tint_factor = tint * 0.3
            img[:, :, 0] += temp_factor
            img[:, :, 2] -= temp_factor
            img[:, :, 1] -= tint_factor
        
        if gamma != 1.0:
            img = img / 255.0
            img = cp.power(cp.maximum(img, 0), 1.0 / gamma)
            img = img * 255.0
        
        if shadows != 0 or highlights != 0:
            lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            lum_norm = lum / 255.0
            shadow_mask = cp.power(cp.clip(1.0 - lum_norm, 0, 1), 1.5)
            highlight_mask = cp.power(cp.clip(lum_norm, 0, 1), 1.5)
            shadow_adj = (shadows / 100.0) * 100.0
            highlight_adj = (highlights / 100.0) * 100.0
            adjustment = shadow_mask * shadow_adj + highlight_mask * highlight_adj
            img = img + adjustment[:, :, cp.newaxis]
        
        if vibrance != 0:
            rgb = img / 255.0
            max_c = cp.maximum(cp.maximum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
            min_c = cp.minimum(cp.minimum(rgb[:, :, 0], rgb[:, :, 1]), rgb[:, :, 2])
            current_sat = cp.where(max_c > 0, (max_c - min_c) / max_c, 0)
            factor = vibrance / 100.0
            boost = factor * (1.0 - current_sat)
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            img = (rgb + (rgb - lum[:, :, cp.newaxis]) * boost[:, :, cp.newaxis]) * 255.0
        
        if saturation != 0:
            factor = 1.0 + (saturation / 100.0)
            rgb = img / 255.0
            lum = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            img = (lum[:, :, cp.newaxis] + (rgb - lum[:, :, cp.newaxis]) * factor) * 255.0
        
        # Hue shift (requires HSV conversion - done on CPU for CuPy)
        if hue != 0:
            img_cpu = cp.asnumpy(cp.clip(img, 0, 255).astype(cp.uint8))
            import cv2
            hsv = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + hue / 2.0) % 180  # OpenCV uses 0-180 for hue
            img_cpu = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            img = cp.asarray(img_cpu, dtype=cp.float32)

        # Stage 5: Levels
        if levels:
            in_black = levels.get("in_black", 0)
            in_white = levels.get("in_white", 255)
            lvl_gamma = levels.get("gamma", 1.0)
            out_black = levels.get("out_black", 0)
            out_white = levels.get("out_white", 255)
            
            in_range = max(in_white - in_black, 0.001)
            out_range = out_white - out_black
            
            img = (img - in_black) / in_range
            img = cp.clip(img, 0, 1)
            if lvl_gamma != 1.0:
                img = cp.power(img, 1.0 / lvl_gamma)
            img = img * out_range + out_black
        
        # Stage 6: Channel Mixer
        if channel_mixer:
            img_clamped = cp.clip(img, 0, 255)
            r_cfg = channel_mixer.get('red', {'r': 100, 'g': 0, 'b': 0, 'constant': 0})
            g_cfg = channel_mixer.get('green', {'r': 0, 'g': 100, 'b': 0, 'constant': 0})
            b_cfg = channel_mixer.get('blue', {'r': 0, 'g': 0, 'b': 100, 'constant': 0})
            
            r_out = (img_clamped[:,:,0] * r_cfg.get('r', 100) / 100.0 +
                     img_clamped[:,:,1] * r_cfg.get('g', 0) / 100.0 +
                     img_clamped[:,:,2] * r_cfg.get('b', 0) / 100.0 +
                     r_cfg.get('constant', 0) * 1.275)
            g_out = (img_clamped[:,:,0] * g_cfg.get('r', 0) / 100.0 +
                     img_clamped[:,:,1] * g_cfg.get('g', 100) / 100.0 +
                     img_clamped[:,:,2] * g_cfg.get('b', 0) / 100.0 +
                     g_cfg.get('constant', 0) * 1.275)
            b_out = (img_clamped[:,:,0] * b_cfg.get('r', 0) / 100.0 +
                     img_clamped[:,:,1] * b_cfg.get('g', 0) / 100.0 +
                     img_clamped[:,:,2] * b_cfg.get('b', 100) / 100.0 +
                     b_cfg.get('constant', 0) * 1.275)
            
            img = cp.stack([r_out, g_out, b_out], axis=2)
        
        # Stage 7: Curves (via LUT)
        if curves:
            img_clamped = cp.clip(img, 0, 255).astype(cp.int32)
            
            # Apply RGB master curve first if present
            if 'rgb' in curves and curves['rgb'] is not None:
                lut_rgb = cp.asarray(curves['rgb'], dtype=cp.float32)
                img = cp.stack([
                    lut_rgb[img_clamped[:,:,0]],
                    lut_rgb[img_clamped[:,:,1]],
                    lut_rgb[img_clamped[:,:,2]]
                ], axis=2)
                img_clamped = cp.clip(img, 0, 255).astype(cp.int32)
            
            # Apply per-channel curves
            if 'r' in curves and curves['r'] is not None:
                lut_r = cp.asarray(curves['r'], dtype=cp.float32)
                img[:,:,0] = lut_r[img_clamped[:,:,0]]
            if 'g' in curves and curves['g'] is not None:
                lut_g = cp.asarray(curves['g'], dtype=cp.float32)
                img[:,:,1] = lut_g[img_clamped[:,:,1]]
            if 'b' in curves and curves['b'] is not None:
                lut_b = cp.asarray(curves['b'], dtype=cp.float32)
                img[:,:,2] = lut_b[img_clamped[:,:,2]]
        
        # Stage 8: HSL per-range (done on CPU for CuPy - complex color logic)
        if hsl and any(hsl.get(k, {}).get('h', 0) != 0 or 
                       hsl.get(k, {}).get('s', 0) != 0 or 
                       hsl.get(k, {}).get('l', 0) != 0 
                       for k in hsl):
            # Transfer to CPU for HSL adjustments
            img_cpu = cp.asnumpy(cp.clip(img, 0, 255).astype(cp.uint8))
            import cv2
            for color_range, adj in hsl.items():
                h_shift = adj.get('h', 0)
                s_shift = adj.get('s', 0)
                l_shift = adj.get('l', 0)
                if h_shift != 0 or s_shift != 0 or l_shift != 0:
                    from ..processing.adjustments import AdvancedAdjustments
                    img_cpu = AdvancedAdjustments.adjust_hsl_by_range(
                        img_cpu, color_range.capitalize(), h_shift, s_shift, l_shift
                    )
            img = cp.asarray(img_cpu, dtype=cp.float32)
        
        # Stage 9: Selective Color (done on CPU for CuPy)
        if selective_color and any(
            any(selective_color.get(k, {}).get(c, 0) != 0 for c in ['c', 'm', 'y', 'k'])
            for k in selective_color
        ):
            img_cpu = cp.asnumpy(cp.clip(img, 0, 255).astype(cp.uint8))
            from ..processing.adjustments import AdvancedAdjustments
            for color_range, adj in selective_color.items():
                c_adj = adj.get('c', 0)
                m_adj = adj.get('m', 0)
                y_adj = adj.get('y', 0)
                k_adj = adj.get('k', 0)
                if c_adj != 0 or m_adj != 0 or y_adj != 0 or k_adj != 0:
                    img_cpu = AdvancedAdjustments.adjust_selective_color(
                        img_cpu, color_range.capitalize(), c_adj, m_adj, y_adj, k_adj,
                        relative=selective_color_relative
                    )
            img = cp.asarray(img_cpu, dtype=cp.float32)
        
        # Stage 10: LAB grading (done on CPU for CuPy)
        if lab_grading:
            img_cpu = cp.asnumpy(cp.clip(img, 0, 255).astype(cp.uint8))
            import cv2
            lab = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Apply shifts
            lab[:, :, 0] = np.clip(lab[:, :, 0] + lab_grading.get('l_shift', 0), 0, 255)
            lab[:, :, 1] = np.clip(lab[:, :, 1] + lab_grading.get('a_shift', 0), 0, 255)
            lab[:, :, 2] = np.clip(lab[:, :, 2] + lab_grading.get('b_shift', 0), 0, 255)
            
            # Apply target-based correction
            a_factor = lab_grading.get('a_factor', 0)
            if a_factor != 0:
                a_target = lab_grading.get('a_target', 128)
                a_avg = np.mean(lab[:, :, 1])
                lab[:, :, 1] = lab[:, :, 1] - (a_avg - a_target) * a_factor
            
            b_factor = lab_grading.get('b_factor', 0)
            if b_factor != 0:
                b_target = lab_grading.get('b_target', 128)
                b_avg = np.mean(lab[:, :, 2])
                lab[:, :, 2] = lab[:, :, 2] - (b_avg - b_target) * b_factor
            
            lab = np.clip(lab, 0, 255).astype(np.uint8)
            img_cpu = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = cp.asarray(img_cpu, dtype=cp.float32)
        
        # Stage 11: Smoothing (bilateral filter on CPU for CuPy)
        if smoothing and smoothing.get('radius', 0) > 0:
            img_cpu = cp.asnumpy(cp.clip(img, 0, 255).astype(cp.uint8))
            import cv2
            d = int(smoothing.get('radius', 3) * 2 + 1)
            sigma_color = smoothing.get('sigma_r', 75)
            sigma_space = smoothing.get('sigma_s', 75)
            img_cpu = cv2.bilateralFilter(img_cpu, d, sigma_color, sigma_space)
            img = cp.asarray(img_cpu, dtype=cp.float32)
        
        # Single transfer back to CPU
        result = cp.clip(img, 0, 255).astype(cp.uint8)
        return cp.asnumpy(result)

    def _process_full_pipeline_wgpu(
        self,
        image: np.ndarray,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray],
        brightness: float,
        contrast: float,
        saturation: float,
        temp: float,
        tint: float,
        gamma: float,
        shadows: float,
        highlights: float,
        vibrance: float,
        hue: float,
        levels: Optional[Dict[str, float]],
        channel_mixer: Optional[Dict[str, Dict[str, float]]],
        curves: Optional[Dict[str, np.ndarray]],
        hsl: Optional[Dict[str, Dict[str, float]]],
        selective_color: Optional[Dict[str, Dict[str, float]]],
        selective_color_relative: bool,
        lab_grading: Optional[Dict[str, float]],
        smoothing: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """wgpu implementation - single shader dispatch."""
        self._init_wgpu_resources()
        
        if not self._pipeline:
            raise RuntimeError("wgpu pipeline not initialized")
        
        import wgpu
        device = self.gpu.wgpu_device
        h, w = image.shape[:2]
        
        # Get textures
        tex_input = self._get_texture(w, h, "input")
        tex_output = self._get_texture(w, h, "output")
        
        # Single upload
        tex_input.upload(image.astype(np.float32))
        
        # Update curves LUT if provided
        curves_lut_view = self._update_curves_lut(curves)
        
        # Build uniform data
        uniform_data = self._build_uniform_data_extended(
            invert, wb_scales, color_matrix,
            brightness, contrast, saturation, temp, tint, gamma,
            shadows, highlights, vibrance, hue, levels,
            channel_mixer, curves, hsl,
            selective_color, selective_color_relative, lab_grading, smoothing
        )
        device.queue.write_buffer(self._uniform_buffer, 0, uniform_data)
        
        # Create bind group with curves LUT
        bind_group = device.create_bind_group(
            layout=self._pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_input.view},
                {"binding": 1, "resource": tex_output.view},
                {"binding": 2, "resource": {"buffer": self._uniform_buffer}},
                {"binding": 3, "resource": curves_lut_view},
            ]
        )
        
        # Single dispatch
        encoder = device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self._pipeline)
        compute_pass.set_bind_group(0, bind_group)
        
        wg_x = (w + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        wg_y = (h + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE
        compute_pass.dispatch_workgroups(wg_x, wg_y, 1)
        compute_pass.end()
        
        device.queue.submit([encoder.finish()])
        
        # Single readback
        result = tex_output.readback()
        return np.clip(result[:, :, :3], 0, 255).astype(np.uint8)
    
    def _update_curves_lut(self, curves: Optional[Dict[str, np.ndarray]]) -> Any:
        """Update curves LUT texture and return view."""
        device = self.gpu.wgpu_device
        
        if curves:
            # Build LUT data: 256x4 (R, G, B, RGB)
            lut_data = np.zeros((4, 256), dtype=np.float32)
            
            # Default to identity
            for i in range(256):
                lut_data[0, i] = float(i)
                lut_data[1, i] = float(i)
                lut_data[2, i] = float(i)
                lut_data[3, i] = float(i)
            
            # Override with provided curves
            if 'r' in curves and curves['r'] is not None:
                lut_data[0, :] = curves['r'].astype(np.float32)
            if 'g' in curves and curves['g'] is not None:
                lut_data[1, :] = curves['g'].astype(np.float32)
            if 'b' in curves and curves['b'] is not None:
                lut_data[2, :] = curves['b'].astype(np.float32)
            if 'rgb' in curves and curves['rgb'] is not None:
                lut_data[3, :] = curves['rgb'].astype(np.float32)
            
            device.queue.write_texture(
                {"texture": self._curves_lut_texture},
                lut_data.tobytes(),
                {"bytes_per_row": 256 * 4, "rows_per_image": 4},
                (256, 4, 1),
            )
        
        return self._curves_lut_texture.create_view()

    def _build_uniform_data_extended(
        self,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray],
        brightness: float,
        contrast: float,
        saturation: float,
        temp: float,
        tint: float,
        gamma: float,
        shadows: float,
        highlights: float,
        vibrance: float,
        hue: float,
        levels: Optional[Dict[str, float]],
        channel_mixer: Optional[Dict[str, Dict[str, float]]],
        curves: Optional[Dict[str, np.ndarray]],
        hsl: Optional[Dict[str, Dict[str, float]]],
        selective_color: Optional[Dict[str, Dict[str, float]]] = None,
        selective_color_relative: bool = True,
        lab_grading: Optional[Dict[str, float]] = None,
        smoothing: Optional[Dict[str, float]] = None,
    ) -> bytes:
        """Build the uniform buffer data for the extended shader."""
        # Stage flags
        do_invert = 1 if invert else 0
        do_wb = 1 if wb_scales else 0
        do_matrix = 1 if color_matrix is not None else 0
        do_adjustments = 1 if any([
            brightness != 0, contrast != 0, saturation != 0,
            temp != 0, tint != 0, gamma != 1.0,
            shadows != 0, highlights != 0, vibrance != 0, hue != 0
        ]) else 0
        
        # White balance
        wb_r = wb_scales[0] if wb_scales else 1.0
        wb_g = wb_scales[1] if wb_scales else 1.0
        wb_b = wb_scales[2] if wb_scales else 1.0
        
        # Color matrix
        if color_matrix is not None:
            m = color_matrix.flatten()
        else:
            m = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32)
        
        # Contrast factor
        contrast_factor = 1.0
        if contrast != 0:
            contrast_factor = (259.0 * (contrast + 255.0)) / (255.0 * (259.0 - contrast))
        
        # Saturation factor
        saturation_factor = 1.0 + (saturation / 100.0)
        
        # Levels
        do_levels = 1 if levels else 0
        in_black = levels.get("in_black", 0) if levels else 0
        in_white = levels.get("in_white", 255) if levels else 255
        lvl_gamma = levels.get("gamma", 1.0) if levels else 1.0
        out_black = levels.get("out_black", 0) if levels else 0
        out_white = levels.get("out_white", 255) if levels else 255
        
        # Channel mixer
        do_mixer = 1 if channel_mixer else 0
        mixer_r = channel_mixer.get('red', {}) if channel_mixer else {}
        mixer_g = channel_mixer.get('green', {}) if channel_mixer else {}
        mixer_b = channel_mixer.get('blue', {}) if channel_mixer else {}
        
        # Curves
        do_curves = 1 if curves else 0
        
        # HSL
        do_hsl = 1 if hsl else 0
        hsl_reds = hsl.get('reds', {}) if hsl else {}
        hsl_yellows = hsl.get('yellows', {}) if hsl else {}
        hsl_greens = hsl.get('greens', {}) if hsl else {}
        hsl_cyans = hsl.get('cyans', {}) if hsl else {}
        hsl_blues = hsl.get('blues', {}) if hsl else {}
        hsl_magentas = hsl.get('magentas', {}) if hsl else {}

        # Pack uniform data (must match shader struct layout)
        data = struct.pack(
            "IIII",  # Stage flags
            do_invert, do_wb, do_matrix, do_adjustments,
        )
        data += struct.pack(
            "ffff",  # White balance
            wb_r, wb_g, wb_b, 0.0,
        )
        data += struct.pack(
            "ffffffffffff",  # Color matrix rows (3 x vec4)
            m[0], m[1], m[2], 0.0,
            m[3], m[4], m[5], 0.0,
            m[6], m[7], m[8], 0.0,
        )
        data += struct.pack(
            "ffffffffffff",  # Adjustments (12 f32)
            brightness, contrast_factor, saturation_factor, temp,
            tint, gamma, shadows, highlights,
            vibrance, hue, 0.0, 0.0,
        )
        data += struct.pack(
            "Ifffffff",  # Levels (1 u32 + 7 f32)
            do_levels, in_black, in_white, lvl_gamma, out_black, out_white, 0.0, 0.0,
        )
        data += struct.pack(
            "I",  # do_channel_mixer
            do_mixer,
        )
        # Pad to align mixer rows
        data += struct.pack("fff", 0.0, 0.0, 0.0)
        data += struct.pack(
            "ffff",  # mixer_row0 (R output)
            mixer_r.get('r', 100), mixer_r.get('g', 0), mixer_r.get('b', 0), mixer_r.get('constant', 0),
        )
        data += struct.pack(
            "ffff",  # mixer_row1 (G output)
            mixer_g.get('r', 0), mixer_g.get('g', 100), mixer_g.get('b', 0), mixer_g.get('constant', 0),
        )
        data += struct.pack(
            "ffff",  # mixer_row2 (B output)
            mixer_b.get('r', 0), mixer_b.get('g', 0), mixer_b.get('b', 100), mixer_b.get('constant', 0),
        )
        data += struct.pack(
            "Ifff",  # do_curves + padding
            do_curves, 0.0, 0.0, 0.0,
        )
        data += struct.pack(
            "I",  # do_hsl
            do_hsl,
        )
        # Pad to align HSL vec4s
        data += struct.pack("fff", 0.0, 0.0, 0.0)
        data += struct.pack(
            "ffff",  # hsl_reds
            hsl_reds.get('h', 0), hsl_reds.get('s', 0), hsl_reds.get('l', 0), 0.0,
        )
        data += struct.pack(
            "ffff",  # hsl_yellows
            hsl_yellows.get('h', 0), hsl_yellows.get('s', 0), hsl_yellows.get('l', 0), 0.0,
        )
        data += struct.pack(
            "ffff",  # hsl_greens
            hsl_greens.get('h', 0), hsl_greens.get('s', 0), hsl_greens.get('l', 0), 0.0,
        )
        data += struct.pack(
            "ffff",  # hsl_cyans
            hsl_cyans.get('h', 0), hsl_cyans.get('s', 0), hsl_cyans.get('l', 0), 0.0,
        )
        data += struct.pack(
            "ffff",  # hsl_blues
            hsl_blues.get('h', 0), hsl_blues.get('s', 0), hsl_blues.get('l', 0), 0.0,
        )
        data += struct.pack(
            "ffff",  # hsl_magentas
            hsl_magentas.get('h', 0), hsl_magentas.get('s', 0), hsl_magentas.get('l', 0), 0.0,
        )
        
        # Selective Color
        do_selective = 1 if selective_color else 0
        sel_relative = 1 if selective_color_relative else 0
        sel_reds = selective_color.get('reds', {}) if selective_color else {}
        sel_yellows = selective_color.get('yellows', {}) if selective_color else {}
        sel_greens = selective_color.get('greens', {}) if selective_color else {}
        sel_cyans = selective_color.get('cyans', {}) if selective_color else {}
        sel_blues = selective_color.get('blues', {}) if selective_color else {}
        sel_magentas = selective_color.get('magentas', {}) if selective_color else {}
        sel_whites = selective_color.get('whites', {}) if selective_color else {}
        sel_neutrals = selective_color.get('neutrals', {}) if selective_color else {}
        sel_blacks = selective_color.get('blacks', {}) if selective_color else {}
        
        data += struct.pack(
            "II",  # do_selective_color, sel_relative
            do_selective, sel_relative,
        )
        # Pad to align vec4
        data += struct.pack("ff", 0.0, 0.0)
        data += struct.pack(
            "ffff",  # sel_reds (c, m, y, k)
            sel_reds.get('c', 0), sel_reds.get('m', 0), sel_reds.get('y', 0), sel_reds.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_yellows
            sel_yellows.get('c', 0), sel_yellows.get('m', 0), sel_yellows.get('y', 0), sel_yellows.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_greens
            sel_greens.get('c', 0), sel_greens.get('m', 0), sel_greens.get('y', 0), sel_greens.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_cyans
            sel_cyans.get('c', 0), sel_cyans.get('m', 0), sel_cyans.get('y', 0), sel_cyans.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_blues
            sel_blues.get('c', 0), sel_blues.get('m', 0), sel_blues.get('y', 0), sel_blues.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_magentas
            sel_magentas.get('c', 0), sel_magentas.get('m', 0), sel_magentas.get('y', 0), sel_magentas.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_whites
            sel_whites.get('c', 0), sel_whites.get('m', 0), sel_whites.get('y', 0), sel_whites.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_neutrals
            sel_neutrals.get('c', 0), sel_neutrals.get('m', 0), sel_neutrals.get('y', 0), sel_neutrals.get('k', 0),
        )
        data += struct.pack(
            "ffff",  # sel_blacks
            sel_blacks.get('c', 0), sel_blacks.get('m', 0), sel_blacks.get('y', 0), sel_blacks.get('k', 0),
        )
        
        # LAB grading
        do_lab = 1 if lab_grading else 0
        lab_l_shift = lab_grading.get('l_shift', 0) if lab_grading else 0
        lab_a_shift = lab_grading.get('a_shift', 0) if lab_grading else 0
        lab_b_shift = lab_grading.get('b_shift', 0) if lab_grading else 0
        lab_a_target = lab_grading.get('a_target', 128) if lab_grading else 128
        lab_a_factor = lab_grading.get('a_factor', 0) if lab_grading else 0
        lab_b_target = lab_grading.get('b_target', 128) if lab_grading else 128
        lab_b_factor = lab_grading.get('b_factor', 0) if lab_grading else 0
        
        data += struct.pack(
            "Ifffffff",  # do_lab_grading + 7 floats
            do_lab, lab_l_shift, lab_a_shift, lab_b_shift,
            lab_a_target, lab_a_factor, lab_b_target, lab_b_factor,
        )
        
        # Smoothing
        do_smooth = 1 if smoothing and smoothing.get('radius', 0) > 0 else 0
        smooth_radius = smoothing.get('radius', 0) if smoothing else 0
        smooth_sigma_s = smoothing.get('sigma_s', 75) if smoothing else 75
        smooth_sigma_r = smoothing.get('sigma_r', 75) if smoothing else 75
        
        data += struct.pack(
            "Ifff",  # do_smoothing + 3 floats
            do_smooth, smooth_radius, smooth_sigma_s, smooth_sigma_r,
        )
        
        # Pad to UNIFORM_SIZE
        if len(data) < UNIFORM_SIZE:
            data += b'\x00' * (UNIFORM_SIZE - len(data))
        
        return data

    # =========================================================================
    # Legacy API (backward compatible)
    # =========================================================================
    
    def _build_uniform_data(
        self,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray],
        brightness: float,
        contrast: float,
        saturation: float,
        temp: float,
        tint: float,
        gamma: float,
        shadows: float,
        highlights: float,
        vibrance: float,
        levels: Optional[Dict[str, float]]
    ) -> bytes:
        """Build uniform data (legacy, calls extended version)."""
        return self._build_uniform_data_extended(
            invert, wb_scales, color_matrix,
            brightness, contrast, saturation, temp, tint, gamma,
            shadows, highlights, vibrance, 0.0, levels,
            None, None, None
        )
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def process_negative_conversion(
        self,
        image: np.ndarray,
        wb_scales: Tuple[float, float, float],
        color_matrix: np.ndarray
    ) -> np.ndarray:
        """Process negative to positive conversion."""
        return self.process_full_pipeline(
            image,
            invert=True,
            wb_scales=wb_scales,
            color_matrix=color_matrix
        )
    
    def process_adjustments(
        self,
        image: np.ndarray,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        temp: float = 0,
        tint: float = 0,
        gamma: float = 1.0,
        shadows: float = 0,
        highlights: float = 0,
        vibrance: float = 0,
        hue: float = 0,
        levels: Optional[Dict[str, float]] = None,
        channel_mixer: Optional[Dict[str, Dict[str, float]]] = None,
        curves: Optional[Dict[str, np.ndarray]] = None,
        hsl: Optional[Dict[str, Dict[str, float]]] = None,
        selective_color: Optional[Dict[str, Dict[str, float]]] = None,
        selective_color_relative: bool = True,
        lab_grading: Optional[Dict[str, float]] = None,
        smoothing: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Apply image adjustments."""
        return self.process_full_pipeline(
            image,
            invert=False,
            wb_scales=None,
            color_matrix=None,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            temp=temp,
            tint=tint,
            gamma=gamma,
            shadows=shadows,
            highlights=highlights,
            vibrance=vibrance,
            hue=hue,
            levels=levels,
            channel_mixer=channel_mixer,
            curves=curves,
            hsl=hsl,
            selective_color=selective_color,
            selective_color_relative=selective_color_relative,
            lab_grading=lab_grading,
            smoothing=smoothing,
        )
    
    # =========================================================================
    # Resource Management
    # =========================================================================
    
    def cleanup(self) -> None:
        """Release temporary resources (keeps pipeline)."""
        if self._texture_pool:
            self._texture_pool.clear()
    
    def destroy(self) -> None:
        """Release all GPU resources."""
        self.cleanup()
        self._pipeline = None
        self._uniform_buffer = None
        self._curves_lut_texture = None
        self._initialized = False
