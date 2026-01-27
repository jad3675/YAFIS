"""
Processing strategies for negative conversion.

This module provides an abstraction layer over GPU and CPU processing paths,
reducing code duplication and making it easier to add new backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProcessingStrategy(ABC):
    """Abstract base class for image processing strategies."""
    
    @abstractmethod
    def process_core_pipeline(
        self,
        image: np.ndarray,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Process the core conversion pipeline (invert, WB, color matrix).
        
        Args:
            image: Input uint8 RGB image.
            invert: Whether to invert the image.
            wb_scales: White balance scale factors (R, G, B) or None.
            color_matrix: 3x3 color correction matrix or None.
            
        Returns:
            Processed image as float32 array.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this processing strategy is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this strategy."""
        pass


class GPUStrategy(ProcessingStrategy):
    """GPU-accelerated processing using the GPU engine."""
    
    def __init__(self):
        self._engine = None
        self._available = None
    
    @property
    def name(self) -> str:
        return "GPU"
    
    def is_available(self) -> bool:
        if self._available is None:
            try:
                from ..utils.gpu import has_gpu_engine
                self._available = has_gpu_engine()
            except Exception:
                self._available = False
        return self._available
    
    def _get_engine(self):
        if self._engine is None:
            from ..utils.gpu import get_gpu_engine
            self._engine = get_gpu_engine()
        return self._engine
    
    def process_core_pipeline(
        self,
        image: np.ndarray,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        """Process using GPU engine."""
        engine = self._get_engine()
        
        result_u8 = engine.process_full_pipeline(
            image,
            invert=invert,
            wb_scales=wb_scales,
            color_matrix=color_matrix
        )
        
        return result_u8.astype(np.float32)


class CPUStrategy(ProcessingStrategy):
    """CPU-based processing with optional CuPy acceleration."""
    
    @property
    def name(self) -> str:
        return "CPU"
    
    def is_available(self) -> bool:
        return True  # CPU is always available
    
    def process_core_pipeline(
        self,
        image: np.ndarray,
        invert: bool,
        wb_scales: Optional[Tuple[float, float, float]],
        color_matrix: Optional[np.ndarray]
    ) -> np.ndarray:
        """Process using CPU (with optional CuPy for array ops)."""
        from ..utils.gpu import xp, is_cupy_backend
        
        img_float = xp.asarray(image, dtype=xp.float32)
        
        # Step 1: Invert
        if invert:
            img_float = 255.0 - img_float
        
        # Step 2: White Balance
        if wb_scales is not None:
            scales = xp.asarray(wb_scales, dtype=xp.float32)
            img_float = img_float * scales
        
        # Step 3: Color Matrix
        if color_matrix is not None:
            matrix = xp.asarray(color_matrix)
            h, w, c = img_float.shape
            flat = img_float.reshape(-1, 3)
            img_float = xp.dot(flat, matrix.T).reshape(h, w, 3)
        
        # Transfer back to CPU if needed
        if is_cupy_backend():
            return xp.asnumpy(img_float)
        return img_float


class ProcessingContext:
    """
    Context for selecting and using processing strategies.
    
    Automatically selects the best available strategy and provides
    fallback to CPU if GPU processing fails.
    """
    
    def __init__(self, prefer_gpu: bool = True):
        """
        Initialize processing context.
        
        Args:
            prefer_gpu: Whether to prefer GPU processing when available.
        """
        self._gpu_strategy = GPUStrategy()
        self._cpu_strategy = CPUStrategy()
        self._prefer_gpu = prefer_gpu
    
    def get_strategy(self) -> ProcessingStrategy:
        """Get the best available processing strategy."""
        if self._prefer_gpu and self._gpu_strategy.is_available():
            return self._gpu_strategy
        return self._cpu_strategy
    
    def process_core_pipeline(
        self,
        image: np.ndarray,
        invert: bool = False,
        wb_scales: Optional[Tuple[float, float, float]] = None,
        color_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, str]:
        """
        Process the core pipeline with automatic fallback.
        
        Args:
            image: Input uint8 RGB image.
            invert: Whether to invert the image.
            wb_scales: White balance scale factors or None.
            color_matrix: Color correction matrix or None.
            
        Returns:
            Tuple of (processed float32 image, strategy name used).
        """
        strategy = self.get_strategy()
        
        try:
            result = strategy.process_core_pipeline(
                image, invert, wb_scales, color_matrix
            )
            return result, strategy.name
        except Exception as e:
            if strategy.name == "GPU":
                logger.warning(f"GPU processing failed, falling back to CPU: {e}")
                result = self._cpu_strategy.process_core_pipeline(
                    image, invert, wb_scales, color_matrix
                )
                return result, "CPU (fallback)"
            raise


class WhiteBalanceCalculator:
    """Calculates white balance scale factors for different film types."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
    
    def calculate_for_c41(self, mask_color: np.ndarray) -> np.ndarray:
        """Calculate WB scales for C-41 film based on mask color."""
        inverted = 255.0 - mask_color
        inverted = np.maximum(inverted, 1e-6)
        target = self.params.get('wb_target_gray', 128.0)
        scales = target / inverted
        return np.clip(scales,
                       self.params.get('wb_clamp_min', 0.8),
                       self.params.get('wb_clamp_max', 1.3))
    
    def calculate_for_ecn2(self, mask_color: np.ndarray) -> np.ndarray:
        """Calculate WB scales for ECN-2 film based on mask color."""
        inverted = 255.0 - mask_color
        inverted = np.maximum(inverted, 1e-6)
        target = self.params.get('wb_target_gray_ecn2', 140.0)
        scales = target / inverted
        return np.clip(scales,
                       self.params.get('wb_ecn2_clamp_min', 0.7),
                       self.params.get('wb_ecn2_clamp_max', 1.5))
    
    def calculate_gray_world(
        self,
        inverted_image: np.ndarray,
        gentle: bool = False
    ) -> np.ndarray:
        """
        Calculate Gray World AWB scales from inverted image.
        
        Args:
            inverted_image: Already inverted image as float32.
            gentle: If True, use very conservative clamping (for E-6).
        """
        avg_r = max(np.mean(inverted_image[:, :, 0]), 1e-6)
        avg_g = max(np.mean(inverted_image[:, :, 1]), 1e-6)
        avg_b = max(np.mean(inverted_image[:, :, 2]), 1e-6)
        overall_avg = (avg_r + avg_g + avg_b) / 3.0
        
        scales = np.array([
            overall_avg / avg_r,
            overall_avg / avg_g,
            overall_avg / avg_b
        ], dtype=np.float32)
        
        if gentle:
            # E-6 slide film - very gentle correction
            return np.clip(scales,
                           self.params.get('wb_e6_clamp_min', 0.95),
                           self.params.get('wb_e6_clamp_max', 1.05))
        elif self.params.get('gray_world_clamp_enabled', True):
            return np.clip(scales,
                           self.params.get('wb_clamp_min', 0.8),
                           self.params.get('wb_clamp_max', 1.3))
        return scales
    
    def get_scales_for_classification(
        self,
        classification: str,
        mask_color: np.ndarray,
        image: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Get appropriate WB scales based on film classification.
        
        Args:
            classification: Film type classification string.
            mask_color: Detected mask color.
            image: Original image (needed for gray world calculation).
            
        Returns:
            WB scale factors or None if no WB needed.
            Returns string "gray_world" or "gray_world_gentle" if 
            calculation needs to happen after inversion.
        """
        if classification == "Likely C-41":
            return self.calculate_for_c41(mask_color)
        elif classification == "Likely ECN-2":
            return self.calculate_for_ecn2(mask_color)
        elif classification == "Likely E-6":
            return "gray_world_gentle"  # Marker for later calculation
        elif classification == "Likely B&W":
            return None  # No color correction
        elif classification == "Clear/Near Clear":
            return None
        else:  # Unknown/Other
            return "gray_world"  # Marker for later calculation
