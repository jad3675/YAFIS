# GPU pipeline for full resolution processing
"""
GPU-accelerated pipeline for full resolution image processing and export.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import numpy as np
import time

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStage:
    """A stage in the processing pipeline."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    image: np.ndarray
    stages_executed: List[str]
    total_time: float
    stage_times: Dict[str, float]
    used_gpu: bool


class GPUPipeline:
    """
    GPU-accelerated image processing pipeline.
    
    Provides a unified interface for processing images through multiple
    stages, automatically using GPU when available.
    """
    
    def __init__(self):
        self._stages: List[PipelineStage] = []
        self._gpu_available = False
        self._gpu_backend = None
        self._check_gpu()
    
    def _check_gpu(self):
        """Check GPU availability."""
        try:
            from .gpu import get_gpu_info
            info = get_gpu_info()
            self._gpu_available = info.get("enabled", False)
            self._gpu_backend = info.get("backend")
            logger.info("GPU pipeline: %s (backend: %s)", 
                       "enabled" if self._gpu_available else "disabled",
                       self._gpu_backend or "none")
        except Exception:
            self._gpu_available = False
            logger.debug("GPU not available for pipeline")
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self._gpu_available
    
    def add_stage(self, stage: PipelineStage):
        """Add a processing stage."""
        self._stages.append(stage)
    
    def clear_stages(self):
        """Clear all stages."""
        self._stages.clear()
    
    def set_stages_from_adjustments(self, adjustments: Dict[str, Any]):
        """
        Configure pipeline stages from adjustment dictionary.
        
        Args:
            adjustments: Dictionary of adjustment parameters.
        """
        self.clear_stages()
        
        # Basic adjustments
        basic_params = {
            k: v for k, v in adjustments.items()
            if k in ('brightness', 'contrast', 'saturation', 'hue', 'temp', 'tint')
        }
        if any(v != 0 for v in basic_params.values()):
            self.add_stage(PipelineStage("basic", params=basic_params))
        
        # Levels
        levels_params = {
            k: v for k, v in adjustments.items()
            if k.startswith('levels_')
        }
        if levels_params:
            has_levels = (
                levels_params.get('levels_in_black', 0) != 0 or
                levels_params.get('levels_in_white', 255) != 255 or
                levels_params.get('levels_gamma', 1.0) != 1.0 or
                levels_params.get('levels_out_black', 0) != 0 or
                levels_params.get('levels_out_white', 255) != 255
            )
            if has_levels:
                self.add_stage(PipelineStage("levels", params=levels_params))
        
        # Curves
        curves_params = {
            k: v for k, v in adjustments.items()
            if k.startswith('curves_')
        }
        if curves_params:
            self.add_stage(PipelineStage("curves", params=curves_params))
        
        # HSL
        hsl_params = {
            k: v for k, v in adjustments.items()
            if k.startswith('hsl_')
        }
        if any(v != 0 for k, v in hsl_params.items() if isinstance(v, (int, float))):
            self.add_stage(PipelineStage("hsl", params=hsl_params))
        
        # Channel mixer
        mixer_params = {
            k: v for k, v in adjustments.items()
            if k.startswith('mixer_')
        }
        if mixer_params:
            self.add_stage(PipelineStage("mixer", params=mixer_params))
        
        # Selective color
        sel_params = {
            k: v for k, v in adjustments.items()
            if k.startswith('sel_')
        }
        if any(v != 0 for k, v in sel_params.items() if isinstance(v, (int, float))):
            self.add_stage(PipelineStage("selective_color", params=sel_params))
        
        # Noise reduction
        nr_strength = adjustments.get('noise_reduction_strength', 0)
        if nr_strength > 0:
            self.add_stage(PipelineStage("noise_reduction", params={'strength': nr_strength}))
        
        # Dust removal
        if adjustments.get('dust_removal_enabled', False):
            self.add_stage(PipelineStage("dust_removal", params={
                'sensitivity': adjustments.get('dust_removal_sensitivity', 50),
                'radius': adjustments.get('dust_removal_radius', 3)
            }))
    
    def execute(
        self,
        image: np.ndarray,
        force_cpu: bool = False,
        progress_callback: Callable[[str, float], None] = None
    ) -> PipelineResult:
        """
        Execute the pipeline on an image.
        
        Args:
            image: Input image (RGB uint8).
            force_cpu: Force CPU processing even if GPU available.
            progress_callback: Optional callback(stage_name, progress).
            
        Returns:
            PipelineResult with processed image and timing info.
        """
        if image is None:
            return PipelineResult(
                image=None,
                stages_executed=[],
                total_time=0,
                stage_times={},
                used_gpu=False
            )
        
        use_gpu = self._gpu_available and not force_cpu
        result_image = image.copy()
        stages_executed = []
        stage_times = {}
        total_start = time.time()
        
        for i, stage in enumerate(self._stages):
            if not stage.enabled:
                continue
            
            stage_start = time.time()
            
            if progress_callback:
                progress = (i / len(self._stages)) * 100
                progress_callback(stage.name, progress)
            
            try:
                if use_gpu:
                    result_image = self._execute_stage_gpu(stage, result_image)
                else:
                    result_image = self._execute_stage_cpu(stage, result_image)
                
                stages_executed.append(stage.name)
                stage_times[stage.name] = time.time() - stage_start
                
            except Exception as e:
                logger.warning("Stage %s failed: %s, falling back to CPU", stage.name, e)
                if use_gpu:
                    try:
                        result_image = self._execute_stage_cpu(stage, result_image)
                        stages_executed.append(stage.name)
                        stage_times[stage.name] = time.time() - stage_start
                    except Exception as e2:
                        logger.error("CPU fallback also failed for %s: %s", stage.name, e2)
        
        if progress_callback:
            progress_callback("complete", 100)
        
        return PipelineResult(
            image=result_image,
            stages_executed=stages_executed,
            total_time=time.time() - total_start,
            stage_times=stage_times,
            used_gpu=use_gpu
        )
    
    def _execute_stage_gpu(self, stage: PipelineStage, image: np.ndarray) -> np.ndarray:
        """Execute a stage using GPU."""
        try:
            from .gpu_engine import GPUEngine
            engine = GPUEngine()
            
            if not engine.is_available():
                raise RuntimeError("GPU engine not available")
            
            if stage.name == "basic":
                return engine.apply_adjustments(image, stage.params)
            elif stage.name == "levels":
                return engine.apply_levels(image, stage.params)
            elif stage.name == "curves":
                return engine.apply_curves(image, stage.params)
            else:
                # Fall back to CPU for unsupported stages
                return self._execute_stage_cpu(stage, image)
                
        except Exception as e:
            logger.debug("GPU execution failed for %s: %s", stage.name, e)
            raise
    
    def _execute_stage_cpu(self, stage: PipelineStage, image: np.ndarray) -> np.ndarray:
        """Execute a stage using CPU."""
        from ..processing.adjustments import apply_all_adjustments
        
        if stage.name == "basic":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "levels":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "curves":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "hsl":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "mixer":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "selective_color":
            return apply_all_adjustments(image, stage.params)
        elif stage.name == "noise_reduction":
            return self._apply_noise_reduction(image, stage.params)
        elif stage.name == "dust_removal":
            return self._apply_dust_removal(image, stage.params)
        else:
            logger.warning("Unknown pipeline stage: %s", stage.name)
            return image
    
    def _apply_noise_reduction(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Apply noise reduction."""
        import cv2
        
        strength = params.get('strength', 0)
        if strength <= 0:
            return image
        
        # Scale strength to OpenCV parameters
        h = int(strength / 10)  # 0-10 range
        
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    def _apply_dust_removal(self, image: np.ndarray, params: dict) -> np.ndarray:
        """Apply dust removal."""
        from ..processing.dust_detection import auto_clean_image
        
        sensitivity = params.get('sensitivity', 50) / 100.0
        radius = params.get('radius', 3)
        
        cleaned, _ = auto_clean_image(image, sensitivity, radius)
        return cleaned


def process_for_export(
    image: np.ndarray,
    adjustments: Dict[str, Any],
    use_gpu: bool = True,
    progress_callback: Callable[[str, float], None] = None
) -> np.ndarray:
    """
    Process an image for export using the GPU pipeline.
    
    This is the main entry point for full-resolution processing.
    
    Args:
        image: Input image (RGB uint8).
        adjustments: Adjustment parameters.
        use_gpu: Whether to use GPU if available.
        progress_callback: Optional progress callback.
        
    Returns:
        Processed image.
    """
    pipeline = GPUPipeline()
    pipeline.set_stages_from_adjustments(adjustments)
    
    result = pipeline.execute(
        image,
        force_cpu=not use_gpu,
        progress_callback=progress_callback
    )
    
    logger.info(
        "Pipeline complete: %d stages in %.2fs (GPU: %s)",
        len(result.stages_executed),
        result.total_time,
        result.used_gpu
    )
    
    return result.image


def benchmark_pipeline(
    image: np.ndarray,
    adjustments: Dict[str, Any],
    iterations: int = 5
) -> Dict[str, Any]:
    """
    Benchmark pipeline performance.
    
    Args:
        image: Test image.
        adjustments: Test adjustments.
        iterations: Number of iterations.
        
    Returns:
        Benchmark results.
    """
    pipeline = GPUPipeline()
    pipeline.set_stages_from_adjustments(adjustments)
    
    # CPU benchmark
    cpu_times = []
    for _ in range(iterations):
        result = pipeline.execute(image, force_cpu=True)
        cpu_times.append(result.total_time)
    
    # GPU benchmark (if available)
    gpu_times = []
    if pipeline.gpu_available:
        for _ in range(iterations):
            result = pipeline.execute(image, force_cpu=False)
            gpu_times.append(result.total_time)
    
    return {
        "cpu_avg": sum(cpu_times) / len(cpu_times) if cpu_times else 0,
        "cpu_min": min(cpu_times) if cpu_times else 0,
        "cpu_max": max(cpu_times) if cpu_times else 0,
        "gpu_avg": sum(gpu_times) / len(gpu_times) if gpu_times else 0,
        "gpu_min": min(gpu_times) if gpu_times else 0,
        "gpu_max": max(gpu_times) if gpu_times else 0,
        "gpu_available": pipeline.gpu_available,
        "speedup": (sum(cpu_times) / sum(gpu_times)) if gpu_times and sum(gpu_times) > 0 else 0,
    }
