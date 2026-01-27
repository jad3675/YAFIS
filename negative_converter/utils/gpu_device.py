"""
GPU Device Manager - Unified interface for wgpu and CuPy backends.

This module provides a singleton GPU device that abstracts the underlying
GPU backend (wgpu for Vulkan/Metal/DX12, or CuPy for CUDA/ROCm).

The device is initialized once and reused throughout the application lifetime.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)


class GPUDevice:
    """
    Singleton GPU device manager supporting multiple backends.
    
    Backends (in priority order):
    1. CuPy (CUDA/ROCm) - Best performance for NVIDIA/AMD with native drivers
    2. wgpu (Vulkan/Metal/DX12) - Cross-platform, works on most modern GPUs
    3. CPU fallback - NumPy-based processing
    """
    
    _instance: Optional["GPUDevice"] = None
    
    def __init__(self) -> None:
        if GPUDevice._instance is not None:
            raise RuntimeError("GPUDevice is a singleton - use GPUDevice.get()")
        
        self.backend: Optional[str] = None  # "cupy", "wgpu", or None
        self.device_name: Optional[str] = None
        
        # wgpu-specific
        self._wgpu_adapter: Optional[Any] = None
        self._wgpu_device: Optional[Any] = None
        self._wgpu_limits: Dict[str, Any] = {}
        
        # CuPy-specific
        self._cupy_module: Optional[Any] = None
        
        self._initialize()
    
    @classmethod
    def get(cls) -> "GPUDevice":
        """Get the singleton GPU device instance."""
        if cls._instance is None:
            cls._instance = GPUDevice()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for testing)."""
        if cls._instance is not None:
            cls._instance._cleanup()
            cls._instance = None
    
    def _initialize(self) -> None:
        """Initialize GPU backend in priority order."""
        # Try CuPy first (best performance)
        if self._try_cupy():
            return
        
        # Try wgpu second (cross-platform)
        if self._try_wgpu():
            return
        
        # CPU fallback
        logger.info("No GPU acceleration available. Using CPU fallback.")
        self.backend = None
        self.device_name = "CPU"
    
    def _try_cupy(self) -> bool:
        """Attempt to initialize CuPy backend."""
        try:
            import cupy as cp
            
            # Check for available device
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count == 0:
                logger.debug("CuPy available but no GPU devices found")
                return False
            
            # Get device info
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            device_name = device_props.get("name", b"Unknown GPU")
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8", errors="ignore")
            
            # Determine if CUDA or ROCm
            cupy_path = cp.__file__.lower() if cp.__file__ else ""
            if "rocm" in cupy_path or "hip" in cupy_path:
                backend_type = "cupy-rocm"
                backend_label = "ROCm"
            else:
                backend_type = "cupy-cuda"
                backend_label = "CUDA"
            
            # Test basic operation
            test_arr = cp.array([1, 2, 3])
            _ = cp.sum(test_arr)
            
            self._cupy_module = cp
            self.backend = backend_type
            self.device_name = f"{device_name} ({backend_label})"
            logger.info(f"GPU acceleration enabled: {self.device_name}")
            return True
            
        except ImportError:
            logger.debug("CuPy not installed")
        except Exception as e:
            logger.debug(f"CuPy initialization failed: {e}")
        
        return False
    
    def _try_wgpu(self) -> bool:
        """Attempt to initialize wgpu backend."""
        try:
            import wgpu
            
            # Request high-performance adapter
            adapter = wgpu.gpu.request_adapter_sync(
                power_preference="high-performance"
            )
            
            if adapter is None:
                logger.debug("wgpu: No compatible GPU adapter found")
                return False
            
            # Request device with default limits
            device = adapter.request_device_sync()
            
            if device is None:
                logger.debug("wgpu: Failed to create device")
                return False
            
            # Extract backend info from adapter summary
            summary = str(adapter.summary)
            # Parse backend from summary (e.g., "AMD Radeon... (Vulkan)")
            backend_name = "WebGPU"
            if "(" in summary:
                backend_name = summary.split("(")[-1].replace(")", "").strip()
            
            self._wgpu_adapter = adapter
            self._wgpu_device = device
            self._wgpu_limits = dict(device.limits) if hasattr(device, 'limits') else {}
            self.backend = "wgpu"
            self.device_name = f"{summary.split('(')[0].strip()} ({backend_name})"
            
            logger.info(f"GPU acceleration enabled: {self.device_name}")
            return True
            
        except ImportError:
            logger.debug("wgpu not installed")
        except Exception as e:
            logger.debug(f"wgpu initialization failed: {e}")
        
        return False
    
    def _cleanup(self) -> None:
        """Release GPU resources."""
        self._wgpu_adapter = None
        self._wgpu_device = None
        self._cupy_module = None
    
    @property
    def is_available(self) -> bool:
        """True if any GPU backend is available."""
        return self.backend is not None
    
    @property
    def is_cupy(self) -> bool:
        """True if using CuPy backend."""
        return self.backend in ("cupy-cuda", "cupy-rocm")
    
    @property
    def is_wgpu(self) -> bool:
        """True if using wgpu backend."""
        return self.backend == "wgpu"
    
    @property
    def wgpu_device(self) -> Optional[Any]:
        """Get the wgpu device (None if not using wgpu)."""
        return self._wgpu_device
    
    @property
    def cupy(self) -> Optional[Any]:
        """Get the CuPy module (None if not using CuPy)."""
        return self._cupy_module
    
    @property
    def uniform_alignment(self) -> int:
        """Get minimum uniform buffer alignment for wgpu."""
        if self.is_wgpu and self._wgpu_limits:
            return self._wgpu_limits.get("min_uniform_buffer_offset_alignment", 256)
        return 256
    
    def get_info(self) -> Dict[str, Any]:
        """Get GPU information for display."""
        return {
            "enabled": self.is_available,
            "backend": self.backend,
            "device_name": self.device_name,
            "is_cupy": self.is_cupy,
            "is_wgpu": self.is_wgpu,
        }
    
    def poll(self) -> None:
        """Force GPU queue processing (wgpu only)."""
        if self.is_wgpu and self._wgpu_device:
            if hasattr(self._wgpu_device, "poll"):
                self._wgpu_device.poll()
            elif hasattr(self._wgpu_device, "_poll"):
                self._wgpu_device._poll()
