"""
GPU acceleration module supporting multiple backends.

Backend priority:
1. CuPy (CUDA for NVIDIA, ROCm for AMD) - Best performance for array operations
2. wgpu (Vulkan/Metal/DX12) - Cross-platform GPU compute via shaders
3. NumPy (CPU fallback)

The module provides two levels of GPU acceleration:
- Array-level: CuPy drop-in replacement for NumPy (xp module)
- Pipeline-level: GPUEngine for multi-stage processing without intermediate readbacks

Install options:
- NVIDIA CUDA: pip install cupy-cuda12x
- AMD ROCm: pip install cupy-rocm-6-0
- Cross-platform: pip install wgpu
"""

import numpy as np
from .logger import get_logger

logger = get_logger(__name__)

# Global state for backward compatibility
_gpu_enabled = None
_array_module = None
_cp_module = None
_gpu_backend = None  # "cuda", "rocm", "wgpu", or None
_device_name = None
_gpu_available = None  # Tracks if GPU hardware is available (separate from user toggle)
_user_gpu_disabled = False  # User preference to disable GPU


# =============================================================================
# CuPy Detection (CUDA / ROCm)
# =============================================================================

def _detect_cuda(cp):
    """Attempt CUDA detection. Returns (success, device_name)."""
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            device_name = device_props.get("name", b"Unknown")
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8", errors="ignore")
            return True, device_name
    except (cp.cuda.runtime.CUDARuntimeError, AttributeError):
        pass
    except Exception as e:
        logger.debug("CUDA detection error: %s", e)
    return False, None


def _detect_rocm(cp):
    """Attempt ROCm/HIP detection. Returns (success, device_name)."""
    try:
        if hasattr(cp, "hip") or hasattr(cp.cuda.runtime, "hipGetDeviceCount"):
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                try:
                    device_props = cp.cuda.runtime.getDeviceProperties(0)
                    device_name = device_props.get("name", b"Unknown")
                    if isinstance(device_name, bytes):
                        device_name = device_name.decode("utf-8", errors="ignore")
                except Exception:
                    device_name = "AMD GPU (ROCm)"
                return True, device_name
    except Exception as e:
        logger.debug("ROCm detection error: %s", e)
    return False, None


def _get_cupy_backend_info(cp):
    """Determine which backend CuPy was built with."""
    try:
        cupy_path = cp.__file__.lower() if cp.__file__ else ""
        if "rocm" in cupy_path or "hip" in cupy_path:
            return "rocm"
    except Exception:
        pass
    
    try:
        if hasattr(cp.cuda, "hip") or "hip" in str(type(cp.cuda.runtime)).lower():
            return "rocm"
    except Exception:
        pass
    
    return "cuda"


def _try_cupy():
    """Try to initialize CuPy backend. Returns (success, backend_type, device_name, cp_module)."""
    try:
        import cupy as cp
        
        expected_backend = _get_cupy_backend_info(cp)
        
        if expected_backend == "rocm":
            success, device_name = _detect_rocm(cp)
            if success:
                return True, "rocm", device_name, cp
        
        success, device_name = _detect_cuda(cp)
        if success:
            return True, "cuda", device_name, cp
            
    except ImportError:
        logger.debug("CuPy not installed")
    except Exception as e:
        logger.debug("CuPy initialization failed: %s", e)
    
    return False, None, None, None


def _try_wgpu():
    """Try to initialize wgpu backend. Returns (success, device_name)."""
    try:
        import wgpu
        
        # Request high-performance adapter
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        )
        
        if adapter is None:
            logger.debug("wgpu: No compatible GPU adapter found")
            return False, None
        
        # Request device
        device = adapter.request_device_sync()
        
        if device is None:
            logger.debug("wgpu: Failed to create device")
            return False, None
        
        # Extract device info
        summary = str(adapter.summary)
        backend_name = "WebGPU"
        if "(" in summary:
            backend_name = summary.split("(")[-1].replace(")", "").strip()
        
        device_name = f"{summary.split('(')[0].strip()} ({backend_name})"
        
        logger.debug("wgpu available: %s", device_name)
        return True, device_name
        
    except ImportError:
        logger.debug("wgpu not installed")
    except Exception as e:
        logger.debug("wgpu initialization failed: %s", e)
    
    return False, None


# =============================================================================
# Main Initialization
# =============================================================================

def initialize_gpu():
    """
    Detect and initialize GPU acceleration.
    
    Tries backends in order of preference:
    1. CuPy (CUDA/ROCm) - Best performance for NVIDIA/AMD
    2. wgpu (Vulkan/Metal/DX12) - Cross-platform GPU compute
    3. NumPy (CPU fallback)
    
    Returns:
        tuple: (gpu_enabled, array_module, cp_module)
    """
    global _gpu_enabled, _array_module, _cp_module, _gpu_backend, _device_name, _gpu_available, _user_gpu_disabled
    
    if _gpu_enabled is not None:
        return _gpu_enabled, _array_module, _cp_module
    
    # Check user preference from settings
    try:
        from ..config.settings import UI_DEFAULTS
        _user_gpu_disabled = not UI_DEFAULTS.get("gpu_acceleration_enabled", True)
    except Exception:
        _user_gpu_disabled = False
    
    # Try CuPy first (best performance for array operations)
    success, backend, device_name, cp = _try_cupy()
    if success:
        _gpu_available = True
        _cp_module = cp
        
        if _user_gpu_disabled:
            # GPU available but user disabled it
            _gpu_enabled = False
            _array_module = np
            _gpu_backend = None
            _device_name = device_name  # Keep device name for info display
            logger.info("GPU available (%s) but disabled by user preference", device_name)
            return _gpu_enabled, _array_module, _cp_module
        
        _gpu_enabled = True
        _array_module = cp
        _gpu_backend = backend
        _device_name = device_name
        backend_label = "CUDA (NVIDIA)" if backend == "cuda" else "ROCm (AMD)"
        logger.info("GPU acceleration enabled: %s via %s", device_name, backend_label)
        return _gpu_enabled, _array_module, _cp_module
    
    # Try wgpu second (cross-platform GPU compute)
    success, device_name = _try_wgpu()
    if success:
        _gpu_available = True
        _cp_module = None  # No CuPy, but GPU is available via wgpu
        
        if _user_gpu_disabled:
            # GPU available but user disabled it
            _gpu_enabled = False
            _array_module = np
            _gpu_backend = None
            _device_name = device_name  # Keep device name for info display
            logger.info("GPU available (%s) but disabled by user preference", device_name)
            return _gpu_enabled, _array_module, _cp_module
        
        _gpu_enabled = True
        _array_module = np  # Use NumPy for array operations
        _gpu_backend = "wgpu"
        _device_name = device_name
        logger.info("GPU acceleration enabled: %s via wgpu", device_name)
        return _gpu_enabled, _array_module, _cp_module
    
    # CPU fallback
    _gpu_available = False
    _gpu_enabled = False
    _cp_module = None
    _array_module = np
    _gpu_backend = None
    _device_name = "CPU"
    logger.info(
        "No GPU acceleration available. Install for GPU support:\n"
        "  - NVIDIA: pip install cupy-cuda12x\n"
        "  - AMD ROCm: pip install cupy-rocm-6-0\n"
        "  - Cross-platform: pip install wgpu\n"
        "Using CPU."
    )
    
    return _gpu_enabled, _array_module, _cp_module


# --- Initialize on import ---
GPU_ENABLED, xp, cp_module = initialize_gpu()


# =============================================================================
# Public API - Backward Compatible
# =============================================================================

def get_gpu_state():
    """Returns (gpu_enabled, array_module, cp_module) without reinitializing."""
    if _gpu_enabled is None:
        return initialize_gpu()
    return _gpu_enabled, _array_module, _cp_module


def get_array_module():
    """Returns the configured array module (cupy or numpy)."""
    return xp


def is_gpu_enabled():
    """True if any GPU acceleration is enabled (CuPy or wgpu)."""
    return bool(GPU_ENABLED)


def get_gpu_backend():
    """Returns the GPU backend type: 'cuda', 'rocm', 'wgpu', or None."""
    return _gpu_backend


def is_cupy_backend():
    """True if using CuPy (CUDA or ROCm) backend."""
    return _gpu_backend in ("cuda", "rocm")


def is_wgpu_backend():
    """True if using wgpu (Vulkan/Metal/DX12) backend."""
    return _gpu_backend == "wgpu"


def get_gpu_info():
    """
    Returns a dict with GPU status information for display purposes.
    
    Returns:
        dict with keys:
            - enabled: bool
            - available: bool (True if GPU hardware detected, even if disabled)
            - backend: 'cuda', 'rocm', 'wgpu', or None
            - device_name: str or None
            - user_disabled: bool (True if user has disabled GPU)
            - message: human-readable status string
    """
    if not _gpu_available:
        return {
            "enabled": False,
            "available": False,
            "backend": None,
            "device_name": None,
            "user_disabled": False,
            "message": "No GPU detected (using CPU)",
        }
    
    if _user_gpu_disabled or not _gpu_enabled:
        return {
            "enabled": False,
            "available": True,
            "backend": None,
            "device_name": _device_name,
            "user_disabled": True,
            "message": f"GPU disabled by user ({_device_name} available)",
        }
    
    # Get device name
    device_name = _device_name
    
    if _gpu_backend == "cuda":
        backend_label = "CUDA (NVIDIA)"
    elif _gpu_backend == "rocm":
        backend_label = "ROCm (AMD)"
    elif _gpu_backend == "wgpu":
        backend_label = "wgpu (Vulkan/Metal/DX12)"
    else:
        backend_label = "Unknown"
    
    return {
        "enabled": True,
        "available": True,
        "backend": _gpu_backend,
        "device_name": device_name,
        "user_disabled": False,
        "message": f"GPU acceleration enabled: {device_name} via {backend_label}",
    }


def set_gpu_enabled(enabled):
    """
    Enable or disable GPU acceleration at runtime.
    
    Note: This requires reinitializing the GPU state. Changes take effect
    on the next operation that uses GPU.
    
    Args:
        enabled: bool - True to enable GPU, False to disable
        
    Returns:
        bool - True if the change was applied successfully
    """
    global _gpu_enabled, _array_module, _gpu_backend, _user_gpu_disabled, _gpu_engine
    
    if not _gpu_available:
        logger.warning("Cannot enable GPU - no GPU hardware detected")
        return False
    
    _user_gpu_disabled = not enabled
    
    if enabled:
        # Re-enable GPU
        if _cp_module is not None:
            _gpu_enabled = True
            _array_module = _cp_module
            _gpu_backend = "cuda" if hasattr(_cp_module, 'cuda') else "rocm"
            logger.info("GPU acceleration re-enabled")
        else:
            # wgpu backend
            _gpu_enabled = True
            _array_module = np
            _gpu_backend = "wgpu"
            logger.info("GPU acceleration re-enabled (wgpu)")
    else:
        # Disable GPU
        _gpu_enabled = False
        _array_module = np
        _gpu_backend = None
        # Clear the GPU engine so it gets recreated if re-enabled
        _gpu_engine = None
        logger.info("GPU acceleration disabled by user")
    
    return True


def is_gpu_available():
    """True if GPU hardware is detected (regardless of user preference)."""
    return bool(_gpu_available)


# =============================================================================
# GPU Engine Access
# =============================================================================

_gpu_engine = None


def get_gpu_engine():
    """
    Get the GPU processing engine for multi-stage pipeline operations.
    
    The engine provides efficient GPU processing by:
    - Keeping data on GPU between operations
    - Using persistent textures/buffers
    - Submitting batched commands
    
    Returns:
        GPUEngine instance or None if no GPU available
    """
    global _gpu_engine
    
    if not _gpu_enabled:
        return None
    
    if _gpu_engine is None:
        try:
            from .gpu_engine import GPUEngine
            _gpu_engine = GPUEngine()
        except Exception as e:
            logger.warning("Failed to create GPU engine: %s", e)
            return None
    
    return _gpu_engine


def has_gpu_engine():
    """True if GPU engine is available for pipeline processing."""
    engine = get_gpu_engine()
    return engine is not None and engine.is_available()
