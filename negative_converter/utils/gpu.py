import numpy as np

from negative_converter.utils.logger import get_logger

logger = get_logger(__name__)

# Global state to store the result after the first check
_gpu_enabled = None
_array_module = None
_cp_module = None


def initialize_gpu():
    """
    Detect CuPy + CUDA availability once and cache the result.

    Returns:
        tuple[bool, module, module|None]:
            (gpu_enabled, array_module, cp_module)

        - gpu_enabled: True when CuPy is importable and a CUDA device is accessible
        - array_module: cupy if enabled, otherwise numpy
        - cp_module: cupy module when enabled, otherwise None
    """
    global _gpu_enabled, _array_module, _cp_module

    if _gpu_enabled is not None:
        return _gpu_enabled, _array_module, _cp_module

    try:
        import cupy as cp

        try:
            cp.cuda.runtime.getDeviceCount()
            _gpu_enabled = True
            _cp_module = cp
            _array_module = cp
            logger.info("CuPy found and CUDA device detected. GPU acceleration enabled.")
        except cp.cuda.runtime.CUDARuntimeError as e:
            _gpu_enabled = False
            _cp_module = None
            _array_module = np
            logger.warning(
                "CuPy found but no compatible CUDA GPU detected or driver issue. Using CPU. (%s)",
                e,
            )
        except Exception as e:
            _gpu_enabled = False
            _cp_module = None
            _array_module = np
            logger.exception("Error during CuPy/CUDA initialization. Using CPU. (%s)", e)

    except ImportError:
        _gpu_enabled = False
        _cp_module = None
        _array_module = np
        logger.info("CuPy not found. Using CPU.")
    except Exception as e:
        _gpu_enabled = False
        _cp_module = None
        _array_module = np
        logger.exception("Unexpected error importing/checking CuPy. Using CPU. (%s)", e)

    return _gpu_enabled, _array_module, _cp_module


# --- Initialize on import ---
GPU_ENABLED, xp, cp_module = initialize_gpu()


def get_gpu_state():
    """Returns (gpu_enabled, array_module, cp_module) without reinitializing."""
    if _gpu_enabled is None:
        return initialize_gpu()
    return _gpu_enabled, _array_module, _cp_module


def get_array_module():
    """Returns the configured array module (cupy or numpy)."""
    return xp


def is_gpu_enabled():
    """True if GPU acceleration is enabled."""
    return bool(GPU_ENABLED)