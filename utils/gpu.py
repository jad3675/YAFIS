import numpy as np
import sys

# Global state to store the result after the first check
_gpu_enabled = None
_array_module = None
_cp_module = None # Explicitly define cp_module global

def initialize_gpu():
    """
    Checks for CuPy installation and CUDA device availability.

    Sets global state for GPU enabled status and the array module (CuPy or NumPy).
    Prints informative messages about the detected environment.

    Returns:
        tuple: (bool, module) - A tuple containing:
               - gpu_enabled (bool): True if CuPy is found and a CUDA device is accessible, False otherwise.
               - xp (module): The array module to use (cupy or numpy).
    """
    global _gpu_enabled, _array_module, _cp_module
    if _gpu_enabled is not None:
        # Already initialized
        return _gpu_enabled, _array_module

    try:
        import cupy as cp
        _cp_module = cp # Assign to global
        # Check if CUDA is available and a device is accessible
        try:
            cp.cuda.runtime.getDeviceCount()
            _gpu_enabled = True
            # _array_module = cp # Already assigned below
            _array_module = _cp_module
            print("[GPU Util] CuPy found and GPU detected. GPU acceleration enabled.")
        except cp.cuda.runtime.CUDARuntimeError:
            _gpu_enabled = False
            _cp_module = None # Ensure it's None if import failed
            _cp_module = None # Ensure it's None if import failed
            # _cp_module remains assigned if import succeeded
            # _cp_module remains assigned if import succeeded
            _array_module = np
            print("[GPU Util Warning] CuPy found but no compatible CUDA GPU detected or driver issue. Using CPU.")
        except Exception as e: # Catch other potential CUDA/CuPy errors during init
             _gpu_enabled = False
             _array_module = np
             print(f"[GPU Util Error] Error during CuPy/CUDA initialization: {e}. Using CPU.")

    except ImportError:
        print("[GPU Util Warning] CuPy not found. Install CuPy (e.g., 'pip install cupy-cudaXXX') for GPU acceleration. Using CPU.")
        _gpu_enabled = False
        _array_module = np
    except Exception as e: # Catch other potential errors during import
        print(f"[GPU Util Error] Unexpected error importing or checking CuPy: {e}. Using CPU.")
        _gpu_enabled = False
        _array_module = np

    return _gpu_enabled, _array_module, _cp_module

# --- Initialize on import ---
# Unpack all three return values
GPU_ENABLED, xp, _cp_module_init = initialize_gpu()
# Export the determined cp_module using the value returned from the init call
cp_module = _cp_module_init

# --- Optional: Function to get the current state without re-initializing ---
def get_gpu_state():
    """Returns the determined GPU state and array module."""
    if _gpu_enabled is None:
        # Should have been initialized on import, but as a fallback:
        return initialize_gpu()
    return _gpu_enabled, _array_module