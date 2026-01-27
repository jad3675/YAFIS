"""
GPU Benchmark Script - Compare CPU vs GPU performance for image processing.

This script benchmarks the GPU engine against CPU implementations to verify
that GPU acceleration provides meaningful speedup.

Usage:
    python -m negative_converter.benchmark_gpu
"""

import time
import numpy as np
from typing import Callable, Tuple, Dict, Any

from .utils.logger import get_logger
from .utils.gpu import (
    get_gpu_info, is_gpu_enabled, get_gpu_backend,
    is_cupy_backend, is_wgpu_backend, get_gpu_engine
)

logger = get_logger(__name__)


def create_test_image(width: int = 2000, height: int = 1500) -> np.ndarray:
    """Create a test image with realistic color distribution."""
    np.random.seed(42)
    
    # Create base image with gradient
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xx, yy = np.meshgrid(x, y)
    
    # RGB channels with different patterns
    r = (xx * 0.5 + yy * 0.3 + np.random.rand(height, width) * 30).astype(np.float32)
    g = (xx * 0.3 + yy * 0.5 + np.random.rand(height, width) * 30).astype(np.float32)
    b = (xx * 0.2 + yy * 0.4 + np.random.rand(height, width) * 30).astype(np.float32)
    
    image = np.stack([r, g, b], axis=2)
    return np.clip(image, 0, 255).astype(np.uint8)


def benchmark_function(
    func: Callable,
    args: Tuple,
    iterations: int = 5,
    warmup: int = 2
) -> Dict[str, float]:
    """
    Benchmark a function with warmup and multiple iterations.
    
    Returns dict with min, max, mean, and std times in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        "min": min(times),
        "max": max(times),
        "mean": np.mean(times),
        "std": np.std(times),
    }


def cpu_full_pipeline(image: np.ndarray) -> np.ndarray:
    """CPU implementation of full pipeline (invert + WB + matrix + adjustments)."""
    img = image.astype(np.float32)
    
    # Invert
    img = 255.0 - img
    
    # White balance
    scales = np.array([1.1, 1.0, 0.9], dtype=np.float32)
    img = img * scales
    
    # Color matrix
    matrix = np.array([
        [1.50, -0.20, -0.30],
        [-0.30, 1.60, -0.30],
        [-0.20, -0.20, 1.40]
    ], dtype=np.float32)
    h, w, c = img.shape
    flat = img.reshape(-1, 3)
    img = np.dot(flat, matrix.T).reshape(h, w, 3)
    
    # Brightness
    img = img + 10.0
    
    # Contrast
    factor = 1.2
    img = factor * (img - 128.0) + 128.0
    
    # Temperature
    img[:, :, 0] += 5.0
    img[:, :, 2] -= 5.0
    
    # Saturation
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    sat_factor = 1.1
    img = lum[:, :, np.newaxis] + (img - lum[:, :, np.newaxis]) * sat_factor
    
    return np.clip(img, 0, 255).astype(np.uint8)


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 70)
    print("YAFIS GPU Benchmark - Unified Pipeline")
    print("=" * 70)
    
    # GPU Info
    gpu_info = get_gpu_info()
    print(f"\nGPU Status: {gpu_info['message']}")
    print(f"Backend: {gpu_info['backend'] or 'None'}")
    print(f"Device: {gpu_info['device_name'] or 'N/A'}")
    
    # Create test images
    sizes = [
        (1000, 750, "1000x750 (0.75 MP)"),
        (2000, 1500, "2000x1500 (3 MP)"),
        (4000, 3000, "4000x3000 (12 MP)"),
        (6000, 4000, "6000x4000 (24 MP)"),
    ]
    
    print("\n" + "-" * 70)
    print("Full Pipeline Benchmark (Invert + WB + Matrix + Adjustments)")
    print("-" * 70)
    print("\nThis tests the unified pipeline where all operations are performed")
    print("in a single GPU dispatch with one upload and one readback.\n")
    
    engine = get_gpu_engine()
    
    for width, height, label in sizes:
        print(f"Image size: {label}")
        
        image = create_test_image(width, height)
        
        # CPU benchmark
        cpu_result = benchmark_function(cpu_full_pipeline, (image,))
        print(f"  CPU:  {cpu_result['mean']:7.1f} ms (±{cpu_result['std']:.1f})")
        
        # GPU benchmark (if available)
        if engine and engine.is_available():
            wb_scales = (1.1, 1.0, 0.9)
            color_matrix = np.array([
                [1.50, -0.20, -0.30],
                [-0.30, 1.60, -0.30],
                [-0.20, -0.20, 1.40]
            ], dtype=np.float32)
            
            def gpu_full_pipeline(img):
                return engine.process_full_pipeline(
                    img,
                    invert=True,
                    wb_scales=wb_scales,
                    color_matrix=color_matrix,
                    brightness=10,
                    contrast=20,
                    saturation=10,
                    temp=5
                )
            
            try:
                gpu_result = benchmark_function(gpu_full_pipeline, (image,))
                speedup = cpu_result['mean'] / gpu_result['mean']
                speedup_str = f"{speedup:.1f}x" if speedup >= 1 else f"{1/speedup:.1f}x slower"
                print(f"  GPU:  {gpu_result['mean']:7.1f} ms (±{gpu_result['std']:.1f}) - {speedup_str}")
            except Exception as e:
                print(f"  GPU:  Error - {e}")
        else:
            print("  GPU:  Not available")
        
        print()
    
    # Benchmark negative conversion specifically
    print("-" * 70)
    print("Negative Conversion Benchmark (NegativeConverter.convert)")
    print("-" * 70)
    print("\nThis tests the actual negative-to-positive conversion pipeline.\n")
    
    from .processing.converter import NegativeConverter
    converter = NegativeConverter(film_profile="C41")
    
    # Use a smaller image for this test since convert() does more work
    test_sizes = [
        (1000, 750, "1000x750 (0.75 MP)"),
        (2000, 1500, "2000x1500 (3 MP)"),
    ]
    
    for width, height, label in test_sizes:
        print(f"Image size: {label}")
        image = create_test_image(width, height)
        
        def convert_image(img):
            result, _ = converter.convert(img)
            return result
        
        try:
            result = benchmark_function(convert_image, (image,), iterations=3, warmup=1)
            print(f"  Time: {result['mean']:7.1f} ms (±{result['std']:.1f})")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    print("=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    
    # Summary and recommendations
    print("\nNotes:")
    if is_gpu_enabled():
        backend = get_gpu_backend()
        if backend in ("cuda", "rocm"):
            print("- Using CuPy backend - optimal for all image sizes")
        elif backend == "wgpu":
            print("- Using wgpu backend (Vulkan/Metal/DX12)")
            print("- Integrated GPUs may be slower than CPU for small images")
            print("- Discrete GPUs with dedicated VRAM will show better speedup")
            print("- GPU benefits increase with larger images (more parallelism)")
    else:
        print("- No GPU acceleration available")
        print("- Install one of:")
        print("    pip install wgpu          # Cross-platform (Vulkan/Metal/DX12)")
        print("    pip install cupy-cuda12x  # NVIDIA CUDA")
        print("    pip install cupy-rocm-6-0 # AMD ROCm")


if __name__ == "__main__":
    run_benchmark()
