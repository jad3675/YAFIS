"""
GPU Shader Loader - Compiles and caches WGSL shaders.

The primary shader is 'full_pipeline.wgsl' which performs all operations
in a single dispatch for maximum efficiency. Individual operation shaders
are also available for specialized use cases.
"""

import os
from typing import Any, Dict, Optional
from .logger import get_logger

logger = get_logger(__name__)

# Shader directory
SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


class ShaderLoader:
    """
    On-demand WGSL shader compiler with caching.
    Reduces pipeline initialization overhead by reusing compiled modules.
    """
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def load(cls, shader_name: str) -> Any:
        """
        Load and compile a shader by name.
        
        Args:
            shader_name: Name of the shader file (without .wgsl extension)
            
        Returns:
            Compiled wgpu shader module
        """
        if shader_name in cls._cache:
            return cls._cache[shader_name]
        
        path = os.path.join(SHADER_DIR, f"{shader_name}.wgsl")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Shader not found: {path}")
        
        with open(path, "r") as f:
            code = f.read()
        
        from .gpu_device import GPUDevice
        gpu = GPUDevice.get()
        
        if not gpu.is_wgpu or not gpu.wgpu_device:
            raise RuntimeError("wgpu device required for shader compilation")
        
        module = gpu.wgpu_device.create_shader_module(code=code)
        cls._cache[shader_name] = module
        
        logger.debug(f"Compiled shader: {shader_name}")
        return module
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the shader cache."""
        cls._cache.clear()
    
    @classmethod
    def get_shader_path(cls, shader_name: str) -> str:
        """Get the full path to a shader file."""
        return os.path.join(SHADER_DIR, f"{shader_name}.wgsl")
    
    @classmethod
    def shader_exists(cls, shader_name: str) -> bool:
        """Check if a shader file exists."""
        return os.path.exists(cls.get_shader_path(shader_name))
