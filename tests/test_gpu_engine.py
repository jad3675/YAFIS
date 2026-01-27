"""
Tests for GPU Engine and related infrastructure.
"""

import pytest
import numpy as np


class TestGPUDevice:
    """Tests for GPUDevice singleton."""
    
    def test_singleton_pattern(self):
        """GPUDevice should be a singleton."""
        from negative_converter.utils.gpu_device import GPUDevice
        
        device1 = GPUDevice.get()
        device2 = GPUDevice.get()
        
        assert device1 is device2
    
    def test_device_info(self):
        """GPUDevice should provide device info."""
        from negative_converter.utils.gpu_device import GPUDevice
        
        device = GPUDevice.get()
        info = device.get_info()
        
        assert "enabled" in info
        assert "backend" in info
        assert "device_name" in info
        assert isinstance(info["enabled"], bool)
    
    def test_backend_detection(self):
        """Backend should be one of the valid options."""
        from negative_converter.utils.gpu_device import GPUDevice
        
        device = GPUDevice.get()
        
        valid_backends = [None, "cupy-cuda", "cupy-rocm", "wgpu"]
        assert device.backend in valid_backends


class TestGPUModule:
    """Tests for the main gpu.py module."""
    
    def test_gpu_info(self):
        """get_gpu_info should return valid info dict."""
        from negative_converter.utils.gpu import get_gpu_info
        
        info = get_gpu_info()
        
        assert "enabled" in info
        assert "backend" in info
        assert "device_name" in info
        assert "message" in info
    
    def test_backend_functions(self):
        """Backend detection functions should be consistent."""
        from negative_converter.utils.gpu import (
            is_gpu_enabled, is_cupy_backend, is_wgpu_backend, get_gpu_backend
        )
        
        backend = get_gpu_backend()
        
        if backend in ("cuda", "rocm"):
            assert is_cupy_backend()
            assert not is_wgpu_backend()
        elif backend == "wgpu":
            assert not is_cupy_backend()
            assert is_wgpu_backend()
        else:
            assert not is_cupy_backend()
            assert not is_wgpu_backend()
    
    def test_array_module(self):
        """Array module should be numpy or cupy."""
        from negative_converter.utils.gpu import get_array_module
        
        xp = get_array_module()
        
        # Should have basic array operations
        assert hasattr(xp, "array")
        assert hasattr(xp, "zeros")
        assert hasattr(xp, "ones")


class TestGPUEngine:
    """Tests for GPUEngine processing."""
    
    @pytest.fixture
    def test_image(self):
        """Create a test image."""
        np.random.seed(42)
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def engine(self):
        """Get GPU engine if available."""
        from negative_converter.utils.gpu import get_gpu_engine
        return get_gpu_engine()
    
    def test_engine_availability(self, engine):
        """Engine should report availability correctly."""
        from negative_converter.utils.gpu import is_gpu_enabled
        
        if is_gpu_enabled():
            assert engine is not None
            assert engine.is_available()
        else:
            # Engine might be None or not available
            pass
    
    def test_negative_conversion(self, engine, test_image):
        """Test negative conversion processing."""
        if engine is None or not engine.is_available():
            pytest.skip("GPU not available")
        
        wb_scales = (1.0, 1.0, 1.0)
        color_matrix = np.eye(3, dtype=np.float32)
        
        result = engine.process_negative_conversion(
            test_image, wb_scales, color_matrix
        )
        
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
    
    def test_adjustments(self, engine, test_image):
        """Test adjustment processing."""
        if engine is None or not engine.is_available():
            pytest.skip("GPU not available")
        
        result = engine.process_adjustments(
            test_image,
            brightness=10,
            contrast=10,
            saturation=10
        )
        
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
    
    def test_identity_adjustments(self, engine, test_image):
        """Identity adjustments should not change image significantly."""
        if engine is None or not engine.is_available():
            pytest.skip("GPU not available")
        
        result = engine.process_adjustments(
            test_image,
            brightness=0,
            contrast=0,
            saturation=0,
            gamma=1.0
        )
        
        # Should be very close to original
        diff = np.abs(result.astype(np.float32) - test_image.astype(np.float32))
        assert np.mean(diff) < 2.0  # Allow small rounding differences
    
    def test_curves_processing(self, engine, test_image):
        """Test curves LUT processing."""
        if engine is None or not engine.is_available():
            pytest.skip("GPU not available")
        
        # Create a simple contrast curve (S-curve)
        identity_lut = np.arange(256, dtype=np.float32)
        
        # Test with identity curves (should not change image much)
        curves = {
            'r': identity_lut.copy(),
            'g': identity_lut.copy(),
            'b': identity_lut.copy(),
            'rgb': identity_lut.copy(),
        }
        
        result = engine.process_adjustments(
            test_image,
            curves=curves
        )
        
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8
        
        # Identity curves should not change image significantly
        diff = np.abs(result.astype(np.float32) - test_image.astype(np.float32))
        assert np.mean(diff) < 2.0
    
    def test_selective_color(self, engine, test_image):
        """Test selective color processing."""
        if engine is None or not engine.is_available():
            pytest.skip("GPU not available")
        
        selective_color = {
            'reds': {'c': 10, 'm': 0, 'y': 0, 'k': 0},
        }
        
        result = engine.process_adjustments(
            test_image,
            selective_color=selective_color,
            selective_color_relative=True
        )
        
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8


class TestGPUResources:
    """Tests for GPU resource management (wgpu only)."""
    
    def test_texture_pool(self):
        """Test texture pool functionality."""
        from negative_converter.utils.gpu import is_wgpu_backend
        
        if not is_wgpu_backend():
            pytest.skip("wgpu not available")
        
        from negative_converter.utils.gpu_resources import TexturePool
        
        pool = TexturePool()
        
        # Get texture
        import wgpu
        usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.STORAGE_BINDING
        tex1 = pool.get(100, 100, usage, "test1")
        
        # Same params should return same texture
        tex2 = pool.get(100, 100, usage, "test1")
        assert tex1 is tex2
        
        # Different size should return different texture
        tex3 = pool.get(200, 200, usage, "test2")
        assert tex1 is not tex3
        
        # Cleanup
        pool.clear()
        assert len(pool) == 0


class TestShaderLoader:
    """Tests for shader loading (wgpu only)."""
    
    def test_shader_exists(self):
        """Test shader existence check."""
        from negative_converter.utils.gpu import is_wgpu_backend
        
        if not is_wgpu_backend():
            pytest.skip("wgpu not available")
        
        from negative_converter.utils.gpu_shaders import ShaderLoader
        
        # These shaders should exist
        assert ShaderLoader.shader_exists("invert")
        assert ShaderLoader.shader_exists("white_balance")
        assert ShaderLoader.shader_exists("adjustments")
        
        # This should not exist
        assert not ShaderLoader.shader_exists("nonexistent_shader")
    
    def test_shader_loading(self):
        """Test shader compilation."""
        from negative_converter.utils.gpu import is_wgpu_backend
        
        if not is_wgpu_backend():
            pytest.skip("wgpu not available")
        
        from negative_converter.utils.gpu_shaders import ShaderLoader
        
        # Should compile without error
        module = ShaderLoader.load("invert")
        assert module is not None
        
        # Should be cached
        module2 = ShaderLoader.load("invert")
        assert module is module2
