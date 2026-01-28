"""Tests for image proxy/preview functionality."""

import numpy as np
import pytest
from negative_converter.utils.image_proxy import (
    ImageProxyInfo,
    estimate_memory_usage,
    get_pixel_count,
    should_use_proxy,
    is_large_image,
    calculate_scale_factor,
    create_proxy,
    upscale_to_original,
    process_with_proxy,
    TiledProcessor,
    get_memory_warning_message,
    DEFAULT_PREVIEW_MAX_PIXELS,
    DEFAULT_WARNING_THRESHOLD_PIXELS,
)


class TestImageProxyInfo:
    """Tests for ImageProxyInfo dataclass."""
    
    def test_megapixels_calculation(self):
        """Should calculate megapixels correctly."""
        info = ImageProxyInfo(
            original_shape=(2000, 3000, 3),
            proxy_shape=(1000, 1500, 3),
            scale_factor=0.5,
            is_proxy=True,
        )
        assert info.original_megapixels == 6.0
        assert info.proxy_megapixels == 1.5
    
    def test_memory_saved_calculation(self):
        """Should calculate memory saved correctly."""
        info = ImageProxyInfo(
            original_shape=(2000, 3000, 3),
            proxy_shape=(1000, 1500, 3),
            scale_factor=0.5,
            is_proxy=True,
        )
        # Original: 2000*3000*3 = 18MB, Proxy: 1000*1500*3 = 4.5MB
        # Saved: ~13.5MB
        assert info.memory_saved_mb > 10


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_estimate_memory_usage(self):
        """Should estimate memory usage correctly."""
        # 100x100x3 uint8 = 30,000 bytes = ~0.03 MB
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        memory = estimate_memory_usage(img)
        assert 0.02 < memory < 0.04
    
    def test_estimate_memory_usage_none(self):
        """Should handle None image."""
        assert estimate_memory_usage(None) == 0.0
    
    def test_get_pixel_count(self):
        """Should count pixels correctly."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        assert get_pixel_count(img) == 20000
    
    def test_get_pixel_count_none(self):
        """Should handle None image."""
        assert get_pixel_count(None) == 0
    
    def test_should_use_proxy_small_image(self):
        """Small images should not need proxy."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert should_use_proxy(img) == False
    
    def test_should_use_proxy_large_image(self):
        """Large images should use proxy."""
        # 3000x3000 = 9 million pixels > 4 million default
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        assert should_use_proxy(img) == True
    
    def test_is_large_image(self):
        """Should detect very large images."""
        # 10000x10000 = 100 million pixels > 50 million threshold
        img = np.zeros((10000, 10000, 3), dtype=np.uint8)
        assert is_large_image(img) == True
        
        # 1000x1000 = 1 million pixels < threshold
        small_img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        assert is_large_image(small_img) == False
    
    def test_calculate_scale_factor_no_scaling(self):
        """Small images should have scale factor 1.0."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert calculate_scale_factor(img) == 1.0
    
    def test_calculate_scale_factor_needs_scaling(self):
        """Large images should have scale factor < 1.0."""
        # 3000x3000 = 9 million pixels, target 4 million
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        scale = calculate_scale_factor(img)
        assert 0 < scale < 1.0
        # Should scale to approximately sqrt(4M/9M) ≈ 0.67
        assert 0.6 < scale < 0.75


class TestCreateProxy:
    """Tests for create_proxy function."""
    
    def test_create_proxy_small_image(self):
        """Small images should not be proxied."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        proxy, info = create_proxy(img)
        
        assert info.is_proxy == False
        assert info.scale_factor == 1.0
        assert proxy.shape == img.shape
    
    def test_create_proxy_large_image(self):
        """Large images should be proxied."""
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        proxy, info = create_proxy(img)
        
        assert info.is_proxy == True
        assert info.scale_factor < 1.0
        assert proxy.shape[0] < img.shape[0]
        assert proxy.shape[1] < img.shape[1]
    
    def test_create_proxy_preserves_content(self):
        """Proxy should preserve image content (approximately)."""
        # Create image with distinct regions
        img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        img[:1000, :1000] = [255, 0, 0]  # Red quadrant
        img[:1000, 1000:] = [0, 255, 0]  # Green quadrant
        img[1000:, :1000] = [0, 0, 255]  # Blue quadrant
        img[1000:, 1000:] = [255, 255, 0]  # Yellow quadrant
        
        proxy, info = create_proxy(img, max_pixels=1_000_000)
        
        # Check that colors are preserved in proxy
        h, w = proxy.shape[:2]
        # Top-left should be reddish
        assert proxy[h//4, w//4, 0] > 200
    
    def test_create_proxy_none_image(self):
        """Should handle None image."""
        proxy, info = create_proxy(None)
        assert proxy is None
        assert info.is_proxy == False


class TestUpscaleToOriginal:
    """Tests for upscale_to_original function."""
    
    def test_upscale_non_proxy(self):
        """Non-proxy images should be returned unchanged."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        info = ImageProxyInfo(
            original_shape=(100, 100, 3),
            proxy_shape=(100, 100, 3),
            scale_factor=1.0,
            is_proxy=False,
        )
        
        result = upscale_to_original(img, info)
        assert result.shape == img.shape
    
    def test_upscale_proxy(self):
        """Proxy images should be upscaled to original size."""
        proxy = np.zeros((500, 500, 3), dtype=np.uint8)
        info = ImageProxyInfo(
            original_shape=(1000, 1000, 3),
            proxy_shape=(500, 500, 3),
            scale_factor=0.5,
            is_proxy=True,
        )
        
        result = upscale_to_original(proxy, info)
        assert result.shape == (1000, 1000, 3)


class TestProcessWithProxy:
    """Tests for process_with_proxy function."""
    
    def test_process_small_image_no_proxy(self):
        """Small images should be processed directly."""
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        
        def add_brightness(image, amount=50):
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        
        result = process_with_proxy(img, add_brightness, preview_mode=True, amount=50)
        
        assert result is not None
        assert result.shape == img.shape
        assert result.mean() > img.mean()
    
    def test_process_large_image_with_proxy(self):
        """Large images should use proxy in preview mode."""
        img = np.full((3000, 3000, 3), 100, dtype=np.uint8)
        
        def add_brightness(image, amount=50):
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        
        result = process_with_proxy(
            img, add_brightness,
            max_preview_pixels=1_000_000,
            preview_mode=True,
            amount=50
        )
        
        assert result is not None
        assert result.shape == img.shape  # Should be upscaled back
    
    def test_process_full_resolution(self):
        """preview_mode=False should process at full resolution."""
        img = np.full((3000, 3000, 3), 100, dtype=np.uint8)
        
        processed_shapes = []
        
        def track_shape(image, amount=50):
            processed_shapes.append(image.shape)
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        
        result = process_with_proxy(
            img, track_shape,
            preview_mode=False,
            amount=50
        )
        
        # Should have processed at full resolution
        assert processed_shapes[0] == img.shape


class TestTiledProcessor:
    """Tests for TiledProcessor class."""
    
    def test_tiled_processor_small_image(self):
        """Small images should work with tiled processor."""
        processor = TiledProcessor(tile_size=256, overlap=32)
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        
        def add_brightness(image, amount=50):
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        
        result = processor.process(img, add_brightness, amount=50)
        
        assert result is not None
        assert result.shape == img.shape
    
    def test_tiled_processor_large_image(self):
        """Large images should be processed in tiles."""
        processor = TiledProcessor(tile_size=512, overlap=64)
        img = np.full((2000, 2000, 3), 100, dtype=np.uint8)
        
        def add_brightness(image, amount=50):
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)
        
        result = processor.process(img, add_brightness, amount=50)
        
        assert result is not None
        assert result.shape == img.shape
        # Result should be brighter
        assert result.mean() > img.mean()
    
    def test_tiled_processor_seamless(self):
        """Tiled processing should produce seamless results in the interior."""
        processor = TiledProcessor(tile_size=256, overlap=64)
        
        # Create gradient image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img[i, :, :] = i // 2
        
        def identity(image):
            return image
        
        result = processor.process(img, identity)
        
        # Check interior region (away from edges where blending occurs)
        # The interior should be very close to original
        margin = 64  # Skip edge regions affected by blending
        interior_result = result[margin:-margin, margin:-margin]
        interior_orig = img[margin:-margin, margin:-margin]
        
        assert np.allclose(interior_result, interior_orig, atol=5)


class TestMemoryWarning:
    """Tests for memory warning functionality."""
    
    def test_no_warning_small_image(self):
        """Small images should not trigger warning."""
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        warning = get_memory_warning_message(img)
        assert warning is None
    
    def test_warning_large_image(self):
        """Very large images should trigger warning."""
        # 10000x10000 = 100 million pixels
        img = np.zeros((10000, 10000, 3), dtype=np.uint8)
        warning = get_memory_warning_message(img)
        assert warning is not None
        assert "Large image" in warning
        assert "MP" in warning
    
    def test_no_warning_none_image(self):
        """None image should not trigger warning."""
        warning = get_memory_warning_message(None)
        assert warning is None
