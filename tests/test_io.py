"""Tests for image I/O functionality."""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path


class TestImageLoader:
    """Tests for image loading functionality."""
    
    def test_load_image_imports(self):
        """Image loader module should import successfully."""
        from negative_converter.io import image_loader
        assert image_loader.PILLOW_AVAILABLE
    
    def test_load_nonexistent_file(self):
        """Loading nonexistent file should return None."""
        from negative_converter.io.image_loader import load_image
        result = load_image("/nonexistent/path/to/image.jpg")
        assert result == (None, None, None)
    
    def test_load_invalid_path(self):
        """Loading with invalid path should return None."""
        from negative_converter.io.image_loader import load_image
        result = load_image("")
        assert result == (None, None, None)
        
        result = load_image(None)
        assert result == (None, None, None)
    
    def test_supported_formats_filter(self):
        """Supported formats filter should be defined."""
        from negative_converter.io.image_loader import SUPPORTED_FORMATS_FILTER
        assert SUPPORTED_FORMATS_FILTER is not None
        assert "*.jpg" in SUPPORTED_FORMATS_FILTER.lower()
        assert "*.png" in SUPPORTED_FORMATS_FILTER.lower()
    
    def test_raw_extensions_defined(self):
        """RAW extensions should be defined."""
        from negative_converter.io.image_loader import RAW_EXTENSIONS
        assert RAW_EXTENSIONS is not None
        assert '.dng' in RAW_EXTENSIONS
        assert '.cr2' in RAW_EXTENSIONS
        assert '.nef' in RAW_EXTENSIONS
    
    def test_is_raw_file_function(self):
        """is_raw_file should correctly identify RAW files."""
        from negative_converter.io.image_loader import is_raw_file
        
        assert is_raw_file("photo.dng") == True
        assert is_raw_file("photo.CR2") == True
        assert is_raw_file("photo.nef") == True
        assert is_raw_file("photo.jpg") == False
        assert is_raw_file("photo.png") == False
    
    def test_image_metadata_class(self):
        """ImageMetadata class should work correctly."""
        from negative_converter.io.image_loader import ImageMetadata
        
        metadata = ImageMetadata()
        assert metadata.exif_data is None
        assert metadata.icc_profile is None
        assert metadata.has_exif() == False
        assert metadata.has_icc_profile() == False
        
        # Set some data
        metadata.exif_data = b"test exif data"
        metadata.icc_profile = b"test icc profile"
        assert metadata.has_exif() == True
        assert metadata.has_icc_profile() == True


class TestImageSaver:
    """Tests for image saving functionality."""
    
    def test_save_image_imports(self):
        """Image saver module should import successfully."""
        from negative_converter.io import image_saver
        assert image_saver.PILLOW_AVAILABLE
    
    def test_save_none_image(self):
        """Saving None image should return False."""
        from negative_converter.io.image_saver import save_image
        result = save_image(None, "/tmp/test.jpg")
        assert result == False
    
    def test_save_empty_image(self):
        """Saving empty image should return False."""
        from negative_converter.io.image_saver import save_image
        result = save_image(np.array([]), "/tmp/test.jpg")
        assert result == False
    
    def test_save_invalid_path(self):
        """Saving with invalid path should return False."""
        from negative_converter.io.image_saver import save_image
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = save_image(img, "")
        assert result == False
    
    def test_save_wrong_channels(self):
        """Saving image with wrong channels should return False."""
        from negative_converter.io.image_saver import save_image
        img = np.zeros((10, 10, 4), dtype=np.uint8)  # RGBA instead of RGB
        result = save_image(img, "/tmp/test.jpg")
        assert result == False
    
    def test_save_and_load_jpeg(self, sample_image_uint8):
        """Should save and load JPEG correctly."""
        from negative_converter.io.image_saver import save_image
        from negative_converter.io.image_loader import load_image
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            result = save_image(sample_image_uint8, temp_path, quality=95)
            assert result == True
            assert os.path.exists(temp_path)
            
            # Load back
            loaded, mode, size = load_image(temp_path)
            assert loaded is not None
            assert loaded.shape == sample_image_uint8.shape
            assert loaded.dtype == np.uint8
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_and_load_png(self, sample_image_uint8):
        """Should save and load PNG correctly."""
        from negative_converter.io.image_saver import save_image
        from negative_converter.io.image_loader import load_image
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save
            result = save_image(sample_image_uint8, temp_path, png_compression=6)
            assert result == True
            
            # Load back - PNG is lossless
            loaded, mode, size = load_image(temp_path)
            assert loaded is not None
            assert np.array_equal(loaded, sample_image_uint8)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_save_creates_directory(self, sample_image_uint8):
        """Should create output directory if it doesn't exist."""
        from negative_converter.io.image_saver import save_image
        
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = os.path.join(temp_dir, "subdir", "nested", "image.png")
            result = save_image(sample_image_uint8, nested_path)
            assert result == True
            assert os.path.exists(nested_path)
    
    def test_save_with_metadata(self, sample_image_uint8):
        """Should save with metadata when provided."""
        from negative_converter.io.image_saver import save_image
        from negative_converter.io.image_loader import ImageMetadata
        
        metadata = ImageMetadata()
        # Note: We can't easily create valid EXIF bytes without a real image,
        # but we can test that the function accepts metadata
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_path = f.name
        
        try:
            result = save_image(
                sample_image_uint8, temp_path,
                quality=95, metadata=metadata, preserve_exif=True
            )
            assert result == True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIOIntegration:
    """Integration tests for I/O functionality."""
    
    def test_load_with_metadata_flag(self, sample_image_uint8):
        """Should support return_metadata flag."""
        from negative_converter.io.image_saver import save_image
        from negative_converter.io.image_loader import load_image, ImageMetadata
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save first
            save_image(sample_image_uint8, temp_path)
            
            # Load with metadata
            result = load_image(temp_path, return_metadata=True)
            assert len(result) == 2
            image, metadata = result
            assert image is not None
            assert isinstance(metadata, ImageMetadata)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_roundtrip_preserves_dimensions(self, sample_image_uint8):
        """Save/load roundtrip should preserve image dimensions."""
        from negative_converter.io.image_saver import save_image
        from negative_converter.io.image_loader import load_image
        
        for ext in ['.jpg', '.png', '.tif', '.webp']:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                temp_path = f.name
            
            try:
                save_image(sample_image_uint8, temp_path)
                loaded, _, _ = load_image(temp_path)
                assert loaded is not None, f"Failed to load {ext}"
                assert loaded.shape == sample_image_uint8.shape, f"Shape mismatch for {ext}"
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


class TestErrorHandling:
    """Tests for error handling in I/O."""
    
    def test_load_corrupted_file(self):
        """Should handle corrupted files gracefully."""
        from negative_converter.io.image_loader import load_image
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b"not a valid image file")
            temp_path = f.name
        
        try:
            result = load_image(temp_path)
            assert result == (None, None, None)
        finally:
            os.unlink(temp_path)
    
    def test_save_to_readonly_location(self, sample_image_uint8):
        """Should handle permission errors gracefully."""
        from negative_converter.io.image_saver import save_image
        import sys
        
        # Use a path that definitely won't work on any platform
        if sys.platform == 'win32':
            # On Windows, try a path with invalid characters
            result = save_image(sample_image_uint8, "Z:\\nonexistent\\path\\<>:\"|?*\\image.jpg")
        else:
            # On Unix, try root directory which typically isn't writable
            result = save_image(sample_image_uint8, "/root/definitely/not/writable/image.jpg")
        assert result == False
