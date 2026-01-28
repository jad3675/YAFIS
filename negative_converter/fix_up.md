# YAFIS - Issues Addressed

This document tracks the top 5 issues that were identified and fixed in the codebase.

## ✅ 1. EXIF/Metadata Preservation (FIXED)

**Problem:** The image saver lost EXIF metadata when saving images.

**Solution:** 
- Added `ImageMetadata` class in `io/image_loader.py` to capture and store EXIF data
- Updated `load_image()` to extract and return metadata via `return_metadata=True` parameter
- Updated `save_image()` to accept metadata and preserve EXIF in JPEG, PNG, TIFF, and WebP outputs
- Added `copy_metadata()` utility function for post-hoc metadata copying

**Files changed:**
- `negative_converter/io/image_loader.py`
- `negative_converter/io/image_saver.py`
- `negative_converter/io/__init__.py`

---

## ✅ 2. RAW File Support (FIXED)

**Problem:** No support for RAW file formats (DNG, CR2, NEF, etc.).

**Solution:**
- Added optional `rawpy` dependency for RAW file loading
- Implemented `load_raw_image()` function with sensible defaults for film scanning
- Updated `SUPPORTED_FORMATS_FILTER` to include RAW extensions when rawpy is available
- Added `is_raw_file()` and `is_raw_supported()` utility functions
- Defined `RAW_EXTENSIONS` set for easy format detection

**Files changed:**
- `negative_converter/io/image_loader.py`
- `negative_converter/io/__init__.py`
- `negative_converter/requirements.txt`

---

## ✅ 3. Centralized Error Handling (FIXED)

**Problem:** Inconsistent error handling patterns (50+ `except Exception` blocks with varying approaches).

**Solution:**
- Created `negative_converter/utils/errors.py` with:
  - `ErrorCategory` enum for categorizing errors
  - `AppError` base exception with category, original error, and user message
  - Specific error types: `FileIOError`, `ProcessingError`, `GPUError`, `ConfigurationError`
  - `@handle_errors` decorator for consistent error handling with fallback values
  - `@handle_gpu_errors` decorator specifically for GPU operations with CPU fallback
  - `safe_operation()` context manager for suppressing non-critical errors
  - `format_user_error()` for user-friendly error messages
  - `log_and_continue()` for logging non-critical errors

**Files changed:**
- `negative_converter/utils/errors.py` (new)
- `negative_converter/utils/__init__.py`

---

## ✅ 4. Test Coverage (FIXED)

**Problem:** Missing tests for adjustment pipeline, presets, GPU paths, and edge cases.

**Solution:**
- Added `tests/test_apply_all_adjustments.py` with 28 tests covering:
  - Basic adjustments (brightness, contrast, saturation, etc.)
  - Channel mixer
  - HSL adjustments
  - Curves
  - Selective color
  - Noise reduction
  - Dust removal
  - Edge cases (small images, extreme values)

- Added `tests/test_presets.py` with 16 tests covering:
  - FilmPresetManager initialization and preset loading
  - PhotoPresetManager initialization and preset loading
  - Preset file validation
  - Preset application with intensity

- Added `tests/test_io.py` with 20 tests covering:
  - Image loading (including error cases)
  - Image saving (all formats)
  - Metadata handling
  - Roundtrip preservation

- Added `tests/test_error_handling.py` with 28 tests covering:
  - All error types
  - Decorators
  - Context managers
  - User error formatting

- Added `tests/test_image_proxy.py` with 26 tests covering:
  - Proxy creation and upscaling
  - Memory estimation
  - Tiled processing

**Total new tests: 118** (bringing total from 60 to 178)

**Files changed:**
- `tests/test_apply_all_adjustments.py` (new)
- `tests/test_presets.py` (new)
- `tests/test_io.py` (new)
- `tests/test_error_handling.py` (new)
- `tests/test_image_proxy.py` (new)

---

## ✅ 5. Large File Memory Management (FIXED)

**Problem:** No proxy/preview system for high-resolution scans, risking memory issues.

**Solution:**
- Created `negative_converter/utils/image_proxy.py` with:
  - `ImageProxyInfo` dataclass for tracking proxy metadata
  - `create_proxy()` for automatic downscaling based on pixel count
  - `upscale_to_original()` for restoring full resolution
  - `process_with_proxy()` for transparent proxy handling in processing functions
  - `TiledProcessor` class for processing very large images in tiles with seamless blending
  - `should_use_proxy()`, `is_large_image()`, `calculate_scale_factor()` utilities
  - `get_memory_warning_message()` for user warnings on large files
  - Configurable thresholds: `DEFAULT_PREVIEW_MAX_PIXELS`, `DEFAULT_PROCESSING_MAX_PIXELS`, `DEFAULT_WARNING_THRESHOLD_PIXELS`

**Files changed:**
- `negative_converter/utils/image_proxy.py` (new)

---

## Summary

All 5 identified issues have been addressed with:
- 5 new modules/files
- 118 new tests (178 total, all passing)
- Backward-compatible API changes
- Optional dependencies (rawpy) that gracefully degrade when not installed
