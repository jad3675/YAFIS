"""Tests for centralized error handling utilities."""

import pytest
import numpy as np
from negative_converter.utils.errors import (
    AppError,
    FileIOError,
    ProcessingError,
    GPUError,
    ConfigurationError,
    ErrorCategory,
    handle_errors,
    handle_gpu_errors,
    safe_operation,
    log_and_continue,
    format_user_error,
)


class TestAppError:
    """Tests for AppError base class."""
    
    def test_basic_error(self):
        """Basic error creation should work."""
        error = AppError("Test error")
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.RECOVERABLE
        assert error.user_message == "Test error"
    
    def test_error_with_category(self):
        """Error with specific category should work."""
        error = AppError("Test error", category=ErrorCategory.FATAL)
        assert error.category == ErrorCategory.FATAL
    
    def test_error_with_original(self):
        """Error wrapping original exception should work."""
        original = ValueError("Original error")
        error = AppError("Wrapped error", original_error=original)
        assert error.original_error is original
        assert "ValueError" in str(error)
    
    def test_error_with_user_message(self):
        """Error with custom user message should work."""
        error = AppError(
            "Technical error details",
            user_message="Something went wrong. Please try again."
        )
        assert error.user_message == "Something went wrong. Please try again."


class TestSpecificErrors:
    """Tests for specific error types."""
    
    def test_file_io_error(self):
        """FileIOError should include file path."""
        error = FileIOError("File not found", file_path="/path/to/file.jpg")
        assert error.category == ErrorCategory.FILE_IO
        assert error.file_path == "/path/to/file.jpg"
    
    def test_processing_error(self):
        """ProcessingError should include step info."""
        error = ProcessingError("Processing failed", step="brightness")
        assert error.category == ErrorCategory.PROCESSING
        assert error.step == "brightness"
    
    def test_gpu_error(self):
        """GPUError should include fallback info."""
        error = GPUError("GPU operation failed", fallback_available=True)
        assert error.category == ErrorCategory.GPU
        assert error.fallback_available == True
    
    def test_configuration_error(self):
        """ConfigurationError should include setting name."""
        error = ConfigurationError("Invalid setting", setting_name="quality")
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.setting_name == "quality"


class TestHandleErrorsDecorator:
    """Tests for handle_errors decorator."""
    
    def test_successful_function(self):
        """Decorator should not affect successful functions."""
        @handle_errors(fallback_value=None)
        def successful_func():
            return "success"
        
        assert successful_func() == "success"
    
    def test_fallback_on_error(self):
        """Decorator should return fallback on error."""
        @handle_errors(fallback_value="fallback")
        def failing_func():
            raise ValueError("Test error")
        
        assert failing_func() == "fallback"
    
    def test_callable_fallback(self):
        """Decorator should support callable fallback."""
        @handle_errors(fallback_value=lambda: "dynamic fallback")
        def failing_func():
            raise ValueError("Test error")
        
        assert failing_func() == "dynamic fallback"
    
    def test_reraise_option(self):
        """Decorator should reraise when configured."""
        @handle_errors(fallback_value=None, reraise=True)
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(AppError):
            failing_func()
    
    def test_preserves_function_metadata(self):
        """Decorator should preserve function metadata."""
        @handle_errors(fallback_value=None)
        def documented_func():
            """This is a docstring."""
            return "result"
        
        assert documented_func.__name__ == "documented_func"
        assert "docstring" in documented_func.__doc__


class TestHandleGPUErrorsDecorator:
    """Tests for handle_gpu_errors decorator."""
    
    def test_successful_gpu_operation(self):
        """Decorator should not affect successful GPU operations."""
        @handle_gpu_errors()
        def gpu_func(image):
            return image * 2
        
        result = gpu_func(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([2, 4, 6]))
    
    def test_fallback_to_cpu(self):
        """Decorator should call fallback on GPU error."""
        def cpu_fallback(image):
            return image + 1
        
        @handle_gpu_errors(fallback_func=cpu_fallback)
        def gpu_func(image):
            raise RuntimeError("GPU error")
        
        result = gpu_func(np.array([1, 2, 3]))
        assert np.array_equal(result, np.array([2, 3, 4]))
    
    def test_returns_input_without_fallback(self):
        """Without fallback, should return first argument."""
        @handle_gpu_errors()
        def gpu_func(image, param):
            raise RuntimeError("GPU error")
        
        input_image = np.array([1, 2, 3])
        result = gpu_func(input_image, "param")
        assert np.array_equal(result, input_image)


class TestSafeOperation:
    """Tests for safe_operation context manager."""
    
    def test_successful_operation(self):
        """Successful operation should set success flag."""
        with safe_operation("test operation") as ctx:
            result = 1 + 1
        
        assert ctx.success == True
        assert ctx.error is None
    
    def test_failed_operation(self):
        """Failed operation should capture error."""
        with safe_operation("test operation") as ctx:
            raise ValueError("Test error")
        
        assert ctx.success == False
        assert ctx.error is not None
        assert isinstance(ctx.error, ValueError)
    
    def test_suppresses_exception(self):
        """Context manager should suppress exceptions."""
        # This should not raise
        with safe_operation("test operation"):
            raise RuntimeError("Should be suppressed")
        
        # If we get here, exception was suppressed
        assert True


class TestFormatUserError:
    """Tests for format_user_error function."""
    
    def test_format_app_error(self):
        """Should use user_message from AppError."""
        error = AppError("Technical details", user_message="User friendly message")
        result = format_user_error(error)
        assert result == "User friendly message"
    
    def test_format_file_not_found(self):
        """Should format file not found errors nicely."""
        error = FileNotFoundError("No such file or directory: '/path/to/file'")
        result = format_user_error(error, context="loading image")
        assert "File not found" in result
        assert "loading image" in result
    
    def test_format_permission_denied(self):
        """Should format permission errors nicely."""
        error = PermissionError("Permission denied: '/path/to/file'")
        result = format_user_error(error)
        assert "Permission denied" in result
    
    def test_format_memory_error(self):
        """Should format memory errors nicely."""
        error = MemoryError("Out of memory")
        result = format_user_error(error)
        assert "memory" in result.lower()
    
    def test_format_generic_error(self):
        """Should format generic errors with context."""
        error = RuntimeError("Something went wrong")
        result = format_user_error(error, context="processing")
        assert "processing" in result
        assert "Something went wrong" in result


class TestLogAndContinue:
    """Tests for log_and_continue function."""
    
    def test_does_not_raise(self):
        """Function should not raise exceptions."""
        # This should not raise
        log_and_continue("Test message", ErrorCategory.RECOVERABLE)
        assert True
    
    def test_accepts_all_categories(self):
        """Should accept all error categories."""
        for category in ErrorCategory:
            log_and_continue(f"Test {category.value}", category)
        assert True


class TestErrorCategoryEnum:
    """Tests for ErrorCategory enum."""
    
    def test_all_categories_defined(self):
        """All expected categories should be defined."""
        assert ErrorCategory.RECOVERABLE is not None
        assert ErrorCategory.USER_INPUT is not None
        assert ErrorCategory.FILE_IO is not None
        assert ErrorCategory.PROCESSING is not None
        assert ErrorCategory.GPU is not None
        assert ErrorCategory.CONFIGURATION is not None
        assert ErrorCategory.FATAL is not None
    
    def test_category_values(self):
        """Categories should have string values."""
        for category in ErrorCategory:
            assert isinstance(category.value, str)
