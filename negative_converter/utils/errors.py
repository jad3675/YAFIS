# Centralized error handling utilities
"""
Provides consistent error handling patterns across the application.

This module defines:
- Custom exception classes for different error categories
- Error handling decorators for common patterns
- Utility functions for error logging and user messaging
"""

import functools
import traceback
from typing import Any, Callable, Optional, TypeVar, Union
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)

# Type variable for generic function signatures
F = TypeVar('F', bound=Callable[..., Any])


class ErrorCategory(Enum):
    """Categories of errors for consistent handling."""
    RECOVERABLE = "recoverable"      # Can continue with fallback
    USER_INPUT = "user_input"        # Invalid user input
    FILE_IO = "file_io"              # File system errors
    PROCESSING = "processing"        # Image processing errors
    GPU = "gpu"                      # GPU-related errors
    CONFIGURATION = "configuration"  # Settings/config errors
    FATAL = "fatal"                  # Unrecoverable errors


class AppError(Exception):
    """Base exception for application-specific errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.RECOVERABLE,
        original_error: Optional[Exception] = None,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.category = category
        self.original_error = original_error
        self.user_message = user_message or message
    
    def __str__(self) -> str:
        if self.original_error:
            return f"{self.args[0]} (caused by: {type(self.original_error).__name__})"
        return self.args[0]


class FileIOError(AppError):
    """File I/O related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.FILE_IO, **kwargs)
        self.file_path = file_path


class ProcessingError(AppError):
    """Image processing errors."""
    
    def __init__(self, message: str, step: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.PROCESSING, **kwargs)
        self.step = step


class GPUError(AppError):
    """GPU-related errors."""
    
    def __init__(self, message: str, fallback_available: bool = True, **kwargs):
        super().__init__(message, category=ErrorCategory.GPU, **kwargs)
        self.fallback_available = fallback_available


class ConfigurationError(AppError):
    """Configuration/settings errors."""
    
    def __init__(self, message: str, setting_name: Optional[str] = None, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)
        self.setting_name = setting_name


def handle_errors(
    fallback_value: Any = None,
    category: ErrorCategory = ErrorCategory.RECOVERABLE,
    log_level: str = "warning",
    reraise: bool = False,
    user_message: Optional[str] = None,
) -> Callable[[F], F]:
    """
    Decorator for consistent error handling.
    
    Args:
        fallback_value: Value to return on error (can be callable for dynamic fallback).
        category: Error category for logging context.
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'exception').
        reraise: If True, re-raise the exception after logging.
        user_message: Optional user-friendly message for UI display.
    
    Example:
        @handle_errors(fallback_value=None, category=ErrorCategory.PROCESSING)
        def process_image(image):
            # ... processing code ...
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AppError:
                # Re-raise our custom errors
                raise
            except Exception as e:
                # Log the error
                log_func = getattr(logger, log_level, logger.warning)
                log_func(
                    "%s failed in %s.%s: %s",
                    category.value,
                    func.__module__,
                    func.__name__,
                    str(e),
                )
                
                if log_level == "exception":
                    logger.debug("Full traceback:\n%s", traceback.format_exc())
                
                if reraise:
                    raise AppError(
                        str(e),
                        category=category,
                        original_error=e,
                        user_message=user_message,
                    ) from e
                
                # Return fallback value
                if callable(fallback_value):
                    return fallback_value()
                return fallback_value
        
        return wrapper  # type: ignore
    return decorator


def handle_gpu_errors(fallback_func: Optional[Callable] = None) -> Callable[[F], F]:
    """
    Decorator specifically for GPU operations with CPU fallback.
    
    Args:
        fallback_func: Optional CPU fallback function to call on GPU error.
    
    Example:
        @handle_gpu_errors(fallback_func=cpu_process)
        def gpu_process(image):
            # ... GPU processing code ...
            return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "GPU operation %s failed: %s. Falling back to CPU.",
                    func.__name__,
                    str(e),
                )
                
                if fallback_func is not None:
                    return fallback_func(*args, **kwargs)
                
                # If no fallback, try to return first argument (usually the image)
                if args:
                    return args[0]
                return None
        
        return wrapper  # type: ignore
    return decorator


def safe_operation(
    operation_name: str,
    category: ErrorCategory = ErrorCategory.RECOVERABLE,
):
    """
    Context manager for safe operations with consistent error handling.
    
    Example:
        with safe_operation("loading preset", ErrorCategory.FILE_IO):
            preset = load_preset(path)
    """
    class SafeOperationContext:
        def __init__(self):
            self.error: Optional[Exception] = None
            self.success = False
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.error = exc_val
                logger.warning(
                    "%s error during %s: %s",
                    category.value,
                    operation_name,
                    str(exc_val),
                )
                # Suppress the exception
                return True
            self.success = True
            return False
    
    return SafeOperationContext()


def log_and_continue(
    message: str,
    category: ErrorCategory = ErrorCategory.RECOVERABLE,
    level: str = "warning",
) -> None:
    """
    Log an error and continue execution.
    
    Use this for non-critical errors that shouldn't stop processing.
    
    Args:
        message: Error message to log.
        category: Error category for context.
        level: Log level.
    """
    log_func = getattr(logger, level, logger.warning)
    log_func("[%s] %s", category.value, message)


def format_user_error(error: Union[Exception, str], context: Optional[str] = None) -> str:
    """
    Format an error message for user display.
    
    Args:
        error: The error or error message.
        context: Optional context about what operation failed.
    
    Returns:
        User-friendly error message.
    """
    if isinstance(error, AppError):
        return error.user_message
    
    error_str = str(error)
    
    # Clean up common technical error messages
    if "No such file or directory" in error_str:
        return f"File not found{f' while {context}' if context else ''}"
    if "Permission denied" in error_str:
        return f"Permission denied{f' while {context}' if context else ''}"
    if "out of memory" in error_str.lower():
        return "Not enough memory to complete this operation. Try with a smaller image."
    
    # Generic fallback
    if context:
        return f"Error {context}: {error_str}"
    return f"An error occurred: {error_str}"
