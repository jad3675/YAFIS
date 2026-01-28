# This file makes the 'utils' directory a Python package.

from .errors import (
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

__all__ = [
    'AppError',
    'FileIOError',
    'ProcessingError',
    'GPUError',
    'ConfigurationError',
    'ErrorCategory',
    'handle_errors',
    'handle_gpu_errors',
    'safe_operation',
    'log_and_continue',
    'format_user_error',
]