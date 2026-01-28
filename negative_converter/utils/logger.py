"""
Logging configuration for YAFIS.

Provides centralized logging setup with appropriate log levels:
- DEBUG: Detailed diagnostic info (pixel values, algorithm internals)
- INFO: General operational messages (file loaded, processing complete)
- WARNING: Recoverable issues (fallback used, deprecated feature)
- ERROR: Failures that prevent operation completion
- CRITICAL: Application-level failures

Guidelines for log levels:
- Use DEBUG for: loop iterations, intermediate values, cache hits/misses
- Use INFO for: user-initiated actions, major state changes
- Use WARNING for: recoverable errors, deprecations, performance issues
- Use ERROR for: operation failures, invalid input
- Use CRITICAL for: unrecoverable errors, data corruption risks
"""

import logging
import sys
from typing import Optional, Dict

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Module-specific log level overrides
# Use this to quiet noisy modules or increase verbosity for debugging
MODULE_LOG_LEVELS: Dict[str, int] = {
    # Quiet down verbose modules in production
    "negative_converter.ui.filmstrip_widget": logging.WARNING,
    "negative_converter.ui.lazy_filmstrip": logging.WARNING,
    "negative_converter.ui.preset_preview_cache": logging.WARNING,
    "negative_converter.processing.adjustments": logging.WARNING,
    "negative_converter.utils.gpu_engine": logging.WARNING,
    "negative_converter.utils.gpu_resources": logging.WARNING,
}

_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
_CONSOLE_HANDLER.setFormatter(_LOG_FORMATTER)

_CONFIGURED = False


def _normalize_level(level: Optional[str]) -> int:
    """Convert string log level to logging constant."""
    if not level:
        return logging.INFO
    return LOG_LEVEL_MAP.get(str(level).upper(), logging.INFO)


def configure_logging(level: Optional[str] = None) -> None:
    """
    Configure root logging once for the application.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(_normalize_level(level))

    if not _CONFIGURED:
        root.addHandler(_CONSOLE_HANDLER)
        _CONFIGURED = True
    
    # Apply module-specific overrides
    _apply_module_levels()


def _apply_module_levels() -> None:
    """Apply module-specific log level overrides."""
    for module_name, level in MODULE_LOG_LEVELS.items():
        logging.getLogger(module_name).setLevel(level)


def set_log_level(level: Optional[str]) -> None:
    """
    Update the root logger level.
    
    Args:
        level: Log level string.
    """
    logging.getLogger().setLevel(_normalize_level(level))
    _apply_module_levels()


def set_module_log_level(module_name: str, level: str) -> None:
    """
    Set log level for a specific module.
    
    Args:
        module_name: Full module name (e.g., "negative_converter.ui.main_window").
        level: Log level string.
    """
    MODULE_LOG_LEVELS[module_name] = _normalize_level(level)
    logging.getLogger(module_name).setLevel(_normalize_level(level))


def get_logger(name: str) -> logging.Logger:
    """
    Return a module logger.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Apply module-specific level if configured
    if name in MODULE_LOG_LEVELS:
        logger.setLevel(MODULE_LOG_LEVELS[name])
    
    return logger


def log_once(logger: logging.Logger, level: int, message: str, *args) -> None:
    """
    Log a message only once (useful for warnings in loops).
    
    Args:
        logger: Logger instance.
        level: Log level.
        message: Message format string.
        *args: Format arguments.
    """
    if not hasattr(logger, '_logged_once'):
        logger._logged_once = set()
    
    key = (level, message)
    if key not in logger._logged_once:
        logger._logged_once.add(key)
        logger.log(level, message, *args)