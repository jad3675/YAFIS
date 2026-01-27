import logging
import sys
from typing import Optional

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LOG_FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
_CONSOLE_HANDLER = logging.StreamHandler(sys.stdout)
_CONSOLE_HANDLER.setFormatter(_LOG_FORMATTER)

_CONFIGURED = False


def _normalize_level(level: Optional[str]) -> int:
    if not level:
        return logging.INFO
    return LOG_LEVEL_MAP.get(str(level).upper(), logging.INFO)


def configure_logging(level: Optional[str] = None) -> None:
    """Configure root logging once for the application."""
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(_normalize_level(level))

    if not _CONFIGURED:
        root.addHandler(_CONSOLE_HANDLER)
        _CONFIGURED = True


def set_log_level(level: Optional[str]) -> None:
    """Update the root logger level (does not add handlers)."""
    logging.getLogger().setLevel(_normalize_level(level))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)