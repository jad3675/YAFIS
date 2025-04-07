import logging
import sys
from negative_converter.config import settings # Use absolute import

# Map string level names to logging constants
LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

# Get the desired level from settings, default to INFO if invalid or not found
log_level_str = getattr(settings, 'LOGGING_LEVEL', 'INFO').upper()
log_level = LOG_LEVEL_MAP.get(log_level_str, logging.INFO)

# Basic configuration
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Console Handler
console_handler = logging.StreamHandler(sys.stdout) # Use stdout for console output
console_handler.setFormatter(log_formatter)

# --- Optional: File Handler (Example - uncomment and configure if needed) ---
# try:
#     log_file_path = 'app.log' # Consider making this configurable
#     file_handler = logging.FileHandler(log_file_path)
#     file_handler.setFormatter(log_formatter)
# except Exception as e:
#     print(f"Warning: Could not configure file logging: {e}")
#     file_handler = None
# --------------------------------------------------------------------------

def get_logger(name):
    """
    Gets a logger instance configured with the application's settings.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times if get_logger is called repeatedly for the same name
    if not logger.handlers:
        logger.addHandler(console_handler)
        # if file_handler:
        #     logger.addHandler(file_handler)

    # Prevent messages from propagating to the root logger if handlers are added
    logger.propagate = False

    return logger

# Example usage (can be removed or kept for testing):
# if __name__ == '__main__':
#     logger = get_logger('TestLogger')
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     logger.critical("This is a critical message.")