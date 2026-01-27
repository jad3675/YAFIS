"""Application entrypoint.

Supported run modes:
- Recommended: `python -m negative_converter`

Notes:
- Running as a script (`python negative_converter/main.py`) is intentionally not supported
  because it breaks Python package imports unless you mutate `sys.path`. Keeping the
  entrypoint import-clean avoids hiding packaging/import issues.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from negative_converter.config import settings
from negative_converter.ui.main_window import MainWindow
from negative_converter.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the Qt application."""

    # Configure logging once at process start.
    configure_logging(settings.LOGGING_LEVEL)
    logger.info("Logging configured: level=%s", settings.LOGGING_LEVEL)

    app = QApplication(sys.argv)
    app.setApplicationName("YAFIS")
    app.setOrganizationName("YAFIS")

    main_window = MainWindow()
    main_window.show()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()