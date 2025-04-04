# Application entry point
import sys
import os
from PyQt6.QtWidgets import QApplication

# Ensure the package structure is recognized when running main.py directly
# Add the parent directory (where negative_converter package resides) to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(script_dir) # Go up one level from negative_converter/
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Import the MainWindow class from the ui subpackage
from negative_converter.ui.main_window import MainWindow


def main():
    """Main function to run the application."""
    # Create the Qt Application
    app = QApplication(sys.argv)

    # Optional: Set application details
    app.setApplicationName("Negative Converter")
    app.setOrganizationName("ExampleOrg") # Replace if desired
    # app.setWindowIcon(QIcon("path/to/icon.png")) # Set application icon

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    # Start the Qt event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    # This block ensures the code runs only when the script is executed directly
    main()