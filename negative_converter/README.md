# Negative-to-Positive Converter

## Description

This application converts digital scans of film negatives into positive images. It provides a suite of tools for adjusting the resulting image, including basic tonal and color controls, advanced adjustments like curves and selective color, and presets simulating various film stocks and photo styles. The application also supports batch processing of multiple negative images.

## Key Features

*   **Negative Conversion:** Core functionality to invert and color-correct scanned film negatives (primarily targeting C-41 process, with basic handling for others).
*   **Manual Adjustments:** Comprehensive controls for brightness, contrast, saturation, hue, temperature, tint, levels, curves, shadows/highlights, color balance, channel mixer, HSL, selective color, vibrance, clarity, noise reduction, and more.
*   **Film Simulation Presets:** Apply looks based on classic film stocks (loaded from JSON files in `config/presets/film/`).
*   **Photo Style Presets:** Apply creative styles (loaded from JSON files in `config/presets/photo/` and user-specific directory).
*   **Batch Processing:** Process multiple images using the current settings and save them to a specified directory.
*   **GPU Acceleration:** Attempts to use NVIDIA GPU acceleration via CuPy for certain processing steps if available, falling back to CPU (NumPy/OpenCV) otherwise.
*   **User Interface:** Built with PyQt6, featuring a central image viewer and dockable panels for adjustments and presets.

## Technology Stack

*   **Language:** Python 3
*   **UI Framework:** PyQt6
*   **Core Processing:** NumPy, OpenCV-Python (`opencv-python`, `opencv-contrib-python` recommended for all features)
*   **GPU Acceleration (Optional):** CuPy (`cupy-cudaXXX` matching your CUDA version)
*   **User Config (Optional):** appdirs

## Basic Usage

1.  **Dependencies:** Ensure Python 3 and the required libraries (PyQt6, NumPy, OpenCV-Python, optionally CuPy and appdirs) are installed.
    ```bash
    pip install PyQt6 numpy opencv-python opencv-contrib-python appdirs
    # Optional: pip install cupy-cudaXXX (replace XXX with your CUDA version, e.g., 118 or 12x)
    ```
2.  **Run:** Execute the main script from the project's root directory:
    ```bash
    python main.py
    ```
3.  **Open Image:** Use `File > Open Negative...` to load a scanned negative.
4.  **Adjust:** Use the panels on the right to apply adjustments and presets.
5.  **Save:** Use `File > Save Positive As...` to save the result.
6.  **Batch:** Use `File > Open Folder for Batch...` to process multiple images.

## Directory Structure

```
negative_converter/
├── main.py                 # Application entry point
├── config/                 # Configuration files
│   ├── settings.py         # (Potentially unused)
│   └── presets/            # Preset definitions
│       ├── film/           # Film simulation JSON presets
│       └── photo/          # Photo style JSON presets
├── io/                     # Input/Output operations
│   ├── image_loader.py
│   └── image_saver.py
├── processing/             # Core image processing logic
│   ├── converter.py        # Negative-to-positive conversion
│   ├── adjustments.py      # Manual and auto adjustments
│   ├── film_simulation.py  # Film preset application logic
│   ├── photo_presets.py    # Photo preset application logic
│   └── batch.py            # Batch processing implementation
├── ui/                     # User interface components (PyQt6)
│   ├── main_window.py      # Main application window
│   ├── image_viewer.py     # Central image display widget
│   ├── adjustment_panel.py # Panel for manual adjustments
│   ├── preset_panel.py     # Base class/widgets for presets
│   ├── film_preset_panel.py # Panel for film presets
│   ├── photo_preset_panel.py # Panel for photo presets
│   └── filmstrip_widget.py # Widget for batch image selection
├── utils/                  # Utility functions
│   ├── imaging.py          # Core image manipulation helpers (e.g., apply_curve)
│   └── color_profiles.py   # (Potentially unused)
├── photo_presets.json      # Default location for user-saved photo presets (if appdirs not used)
└── README.md               # This file
```

## Contributing

(Placeholder - Contributions are welcome. Please follow standard Git practices: fork, branch, pull request.)

## License

(Placeholder - Specify license, e.g., MIT, GPL, etc.)