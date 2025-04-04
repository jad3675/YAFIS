# Negative-to-Positive Converter

## Overview

The Negative-to-Positive Converter is a powerful Python application designed for photographers and film enthusiasts who need to convert digital scans of film negatives into positive images. It implements a professional-grade image processing pipeline with advanced color correction algorithms, customizable adjustments, and film simulation features.



## Key Features

- **Intelligent Negative Conversion**: Automatically detects film type (C-41 color negative, B&W, etc.) and applies appropriate conversion algorithms
- **GPU Acceleration**: Uses NVIDIA GPU via CuPy when available for faster processing
- **Comprehensive Adjustment Tools**:
  - Basic: Brightness, Contrast, Saturation, Hue, Temperature, Tint
  - Advanced: Levels, Curves, Color Balance, HSL, Selective Color, Channel Mixer
  - Detail: Shadows & Highlights, Vibrance, Clarity, Noise Reduction
  - Automated: Auto White Balance, Auto Levels, Auto Color, Auto Tone
- **Film Simulation**: Apply presets that mimic the look of classic film stocks (Kodachrome, Portra, etc.)
- **Photo Style Presets**: Quick application of creative looks (Black & White, Vintage, etc.)
- **White Balance Picker**: Sample neutral areas to precisely correct color casts
- **Batch Processing**: Process multiple images with the same settings
- **Modern UI**: Intuitive interface with dockable panels and real-time preview

## Technology Stack

- **Language**: Python 3
- **UI Framework**: PyQt6
- **Image Processing**: NumPy, OpenCV-Python
- **GPU Acceleration**: CuPy (optional)
- **File Management**: appdirs (optional)

## Requirements

- Python 3.8 or higher
- PyQt6
- NumPy
- OpenCV-Python (opencv-contrib-python recommended for all features)
- CuPy (optional, for GPU acceleration)
- appdirs (optional, for user config storage)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jad3675/negative_2_positive.git
cd negative_2_positive
```

### 2. Install Dependencies

```bash
pip install PyQt6 numpy opencv-python opencv-contrib-python appdirs pillow
```

For GPU acceleration (optional, NVIDIA GPU required):
```bash
# Replace XXX with your CUDA version (e.g., 12x for CUDA 12.x)
pip install cupy-cudaXXX
```

## Quick Start

1. **Launch the Application**:
   ```bash
   python -m negative_converter.main
   ```

2. **Open a Negative**:
   - Go to `File > Open Negative...` or press `Ctrl+O`
   - Select a scanned film negative image
   - The application automatically performs the initial conversion

3. **Adjust the Image**:
   - Use the panels on the right to fine-tune the result
   - Try the auto adjustment buttons for quick results
   - Apply film simulation or photo style presets as desired

4. **Save the Result**:
   - Go to `File > Save Positive As...` or press `Ctrl+Shift+S`
   - Choose a location and format (JPEG, PNG, TIFF)

5. **Batch Processing**:
   - Go to `File > Open Folder for Batch...`
   - Select images in the filmstrip
   - Configure settings and press "Process Batch"

## Documentation

- [How-to Guide](how-to-use.md): Detailed usage instructions for all features
- [Technical Details](technical_theory.md): In-depth explanation of the processing pipeline and architecture

## Directory Structure

```
negative_converter/
├── main.py                 # Application entry point
├── config/                 # Configuration and preset files
│   └── presets/            # Film and photo presets
│       ├── film/           # Film simulation presets
│       └── photo/          # Photo style presets
├── io/                     # Image loading/saving
├── processing/             # Core image processing algorithms
├── ui/                     # User interface components
├── utils/                  # Utility functions
└── README.md               # This file
```

## License

[Specify license here, e.g., MIT, GPL, etc.]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

[Add credits, inspiration, etc. as appropriate]
