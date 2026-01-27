# YAFIS - Yet Another Film Inversion Software

Desktop application for converting scanned film negatives into positives with automatic film type detection, GPU acceleration, and a non-destructive adjustment pipeline.

> Run via `python -m negative_converter.main`

## Features

### Film Type Detection
Automatic detection and classification of film base types:
- **C-41** - Standard color negative (Kodak Portra, Fuji 400H, etc.)
- **ECN-2** - Motion picture negative (Kodak Vision3, etc.)
- **E-6** - Slide/reversal film (Ektachrome, Velvia, etc.)
- **B&W** - Black and white negative
- **Clear/Unknown** - Fallback with gray world white balance

### Processing
- Automatic orange mask removal with per-film-type white balance
- GPU acceleration via wgpu (WebGPU) or CuPy (CUDA)
- Non-destructive adjustment pipeline (brightness, contrast, curves, levels, HSL, etc.)
- LAB color grading and HSV saturation control

### Presets
- Film simulations: [`config/presets/film/`](negative_converter/config/presets/film)
- Photo styles: [`config/presets/photo/`](negative_converter/config/presets/photo)
- Custom film profiles: [`config/film_profiles/`](negative_converter/config/film_profiles)

### Batch Processing
- Process entire folders of scans
- Configurable output format (JPEG, PNG, TIFF)
- Progress tracking and error handling

## Installation

### Requirements
- Python 3.10+
- PyQt6, NumPy, OpenCV

```bash
pip install PyQt6 numpy opencv-python
```

### Optional: GPU Acceleration

For wgpu (recommended, cross-platform):
```bash
pip install wgpu
```

For CuPy (NVIDIA CUDA):
```bash
pip install cupy-cuda12x  # Replace with your CUDA version
```

## Usage

### Quick Start
```bash
python -m negative_converter.main
```

1. **Open** a scanned negative: `File > Open Negative...`
2. **Review** the auto-detected film type (shown in status bar, click to override)
3. **Adjust** using the right-side panels (Adjustments, Film Simulation, Photo Styles)
4. **Compare** before/after: `Edit > Compare Before/After` or use the wipe slider
5. **Save**: `File > Save Positive As...`

### Batch Processing
1. `File > Open Folder for Batch...`
2. Select images to process
3. Choose output directory and format
4. Click Process

### Manual Film Type Override
Click the detected film type in the status bar to manually select:
- Auto (use detection)
- C-41 (Color Negative)
- ECN-2 (Motion Picture)
- E-6 (Slide/Reversal)
- B&W (Black & White)
- Clear/Near Clear
- Unknown/Other

## Configuration

### Settings
`Edit > Settings...` opens the configuration dialog with collapsible sections:
- **Mask Detection** - HSV thresholds for each film type
- **White Balance** - Per-film-type WB parameters
- **Channel Curves** - Gamma and clipping settings
- **Color Grading** - LAB correction and saturation

### Custom Film Profiles
Create custom profiles in `config/film_profiles/`. See the [Film Profiles README](negative_converter/config/film_profiles/README.md) for details.

## Architecture

```
negative_converter/
├── config/
│   ├── film_profiles/     # Film type processing profiles
│   ├── presets/           # Film simulation & photo style presets
│   └── settings.py        # Application settings
├── processing/
│   ├── converter.py       # Main conversion pipeline
│   ├── mask_detection.py  # Film base detection & classification
│   ├── processing_strategy.py  # GPU/CPU abstraction
│   └── adjustments.py     # Image adjustment algorithms
├── ui/
│   ├── main_window.py     # Main application window
│   ├── settings_dialog.py # Settings with collapsible sections
│   └── ...
└── utils/
    ├── gpu.py             # GPU backend detection
    ├── gpu_engine.py      # Unified GPU processing engine
    └── shaders/           # WGSL compute shaders
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_converter.py -v
python -m pytest tests/test_mask_detection.py -v
python -m pytest tests/test_ui_integration.py -v
```

Test coverage includes:
- Mask detection for all film types
- Edge cases (dark scans, overexposed, mixed lighting)
- White balance calculation
- UI component integration
- Film profile loading

## Documentation

- [How to Use](negative_converter/how-to-use.md) - User guide
- [Technical Overview](negative_converter/technical.md) - Architecture details
- [Film Profiles](negative_converter/config/film_profiles/README.md) - Creating custom profiles

## Performance Notes

- GPU acceleration significantly improves processing speed for large scans
- wgpu backend works on Windows, macOS, and Linux
- CuPy backend requires NVIDIA GPU with CUDA
- For very large TIFFs, consider hiding the histogram panel for better UI responsiveness

## Contributing

PRs welcome. Please:
- Add tests for new functionality
- Keep changes focused and testable
- Run the test suite before submitting

## License

Not specified yet.
