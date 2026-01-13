# Negative-to-Positive Converter (YAFIS)

Desktop application to convert scanned film negatives into positives with a non-destructive adjustment pipeline, film/photo presets, and batch export.

> Project entry point: run the app via [`negative_converter.main.main()`](negative_converter/main.py:1)

## Status / Known issues (current)
- **Compare wipe slider can still be CPU-heavy on large images.** We added throttling and removed a known UI-thread recompute at the slider endpoint, but rendering + histogram work can still be expensive on very large frames. See “Performance notes” below.
- **Auto Tone is async**, but depending on image size and CPU/GPU availability you may still see CPU spikes during compute and during final apply/re-render.

## Key features
- Negative conversion with basic mask detection (C-41, clear, other).
- Non-destructive adjustments (brightness/contrast/saturation/hue/temp/tint, levels, curves, etc.).
- Presets:
  - Film simulations from [`negative_converter/config/presets/film/`](negative_converter/config/presets/film:1)
  - Photo styles from [`negative_converter/config/presets/photo/`](negative_converter/config/presets/photo:1)
- Batch processing (select folder, choose output dir/format/quality, process checked images).
- Optional GPU acceleration (CuPy) for some operations.
- Before/After comparison:
  - Toggle compare (Edit menu)
  - Wipe slider in the View toolbar (“Compare: Before … After”)

## Requirements
- Python 3.10+ recommended
- Core deps: PyQt6, numpy, opencv-python
- Recommended: `opencv-contrib-python` (enables certain features like xphoto-based WB / better denoise availability)

Example install:
```bash
pip install PyQt6 numpy opencv-python opencv-contrib-python
```

Optional GPU (NVIDIA/CUDA):
```bash
pip install cupy-cudaXXX
```
Replace `XXX` with your CUDA version (example: `cupy-cuda118`).

## Run
From repo root:
```bash
python -m negative_converter.main
```

(Equivalent alternative for this repo layout:)
```bash
python negative_converter/main.py
```

## Basic usage
1. Open a scan: `File > Open Negative...`
2. Adjust using the right-side docks:
   - Adjustments
   - Film Simulation
   - Photo Styles
3. Compare:
   - Toggle “Compare Before/After” (Edit menu)
   - Or use the “Compare” wipe slider (View toolbar)
4. Save: `File > Save Positive As...`
5. Batch: `File > Open Folder for Batch...`

## Performance notes (practical)
- Large scans (especially TIFFs) can be expensive to process/render.
- Best results for responsiveness:
  - Avoid dragging sliders extremely fast on very large images.
  - Use GPU acceleration if available.
  - Keep histogram visible if you need it; hide it if you’re chasing max UI responsiveness.

## Documentation
- User manual: [`negative_converter/how-to-use.md`](negative_converter/how-to-use.md)
- Technical overview: [`negative_converter/technical.md`](negative_converter/technical.md)
- Historical analysis/ideas: [`negative_converter/fix_up.md`](negative_converter/fix_up.md)

## Repo layout (high level)
```
negative_converter/
  main.py
  config/
  io/
  processing/
  services/
  ui/
  utils/
```

## Contributing
PRs welcome. Keep changes small, testable, and prefer non-breaking refactors.

## License
Not specified yet.
