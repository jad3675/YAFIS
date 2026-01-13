# Negative-to-Positive Converter

Desktop app to convert scanned film negatives into positives, then apply non-destructive adjustments and presets. Includes batch export and optional GPU acceleration.

## Run
From the repo root:
```bash
python -m negative_converter.main
```

(Alternative equivalent for this repo layout:)
```bash
python negative_converter/main.py
```

## Key features (current)
- Initial negative conversion (C-41 focused; “clear/other” handling via mask classification).
- Adjustment pipeline (basic + advanced: levels, curves, channel mixer, HSL, selective color, etc.).
- Presets:
  - Film simulations from `negative_converter/config/presets/film/`
  - Photo styles from `negative_converter/config/presets/photo/`
- Batch processing with output format + quality controls.
- Before/After comparison:
  - Toggle compare (Edit menu)
  - Wipe slider in the View toolbar (“Compare: Before … After”)
- Optional ICC handling (if enabled via settings): apply embedded ICC profile and convert to sRGB on load.

## Install
Minimum:
```bash
pip install PyQt6 numpy opencv-python
```

Recommended:
```bash
pip install opencv-contrib-python
```

Optional GPU:
```bash
pip install cupy-cudaXXX
```

## Notes on presets
- Film simulations are JSON files under `negative_converter/config/presets/film/`.
- Photo styles are JSON files under `negative_converter/config/presets/photo/`.
- Preset “preview” is non-destructive; “apply” modifies the base image and enables Undo.

## Docs
- User manual: `negative_converter/how-to-use.md`
- Technical overview: `negative_converter/technical.md`

## License
Not specified yet.