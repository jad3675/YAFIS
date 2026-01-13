# Negative Converter - Technical Details

This document provides a technical overview of the Negative Converter’s architecture and major subsystems.

## Architecture (high level)

- Entry point: [`negative_converter.main.main()`](negative_converter/main.py:1)
- UI (PyQt6): `negative_converter/ui/`
- Processing pipeline: `negative_converter/processing/`
- IO: `negative_converter/io/`
- Config/presets: `negative_converter/config/`
- Utilities: `negative_converter/utils/`
- Service facade (UI → processing/IO): `negative_converter/services/`

## Core conversion pipeline

- The initial negative → positive conversion is handled by the converter in `negative_converter/processing/`.
- It performs:
  - inversion
  - mask/base classification (C‑41 / clear / other)
  - base neutralization and channel correction
  - curve and grading steps to produce an initial positive image

This initial result becomes the “before” image for compare.

## Adjustment pipeline (non-destructive)

The app uses a non-destructive adjustment pipeline applied on top of the converted base image.

Main entry point is via the service facade, ultimately calling:
- [`negative_converter.processing.adjustments.apply_all_adjustments()`](negative_converter/processing/adjustments.py:1477)

The pipeline applies adjustments in a fixed order (basic → WB → levels → curves → advanced color → detail, etc.). Preset preview is injected into this same pipeline through `preset_info` in the adjustments dict.

## Presets

### Film simulations
- Source: `negative_converter/config/presets/film/*.json`
- Applied via the film preset manager. Typical steps include matrix, curves, color balance, dynamic range, and grain.

### Photo styles
- Source: `negative_converter/config/presets/photo/*.json`
- Implemented as a set of adjustment-like parameters (and some effects like grading/grain/vignette).

## Threading model (PyQt6)

To keep UI responsive, the app uses worker objects moved to dedicated `QThread`s, communicating via signals.

Primary workers (defined in [`negative_converter.ui.main_window`](negative_converter/ui/main_window.py:1)):

- `InitialConversionWorker`: runs initial conversion.
- `ImageProcessingWorker`: runs “apply adjustments” requests.
- `BatchProcessingWorker`: runs batch export.
- `AutoToneWorker`: computes auto tone parameters off the UI thread.

Key point: heavy work should not run on the UI thread; use signals to queue work onto worker threads.

## Before/After compare

Compare is implemented in two modes:

1. Toggle compare (before vs after)
2. Wipe compare slider (split view)

Implementation details:
- Wipe compare is controlled from [`negative_converter.ui.main_window.MainWindow.create_view_toolbar()`](negative_converter/ui/main_window.py:742) and applied through `ImageViewer`.
- Slider updates are throttled to reduce repaint storms.

## ICC color management (optional)

If enabled in settings:
- On load, embedded ICC profiles can be applied and converted to sRGB before further processing. This helps avoid “wrong colors” when scanning software embeds profiles.

## Performance considerations

Common hotspots:
- Repeated full-frame processing on large images
- Repeated full-frame repaint/compositing during slider drags
- Histogram computation on large frames

Mitigations used in the codebase include:
- throttling/debouncing certain UI updates
- computing expensive operations off the UI thread where possible

## Related docs

- User manual: [`negative_converter/how-to-use.md`](negative_converter/how-to-use.md:1)
- Root README: [`README.md`](README.md:1)
- App README: [`negative_converter/README.md`](negative_converter/README.md:1)
- Historical analysis (kept as-is): [`negative_converter/fix_up.md`](negative_converter/fix_up.md:1)