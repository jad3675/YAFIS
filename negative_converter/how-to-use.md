# Negative Converter - User Manual

This guide explains how to use the Negative Converter application to convert scanned film negatives into positive images, apply adjustments/presets, compare before/after, and batch export.

## Table of Contents

1. [Installation](#installation)
2. [Launching the Application](#launching-the-application)
3. [Main Window Overview](#main-window-overview)
4. [Opening Images](#opening-images)
5. [Using the Interface](#using-the-interface)
6. [Applying Adjustments](#applying-adjustments)
7. [Presets](#presets)
8. [Before/After Compare](#beforeafter-compare)
9. [Saving Images](#saving-images)
10. [Batch Processing](#batch-processing)
11. [Troubleshooting](#troubleshooting)

---

## Installation

Install dependencies:

```bash
pip install PyQt6 numpy opencv-python
```

Recommended for best feature coverage:

```bash
pip install opencv-contrib-python
```

Optional GPU acceleration (NVIDIA + CUDA):

```bash
pip install cupy-cudaXXX
```

## Launching the Application

From the repo root:

```bash
python -m negative_converter.main
```

(Alternative equivalent command for this repo layout:)

```bash
python negative_converter/main.py
```

## Main Window Overview

The main window consists of:

- **Menu bar**: File, Edit, View
- **Toolbars**:
  - View controls (zoom + compare slider)
  - Batch processing toolbar (shows when batch items exist)
- **Image Viewer** (center): shows the current image
- **Dock panels** (right): Adjustments, Film Simulation, Photo Styles
- **Batch Filmstrip** (bottom): appears when batch folder is opened
- **Status bar** (bottom): messages + current file info

## Opening Images

### Single Negative

- `File > Open Negative...` (Ctrl+O)
- The app runs an initial negative→positive conversion and shows the result.

### Folder for Batch Processing

- `File > Open Folder for Batch...`
- The **Batch Filmstrip** appears with thumbnails. Check images to batch-process.

## Using the Interface

### Image Viewer

- Pan: click-drag
- Zoom: mouse wheel or toolbar buttons (Zoom In/Out, 1:1, Fit)
- White Balance picker: click a neutral area after enabling the picker from the adjustment panel.

### Adjustment Panel

Contains grouped controls (varies by build), commonly including:

- Basic: Brightness, Contrast, Saturation, Hue, Temperature, Tint
- Levels: input/output black/white and gamma
- Curves: RGB and per-channel curve editing
- Advanced color: Color balance, HSL, Selective Color, Channel Mixer
- Detail: Vibrance, Clarity, Noise reduction, etc.
- Auto adjustments: AWB / Auto Levels / Auto Color / Auto Tone

Most adjustments update the preview live (with short throttling to keep UI responsive).

### Film Simulation Panel

- Click a preset to preview it.
- Apply it to make it “stick” to the base image.
- Intensity controls the blend amount.
- Some film presets include grain; grain scale controls grain strength.

### Photo Style Panel

- Similar to Film Simulation, but for photo-style looks.
- Photo styles are applied using the current adjustment pipeline + preset-defined parameters.

### Batch Filmstrip

- Single click: preview an image
- Checkboxes: select images for batch processing
- Batch processing uses the current settings (adjustments + applied preset state).

### Toolbars

- **View Controls**: zoom actions and a Compare slider.
- **Batch Processing**:
  - Set Output Dir
  - Format (.jpg / .png / .tif)
  - Quality (JPEG quality or PNG compression depending on format)
  - Process Batch + progress

## Applying Adjustments

Adjustments are cumulative and applied on top of the current converted “base image”.

Tip: If you need to undo a major step (like applying a preset), use Undo (Edit menu).

## Presets

There are two preset systems:

- **Film Simulation**: film stock looks
- **Photo Styles**: creative looks

Terminology:
- **Preview**: temporary (non-destructive), replaced by the next preview/adjustment.
- **Apply**: commits a preset to the base image (undoable).

## Before/After Compare

There are two ways to compare:

1. **Toggle compare** (Edit menu: “Compare Before/After”)
   - Switches between the initial converted image (“before”) and the current adjusted image (“after”).

2. **Wipe compare slider** (View toolbar: “Compare: Before … After”)
   - Shows a split view: before on the left, after on the right.
   - Tip: for very large images, rapid slider movement can be CPU-heavy.

## Saving Images

Use `File > Save Positive As...`.

Note:
- Saving applies the current full adjustment pipeline to ensure the saved image matches your preview intent.

## Batch Processing

1. Open a folder (`File > Open Folder for Batch...`)
2. Check images in the filmstrip
3. Click `Set Output Dir`
4. Choose output format and quality
5. Click `Process Batch`

## Troubleshooting

- **Slow performance / high CPU**: large images are expensive; try smaller previews, reduce rapid slider dragging, or use GPU (CuPy) if available.
- **Noise reduction / xphoto features not working**: install `opencv-contrib-python`.
- **Presets not found**: ensure JSON files exist under:
  - `negative_converter/config/presets/film/`
  - `negative_converter/config/presets/photo/`
- **Color looks “off”**: check Settings for ICC conversion and ensure your scanner profile behavior matches expectations.