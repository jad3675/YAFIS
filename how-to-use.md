# Negative Converter - User Manual

This guide provides detailed instructions on how to use the Negative Converter application to convert your scanned film negatives into positive images and apply various adjustments and styles.

## Table of Contents

1.  [Installation](#installation)
2.  [Launching the Application](#launching-the-application)
3.  [Main Window Overview](#main-window-overview)
4.  [Opening Images](#opening-images)
    *   [Single Negative](#single-negative)
    *   [Folder for Batch Processing](#folder-for-batch-processing)
5.  [Using the Interface](#using-the-interface)
    *   [Image Viewer](#image-viewer)
    *   [Adjustment Panel](#adjustment-panel)
    *   [Film Simulation Panel](#film-simulation-panel)
    *   [Photo Style Panel](#photo-style-panel)
    *   [Batch Filmstrip](#batch-filmstrip)
    *   [Toolbars](#toolbars)
    *   [Menu Bar](#menu-bar)
6.  [Applying Adjustments](#applying-adjustments)
    *   [Basic Adjustments](#basic-adjustments)
    *   [Levels](#levels)
    *   [Curves](#curves)
    *   [Color Balance](#color-balance)
    *   [HSL (Hue/Saturation/Lightness)](#hsl-huesaturationlightness)
    *   [Selective Color](#selective-color)
    *   [Channel Mixer](#channel-mixer)
    *   [Shadows & Highlights](#shadows--highlights)
    *   [Vibrance & Clarity](#vibrance--clarity)
    *   [Noise Reduction](#noise-reduction)
    *   [White Balance Picker](#white-balance-picker)
    *   [Automatic Adjustments](#automatic-adjustments)
7.  [Applying Presets](#applying-presets)
    *   [Film Simulation](#film-simulation)
    *   [Photo Styles](#photo-styles)
    *   [Preview vs. Apply](#preview-vs-apply)
    *   [Intensity Control](#intensity-control)
8.  [Saving Images](#saving-images)
    *   [Saving a Single Image](#saving-a-single-image)
9.  [Batch Processing](#batch-processing)
    *   [Selecting Images](#selecting-images)
    *   [Setting Output Directory](#setting-output-directory)
    *   [Choosing Format and Quality](#choosing-format-and-quality)
    *   [Starting the Batch](#starting-the-batch)
    *   [Monitoring Progress](#monitoring-progress)
10. [Troubleshooting](#troubleshooting)

---

## 1. Installation

Before using the application, ensure you have Python 3 installed along with the necessary libraries. Open your terminal or command prompt and run:

```bash
pip install PyQt6 numpy opencv-python opencv-contrib-python appdirs
```

**Optional (for GPU Acceleration):** If you have an NVIDIA GPU and CUDA installed, you can install CuPy for potential performance improvements:

```bash
# Replace XXX with your CUDA version (e.g., 118 for CUDA 11.8, 12x for CUDA 12.x)
pip install cupy-cudaXXX
```

## 2. Launching the Application

Navigate to the project's root directory in your terminal and run:

```bash
python main.py
```

## 3. Main Window Overview

The main application window consists of:

*   **Menu Bar:** Access file operations, editing, view options, etc.
*   **Toolbars:** Quick access buttons for common actions (View Controls, Batch Processing).
*   **Image Viewer:** The central area where the loaded image is displayed.
*   **Dockable Panels (Right):** Tabs for Adjustments, Film Simulation, and Photo Styles. You can rearrange or hide these via the `View` menu.
*   **Batch Filmstrip (Bottom):** Appears when a folder is opened for batch processing. Displays thumbnails of images in the folder.
*   **Status Bar:** Displays messages, tips, and progress information.

## 4. Opening Images

### Single Negative

*   Go to `File > Open Negative...` or use the shortcut `Ctrl+O`.
*   Navigate to and select your scanned negative image file (common formats like JPG, PNG, TIFF are supported).
*   The application will automatically perform the initial negative-to-positive conversion and display the result in the Image Viewer.

### Folder for Batch Processing

*   Go to `File > Open Folder for Batch...`.
*   Select the folder containing the negative images you want to process.
*   The **Batch Filmstrip** panel will appear at the bottom, showing thumbnails of the detected image files. The **Batch Processing** toolbar will also become visible.

## 5. Using the Interface

### Image Viewer

*   **Panning:** Click and drag the image to pan.
*   **Zooming:** Use the mouse wheel or the zoom buttons (+, -, 1:1, Fit) on the **View Controls** toolbar. Shortcuts: `Ctrl++` (Zoom In), `Ctrl+-` (Zoom Out).
*   **White Balance Picker:** When activated from the Adjustment Panel, click on an area in the image that should be neutral gray or white to set the white balance.

### Adjustment Panel

This panel contains various sliders and controls to fine-tune the image. It's organized into collapsible sections:

*   **Basic:** Brightness, Contrast, Saturation, Hue, Temperature, Tint.
*   **Levels:** Input/Output sliders for Black/White points and a Gamma slider.
*   **Curves:** A graphical curve editor. Select RGB, Red, Green, or Blue channel. Click to add points, drag points to adjust the curve. Double-click to remove a point. Right-click for options like Reset.
*   **Color Balance:** Sliders for Red/Green/Blue balance (multiplicative) and Cyan-Red/Magenta-Green/Yellow-Blue shifts (additive, often used for split toning effects).
*   **HSL:** Adjust Hue, Saturation, and Lightness for specific color ranges (Reds, Yellows, Greens, Cyans, Blues, Magentas).
*   **Selective Color:** Adjust the CMYK components within specific color ranges (Reds, Yellows, ..., Whites, Neutrals, Blacks). Choose Relative (adjusts based on existing color) or Absolute mode.
*   **Channel Mixer:** Modify output channels (Red, Green, Blue) based on a mix of the input channels. Includes a Monochrome option.
*   **Detail:** Shadows, Highlights, Vibrance, Clarity, Noise Reduction.
*   **Auto Adjustments:** Buttons for Auto White Balance (AWB), Auto Levels, Auto Color, Auto Tone.
*   **White Balance Picker:** Button to activate the eyedropper tool for sampling a neutral color in the Image Viewer.

Changes made in this panel are typically reflected live (or with a short delay) in the Image Viewer.

### Film Simulation Panel

*   Displays a list of available film simulation presets (e.g., Kodachrome, Velvia).
*   **Preview:** Click a preset name to see a temporary preview in the Image Viewer.
*   **Apply:** Double-click a preset name or click 'Apply' after selecting to permanently apply it to the current image state.
*   **Intensity:** Controls the strength of the applied preset (0% = original, 100% = full effect).
*   **Grain Scale:** Adjusts the intensity/visibility of the film grain effect defined in the preset.

### Photo Style Panel

*   Similar to the Film Simulation panel, but lists photo style presets (e.g., Vivid Cool, Golden, BW).
*   **Preview:** Click a preset name for a temporary preview.
*   **Apply:** Double-click or click 'Apply' to apply the style.
*   **Intensity:** Controls the strength of the applied style.

### Batch Filmstrip

*   Appears when a folder is opened via `File > Open Folder for Batch...`.
*   Shows thumbnails of images in the selected folder.
*   **Checkbox:** Check the box on each thumbnail you want to include in the batch process.
*   **Single Click:** Previews the selected image in the main viewer *without* applying current adjustments (useful for quickly checking images).
*   **Double Click:** Loads the selected image into the main viewer and applies the initial conversion (same as opening it individually).

### Toolbars

*   **View Controls:** Buttons for Zoom In, Zoom Out, Reset Zoom (1:1), Fit to Window. Enabled when an image is loaded.
*   **Batch Processing:** Controls for batch operations. Becomes visible when a batch folder is opened. Contains:
    *   `Set Output Dir`: Button to select the destination folder for batch results.
    *   `Output: [Path]`: Label showing the currently selected output directory.
    *   `Format:`: Dropdown to select the output file format (JPG, PNG, TIF).
    *   `Quality:`: Spinner to set quality (1-100 for JPG, 0-9 for PNG). Only relevant for selected formats.
    *   `Process Batch`: Button to start processing checked images. Enabled only when an output directory is set and at least one image is checked.
    *   `Progress Bar`: Shows the progress during batch processing.

Toolbars can be shown/hidden via the `View` menu.

### Menu Bar

*   **File:** Open Negative, Open Folder for Batch, Save As, Exit.
*   **Edit:** Undo (reverts the last *destructive* operation like applying a preset or auto-adjustment).
*   **View:** Toggle visibility of panels (Adjustment, Film Sim, Photo Style, Batch Filmstrip) and toolbars (Batch, View Controls).

## 6. Applying Adjustments

Adjustments are applied cumulatively based on the current state of the image.

### Basic Adjustments

Use the sliders in the **Basic** section of the Adjustment Panel.

### Levels

Use the sliders in the **Levels** section. Adjust input black/white points to stretch contrast, gamma for midtones, and output black/white to limit the output range.

### Curves

*   Select the channel (RGB, R, G, B) you want to adjust.
*   Click on the curve line to add points.
*   Drag points to reshape the curve. Higher points brighten, lower points darken the corresponding tonal range.
*   Double-click a point to remove it.
*   Right-click for options like 'Reset Curve'.

### Color Balance

Use the sliders in the **Color Balance** section. The top three sliders (Red, Green, Blue Balance) multiply the respective channels, while the bottom three (Cyan-Red, etc.) add/subtract color casts.

### HSL (Hue/Saturation/Lightness)

*   Select a color range (e.g., 'Greens') from the dropdown.
*   Use the Hue, Saturation, and Lightness sliders to adjust only that specific color range in the image.

### Selective Color

*   Select a color range (e.g., 'Reds', 'Neutrals').
*   Adjust the Cyan, Magenta, Yellow, and Black sliders to modify the composition of the selected color.
    *   **Relative Mode (Default):** Adjustments are proportional to the amount of CMYK already present in the selected color.
    *   **Absolute Mode:** (May behave similarly to Relative in current implementation) Aims to add/subtract absolute amounts of CMYK.

### Channel Mixer

*   Select the **Output Channel** (Red, Green, or Blue) you want to modify.
*   Adjust the **Red, Green, Blue sliders** to control how much of each *input* channel contributes to the selected *output* channel.
*   Use the **Constant** slider to add a flat offset.
*   Check **Monochrome** to convert the image to grayscale based on the current mix settings.

### Shadows & Highlights

Use the sliders in the **Detail** section to recover detail in dark or bright areas.

### Vibrance & Clarity

*   **Vibrance:** Intelligently boosts saturation, primarily affecting less saturated colors while protecting skin tones.
*   **Clarity:** Adds local contrast to enhance texture and detail (can look like sharpening).

### Noise Reduction

Adjust the **Strength** slider in the **Detail** section. Higher values apply stronger denoising (using Non-Local Means). *Note: Requires `opencv-contrib-python` to be installed.*

### White Balance Picker

1.  Click the **WB Picker** button in the Adjustment Panel (eyedropper icon).
2.  The mouse cursor changes over the Image Viewer.
3.  Click on an area of the image that *should* be neutral gray or white.
4.  The application calculates and applies new Temperature and Tint values based on your selection.

### Automatic Adjustments

Click the buttons in the **Auto Adjustments** section:

*   **AWB:** Automatically attempts to correct the white balance (various methods available via dropdown).
*   **Auto Levels:** Automatically adjusts levels based on image content (modes available via dropdown).
*   **Auto Color:** Attempts automatic color correction (methods available via dropdown).
*   **Auto Tone:** Applies a combination of AWB, Auto Levels, and potentially Clarity for a general tonal improvement.

## 7. Applying Presets

Presets offer quick ways to apply complex looks.

### Film Simulation

*   Select the **Film Simulation** tab.
*   Choose a preset from the list.

### Photo Styles

*   Select the **Photo Style** tab.
*   Choose a preset from the list.

### Preview vs. Apply

*   **Single-clicking** a preset name shows a *temporary preview*. Making any other adjustment or clicking another preset replaces the preview.
*   **Double-clicking** or selecting and clicking **Apply** *permanently applies* the preset to the current image state. This is an undoable step.

### Intensity Control

*   After applying a preset, use the **Intensity** slider (available in both preset panels) to blend the preset effect with the image state *before* the preset was applied. 100% is the full preset effect.

## 8. Saving Images

### Saving a Single Image

*   Once you are satisfied with the adjustments and presets applied to the image in the viewer:
*   Go to `File > Save Positive As...` or use the shortcut `Ctrl+Shift+S`.
*   Choose a location, filename, and format (JPG, PNG, TIF).
*   Click **Save**. Quality settings are typically default (high for JPG, standard compression for PNG).

## 9. Batch Processing

This allows you to apply the *current* adjustment and preset settings to multiple images efficiently.

### Selecting Images

1.  Open a folder using `File > Open Folder for Batch...`.
2.  In the **Batch Filmstrip** at the bottom, check the boxes next to the thumbnails of the images you want to process.

### Setting Output Directory

1.  Click the `Set Output Dir` button on the **Batch Processing** toolbar.
2.  Choose the folder where the processed positive images will be saved. The selected path will appear next to the button.

### Choosing Format and Quality

1.  Use the `Format:` dropdown on the **Batch Processing** toolbar to select the desired output file type (.jpg, .png, .tif).
2.  If JPG or PNG is selected, use the `Quality:` spinner to set the desired quality/compression level.

### Starting the Batch

1.  Ensure you have:
    *   Checked the desired images in the filmstrip.
    *   Set a valid output directory.
    *   Configured the desired adjustments and presets in the main panels (these settings will be applied to all checked images).
2.  Click the `Process Batch` button on the **Batch Processing** toolbar.

### Monitoring Progress

*   The **Progress Bar** on the toolbar will fill as images are processed.
*   The **Status Bar** will show progress messages (e.g., "Batch processing: 5/20 images...").
*   A message box will appear upon completion, summarizing the results (successes and failures).

## 10. Troubleshooting

*   **Slow Performance:** Image processing can be demanding. Larger images take longer. If you don't have a compatible NVIDIA GPU or CuPy installed, processing will rely solely on the CPU.
*   **Incorrect Conversion:** The default conversion targets C-41 negatives. Other film types might require different initial settings or manual adjustments for optimal results. The automatic mask detection might struggle with unusual scans or non-standard film bases.
*   **Noise Reduction Not Working:** Ensure you have installed the `opencv-contrib-python` package.
*   **Presets Not Loading:**
    *   Default film presets should be in `config/presets/film/`.
    *   Default photo presets should be in `config/presets/photo/`.
    *   User-saved photo presets are typically stored in a user-specific data directory (location shown in terminal on startup if `appdirs` is installed) or in `photo_presets.json` in the project root as a fallback. Check these locations and file permissions.
*   **Errors During Processing:** Check the terminal window where you launched the application for detailed error messages, which can help diagnose the problem.