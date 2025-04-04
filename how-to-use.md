# Negative Converter - Comprehensive User Guide

This guide provides detailed instructions on how to use the Negative Converter application to transform your scanned film negatives into beautiful positive images.

## Table of Contents

1. [Installation](#1-installation)
2. [Getting Started](#2-getting-started)
3. [Interface Overview](#3-interface-overview)
4. [Working with Images](#4-working-with-images)
5. [Basic Adjustments](#5-basic-adjustments)
6. [Advanced Adjustments](#6-advanced-adjustments)
7. [Automatic Adjustments](#7-automatic-adjustments)
8. [Film Simulation](#8-film-simulation)
9. [Photo Styles](#9-photo-styles)
10. [Batch Processing](#10-batch-processing)
11. [Tips and Best Practices](#11-tips-and-best-practices)
12. [Troubleshooting](#12-troubleshooting)

## 1. Installation

### System Requirements

* **Operating System**: Windows, macOS, or Linux
* **Python**: Version 3.8 or higher
* **RAM**: 4GB minimum, 8GB or more recommended for larger images
* **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)

### Installing Dependencies

1. Ensure Python 3.8+ is installed on your system
2. Install the required Python packages:

```bash
pip install PyQt6 numpy opencv-python opencv-contrib-python appdirs
```

3. For GPU acceleration (optional, NVIDIA GPU required):

```bash
# Replace XXX with your CUDA version (e.g., 12x for CUDA 12.x)
pip install cupy-cudaXXX
```

### Launching the Application

Navigate to the project's root directory and run:

```bash
python main.py
```

## 2. Getting Started

### Opening Your First Negative

1. Click `File > Open Negative...` or press `Ctrl+O`
2. Navigate to and select your scanned negative image file
3. The application will automatically perform the initial conversion
4. Once complete, the converted positive image will appear in the viewer

### Initial Conversion Process

During the initial conversion, the application:

1. Detects the type of film negative (C-41 color, B&W, etc.)
2. Inverts the colors
3. Removes the orange mask (for C-41 negatives)
4. Applies initial color correction
5. Adjusts contrast and balance

This process may take a few seconds, especially for large images or without GPU acceleration.

## 3. Interface Overview

The application interface consists of several key components:

### Main Window

* **Central Image Viewer**: Displays the current image with pan and zoom capabilities
* **Menu Bar**: Access to file operations, editing functions, and view options
* **Status Bar**: Shows current status, tips, and processing information

### Panels and Toolbars

* **Adjustment Panel**: Controls for modifying the image appearance (right side)
* **Film Simulation Panel**: Film stock presets (right side, tabbed with Adjustments)
* **Photo Style Panel**: Creative look presets (right side, tabbed with Adjustments)
* **Batch Filmstrip**: Thumbnails for batch processing (bottom, visible in batch mode)
* **View Controls Toolbar**: Zoom and fit controls (top)
* **Batch Processing Toolbar**: Batch operation controls (top, visible in batch mode)

### Panel Navigation

* Use tabs on the right side to switch between Adjustments, Film Simulation, and Photo Styles
* Panels can be rearranged, detached, or hidden via the `View` menu

## 4. Working with Images

### Navigating the Image

* **Pan**: Click and drag the image
* **Zoom In**: Mouse wheel up, `Ctrl++`, or the zoom in button
* **Zoom Out**: Mouse wheel down, `Ctrl+-`, or the zoom out button
* **Fit to Window**: Click the "Fit to Window" button
* **100% View**: Click the "Reset Zoom (1:1)" button

### Saving Your Work

1. Click `File > Save Positive As...` or press `Ctrl+Shift+S`
2. Choose a location and filename
3. Select a format (JPEG, PNG, TIFF)
4. For JPEG and PNG, you can specify quality/compression settings
5. Click "Save"

### Undo Functionality

* To undo a destructive operation (applying a preset or auto adjustment), click `Edit > Undo` or press `Ctrl+Z`
* Note: The application currently supports a single level of undo

## 5. Basic Adjustments

The Basic section in the Adjustment Panel provides fundamental controls:

### Brightness and Contrast

* **Brightness**: Adjusts the overall lightness of the image (-100 to +100)
* **Contrast**: Controls the difference between light and dark areas (-100 to +100)

### Color Controls

* **Saturation**: Adjusts color intensity (-100 to +100)
* **Hue**: Shifts all colors around the color wheel (-180 to +180 degrees)

### White Balance

* **Temperature**: Adjusts from cool (blue) to warm (yellow) (-100 to +100)
* **Tint**: Adjusts from green to magenta (-100 to +100)

## 6. Advanced Adjustments

Expand the collapsible sections to access advanced controls:

### Levels

* **Black Point**: Sets the darkest point in the image (0-255)
* **White Point**: Sets the brightest point in the image (0-255)
* **Gamma**: Adjusts midtones without affecting blacks or whites (0.1-5.0)
* **Output Black**: Limits how dark shadows can be (0-255)
* **Output White**: Limits how bright highlights can be (0-255)

### Curves

A powerful tool for precise tonal control:

1. Select a channel (RGB, Red, Green, or Blue)
2. Click on the curve line to add control points
3. Drag points to reshape the curve:
   * Moving a point up brightens that tonal range
   * Moving a point down darkens that tonal range
4. Double-click a point to remove it
5. Right-click for additional options (Reset Curve)

### Color Balance

* **Red/Green/Blue Balance**: Multiplicative adjustment of individual channels
* **Cyan-Red/Magenta-Green/Yellow-Blue**: Additive adjustments for shadows, midtones, or highlights

### HSL (Hue/Saturation/Lightness)

1. Select a color range from the dropdown (Reds, Yellows, Greens, etc.)
2. Adjust:
   * **Hue**: Shifts the selected color (-180 to +180 degrees)
   * **Saturation**: Adjusts intensity of the selected color (-100 to +100)
   * **Lightness**: Brightens or darkens the selected color (-100 to +100)

### Selective Color

1. Choose a color range (Reds, Yellows, Greens, Cyans, Blues, Magentas, Whites, Neutrals, Blacks)
2. Adjust CMYK components within that range:
   * **Cyan**: Shifts between cyan and red
   * **Magenta**: Shifts between magenta and green
   * **Yellow**: Shifts between yellow and blue
   * **Black**: Adjusts darkness
3. Select mode:
   * **Relative**: Adjustments proportional to existing color amounts
   * **Absolute**: Fixed adjustment amounts

### Channel Mixer

1. Select the output channel (Red, Green, Blue, or Monochrome)
2. Adjust the contribution from each input channel:
   * **Red Source**: How much the red channel contributes
   * **Green Source**: How much the green channel contributes
   * **Blue Source**: How much the blue channel contributes
   * **Constant**: Adds a flat offset
3. Enable **Monochrome** mode for black and white conversion with custom channel mixing

### Detail Enhancement

* **Shadows**: Recovers detail in dark areas (0-100)
* **Highlights**: Recovers detail in bright areas (0-100)
* **Vibrance**: Intelligently boosts less saturated colors while protecting skin tones (0-100)
* **Clarity**: Enhances local contrast for more definition (0-100)
* **Noise Reduction**: Reduces digital noise while preserving detail (0-100)

## 7. Automatic Adjustments

### White Balance Picker

1. Click the eyedropper icon in the Adjustment Panel
2. Click on an area in the image that should be neutral gray or white
3. The application will automatically adjust temperature and tint

### Auto White Balance (AWB)

1. Click the "AWB" button
2. Choose a method from the dropdown:
   * **Gray World**: Assumes the average color should be neutral gray
   * **White Patch**: Assumes the brightest area should be white
   * **Learning-Based**: Uses advanced algorithms to detect white balance

### Auto Levels

1. Click the "Auto Levels" button
2. Choose a mode from the dropdown:
   * **RGB**: Adjusts each channel independently
   * **Luminance**: Preserves color relationships while adjusting brightness

### Auto Color

1. Click the "Auto Color" button
2. Choose a method from the dropdown:
   * **Gamma**: Uses gamma correction to normalize colors
   * **Gray World**: Balances colors assuming gray world average

### Auto Tone

Click the "Auto Tone" button to apply a combination of Auto White Balance, Auto Levels, and Clarity for a general improvement.

## 8. Film Simulation

The Film Simulation panel allows you to apply looks based on classic film stocks:

### Using Film Presets

1. Select the "Film Simulation" tab on the right
2. Click on a preset name to preview it (e.g., "Kodachrome", "Portra 400")
3. Adjust the **Intensity** slider to control the strength of the effect (0-100%)
4. Adjust the **Grain Scale** slider to control the amount of film grain (0-5x)
5. Click "Apply" or double-click the preset name to permanently apply it

### Understanding Film Simulation

Each film preset emulates characteristics of specific film stocks:

* **Color Palette**: The distinctive color rendering of each film
* **Tone Curve**: The contrast and dynamic range behavior
* **Grain Pattern**: The size and distribution of film grain

### Creating Custom Film Presets

Currently, film presets are defined in JSON files in the `config/presets/film/` directory. While the UI doesn't provide tools to create custom film presets, you can:

1. Copy an existing preset JSON file
2. Modify parameters to create your own look
3. Save it with a new name in the film presets directory

## 9. Photo Styles

The Photo Style panel offers creative looks similar to those found in photo editing apps:

### Using Photo Styles

1. Select the "Photo Styles" tab on the right
2. Click on a style name to preview it (e.g., "Vivid", "Black & White", "Vintage")
3. Adjust the **Intensity** slider to control the strength of the effect (0-100%)
4. Click "Apply" or double-click the style name to permanently apply it

### Saving Custom Photo Styles

1. Adjust your image using the controls in the Adjustment Panel
2. In the Photo Styles panel, click the "Save Current Look..." button
3. Enter a name for your style
4. Click "OK" to save
5. Your custom style will appear at the bottom of the list

## 10. Batch Processing

Process multiple images with the same settings:

### Setting Up Batch Processing

1. Click `File > Open Folder for Batch...`
2. Select a folder containing negative images
3. The Batch Filmstrip will appear at the bottom showing thumbnails
4. The Batch Processing toolbar will appear at the top

### Selecting Images

1. In the Batch Filmstrip, check the boxes on images you want to process
2. Click on a thumbnail (single-click) to preview it without applying adjustments
3. Double-click a thumbnail to load it for detailed adjustment

### Configuring Output

1. Click "Set Output Dir" to choose a destination folder
2. Select an output format (.jpg, .png, .tif) from the dropdown
3. Set the quality level if applicable:
   * JPEG: 1-100 (higher is better quality)
   * PNG: 0-9 (higher is more compression)

### Processing the Batch

1. Set up all adjustments, film simulation, or photo style as desired
2. Ensure at least one image is checked and an output directory is set
3. Click "Process Batch"
4. A progress bar will show the status
5. When complete, a summary dialog will appear

### Understanding Batch Processing

* All selected images will be processed with the **same settings**
* The original negative files are never modified
* Output filenames maintain the original name with "_positive" added
* The application can be used normally during batch processing, but some controls will be disabled

## 11. Tips and Best Practices

### For Best Results

* **Scan Quality**: Start with the highest quality scans possible
* **Film Base Detection**: Ensure your scans include some of the film border for better mask detection
* **Adjustment Order**: Make global adjustments (WB, exposure) before local adjustments (selective color)
* **Preview vs. Apply**: Use the preview feature to experiment with presets before applying them
* **Presets as Starting Points**: Apply a preset first, then fine-tune with manual adjustments

### Performance Tips

* **GPU Acceleration**: Install CuPy for faster processing if you have an NVIDIA GPU
* **Image Size**: Extremely large images (>20MP) may process slower, especially without GPU
* **Batch at Night**: Set up large batch jobs to run when you don't need to use the computer

### Workflow Recommendations

1. **Rough Cut**: First go through your scans and select the keepers
2. **Basic Conversion**: Process these images with default or minimal adjustment
3. **Group Similar Images**: Process similar images (same film, lighting) with the same settings
4. **Final Adjustments**: Fine-tune individual images that need special attention

## 12. Troubleshooting

### Common Issues and Solutions

#### Slow Performance

* **Cause**: Large images or CPU-only processing
* **Solution**: 
  * Install CuPy for GPU acceleration
  * Reduce image resolution for preview purposes
  * Close other applications while processing

#### Incorrect Colors After Conversion

* **Cause**: Film type misdetection or unusual color cast
* **Solution**:
  * Use the White Balance Picker on a neutral area
  * Try Auto White Balance with different methods
  * Manually adjust Temperature and Tint
  * Use the Color Balance controls for fine-tuning

#### Film Base Not Detected Correctly

* **Cause**: Scan doesn't include enough film border or has untypical characteristics
* **Solution**:
  * Ensure scans include some film border/rebate area
  * Try Auto White Balance
  * Manually adjust colors using Curves and Color Balance

#### Batch Processing Errors

* **Cause**: File permission issues or incompatible image formats
* **Solution**:
  * Ensure you have write permissions to the output directory
  * Check that all images are in supported formats
  * Process fewer images at once

#### Missing Presets

* **Cause**: Preset files not found in expected locations
* **Solution**:
  * Check that the `config/presets/` directories contain the preset JSON files
  * Restart the application after adding new preset files

### Error Messages

| Message | Possible Cause | Solution |
|---------|----------------|----------|
| "Failed to load image" | Corrupted or unsupported file | Try converting the image to JPEG or PNG first |
| "Processing function returned None" | Processing error, possibly due to invalid input | Try reopening the image or restarting the application |
| "Conversion failed" | Initial negative-to-positive conversion error | Check if the image is a valid negative scan |
| "White balance scale calculation failed" | No valid neutral reference area found | Try picking a different area or use Auto White Balance |

### Getting Help

If you encounter persistent issues:

1. Check the terminal output for detailed error messages
2. Restart the application
3. Try processing a different image to see if the issue is file-specific
4. Search for similar issues in the project repository

---

This guide covers the main features and usage of the Negative Converter application. As the software evolves, new features and adjustments may be added. Refer to the project repository for the latest updates and information.
