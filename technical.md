# Negative Converter - Technical Details

This document provides a technical overview of the Negative Converter application's architecture, core algorithms, and implementation details.

## Table of Contents

1.  [Architecture Overview](#architecture-overview)
2.  [Core Conversion Pipeline (`processing/converter.py`)](#core-conversion-pipeline-processingconverterpy)
    *   [GPU Acceleration (CuPy)](#gpu-acceleration-cupy)
    *   [Step 0: Float Conversion](#step-0-float-conversion)
    *   [Step 1: Inversion](#step-1-inversion)
    *   [Step 2: Mask Detection & Neutralization](#step-2-mask-detection--neutralization)
    *   [Step 3: Channel Correction Matrix](#step-3-channel-correction-matrix)
    *   [Step 4: Channel-Specific Curves](#step-4-channel-specific-curves)
    *   [Step 5: Final Color Grading (LAB/HSV)](#step-5-final-color-grading-labhsv)
    *   [Step 6: Final Conversion to uint8](#step-6-final-conversion-to-uint8)
3.  [Adjustment System (`processing/adjustments.py`)](#adjustment-system-processingadjustmentspy)
    *   [`ImageAdjustments` Class](#imageadjustments-class)
    *   [`AdvancedAdjustments` Class](#advancedadjustments-class)
    *   [`apply_all_adjustments` Function](#apply_all_adjustments-function)
    *   [GPU Acceleration](#gpu-acceleration)
    *   [Utility Functions (`utils/imaging.py`)](#utility-functions-utilsimagingpy)
4.  [Preset Systems](#preset-systems)
    *   [Film Simulation (`processing/film_simulation.py`)](#film-simulation-processingfilm_simulationpy)
    *   [Photo Styles (`processing/photo_presets.py`)](#photo-styles-processingphoto_presetspy)
    *   [Preset Application Flow](#preset-application-flow)
    *   [Intensity Blending](#intensity-blending)
5.  [User Interface (`ui/`)](#user-interface-ui)
    *   [Framework: PyQt6](#framework-pyqt6)
    *   [Main Window (`main_window.py`)](#main-window-main_windowpy)
    *   [Threading Model](#threading-model)
    *   [UI Components](#ui-components)
6.  [Input/Output (`io/`)](#inputoutput-io)
7.  [Configuration and Presets (`config/`)](#configuration-and-presets-config)

---

## 1. Architecture Overview

The application follows a modular structure separating concerns into distinct packages:

*   **`main.py`:** Entry point, initializes the PyQt6 application and the main window.
*   **`config/`:** Stores configuration files, primarily JSON definitions for film and photo presets.
*   **`io/`:** Handles image loading (`image_loader.py`) and saving (`image_saver.py`) using OpenCV.
*   **`processing/`:** Contains the core image processing logic:
    *   `converter.py`: Negative-to-positive conversion algorithm.
    *   `adjustments.py`: Functions for applying various image adjustments.
    *   `film_simulation.py`: Logic for applying film simulation presets.
    *   `photo_presets.py`: Logic for applying photo style presets.
    *   `batch.py`: Implements the batch processing workflow.
*   **`ui/`:** Manages the graphical user interface using PyQt6:
    *   `main_window.py`: Orchestrates the main application window, panels, menus, and connects UI events to processing logic.
    *   Other files define specific widgets (image viewer, adjustment panels, preset lists, curves editor, filmstrip).
*   **`utils/`:** Contains shared utility functions, notably `imaging.py` for core operations like curve application.

## 2. Core Conversion Pipeline (`processing/converter.py`)

The `NegativeConverter` class encapsulates the primary negative-to-positive conversion process. It aims to replicate aspects of traditional film processing digitally.

### GPU Acceleration (CuPy)

*   The module attempts to import `cupy`. If successful and a CUDA-enabled GPU is detected (`cp.cuda.runtime.getDeviceCount()` succeeds), `GPU_ENABLED` is set to `True`.
*   Where feasible, array operations (inversion, matrix multiplication, scaling) are performed using CuPy (`xp = cp`) for potential speedup.
*   Error handling (e.g., `CUDARuntimeError`) is implemented to fall back to NumPy (`xp = np`) if GPU operations fail.
*   Data transfer between CPU (NumPy) and GPU (CuPy) occurs via `xp.asarray()` and `xp.asnumpy()`. OpenCV functions (`cv2.cvtColor`, histogramming) still require NumPy arrays, necessitating temporary transfers back to the CPU if data is on the GPU.

### Step 0: Float Conversion

*   The input `uint8` image is converted to `float32` using `xp.asarray()`. This happens once at the beginning. If GPU is enabled, this also transfers the image data to the GPU.

### Step 1: Inversion

*   A simple pixel-wise inversion is performed on the float32 image: `inverted_float = 255.0 - img_float`.

### Step 2: Mask Detection & Neutralization

*   **Mask Detection (`detect_orange_mask`):** Samples corner regions of the *original uint8* negative image to estimate the average color of the film base (orange mask for C-41).
*   **Mask Classification:** The detected mask color (RGB) is converted to HSV. Based on Hue, Saturation, and Value thresholds, the base is classified as "Likely C-41", "Clear/Near Clear", or "Unknown/Other".
*   **Neutralization:**
    *   **C-41:** The detected mask color is inverted (`255.0 - mask_color`). Scaling factors are calculated to shift this inverted mask color towards a target neutral gray (currently fixed at 128.0). Factors are clamped (`CLAMP_MIN`, `CLAMP_MAX`) to prevent extreme shifts. These factors are applied multiplicatively to the *inverted float* image (`neutralized_float = inverted_float * scale_factors`).
    *   **Clear/Near Clear:** No neutralization is applied (`neutralized_float = inverted_float`).
    *   **Unknown/Other:** A basic "Gray World" algorithm is applied. The average R, G, B values of the *inverted float* image are calculated. Scaling factors are derived to make these averages equal. Factors are clamped and applied multiplicatively (`neutralized_float = inverted_float * scale_factors`).

### Step 3: Channel Correction Matrix

*   A 3x3 color correction matrix (hardcoded, potentially should be profile-dependent) is applied to the `neutralized_float` image using matrix multiplication (`xp.dot(flat_image_float, correction_matrix.T)`). This step aims to correct color casts inherent in the film/scanning process. The operation is performed in float32 space.

### Step 4: Channel-Specific Curves

*   This step applies automatic contrast adjustments independently to each R, G, B channel using curves.
*   A histogram is calculated for each channel of a temporary `uint8` version of the image (clipped from the current float state).
*   Black and white points are determined by clipping a small percentage (`clip_percent`) from the histogram's cumulative distribution function (CDF).
*   A gamma adjustment is applied (different defaults for R and B channels).
*   Curve points representing this black/white point stretch and gamma adjustment are generated.
*   The `utils.imaging.apply_curve` function is called for each channel on the *float32* image data (`curves_result_float`) using the generated curve points.

### Step 5: Final Color Grading (LAB/HSV)

*   The result from the curves step (`curves_result_float`) is converted to `uint8` temporarily for color space conversions using OpenCV (which requires `uint8`).
*   **LAB Adjustments:** The image is converted to LAB color space. The average 'a' (green-magenta) and 'b' (blue-yellow) values are calculated. Minor adjustments are applied to nudge the average 'a' and 'b' values closer to the neutral point (128), aiming to reduce residual color casts.
*   **HSV Adjustments:** The LAB-adjusted image is converted back to RGB, then to HSV. Saturation (S channel) is boosted slightly (`*= 1.15`).
*   The image is converted back to RGB `uint8`.

### Step 6: Final Conversion to uint8

*   The result of the HSV adjustment (`final_graded_np`) is already `uint8` RGB and is returned as the final converted positive image.

## 3. Adjustment System (`processing/adjustments.py`)

This module provides functions for applying various user-controlled adjustments *after* the initial negative conversion.

### `ImageAdjustments` Class

Contains static methods for basic adjustments:
*   Brightness, Contrast, Saturation, Hue, Temp/Tint, Levels.
*   These generally operate on `uint8` RGB images, though some use internal float conversions for calculations (e.g., Brightness, Contrast, Temp/Tint attempt GPU acceleration). Saturation/Hue use OpenCV's HSV conversion. Levels uses `cv2.LUT`.

### `AdvancedAdjustments` Class

Contains static methods for more complex adjustments:
*   `apply_curves`: Applies separate R, G, B, or combined RGB curves using `utils.imaging.apply_curve`. Handles precedence (channel-specific overrides RGB).
*   `adjust_shadows_highlights`: Uses LAB color space to selectively adjust L channel based on masks derived from lightness.
*   `adjust_color_balance_additive`: Simple additive color shifts.
*   `apply_color_balance`: Combines additive shifts and multiplicative balance factors.
*   `adjust_channel_mixer`: Recomputes output channels based on weighted input channels.
*   `apply_noise_reduction`: Uses `cv2.fastNlMeansDenoisingColored` (requires `opencv-contrib-python`).
*   `adjust_hsl_by_range`: Modifies H, L, S based on masks derived from input Hue ranges.
*   `adjust_selective_color`: Simulates CMYK adjustments within specific color ranges (defined by Hue or lightness/saturation for grays).
*   `adjust_vibrance`: Modifies saturation weighted by inverse saturation.
*   `adjust_clarity`: Uses unsharp masking on the L channel of LAB space.
*   `apply_vignette`: Creates radial masks to darken/lighten edges.
*   `apply_bw_mix`: Converts to grayscale based on weighted R, G, B channels.
*   `apply_color_grading`: Applies color shifts to shadows, midtones, highlights (implementation likely simplified).
*   `apply_film_grain`: Adds synthetic noise (GPU accelerated).
*   `apply_auto_white_balance`: Implements various AWB algorithms (Gray World, White Patch, Learning-Based via `cv2.xphoto`).
*   `apply_auto_levels`: Automatic levels adjustment based on histogram analysis (Luminance or RGB modes).
*   `apply_auto_color`: Automatic color correction (Gamma or Gray World based).
*   `apply_auto_tone`: Combines several auto adjustments (NR, AWB, Levels, Clarity).

### `apply_all_adjustments` Function

*   This is the main entry point called by the UI to apply all user-selected adjustments.
*   It takes the base (converted positive) image and a dictionary (`adjustments_dict`) containing all adjustment parameters.
*   It applies the adjustments sequentially in a predefined, logical order (approximating a standard photo editing workflow) using the static methods from `ImageAdjustments` and `AdvancedAdjustments`.
*   It includes checks after each step to ensure the image is still valid.

### GPU Acceleration

*   Similar to `converter.py`, this module checks for CuPy and enables `GPU_ENABLED`.
*   Functions like Brightness, Contrast, Temp/Tint, Color Balance (Additive/Multiplicative), Channel Mixer, and Film Grain attempt to use CuPy for calculations if available, with NumPy fallbacks.
*   Operations relying heavily on OpenCV color space conversions (Saturation, Hue, Shadows/Highlights, HSL, Vibrance, Clarity, Vignette, Auto adjustments) primarily run on the CPU via OpenCV, even if `GPU_ENABLED` is true, due to the need for `cv2.cvtColor` etc.

### Utility Functions (`utils/imaging.py`)

*   `apply_curve`: A crucial utility function used by `converter.py`, `adjustments.py`, and `film_simulation.py`. It takes an image channel (NumPy or CuPy, uint8 or float32) and a list of curve points `[[x1, y1], [x2, y2], ...]`. It interpolates these points to create a lookup table (LUT) and applies it efficiently. It handles different input types and returns an array of the same type/backend.

## 4. Preset Systems

Separate managers handle Film Simulation and Photo Style presets.

### Film Simulation (`processing/film_simulation.py`)

*   **`FilmPresetManager`:** Loads JSON files from `config/presets/film/`.
*   **Preset Structure:** Each JSON defines parameters like `colorMatrix`, `toneCurves` (R, G, B, RGB), `colorBalance`, `dynamicRange`, and `grainParams`.
*   **Application (`_apply_full_preset`):**
    1.  Converts input `uint8` sRGB image to *linear float32* RGB (GPU/CPU).
    2.  Applies `colorMatrix` (linear space).
    3.  Applies `toneCurves` using `utils.imaging.apply_curve` (linear space).
    4.  Converts back to *sRGB float32* RGB (GPU/CPU).
    5.  Applies `colorBalance` (sRGB space).
    6.  Applies `dynamicRange` compression (sRGB space, uses LAB internally).
    7.  Applies `grainParams` using `AdvancedAdjustments.apply_film_grain` (sRGB space).
    8.  Clips result and converts final image to `uint8` sRGB NumPy array.
*   **GPU Usage:** Leverages GPU for linear/sRGB conversions, matrix math, curve application, color balance, and grain where possible. Dynamic range involves CPU steps for LAB conversion.

### Photo Styles (`processing/photo_presets.py`)

*   **`PhotoPresetManager`:** Loads default presets from `config/presets/photo/` and user presets from a user data directory (via `appdirs`) or a fallback `photo_presets.json` file. Saves user presets back to the user file.
*   **Preset Structure:** JSON files define parameters corresponding directly to the controls available in the `AdjustmentPanel` (e.g., `brightness`, `contrast`, `saturation`, `temperature`, `tint`, `shadows`, `highlights`, `vibrance`, `clarity`, `colorBalance`, `colorGrading`, `toneCurve` (RGB only), `grainParams`, `vignette`, `bwMix`).
*   **Application (`_apply_full_photo_preset`):**
    1.  Takes `uint8` sRGB image as input.
    2.  Applies adjustments sequentially using methods from `ImageAdjustments` and `AdvancedAdjustments` in a predefined order (Tonal -> Color -> Advanced Color -> Curve -> Effects -> BW).
    3.  Operates primarily in `uint8` sRGB space, except for grain application which involves a temporary float conversion.
    4.  Returns `uint8` sRGB NumPy array.
*   **GPU Usage:** Relies on the GPU acceleration implemented within the individual `ImageAdjustments` and `AdvancedAdjustments` methods called.

### Preset Application Flow

*   The UI calls `FilmPresetManager.apply_preset` or `PhotoPresetManager.apply_photo_preset`.
*   These methods retrieve the preset data and call their respective internal `_apply_full_preset` or `_apply_full_photo_preset` methods.
*   The internal methods execute the sequence of adjustments defined by the preset parameters.

### Intensity Blending

*   Both preset managers implement an `intensity` parameter (0.0 to 1.0).
*   If intensity is less than 1.0, the final result of the preset application is alpha-blended (`cv2.addWeighted`) with the original image passed to the `apply_preset` function.

## 5. User Interface (`ui/`)

### Framework: PyQt6

The GUI is built using the PyQt6 framework, providing widgets, layout management, and event handling.

### Main Window (`main_window.py`)

*   `MainWindow` class (inherits `QMainWindow`) is the central orchestrator.
*   Sets up the main layout with a central `ImageViewer` and `QDockWidget`s for the adjustment and preset panels. Docks are tabified.
*   Creates menus (`QMenu`), toolbars (`QToolBar`), and actions (`QAction`).
*   Connects signals from UI elements (sliders, buttons, list selections) to slots (methods) within `MainWindow`.
*   Manages the application state (e.g., enabling/disabling actions based on whether an image is loaded or batch mode is active).
*   Holds instances of the processing engines (`NegativeConverter`, `FilmPresetManager`, `PhotoPresetManager`, `ImageAdjustments`, `AdvancedAdjustments`).

### Threading Model

To prevent the UI from freezing during potentially long-running image processing tasks, `QThread` and worker `QObject`s are used:

*   **`InitialConversionWorker`:** Runs the `NegativeConverter.convert` method in a separate thread when an image is first loaded. Signals `finished` or `error`.
*   **`ImageProcessingWorker`:** Runs the `MainWindow._get_fully_adjusted_image` method (which calls `apply_all_adjustments` and potentially `apply_preset`) in a separate thread whenever an adjustment or preset preview/apply is triggered. Signals `finished` or `error`.
*   **`BatchProcessingWorker`:** Runs the `processing.batch.process_batch_with_adjustments` function in a separate thread when batch processing is started. Signals `progress`, `finished`, or `error`.

Signals and slots (`pyqtSignal`, `pyqtSlot`) are used for communication between the main UI thread and the worker threads. `QMetaObject.invokeMethod` with `Qt.ConnectionType.QueuedConnection` is used to trigger worker methods safely from the main thread.

### UI Components

*   **`ImageViewer`:** Custom widget (likely based on `QLabel` or `QGraphicsView`) for displaying the image, handling panning and zooming. Emits `color_sampled` signal for WB picker.
*   **`AdjustmentPanel`:** Contains various input widgets (sliders, spin boxes, combo boxes, curve widget) grouped into sections. Emits `adjustment_changed` signal when any value changes, and specific signals for auto adjustments or WB picker requests.
*   **`FilmPresetPanel` / `PhotoPresetPanel`:** Display lists of presets. Emit `preview_requested` and `apply_requested` signals. Include intensity/grain sliders.
*   **`CurvesWidget`:** Custom widget for interactive curve editing.
*   **`BatchFilmstripWidget`:** Displays image thumbnails with checkboxes for batch selection. Emits `preview_requested` (single click) and `checked_items_changed`.

## 6. Input/Output (`io/`)

*   **`image_loader.py`:** Uses `cv2.imread` to load images. Handles basic file checking. Includes functions to find compatible image files within a directory for batch mode.
*   **`image_saver.py`:** Uses `cv2.imwrite` to save images. Handles different formats (JPG, PNG, TIF) and associated quality/compression parameters. Ensures output directory exists.

## 7. Configuration and Presets (`config/`)

*   **`config/presets/film/`:** Contains default JSON files defining film simulation presets.
*   **`config/presets/photo/`:** Contains default JSON files defining photo style presets.
*   **User Photo Presets:** Stored in a platform-specific user data directory (identified by `appdirs`) or `photo_presets.json` in the project root. This allows users to save custom photo styles without modifying the core application files.