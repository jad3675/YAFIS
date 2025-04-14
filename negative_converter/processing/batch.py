# Batch processing handler
import os
import cv2
import concurrent.futures
import numpy as np # Needed for potential errors during processing

# Assuming converter and film_simulation modules are in the same package level
from .converter import NegativeConverter
from .film_simulation import FilmPresetManager, GPU_ENABLED as FILM_SIM_GPU_ENABLED # Import GPU status
from .photo_presets import PhotoPresetManager # Import PhotoPresetManager

def process_batch(file_paths, preset_id, output_dir, preset_manager, negative_converter):
    """Process multiple images in parallel using NegativeConverter and FilmPresetManager.

    Args:
        file_paths (list): List of paths to input image files.
        preset_id (str): The ID of the film simulation preset to apply.
        output_dir (str): Directory to save the processed images.
        preset_manager (FilmPresetManager): An instance of the preset manager.
        negative_converter (NegativeConverter): An instance of the negative converter.

    Returns:
        list: A list of tuples, where each tuple contains:
              (file_path, success_status (bool), error_message (str or None))
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            # Return failure for all files if output dir cannot be created
            return [(fp, False, f"Output directory creation failed: {e}") for fp in file_paths]

    results = []

    def process_single_file(file_path):
        """Worker function to process a single image file."""
        try:
            # Basic check if file exists
            if not os.path.isfile(file_path):
                return (file_path, False, "File not found")

            # Load image using OpenCV (loads as BGR by default)
            image_bgr = cv2.imread(file_path)
            if image_bgr is None or image_bgr.size == 0:
                return (file_path, False, "Failed to load image or image is empty")

            # Convert to RGB for processing
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # --- Core Processing Steps ---
            # 1. Convert negative to positive
            #    Instantiate converter inside worker or pass pre-initialized?
            #    Passing pre-initialized is likely better for efficiency if stateful.
            positive = negative_converter.convert(image_rgb)
            if positive is None or positive.size == 0:
                 return (file_path, False, "Negative conversion failed")

            # 2. Apply film simulation
            #    Instantiate manager inside worker or pass pre-initialized?
            #    Passing pre-initialized is better.
            result_rgb = preset_manager.apply_preset(positive, preset_id)
            if result_rgb is None or result_rgb.size == 0:
                 return (file_path, False, "Film simulation failed")
            # --- End Core Processing Steps ---

            # Prepare output path
            filename = os.path.basename(file_path)
            # Consider adding a suffix/prefix to avoid overwriting originals if output_dir is same as input
            output_filename = f"processed_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            # Convert back to BGR for saving with OpenCV
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            # Save result
            save_success = cv2.imwrite(output_path, result_bgr)
            if not save_success:
                return (file_path, False, f"Failed to save image to {output_path}")

            return (file_path, True, None) # Success

        except Exception as e:
            # Catch any other exceptions during processing
            return (file_path, False, f"Error processing file: {e}")

    # Determine executor and max workers based on GPU availability
    # Use ThreadPoolExecutor if GPU is enabled (assumes CuPy releases GIL)
    # Use ProcessPoolExecutor if CPU only
    if FILM_SIM_GPU_ENABLED: # Check imported GPU status
        # Use fewer workers for GPU to avoid VRAM exhaustion and context switching overhead
        max_workers = min(4, os.cpu_count() or 1) # Limit GPU threads (e.g., 4)
        Executor = concurrent.futures.ThreadPoolExecutor
        print(f"[Batch Info] GPU Enabled. Using ThreadPoolExecutor with max_workers={max_workers}")
    else:
        # Use ProcessPoolExecutor for CPU-bound tasks
        max_workers = os.cpu_count() or 1
        Executor = concurrent.futures.ProcessPoolExecutor
        print(f"[Batch Info] GPU Disabled. Using ProcessPoolExecutor with max_workers={max_workers}")

    with Executor(max_workers=max_workers) as executor:
        # Submit all files for processing
        future_to_file = {
            executor.submit(process_single_file, file_path): file_path
            for file_path in file_paths
        }

        # Collect results as they complete
        processed_count = 0
        total_files = len(file_paths)
        for future in concurrent.futures.as_completed(future_to_file):
            processed_count += 1
            try:
                result = future.result()
                results.append(result)
                print(f"({processed_count}/{total_files}) Processed: {result[0]} - Success: {result[1]}{f' - Error: {result[2]}' if not result[1] else ''}")
            except Exception as exc:
                # Handle exceptions raised by the worker function itself (should be caught inside ideally)
                file_path = future_to_file[future]
                print(f"({processed_count}/{total_files}) Exception for {file_path}: {exc}")
                results.append((file_path, False, f"Executor error: {exc}"))

    return sorted(results, key=lambda x: x[0]) # Sort results by original file path


# --- NEW: Batch Processing with Live Adjustments ---

# Import the adjustment function
from .adjustments import apply_all_adjustments, GPU_ENABLED as ADJUSTMENTS_GPU_ENABLED # Import GPU status
import traceback # Moved import here as it's only used in the worker

# --- Top-level worker function for batch processing with adjustments ---
# Updated signature
def _process_single_file_worker(file_path, output_dir, adjustments_dict, active_preset_info,
                                negative_converter, film_preset_manager, photo_preset_manager,
                                fmt, quality):
    """Worker function (top-level) to process a single image file with preset and adjustments."""
    try:
        if not os.path.isfile(file_path):
            return (file_path, False, "File not found")

        # Load image (BGR)
        image_bgr = cv2.imread(file_path)
        if image_bgr is None or image_bgr.size == 0:
            return (file_path, False, "Failed to load image or image is empty")

        # Convert to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # --- Core Processing Steps ---
        # 1. Convert negative to positive
        # Unpack the tuple: (image, mask_classification)
        positive, _ = negative_converter.convert(image_rgb)
        if positive is None or positive.size == 0:
             return (file_path, False, "Negative conversion failed")

        # 2. Apply Preset (if active)
        image_after_preset = positive # Start with the positive image
        if active_preset_info:
            preset_type = active_preset_info['type']
            preset_id = active_preset_info['id']
            intensity = active_preset_info['intensity']
            print(f"Worker applying {preset_type} preset '{preset_id}' (Intensity: {intensity:.2f}) to {os.path.basename(file_path)}")
            try:
                if preset_type == 'film':
                    grain_scale = active_preset_info.get('grain_scale', 1.0)
                    # Need to get modified params based on grain scale for film presets
                    # This requires access to the preset data itself, which the manager has.
                    # Let's assume the manager's apply_preset handles intensity/grain internally for simplicity here.
                    # If not, we'd need to replicate the logic from FilmPresetPanel.get_modified_preset_params
                    image_after_preset = film_preset_manager.apply_preset(positive, preset_id, intensity=intensity) # Assuming manager handles grain internally based on preset data + intensity
                elif preset_type == 'photo':
                    image_after_preset = photo_preset_manager.apply_photo_preset(positive, preset_id, intensity=intensity)

                if image_after_preset is None:
                    print(f"Warning: Preset application returned None for {os.path.basename(file_path)}. Skipping preset.")
                    image_after_preset = positive # Revert to pre-preset image
            except Exception as preset_e:
                print(f"Error applying preset {preset_id} to {os.path.basename(file_path)}: {preset_e}")
                # Decide whether to continue with adjustments or fail the file
                # For now, continue with adjustments on the pre-preset image
                image_after_preset = positive

        # 3. Apply live adjustments (to the result of the preset application, or the original positive if no preset)
        adjusted_rgb = apply_all_adjustments(image_after_preset, adjustments_dict)
        if adjusted_rgb is None or adjusted_rgb.size == 0:
             return (file_path, False, "Adjustment application failed")
        # --- End Core Processing Steps ---

        # Prepare output path
        base_filename, _ = os.path.splitext(os.path.basename(file_path))
        output_filename = f"adjusted_{base_filename}{fmt}"
        output_path = os.path.join(output_dir, output_filename)

        # Convert back to BGR for saving
        result_bgr = cv2.cvtColor(adjusted_rgb, cv2.COLOR_RGB2BGR)

        # Save result
        imwrite_params = []
        if quality:
            if fmt.lower() in ['.jpg', '.jpeg']:
                if 'jpeg_quality' in quality:
                    imwrite_params = [cv2.IMWRITE_JPEG_QUALITY, quality['jpeg_quality']]
            elif fmt.lower() == '.png':
                if 'png_compression' in quality:
                    imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION, quality['png_compression']]
            # Add more formats if needed

        save_success = cv2.imwrite(output_path, result_bgr, imwrite_params)
        if not save_success:
            return (file_path, False, f"Failed to save image to {output_path} with params {imwrite_params}")

        return (file_path, True, None) # Success

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return (file_path, False, f"Error processing file: {e}")


# Updated signature to accept preset info and managers
def process_batch_with_adjustments(file_paths, output_dir, adjustments_dict, active_preset_info,
                                   negative_converter, film_preset_manager, photo_preset_manager,
                                   output_format='.jpg', quality_settings=None):
    """Process multiple images in parallel using NegativeConverter, optional preset, and live adjustments.

    Args:
        file_paths (list): List of paths to input image files.
        output_dir (str): Directory to save the processed images.
        adjustments_dict (dict): Dictionary containing the adjustment settings to apply.
        active_preset_info (dict or None): Info about the active preset {'type': 'film'/'photo', 'id': ..., ...} or None.
        negative_converter (NegativeConverter): An instance of the negative converter.
        film_preset_manager (FilmPresetManager): Instance for applying film presets.
        photo_preset_manager (PhotoPresetManager): Instance for applying photo presets.
        output_format (str): The desired output file format extension (e.g., '.jpg', '.png').
        quality_settings (dict): Dictionary with format-specific quality parameters for cv2.imwrite.

    Returns:
        list: A list of tuples, where each tuple contains:
              (file_path, success_status (bool), error_message (str or None))
    """
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return [(fp, False, f"Output directory creation failed: {e}") for fp in file_paths]

    results = []

    # (Worker function is now defined above)

    # Determine executor and max workers based on GPU availability
    # Check GPU status from relevant modules (adjustments likely covers converter too)
    gpu_mode = ADJUSTMENTS_GPU_ENABLED # Assume adjustments reflects overall GPU capability needed
    if gpu_mode:
        max_workers = min(4, os.cpu_count() or 1) # Limit GPU threads
        Executor = concurrent.futures.ThreadPoolExecutor
        print(f"[Batch Info] GPU Enabled. Using ThreadPoolExecutor with max_workers={max_workers}")
    else:
        max_workers = os.cpu_count() or 1
        Executor = concurrent.futures.ProcessPoolExecutor
        print(f"[Batch Info] GPU Disabled. Using ProcessPoolExecutor with max_workers={max_workers}")


    with Executor(max_workers=max_workers) as executor:
        # Submit all files for processing using the new worker
        # Pass format and quality settings to the worker
        future_to_file = {
            # Submit tasks using the top-level worker function and pass necessary arguments
            # Submit tasks using the top-level worker function and pass ALL necessary arguments
            executor.submit(_process_single_file_worker,
                            file_path, output_dir, adjustments_dict, active_preset_info,
                            negative_converter, film_preset_manager, photo_preset_manager,
                            output_format, quality_settings): file_path
            for file_path in file_paths
        }

        # Collect results as they complete
        processed_count = 0
        total_files = len(file_paths)
        for future in concurrent.futures.as_completed(future_to_file):
            processed_count += 1
            try:
                result = future.result()
                results.append(result)
                print(f"({processed_count}/{total_files}) Processed (Adjustments): {result[0]} - Success: {result[1]}{f' - Error: {result[2]}' if not result[1] else ''}")
            except Exception as exc:
                file_path = future_to_file[future]
                print(f"({processed_count}/{total_files}) Exception for {file_path} in adjustment batch: {exc}")
                results.append((file_path, False, f"Executor error: {exc}"))

    return sorted(results, key=lambda x: x[0])