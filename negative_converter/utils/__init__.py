# This file makes the 'utils' directory a Python package.

from .errors import (
    AppError,
    FileIOError,
    ProcessingError,
    GPUError,
    ConfigurationError,
    ErrorCategory,
    handle_errors,
    handle_gpu_errors,
    safe_operation,
    log_and_continue,
    format_user_error,
)

from .history import HistoryStack, ImageHistoryStack, HistoryEntry
from .keyboard_shortcuts import ShortcutManager, get_shortcut_manager, DEFAULT_SHORTCUTS
from .geometry import (
    CropRect, AspectRatio, ASPECT_RATIOS,
    crop_image, rotate_image, rotate_90, flip_image,
    straighten_image, detect_horizon_angle, auto_crop_borders
)
from .color_sampler import (
    ColorSample, ColorSamplerState,
    sample_color, sample_color_averaged,
    rgb_to_lab, analyze_skin_tone, check_neutral
)
from .preset_validator import (
    validate_preset_file, validate_all_presets, get_preset_info,
    ValidationError
)
from .export_presets import ExportPreset, ExportPresetManager, resize_image
from .session import SessionData, SessionManager, get_session_manager, ImageState
from .gpu_pipeline import GPUPipeline, PipelineStage, PipelineResult, process_for_export

__all__ = [
    # Errors
    'AppError',
    'FileIOError',
    'ProcessingError',
    'GPUError',
    'ConfigurationError',
    'ErrorCategory',
    'handle_errors',
    'handle_gpu_errors',
    'safe_operation',
    'log_and_continue',
    'format_user_error',
    # History
    'HistoryStack',
    'ImageHistoryStack',
    'HistoryEntry',
    # Shortcuts
    'ShortcutManager',
    'get_shortcut_manager',
    'DEFAULT_SHORTCUTS',
    # Geometry
    'CropRect',
    'AspectRatio',
    'ASPECT_RATIOS',
    'crop_image',
    'rotate_image',
    'rotate_90',
    'flip_image',
    'straighten_image',
    'detect_horizon_angle',
    'auto_crop_borders',
    # Color sampler
    'ColorSample',
    'ColorSamplerState',
    'sample_color',
    'sample_color_averaged',
    'rgb_to_lab',
    'analyze_skin_tone',
    'check_neutral',
    # Preset validation
    'validate_preset_file',
    'validate_all_presets',
    'get_preset_info',
    'ValidationError',
    # Export presets
    'ExportPreset',
    'ExportPresetManager',
    'resize_image',
    # Session
    'SessionData',
    'SessionManager',
    'get_session_manager',
    'ImageState',
    # GPU Pipeline
    'GPUPipeline',
    'PipelineStage',
    'PipelineResult',
    'process_for_export',
]