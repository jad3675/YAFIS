# Processing package initialization
from .photo_presets import PhotoPresetManager
from .film_simulation import FilmPresetManager
from .adjustments import ImageAdjustments, AdvancedAdjustments
from .converter import NegativeConverter, detect_orange_mask, apply_color_correction
from .mask_detection import FilmBaseDetector, MaskDetectionResult, classify_film_type
from .processing_strategy import ProcessingContext, ProcessingStrategy
from .dust_detection import (
    DustSpot, Scratch, DetectionParams, DetectionResult,
    detect_dust_spots, detect_scratches, detect_artifacts,
    create_artifact_mask, remove_artifacts, auto_clean_image
)
from .batch_queue import (
    BatchItemSettings, BatchItem, BatchItemStatus, BatchQueueStats, BatchQueue
)