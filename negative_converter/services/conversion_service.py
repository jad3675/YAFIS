from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..io import image_loader, image_saver
from ..processing.adjustments import apply_all_adjustments
from ..utils.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class PresetInfo:
    """Normalized preset info passed from UI."""
    type: str  # "film" | "photo"
    id: str
    intensity: float
    grain_scale: Optional[float] = None


class ConversionService:
    """Thin facade over IO + processing. Keeps UI code smaller and easier to test."""

    def __init__(self, converter, film_preset_manager, photo_preset_manager):
        self._converter = converter
        self._film_preset_manager = film_preset_manager
        self._photo_preset_manager = photo_preset_manager

    def load_image(self, file_path: str):
        return image_loader.load_image(file_path)

    def convert_negative(
        self,
        raw_image: np.ndarray,
        *,
        override_mask_classification: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Delegates to NegativeConverter.convert(...).

        Return shape is preserved (image, mask_classification).
        """
        return self._converter.convert(
            raw_image,
            progress_callback=progress_callback,
            override_mask_classification=override_mask_classification,
        )

    def apply_adjustments(self, base_image: np.ndarray, adjustments: dict) -> np.ndarray:
        return apply_all_adjustments(base_image, adjustments)

    def apply_preset(self, image: np.ndarray, preset: PresetInfo) -> np.ndarray:
        if preset.type == "film":
            preset_data = self._film_preset_manager.get_preset(preset.id)
            if not preset_data:
                raise ValueError(f"Film preset '{preset.id}' not found.")
            return self._film_preset_manager.apply_preset(
                image=image,
                preset=preset_data,
                intensity=preset.intensity,
                grain_scale=preset.grain_scale,
            )

        if preset.type == "photo":
            return self._photo_preset_manager.apply_photo_preset(
                image,
                preset.id,
                preset.intensity,
            )

        raise ValueError(f"Unknown preset type '{preset.type}'.")

    def save_image(self, image: np.ndarray, file_path: str, **kwargs: Any) -> bool:
        return image_saver.save_image(image, file_path, **kwargs)