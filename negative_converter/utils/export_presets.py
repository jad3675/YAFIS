# Export presets for batch output settings
"""
Export presets for saving images with consistent settings.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import os

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExportPreset:
    """Export preset defining output format and quality settings."""
    name: str
    format: str = "jpeg"  # jpeg, png, tiff, webp
    quality: int = 95  # JPEG/WebP quality (1-100)
    png_compression: int = 6  # PNG compression (0-9)
    tiff_compression: str = "lzw"  # none, lzw, zip
    
    # Resize options
    resize_enabled: bool = False
    resize_mode: str = "fit"  # fit, fill, width, height, percentage
    resize_width: int = 0
    resize_height: int = 0
    resize_percentage: int = 100
    resize_filter: str = "lanczos"  # nearest, bilinear, bicubic, lanczos
    
    # Output naming
    suffix: str = "_export"
    preserve_folder_structure: bool = True
    
    # Metadata
    preserve_exif: bool = True
    preserve_icc: bool = True
    
    # Color space
    convert_to_srgb: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExportPreset':
        """Create from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    def get_extension(self) -> str:
        """Get file extension for this format."""
        extensions = {
            "jpeg": ".jpg",
            "png": ".png",
            "tiff": ".tif",
            "webp": ".webp",
        }
        return extensions.get(self.format, ".jpg")
    
    def get_save_params(self) -> Dict[str, Any]:
        """Get parameters for image_saver.save_image()."""
        params = {
            "preserve_exif": self.preserve_exif,
        }
        
        if self.format == "jpeg":
            params["quality"] = self.quality
        elif self.format == "png":
            params["png_compression"] = self.png_compression
        elif self.format == "tiff":
            params["compression"] = f"tiff_{self.tiff_compression}" if self.tiff_compression != "none" else None
        elif self.format == "webp":
            params["quality"] = self.quality
        
        return params


# Default export presets
DEFAULT_EXPORT_PRESETS = [
    ExportPreset(
        name="High Quality JPEG",
        format="jpeg",
        quality=95,
        suffix="_hq"
    ),
    ExportPreset(
        name="Web JPEG",
        format="jpeg",
        quality=80,
        resize_enabled=True,
        resize_mode="fit",
        resize_width=2048,
        resize_height=2048,
        suffix="_web"
    ),
    ExportPreset(
        name="Lossless PNG",
        format="png",
        png_compression=6,
        suffix="_lossless"
    ),
    ExportPreset(
        name="Archive TIFF",
        format="tiff",
        tiff_compression="lzw",
        suffix="_archive"
    ),
    ExportPreset(
        name="Social Media",
        format="jpeg",
        quality=85,
        resize_enabled=True,
        resize_mode="fit",
        resize_width=1080,
        resize_height=1080,
        suffix="_social"
    ),
]


class ExportPresetManager:
    """Manages export presets."""
    
    def __init__(self, presets_file: Optional[str] = None):
        """
        Initialize the export preset manager.
        
        Args:
            presets_file: Path to presets JSON file. If None, uses default location.
        """
        if presets_file is None:
            config_dir = os.path.dirname(os.path.dirname(__file__))
            presets_file = os.path.join(config_dir, "config", "export_presets.json")
        
        self._presets_file = presets_file
        self._presets: Dict[str, ExportPreset] = {}
        self._load_presets()
    
    def _load_presets(self) -> None:
        """Load presets from file, falling back to defaults."""
        # Start with defaults
        for preset in DEFAULT_EXPORT_PRESETS:
            self._presets[preset.name] = preset
        
        # Load custom presets
        if os.path.exists(self._presets_file):
            try:
                with open(self._presets_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for preset_data in data.get("presets", []):
                    try:
                        preset = ExportPreset.from_dict(preset_data)
                        self._presets[preset.name] = preset
                    except Exception as e:
                        logger.warning("Failed to load export preset: %s", e)
                
                logger.info("Loaded %d export presets", len(self._presets))
            except Exception as e:
                logger.warning("Failed to load export presets file: %s", e)
    
    def _save_presets(self) -> bool:
        """Save presets to file."""
        try:
            # Only save non-default presets
            default_names = {p.name for p in DEFAULT_EXPORT_PRESETS}
            custom_presets = [
                p.to_dict() for name, p in self._presets.items()
                if name not in default_names
            ]
            
            # Also save modified defaults
            for default in DEFAULT_EXPORT_PRESETS:
                if default.name in self._presets:
                    current = self._presets[default.name]
                    if current.to_dict() != default.to_dict():
                        custom_presets.append(current.to_dict())
            
            data = {"presets": custom_presets}
            
            os.makedirs(os.path.dirname(self._presets_file), exist_ok=True)
            with open(self._presets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            return True
        except Exception as e:
            logger.exception("Failed to save export presets")
            return False
    
    def get_preset(self, name: str) -> Optional[ExportPreset]:
        """Get a preset by name."""
        return self._presets.get(name)
    
    def get_all_presets(self) -> List[ExportPreset]:
        """Get all presets."""
        return list(self._presets.values())
    
    def get_preset_names(self) -> List[str]:
        """Get all preset names."""
        return list(self._presets.keys())
    
    def add_preset(self, preset: ExportPreset) -> bool:
        """Add or update a preset."""
        self._presets[preset.name] = preset
        return self._save_presets()
    
    def delete_preset(self, name: str) -> bool:
        """Delete a preset."""
        if name in self._presets:
            del self._presets[name]
            return self._save_presets()
        return False
    
    def rename_preset(self, old_name: str, new_name: str) -> bool:
        """Rename a preset."""
        if old_name in self._presets and new_name not in self._presets:
            preset = self._presets[old_name]
            preset.name = new_name
            del self._presets[old_name]
            self._presets[new_name] = preset
            return self._save_presets()
        return False


def resize_image(
    image: 'np.ndarray',
    preset: ExportPreset
) -> 'np.ndarray':
    """
    Resize an image according to export preset settings.
    
    Args:
        image: Input image array.
        preset: Export preset with resize settings.
        
    Returns:
        Resized image array.
    """
    import numpy as np
    import cv2
    
    if not preset.resize_enabled or image is None:
        return image
    
    h, w = image.shape[:2]
    
    # Calculate target dimensions
    if preset.resize_mode == "percentage":
        scale = preset.resize_percentage / 100.0
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif preset.resize_mode == "width":
        if preset.resize_width > 0:
            scale = preset.resize_width / w
            new_w = preset.resize_width
            new_h = int(h * scale)
        else:
            return image
    elif preset.resize_mode == "height":
        if preset.resize_height > 0:
            scale = preset.resize_height / h
            new_w = int(w * scale)
            new_h = preset.resize_height
        else:
            return image
    elif preset.resize_mode == "fit":
        if preset.resize_width > 0 and preset.resize_height > 0:
            scale = min(preset.resize_width / w, preset.resize_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return image
    elif preset.resize_mode == "fill":
        if preset.resize_width > 0 and preset.resize_height > 0:
            scale = max(preset.resize_width / w, preset.resize_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return image
    else:
        return image
    
    # Don't upscale unless explicitly requested
    if new_w >= w and new_h >= h:
        return image
    
    # Select interpolation method
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interpolation_map.get(preset.resize_filter, cv2.INTER_LANCZOS4)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    return resized
