# Preset validation utilities
"""
JSON schema validation for preset files.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import os

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationError:
    """A validation error with location and message."""
    path: str
    field: str
    message: str
    severity: str = "error"  # "error" or "warning"
    
    def __str__(self) -> str:
        return f"{self.severity.upper()}: {self.path} - {self.field}: {self.message}"


# Schema definitions for different preset types
FILM_PRESET_SCHEMA = {
    "required": ["name", "id"],
    "optional": ["description", "category", "adjustments", "color_matrix", "curves", "intensity"],
    "types": {
        "name": str,
        "id": str,
        "description": str,
        "category": str,
        "intensity": (int, float),
    },
    "nested": {
        "adjustments": {
            "optional": [
                "brightness", "contrast", "saturation", "hue",
                "temp", "tint", "shadows", "highlights",
                "clarity", "vibrance", "grain", "vignette"
            ],
            "types": {
                "brightness": (int, float),
                "contrast": (int, float),
                "saturation": (int, float),
                "hue": (int, float),
                "temp": (int, float),
                "tint": (int, float),
                "shadows": (int, float),
                "highlights": (int, float),
                "clarity": (int, float),
                "vibrance": (int, float),
                "grain": (int, float),
                "vignette": (int, float),
            },
            "ranges": {
                "brightness": (-100, 100),
                "contrast": (-100, 100),
                "saturation": (-100, 100),
                "hue": (-180, 180),
                "temp": (-100, 100),
                "tint": (-100, 100),
                "shadows": (-100, 100),
                "highlights": (-100, 100),
                "clarity": (-100, 100),
                "vibrance": (-100, 100),
                "grain": (0, 100),
                "vignette": (-100, 100),
            }
        }
    }
}

PHOTO_PRESET_SCHEMA = {
    "required": ["name", "id"],
    "optional": ["description", "category", "adjustments", "curves", "hsl", "selective_color"],
    "types": {
        "name": str,
        "id": str,
        "description": str,
        "category": str,
    },
    "nested": {
        "adjustments": FILM_PRESET_SCHEMA["nested"]["adjustments"]
    }
}

ADJUSTMENT_PRESET_SCHEMA = {
    "required": ["name"],
    "optional": [
        "brightness", "contrast", "saturation", "hue", "temp", "tint",
        "levels_in_black", "levels_in_white", "levels_gamma",
        "levels_out_black", "levels_out_white",
        "curves_rgb", "curves_red", "curves_green", "curves_blue",
        "noise_reduction_strength"
    ],
    "types": {
        "name": str,
        "brightness": (int, float),
        "contrast": (int, float),
        "saturation": (int, float),
        "hue": (int, float),
        "temp": (int, float),
        "tint": (int, float),
        "levels_in_black": int,
        "levels_in_white": int,
        "levels_gamma": (int, float),
        "levels_out_black": int,
        "levels_out_white": int,
        "noise_reduction_strength": (int, float),
    },
    "ranges": {
        "brightness": (-100, 100),
        "contrast": (-100, 100),
        "saturation": (-100, 100),
        "hue": (-180, 180),
        "temp": (-100, 100),
        "tint": (-100, 100),
        "levels_in_black": (0, 254),
        "levels_in_white": (1, 255),
        "levels_gamma": (0.1, 10.0),
        "levels_out_black": (0, 254),
        "levels_out_white": (1, 255),
        "noise_reduction_strength": (0, 100),
    }
}


def validate_type(value: Any, expected_type: Any, field: str) -> Optional[str]:
    """
    Validate that a value matches the expected type.
    
    Args:
        value: The value to check.
        expected_type: Expected type or tuple of types.
        field: Field name for error message.
        
    Returns:
        Error message or None if valid.
    """
    if isinstance(expected_type, tuple):
        if not isinstance(value, expected_type):
            type_names = " or ".join(t.__name__ for t in expected_type)
            return f"Expected {type_names}, got {type(value).__name__}"
    else:
        if not isinstance(value, expected_type):
            return f"Expected {expected_type.__name__}, got {type(value).__name__}"
    return None


def validate_range(value: Any, min_val: float, max_val: float, field: str) -> Optional[str]:
    """
    Validate that a numeric value is within range.
    
    Args:
        value: The value to check.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        field: Field name for error message.
        
    Returns:
        Error message or None if valid.
    """
    if isinstance(value, (int, float)):
        if value < min_val or value > max_val:
            return f"Value {value} out of range [{min_val}, {max_val}]"
    return None


def validate_curves(curves: Any, field: str) -> List[str]:
    """
    Validate curve control points.
    
    Args:
        curves: Curve data (should be list of [x, y] points).
        field: Field name for error messages.
        
    Returns:
        List of error messages.
    """
    errors = []
    
    if not isinstance(curves, list):
        errors.append(f"{field}: Expected list of points")
        return errors
    
    if len(curves) < 2:
        errors.append(f"{field}: Curve must have at least 2 points")
        return errors
    
    for i, point in enumerate(curves):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            errors.append(f"{field}[{i}]: Expected [x, y] point")
            continue
        
        x, y = point
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            errors.append(f"{field}[{i}]: Point values must be numbers")
            continue
        
        if x < 0 or x > 255 or y < 0 or y > 255:
            errors.append(f"{field}[{i}]: Point values must be in range [0, 255]")
    
    return errors


def validate_preset(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    file_path: str = ""
) -> List[ValidationError]:
    """
    Validate a preset against a schema.
    
    Args:
        data: Preset data dictionary.
        schema: Schema to validate against.
        file_path: File path for error messages.
        
    Returns:
        List of validation errors.
    """
    errors = []
    
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(ValidationError(
                path=file_path,
                field=field,
                message="Required field missing"
            ))
    
    # Check types
    types = schema.get("types", {})
    for field, expected_type in types.items():
        if field in data:
            error = validate_type(data[field], expected_type, field)
            if error:
                errors.append(ValidationError(
                    path=file_path,
                    field=field,
                    message=error
                ))
    
    # Check ranges
    ranges = schema.get("ranges", {})
    for field, (min_val, max_val) in ranges.items():
        if field in data:
            error = validate_range(data[field], min_val, max_val, field)
            if error:
                errors.append(ValidationError(
                    path=file_path,
                    field=field,
                    message=error,
                    severity="warning"
                ))
    
    # Check nested schemas
    nested = schema.get("nested", {})
    for field, nested_schema in nested.items():
        if field in data and isinstance(data[field], dict):
            nested_errors = validate_preset(data[field], nested_schema, f"{file_path}.{field}")
            errors.extend(nested_errors)
    
    # Check for unknown fields (warning only)
    known_fields = set(schema.get("required", [])) | set(schema.get("optional", []))
    for field in data:
        if field not in known_fields and field not in nested:
            errors.append(ValidationError(
                path=file_path,
                field=field,
                message="Unknown field",
                severity="warning"
            ))
    
    # Validate curves if present
    for curve_field in ["curves_rgb", "curves_red", "curves_green", "curves_blue"]:
        if curve_field in data:
            curve_errors = validate_curves(data[curve_field], curve_field)
            for msg in curve_errors:
                errors.append(ValidationError(
                    path=file_path,
                    field=curve_field,
                    message=msg
                ))
    
    return errors


def validate_preset_file(file_path: str, preset_type: str = "auto") -> Tuple[bool, List[ValidationError]]:
    """
    Validate a preset JSON file.
    
    Args:
        file_path: Path to the preset file.
        preset_type: Type of preset ("film", "photo", "adjustment", or "auto").
        
    Returns:
        Tuple of (is_valid, list of errors).
    """
    errors = []
    
    # Check file exists
    if not os.path.exists(file_path):
        errors.append(ValidationError(
            path=file_path,
            field="file",
            message="File not found"
        ))
        return False, errors
    
    # Try to parse JSON
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(ValidationError(
            path=file_path,
            field="json",
            message=f"Invalid JSON: {e}"
        ))
        return False, errors
    except Exception as e:
        errors.append(ValidationError(
            path=file_path,
            field="file",
            message=f"Error reading file: {e}"
        ))
        return False, errors
    
    # Determine schema
    if preset_type == "auto":
        # Try to detect from path or content
        if "film" in file_path.lower():
            preset_type = "film"
        elif "photo" in file_path.lower():
            preset_type = "photo"
        else:
            preset_type = "adjustment"
    
    schema_map = {
        "film": FILM_PRESET_SCHEMA,
        "photo": PHOTO_PRESET_SCHEMA,
        "adjustment": ADJUSTMENT_PRESET_SCHEMA,
    }
    
    schema = schema_map.get(preset_type, ADJUSTMENT_PRESET_SCHEMA)
    
    # Validate
    errors = validate_preset(data, schema, file_path)
    
    # Determine if valid (no errors, warnings are OK)
    is_valid = not any(e.severity == "error" for e in errors)
    
    return is_valid, errors


def validate_all_presets(preset_dir: str) -> Dict[str, List[ValidationError]]:
    """
    Validate all preset files in a directory.
    
    Args:
        preset_dir: Directory containing preset files.
        
    Returns:
        Dict mapping file paths to their validation errors.
    """
    results = {}
    
    if not os.path.isdir(preset_dir):
        logger.warning("Preset directory not found: %s", preset_dir)
        return results
    
    for root, dirs, files in os.walk(preset_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                is_valid, errors = validate_preset_file(file_path)
                if errors:
                    results[file_path] = errors
    
    return results


def get_preset_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Get basic info from a preset file without full validation.
    
    Args:
        file_path: Path to preset file.
        
    Returns:
        Dict with name, id, description, or None if invalid.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "name": data.get("name", "Unknown"),
            "id": data.get("id", os.path.splitext(os.path.basename(file_path))[0]),
            "description": data.get("description", ""),
            "category": data.get("category", ""),
            "file_path": file_path,
        }
    except Exception:
        return None
