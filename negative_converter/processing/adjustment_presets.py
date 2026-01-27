from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)

try:
    import appdirs
except ImportError:  # pragma: no cover
    appdirs = None


@dataclass(frozen=True)
class AdjustmentPreset:
    """A user-defined snapshot of AdjustmentPanel settings."""
    id: str
    name: str
    parameters: Dict


def _slugify(name: str) -> str:
    # Minimal slugify: keep it predictable and filesystem/json safe.
    slug = name.strip().lower().replace(" ", "_").replace("-", "_")
    slug = "".join(ch for ch in slug if (ch.isalnum() or ch == "_"))
    return slug or "preset"


def _validate_adjustment_preset(preset: dict, source: str) -> Tuple[bool, Optional[dict]]:
    if not isinstance(preset, dict):
        logger.warning("Invalid adjustment preset from %s: expected dict", source)
        return False, None

    preset_id = preset.get("id")
    name = preset.get("name")
    params = preset.get("parameters")

    if not isinstance(preset_id, str) or not preset_id.strip():
        logger.warning("Invalid adjustment preset from %s: missing/invalid 'id'", source)
        return False, None
    if not isinstance(name, str) or not name.strip():
        logger.warning("Invalid adjustment preset '%s' from %s: missing/invalid 'name'", preset_id, source)
        return False, None
    if not isinstance(params, dict):
        logger.warning("Invalid adjustment preset '%s' from %s: missing/invalid 'parameters'", preset_id, source)
        return False, None

    return True, preset


class AdjustmentPresetManager:
    """
    Load/save user adjustment presets.

    File format: JSON list of objects:
      { "id": "...", "name": "...", "parameters": { ... } }
    """

    def __init__(self, presets_file: Optional[str] = None):
        if presets_file:
            self.presets_file = presets_file
        elif appdirs:
            app_name = "NegativeConverter"
            app_author = "NegativeConverter"
            data_dir = appdirs.user_data_dir(app_name, app_author)
            os.makedirs(data_dir, exist_ok=True)
            self.presets_file = os.path.join(data_dir, "adjustment_presets.json")
        else:
            # Fallback: project root
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.presets_file = os.path.join(base_dir, "adjustment_presets.json")

        self._presets: Dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        self._presets = {}
        if not os.path.isfile(self.presets_file):
            logger.info("No adjustment presets file found at %s.", self.presets_file)
            return

        try:
            with open(self.presets_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.exception("Failed to decode adjustment presets file %s", self.presets_file)
            return
        except Exception:
            logger.exception("Failed to read adjustment presets file %s", self.presets_file)
            return

        if not isinstance(data, list):
            logger.warning("Adjustment presets file %s must contain a JSON list.", self.presets_file)
            return

        for idx, preset in enumerate(data):
            ok, validated = _validate_adjustment_preset(preset, source=f"{self.presets_file}#{idx}")
            if ok:
                self._presets[validated["id"]] = validated

        logger.info("Loaded %s adjustment presets from %s.", len(self._presets), self.presets_file)

    def list_presets(self) -> List[dict]:
        return sorted(self._presets.values(), key=lambda p: p.get("name", ""))

    def get_preset(self, preset_id: str) -> Optional[dict]:
        return self._presets.get(preset_id)

    def _save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.presets_file), exist_ok=True)
            with open(self.presets_file, "w", encoding="utf-8") as f:
                json.dump(self.list_presets(), f, indent=2)
            logger.info("Saved %s adjustment presets to %s.", len(self._presets), self.presets_file)
            return True
        except Exception:
            logger.exception("Failed saving adjustment presets to %s", self.presets_file)
            return False

    def add_preset(self, name: str, parameters: Dict, preset_id: Optional[str] = None, *, overwrite: bool = False) -> Tuple[bool, str]:
        if not isinstance(parameters, dict):
            return False, "parameters must be a dict"

        resolved_id = (preset_id or _slugify(name)).strip()
        if not overwrite and resolved_id in self._presets:
            return False, f"Preset '{resolved_id}' already exists"

        preset = {"id": resolved_id, "name": name.strip() or resolved_id, "parameters": parameters}
        ok, validated = _validate_adjustment_preset(preset, source="add_preset")
        if not ok:
            return False, "invalid preset payload"

        self._presets[resolved_id] = validated
        if self._save():
            return True, resolved_id
        return False, "failed to save presets file"

    def delete_preset(self, preset_id: str) -> bool:
        if preset_id not in self._presets:
            return False
        self._presets.pop(preset_id, None)
        return self._save()