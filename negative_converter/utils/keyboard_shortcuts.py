# Keyboard shortcuts management
"""
Centralized keyboard shortcut definitions and management.
"""

from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import QWidget

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ShortcutDefinition:
    """Definition of a keyboard shortcut."""
    key: str
    description: str
    category: str = "General"
    default_key: str = ""
    
    def __post_init__(self):
        if not self.default_key:
            self.default_key = self.key


# Default shortcut definitions
DEFAULT_SHORTCUTS: Dict[str, ShortcutDefinition] = {
    # File operations
    "file.open": ShortcutDefinition("Ctrl+O", "Open image", "File"),
    "file.save": ShortcutDefinition("Ctrl+S", "Save image", "File"),
    "file.save_as": ShortcutDefinition("Ctrl+Shift+S", "Save image as", "File"),
    "file.info": ShortcutDefinition("Ctrl+I", "Show image info", "File"),
    
    # Edit operations
    "edit.undo": ShortcutDefinition("Ctrl+Z", "Undo", "Edit"),
    "edit.redo": ShortcutDefinition("Ctrl+Y", "Redo", "Edit"),
    "edit.reset": ShortcutDefinition("Ctrl+R", "Reset all adjustments", "Edit"),
    
    # View operations
    "view.compare": ShortcutDefinition("\\", "Toggle before/after", "View"),
    "view.fit": ShortcutDefinition("Ctrl+0", "Fit to window", "View"),
    "view.zoom_100": ShortcutDefinition("Ctrl+1", "Zoom to 100%", "View"),
    "view.zoom_in": ShortcutDefinition("Ctrl+=", "Zoom in", "View"),
    "view.zoom_out": ShortcutDefinition("Ctrl+-", "Zoom out", "View"),
    
    # Adjustment shortcuts
    "adjust.exposure_up": ShortcutDefinition("]", "Increase exposure", "Adjustments"),
    "adjust.exposure_down": ShortcutDefinition("[", "Decrease exposure", "Adjustments"),
    "adjust.contrast_up": ShortcutDefinition("Shift+]", "Increase contrast", "Adjustments"),
    "adjust.contrast_down": ShortcutDefinition("Shift+[", "Decrease contrast", "Adjustments"),
    "adjust.saturation_up": ShortcutDefinition("Alt+]", "Increase saturation", "Adjustments"),
    "adjust.saturation_down": ShortcutDefinition("Alt+[", "Decrease saturation", "Adjustments"),
    "adjust.temp_warmer": ShortcutDefinition("Shift+.", "Warmer temperature", "Adjustments"),
    "adjust.temp_cooler": ShortcutDefinition("Shift+,", "Cooler temperature", "Adjustments"),
    "adjust.auto_wb": ShortcutDefinition("Ctrl+Shift+U", "Auto white balance", "Adjustments"),
    "adjust.auto_tone": ShortcutDefinition("Ctrl+Shift+T", "Auto tone", "Adjustments"),
    "adjust.auto_levels": ShortcutDefinition("Ctrl+Shift+L", "Auto levels", "Adjustments"),
    
    # Tools
    "tool.crop": ShortcutDefinition("C", "Crop tool", "Tools"),
    "tool.rotate_cw": ShortcutDefinition("Ctrl+Shift+R", "Rotate clockwise", "Tools"),
    "tool.rotate_ccw": ShortcutDefinition("Ctrl+Alt+R", "Rotate counter-clockwise", "Tools"),
    "tool.straighten": ShortcutDefinition("Ctrl+Shift+H", "Straighten", "Tools"),
    "tool.wb_picker": ShortcutDefinition("W", "White balance picker", "Tools"),
    "tool.color_sampler": ShortcutDefinition("S", "Color sampler", "Tools"),
    
    # Navigation
    "nav.next_image": ShortcutDefinition("Right", "Next image", "Navigation"),
    "nav.prev_image": ShortcutDefinition("Left", "Previous image", "Navigation"),
    "nav.first_image": ShortcutDefinition("Home", "First image", "Navigation"),
    "nav.last_image": ShortcutDefinition("End", "Last image", "Navigation"),
}


class ShortcutManager:
    """
    Manages keyboard shortcuts for the application.
    
    Provides centralized shortcut registration, customization, and conflict detection.
    """
    
    def __init__(self):
        self._shortcuts: Dict[str, ShortcutDefinition] = DEFAULT_SHORTCUTS.copy()
        self._registered: Dict[str, QShortcut] = {}
        self._callbacks: Dict[str, Callable] = {}
    
    def get_shortcut(self, action_id: str) -> Optional[str]:
        """Get the key sequence for an action."""
        if action_id in self._shortcuts:
            return self._shortcuts[action_id].key
        return None
    
    def get_description(self, action_id: str) -> Optional[str]:
        """Get the description for an action."""
        if action_id in self._shortcuts:
            return self._shortcuts[action_id].description
        return None
    
    def set_shortcut(self, action_id: str, key: str) -> bool:
        """
        Set a custom shortcut for an action.
        
        Args:
            action_id: The action identifier.
            key: The new key sequence string.
            
        Returns:
            True if successful, False if conflict detected.
        """
        # Check for conflicts
        conflict = self.find_conflict(key, exclude=action_id)
        if conflict:
            logger.warning("Shortcut conflict: %s already assigned to %s", key, conflict)
            return False
        
        if action_id in self._shortcuts:
            self._shortcuts[action_id].key = key
            # Update registered shortcut if exists
            if action_id in self._registered:
                self._registered[action_id].setKey(QKeySequence(key))
            return True
        return False
    
    def reset_shortcut(self, action_id: str) -> None:
        """Reset a shortcut to its default."""
        if action_id in self._shortcuts:
            default = self._shortcuts[action_id].default_key
            self._shortcuts[action_id].key = default
            if action_id in self._registered:
                self._registered[action_id].setKey(QKeySequence(default))
    
    def reset_all(self) -> None:
        """Reset all shortcuts to defaults."""
        for action_id in self._shortcuts:
            self.reset_shortcut(action_id)
    
    def find_conflict(self, key: str, exclude: Optional[str] = None) -> Optional[str]:
        """
        Find if a key sequence conflicts with existing shortcuts.
        
        Args:
            key: The key sequence to check.
            exclude: Action ID to exclude from conflict check.
            
        Returns:
            The conflicting action ID, or None if no conflict.
        """
        key_seq = QKeySequence(key)
        for action_id, shortcut in self._shortcuts.items():
            if action_id == exclude:
                continue
            if QKeySequence(shortcut.key) == key_seq:
                return action_id
        return None
    
    def register(
        self,
        action_id: str,
        parent: QWidget,
        callback: Callable,
        context: Any = None
    ) -> Optional[QShortcut]:
        """
        Register a shortcut with a callback.
        
        Args:
            action_id: The action identifier.
            parent: Parent widget for the shortcut.
            callback: Function to call when shortcut is triggered.
            context: Optional context for the shortcut.
            
        Returns:
            The created QShortcut, or None if action not found.
        """
        if action_id not in self._shortcuts:
            logger.warning("Unknown shortcut action: %s", action_id)
            return None
        
        key = self._shortcuts[action_id].key
        shortcut = QShortcut(QKeySequence(key), parent)
        shortcut.activated.connect(callback)
        
        self._registered[action_id] = shortcut
        self._callbacks[action_id] = callback
        
        return shortcut
    
    def unregister(self, action_id: str) -> None:
        """Unregister a shortcut."""
        if action_id in self._registered:
            self._registered[action_id].setEnabled(False)
            del self._registered[action_id]
        if action_id in self._callbacks:
            del self._callbacks[action_id]
    
    def get_all_by_category(self) -> Dict[str, List[tuple]]:
        """
        Get all shortcuts organized by category.
        
        Returns:
            Dict mapping category names to lists of (action_id, key, description).
        """
        by_category: Dict[str, List[tuple]] = {}
        for action_id, shortcut in self._shortcuts.items():
            category = shortcut.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((
                action_id,
                shortcut.key,
                shortcut.description
            ))
        return by_category
    
    def to_dict(self) -> Dict[str, str]:
        """Export shortcuts to a dictionary for saving."""
        return {
            action_id: shortcut.key
            for action_id, shortcut in self._shortcuts.items()
        }
    
    def from_dict(self, data: Dict[str, str]) -> None:
        """Import shortcuts from a dictionary."""
        for action_id, key in data.items():
            if action_id in self._shortcuts:
                self._shortcuts[action_id].key = key


# Global shortcut manager instance
_shortcut_manager: Optional[ShortcutManager] = None


def get_shortcut_manager() -> ShortcutManager:
    """Get the global shortcut manager instance."""
    global _shortcut_manager
    if _shortcut_manager is None:
        _shortcut_manager = ShortcutManager()
    return _shortcut_manager
