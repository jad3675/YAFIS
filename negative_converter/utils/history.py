# History management for undo/redo functionality
"""
Provides a generic history stack for undo/redo operations.
Can be used for adjustment history, image state history, etc.
"""

from typing import TypeVar, Generic, Optional, List, Callable, Any
from dataclasses import dataclass, field
from copy import deepcopy
import time

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


@dataclass
class HistoryEntry(Generic[T]):
    """A single entry in the history stack."""
    state: T
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


class HistoryStack(Generic[T]):
    """
    Generic history stack supporting undo/redo operations.
    
    Type parameter T represents the state type being tracked.
    """
    
    def __init__(
        self,
        max_size: int = 50,
        deep_copy: bool = True,
        on_change: Optional[Callable[[], None]] = None
    ):
        """
        Initialize the history stack.
        
        Args:
            max_size: Maximum number of history entries to keep.
            deep_copy: Whether to deep copy states when pushing.
            on_change: Optional callback when history changes.
        """
        self._undo_stack: List[HistoryEntry[T]] = []
        self._redo_stack: List[HistoryEntry[T]] = []
        self._max_size = max_size
        self._deep_copy = deep_copy
        self._on_change = on_change
        self._is_applying = False  # Prevent recursive pushes during undo/redo
    
    def push(self, state: T, description: str = "") -> None:
        """
        Push a new state onto the history stack.
        
        Args:
            state: The state to save.
            description: Optional description of the change.
        """
        if self._is_applying:
            return
        
        # Deep copy if enabled
        if self._deep_copy:
            state = deepcopy(state)
        
        entry = HistoryEntry(state=state, description=description)
        self._undo_stack.append(entry)
        
        # Clear redo stack on new action
        self._redo_stack.clear()
        
        # Trim if over max size
        while len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)
        
        logger.debug("History push: %s (stack size: %d)", description or "unnamed", len(self._undo_stack))
        self._notify_change()
    
    def undo(self) -> Optional[T]:
        """
        Undo the last action and return the previous state.
        
        Returns:
            The previous state, or None if nothing to undo.
        """
        if not self.can_undo():
            logger.debug("Nothing to undo")
            return None
        
        self._is_applying = True
        try:
            # Move current state to redo stack
            current = self._undo_stack.pop()
            self._redo_stack.append(current)
            
            # Return the new current state (top of undo stack)
            if self._undo_stack:
                result = self._undo_stack[-1].state
                if self._deep_copy:
                    result = deepcopy(result)
                logger.debug("Undo: restored to '%s'", self._undo_stack[-1].description or "unnamed")
                self._notify_change()
                return result
            
            logger.debug("Undo: stack empty after undo")
            self._notify_change()
            return None
        finally:
            self._is_applying = False
    
    def redo(self) -> Optional[T]:
        """
        Redo the last undone action and return the restored state.
        
        Returns:
            The restored state, or None if nothing to redo.
        """
        if not self.can_redo():
            logger.debug("Nothing to redo")
            return None
        
        self._is_applying = True
        try:
            # Move state from redo to undo stack
            entry = self._redo_stack.pop()
            self._undo_stack.append(entry)
            
            result = entry.state
            if self._deep_copy:
                result = deepcopy(result)
            
            logger.debug("Redo: restored to '%s'", entry.description or "unnamed")
            self._notify_change()
            return result
        finally:
            self._is_applying = False
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 1  # Need at least 2 entries to undo
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        logger.debug("History cleared")
        self._notify_change()
    
    def get_undo_description(self) -> Optional[str]:
        """Get description of the action that would be undone."""
        if len(self._undo_stack) > 0:
            return self._undo_stack[-1].description
        return None
    
    def get_redo_description(self) -> Optional[str]:
        """Get description of the action that would be redone."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return None
    
    def get_undo_count(self) -> int:
        """Get number of available undo steps."""
        return max(0, len(self._undo_stack) - 1)
    
    def get_redo_count(self) -> int:
        """Get number of available redo steps."""
        return len(self._redo_stack)
    
    def get_current_state(self) -> Optional[T]:
        """Get the current state without modifying history."""
        if self._undo_stack:
            state = self._undo_stack[-1].state
            if self._deep_copy:
                return deepcopy(state)
            return state
        return None
    
    def _notify_change(self) -> None:
        """Notify listeners of history change."""
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                logger.exception("Error in history change callback")


class ImageHistoryStack:
    """
    Specialized history stack for image states.
    
    Stores image arrays with memory-efficient compression for large images.
    """
    
    def __init__(
        self,
        max_size: int = 10,
        compress_threshold_mb: float = 50.0,
        on_change: Optional[Callable[[], None]] = None
    ):
        """
        Initialize image history stack.
        
        Args:
            max_size: Maximum number of image states to keep.
            compress_threshold_mb: Compress images larger than this (MB).
            on_change: Optional callback when history changes.
        """
        import numpy as np
        self._undo_stack: List[tuple] = []  # (compressed_data, shape, dtype, description)
        self._redo_stack: List[tuple] = []
        self._max_size = max_size
        self._compress_threshold = compress_threshold_mb * 1024 * 1024
        self._on_change = on_change
        self._is_applying = False
    
    def push(self, image: 'np.ndarray', description: str = "") -> None:
        """Push an image state onto the history stack."""
        import numpy as np
        
        if self._is_applying or image is None:
            return
        
        # Store image data
        data = self._compress_image(image)
        entry = (data, image.shape, image.dtype, description, time.time())
        
        self._undo_stack.append(entry)
        self._redo_stack.clear()
        
        # Trim if over max size
        while len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)
        
        logger.debug("Image history push: %s (stack size: %d)", description or "unnamed", len(self._undo_stack))
        self._notify_change()
    
    def undo(self) -> Optional['np.ndarray']:
        """Undo and return the previous image state."""
        if not self.can_undo():
            return None
        
        self._is_applying = True
        try:
            current = self._undo_stack.pop()
            self._redo_stack.append(current)
            
            if self._undo_stack:
                entry = self._undo_stack[-1]
                image = self._decompress_image(entry[0], entry[1], entry[2])
                logger.debug("Image undo: restored to '%s'", entry[3] or "unnamed")
                self._notify_change()
                return image
            
            self._notify_change()
            return None
        finally:
            self._is_applying = False
    
    def redo(self) -> Optional['np.ndarray']:
        """Redo and return the restored image state."""
        if not self.can_redo():
            return None
        
        self._is_applying = True
        try:
            entry = self._redo_stack.pop()
            self._undo_stack.append(entry)
            
            image = self._decompress_image(entry[0], entry[1], entry[2])
            logger.debug("Image redo: restored to '%s'", entry[3] or "unnamed")
            self._notify_change()
            return image
        finally:
            self._is_applying = False
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 1
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._notify_change()
    
    def _compress_image(self, image: 'np.ndarray') -> bytes:
        """Compress image data if it exceeds threshold."""
        import numpy as np
        
        raw_size = image.nbytes
        if raw_size > self._compress_threshold:
            try:
                import zlib
                compressed = zlib.compress(image.tobytes(), level=1)
                logger.debug("Compressed image: %.1f MB -> %.1f MB", 
                           raw_size / 1024 / 1024, len(compressed) / 1024 / 1024)
                return compressed
            except Exception:
                logger.debug("Compression failed, storing raw")
        
        return image.tobytes()
    
    def _decompress_image(self, data: bytes, shape: tuple, dtype) -> 'np.ndarray':
        """Decompress image data."""
        import numpy as np
        
        expected_size = np.prod(shape) * np.dtype(dtype).itemsize
        
        if len(data) != expected_size:
            try:
                import zlib
                data = zlib.decompress(data)
            except Exception:
                logger.exception("Decompression failed")
                return None
        
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    
    def _notify_change(self) -> None:
        """Notify listeners of history change."""
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                logger.exception("Error in history change callback")
