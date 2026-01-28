# Session persistence for saving/restoring work state
"""
Session management for saving and restoring application state.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import json
import os
import time
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImageState:
    """State of a single image in the session."""
    file_path: str
    adjustments: Dict[str, Any] = field(default_factory=dict)
    film_preset: Optional[str] = None
    photo_preset: Optional[str] = None
    is_converted: bool = False
    negative_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImageState':
        return cls(**data)


@dataclass
class SessionData:
    """Complete session state."""
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    # Current image
    current_image_path: Optional[str] = None
    current_adjustments: Dict[str, Any] = field(default_factory=dict)
    current_film_preset: Optional[str] = None
    current_photo_preset: Optional[str] = None
    current_negative_type: Optional[str] = None
    
    # Batch images
    batch_images: List[ImageState] = field(default_factory=list)
    batch_output_dir: Optional[str] = None
    
    # UI state
    window_geometry: Optional[Dict[str, int]] = None
    dock_states: Dict[str, bool] = field(default_factory=dict)
    zoom_level: float = 1.0
    
    # Export settings
    last_export_preset: Optional[str] = None
    last_save_directory: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert ImageState objects to dicts
        data["batch_images"] = [img.to_dict() if isinstance(img, ImageState) else img 
                                for img in self.batch_images]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        # Convert batch_images dicts to ImageState objects
        if "batch_images" in data:
            data["batch_images"] = [
                ImageState.from_dict(img) if isinstance(img, dict) else img
                for img in data["batch_images"]
            ]
        
        # Filter to known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
    
    def update_modified(self) -> None:
        """Update the modified timestamp."""
        self.modified_at = time.time()
    
    @property
    def created_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.created_at)
    
    @property
    def modified_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.modified_at)


class SessionManager:
    """Manages session persistence."""
    
    def __init__(self, sessions_dir: Optional[str] = None):
        """
        Initialize the session manager.
        
        Args:
            sessions_dir: Directory for session files. If None, uses default.
        """
        if sessions_dir is None:
            config_dir = os.path.dirname(os.path.dirname(__file__))
            sessions_dir = os.path.join(config_dir, "config", "sessions")
        
        self._sessions_dir = sessions_dir
        self._current_session: Optional[SessionData] = None
        self._current_session_file: Optional[str] = None
        self._auto_save_enabled = True
        self._auto_save_interval = 60  # seconds
        self._last_auto_save = 0
        
        os.makedirs(self._sessions_dir, exist_ok=True)
    
    def new_session(self) -> SessionData:
        """Create a new session."""
        self._current_session = SessionData()
        self._current_session_file = None
        logger.info("Created new session")
        return self._current_session
    
    def get_current_session(self) -> Optional[SessionData]:
        """Get the current session."""
        return self._current_session
    
    def save_session(self, file_path: Optional[str] = None) -> bool:
        """
        Save the current session to a file.
        
        Args:
            file_path: Path to save to. If None, uses current session file or generates new.
            
        Returns:
            True if successful.
        """
        if self._current_session is None:
            logger.warning("No session to save")
            return False
        
        if file_path is None:
            if self._current_session_file:
                file_path = self._current_session_file
            else:
                # Generate new filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(self._sessions_dir, f"session_{timestamp}.json")
        
        try:
            self._current_session.update_modified()
            data = self._current_session.to_dict()
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self._current_session_file = file_path
            self._last_auto_save = time.time()
            logger.info("Session saved to %s", file_path)
            return True
            
        except Exception as e:
            logger.exception("Failed to save session")
            return False
    
    def load_session(self, file_path: str) -> Optional[SessionData]:
        """
        Load a session from a file.
        
        Args:
            file_path: Path to the session file.
            
        Returns:
            Loaded SessionData or None if failed.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._current_session = SessionData.from_dict(data)
            self._current_session_file = file_path
            logger.info("Session loaded from %s", file_path)
            return self._current_session
            
        except Exception as e:
            logger.exception("Failed to load session from %s", file_path)
            return None
    
    def load_last_session(self) -> Optional[SessionData]:
        """Load the most recently modified session."""
        sessions = self.list_sessions()
        if sessions:
            # Sort by modified time, most recent first
            sessions.sort(key=lambda x: x.get("modified_at", 0), reverse=True)
            return self.load_session(sessions[0]["file_path"])
        return None
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved sessions.
        
        Returns:
            List of session info dicts with file_path, created_at, modified_at.
        """
        sessions = []
        
        if not os.path.isdir(self._sessions_dir):
            return sessions
        
        for filename in os.listdir(self._sessions_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self._sessions_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    sessions.append({
                        "file_path": file_path,
                        "filename": filename,
                        "created_at": data.get("created_at", 0),
                        "modified_at": data.get("modified_at", 0),
                        "current_image": data.get("current_image_path"),
                        "batch_count": len(data.get("batch_images", [])),
                    })
                except Exception:
                    pass
        
        return sessions
    
    def delete_session(self, file_path: str) -> bool:
        """Delete a session file."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                if self._current_session_file == file_path:
                    self._current_session_file = None
                logger.info("Session deleted: %s", file_path)
                return True
        except Exception as e:
            logger.exception("Failed to delete session")
        return False
    
    def auto_save_if_needed(self) -> bool:
        """
        Auto-save if enough time has passed since last save.
        
        Returns:
            True if auto-save was performed.
        """
        if not self._auto_save_enabled or self._current_session is None:
            return False
        
        if time.time() - self._last_auto_save >= self._auto_save_interval:
            return self.save_session()
        
        return False
    
    def set_auto_save(self, enabled: bool, interval: int = 60) -> None:
        """Configure auto-save settings."""
        self._auto_save_enabled = enabled
        self._auto_save_interval = interval
    
    # Convenience methods for updating session state
    
    def set_current_image(
        self,
        file_path: str,
        adjustments: Optional[Dict[str, Any]] = None,
        negative_type: Optional[str] = None
    ) -> None:
        """Update current image in session."""
        if self._current_session is None:
            self.new_session()
        
        self._current_session.current_image_path = file_path
        if adjustments is not None:
            self._current_session.current_adjustments = adjustments.copy()
        if negative_type is not None:
            self._current_session.current_negative_type = negative_type
        self._current_session.update_modified()
    
    def set_adjustments(self, adjustments: Dict[str, Any]) -> None:
        """Update current adjustments in session."""
        if self._current_session is None:
            self.new_session()
        
        self._current_session.current_adjustments = adjustments.copy()
        self._current_session.update_modified()
    
    def set_presets(
        self,
        film_preset: Optional[str] = None,
        photo_preset: Optional[str] = None
    ) -> None:
        """Update current presets in session."""
        if self._current_session is None:
            self.new_session()
        
        if film_preset is not None:
            self._current_session.current_film_preset = film_preset
        if photo_preset is not None:
            self._current_session.current_photo_preset = photo_preset
        self._current_session.update_modified()
    
    def set_batch_images(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None
    ) -> None:
        """Update batch images in session."""
        if self._current_session is None:
            self.new_session()
        
        self._current_session.batch_images = [
            ImageState(file_path=path) for path in image_paths
        ]
        if output_dir is not None:
            self._current_session.batch_output_dir = output_dir
        self._current_session.update_modified()
    
    def set_window_state(
        self,
        geometry: Optional[Dict[str, int]] = None,
        dock_states: Optional[Dict[str, bool]] = None,
        zoom_level: Optional[float] = None
    ) -> None:
        """Update window state in session."""
        if self._current_session is None:
            self.new_session()
        
        if geometry is not None:
            self._current_session.window_geometry = geometry
        if dock_states is not None:
            self._current_session.dock_states = dock_states
        if zoom_level is not None:
            self._current_session.zoom_level = zoom_level
        self._current_session.update_modified()


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
