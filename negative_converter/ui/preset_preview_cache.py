# Background preset preview caching
"""
Background caching of preset previews for instant hover feedback.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import time
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QMutex, QMutexLocker, QTimer
from PyQt6.QtGui import QPixmap, QImage
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PreviewCacheEntry:
    """A cached preset preview."""
    preset_id: str
    pixmap: QPixmap
    timestamp: float
    thumbnail_size: tuple


class PresetPreviewWorker(QObject):
    """Worker for generating preset previews in background."""
    
    preview_ready = pyqtSignal(str, QPixmap)  # preset_id, pixmap
    batch_complete = pyqtSignal()
    error = pyqtSignal(str, str)  # preset_id, error_message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_image: Optional[np.ndarray] = None
        self._thumbnail_size = (80, 80)
        self._preset_queue: List[tuple] = []  # (preset_id, apply_func)
        self._is_running = False
        self._should_stop = False
        self._mutex = QMutex()
    
    def set_base_image(self, image: np.ndarray, thumbnail_size: tuple = (80, 80)):
        """Set the base image for preview generation."""
        with QMutexLocker(self._mutex):
            if image is not None:
                # Create a small thumbnail for fast preview generation
                import cv2
                h, w = image.shape[:2]
                scale = min(thumbnail_size[0] / w, thumbnail_size[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
                self._base_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                self._base_image = None
            self._thumbnail_size = thumbnail_size
    
    def queue_preset(self, preset_id: str, apply_func: Callable[[np.ndarray], np.ndarray]):
        """Queue a preset for preview generation."""
        with QMutexLocker(self._mutex):
            # Remove if already queued
            self._preset_queue = [(pid, func) for pid, func in self._preset_queue if pid != preset_id]
            self._preset_queue.append((preset_id, apply_func))
    
    def queue_presets(self, presets: List[tuple]):
        """Queue multiple presets."""
        with QMutexLocker(self._mutex):
            for preset_id, apply_func in presets:
                self._preset_queue = [(pid, func) for pid, func in self._preset_queue if pid != preset_id]
                self._preset_queue.append((preset_id, apply_func))
    
    def clear_queue(self):
        """Clear the preset queue."""
        with QMutexLocker(self._mutex):
            self._preset_queue.clear()
    
    def stop(self):
        """Request worker to stop."""
        with QMutexLocker(self._mutex):
            self._should_stop = True
    
    @pyqtSlot()
    def process_queue(self):
        """Process all queued presets."""
        with QMutexLocker(self._mutex):
            if self._is_running:
                return
            self._is_running = True
            self._should_stop = False
        
        try:
            while True:
                with QMutexLocker(self._mutex):
                    if self._should_stop or not self._preset_queue:
                        break
                    
                    if self._base_image is None:
                        break
                    
                    preset_id, apply_func = self._preset_queue.pop(0)
                    base_copy = self._base_image.copy()
                
                try:
                    # Apply preset
                    result = apply_func(base_copy)
                    
                    if result is not None:
                        # Convert to QPixmap
                        h, w = result.shape[:2]
                        bytes_per_line = 3 * w
                        q_image = QImage(result.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_image)
                        
                        self.preview_ready.emit(preset_id, pixmap)
                    
                except Exception as e:
                    self.error.emit(preset_id, str(e))
                    logger.debug("Preview generation failed for %s: %s", preset_id, e)
        
        finally:
            with QMutexLocker(self._mutex):
                self._is_running = False
            self.batch_complete.emit()


class PresetPreviewCache(QObject):
    """
    Cache for preset previews with background generation.
    """
    
    preview_updated = pyqtSignal(str)  # preset_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._cache: Dict[str, PreviewCacheEntry] = {}
        self._max_cache_size = 100
        self._thumbnail_size = (80, 80)
        self._base_image_hash: Optional[int] = None
        
        # Background worker
        self._worker_thread = QThread()
        self._worker = PresetPreviewWorker()
        self._worker.moveToThread(self._worker_thread)
        
        # Connect signals
        self._worker.preview_ready.connect(self._on_preview_ready)
        self._worker.error.connect(self._on_preview_error)
        
        self._worker_thread.start()
        
        # Debounce timer for batch processing
        self._process_timer = QTimer(self)
        self._process_timer.setSingleShot(True)
        self._process_timer.setInterval(50)
        self._process_timer.timeout.connect(self._trigger_processing)
    
    def set_base_image(self, image: np.ndarray):
        """
        Set the base image for preview generation.
        
        This invalidates the cache if the image has changed.
        """
        if image is None:
            self._base_image_hash = None
            self._worker.set_base_image(None)
            return
        
        # Simple hash to detect image changes
        new_hash = hash(image.tobytes()[:1000])  # Hash first 1000 bytes
        
        if new_hash != self._base_image_hash:
            self._base_image_hash = new_hash
            self._cache.clear()
            self._worker.set_base_image(image, self._thumbnail_size)
            logger.debug("Base image changed, cache cleared")
    
    def get_preview(self, preset_id: str) -> Optional[QPixmap]:
        """Get a cached preview if available."""
        if preset_id in self._cache:
            return self._cache[preset_id].pixmap
        return None
    
    def has_preview(self, preset_id: str) -> bool:
        """Check if a preview is cached."""
        return preset_id in self._cache
    
    def request_preview(self, preset_id: str, apply_func: Callable[[np.ndarray], np.ndarray]):
        """
        Request a preview to be generated.
        
        If already cached, returns immediately via preview_updated signal.
        Otherwise queues for background generation.
        """
        if preset_id in self._cache:
            self.preview_updated.emit(preset_id)
            return
        
        self._worker.queue_preset(preset_id, apply_func)
        self._process_timer.start()
    
    def request_previews(self, presets: List[tuple]):
        """
        Request multiple previews.
        
        Args:
            presets: List of (preset_id, apply_func) tuples.
        """
        to_generate = []
        
        for preset_id, apply_func in presets:
            if preset_id not in self._cache:
                to_generate.append((preset_id, apply_func))
        
        if to_generate:
            self._worker.queue_presets(to_generate)
            self._process_timer.start()
    
    def invalidate(self, preset_id: str = None):
        """
        Invalidate cached previews.
        
        Args:
            preset_id: Specific preset to invalidate, or None for all.
        """
        if preset_id is None:
            self._cache.clear()
        elif preset_id in self._cache:
            del self._cache[preset_id]
    
    def clear(self):
        """Clear all cached previews."""
        self._cache.clear()
        self._worker.clear_queue()
    
    def _trigger_processing(self):
        """Trigger background processing."""
        # Use QMetaObject to call slot on worker thread
        from PyQt6.QtCore import QMetaObject, Qt as QtCore_Qt
        QMetaObject.invokeMethod(
            self._worker, "process_queue",
            QtCore_Qt.ConnectionType.QueuedConnection
        )
    
    @pyqtSlot(str, QPixmap)
    def _on_preview_ready(self, preset_id: str, pixmap: QPixmap):
        """Handle completed preview."""
        # Add to cache
        entry = PreviewCacheEntry(
            preset_id=preset_id,
            pixmap=pixmap,
            timestamp=time.time(),
            thumbnail_size=self._thumbnail_size
        )
        self._cache[preset_id] = entry
        
        # Trim cache if needed
        self._trim_cache()
        
        # Notify listeners
        self.preview_updated.emit(preset_id)
    
    @pyqtSlot(str, str)
    def _on_preview_error(self, preset_id: str, error: str):
        """Handle preview generation error."""
        logger.debug("Preview error for %s: %s", preset_id, error)
    
    def _trim_cache(self):
        """Remove oldest entries if cache is too large."""
        if len(self._cache) <= self._max_cache_size:
            return
        
        # Sort by timestamp and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        to_remove = len(self._cache) - self._max_cache_size
        for preset_id, _ in sorted_entries[:to_remove]:
            del self._cache[preset_id]
    
    def shutdown(self):
        """Shutdown the background worker."""
        self._worker.stop()
        self._worker_thread.quit()
        self._worker_thread.wait(1000)
    
    def __del__(self):
        self.shutdown()


class PreviewablePresetPanel:
    """
    Mixin for preset panels that want preview caching.
    
    Add this to your preset panel class to enable background preview generation.
    """
    
    def init_preview_cache(self):
        """Initialize the preview cache. Call in __init__."""
        self._preview_cache = PresetPreviewCache(self)
        self._preview_cache.preview_updated.connect(self._on_preview_cache_updated)
    
    def set_preview_base_image(self, image: np.ndarray):
        """Set the base image for preview generation."""
        if hasattr(self, '_preview_cache'):
            self._preview_cache.set_base_image(image)
    
    def get_cached_preview(self, preset_id: str) -> Optional[QPixmap]:
        """Get a cached preview."""
        if hasattr(self, '_preview_cache'):
            return self._preview_cache.get_preview(preset_id)
        return None
    
    def request_preset_preview(self, preset_id: str, apply_func: Callable):
        """Request a preview to be generated."""
        if hasattr(self, '_preview_cache'):
            self._preview_cache.request_preview(preset_id, apply_func)
    
    def _on_preview_cache_updated(self, preset_id: str):
        """Override to handle preview updates."""
        pass
    
    def shutdown_preview_cache(self):
        """Shutdown the preview cache. Call in closeEvent."""
        if hasattr(self, '_preview_cache'):
            self._preview_cache.shutdown()
