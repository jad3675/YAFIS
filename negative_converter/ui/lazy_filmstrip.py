# Lazy loading filmstrip widget
"""
Filmstrip widget with lazy thumbnail loading for better performance with large batches.
"""

from typing import Dict, List, Optional, Set
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListView,
    QListWidgetItem, QAbstractItemView, QStyle, QApplication,
    QScrollBar
)
from PyQt6.QtGui import QIcon, QPixmap, QImage, QColor
from PyQt6.QtCore import (
    QSize, Qt, QThread, QObject, pyqtSignal, pyqtSlot,
    QMutex, QMutexLocker, QRunnable, QThreadPool, QTimer
)
import cv2

from ..config import settings as app_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ThumbnailCache:
    """
    LRU cache for thumbnails with memory limit.
    """
    
    def __init__(self, max_size_mb: float = 100.0):
        self._cache: Dict[str, QPixmap] = {}
        self._access_order: List[str] = []
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._current_size = 0
        self._lock = QMutex()
    
    def get(self, key: str) -> Optional[QPixmap]:
        """Get a thumbnail from cache."""
        with QMutexLocker(self._lock):
            if key in self._cache:
                # Move to end (most recently used)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        return None
    
    def put(self, key: str, pixmap: QPixmap):
        """Add a thumbnail to cache."""
        if pixmap is None:
            return
        
        with QMutexLocker(self._lock):
            # Estimate pixmap size
            size = pixmap.width() * pixmap.height() * 4  # RGBA
            
            # Remove old entries if needed
            while self._current_size + size > self._max_size_bytes and self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    old_pixmap = self._cache.pop(oldest)
                    self._current_size -= old_pixmap.width() * old_pixmap.height() * 4
            
            # Add new entry
            self._cache[key] = pixmap
            self._access_order.append(key)
            self._current_size += size
    
    def contains(self, key: str) -> bool:
        """Check if key is in cache."""
        with QMutexLocker(self._lock):
            return key in self._cache
    
    def clear(self):
        """Clear the cache."""
        with QMutexLocker(self._lock):
            self._cache.clear()
            self._access_order.clear()
            self._current_size = 0
    
    def size(self) -> int:
        """Get number of cached items."""
        with QMutexLocker(self._lock):
            return len(self._cache)


class LazyThumbnailLoader(QObject):
    """Worker for loading thumbnails on demand."""
    thumbnail_ready = pyqtSignal(str, QPixmap)
    error = pyqtSignal(str, str)
    
    def __init__(self, file_path: str, target_size: QSize, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.target_size = target_size
        self._cancelled = False
        self._mutex = QMutex()
    
    @pyqtSlot()
    def run(self):
        """Load and resize the thumbnail."""
        with QMutexLocker(self._mutex):
            if self._cancelled:
                return
        
        try:
            img = cv2.imread(self.file_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            with QMutexLocker(self._mutex):
                if self._cancelled:
                    return
            
            # Resize
            h, w = img.shape[:2]
            target_w, target_h = self.target_size.width(), self.target_size.height()
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            with QMutexLocker(self._mutex):
                if self._cancelled:
                    return
            
            # Convert to QPixmap
            if len(resized.shape) == 3:
                h, w, ch = resized.shape
                bytes_per_line = ch * w
                q_img = QImage(resized.data, w, h, bytes_per_line, 
                              QImage.Format.Format_BGR888).rgbSwapped()
            else:
                h, w = resized.shape
                q_img = QImage(resized.data, w, h, w, QImage.Format.Format_Grayscale8)
            
            pixmap = QPixmap.fromImage(q_img)
            
            with QMutexLocker(self._mutex):
                if self._cancelled:
                    return
            
            self.thumbnail_ready.emit(self.file_path, pixmap)
            
        except Exception as e:
            self.error.emit(self.file_path, str(e))
    
    def cancel(self):
        with QMutexLocker(self._mutex):
            self._cancelled = True


class ThumbnailRunnable(QRunnable):
    """Runnable wrapper for thumbnail loader."""
    
    def __init__(self, loader: LazyThumbnailLoader):
        super().__init__()
        self.loader = loader
        self.setAutoDelete(True)
    
    def run(self):
        self.loader.run()


class LazyFilmstripWidget(QWidget):
    """
    Filmstrip widget with lazy thumbnail loading.
    
    Only loads thumbnails for visible items, improving performance
    with large batches.
    """
    selection_changed = pyqtSignal(list)
    preview_requested = pyqtSignal(str)
    checked_items_changed = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Settings
        initial_size = app_settings.UI_DEFAULTS.get("filmstrip_thumb_size", 120)
        self.thumbnail_size = QSize(initial_size, initial_size)
        
        # Data
        self._image_paths: List[str] = []
        self._list_items: Dict[str, QListWidgetItem] = {}
        
        # Lazy loading
        self._cache = ThumbnailCache(max_size_mb=100.0)
        self._thread_pool = QThreadPool.globalInstance()
        self._active_loaders: Dict[str, LazyThumbnailLoader] = {}
        self._pending_loads: Set[str] = set()
        self._visible_range = (0, 0)
        
        # Debounce timer for scroll events
        self._scroll_timer = QTimer(self)
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.setInterval(100)  # 100ms debounce
        self._scroll_timer.timeout.connect(self._load_visible_thumbnails)
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.list_widget = QListWidget(self)
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_widget.setMovement(QListView.Movement.Static)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.setSpacing(10)
        self.list_widget.setIconSize(self.thumbnail_size)
        self.list_widget.setWordWrap(True)
        self.list_widget.setUniformItemSizes(True)  # Performance optimization
        
        layout.addWidget(self.list_widget)
    
    def _connect_signals(self):
        self.list_widget.itemSelectionChanged.connect(self._emit_selection_change)
        self.list_widget.itemChanged.connect(self._handle_item_changed)
        
        # Connect scroll events
        scrollbar = self.list_widget.verticalScrollBar()
        scrollbar.valueChanged.connect(self._on_scroll)
    
    def _on_scroll(self):
        """Handle scroll events with debouncing."""
        self._scroll_timer.start()
    
    def _get_visible_range(self) -> tuple:
        """Get the range of visible item indices."""
        viewport = self.list_widget.viewport()
        if viewport is None:
            return (0, 0)
        
        # Get visible rect
        visible_rect = viewport.rect()
        
        # Find first and last visible items
        first_visible = -1
        last_visible = -1
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item_rect = self.list_widget.visualItemRect(item)
            
            if item_rect.intersects(visible_rect):
                if first_visible == -1:
                    first_visible = i
                last_visible = i
        
        # Add buffer for smoother scrolling
        buffer = 5
        first_visible = max(0, first_visible - buffer)
        last_visible = min(self.list_widget.count() - 1, last_visible + buffer)
        
        return (first_visible, last_visible)
    
    def _load_visible_thumbnails(self):
        """Load thumbnails for currently visible items."""
        first, last = self._get_visible_range()
        
        if first < 0 or last < 0:
            return
        
        self._visible_range = (first, last)
        
        for i in range(first, last + 1):
            if i >= len(self._image_paths):
                break
            
            path = self._image_paths[i]
            
            # Skip if already cached or loading
            if self._cache.contains(path) or path in self._pending_loads:
                # Update from cache if available
                cached = self._cache.get(path)
                if cached and path in self._list_items:
                    self._list_items[path].setIcon(QIcon(cached))
                continue
            
            # Start loading
            self._start_thumbnail_load(path)
    
    def _start_thumbnail_load(self, path: str):
        """Start loading a thumbnail."""
        if path in self._active_loaders:
            return
        
        self._pending_loads.add(path)
        
        loader = LazyThumbnailLoader(path, self.thumbnail_size)
        loader.thumbnail_ready.connect(self._on_thumbnail_ready)
        loader.error.connect(self._on_thumbnail_error)
        
        self._active_loaders[path] = loader
        
        runnable = ThumbnailRunnable(loader)
        self._thread_pool.start(runnable)
    
    @pyqtSlot(str, QPixmap)
    def _on_thumbnail_ready(self, path: str, pixmap: QPixmap):
        """Handle loaded thumbnail."""
        self._pending_loads.discard(path)
        
        if path in self._active_loaders:
            del self._active_loaders[path]
        
        # Cache the thumbnail
        self._cache.put(path, pixmap)
        
        # Update the list item
        if path in self._list_items:
            self._list_items[path].setIcon(QIcon(pixmap))
    
    @pyqtSlot(str, str)
    def _on_thumbnail_error(self, path: str, error: str):
        """Handle thumbnail loading error."""
        self._pending_loads.discard(path)
        
        if path in self._active_loaders:
            del self._active_loaders[path]
        
        logger.debug("Thumbnail error for %s: %s", os.path.basename(path), error)
        
        # Set error icon
        if path in self._list_items:
            error_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
            self._list_items[path].setIcon(error_icon)
    
    def add_images(self, file_paths: List[str]):
        """Add images to the filmstrip."""
        self.clear_images()
        self._image_paths = sorted(list(set(file_paths)))
        
        placeholder = self._create_placeholder_icon()
        
        for path in self._image_paths:
            base_name = os.path.basename(path)
            
            # Check cache first
            cached = self._cache.get(path)
            icon = QIcon(cached) if cached else placeholder
            
            item = QListWidgetItem(icon, base_name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            
            self.list_widget.addItem(item)
            self._list_items[path] = item
        
        logger.info("Added %d images to filmstrip", len(self._image_paths))
        
        # Load visible thumbnails after a short delay
        QTimer.singleShot(50, self._load_visible_thumbnails)
    
    def clear_images(self):
        """Clear all images."""
        # Cancel active loaders
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
        self._pending_loads.clear()
        
        self.list_widget.clear()
        self._image_paths.clear()
        self._list_items.clear()
        
        self._emit_selection_change()
        self._emit_checked_items_change()
    
    def _create_placeholder_icon(self) -> QIcon:
        """Create placeholder icon."""
        pixmap = QPixmap(self.thumbnail_size)
        pixmap.fill(QColor('lightgray'))
        return QIcon(pixmap)
    
    def _emit_selection_change(self):
        """Emit selection change signal."""
        selected = self.get_selected_image_paths()
        self.selection_changed.emit(selected)
        
        if len(selected) == 1:
            self.preview_requested.emit(selected[0])
    
    def _handle_item_changed(self, item):
        """Handle item change (checkbox)."""
        if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self._emit_checked_items_change()
    
    def _emit_checked_items_change(self):
        """Emit checked items change signal."""
        checked = self.get_checked_image_paths()
        self.checked_items_changed.emit(checked)
    
    def get_selected_image_paths(self) -> List[str]:
        """Get selected image paths."""
        return [item.data(Qt.ItemDataRole.UserRole) 
                for item in self.list_widget.selectedItems()]
    
    def get_checked_image_paths(self) -> List[str]:
        """Get checked image paths."""
        checked = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked.append(item.data(Qt.ItemDataRole.UserRole))
        return checked
    
    def get_all_image_paths(self) -> List[str]:
        """Get all image paths."""
        return self._image_paths.copy()
    
    def update_thumbnail_size(self):
        """Update thumbnail size from settings."""
        new_size = app_settings.UI_DEFAULTS.get("filmstrip_thumb_size", 120)
        new_size = QSize(new_size, new_size)
        
        if new_size != self.thumbnail_size:
            self.thumbnail_size = new_size
            self.list_widget.setIconSize(self.thumbnail_size)
            
            # Clear cache and reload
            self._cache.clear()
            
            if self._image_paths:
                paths = self._image_paths.copy()
                checked = self.get_checked_image_paths()
                self.add_images(paths)
                
                # Restore checked state
                self.list_widget.blockSignals(True)
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path in checked:
                        item.setCheckState(Qt.CheckState.Checked)
                self.list_widget.blockSignals(False)
                self._emit_checked_items_change()
    
    def prefetch_thumbnails(self, paths: List[str]):
        """Prefetch thumbnails for given paths."""
        for path in paths:
            if not self._cache.contains(path) and path not in self._pending_loads:
                self._start_thumbnail_load(path)
    
    def closeEvent(self, event):
        """Clean up on close."""
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
        super().closeEvent(event)
