# Filmstrip widget for batch processing
import os
import cv2 # For image loading and resizing
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QListWidget, QListView,
                             QListWidgetItem, QAbstractItemView, QStyle, QApplication) # Added QApplication for testing
from PyQt6.QtGui import QIcon, QPixmap, QImage, QPainter, QColor
from PyQt6.QtCore import (QSize, Qt, QThread, QObject, pyqtSignal, pyqtSlot,
                          QMutex, QMutexLocker, QMetaObject, QRunnable, QThreadPool)
from ..config import settings as app_settings # Import settings

# --- Thumbnail Loading Worker ---

class ThumbnailLoader(QObject):
    """Worker to load and resize an image in a background thread."""
    thumbnail_ready = pyqtSignal(str, QPixmap) # path, pixmap
    error = pyqtSignal(str, str) # path, error_message

    def __init__(self, file_path, target_size, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.target_size = target_size
        self._is_cancelled = False
        self._mutex = QMutex()

    @pyqtSlot()
    def run(self):
        """Load, resize, and emit the thumbnail."""
        try:
            with QMutexLocker(self._mutex):
                if self._is_cancelled:
                    return

            # Load image using OpenCV (handles various formats)
            img = cv2.imread(self.file_path)
            if img is None:
                raise ValueError("cv2.imread returned None")

            # Resize while maintaining aspect ratio
            h, w = img.shape[:2]
            target_w, target_h = self.target_size.width(), self.target_size.height()
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Check cancellation again before potentially slow resize
            with QMutexLocker(self._mutex):
                if self._is_cancelled:
                    return

            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Convert OpenCV image (BGR) to QPixmap (RGB)
            if len(resized_img.shape) == 3: # Color image
                h, w, ch = resized_img.shape
                bytes_per_line = ch * w
                q_img = QImage(resized_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).rgbSwapped()
            elif len(resized_img.shape) == 2: # Grayscale image
                 h, w = resized_img.shape
                 bytes_per_line = w
                 q_img = QImage(resized_img.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                 raise ValueError("Unsupported image shape")

            pixmap = QPixmap.fromImage(q_img)

            # Check cancellation one last time before emitting
            with QMutexLocker(self._mutex):
                if self._is_cancelled:
                    return

            self.thumbnail_ready.emit(self.file_path, pixmap)

        except Exception as e:
            self.error.emit(self.file_path, f"Error loading thumbnail: {e}")

    def cancel(self):
        with QMutexLocker(self._mutex):
            self._is_cancelled = True

# --- Helper Runnable for QThreadPool ---

class WorkerRunnable(QRunnable):
    """Takes a QObject worker instance and calls its run method."""
    def __init__(self, worker_instance):
        super().__init__()
        self.worker = worker_instance
        self.setAutoDelete(True) # Auto-delete this runnable when done

    @pyqtSlot()
    def run(self):
        """Execute the worker's run method."""
        # Note: The worker's signals will be emitted from this thread pool thread.
        # Ensure slots connected to these signals are thread-safe or use QueuedConnection.
        self.worker.run()


class BatchFilmstripWidget(QWidget):
    """
    A widget to display thumbnails of images selected for batch processing.
    """
    selection_changed = pyqtSignal(list) # Emits list of *selected* file paths (for preview)
    preview_requested = pyqtSignal(str) # Emits path of single selected item for preview
    checked_items_changed = pyqtSignal(list) # Emits list of *checked* file paths (for batch)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Fetch initial size from settings
        initial_size = app_settings.UI_DEFAULTS.get("filmstrip_thumb_size", 120) # Default 120 if not found
        self.thumbnail_size = QSize(initial_size, initial_size) # Store target size (assuming square)
        self._image_paths = [] # Store full paths of added images
        self._list_items = {} # Map path to QListWidgetItem for easy update
        # Use global thread pool instead of a dedicated thread
        self._thread_pool = QThreadPool.globalInstance()
        # Optional: Limit max threads if needed, though global pool often sizes well
        # print(f"Global Thread Pool Max Threads: {self._thread_pool.maxThreadCount()}")
        self._active_loaders = {} # Keep track of active loaders path -> worker
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize UI elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # No external margins

        self.list_widget = QListWidget(self)
        self.list_widget.setViewMode(QListView.ViewMode.IconMode)
        self.list_widget.setResizeMode(QListView.ResizeMode.Adjust) # Adjust layout on resize
        self.list_widget.setMovement(QListView.Movement.Static) # Items are not movable
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection) # Allow multi-select
        self.list_widget.setSpacing(10) # Increased spacing
        self.list_widget.setIconSize(self.thumbnail_size) # Use stored size
        self.list_widget.setWordWrap(True) # Wrap text if needed

        layout.addWidget(self.list_widget)
        self.setLayout(layout)

    def _connect_signals(self):
        """Connect internal signals."""
        self.list_widget.itemSelectionChanged.connect(self._emit_selection_change) # For preview
        self.list_widget.itemChanged.connect(self._handle_item_changed) # For checkboxes

    def _emit_selection_change(self):
        """Emit selection change and preview request if applicable."""
        selected_paths = self.get_selected_image_paths()
        self.selection_changed.emit(selected_paths)

        # Emit preview request only if exactly one item is selected
        if len(selected_paths) == 1:
            self.preview_requested.emit(selected_paths[0])
        # else: # Optionally clear preview if multiple/none selected
            # self.preview_requested.emit(None) # Or handle in MainWindow

    def _handle_item_changed(self, item):
        """Handle changes to an item, specifically its check state."""
        # Check if the change was the check state
        # This check might not be strictly necessary if only check state changes,
        # but good practice if other item flags could change.
        if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
            self._emit_checked_items_change()

    def _emit_checked_items_change(self):
        """Emit the list of paths for currently checked items."""
        checked_paths = self.get_checked_image_paths()
        self.checked_items_changed.emit(checked_paths)

    def add_images(self, file_paths):
        """
        Add multiple images to the filmstrip, loading thumbnails in the background.
        """
        self.clear_images() # Clear previous batch first
        self._image_paths = sorted(list(set(file_paths))) # Store unique, sorted paths

        placeholder_icon = self._create_placeholder_icon(self.thumbnail_size)

        for path in self._image_paths:
            base_name = os.path.basename(path)
            item = QListWidgetItem(placeholder_icon, base_name)
            item.setData(Qt.ItemDataRole.UserRole, path) # Store full path
            item.setToolTip(path)
            # --- Add Checkbox ---
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable) # Make checkable
            item.setCheckState(Qt.CheckState.Unchecked) # Start unchecked
            # --- End Checkbox ---
            self.list_widget.addItem(item)
            self._list_items[path] = item # Store item reference

            # Start loading thumbnail
            self._load_thumbnail(path)

        print(f"Added {len(self._image_paths)} images to filmstrip. Loading thumbnails...")

    def _load_thumbnail(self, file_path):
        """Initiate loading for a single thumbnail."""
        # Cancel previous loader for this path if any
        if file_path in self._active_loaders:
             self._active_loaders[file_path].cancel()
             # Consider waiting for thread to finish if necessary, but often moveToThread handles cleanup

        loader = ThumbnailLoader(file_path, self.thumbnail_size) # Loader is QObject
        # Connect signals directly - slots (_update_thumbnail, _handle_thumbnail_error)
        # need to be thread-safe or use QueuedConnection implicitly (default for cross-thread).
        # PyQt's default connection type for cross-thread signals/slots is QueuedConnection,
        # which is safe for updating GUI elements from worker threads.
        loader.thumbnail_ready.connect(self._update_thumbnail)
        loader.error.connect(self._handle_thumbnail_error)

        self._active_loaders[file_path] = loader
        runnable = WorkerRunnable(loader) # Wrap loader in QRunnable
        self._thread_pool.start(runnable) # Submit runnable to the pool

    @pyqtSlot(str, QPixmap)
    def _update_thumbnail(self, path, pixmap):
        """Update the icon of the corresponding list item when thumbnail is ready."""
        if path in self._list_items:
            item = self._list_items[path]
            item.setIcon(QIcon(pixmap))
        if path in self._active_loaders:
             del self._active_loaders[path] # Remove loader reference

    @pyqtSlot(str, str)
    def _handle_thumbnail_error(self, path, error_message):
        """Handle errors during thumbnail loading."""
        print(f"Error loading thumbnail for {os.path.basename(path)}: {error_message}")
        if path in self._list_items:
             # Optionally set an error icon
             error_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
             self._list_items[path].setIcon(error_icon)
        if path in self._active_loaders:
             del self._active_loaders[path] # Remove loader reference

    def clear_images(self):
        """Clear all images and stop any active thumbnail loaders."""
        # Cancel and remove active loaders
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()

        self.list_widget.clear()
        self._image_paths = []
        self._list_items = {}
        self._emit_selection_change() # Emit empty list for selection
        self._emit_checked_items_change() # Emit empty list for checked items
        print("Filmstrip cleared.")

    def get_selected_image_paths(self):
        """Return a list of full paths for the selected images."""
        selected_items = self.list_widget.selectedItems()
        return [item.data(Qt.ItemDataRole.UserRole) for item in selected_items]

    def get_checked_image_paths(self):
        """Return a list of full paths for the checked images."""
        checked_paths = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                checked_paths.append(item.data(Qt.ItemDataRole.UserRole))
        return checked_paths

    def get_all_image_paths(self):
        """Return a list of full paths for all images in the filmstrip."""
        return self._image_paths[:] # Return a copy

    def _create_placeholder_icon(self, size):
        """Creates a simple gray placeholder icon."""
        pixmap = QPixmap(size)
        pixmap.fill(QColor('lightgray'))
        # Optional: Draw text or symbol
        # painter = QPainter(pixmap)
        # painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "...")
        # painter.end()
        return QIcon(pixmap)

    def closeEvent(self, event):
        """Clean up thumbnail loading on widget close."""
        print("Cancelling active thumbnail loaders...")
        # Cancel any loaders still running
        for loader in self._active_loaders.values():
            loader.cancel()
        self._active_loaders.clear()
        # Optional: Wait for thread pool tasks to finish, but might hang UI.
        # Cancellation should be enough for loaders to exit quickly.
        # self._thread_pool.waitForDone(3000) # Use with caution
        print("Thumbnail loaders cancelled.")
        super().closeEvent(event)

    def update_thumbnail_size(self):
        """Fetches the latest thumbnail size from settings and updates the view."""
        new_size_val = app_settings.UI_DEFAULTS.get("filmstrip_thumb_size", 120)
        new_size = QSize(new_size_val, new_size_val)

        if new_size != self.thumbnail_size:
            print(f"Updating filmstrip thumbnail size to {new_size_val}x{new_size_val}...")
            self.thumbnail_size = new_size
            self.list_widget.setIconSize(self.thumbnail_size)

            # Easiest way to apply new size is to reload all images
            # This will cancel existing loaders and start new ones with the correct size.
            current_paths = self.get_all_image_paths()
            if current_paths:
                print("Reloading filmstrip images to apply new thumbnail size...")
                # Store checked state before clearing
                checked_before_reload = self.get_checked_image_paths()
                self.add_images(current_paths) # This calls clear_images first
                # Restore checked state
                items_to_recheck = []
                for i in range(self.list_widget.count()):
                    item = self.list_widget.item(i)
                    path = item.data(Qt.ItemDataRole.UserRole)
                    if path in checked_before_reload:
                        items_to_recheck.append(item)

                # Block signals temporarily while re-checking to avoid multiple emissions
                self.list_widget.blockSignals(True)
                for item in items_to_recheck:
                    item.setCheckState(Qt.CheckState.Checked)
                self.list_widget.blockSignals(False)
                # Manually emit the checked items change once after restoring
                self._emit_checked_items_change()

            print("Filmstrip thumbnail size update complete.")

# Example usage (for testing)
# Example usage (for testing - requires QApplication)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = QWidget()
    layout = QVBoxLayout(window)
    filmstrip = BatchFilmstripWidget()
    layout.addWidget(filmstrip)

    # Example files (replace with actual paths if testing)
    test_files = [f"image_{i:02d}.jpg" for i in range(15)]
    filmstrip.add_images(test_files)

    def on_selection(paths):
        print("Selected paths:", paths)

    filmstrip.selection_changed.connect(on_selection)

    window.setWindowTitle("Filmstrip Test")
    window.setGeometry(200, 200, 800, 200)
    window.show()
    sys.exit(app.exec())