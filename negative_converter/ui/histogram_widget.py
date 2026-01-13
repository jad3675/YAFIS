import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import QTimer, pyqtSlot
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Configure PyQtGraph background and foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class HistogramWidget(QWidget):
    """A widget to display image histograms using PyQtGraph."""

    # Keep histogram fast: compute from a proxy image and debounce updates
    _DEFAULT_PROXY_MAX_DIM = 1000
    _DEFAULT_DEBOUNCE_MS = 150

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget(title="Histogram")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', 'Count')
        self.plot_widget.setLabel('bottom', 'Pixel Intensity')
        self.plot_widget.setLogMode(y=True) # Use log scale for better visibility
        self.plot_widget.setXRange(0, 255, padding=0)
        self.plot_widget.setLimits(xMin=0, xMax=255)

        # Store plot items for updating
        self.red_curve = self.plot_widget.plot(pen='r', name='Red')
        self.green_curve = self.plot_widget.plot(pen='g', name='Green')
        self.blue_curve = self.plot_widget.plot(pen='b', name='Blue')
        self.luminance_curve = self.plot_widget.plot(pen=pg.mkPen('k', width=2), name='Luminance')

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)

        self.current_hist_data = None # Cache histogram data

        # Debounce histogram updates (coalesce rapid changes into one computation)
        self._pending_image = None
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.timeout.connect(self._apply_pending_histogram)

        # Settings (no UI yet; keep constants minimal/YAGNI)
        self._proxy_max_dim = self._DEFAULT_PROXY_MAX_DIM
        self._debounce_ms = self._DEFAULT_DEBOUNCE_MS

    @pyqtSlot(object) # Accepts NumPy array
    def update_histogram(self, image):
        """
        Request a histogram update for the given image.

        This method is intentionally debounced to avoid repeated full computations
        during rapid UI updates (slider drags, multi-stage renders, etc.).
        """
        if image is None or getattr(image, "size", 0) == 0:
            self._pending_image = None
            self._hist_timer.stop()
            self.clear_histogram()
            return

        # Keep only latest request; compute once per debounce window
        self._pending_image = image
        self._hist_timer.start(self._debounce_ms)

    @pyqtSlot()
    def _apply_pending_histogram(self):
        """Compute and plot histogram for the latest pending image."""
        image = self._pending_image
        self._pending_image = None

        if image is None or getattr(image, "size", 0) == 0:
            self.clear_histogram()
            return

        # Ensure uint8 RGB
        if image.dtype != np.uint8:
            logger.warning("HistogramWidget expects uint8 image, converting.")
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning("HistogramWidget expects 3-channel RGB image, got shape %s. Cannot update.", getattr(image, "shape", None))
            self.clear_histogram()
            return

        try:
            proxy = self._make_histogram_proxy(image, max_dim=self._proxy_max_dim)

            # Calculate histograms for each channel (fixed 256 bins for 0..255)
            bins = np.arange(257) # 257 edges for 256 bins
            hist_r, _ = np.histogram(proxy[..., 0].ravel(), bins=bins)
            hist_g, _ = np.histogram(proxy[..., 1].ravel(), bins=bins)
            hist_b, _ = np.histogram(proxy[..., 2].ravel(), bins=bins)

            # Luminance approximation (simple average for speed)
            luminance_approx = proxy.mean(axis=2).astype(np.uint8)
            hist_lum, _ = np.histogram(luminance_approx.ravel(), bins=bins)

            self.current_hist_data = (hist_r, hist_g, hist_b, hist_lum)

            # Update plot data (use bin centers for x-axis)
            bin_centers = bins[:-1] + 0.5
            self.red_curve.setData(x=bin_centers, y=hist_r)
            self.green_curve.setData(x=bin_centers, y=hist_g)
            self.blue_curve.setData(x=bin_centers, y=hist_b)
            self.luminance_curve.setData(x=bin_centers, y=hist_lum)

            # Adjust Y range based on max value across histograms (excluding 0 bin)
            max_val = max(hist_r[1:].max(), hist_g[1:].max(), hist_b[1:].max(), hist_lum[1:].max()) if len(bins) > 2 else 1
            self.plot_widget.setYRange(0, np.log10(max(max_val, 10)) * 1.1) # Log scale range, add padding

        except Exception:
            logger.exception("Error calculating or updating histogram")
            self.clear_histogram()

    @staticmethod
    def _make_histogram_proxy(image: np.ndarray, max_dim: int) -> np.ndarray:
        """
        Create a smaller proxy image for histogram calculation using simple striding.

        This avoids extra dependencies and stays very fast.
        """
        if max_dim <= 0:
            return image

        h, w = image.shape[:2]
        if h <= max_dim and w <= max_dim:
            return image

        step = int(max(1, np.ceil(max(h, w) / float(max_dim))))
        return image[::step, ::step, :]

    def clear_histogram(self):
        """Clears the histogram plot."""
        self._pending_image = None
        self._hist_timer.stop()

        self.red_curve.clear()
        self.green_curve.clear()
        self.blue_curve.clear()
        self.luminance_curve.clear()
        self.current_hist_data = None
        self.plot_widget.setYRange(0, 1) # Reset Y range