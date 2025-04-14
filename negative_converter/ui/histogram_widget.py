import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Configure PyQtGraph background and foreground
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class HistogramWidget(QWidget):
    """A widget to display image histograms using PyQtGraph."""

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

    @pyqtSlot(object) # Accepts NumPy array
    def update_histogram(self, image):
        """Calculates and updates the histogram plot for the given image."""
        if image is None or image.size == 0:
            self.clear_histogram()
            return

        if image.dtype != np.uint8:
            logger.warning("HistogramWidget expects uint8 image, converting.")
            image = np.clip(image, 0, 255).astype(np.uint8)

        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.warning(f"HistogramWidget expects 3-channel RGB image, got shape {image.shape}. Cannot update.")
            self.clear_histogram()
            return

        try:
            # Calculate histograms for each channel
            # Use a fixed bin count of 256 for 0-255 range
            bins = np.arange(257) # 257 edges for 256 bins
            hist_r, _ = np.histogram(image[..., 0].ravel(), bins=bins)
            hist_g, _ = np.histogram(image[..., 1].ravel(), bins=bins)
            hist_b, _ = np.histogram(image[..., 2].ravel(), bins=bins)

            # Calculate Luminance approximation (simple average for speed)
            # Could use weighted average: 0.299*R + 0.587*G + 0.114*B
            luminance_approx = image.mean(axis=2).astype(np.uint8)
            hist_lum, _ = np.histogram(luminance_approx.ravel(), bins=bins)

            self.current_hist_data = (hist_r, hist_g, hist_b, hist_lum)

            # Update plot data (use bin centers for x-axis)
            bin_centers = bins[:-1] + 0.5
            self.red_curve.setData(x=bin_centers, y=hist_r)
            self.green_curve.setData(x=bin_centers, y=hist_g)
            self.blue_curve.setData(x=bin_centers, y=hist_b)
            self.luminance_curve.setData(x=bin_centers, y=hist_lum)

            # Adjust Y range based on max value across histograms (excluding 0 bin if desired)
            max_val = max(hist_r[1:].max(), hist_g[1:].max(), hist_b[1:].max(), hist_lum[1:].max()) if len(bins) > 2 else 1
            self.plot_widget.setYRange(0, np.log10(max(max_val, 10)) * 1.1) # Log scale range, add padding

        except Exception as e:
            logger.error(f"Error calculating or updating histogram: {e}")
            self.clear_histogram()

    def clear_histogram(self):
        """Clears the histogram plot."""
        self.red_curve.clear()
        self.green_curve.clear()
        self.blue_curve.clear()
        self.luminance_curve.clear()
        self.current_hist_data = None
        self.plot_widget.setYRange(0, 1) # Reset Y range