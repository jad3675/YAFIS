# Color sampler dialog for displaying sampled colors
"""
Dialog for displaying color samples and analysis.
"""

from typing import Optional, List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QFormLayout, QFrame, QWidget, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QFont

from ..utils.color_sampler import (
    ColorSample, ColorSamplerState,
    analyze_skin_tone, check_neutral
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ColorSwatchWidget(QFrame):
    """Widget displaying a color swatch."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(60, 60)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self._color = QColor(128, 128, 128)
    
    def set_color(self, r: int, g: int, b: int) -> None:
        """Set the displayed color."""
        self._color = QColor(r, g, b)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, self._color)
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        self.update()


class ColorSampleWidget(QWidget):
    """Widget displaying a single color sample with all color space values."""
    
    removed = pyqtSignal(int)  # Emits sample index
    
    def __init__(self, sample: ColorSample, index: int, parent=None):
        super().__init__(parent)
        self._sample = sample
        self._index = index
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Color swatch
        self.swatch = ColorSwatchWidget()
        self.swatch.set_color(*self._sample.rgb)
        layout.addWidget(self.swatch)
        
        # Color values
        values_layout = QVBoxLayout()
        
        # Position
        pos_label = QLabel(f"Position: ({self._sample.x}, {self._sample.y})")
        pos_label.setStyleSheet("font-size: 10px; color: gray;")
        values_layout.addWidget(pos_label)
        
        # RGB
        rgb_label = QLabel(self._sample.format_rgb())
        rgb_label.setFont(QFont("Monospace", 9))
        values_layout.addWidget(rgb_label)
        
        # Hex
        hex_label = QLabel(f"Hex: {self._sample.hex}")
        hex_label.setFont(QFont("Monospace", 9))
        values_layout.addWidget(hex_label)
        
        # HSV
        hsv_label = QLabel(self._sample.format_hsv())
        hsv_label.setFont(QFont("Monospace", 9))
        values_layout.addWidget(hsv_label)
        
        # LAB
        lab_label = QLabel(self._sample.format_lab())
        lab_label.setFont(QFont("Monospace", 9))
        values_layout.addWidget(lab_label)
        
        layout.addLayout(values_layout)
        layout.addStretch()
        
        # Remove button
        remove_btn = QPushButton("×")
        remove_btn.setFixedSize(20, 20)
        remove_btn.setToolTip("Remove this sample")
        remove_btn.clicked.connect(lambda: self.removed.emit(self._index))
        layout.addWidget(remove_btn)


class ColorSamplerDialog(QDialog):
    """Dialog for displaying and analyzing color samples."""
    
    sample_requested = pyqtSignal()  # Request to sample a new color
    apply_wb_correction = pyqtSignal(int, int)  # Emit temp, tint adjustments
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._sampler_state = ColorSamplerState(max_samples=5)
        self._sample_widgets: List[ColorSampleWidget] = []
        self._current_suggestion = None  # Store current WB suggestion
        
        self.setWindowTitle("Color Sampler")
        self.setMinimumSize(450, 450)
        self.resize(480, 520)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Sample button
        sample_btn = QPushButton("Sample Color from Image")
        sample_btn.setToolTip("Click on the image to sample a color (shortcut: S)")
        sample_btn.clicked.connect(self.sample_requested.emit)
        layout.addWidget(sample_btn)
        
        # Samples container
        self.samples_group = QGroupBox("Sampled Colors")
        self.samples_layout = QVBoxLayout(self.samples_group)
        self.samples_layout.setSpacing(5)
        
        self.no_samples_label = QLabel("No colors sampled yet.\nClick 'Sample Color' then click on the image.")
        self.no_samples_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_samples_label.setStyleSheet("color: gray;")
        self.samples_layout.addWidget(self.no_samples_label)
        
        layout.addWidget(self.samples_group)
        
        # Analysis group
        self.analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout(self.analysis_group)
        
        self.analysis_label = QLabel("Select a sample to see analysis")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        
        # Apply suggestion button (hidden by default)
        self.apply_suggestion_btn = QPushButton("Apply White Balance Correction")
        self.apply_suggestion_btn.setToolTip("Apply the suggested white balance adjustment")
        self.apply_suggestion_btn.setStyleSheet("background-color: #4a90d9; color: white; font-weight: bold; padding: 8px;")
        self.apply_suggestion_btn.clicked.connect(self._apply_suggestion)
        self.apply_suggestion_btn.setVisible(False)
        analysis_layout.addWidget(self.apply_suggestion_btn)
        
        layout.addWidget(self.analysis_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_samples)
        button_layout.addWidget(clear_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def add_sample(self, sample: ColorSample) -> None:
        """Add a new color sample."""
        self._sampler_state.add_sample(sample)
        self._rebuild_sample_widgets()
        self._update_analysis(sample)
    
    def _rebuild_sample_widgets(self) -> None:
        """Rebuild the sample widgets list."""
        # Clear existing widgets
        for widget in self._sample_widgets:
            widget.setParent(None)
            widget.deleteLater()
        self._sample_widgets.clear()
        
        # Hide/show no samples label
        has_samples = len(self._sampler_state.samples) > 0
        self.no_samples_label.setVisible(not has_samples)
        
        # Create new widgets
        for i, sample in enumerate(self._sampler_state.samples):
            widget = ColorSampleWidget(sample, i)
            widget.removed.connect(self._remove_sample)
            self._sample_widgets.append(widget)
            self.samples_layout.addWidget(widget)
    
    def _remove_sample(self, index: int) -> None:
        """Remove a sample by index."""
        self._sampler_state.remove_sample(index)
        self._rebuild_sample_widgets()
        
        # Update analysis with last sample if any
        if self._sampler_state.samples:
            self._update_analysis(self._sampler_state.samples[-1])
        else:
            self.analysis_label.setText("Select a sample to see analysis")
    
    def _clear_samples(self) -> None:
        """Clear all samples."""
        self._sampler_state.clear()
        self._rebuild_sample_widgets()
        self.analysis_label.setText("Select a sample to see analysis")
    
    def _update_analysis(self, sample: ColorSample) -> None:
        """Update the analysis section for a sample."""
        lines = []
        self._current_suggestion = None
        
        # Luminance
        lum = sample.luminance
        lines.append(f"Relative Luminance: {lum:.3f}")
        if lum < 0.2:
            lines.append("  → Dark tone (shadow region)")
        elif lum > 0.8:
            lines.append("  → Bright tone (highlight region)")
        else:
            lines.append("  → Midtone region")
        
        # Neutral check
        neutral = check_neutral(sample)
        if neutral["is_neutral"]:
            lines.append("\n✓ This appears to be a neutral gray")
            self.apply_suggestion_btn.setVisible(False)
        else:
            lines.append(f"\n⚠ Not neutral (max channel diff: {neutral['max_channel_diff']})")
            if neutral["wb_suggestion"]:
                lines.append(f"  Suggestion: {neutral['wb_suggestion']}")
                
                # Calculate the WB correction values
                r, g, b = sample.rgb
                temp_diff = b - r
                tint_diff = g - (r + b) / 2.0
                
                # Calculate slider values to neutralize
                temp_val = int(round(max(-100, min(100, -temp_diff / 0.6))))
                tint_val = int(round(max(-100, min(100, -tint_diff / 0.3))))
                
                self._current_suggestion = (temp_val, tint_val)
                self.apply_suggestion_btn.setText(f"Apply WB Correction (Temp: {temp_val:+d}, Tint: {tint_val:+d})")
                self.apply_suggestion_btn.setVisible(True)
            else:
                self.apply_suggestion_btn.setVisible(False)
        
        # Skin tone check
        skin = analyze_skin_tone(sample)
        if skin["is_skin_tone"]:
            lines.append(f"\n🎨 Possible skin tone detected")
            if skin["quality"] == "good":
                lines.append("  → Skin tone looks well balanced")
            else:
                for suggestion in skin["suggestions"]:
                    lines.append(f"  → {suggestion}")
        
        self.analysis_label.setText("\n".join(lines))
    
    def _apply_suggestion(self):
        """Apply the current WB suggestion."""
        if self._current_suggestion:
            temp, tint = self._current_suggestion
            self.apply_wb_correction.emit(temp, tint)


class ColorSamplerPanel(QWidget):
    """
    Compact panel version of color sampler for docking.
    """
    sample_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_sample: Optional[ColorSample] = None
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Sample button
        sample_btn = QPushButton("Sample (S)")
        sample_btn.setToolTip("Click to sample color from image")
        sample_btn.clicked.connect(self.sample_requested.emit)
        layout.addWidget(sample_btn)
        
        # Current sample display
        sample_layout = QHBoxLayout()
        
        self.swatch = ColorSwatchWidget()
        self.swatch.setMinimumSize(40, 40)
        self.swatch.setMaximumSize(40, 40)
        sample_layout.addWidget(self.swatch)
        
        self.values_label = QLabel("No sample")
        self.values_label.setFont(QFont("Monospace", 8))
        self.values_label.setWordWrap(True)
        sample_layout.addWidget(self.values_label)
        
        layout.addLayout(sample_layout)
        
        # Analysis
        self.analysis_label = QLabel("")
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self.analysis_label)
        
        layout.addStretch()
    
    def set_sample(self, sample: ColorSample) -> None:
        """Set the current sample to display."""
        self._current_sample = sample
        
        # Update swatch
        self.swatch.set_color(*sample.rgb)
        
        # Update values
        lines = [
            sample.format_rgb(),
            f"Hex: {sample.hex}",
            sample.format_hsv(),
        ]
        self.values_label.setText("\n".join(lines))
        
        # Update analysis
        neutral = check_neutral(sample)
        if neutral["is_neutral"]:
            self.analysis_label.setText("✓ Neutral gray")
        elif neutral["wb_suggestion"]:
            self.analysis_label.setText(f"WB: {neutral['wb_suggestion']}")
        else:
            self.analysis_label.setText(f"Lum: {sample.luminance:.2f}")
    
    def clear_sample(self) -> None:
        """Clear the current sample."""
        self._current_sample = None
        self.swatch.set_color(128, 128, 128)
        self.values_label.setText("No sample")
        self.analysis_label.setText("")
