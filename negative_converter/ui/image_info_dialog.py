# Image information dialog for displaying EXIF and file metadata
"""
Dialog for displaying image metadata including EXIF data, file info, and ICC profile status.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QWidget, QGroupBox, QFormLayout, QTextEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ImageInfoDialog(QDialog):
    """Dialog displaying image metadata and EXIF information."""
    
    def __init__(self, parent=None, metadata=None, file_path=None, image_shape=None):
        """
        Initialize the image info dialog.
        
        Args:
            parent: Parent widget.
            metadata: ImageMetadata object from image_loader.
            file_path: Path to the current image file.
            image_shape: Tuple of (height, width, channels) for the image.
        """
        super().__init__(parent)
        self.metadata = metadata
        self.file_path = file_path
        self.image_shape = image_shape
        
        self.setWindowTitle("Image Information")
        self.setMinimumSize(500, 400)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        
        # Create tab widget for organized display
        tab_widget = QTabWidget()
        
        # File Info tab
        file_tab = self._create_file_info_tab()
        tab_widget.addTab(file_tab, "File Info")
        
        # EXIF tab
        exif_tab = self._create_exif_tab()
        tab_widget.addTab(exif_tab, "EXIF Data")
        
        # Technical tab
        tech_tab = self._create_technical_tab()
        tab_widget.addTab(tech_tab, "Technical")
        
        layout.addWidget(tab_widget)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
    
    def _create_file_info_tab(self) -> QWidget:
        """Create the file information tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File details group
        file_group = QGroupBox("File Details")
        form = QFormLayout(file_group)
        
        # File path
        if self.file_path:
            import os
            form.addRow("File Name:", QLabel(os.path.basename(self.file_path)))
            form.addRow("Directory:", QLabel(os.path.dirname(self.file_path)))
        else:
            form.addRow("File:", QLabel("No file loaded"))
        
        # File size
        if self.metadata and self.metadata.file_size:
            size = self.metadata.file_size
            if size < 1024:
                size_str = f"{size} bytes"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.2f} MB"
            form.addRow("File Size:", QLabel(size_str))
        
        # Original mode
        if self.metadata and self.metadata.original_mode:
            form.addRow("Color Mode:", QLabel(self.metadata.original_mode))
        
        layout.addWidget(file_group)
        
        # Image dimensions group
        if self.image_shape:
            dim_group = QGroupBox("Image Dimensions")
            dim_form = QFormLayout(dim_group)
            
            h, w = self.image_shape[:2]
            channels = self.image_shape[2] if len(self.image_shape) > 2 else 1
            
            dim_form.addRow("Width:", QLabel(f"{w} pixels"))
            dim_form.addRow("Height:", QLabel(f"{h} pixels"))
            dim_form.addRow("Channels:", QLabel(str(channels)))
            
            megapixels = (w * h) / 1_000_000
            dim_form.addRow("Megapixels:", QLabel(f"{megapixels:.2f} MP"))
            
            # Aspect ratio
            from math import gcd
            divisor = gcd(w, h)
            aspect_w, aspect_h = w // divisor, h // divisor
            # Simplify common ratios
            if aspect_w > 100 or aspect_h > 100:
                ratio = w / h
                if abs(ratio - 1.5) < 0.01:
                    aspect_str = "3:2"
                elif abs(ratio - 1.333) < 0.01:
                    aspect_str = "4:3"
                elif abs(ratio - 1.778) < 0.01:
                    aspect_str = "16:9"
                elif abs(ratio - 1.0) < 0.01:
                    aspect_str = "1:1"
                else:
                    aspect_str = f"{ratio:.3f}:1"
            else:
                aspect_str = f"{aspect_w}:{aspect_h}"
            dim_form.addRow("Aspect Ratio:", QLabel(aspect_str))
            
            layout.addWidget(dim_group)
        
        # ICC Profile group
        icc_group = QGroupBox("Color Profile")
        icc_form = QFormLayout(icc_group)
        
        if self.metadata and self.metadata.has_icc_profile():
            icc_size = len(self.metadata.icc_profile)
            icc_form.addRow("ICC Profile:", QLabel(f"Embedded ({icc_size} bytes)"))
        else:
            icc_form.addRow("ICC Profile:", QLabel("None embedded"))
        
        layout.addWidget(icc_group)
        layout.addStretch()
        
        return widget
    
    def _create_exif_tab(self) -> QWidget:
        """Create the EXIF data tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        if not self.metadata or not self.metadata.exif_dict:
            label = QLabel("No EXIF data available")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)
            return widget
        
        # Create table for EXIF data
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Tag", "Value"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        # Priority tags to show first
        priority_tags = [
            'Make', 'Model', 'DateTime', 'DateTimeOriginal',
            'ExposureTime', 'FNumber', 'ISOSpeedRatings', 'ISO',
            'FocalLength', 'LensModel', 'Software',
            'ImageWidth', 'ImageLength', 'Orientation'
        ]
        
        exif_dict = self.metadata.exif_dict
        
        # Sort: priority tags first, then alphabetically
        sorted_tags = []
        for tag in priority_tags:
            if tag in exif_dict:
                sorted_tags.append(tag)
        for tag in sorted(exif_dict.keys()):
            if tag not in sorted_tags:
                sorted_tags.append(tag)
        
        table.setRowCount(len(sorted_tags))
        
        for row, tag in enumerate(sorted_tags):
            value = exif_dict[tag]
            
            # Format the value nicely
            value_str = self._format_exif_value(tag, value)
            
            tag_item = QTableWidgetItem(tag)
            value_item = QTableWidgetItem(value_str)
            
            # Bold priority tags
            if tag in priority_tags:
                font = tag_item.font()
                font.setBold(True)
                tag_item.setFont(font)
            
            table.setItem(row, 0, tag_item)
            table.setItem(row, 1, value_item)
        
        layout.addWidget(table)
        
        # Show count
        count_label = QLabel(f"{len(exif_dict)} EXIF tags found")
        count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(count_label)
        
        return widget
    
    def _create_technical_tab(self) -> QWidget:
        """Create the technical information tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Memory usage
        mem_group = QGroupBox("Memory Usage")
        mem_form = QFormLayout(mem_group)
        
        if self.image_shape:
            h, w = self.image_shape[:2]
            channels = self.image_shape[2] if len(self.image_shape) > 2 else 1
            
            # Raw memory (uint8)
            raw_bytes = h * w * channels
            raw_mb = raw_bytes / (1024 * 1024)
            mem_form.addRow("Raw Image:", QLabel(f"{raw_mb:.2f} MB"))
            
            # Processing memory (float32, typically 4x)
            proc_mb = raw_mb * 4
            mem_form.addRow("Processing (est.):", QLabel(f"{proc_mb:.2f} MB"))
            
            # With undo buffer
            with_undo_mb = raw_mb * 2
            mem_form.addRow("With Undo Buffer:", QLabel(f"{with_undo_mb:.2f} MB"))
        
        layout.addWidget(mem_group)
        
        # EXIF status
        exif_group = QGroupBox("Metadata Status")
        exif_form = QFormLayout(exif_group)
        
        has_exif = self.metadata and self.metadata.has_exif()
        exif_form.addRow("EXIF Data:", QLabel("✓ Available" if has_exif else "✗ Not available"))
        
        if has_exif:
            exif_bytes = len(self.metadata.exif_data)
            exif_form.addRow("EXIF Size:", QLabel(f"{exif_bytes} bytes"))
        
        has_icc = self.metadata and self.metadata.has_icc_profile()
        exif_form.addRow("ICC Profile:", QLabel("✓ Embedded" if has_icc else "✗ None"))
        
        layout.addWidget(exif_group)
        layout.addStretch()
        
        return widget
    
    def _format_exif_value(self, tag: str, value) -> str:
        """Format an EXIF value for display."""
        if value is None:
            return "N/A"
        
        # Handle nested dicts (like GPS data)
        if isinstance(value, dict):
            parts = [f"{k}: {v}" for k, v in value.items()]
            return "; ".join(parts[:5])  # Limit to 5 items
        
        # Handle tuples/lists
        if isinstance(value, (tuple, list)):
            if len(value) <= 3:
                return ", ".join(str(v) for v in value)
            return f"{value[0]}, {value[1]}, ... ({len(value)} items)"
        
        # Format specific tags nicely
        value_str = str(value)
        
        if tag == 'ExposureTime':
            try:
                if isinstance(value, (int, float)):
                    if value < 1:
                        return f"1/{int(1/value)}s"
                    return f"{value}s"
            except:
                pass
        
        if tag == 'FNumber':
            try:
                return f"f/{float(value):.1f}"
            except:
                pass
        
        if tag == 'FocalLength':
            try:
                return f"{float(value):.1f}mm"
            except:
                pass
        
        if tag in ('ISOSpeedRatings', 'ISO'):
            return f"ISO {value}"
        
        # Truncate very long values
        if len(value_str) > 100:
            return value_str[:97] + "..."
        
        return value_str
