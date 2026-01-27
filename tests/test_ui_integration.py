"""
Integration tests for UI components.

These tests verify that UI components can be instantiated and
interact correctly with the backend processing modules.
"""

import pytest
import numpy as np

# Skip all tests if PyQt6 is not available or display is not available
pytest.importorskip("PyQt6")


@pytest.fixture
def qapp():
    """Create a QApplication for UI tests."""
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestSettingsDialog:
    """Integration tests for SettingsDialog."""
    
    def test_dialog_creation(self, qapp):
        """Settings dialog should be creatable."""
        from negative_converter.ui.settings_dialog import SettingsDialog
        
        dialog = SettingsDialog()
        assert dialog is not None
        assert dialog.windowTitle() == "Application Settings"
    
    def test_dialog_loads_settings(self, qapp):
        """Dialog should load current settings into widgets."""
        from negative_converter.ui.settings_dialog import SettingsDialog
        from negative_converter.config.settings import CONVERSION_DEFAULTS
        
        dialog = SettingsDialog()
        
        # Check that some settings were loaded
        assert dialog.mask_sample_size.value() == CONVERSION_DEFAULTS.get('mask_sample_size', 10)
        assert dialog.wb_target_gray.value() == CONVERSION_DEFAULTS.get('wb_target_gray', 128.0)
    
    def test_collapsible_sections(self, qapp):
        """Collapsible sections should toggle correctly."""
        from negative_converter.ui.settings_dialog import CollapsibleSection
        
        section = CollapsibleSection("Test Section", initially_collapsed=True)
        
        # Initially collapsed
        assert section._is_collapsed
        
        # Expand
        section.expand()
        assert not section._is_collapsed
        
        # Collapse
        section.collapse()
        assert section._is_collapsed


class TestConverterIntegration:
    """Integration tests for converter with UI workflows."""
    
    def test_converter_with_progress_callback(self, sample_image_uint8):
        """Converter should work with progress callbacks."""
        from negative_converter.processing.converter import NegativeConverter
        
        progress_steps = []
        
        def progress_callback(current, total):
            progress_steps.append((current, total))
        
        converter = NegativeConverter()
        result, classification = converter.convert(
            sample_image_uint8,
            progress_callback=progress_callback
        )
        
        assert result is not None
        assert len(progress_steps) > 0
        # Should have reported progress for all steps
        assert progress_steps[-1][0] == progress_steps[-1][1]  # Final step
    
    def test_converter_override_classification(self, sample_image_uint8):
        """Converter should respect classification override."""
        from negative_converter.processing.converter import NegativeConverter
        
        converter = NegativeConverter()
        
        # Force E-6 classification
        result, classification = converter.convert(
            sample_image_uint8,
            override_mask_classification="Likely E-6"
        )
        
        assert classification == "Likely E-6"
        assert result is not None


class TestFilmProfileIntegration:
    """Integration tests for film profiles."""
    
    def test_all_profiles_loadable(self):
        """All defined film profiles should be loadable."""
        from negative_converter.processing.converter import NegativeConverter
        
        profiles = ["C41", "BW", "E6", "ECN2"]
        
        for profile in profiles:
            converter = NegativeConverter(film_profile=profile)
            assert converter.profile_data is not None
            assert "gamma" in converter.profile_data
            assert "lab_correction" in converter.profile_data
    
    def test_profile_affects_conversion(self, sample_image_uint8):
        """Different profiles should produce different results."""
        from negative_converter.processing.converter import NegativeConverter
        
        converter_c41 = NegativeConverter(film_profile="C41")
        converter_bw = NegativeConverter(film_profile="BW")
        
        result_c41, _ = converter_c41.convert(sample_image_uint8)
        result_bw, _ = converter_bw.convert(sample_image_uint8)
        
        # Results should be different (B&W has no saturation boost)
        assert not np.array_equal(result_c41, result_bw)


class TestMaskDetectionIntegration:
    """Integration tests for mask detection with converter."""
    
    def test_detection_result_used_in_conversion(self):
        """Mask detection result should influence conversion."""
        from negative_converter.processing.converter import NegativeConverter
        from negative_converter.processing.mask_detection import FilmBaseDetector
        
        # Create C-41 like image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        orange = [200, 120, 80]
        img[:10, :10] = orange
        img[:10, -10:] = orange
        img[-10:, :10] = orange
        img[-10:, -10:] = orange
        img[20:80, 20:80] = [100, 80, 60]  # Some content
        
        # Detect first
        detector = FilmBaseDetector()
        detection = detector.detect(img)
        
        # Convert
        converter = NegativeConverter()
        result, classification = converter.convert(img)
        
        # Classification should match detection
        assert classification == detection.classification
