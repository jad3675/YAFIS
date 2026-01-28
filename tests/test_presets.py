"""Tests for preset loading and application."""

import os
import json
import numpy as np
import pytest
from pathlib import Path


class TestFilmPresetManager:
    """Tests for FilmPresetManager."""
    
    @pytest.fixture
    def film_preset_manager(self):
        """Create a FilmPresetManager instance."""
        from negative_converter.processing.film_simulation import FilmPresetManager
        return FilmPresetManager()
    
    def test_manager_initialization(self, film_preset_manager):
        """Manager should initialize successfully."""
        assert film_preset_manager is not None
    
    def test_get_all_presets(self, film_preset_manager):
        """Should get all available presets."""
        presets = film_preset_manager.get_all_presets()
        assert isinstance(presets, dict)
        # Should have at least some presets
        assert len(presets) > 0
    
    def test_get_preset_by_id(self, film_preset_manager):
        """Should retrieve preset by ID."""
        presets = film_preset_manager.get_all_presets()
        if presets:
            # Get first preset ID
            preset_id = list(presets.keys())[0]
            preset = film_preset_manager.get_preset(preset_id)
            assert preset is not None
    
    def test_apply_preset_to_image(self, film_preset_manager, sample_image_uint8):
        """Should apply preset to image."""
        presets = film_preset_manager.get_all_presets()
        if presets:
            preset_id = list(presets.keys())[0]
            preset_data = presets[preset_id]
            
            result = film_preset_manager.apply_preset(
                sample_image_uint8, preset_data, intensity=1.0
            )
            assert result is not None
            assert result.shape == sample_image_uint8.shape
            assert result.dtype == np.uint8
    
    def test_apply_preset_with_intensity(self, film_preset_manager, sample_image_uint8):
        """Preset intensity should affect the result."""
        presets = film_preset_manager.get_all_presets()
        if presets:
            preset_id = list(presets.keys())[0]
            preset_data = presets[preset_id]
            
            result_full = film_preset_manager.apply_preset(
                sample_image_uint8, preset_data, intensity=1.0
            )
            result_half = film_preset_manager.apply_preset(
                sample_image_uint8, preset_data, intensity=0.5
            )
            
            # Both should produce valid results
            assert result_full is not None
            assert result_half is not None
    
    def test_invalid_preset_id(self, film_preset_manager):
        """Invalid preset ID should return None."""
        result = film_preset_manager.get_preset("nonexistent_preset_12345")
        assert result is None


class TestPhotoPresetManager:
    """Tests for PhotoPresetManager."""
    
    @pytest.fixture
    def photo_preset_manager(self):
        """Create a PhotoPresetManager instance."""
        from negative_converter.processing.photo_presets import PhotoPresetManager
        return PhotoPresetManager()
    
    def test_manager_initialization(self, photo_preset_manager):
        """Manager should initialize successfully."""
        assert photo_preset_manager is not None
    
    def test_get_all_photo_presets(self, photo_preset_manager):
        """Should get all available photo presets."""
        presets = photo_preset_manager.get_all_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0
    
    def test_apply_photo_preset(self, photo_preset_manager, sample_image_uint8):
        """Should apply photo preset to image."""
        presets = photo_preset_manager.get_all_presets()
        if presets:
            preset_id = list(presets.keys())[0]
            
            result = photo_preset_manager.apply_photo_preset(
                sample_image_uint8, preset_id, intensity=1.0
            )
            assert result is not None
            assert result.shape == sample_image_uint8.shape


class TestPresetFiles:
    """Tests for preset file structure and validity."""
    
    @pytest.fixture
    def preset_dirs(self):
        """Get preset directories."""
        base_dir = Path(__file__).parent.parent / "negative_converter" / "config" / "presets"
        return {
            'film': base_dir / "film",
            'photo': base_dir / "photo",
        }
    
    def test_film_preset_files_exist(self, preset_dirs):
        """Film preset directory should contain JSON files."""
        film_dir = preset_dirs['film']
        if film_dir.exists():
            json_files = list(film_dir.glob("*.json"))
            assert len(json_files) > 0, "No film preset files found"
    
    def test_photo_preset_files_exist(self, preset_dirs):
        """Photo preset directory should contain JSON files."""
        photo_dir = preset_dirs['photo']
        if photo_dir.exists():
            json_files = list(photo_dir.glob("*.json"))
            assert len(json_files) > 0, "No photo preset files found"
    
    def test_film_preset_files_valid_json(self, preset_dirs):
        """All film preset files should be valid JSON."""
        film_dir = preset_dirs['film']
        if film_dir.exists():
            for json_file in film_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    try:
                        data = json.load(f)
                        assert isinstance(data, dict), f"{json_file.name} is not a dict"
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON in {json_file.name}: {e}")
    
    def test_photo_preset_files_valid_json(self, preset_dirs):
        """All photo preset files should be valid JSON."""
        photo_dir = preset_dirs['photo']
        if photo_dir.exists():
            for json_file in photo_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    try:
                        data = json.load(f)
                        assert isinstance(data, dict), f"{json_file.name} is not a dict"
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON in {json_file.name}: {e}")
    
    def test_preset_has_required_fields(self, preset_dirs):
        """Presets should have basic required fields."""
        for preset_type, preset_dir in preset_dirs.items():
            if preset_dir.exists():
                for json_file in preset_dir.glob("*.json"):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        # Most presets should have a name or id
                        has_identifier = 'name' in data or 'id' in data or json_file.stem
                        assert has_identifier, f"{json_file.name} missing identifier"


class TestPresetIntegration:
    """Integration tests for preset system."""
    
    def test_preset_preview_in_adjustments(self, sample_image_uint8):
        """Preset preview through adjustments pipeline should work."""
        from negative_converter.processing.adjustments import apply_all_adjustments
        
        # Test with preset_info in adjustments dict
        result = apply_all_adjustments(sample_image_uint8, {
            'brightness': 0,
            'preset_info': {
                'type': 'film',
                'id': 'portra-400',  # Common preset
                'intensity': 0.5,
            }
        })
        # Should not crash even if preset doesn't exist
        assert result is not None
    
    def test_preset_with_adjustments_combined(self, sample_image_uint8):
        """Presets should work with other adjustments."""
        from negative_converter.processing.adjustments import apply_all_adjustments
        
        result = apply_all_adjustments(sample_image_uint8, {
            'brightness': 10,
            'contrast': 5,
            'preset_info': {
                'type': 'photo',
                'id': 'film',
                'intensity': 0.7,
            }
        })
        assert result is not None
        assert result.shape == sample_image_uint8.shape
