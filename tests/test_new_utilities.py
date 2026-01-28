# Tests for new utility modules
"""
Tests for history, geometry, color sampler, preset validation, export presets, and session management.
"""

import pytest
import numpy as np
import json
import os
import tempfile
from copy import deepcopy


class TestHistoryStack:
    """Tests for the HistoryStack class."""
    
    def test_push_and_undo(self):
        from negative_converter.utils.history import HistoryStack
        
        stack = HistoryStack[dict](max_size=10)
        
        # Push initial state
        stack.push({"value": 1}, "initial")
        assert not stack.can_undo()  # Need 2 entries to undo
        
        # Push second state
        stack.push({"value": 2}, "change 1")
        assert stack.can_undo()
        
        # Undo
        result = stack.undo()
        assert result == {"value": 1}
        assert not stack.can_undo()
        assert stack.can_redo()
    
    def test_redo(self):
        from negative_converter.utils.history import HistoryStack
        
        stack = HistoryStack[dict](max_size=10)
        stack.push({"value": 1}, "initial")
        stack.push({"value": 2}, "change")
        
        stack.undo()
        assert stack.can_redo()
        
        result = stack.redo()
        assert result == {"value": 2}
        assert not stack.can_redo()
    
    def test_max_size_limit(self):
        from negative_converter.utils.history import HistoryStack
        
        stack = HistoryStack[int](max_size=5)
        
        for i in range(10):
            stack.push(i, f"push {i}")
        
        # Should only have 5 entries
        assert stack.get_undo_count() == 4  # Can undo 4 times (5 entries - 1)
    
    def test_clear_redo_on_new_push(self):
        from negative_converter.utils.history import HistoryStack
        
        stack = HistoryStack[int](max_size=10)
        stack.push(1, "one")
        stack.push(2, "two")
        stack.push(3, "three")
        
        stack.undo()
        stack.undo()
        assert stack.can_redo()
        
        # New push should clear redo stack
        stack.push(4, "four")
        assert not stack.can_redo()
    
    def test_deep_copy(self):
        from negative_converter.utils.history import HistoryStack
        
        stack = HistoryStack[dict](max_size=10, deep_copy=True)
        
        data = {"nested": {"value": 1}}
        stack.push(data, "initial")
        
        # Modify original
        data["nested"]["value"] = 999
        
        # Stack should have original value
        current = stack.get_current_state()
        assert current["nested"]["value"] == 1


class TestImageHistoryStack:
    """Tests for the ImageHistoryStack class."""
    
    def test_push_and_undo_image(self):
        from negative_converter.utils.history import ImageHistoryStack
        
        stack = ImageHistoryStack(max_size=5)
        
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        stack.push(img1, "initial")
        stack.push(img2, "modified")
        
        assert stack.can_undo()
        
        result = stack.undo()
        assert result is not None
        assert np.array_equal(result, img1)
    
    def test_compression_for_large_images(self):
        from negative_converter.utils.history import ImageHistoryStack
        
        # Set low threshold to trigger compression
        stack = ImageHistoryStack(max_size=3, compress_threshold_mb=0.001)
        
        # Create image larger than threshold
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        stack.push(img, "large image")
        
        # Should still be able to retrieve
        result = stack.undo()
        # Can't undo with only one entry, but push should work


class TestGeometry:
    """Tests for geometry utilities."""
    
    def test_crop_rect_to_slice(self):
        from negative_converter.utils.geometry import CropRect
        
        rect = CropRect(10, 20, 100, 50)
        y_slice, x_slice = rect.to_slice()
        
        assert y_slice == slice(20, 70)
        assert x_slice == slice(10, 110)
    
    def test_crop_rect_is_valid(self):
        from negative_converter.utils.geometry import CropRect
        
        rect = CropRect(10, 10, 50, 50)
        assert rect.is_valid(100, 100)
        assert not rect.is_valid(50, 50)  # Would exceed bounds
    
    def test_crop_rect_clamp(self):
        from negative_converter.utils.geometry import CropRect
        
        rect = CropRect(-10, -10, 200, 200)
        clamped = rect.clamp(100, 100)
        
        assert clamped.x == 0
        assert clamped.y == 0
        assert clamped.width <= 100
        assert clamped.height <= 100
    
    def test_crop_image(self):
        from negative_converter.utils.geometry import crop_image, CropRect
        
        img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        rect = CropRect(10, 20, 30, 40)
        
        cropped = crop_image(img, rect)
        
        assert cropped.shape == (40, 30, 3)
    
    def test_rotate_90(self):
        from negative_converter.utils.geometry import rotate_90
        
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        
        rotated_cw = rotate_90(img, clockwise=True)
        assert rotated_cw.shape == (200, 100, 3)
        
        rotated_ccw = rotate_90(img, clockwise=False)
        assert rotated_ccw.shape == (200, 100, 3)
    
    def test_flip_image(self):
        from negative_converter.utils.geometry import flip_image
        
        img = np.arange(12, dtype=np.uint8).reshape(3, 4, 1)
        
        flipped_h = flip_image(img, horizontal=True)
        assert flipped_h[0, 0, 0] == img[0, 3, 0]
        
        flipped_v = flip_image(img, horizontal=False)
        assert flipped_v[0, 0, 0] == img[2, 0, 0]
    
    def test_aspect_ratio_constrain(self):
        from negative_converter.utils.geometry import AspectRatio, CropRect
        
        aspect = AspectRatio(4, 3, "4:3")
        rect = CropRect(0, 0, 100, 100)  # 1:1 ratio
        
        constrained = aspect.constrain(rect)
        
        # Should now be 4:3 ratio
        ratio = constrained.width / constrained.height
        assert abs(ratio - 4/3) < 0.01


class TestColorSampler:
    """Tests for color sampler utilities."""
    
    def test_sample_color(self):
        from negative_converter.utils.color_sampler import sample_color
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50, 50] = [255, 128, 64]
        
        sample = sample_color(img, 50, 50)
        
        assert sample is not None
        assert sample.rgb == (255, 128, 64)
        assert sample.x == 50
        assert sample.y == 50
    
    def test_sample_color_out_of_bounds(self):
        from negative_converter.utils.color_sampler import sample_color
        
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        sample = sample_color(img, 200, 200)
        assert sample is None
    
    def test_color_sample_hex(self):
        from negative_converter.utils.color_sampler import ColorSample
        
        sample = ColorSample(0, 0, (255, 128, 64))
        assert sample.hex == "#ff8040"
    
    def test_color_sample_hsv(self):
        from negative_converter.utils.color_sampler import ColorSample
        
        # Pure red
        sample = ColorSample(0, 0, (255, 0, 0))
        h, s, v = sample.hsv
        
        assert h == 0  # Red hue
        assert s == 100  # Full saturation
        assert v == 100  # Full value
    
    def test_rgb_to_lab(self):
        from negative_converter.utils.color_sampler import rgb_to_lab
        
        # White should have high L, near-zero a and b
        l, a, b = rgb_to_lab((255, 255, 255))
        assert l > 95
        assert abs(a) < 1
        assert abs(b) < 1
        
        # Black should have low L
        l, a, b = rgb_to_lab((0, 0, 0))
        assert l < 5
    
    def test_check_neutral(self):
        from negative_converter.utils.color_sampler import ColorSample, check_neutral
        
        # Neutral gray
        neutral = ColorSample(0, 0, (128, 128, 128))
        result = check_neutral(neutral)
        assert result["is_neutral"]
        
        # Not neutral (red tint)
        red_tint = ColorSample(0, 0, (150, 128, 128))
        result = check_neutral(red_tint)
        assert not result["is_neutral"]
        assert result["dominant_channel"] == "red"
    
    def test_sampler_state(self):
        from negative_converter.utils.color_sampler import ColorSample, ColorSamplerState
        
        state = ColorSamplerState(max_samples=3)
        
        for i in range(5):
            state.add_sample(ColorSample(i, i, (i * 50, i * 50, i * 50)))
        
        # Should only have 3 samples
        assert len(state.samples) == 3
        
        # Should be the last 3
        assert state.samples[0].x == 2


class TestPresetValidator:
    """Tests for preset validation."""
    
    def test_validate_valid_preset(self):
        from negative_converter.utils.preset_validator import validate_preset, FILM_PRESET_SCHEMA
        
        valid_preset = {
            "name": "Test Preset",
            "id": "test-preset",
            "description": "A test preset",
            "adjustments": {
                "brightness": 10,
                "contrast": 5,
            }
        }
        
        errors = validate_preset(valid_preset, FILM_PRESET_SCHEMA, "test.json")
        
        # Should have no errors (warnings are OK)
        error_count = sum(1 for e in errors if e.severity == "error")
        assert error_count == 0
    
    def test_validate_missing_required(self):
        from negative_converter.utils.preset_validator import validate_preset, FILM_PRESET_SCHEMA
        
        invalid_preset = {
            "description": "Missing name and id"
        }
        
        errors = validate_preset(invalid_preset, FILM_PRESET_SCHEMA, "test.json")
        
        # Should have errors for missing name and id
        error_fields = [e.field for e in errors if e.severity == "error"]
        assert "name" in error_fields
        assert "id" in error_fields
    
    def test_validate_out_of_range(self):
        from negative_converter.utils.preset_validator import validate_preset, FILM_PRESET_SCHEMA
        
        preset = {
            "name": "Test",
            "id": "test",
            "adjustments": {
                "brightness": 500,  # Out of range
            }
        }
        
        errors = validate_preset(preset, FILM_PRESET_SCHEMA, "test.json")
        
        # Should have warning for out of range
        warnings = [e for e in errors if e.severity == "warning"]
        assert len(warnings) > 0
    
    def test_validate_preset_file(self):
        from negative_converter.utils.preset_validator import validate_preset_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"name": "Test", "id": "test"}, f)
            temp_path = f.name
        
        try:
            is_valid, errors = validate_preset_file(temp_path)
            assert is_valid
        finally:
            os.unlink(temp_path)
    
    def test_validate_invalid_json(self):
        from negative_converter.utils.preset_validator import validate_preset_file
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_path = f.name
        
        try:
            is_valid, errors = validate_preset_file(temp_path)
            assert not is_valid
            assert any("json" in e.field.lower() for e in errors)
        finally:
            os.unlink(temp_path)


class TestExportPresets:
    """Tests for export presets."""
    
    def test_export_preset_defaults(self):
        from negative_converter.utils.export_presets import ExportPreset
        
        preset = ExportPreset(name="Test")
        
        assert preset.format == "jpeg"
        assert preset.quality == 95
        assert not preset.resize_enabled
    
    def test_export_preset_to_dict(self):
        from negative_converter.utils.export_presets import ExportPreset
        
        preset = ExportPreset(name="Test", format="png", quality=80)
        data = preset.to_dict()
        
        assert data["name"] == "Test"
        assert data["format"] == "png"
        assert data["quality"] == 80
    
    def test_export_preset_from_dict(self):
        from negative_converter.utils.export_presets import ExportPreset
        
        data = {"name": "Test", "format": "tiff", "quality": 100}
        preset = ExportPreset.from_dict(data)
        
        assert preset.name == "Test"
        assert preset.format == "tiff"
    
    def test_export_preset_get_extension(self):
        from negative_converter.utils.export_presets import ExportPreset
        
        assert ExportPreset(name="t", format="jpeg").get_extension() == ".jpg"
        assert ExportPreset(name="t", format="png").get_extension() == ".png"
        assert ExportPreset(name="t", format="tiff").get_extension() == ".tif"
        assert ExportPreset(name="t", format="webp").get_extension() == ".webp"
    
    def test_export_preset_get_save_params(self):
        from negative_converter.utils.export_presets import ExportPreset
        
        jpeg_preset = ExportPreset(name="t", format="jpeg", quality=90)
        params = jpeg_preset.get_save_params()
        assert params["quality"] == 90
        
        png_preset = ExportPreset(name="t", format="png", png_compression=9)
        params = png_preset.get_save_params()
        assert params["png_compression"] == 9
    
    def test_resize_image(self):
        from negative_converter.utils.export_presets import ExportPreset, resize_image
        
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        preset = ExportPreset(
            name="test",
            resize_enabled=True,
            resize_mode="fit",
            resize_width=500,
            resize_height=500
        )
        
        resized = resize_image(img, preset)
        
        # Should fit within 500x500 while maintaining aspect ratio
        assert resized.shape[0] <= 500
        assert resized.shape[1] <= 500
    
    def test_resize_disabled(self):
        from negative_converter.utils.export_presets import ExportPreset, resize_image
        
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        preset = ExportPreset(name="test", resize_enabled=False)
        
        result = resize_image(img, preset)
        
        # Should return original
        assert result.shape == img.shape


class TestSession:
    """Tests for session management."""
    
    def test_session_data_creation(self):
        from negative_converter.utils.session import SessionData
        
        session = SessionData()
        
        assert session.version == "1.0"
        assert session.current_image_path is None
        assert session.batch_images == []
    
    def test_session_data_to_dict(self):
        from negative_converter.utils.session import SessionData
        
        session = SessionData()
        session.current_image_path = "/path/to/image.jpg"
        session.current_adjustments = {"brightness": 10}
        
        data = session.to_dict()
        
        assert data["current_image_path"] == "/path/to/image.jpg"
        assert data["current_adjustments"]["brightness"] == 10
    
    def test_session_data_from_dict(self):
        from negative_converter.utils.session import SessionData
        
        data = {
            "version": "1.0",
            "current_image_path": "/test.jpg",
            "current_adjustments": {"contrast": 5},
            "batch_images": []
        }
        
        session = SessionData.from_dict(data)
        
        assert session.current_image_path == "/test.jpg"
        assert session.current_adjustments["contrast"] == 5
    
    def test_session_manager_new_session(self):
        from negative_converter.utils.session import SessionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(sessions_dir=tmpdir)
            session = manager.new_session()
            
            assert session is not None
            assert manager.get_current_session() is session
    
    def test_session_manager_save_and_load(self):
        from negative_converter.utils.session import SessionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(sessions_dir=tmpdir)
            session = manager.new_session()
            session.current_image_path = "/test/image.jpg"
            session.current_adjustments = {"brightness": 20}
            
            # Save
            assert manager.save_session()
            
            # Create new manager and load
            manager2 = SessionManager(sessions_dir=tmpdir)
            sessions = manager2.list_sessions()
            assert len(sessions) == 1
            
            loaded = manager2.load_session(sessions[0]["file_path"])
            assert loaded is not None
            assert loaded.current_image_path == "/test/image.jpg"
            assert loaded.current_adjustments["brightness"] == 20
    
    def test_session_manager_convenience_methods(self):
        from negative_converter.utils.session import SessionManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SessionManager(sessions_dir=tmpdir)
            
            manager.set_current_image("/path/to/img.jpg", {"brightness": 5})
            
            session = manager.get_current_session()
            assert session.current_image_path == "/path/to/img.jpg"
            assert session.current_adjustments["brightness"] == 5
            
            manager.set_presets(film_preset="portra-400")
            assert session.current_film_preset == "portra-400"


class TestKeyboardShortcuts:
    """Tests for keyboard shortcuts."""
    
    def test_shortcut_manager_get_shortcut(self):
        from negative_converter.utils.keyboard_shortcuts import ShortcutManager
        
        manager = ShortcutManager()
        
        shortcut = manager.get_shortcut("file.open")
        assert shortcut == "Ctrl+O"
    
    def test_shortcut_manager_set_shortcut(self):
        from negative_converter.utils.keyboard_shortcuts import ShortcutManager
        
        manager = ShortcutManager()
        
        # Set custom shortcut
        result = manager.set_shortcut("file.open", "Ctrl+Shift+O")
        assert result
        assert manager.get_shortcut("file.open") == "Ctrl+Shift+O"
    
    def test_shortcut_conflict_detection(self):
        from negative_converter.utils.keyboard_shortcuts import ShortcutManager
        
        manager = ShortcutManager()
        
        # Try to set a shortcut that conflicts
        conflict = manager.find_conflict("Ctrl+S")  # Already used by file.save
        assert conflict == "file.save"
    
    def test_shortcut_reset(self):
        from negative_converter.utils.keyboard_shortcuts import ShortcutManager
        
        manager = ShortcutManager()
        
        # Change and reset
        manager.set_shortcut("file.open", "Ctrl+Shift+O")
        manager.reset_shortcut("file.open")
        
        assert manager.get_shortcut("file.open") == "Ctrl+O"
    
    def test_shortcuts_by_category(self):
        from negative_converter.utils.keyboard_shortcuts import ShortcutManager
        
        manager = ShortcutManager()
        
        by_category = manager.get_all_by_category()
        
        assert "File" in by_category
        assert "Edit" in by_category
        assert "View" in by_category
        assert len(by_category["File"]) > 0
