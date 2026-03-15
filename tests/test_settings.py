"""
=============================================================================
Tests for Settings Manager — Configuration Hierarchy
=============================================================================
These tests verify that the SettingsManager correctly implements the
3-tier configuration priority (defaults → .env → settings.json) and
that API key masking works properly.
=============================================================================
"""

import os
import json
import pytest
import tempfile


class TestSettingsManager:
    """Tests for the SettingsManager class."""

    def _create_settings_manager(self, temp_dir, saved_settings=None):
        """
        Helper to create a SettingsManager with controlled inputs.

        We patch SETTINGS_FILE and DB_DIR at the module level before
        creating a fresh SettingsManager instance.
        """
        from unittest.mock import patch

        settings_file = os.path.join(temp_dir, "settings.json")

        if saved_settings:
            with open(settings_file, 'w') as f:
                json.dump(saved_settings, f)

        # Patch module-level constants individually (not dotted keys!)
        with patch('src.core.config.SETTINGS_FILE', settings_file), \
             patch('src.core.config.DB_DIR', temp_dir):
            from src.core.config import SettingsManager
            manager = SettingsManager()

        return manager

    def test_default_values(self, temp_dir):
        """
        Without any settings.json, the SettingsManager should
        return the env/hardcoded defaults.
        """
        manager = self._create_settings_manager(temp_dir)
        settings = manager.get_all()

        assert "gemini_api_key" in settings
        assert "use_local_embeddings" in settings
        assert "generation_model" in settings

    def test_mask_api_key_long(self, temp_dir):
        """
        API keys longer than 8 chars should be masked: first 4 + *** + last 4.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={"gemini_api_key": "sk-1234567890abcdef"}
        )
        safe = manager.get_all()

        assert "***" in safe["gemini_api_key"]
        assert safe["gemini_api_key"].startswith("sk-1")
        assert safe["gemini_api_key"].endswith("cdef")

    def test_mask_api_key_short(self, temp_dir):
        """
        Short API keys (<=8 chars) should be fully masked.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={"gemini_api_key": "short"}
        )
        safe = manager.get_all()
        assert safe["gemini_api_key"] == "***"

    def test_mask_empty_api_key(self, temp_dir):
        """
        Empty API keys should remain empty in the masked output.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={"gemini_api_key": ""}
        )
        safe = manager.get_all()
        assert safe["gemini_api_key"] == ""

    def test_settings_json_overrides_env(self, temp_dir):
        """
        Values in settings.json should override environment defaults.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={
                "generation_model": "models/gemini-custom-model",
                "chunk_size": 1000,
            }
        )
        assert manager.get("generation_model") == "models/gemini-custom-model"
        assert manager.get("chunk_size") == 1000

    def test_update_does_not_overwrite_masked_key(self, temp_dir):
        """
        If the front-end sends back a masked API key (containing ***),
        the update should NOT overwrite the real key with the mask.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={"gemini_api_key": "real-secret-key-123"}
        )

        # Simulate the front-end sending back masked key
        manager.update({"gemini_api_key": "real***123"})

        # The real key should still be intact
        assert manager.get("gemini_api_key") == "real-secret-key-123"

    def test_update_allows_new_api_key(self, temp_dir):
        """
        If the user enters a new API key (no *** in it), it should
        be saved normally.
        """
        manager = self._create_settings_manager(
            temp_dir,
            saved_settings={"gemini_api_key": "old-key"}
        )

        manager.update({"gemini_api_key": "brand-new-key-456"})
        assert manager.get("gemini_api_key") == "brand-new-key-456"

    def test_save_creates_file(self, temp_dir):
        """
        save() should create a settings.json file in the db/ directory.
        """
        settings_file = os.path.join(temp_dir, "settings.json")
        from unittest.mock import patch
        from src.core.config import SettingsManager

        with patch('src.core.config.SETTINGS_FILE', settings_file), \
             patch('src.core.config.DB_DIR', temp_dir):
            manager = SettingsManager()
            manager.save()

        assert os.path.exists(settings_file)

        with open(settings_file) as f:
            saved = json.load(f)
        assert isinstance(saved, dict)

    def test_get_nonexistent_key_returns_default(self, temp_dir):
        """
        get() with a non-existent key should return the provided default.
        """
        manager = self._create_settings_manager(temp_dir)
        result = manager.get("nonexistent_key", "fallback_value")
        assert result == "fallback_value"
