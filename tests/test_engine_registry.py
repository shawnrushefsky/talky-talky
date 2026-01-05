"""Tests for TTS engine registry."""

import pytest
from talky_talky.tools.tts import (
    get_engine,
    list_engines,
    get_available_engines,
    TTSEngine,
    EngineInfo,
)


class TestEngineRegistry:
    """Tests for the engine registry system."""

    def test_list_engines_returns_dict(self):
        """list_engines should return a dict of EngineInfo."""
        engines = list_engines()
        assert isinstance(engines, dict)
        assert len(engines) >= 2  # At least maya1 and chatterbox

    def test_list_engines_contains_maya1(self):
        """Maya1 engine should be registered."""
        engines = list_engines()
        assert "maya1" in engines
        info = engines["maya1"]
        assert isinstance(info, EngineInfo)
        assert info.name == "Maya1"
        assert info.engine_type == "text_prompted"

    def test_list_engines_contains_chatterbox(self):
        """Chatterbox engine should be registered."""
        engines = list_engines()
        assert "chatterbox" in engines
        info = engines["chatterbox"]
        assert isinstance(info, EngineInfo)
        assert info.name == "Chatterbox TTS"
        assert info.engine_type == "audio_prompted"

    def test_get_engine_maya1(self):
        """get_engine should return Maya1 engine instance."""
        engine = get_engine("maya1")
        assert isinstance(engine, TTSEngine)
        assert engine.engine_id == "maya1"
        assert engine.name == "Maya1"

    def test_get_engine_chatterbox(self):
        """get_engine should return Chatterbox engine instance."""
        engine = get_engine("chatterbox")
        assert isinstance(engine, TTSEngine)
        assert engine.engine_id == "chatterbox"
        assert engine.name == "Chatterbox TTS"

    def test_get_engine_invalid_raises(self):
        """get_engine should raise ValueError for unknown engine."""
        with pytest.raises(ValueError) as exc_info:
            get_engine("nonexistent_engine")
        assert "not found" in str(exc_info.value).lower()

    def test_get_available_engines_returns_list(self):
        """get_available_engines should return a list of strings."""
        available = get_available_engines()
        assert isinstance(available, list)
        for engine_id in available:
            assert isinstance(engine_id, str)

    def test_engine_info_has_required_fields(self):
        """EngineInfo should have all required fields."""
        engines = list_engines()
        for engine_id, info in engines.items():
            assert hasattr(info, "name")
            assert hasattr(info, "engine_type")
            assert hasattr(info, "description")
            assert hasattr(info, "requirements")
            assert hasattr(info, "max_duration_secs")
            assert hasattr(info, "chunk_size_chars")
            assert hasattr(info, "sample_rate")

    def test_engine_has_is_available_method(self):
        """Each engine should implement is_available()."""
        for engine_id in list_engines():
            engine = get_engine(engine_id)
            result = engine.is_available()
            assert isinstance(result, bool)

    def test_engine_has_get_info_method(self):
        """Each engine should implement get_info()."""
        for engine_id in list_engines():
            engine = get_engine(engine_id)
            info = engine.get_info()
            assert isinstance(info, EngineInfo)

    def test_engine_instance_caching(self):
        """get_engine should return cached instances."""
        engine1 = get_engine("maya1")
        engine2 = get_engine("maya1")
        assert engine1 is engine2  # Same instance
