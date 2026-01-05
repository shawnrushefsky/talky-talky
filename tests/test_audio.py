"""Tests for audio utilities."""

from talky_talky.tools.audio import (
    AudioInfo,
    ConvertResult,
    ConcatenateResult,
    NormalizeResult,
    get_audio_info,
    is_ffmpeg_available,
)


class TestAudioInfo:
    """Tests for audio info functionality."""

    def test_audio_info_file_not_found(self):
        """get_audio_info should return error for missing file."""
        result = get_audio_info("/nonexistent/path/audio.wav")
        assert isinstance(result, AudioInfo)
        assert result.exists is False
        assert "not found" in result.error.lower()

    def test_audio_info_dataclass_fields(self):
        """AudioInfo should have all required fields."""
        info = AudioInfo(
            path="/test/path.wav",
            exists=True,
            format="wav",
            duration_ms=1000,
            size_bytes=44100,
            valid=True,
        )
        assert info.path == "/test/path.wav"
        assert info.exists is True
        assert info.format == "wav"
        assert info.duration_ms == 1000
        assert info.size_bytes == 44100
        assert info.valid is True
        assert info.error is None


class TestConvertResult:
    """Tests for ConvertResult dataclass."""

    def test_convert_result_dataclass(self):
        """ConvertResult should have all required fields."""
        result = ConvertResult(
            input_path="/input.wav",
            output_path="/output.mp3",
            input_format="wav",
            output_format="mp3",
            input_size_bytes=1000000,
            output_size_bytes=100000,
            compression_ratio=10.0,
            duration_ms=5000,
        )
        assert result.input_path == "/input.wav"
        assert result.output_path == "/output.mp3"
        assert result.compression_ratio == 10.0


class TestConcatenateResult:
    """Tests for ConcatenateResult dataclass."""

    def test_concatenate_result_dataclass(self):
        """ConcatenateResult should have all required fields."""
        result = ConcatenateResult(
            output_path="/output.mp3",
            input_count=3,
            total_duration_ms=15000,
            output_format="mp3",
        )
        assert result.output_path == "/output.mp3"
        assert result.input_count == 3
        assert result.total_duration_ms == 15000


class TestNormalizeResult:
    """Tests for NormalizeResult dataclass."""

    def test_normalize_result_dataclass(self):
        """NormalizeResult should have all required fields."""
        result = NormalizeResult(
            input_path="/input.wav",
            output_path="/output_normalized.wav",
            duration_ms=10000,
        )
        assert result.input_path == "/input.wav"
        assert result.output_path == "/output_normalized.wav"
        assert result.duration_ms == 10000


class TestFfmpegAvailability:
    """Tests for ffmpeg availability check."""

    def test_is_ffmpeg_available_returns_bool(self):
        """is_ffmpeg_available should return a boolean."""
        result = is_ffmpeg_available()
        assert isinstance(result, bool)
