"""Tests for TTS utilities."""

from talky_talky.tools.tts.utils import (
    convert_angle_to_bracket_tags,
    convert_bracket_to_angle_tags,
    split_text_into_chunks,
    get_best_device,
)


class TestTagConversion:
    """Tests for emotion tag format conversion."""

    def test_angle_to_bracket_single_tag(self):
        """Convert single angle tag to bracket."""
        text = "Hello <laugh> world"
        result = convert_angle_to_bracket_tags(text)
        assert result == "Hello [laugh] world"

    def test_angle_to_bracket_multiple_tags(self):
        """Convert multiple angle tags to brackets."""
        text = "<whisper> Be quiet <gasp> what was that?"
        result = convert_angle_to_bracket_tags(text)
        assert result == "[whisper] Be quiet [gasp] what was that?"

    def test_angle_to_bracket_no_tags(self):
        """Text without tags should remain unchanged."""
        text = "Hello world, no tags here."
        result = convert_angle_to_bracket_tags(text)
        assert result == text

    def test_bracket_to_angle_single_tag(self):
        """Convert single bracket tag to angle."""
        text = "Hello [laugh] world"
        result = convert_bracket_to_angle_tags(text)
        assert result == "Hello <laugh> world"

    def test_bracket_to_angle_multiple_tags(self):
        """Convert multiple bracket tags to angles."""
        text = "[whisper] Be quiet [gasp] what was that?"
        result = convert_bracket_to_angle_tags(text)
        assert result == "<whisper> Be quiet <gasp> what was that?"

    def test_bracket_to_angle_no_tags(self):
        """Text without tags should remain unchanged."""
        text = "Hello world, no tags here."
        result = convert_bracket_to_angle_tags(text)
        assert result == text

    def test_roundtrip_conversion(self):
        """Converting angle->bracket->angle should preserve original."""
        original = "Hello <laugh> world <sigh> goodbye"
        bracket = convert_angle_to_bracket_tags(original)
        back = convert_bracket_to_angle_tags(bracket)
        assert back == original


class TestTextChunking:
    """Tests for text chunking functionality."""

    def test_short_text_no_chunking(self):
        """Short text should not be chunked."""
        text = "This is a short sentence."
        chunks = split_text_into_chunks(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_multiple_sentences(self):
        """Multiple sentences should be grouped up to max_chars."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = split_text_into_chunks(text, max_chars=50)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 50 or "." not in chunk[:-1]  # Allow long single sentences

    def test_empty_text(self):
        """Empty text should return empty list or single empty chunk."""
        text = ""
        chunks = split_text_into_chunks(text, max_chars=100)
        assert chunks == []

    def test_whitespace_only(self):
        """Whitespace-only text should return empty list."""
        text = "   \n\t   "
        chunks = split_text_into_chunks(text, max_chars=100)
        assert chunks == []

    def test_respects_sentence_boundaries(self):
        """Chunks should end at sentence boundaries when possible."""
        text = "Hello world. This is a test. Another sentence here."
        chunks = split_text_into_chunks(text, max_chars=30)
        for chunk in chunks:
            # Each chunk should end with a sentence-ending punctuation or be the full text
            assert chunk.rstrip().endswith((".", "!", "?")) or len(chunk) <= 30


class TestDeviceDetection:
    """Tests for device detection utilities."""

    def test_get_best_device_returns_tuple(self):
        """get_best_device should return a 3-tuple."""
        result = get_best_device()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_get_best_device_valid_device(self):
        """Device should be one of cuda, mps, or cpu."""
        device, name, vram = get_best_device()
        assert device in ("cuda", "mps", "cpu")
        assert isinstance(name, str)
        assert vram is None or isinstance(vram, (int, float))
