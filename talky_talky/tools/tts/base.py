"""Base classes and interfaces for TTS engines.

This module defines the abstract interface that all TTS engines must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TTSResult:
    """Result from a TTS generation."""

    status: str  # "success" or "error"
    output_path: str
    duration_ms: int
    sample_rate: int
    chunks_used: int = 1
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptingGuide:
    """Guidance for prompting a TTS engine effectively."""

    overview: str  # Brief description of how to prompt this engine
    text_formatting: list[str]  # Tips for formatting input text
    emotion_tags: dict = field(default_factory=dict)  # Tag format, list, examples
    voice_guidance: dict = field(
        default_factory=dict
    )  # How to specify voice (description or reference)
    parameters: dict = field(default_factory=dict)  # Parameter descriptions and recommended values
    tips: list[str] = field(default_factory=list)  # General tips
    examples: list[dict] = field(default_factory=list)  # Full examples with text, params, and notes


@dataclass
class EngineInfo:
    """Information about a TTS engine."""

    name: str
    engine_type: str  # "text_prompted" or "audio_prompted"
    description: str
    requirements: str
    max_duration_secs: int
    chunk_size_chars: int
    sample_rate: int
    supports_emotions: bool = False
    emotion_format: Optional[str] = None  # "<tag>" or "[tag]"
    emotion_tags: list[str] = field(default_factory=list)
    prompting_guide: Optional[PromptingGuide] = None
    extra_info: dict = field(default_factory=dict)


class TTSEngine(ABC):
    """Abstract base class for TTS engines.

    All TTS engines must implement this interface to be usable in the system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine (e.g., 'maya1', 'chatterbox')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available (dependencies installed, etc.)."""
        pass

    @abstractmethod
    def get_info(self) -> EngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: Path,
        **kwargs,
    ) -> TTSResult:
        """Generate audio from text.

        Args:
            text: The text to synthesize.
            output_path: Where to save the generated audio.
            **kwargs: Engine-specific parameters.

        Returns:
            TTSResult with status and metadata.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."


class TextPromptedEngine(TTSEngine):
    """Base class for text-prompted TTS engines.

    These engines specify voices via natural language descriptions.
    """

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: Path,
        voice_description: str,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with a designed voice.

        Args:
            text: The text to synthesize.
            output_path: Where to save the generated audio.
            voice_description: Natural language description of the voice.
            **kwargs: Engine-specific parameters.

        Returns:
            TTSResult with status and metadata.
        """
        pass


class AudioPromptedEngine(TTSEngine):
    """Base class for audio-prompted TTS engines.

    These engines specify voices via reference audio samples.
    """

    @abstractmethod
    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        **kwargs,
    ) -> TTSResult:
        """Generate audio with a cloned voice.

        Args:
            text: The text to synthesize.
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio files.
            **kwargs: Engine-specific parameters.

        Returns:
            TTSResult with status and metadata.
        """
        pass
