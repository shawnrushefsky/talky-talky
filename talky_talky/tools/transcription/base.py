"""Base classes and interfaces for transcription engines.

This module defines the abstract interface that all transcription engines must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class WordSegment:
    """A single word with timing information."""

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: Optional[float] = None


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text with timing information."""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    words: list[WordSegment] = field(default_factory=list)
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Result from a transcription."""

    status: str  # "success" or "error"
    text: str  # Full transcribed text
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None  # Detected language code
    language_probability: Optional[float] = None
    duration_seconds: Optional[float] = None  # Audio duration
    processing_time_ms: Optional[int] = None  # How long transcription took
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TranscriptionEngineInfo:
    """Information about a transcription engine."""

    name: str
    engine_id: str
    description: str
    requirements: str
    supported_languages: list[str]
    supports_word_timestamps: bool = False
    supports_language_detection: bool = True
    model_sizes: list[str] = field(default_factory=list)
    default_model_size: str = "base"
    extra_info: dict = field(default_factory=dict)


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines.

    All transcription engines must implement this interface to be usable in the system.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this engine."""
        pass

    @property
    @abstractmethod
    def engine_id(self) -> str:
        """Unique identifier for this engine (e.g., 'whisper', 'faster_whisper')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this engine is available (dependencies installed, etc.)."""
        pass

    @abstractmethod
    def get_info(self) -> TranscriptionEngineInfo:
        """Get detailed information about this engine."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio_path: Path to the audio file to transcribe.
            language: Optional language code (e.g., 'en', 'es'). If None, auto-detect.
            **kwargs: Engine-specific parameters (model_size, word_timestamps, etc.)

        Returns:
            TranscriptionResult with transcribed text and metadata.
        """
        pass

    def get_setup_instructions(self) -> str:
        """Get setup instructions for this engine."""
        return f"No setup instructions available for {self.name}."
