"""Speech-to-Text transcription module with pluggable engine architecture.

This module provides a unified interface for multiple transcription engines:
- Whisper: OpenAI's robust speech recognition (via transformers)
- Faster-Whisper: CTranslate2-optimized Whisper (4x faster)

Usage:
    from talky_talky.tools.transcription import get_engine, list_engines, transcribe

    # List available engines
    engines = list_engines()

    # Get a specific engine
    whisper = get_engine("whisper")
    result = whisper.transcribe(audio_path="audio.wav")

    # Or use the unified transcribe function
    result = transcribe(
        audio_path="audio.wav",
        engine="faster_whisper",
        model_size="base",
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type

from .base import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
    TranscriptionEngineInfo,
)

# Import engines
from .whisper import WhisperEngine
from .faster_whisper import FasterWhisperEngine

# Reuse utilities from TTS module
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tts.utils import get_best_device, check_ffmpeg


# ============================================================================
# Engine Registry
# ============================================================================

_engine_registry: dict[str, Type[TranscriptionEngine]] = {}
_engine_instances: dict[str, TranscriptionEngine] = {}


def register_engine(engine_class: Type[TranscriptionEngine]) -> Type[TranscriptionEngine]:
    """Register a transcription engine class.

    Can be used as a decorator:
        @register_engine
        class MyEngine(TranscriptionEngine):
            ...

    Or called directly:
        register_engine(MyEngine)
    """
    temp_instance = engine_class()
    _engine_registry[temp_instance.engine_id] = engine_class
    return engine_class


def get_engine(engine_id: str) -> TranscriptionEngine:
    """Get a transcription engine instance by ID.

    Args:
        engine_id: The engine identifier (e.g., 'whisper', 'faster_whisper')

    Returns:
        TranscriptionEngine instance

    Raises:
        ValueError: If engine not found
    """
    if engine_id not in _engine_registry:
        available = list(_engine_registry.keys())
        raise ValueError(f"Engine '{engine_id}' not found. Available: {available}")

    if engine_id not in _engine_instances:
        _engine_instances[engine_id] = _engine_registry[engine_id]()

    return _engine_instances[engine_id]


def list_engines() -> dict[str, TranscriptionEngineInfo]:
    """List all registered transcription engines with their info.

    Returns:
        Dict mapping engine_id to TranscriptionEngineInfo
    """
    result = {}
    for engine_id in _engine_registry:
        engine = get_engine(engine_id)
        info = engine.get_info()
        result[engine_id] = info
    return result


def get_available_engines() -> list[str]:
    """Get list of engine IDs that are currently available (installed).

    Returns:
        List of engine IDs that have all dependencies installed
    """
    available = []
    for engine_id in _engine_registry:
        engine = get_engine(engine_id)
        if engine.is_available():
            available.append(engine_id)
    return available


# ============================================================================
# Unified Transcription Interface
# ============================================================================


def transcribe(
    audio_path: str | Path,
    engine: str = "faster_whisper",
    **kwargs,
) -> TranscriptionResult:
    """Transcribe audio using the specified engine.

    This is a convenience function that dispatches to the appropriate engine.

    Args:
        audio_path: Path to the audio file to transcribe.
        engine: Engine ID ('whisper', 'faster_whisper')
        **kwargs: Engine-specific parameters

    Common parameters:
        language (str): Language code (e.g., 'en'). Auto-detect if None.
        model_size (str): Model size ('tiny', 'base', 'small', 'medium', 'large-v3', etc.)

    For Whisper:
        return_timestamps (bool|str): True for segments, 'word' for word-level.

    For Faster-Whisper:
        word_timestamps (bool): Enable word-level timestamps.
        vad_filter (bool): Filter silence using VAD (default: True).
        beam_size (int): Beam size for decoding (default: 5).

    Returns:
        TranscriptionResult with transcribed text and metadata
    """
    transcription_engine = get_engine(engine)
    return transcription_engine.transcribe(audio_path=Path(audio_path), **kwargs)


# ============================================================================
# Status and Diagnostics
# ============================================================================


@dataclass
class TranscriptionStatus:
    """Overall transcription system status."""

    status: str  # "ok", "partial", "error"
    available_engines: list[str] = field(default_factory=list)
    unavailable_engines: list[str] = field(default_factory=list)
    device: Optional[str] = None
    device_name: Optional[str] = None
    vram_gb: Optional[float] = None
    ffmpeg_available: bool = False
    warnings: list[str] = field(default_factory=list)
    setup_instructions: dict[str, str] = field(default_factory=dict)


def check_transcription() -> TranscriptionStatus:
    """Check overall transcription system status.

    Returns detailed status including which engines are available
    and setup instructions for unavailable engines.
    """
    result = TranscriptionStatus(status="ok")

    # Check device
    device, device_name, vram_gb = get_best_device()
    result.device = device
    result.device_name = device_name
    result.vram_gb = vram_gb

    if device == "cpu":
        result.warnings.append("No GPU detected - transcription will run on CPU (slower)")

    # Check ffmpeg
    result.ffmpeg_available = check_ffmpeg()
    if not result.ffmpeg_available:
        result.warnings.append(
            "ffmpeg not installed - may be needed for some audio formats. "
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    # Check each engine
    for engine_id in _engine_registry:
        engine = get_engine(engine_id)
        if engine.is_available():
            result.available_engines.append(engine_id)
        else:
            result.unavailable_engines.append(engine_id)
            result.setup_instructions[engine_id] = engine.get_setup_instructions()

    # Determine overall status
    if not result.available_engines:
        result.status = "error"
        result.warnings.append("No transcription engines available. See setup_instructions.")
    elif result.unavailable_engines:
        result.status = "partial"

    return result


def get_transcription_info() -> dict:
    """Get detailed information about all transcription engines.

    Returns comprehensive info for agent consumption.
    """
    engines_info = {}

    for engine_id in _engine_registry:
        engine = get_engine(engine_id)
        info = engine.get_info()
        engine_data = {
            "name": info.name,
            "description": info.description,
            "available": engine.is_available(),
            "requirements": info.requirements,
            "supported_languages": info.supported_languages[:20],  # First 20 for brevity
            "total_languages": len(info.supported_languages),
            "supports_word_timestamps": info.supports_word_timestamps,
            "supports_language_detection": info.supports_language_detection,
            "model_sizes": info.model_sizes,
            "default_model_size": info.default_model_size,
            **info.extra_info,
        }
        engines_info[engine_id] = engine_data

    return {"engines": engines_info}


# ============================================================================
# Register Built-in Engines
# ============================================================================

register_engine(WhisperEngine)
register_engine(FasterWhisperEngine)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Core types
    "TranscriptionEngine",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WordSegment",
    "TranscriptionEngineInfo",
    "TranscriptionStatus",
    # Engine registry
    "register_engine",
    "get_engine",
    "list_engines",
    "get_available_engines",
    # Transcription
    "transcribe",
    # Status
    "check_transcription",
    "get_transcription_info",
    # Engine classes
    "WhisperEngine",
    "FasterWhisperEngine",
]
