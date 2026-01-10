"""Text-to-Speech module with pluggable engine architecture.

This module provides a unified interface for multiple TTS engines:
- Maya1: Text-prompted voice design (describe the voice you want)
- Chatterbox: Audio-prompted voice cloning (clone from reference audio)

Usage:
    from talky_talky.tools.tts import get_engine, list_engines, generate

    # List available engines
    engines = list_engines()

    # Get a specific engine
    maya1 = get_engine("maya1")
    result = maya1.generate(text="Hello", output_path="out.wav", voice_description="...")

    # Or use the unified generate function
    result = generate(
        text="Hello world",
        output_path="output.wav",
        engine="maya1",
        voice_description="Female narrator, warm voice"
    )
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Type

from .base import (
    TTSEngine,
    TTSResult,
    EngineInfo,
    SpeedEstimate,
    TextPromptedEngine,
    AudioPromptedEngine,
    VoiceSelectionEngine,
)
from .utils import check_ffmpeg, get_best_device

# Import engines at module level (registered below after registry is defined)
from .maya1 import Maya1Engine
from .chatterbox import ChatterboxEngine
from .mira import MiraEngine
from .xtts import XTTSEngine
from .kokoro import KokoroEngine
from .soprano import SopranoEngine
from .chatterbox_turbo import ChatterboxTurboEngine
from .vibevoice import VibeVoiceRealtimeEngine, VibeVoiceLongformEngine
from .cosyvoice import CosyVoice3Engine


# ============================================================================
# Engine Registry
# ============================================================================

_engine_registry: dict[str, Type[TTSEngine]] = {}
_engine_instances: dict[str, TTSEngine] = {}


def register_engine(engine_class: Type[TTSEngine]) -> Type[TTSEngine]:
    """Register a TTS engine class.

    Can be used as a decorator:
        @register_engine
        class MyEngine(TTSEngine):
            ...

    Or called directly:
        register_engine(MyEngine)
    """
    # Create a temporary instance to get the engine_id
    temp_instance = engine_class()
    _engine_registry[temp_instance.engine_id] = engine_class
    return engine_class


def get_engine(engine_id: str) -> TTSEngine:
    """Get a TTS engine instance by ID.

    Args:
        engine_id: The engine identifier (e.g., 'maya1', 'chatterbox')

    Returns:
        TTSEngine instance

    Raises:
        ValueError: If engine not found
    """
    if engine_id not in _engine_registry:
        available = list(_engine_registry.keys())
        raise ValueError(f"Engine '{engine_id}' not found. Available: {available}")

    # Use cached instance if available
    if engine_id not in _engine_instances:
        _engine_instances[engine_id] = _engine_registry[engine_id]()

    return _engine_instances[engine_id]


def list_engines() -> dict[str, EngineInfo]:
    """List all registered TTS engines with their info.

    Returns:
        Dict mapping engine_id to EngineInfo
    """
    result = {}
    for engine_id, engine_class in _engine_registry.items():
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


def get_engine_speed_for_current_device(engine_id: str) -> Optional[SpeedEstimate]:
    """Get the speed estimate for an engine on the current device.

    Args:
        engine_id: The engine identifier (e.g., 'maya1', 'chatterbox')

    Returns:
        SpeedEstimate for the current device type, or None if not available
    """
    engine = get_engine(engine_id)
    info = engine.get_info()

    if not info.speed_estimates:
        return None

    # Get current device type
    device, _, _ = get_best_device()

    # Try to get estimate for current device, falling back to CPU if not found
    if device in info.speed_estimates:
        return info.speed_estimates[device]

    # If current device not found but CPU estimate exists, return that with a note
    if "cpu" in info.speed_estimates:
        return info.speed_estimates["cpu"]

    return None


def get_all_engine_speeds() -> dict[str, Optional[dict]]:
    """Get speed estimates for all engines on the current device.

    Returns:
        Dict mapping engine_id to speed estimate dict (or None if unavailable)
    """
    device, device_name, _ = get_best_device()
    result = {}

    for engine_id in _engine_registry:
        estimate = get_engine_speed_for_current_device(engine_id)
        if estimate:
            result[engine_id] = {
                "realtime_factor": estimate.realtime_factor,
                "device_type": estimate.device_type,
                "reference_hardware": estimate.reference_hardware,
                "notes": estimate.notes,
                "matches_current_device": estimate.device_type == device,
            }
        else:
            result[engine_id] = None

    return result


# ============================================================================
# Unified Generation Interface
# ============================================================================


def generate(
    text: str,
    output_path: str | Path,
    engine: str = "maya1",
    **kwargs,
) -> TTSResult:
    """Generate audio using the specified TTS engine.

    This is a convenience function that dispatches to the appropriate engine.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio.
        engine: Engine ID ('maya1', 'chatterbox', etc.)
        **kwargs: Engine-specific parameters

    For Maya1 (voice_description required):
        generate(
            text="Hello world",
            output_path="out.wav",
            engine="maya1",
            voice_description="Female narrator, warm voice",
            temperature=0.4,  # optional
        )

    For Chatterbox (reference_audio_paths required):
        generate(
            text="Hello world",
            output_path="out.wav",
            engine="chatterbox",
            reference_audio_paths=["/path/to/reference.wav"],
            exaggeration=0.5,  # optional
        )

    Returns:
        TTSResult with status and metadata
    """
    tts_engine = get_engine(engine)
    return tts_engine.generate(text=text, output_path=Path(output_path), **kwargs)


# ============================================================================
# Status and Diagnostics
# ============================================================================


@dataclass
class TTSStatus:
    """Overall TTS system status."""

    status: str  # "ok", "partial", "error"
    available_engines: list[str] = field(default_factory=list)
    unavailable_engines: list[str] = field(default_factory=list)
    device: Optional[str] = None
    device_name: Optional[str] = None
    vram_gb: Optional[float] = None
    ffmpeg_available: bool = False
    warnings: list[str] = field(default_factory=list)
    setup_instructions: dict[str, str] = field(default_factory=dict)


def check_tts() -> TTSStatus:
    """Check overall TTS system status.

    Returns detailed status including which engines are available
    and setup instructions for unavailable engines.
    """
    result = TTSStatus(status="ok")

    # Check device
    device, device_name, vram_gb = get_best_device()
    result.device = device
    result.device_name = device_name
    result.vram_gb = vram_gb

    if device == "cpu":
        result.warnings.append("No GPU detected - TTS will run on CPU (slower)")

    # Check ffmpeg
    result.ffmpeg_available = check_ffmpeg()
    if not result.ffmpeg_available:
        result.warnings.append(
            "ffmpeg not installed - required for audio processing. "
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
        result.warnings.append("No TTS engines available. See setup_instructions.")
    elif result.unavailable_engines:
        result.status = "partial"

    return result


def get_tts_info() -> dict:
    """Get detailed information about all TTS engines.

    Returns comprehensive info for agent consumption, including speed
    estimates for the current device.
    """
    device, device_name, _ = get_best_device()
    engines_info = {}

    for engine_id in _engine_registry:
        engine = get_engine(engine_id)
        info = engine.get_info()
        engine_data = {
            "name": info.name,
            "type": info.engine_type,
            "description": info.description,
            "available": engine.is_available(),
            "requirements": info.requirements,
            "max_duration_secs": info.max_duration_secs,
            "chunk_size_chars": info.chunk_size_chars,
            "sample_rate": info.sample_rate,
            "supports_emotions": info.supports_emotions,
            "emotion_format": info.emotion_format,
            "emotion_tags": info.emotion_tags,
            **info.extra_info,
        }

        # Include prompting guide if available
        if info.prompting_guide:
            engine_data["prompting_guide"] = asdict(info.prompting_guide)

        # Include speed estimate for current device
        speed_estimate = get_engine_speed_for_current_device(engine_id)
        if speed_estimate:
            engine_data["speed_estimate"] = {
                "realtime_factor": speed_estimate.realtime_factor,
                "device_type": speed_estimate.device_type,
                "reference_hardware": speed_estimate.reference_hardware,
                "notes": speed_estimate.notes,
                "matches_current_device": speed_estimate.device_type == device,
            }

        # Include all speed estimates for reference
        if info.speed_estimates:
            engine_data["all_speed_estimates"] = {
                dev: {
                    "realtime_factor": est.realtime_factor,
                    "reference_hardware": est.reference_hardware,
                    "notes": est.notes,
                }
                for dev, est in info.speed_estimates.items()
            }

        engines_info[engine_id] = engine_data

    return {
        "current_device": {
            "type": device,
            "name": device_name,
        },
        "engines": engines_info,
    }


# ============================================================================
# Register Built-in Engines
# ============================================================================

register_engine(Maya1Engine)
register_engine(ChatterboxEngine)
register_engine(MiraEngine)
register_engine(XTTSEngine)
register_engine(KokoroEngine)
register_engine(SopranoEngine)
register_engine(ChatterboxTurboEngine)
register_engine(VibeVoiceRealtimeEngine)
register_engine(VibeVoiceLongformEngine)
register_engine(CosyVoice3Engine)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Core types
    "TTSEngine",
    "TTSResult",
    "EngineInfo",
    "SpeedEstimate",
    "TextPromptedEngine",
    "AudioPromptedEngine",
    "VoiceSelectionEngine",
    "TTSStatus",
    # Engine registry
    "register_engine",
    "get_engine",
    "list_engines",
    "get_available_engines",
    # Speed estimates
    "get_engine_speed_for_current_device",
    "get_all_engine_speeds",
    # Generation
    "generate",
    # Status
    "check_tts",
    "get_tts_info",
    # Engine classes (for direct use)
    "Maya1Engine",
    "ChatterboxEngine",
    "MiraEngine",
    "XTTSEngine",
    "KokoroEngine",
    "SopranoEngine",
    "ChatterboxTurboEngine",
    "VibeVoiceRealtimeEngine",
    "VibeVoiceLongformEngine",
    "CosyVoice3Engine",
]
