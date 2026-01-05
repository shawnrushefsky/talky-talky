#!/usr/bin/env python3
"""Talky Talky - Text-to-Speech MCP Server for AI Agents.

This MCP server provides TTS capabilities with pluggable engine support:
- Maya1: Text-prompted voice design (describe the voice you want)
- Chatterbox: Audio-prompted voice cloning (clone from reference audio)
- MiraTTS: Fast voice cloning with high-quality 48kHz output

Plus audio utilities for format conversion and concatenation.
"""

from dataclasses import asdict
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import TTS module
from .tools.tts import (
    check_tts,
    get_tts_info,
    get_available_engines,
    list_engines,
    generate,
)
from .tools.tts.maya1 import (
    check_models_downloaded as check_maya1_models,
    download_models as download_maya1_models,
)

# Import audio utilities
from .tools.audio import (
    get_audio_info,
    convert_audio,
    concatenate_audio,
    normalize_audio,
    is_ffmpeg_available,
)


# Server version
VERSION = "0.2.0"

# Initialize MCP server
mcp = FastMCP("talky-talky")


# ============================================================================
# Helper Functions
# ============================================================================


def to_dict(obj) -> dict:
    """Convert dataclass to dict, handling nested objects."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    elif isinstance(obj, dict):
        return obj
    else:
        return {"value": obj}


# ============================================================================
# TTS Engine Tools
# ============================================================================


@mcp.tool()
def check_tts_availability() -> dict:
    """Check if TTS engines are available and properly configured.

    Returns detailed status including:
    - Available engines
    - Device info (CUDA/MPS/CPU)
    - Setup instructions for unavailable engines
    """
    status = check_tts()
    return to_dict(status)


@mcp.tool()
def get_tts_engines_info() -> dict:
    """Get detailed information about all TTS engines.

    Returns info for each engine including:
    - Name, type, and description
    - Requirements and availability
    - Supported emotion tags and format
    - Engine-specific parameters
    """
    return get_tts_info()


@mcp.tool()
def list_available_engines() -> dict:
    """List TTS engines that are currently available (installed).

    Returns:
        Dict with list of available engine IDs and their basic info.
    """
    available = get_available_engines()
    engines = list_engines()

    return {
        "available_engines": available,
        "engines": {
            engine_id: {
                "name": info.name,
                "type": info.engine_type,
                "description": info.description,
            }
            for engine_id, info in engines.items()
            if engine_id in available
        },
    }


@mcp.tool()
def get_tts_model_status() -> dict:
    """Get the download status of TTS models (Maya1 and SNAC).

    Returns information about which models are downloaded and their cache locations.
    """
    return check_maya1_models()


@mcp.tool()
def download_tts_models(force: bool = False) -> dict:
    """Download Maya1 TTS model weights from HuggingFace.

    Downloads both the Maya1 language model and SNAC audio codec.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results for each model.
    """
    return download_maya1_models(force=force)


# ============================================================================
# Speech Generation Tools
# ============================================================================


@mcp.tool()
def speak_maya1(
    text: str,
    output_path: str,
    voice_description: str,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> dict:
    """Generate speech using Maya1 (text-prompted voice design).

    Creates audio from text using a natural language voice description.
    Supports inline emotion tags like <laugh>, <sigh>, <angry>, <whisper>.

    Args:
        text: The text to synthesize. Can include emotion tags.
        output_path: Where to save the generated audio (e.g., "/tmp/output.wav").
        voice_description: Natural language description of the voice.
            Example: "Gruff male pirate, 50s, British accent, slow pacing"
        temperature: Sampling temperature (0.1-1.0, default 0.4). Lower = more stable.
        top_p: Nucleus sampling parameter (0.1-1.0, default 0.9).
        repetition_penalty: Penalty for repetition (1.0-2.0, default 1.1).

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Emotion tags supported: <laugh>, <sigh>, <gasp>, <whisper>, <angry>, <excited>, etc.
    """
    result = generate(
        text=text,
        output_path=output_path,
        engine="maya1",
        voice_description=voice_description,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return to_dict(result)


@mcp.tool()
def speak_chatterbox(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> dict:
    """Generate speech using Chatterbox (audio-prompted voice cloning).

    Clones a voice from reference audio samples with emotion control.
    Supports paralinguistic tags like [laugh], [cough], [chuckle].

    Args:
        text: The text to synthesize. Can include emotion tags.
        output_path: Where to save the generated audio (e.g., "/tmp/output.wav").
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 10+ seconds of clear speech recommended.
        exaggeration: Controls speech expressiveness (0.0-1.0+, default 0.5).
            0.0 = flat/monotone, 0.5 = natural, 0.7+ = dramatic.
        cfg_weight: Controls pacing/adherence to reference (0.0-1.0, default 0.5).
            Lower values = slower, more deliberate speech.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Emotion tags supported: [laugh], [chuckle], [cough], [sigh], [gasp]
    """
    result = generate(
        text=text,
        output_path=output_path,
        engine="chatterbox",
        reference_audio_paths=reference_audio_paths,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    return to_dict(result)


@mcp.tool()
def speak_mira(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
) -> dict:
    """Generate speech using MiraTTS (fast audio-prompted voice cloning).

    Fast voice cloning with high-quality 48kHz output.
    Over 100x realtime performance with only 6GB VRAM.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio (e.g., "/tmp/output.wav").
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. Clear speech samples work best.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: MiraTTS does not support emotion tags but produces high-quality 48kHz audio.
    """
    result = generate(
        text=text,
        output_path=output_path,
        engine="mira",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)


# ============================================================================
# Audio Utility Tools
# ============================================================================


@mcp.tool()
def get_audio_file_info(audio_path: str) -> dict:
    """Get information about an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dict with path, exists, format, duration_ms, size_bytes, and validity.
    """
    info = get_audio_info(audio_path)
    return to_dict(info)


@mcp.tool()
def convert_audio_format(
    input_path: str,
    output_format: str = "mp3",
    output_path: Optional[str] = None,
) -> dict:
    """Convert an audio file to a different format.

    Args:
        input_path: Path to the input audio file.
        output_format: Target format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        output_path: Optional output path. If not provided, creates a file
            with the same name but new extension in the same directory.

    Returns:
        Dict with input/output paths, formats, sizes, and compression ratio.
    """
    result = convert_audio(
        input_path=input_path,
        output_format=output_format,
        output_path=output_path,
    )
    return to_dict(result)


@mcp.tool()
def join_audio_files(
    audio_paths: list[str],
    output_path: str,
    output_format: str = "mp3",
) -> dict:
    """Concatenate multiple audio files into one.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'mp3'.

    Returns:
        Dict with output_path, input_count, total_duration_ms, and output_format.
    """
    result = concatenate_audio(
        audio_paths=audio_paths,
        output_path=output_path,
        output_format=output_format,
    )
    return to_dict(result)


@mcp.tool()
def normalize_audio_levels(
    input_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """Normalize audio levels to broadcast standard (-16 LUFS).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_normalized' suffix.

    Returns:
        Dict with input_path, output_path, and duration_ms.
    """
    result = normalize_audio(input_path=input_path, output_path=output_path)
    return to_dict(result)


@mcp.tool()
def check_ffmpeg_available() -> dict:
    """Check if ffmpeg is installed and available.

    ffmpeg is required for audio format conversion and concatenation.

    Returns:
        Dict with available status and install instructions if not available.
    """
    available = is_ffmpeg_available()
    return {
        "available": available,
        "message": "ffmpeg is available"
        if available
        else "ffmpeg not found. Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)",
    }


# ============================================================================
# Server Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
