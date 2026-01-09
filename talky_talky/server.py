#!/usr/bin/env python3
"""Talky Talky - Text-to-Speech and Speech-to-Text MCP Server for AI Agents.

This MCP server provides TTS and transcription capabilities with pluggable engine support:

TTS Engines:
- Maya1: Text-prompted voice design (describe the voice you want)
- Chatterbox: Audio-prompted voice cloning (clone from reference audio)
- Chatterbox Turbo: Fast voice cloning optimized for production
- MiraTTS: Fast voice cloning with high-quality 48kHz output
- XTTS-v2: Multilingual voice cloning with 17 language support
- Kokoro: Voice selection from 54 pre-built voices across 8 languages
- Soprano: Ultra-fast CUDA TTS with 2000x realtime speed
- VibeVoice Realtime: Real-time TTS with ~300ms latency (Microsoft)
- VibeVoice Long-form: Long-form multi-speaker TTS up to 90 minutes (Microsoft)
- CosyVoice3: Zero-shot multilingual voice cloning with 9 languages (Alibaba)

Transcription Engines:
- Whisper: OpenAI's robust speech recognition (99+ languages)
- Faster-Whisper: CTranslate2-optimized Whisper (4x faster)

Plus audio utilities for format conversion and concatenation.
"""

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
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

# Import transcription module
from .tools.transcription import (
    check_transcription,
    get_transcription_info,
    get_available_engines as get_available_transcription_engines,
    list_engines as list_transcription_engines,
    transcribe,
)


# Server version
VERSION = "0.2.0"

# Initialize MCP server
mcp = FastMCP("talky-talky")


# ============================================================================
# Configuration
# ============================================================================

CONFIG_DIR = Path.home() / ".config" / "talky-talky"
CONFIG_FILE = CONFIG_DIR / "config.json"
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "talky-talky"


def _load_config() -> dict:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_config(config: dict) -> None:
    """Save configuration to file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_output_dir() -> Path:
    """Get the configured output directory, creating it if needed."""
    config = _load_config()
    output_dir = Path(config.get("output_directory", str(DEFAULT_OUTPUT_DIR)))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_output_path(output_path: str) -> str:
    """Resolve output path, using default directory if path is just a filename."""
    path = Path(output_path)
    # If it's just a filename (no directory components), use the configured output dir
    if path.parent == Path(".") or str(path.parent) == "":
        return str(get_output_dir() / path.name)
    # Otherwise, ensure the parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


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
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
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
        output_path=resolve_output_path(output_path),
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
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
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
        output_path=resolve_output_path(output_path),
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
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. Clear speech samples work best.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: MiraTTS does not support emotion tags but produces high-quality 48kHz audio.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="mira",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)


@mcp.tool()
def speak_xtts(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    language: str = "en",
) -> dict:
    """Generate speech using XTTS-v2 (multilingual voice cloning).

    Multilingual voice cloning supporting 17 languages with cross-language cloning.
    Only requires ~6 seconds of reference audio.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 6+ seconds of clear speech recommended.
        language: Target language code (default: "en").
            Supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: XTTS-v2 supports cross-language cloning - clone a voice from English
    audio and generate speech in Japanese, for example.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="xtts",
        reference_audio_paths=reference_audio_paths,
        language=language,
    )
    return to_dict(result)


@mcp.tool()
def speak_kokoro(
    text: str,
    output_path: str,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> dict:
    """Generate speech using Kokoro (voice selection from 54 pre-built voices).

    Lightweight, fast TTS with 54 high-quality voices across 8 languages.
    No voice cloning needed - select from pre-built voices.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        voice: Voice ID to use (default: "af_heart").
            Format: [lang][gender]_[name]
            Examples: af_heart (American Female Heart), bm_george (British Male George)
            Languages: a=American, b=British, j=Japanese, z=Mandarin, e=Spanish,
                      f=French, h=Hindi, i=Italian, p=Portuguese
        speed: Speech rate multiplier (default: 1.0). Range: 0.5-2.0.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Popular voices:
    - af_heart, af_bella (American English female, quality A)
    - am_fenrir, am_michael (American English male, quality B)
    - bf_emma, bm_george (British English)
    - jf_alpha, jm_kumo (Japanese)
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="kokoro",
        voice=voice,
        speed=speed,
    )
    return to_dict(result)


@mcp.tool()
def speak_soprano(
    text: str,
    output_path: str,
    temperature: float = 0.3,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
) -> dict:
    """Generate speech using Soprano (ultra-fast CUDA TTS).

    Ultra-lightweight 80M model with 2000x realtime speed and 32kHz output.
    Requires NVIDIA GPU with CUDA. No voice selection or cloning.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        temperature: Sampling temperature (default: 0.3). Lower = more consistent.
        top_p: Nucleus sampling parameter (default: 0.95).
        repetition_penalty: Penalty for repetition (default: 1.2).

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Soprano requires CUDA GPU. CPU and MPS are not supported.
    Best for batch processing where speed is critical.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="soprano",
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return to_dict(result)


@mcp.tool()
def speak_chatterbox_turbo(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
) -> dict:
    """Generate speech using Chatterbox Turbo (fast voice cloning).

    Streamlined 350M model optimized for low-latency voice cloning.
    Faster than standard Chatterbox with simpler API (no tuning parameters).
    Supports paralinguistic tags like [laugh], [chuckle], [cough].

    Args:
        text: The text to synthesize. Can include emotion tags.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 10+ seconds of clear speech recommended.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Emotion tags supported: [laugh], [chuckle], [cough]

    Note: For more control over expressiveness and pacing, use speak_chatterbox
    which has exaggeration and cfg_weight parameters.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="chatterbox_turbo",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)


@mcp.tool()
def speak_vibevoice_realtime(
    text: str,
    output_path: str,
    speaker_name: str = "Carter",
) -> dict:
    """Generate speech using VibeVoice Realtime (fast single-speaker TTS).

    Microsoft's real-time TTS with ~300ms first-audio latency.
    Single speaker, up to 10 minutes per generation.

    Args:
        text: The text to synthesize.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        speaker_name: Name of the speaker voice to use (default: "Carter").
            Available speakers: Carter, Emily, Nova, Michael, Sarah

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Primarily supports English. Other languages are experimental.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="vibevoice_realtime",
        voice_description=speaker_name,
    )
    return to_dict(result)


@mcp.tool()
def speak_vibevoice_longform(
    text: str,
    output_path: str,
    speaker_name: str = "Carter",
    speakers: Optional[list[str]] = None,
) -> dict:
    """Generate speech using VibeVoice Long-form (multi-speaker TTS).

    Microsoft's long-form TTS supporting up to 90 minutes and 4 speakers.
    Ideal for podcasts, audiobooks, and conversations.

    Args:
        text: The text to synthesize. Can include speaker labels for multi-speaker.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        speaker_name: Primary speaker name for single-speaker generation (default: "Carter").
        speakers: List of speaker names for multi-speaker generation (max 4).
            If provided, overrides speaker_name.

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Note: Supports English and Chinese. Use speaker labels in text for multi-speaker.
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="vibevoice_longform",
        voice_description=speaker_name,
        speakers=speakers,
    )
    return to_dict(result)


@mcp.tool()
def speak_cosyvoice(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],
    prompt_text: Optional[str] = None,
    instruction: Optional[str] = None,
    language: str = "auto",
) -> dict:
    """Generate speech using CosyVoice3 (multilingual voice cloning).

    Alibaba's zero-shot voice cloning with 9 languages and instruction control.
    Excellent for multilingual content and dialect control.

    Args:
        text: The text to synthesize. Can include [breath] tags for breathing sounds.
        output_path: Where to save the generated audio. Can be a full path or just
            a filename (e.g., "speech.wav") which will be saved to the configured
            output directory (default: ~/Documents/talky-talky).
        reference_audio_paths: Paths to reference audio files for voice cloning.
            At least one required. 5-10 seconds of clear speech recommended.
        prompt_text: Transcript of reference audio (improves quality).
        instruction: Natural language instruction for style control.
            Examples: "请用广东话表达。" (Cantonese), "请用尽可能快地语速说。" (fast speed)
        language: Target language code (default: "auto" for auto-detection).
            Supported: zh, en, ja, ko, de, es, fr, it, ru

    Returns:
        Dict with status, output_path, duration_ms, sample_rate, and metadata.

    Features:
    - 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
    - 18+ Chinese dialects via instruction control
    - Cross-lingual cloning (clone voice in one language, output in another)
    - [breath] tags for fine-grained breathing control
    """
    result = generate(
        text=text,
        output_path=resolve_output_path(output_path),
        engine="cosyvoice3",
        reference_audio_paths=reference_audio_paths,
        prompt_text=prompt_text,
        instruction=instruction,
        language=language,
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
def set_output_directory(directory: str) -> dict:
    """Set the default directory where audio files will be saved.

    When generating audio, if you provide just a filename (e.g., "speech.wav")
    instead of a full path, it will be saved to this directory.

    Args:
        directory: Path to the directory for saving audio files.
            Use "default" to reset to ~/Documents/talky-talky.

    Returns:
        Dict with status, the configured directory path, and whether it was created.
    """
    if directory.lower() == "default":
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(directory).expanduser().resolve()

    # Create the directory if it doesn't exist
    created = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to config
    config = _load_config()
    config["output_directory"] = str(output_dir)
    _save_config(config)

    return {
        "status": "success",
        "output_directory": str(output_dir),
        "created": created,
        "message": f"Audio files will be saved to: {output_dir}",
    }


@mcp.tool()
def get_output_directory() -> dict:
    """Get the current default directory where audio files are saved.

    Returns:
        Dict with the current output directory path and whether it exists.
    """
    output_dir = get_output_dir()
    return {
        "output_directory": str(output_dir),
        "exists": output_dir.exists(),
        "default": str(DEFAULT_OUTPUT_DIR),
        "is_default": str(output_dir) == str(DEFAULT_OUTPUT_DIR),
    }


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


@mcp.tool()
def play_audio(audio_path: str) -> dict:
    """Play an audio file using the system's default audio player.

    Opens the audio file with the platform's default application for audio playback.
    This is useful for previewing generated TTS audio.

    Args:
        audio_path: Path to the audio file to play.

    Returns:
        Dict with status and message indicating success or failure.

    Platform behavior:
    - macOS: Uses 'open' command (opens in default app like QuickTime/Music)
    - Linux: Uses 'xdg-open' command (opens in default audio player)
    - Windows: Uses 'start' command (opens in default app like Windows Media Player)
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return {
            "status": "error",
            "message": f"Audio file not found: {audio_path}",
        }

    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.Popen(["open", str(audio_path)])
        elif sys.platform == "win32":
            # Windows
            subprocess.Popen(["start", "", str(audio_path)], shell=True)
        else:
            # Linux and other Unix-like systems
            subprocess.Popen(["xdg-open", str(audio_path)])

        return {
            "status": "success",
            "message": f"Opened {audio_path.name} in default audio player",
            "path": str(audio_path),
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "message": f"Could not find system command to open audio: {e}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to play audio: {e}",
        }


# ============================================================================
# Transcription Tools
# ============================================================================


@mcp.tool()
def check_transcription_availability() -> dict:
    """Check if transcription engines are available and properly configured.

    Returns detailed status including:
    - Available engines
    - Device info (CUDA/MPS/CPU)
    - Setup instructions for unavailable engines
    """
    status = check_transcription()
    return to_dict(status)


@mcp.tool()
def get_transcription_engines_info() -> dict:
    """Get detailed information about all transcription engines.

    Returns info for each engine including:
    - Name and description
    - Requirements and availability
    - Supported languages and model sizes
    - Engine-specific parameters
    """
    return get_transcription_info()


@mcp.tool()
def list_available_transcription_engines() -> dict:
    """List transcription engines that are currently available (installed).

    Returns:
        Dict with list of available engine IDs and their basic info.
    """
    available = get_available_transcription_engines()
    engines = list_transcription_engines()

    return {
        "available_engines": available,
        "engines": {
            engine_id: {
                "name": info.name,
                "description": info.description,
                "supports_word_timestamps": info.supports_word_timestamps,
            }
            for engine_id, info in engines.items()
            if engine_id in available
        },
    }


@mcp.tool()
def transcribe_audio(
    audio_path: str,
    engine: str = "faster_whisper",
    language: Optional[str] = None,
    model_size: str = "base",
) -> dict:
    """Transcribe audio to text using speech recognition.

    Use this to verify TTS output or transcribe any audio file.
    Supports 99+ languages with automatic language detection.

    Args:
        audio_path: Path to the audio file to transcribe.
        engine: Transcription engine to use (default: "faster_whisper").
            Options: "whisper", "faster_whisper"
            - whisper: Best accuracy via transformers
            - faster_whisper: 4x faster with same accuracy (recommended)
        language: Language code (e.g., "en", "es", "ja"). Auto-detects if not specified.
        model_size: Model size (default: "base").
            Options: tiny, base, small, medium, large-v3, large-v3-turbo
            Larger models = more accurate but slower.

    Returns:
        Dict with:
        - status: "success" or "error"
        - text: Full transcribed text
        - segments: List of segments with timing info
        - language: Detected/used language
        - duration_seconds: Audio duration
        - processing_time_ms: How long transcription took

    Example:
        # Verify TTS output contains expected text
        result = transcribe_audio("generated_speech.wav")
        if "hello world" in result["text"].lower():
            print("TTS verification passed!")
    """
    result = transcribe(
        audio_path=audio_path,
        engine=engine,
        language=language,
        model_size=model_size,
    )
    return to_dict(result)


@mcp.tool()
def transcribe_with_timestamps(
    audio_path: str,
    engine: str = "faster_whisper",
    language: Optional[str] = None,
    model_size: str = "base",
    word_level: bool = False,
) -> dict:
    """Transcribe audio with detailed timing information.

    Returns segment-level or word-level timestamps for precise alignment.
    Useful for subtitles, karaoke, or audio analysis.

    Args:
        audio_path: Path to the audio file to transcribe.
        engine: Transcription engine ("whisper" or "faster_whisper").
        language: Language code (auto-detected if not specified).
        model_size: Model size (tiny, base, small, medium, large-v3, etc.)
        word_level: If True, include word-level timestamps (slower but more precise).

    Returns:
        Dict with:
        - status: "success" or "error"
        - text: Full transcribed text
        - segments: List of segments, each containing:
            - text: Segment text
            - start: Start time in seconds
            - end: End time in seconds
            - words: (if word_level=True) List of words with timing
        - language: Detected language
        - duration_seconds: Audio duration

    Note: Word-level timestamps are more accurate with faster_whisper engine.
    """
    # Set appropriate parameters for each engine
    if engine == "faster_whisper":
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
            word_timestamps=word_level,
            vad_filter=True,
        )
    else:
        # Whisper via transformers
        result = transcribe(
            audio_path=audio_path,
            engine=engine,
            language=language,
            model_size=model_size,
            return_timestamps="word" if word_level else True,
        )
    return to_dict(result)


@mcp.tool()
def verify_tts_output(
    audio_path: str,
    expected_text: str,
    engine: str = "faster_whisper",
    model_size: str = "base",
    similarity_threshold: float = 0.8,
) -> dict:
    """Verify that TTS-generated audio contains the expected text.

    Transcribes the audio and compares it to the expected text.
    Useful for automated testing of TTS output quality.

    Args:
        audio_path: Path to the TTS-generated audio file.
        expected_text: The text that should be in the audio.
        engine: Transcription engine to use (default: "faster_whisper").
        model_size: Model size (default: "base"). Use "large-v3" for best accuracy.
        similarity_threshold: Minimum similarity ratio to consider a match (0.0-1.0).
            Default 0.8 (80% similar). Lower for lenient matching.

    Returns:
        Dict with:
        - status: "success" or "error"
        - verified: True if transcription matches expected text
        - similarity: Similarity ratio (0.0 to 1.0)
        - expected_text: The expected text (normalized)
        - transcribed_text: What was actually transcribed (normalized)
        - match_details: Additional matching information

    Example:
        # After generating TTS audio
        result = verify_tts_output(
            audio_path="greeting.wav",
            expected_text="Hello, how are you today?",
        )
        if result["verified"]:
            print("TTS output verified successfully!")
    """
    from difflib import SequenceMatcher

    # Transcribe the audio
    transcription_result = transcribe(
        audio_path=audio_path,
        engine=engine,
        model_size=model_size,
    )

    if transcription_result.status == "error":
        return {
            "status": "error",
            "verified": False,
            "error": transcription_result.error,
        }

    # Normalize texts for comparison
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        import re

        # Convert to lowercase
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    expected_normalized = normalize_text(expected_text)
    transcribed_normalized = normalize_text(transcription_result.text)

    # Calculate similarity
    matcher = SequenceMatcher(None, expected_normalized, transcribed_normalized)
    similarity = matcher.ratio()

    # Check for exact word match (more lenient than character-level)
    expected_words = set(expected_normalized.split())
    transcribed_words = set(transcribed_normalized.split())
    word_overlap = len(expected_words & transcribed_words) / max(len(expected_words), 1)

    # Use the higher of the two similarity measures
    effective_similarity = max(similarity, word_overlap)

    verified = effective_similarity >= similarity_threshold

    return {
        "status": "success",
        "verified": verified,
        "similarity": round(effective_similarity, 3),
        "character_similarity": round(similarity, 3),
        "word_overlap": round(word_overlap, 3),
        "expected_text": expected_normalized,
        "transcribed_text": transcribed_normalized,
        "threshold": similarity_threshold,
        "match_details": {
            "expected_word_count": len(expected_words),
            "transcribed_word_count": len(transcribed_words),
            "matching_words": len(expected_words & transcribed_words),
        },
        "transcription_metadata": {
            "engine": engine,
            "model_size": model_size,
            "language": transcription_result.language,
            "processing_time_ms": transcription_result.processing_time_ms,
        },
    }


# ============================================================================
# Server Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
