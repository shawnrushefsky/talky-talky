"""Audio utility tools for format conversion, concatenation, and info.

These are standalone utilities with no project/database dependencies.
Agents can use these to manipulate audio files directly by path.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from ..utils.ffmpeg import (
    check_ffmpeg,
    validate_audio_file,
    get_audio_duration,
    concatenate_audio_files as _concat_files,
    convert_audio_format as _convert_format,
    normalize_audio as _normalize,
)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AudioInfo:
    """Information about an audio file."""

    path: str
    exists: bool
    format: Optional[str] = None
    duration_ms: Optional[int] = None
    size_bytes: Optional[int] = None
    valid: bool = False
    error: Optional[str] = None


@dataclass
class ConvertResult:
    """Result of an audio conversion operation."""

    input_path: str
    output_path: str
    input_format: str
    output_format: str
    input_size_bytes: int
    output_size_bytes: int
    compression_ratio: float
    duration_ms: int


@dataclass
class ConcatenateResult:
    """Result of audio concatenation."""

    output_path: str
    input_count: int
    total_duration_ms: int
    output_format: str


@dataclass
class NormalizeResult:
    """Result of audio normalization."""

    input_path: str
    output_path: str
    duration_ms: int


# ============================================================================
# Audio Utilities
# ============================================================================


def get_audio_info(audio_path: str) -> AudioInfo:
    """Get information about an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        AudioInfo with duration, format, size, and validity.
    """
    path = Path(audio_path)

    if not path.exists():
        return AudioInfo(
            path=audio_path,
            exists=False,
            error="File not found",
        )

    # Check if ffprobe is available
    if not check_ffmpeg():
        return AudioInfo(
            path=audio_path,
            exists=True,
            size_bytes=path.stat().st_size,
            valid=False,
            error="ffprobe not available - install ffmpeg to get audio metadata",
        )

    # Validate and get info
    validation = validate_audio_file(audio_path)

    if not validation.valid:
        return AudioInfo(
            path=audio_path,
            exists=True,
            valid=False,
            error=validation.error,
        )

    return AudioInfo(
        path=audio_path,
        exists=True,
        format=validation.format,
        duration_ms=validation.duration_ms,
        size_bytes=path.stat().st_size,
        valid=True,
    )


def convert_audio(
    input_path: str,
    output_format: str = "mp3",
    output_path: Optional[str] = None,
) -> ConvertResult:
    """Convert an audio file to a different format.

    Args:
        input_path: Path to the input audio file.
        output_format: Target format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        output_path: Optional output path. If not provided, creates a file
            with the same name but new extension in the same directory.

    Returns:
        ConvertResult with paths and size comparison.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
        ValueError: If output format is unsupported.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if output_format not in ("mp3", "wav", "m4a"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'mp3', 'wav', or 'm4a'.")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        output_path = str(input_path_obj.with_suffix(f".{output_format}"))

    output_path_obj = Path(output_path)

    # Get input info
    input_format = input_path_obj.suffix.lstrip(".")
    input_size = input_path_obj.stat().st_size

    # Convert
    _convert_format(input_path, output_path, output_format)

    # Get output info
    output_size = output_path_obj.stat().st_size
    duration = get_audio_duration(output_path)

    return ConvertResult(
        input_path=input_path,
        output_path=output_path,
        input_format=input_format,
        output_format=output_format,
        input_size_bytes=input_size,
        output_size_bytes=output_size,
        compression_ratio=round(input_size / output_size, 2) if output_size else 0,
        duration_ms=duration,
    )


def concatenate_audio(
    audio_paths: list[str],
    output_path: str,
    output_format: str = "mp3",
    silence_ms: int = 0,
) -> ConcatenateResult:
    """Concatenate multiple audio files into one.

    Args:
        audio_paths: List of paths to audio files to concatenate (in order).
        output_path: Path for the output file.
        output_format: Output format ('mp3', 'wav', 'm4a'). Default: 'mp3'.
        silence_ms: Milliseconds of silence to insert between clips (not yet implemented).

    Returns:
        ConcatenateResult with output info.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If any input file doesn't exist.
        ValueError: If no audio paths provided or format unsupported.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    if not audio_paths:
        raise ValueError("No audio paths provided")

    if output_format not in ("mp3", "wav", "m4a"):
        raise ValueError(f"Unsupported format: {output_format}. Use 'mp3', 'wav', or 'm4a'.")

    # Validate all input files exist
    for path in audio_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")

    # Create output directory if needed
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate
    _concat_files(audio_paths, output_path, output_format)

    # Get output info
    duration = get_audio_duration(output_path)

    return ConcatenateResult(
        output_path=output_path,
        input_count=len(audio_paths),
        total_duration_ms=duration,
        output_format=output_format,
    )


def normalize_audio(
    input_path: str,
    output_path: Optional[str] = None,
) -> NormalizeResult:
    """Normalize audio levels to broadcast standard (-16 LUFS).

    Args:
        input_path: Path to the input audio file.
        output_path: Optional output path. If not provided, creates a file
            with '_normalized' suffix.

    Returns:
        NormalizeResult with input_path, output_path, and duration_ms.

    Raises:
        RuntimeError: If ffmpeg is not installed.
        FileNotFoundError: If input file doesn't exist.
    """
    if not check_ffmpeg():
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    input_path_obj = Path(input_path)

    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine output path
    if output_path is None:
        stem = input_path_obj.stem
        suffix = input_path_obj.suffix
        output_path = str(input_path_obj.with_name(f"{stem}_normalized{suffix}"))

    # Normalize
    _normalize(input_path, output_path)

    duration = get_audio_duration(output_path)

    return NormalizeResult(
        input_path=input_path,
        output_path=output_path,
        duration_ms=duration,
    )


def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is installed and available.

    Returns:
        True if ffmpeg is available, False otherwise.
    """
    return check_ffmpeg()
