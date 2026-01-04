"""FFmpeg wrapper functions for audio processing."""

import subprocess
import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChapterMarker:
    title: str
    start_ms: int


@dataclass
class AudioValidation:
    valid: bool
    duration_ms: Optional[int] = None
    format: Optional[str] = None
    error: Optional[str] = None


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_audio_duration(file_path: str) -> int:
    """Get the duration of an audio file in milliseconds."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        seconds = float(result.stdout.strip())
        return round(seconds * 1000)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to get audio duration: {e.stderr}")


def validate_audio_file(file_path: str) -> AudioValidation:
    """Validate that an audio file exists and is readable."""
    if not os.path.exists(file_path):
        return AudioValidation(valid=False, error="File not found")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration,format_name",
                "-of", "json",
                file_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        probe = json.loads(result.stdout)

        return AudioValidation(
            valid=True,
            duration_ms=round(float(probe["format"]["duration"]) * 1000),
            format=probe["format"]["format_name"]
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        return AudioValidation(valid=False, error=f"Invalid audio file: {e}")


def concatenate_audio_files(
    input_files: list[str],
    output_path: str,
    format: str = "mp3"
) -> None:
    """Concatenate multiple audio files into one."""
    if not input_files:
        raise ValueError("No input files provided")

    # Validate all input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create a file list for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        list_path = f.name
        for file in input_files:
            # Escape single quotes in file paths
            escaped = file.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        # Set codec based on format
        if format == "mp3":
            codec = ["-c:a", "libmp3lame", "-q:a", "2"]
        elif format == "wav":
            codec = ["-c:a", "pcm_s16le"]
        elif format == "m4a":
            codec = ["-c:a", "aac", "-b:a", "192k"]
        else:
            codec = []

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                *codec,
                output_path
            ],
            capture_output=True,
            check=True
        )
    finally:
        os.unlink(list_path)


def create_audiobook_with_chapters(
    input_files: list[str],
    output_path: str,
    chapters: list[ChapterMarker],
    metadata: Optional[dict] = None
) -> None:
    """Create an MP3 with chapter markers (ID3v2 chapters)."""
    if not input_files:
        raise ValueError("No input files provided")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # First concatenate all files to a temp file
    temp_path = os.path.join(output_dir or ".", ".temp_concat.mp3")

    try:
        concatenate_audio_files(input_files, temp_path, "mp3")

        # Get total duration for the last chapter end
        total_duration = get_audio_duration(temp_path)

        # Create FFMETADATA file for chapters
        metadata_content = ";FFMETADATA1\n"

        if metadata:
            if metadata.get("title"):
                metadata_content += f"title={metadata['title']}\n"
            if metadata.get("artist"):
                metadata_content += f"artist={metadata['artist']}\n"
            if metadata.get("album"):
                metadata_content += f"album={metadata['album']}\n"

        # Add chapter markers
        for i, chapter in enumerate(chapters):
            start_ms = chapter.start_ms
            # End is either the start of the next chapter or total duration
            end_ms = chapters[i + 1].start_ms if i + 1 < len(chapters) else total_duration

            metadata_content += f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start_ms}\nEND={end_ms}\ntitle={chapter.title}\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            metadata_path = f.name
            f.write(metadata_content)

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", temp_path,
                    "-i", metadata_path,
                    "-map_metadata", "1",
                    "-c", "copy",
                    output_path
                ],
                capture_output=True,
                check=True
            )
        finally:
            os.unlink(metadata_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def convert_audio_format(
    input_path: str,
    output_path: str,
    format: str = "mp3"
) -> None:
    """Convert audio file to a specific format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if format == "mp3":
        codec = ["-c:a", "libmp3lame", "-q:a", "2"]
    elif format == "wav":
        codec = ["-c:a", "pcm_s16le"]
    elif format == "m4a":
        codec = ["-c:a", "aac", "-b:a", "192k"]
    else:
        codec = []

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, *codec, output_path],
        capture_output=True,
        check=True
    )


def normalize_audio(input_path: str, output_path: str) -> None:
    """Normalize audio levels (to -16 LUFS for podcast/audiobook standard)."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            output_path
        ],
        capture_output=True,
        check=True
    )
