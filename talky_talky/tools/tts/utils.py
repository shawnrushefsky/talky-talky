"""Shared utilities for TTS engines.

This module contains utilities used by multiple TTS engines:
- Emotion tag conversion
- Text chunking
- Device detection
- Memory utilities
- stdout redirection for MCP compatibility
"""

import re
import shutil
import sys
from contextlib import contextmanager
from typing import Optional


# ============================================================================
# stdout Redirection (MCP Protocol Compatibility)
# ============================================================================


@contextmanager
def redirect_stdout_to_stderr():
    """Redirect stdout to stderr to prevent library output from breaking MCP JSON protocol.

    Many TTS libraries print progress messages, loading status, etc. to stdout.
    Since MCP uses stdout for JSON-RPC communication, this output breaks the protocol.
    Use this context manager around library imports and model operations.

    Usage:
        with redirect_stdout_to_stderr():
            from some_tts_library import Model
            model = Model.from_pretrained("model-id")
            result = model.generate(text)
    """
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# ============================================================================
# Emotion Tag Conversion
# ============================================================================

# Maya1-style tags: <tag>
# Chatterbox-style tags: [tag]


def convert_angle_to_bracket_tags(text: str) -> str:
    """Convert <tag> format to [tag] format.

    Used when passing text to engines that use bracket notation.
    """
    pattern = r"<(\w+)>"
    replacement = r"[\1]"
    return re.sub(pattern, replacement, text)


def convert_bracket_to_angle_tags(text: str) -> str:
    """Convert [tag] format to <tag> format.

    Used when passing text to engines that use angle bracket notation.
    """
    pattern = r"\[(\w+)\]"
    replacement = r"<\1>"
    return re.sub(pattern, replacement, text)


# ============================================================================
# Text Chunking
# ============================================================================


def split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Tries to keep chunks under max_chars while respecting sentence boundaries.
    Never splits mid-sentence when possible.

    Args:
        text: The text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    # Split into sentences (handle various punctuation)
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$"
    sentences = re.split(sentence_pattern, text.strip())

    # Clean up empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [text] if text.strip() else []

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence is too long, we have to include it as its own chunk
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Try to split long sentence at commas or semicolons
            if len(sentence) > max_chars * 2:
                sub_parts = re.split(r"(?<=[,;:])\s+", sentence)
                sub_chunk = ""
                for part in sub_parts:
                    if len(sub_chunk) + len(part) + 1 <= max_chars:
                        sub_chunk = (sub_chunk + " " + part).strip() if sub_chunk else part
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = part
                if sub_chunk:
                    chunks.append(sub_chunk.strip())
            else:
                chunks.append(sentence.strip())
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================================
# Device Detection
# ============================================================================


def get_best_device() -> tuple[str, Optional[str], Optional[float]]:
    """Detect the best available device for PyTorch.

    Returns:
        (device_string, device_name, vram_gb)
        Priority: CUDA > MPS > CPU
    """
    try:
        import torch
    except ImportError:
        return ("cpu", "CPU (PyTorch not installed)", None)

    # Check CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return ("cuda", device_name, vram_gb)

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ("mps", "Apple Silicon (MPS)", None)

    # Fall back to CPU
    return ("cpu", "CPU", None)


def get_system_memory_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import subprocess
        import platform

        if platform.system() == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                bytes_mem = int(result.stdout.strip())
                return round(bytes_mem / (1024**3), 2)
        else:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
    except Exception:
        pass

    return 16.0  # Fallback


def get_available_memory_gb() -> tuple[float, str]:
    """Get available GPU/unified memory and memory type.

    Returns:
        (memory_gb, memory_type) where memory_type is 'cuda', 'mps', or 'cpu'.
    """
    device, _, vram_gb = get_best_device()

    if device == "cuda" and vram_gb:
        return (vram_gb, "cuda")

    if device == "mps":
        system_mem = get_system_memory_gb()
        return (system_mem, "mps")

    system_mem = get_system_memory_gb()
    return (system_mem, "cpu")


# ============================================================================
# Package Manager Detection
# ============================================================================


def detect_package_manager() -> dict:
    """Detect the available package manager and virtual environment setup."""
    info = {
        "in_virtualenv": False,
        "venv_path": None,
        "has_uv": shutil.which("uv") is not None,
        "has_pip": shutil.which("pip") is not None or shutil.which("pip3") is not None,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "install_command": "pip install",
        "extra_install_command": "pip install 'talky-talky[maya1]'",
    }

    # Check if in virtual environment
    info["in_virtualenv"] = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if info["in_virtualenv"]:
        info["venv_path"] = sys.prefix

    # Determine best install command
    if info["has_uv"]:
        info["install_command"] = "uv pip install"
        info["extra_install_command"] = "uv pip install 'talky-talky[maya1]'"
    elif info["has_pip"]:
        info["install_command"] = "pip install"
        info["extra_install_command"] = "pip install 'talky-talky[maya1]'"
    else:
        info["install_command"] = "python -m pip install"
        info["extra_install_command"] = "python -m pip install 'talky-talky[maya1]'"

    return info


# ============================================================================
# ffmpeg check
# ============================================================================


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None
