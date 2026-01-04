"""Text-to-Speech tools with Maya1 and Fish Speech integration.

This module provides TTS capabilities using:
- Maya1: Voice design via natural language descriptions + emotion tags
- Fish Speech: Voice cloning from reference audio samples (local or cloud API)
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ..db.connection import get_database, get_audiobook_dir, get_current_project_path, get_db_lock
from .segments import get_segment
from .characters import get_character
from .voice_samples import add_voice_sample, list_voice_samples


# ============================================================================
# Voice Presets and Constants
# ============================================================================

EMOTION_TAGS = [
    "laugh",
    "laugh_harder",
    "chuckle",
    "giggle",
    "snort",
    "cry",
    "sob",
    "sigh",
    "gasp",
    "groan",
    "whisper",
    "angry",
    "yell",
    "scream",
    "cough",
    "clear_throat",
    "sniff",
    "hum",
    "mumble",
    "stutter",
]

VOICE_PRESETS = {
    "narrator_male": "Realistic male voice in the 40s age with american accent. Low pitch, warm timbre, measured pacing, professional tone.",
    "narrator_female": "Realistic female voice in the 30s age with american accent. Medium pitch, warm timbre, measured pacing, professional tone.",
    "young_male": "Realistic male voice in the 20s age with american accent. Medium-high pitch, bright timbre, energetic pacing, enthusiastic tone.",
    "young_female": "Realistic female voice in the 20s age with american accent. High pitch, bright timbre, energetic pacing, enthusiastic tone.",
    "old_male": "Realistic male voice in the 60s age with american accent. Low pitch, gravelly timbre, slow pacing, wise tone.",
    "old_female": "Realistic female voice in the 60s age with american accent. Medium pitch, gentle timbre, measured pacing, warm tone.",
    "child_male": "Realistic male voice in the 10s age with american accent. High pitch, bright timbre, fast pacing, excited tone.",
    "child_female": "Realistic female voice in the 10s age with american accent. High pitch, bright timbre, fast pacing, excited tone.",
    "villain": "Realistic male voice in the 40s age with british accent. Low pitch, cold timbre, slow pacing, menacing tone.",
    "hero": "Realistic male voice in the 30s age with american accent. Medium-low pitch, strong timbre, confident pacing, determined tone.",
    "mysterious": "Realistic voice in the 30s age. Low pitch, hushed timbre, slow pacing, enigmatic tone.",
}

# Voice description options for the builder
VOICE_GENDERS = ["male", "female"]
VOICE_AGES = ["10s", "20s", "30s", "40s", "50s", "60s", "70s"]
VOICE_ACCENTS = ["american", "british", "australian", "irish", "scottish", "indian"]
VOICE_PITCHES = ["low", "medium-low", "medium", "medium-high", "high"]
VOICE_TIMBRES = ["warm", "cold", "bright", "gravelly", "gentle", "strong", "smooth", "husky"]
VOICE_PACINGS = ["slow", "measured", "moderate", "energetic", "fast"]
VOICE_TONES = [
    "professional",
    "friendly",
    "menacing",
    "wise",
    "enthusiastic",
    "mysterious",
    "warm",
    "determined",
    "calm",
    "excited",
]

DEFAULT_DESCRIPTION = VOICE_PRESETS["narrator_female"]
SAMPLE_RATE = 24000

# Maya1 SNAC token format constants
SNAC_TOKENS_PER_FRAME = 7  # Maya1 outputs 7 tokens per audio frame
CODE_TOKEN_OFFSET = 128266  # Offset for SNAC codes in Maya1 vocabulary
CODE_START_TOKEN_ID = 128257  # Start of speech token
CODE_END_TOKEN_ID = 128258  # End of speech token

# Maya1 special tokens for prompt construction
SOH_ID = 128259  # Start of header
EOH_ID = 128260  # End of header
SOA_ID = 128261  # Start of audio
TEXT_EOT_ID = 128009  # End of text

# Chatterbox TTS settings
CHATTERBOX_DEFAULT_EXAGGERATION = 0.5
CHATTERBOX_DEFAULT_CFG_WEIGHT = 0.5
CHATTERBOX_MAX_DURATION_SECS = 40  # Model has ~40 second max duration

# Chunking settings based on VRAM/RAM availability
# Characters per chunk for different memory tiers
CHUNK_SIZE_LOW_VRAM = 200  # For systems with < 8GB VRAM
CHUNK_SIZE_MEDIUM_VRAM = 500  # For systems with 8-16GB VRAM
CHUNK_SIZE_HIGH_VRAM = 1000  # For systems with 16-32GB VRAM
CHUNK_SIZE_VERY_HIGH_VRAM = 2000  # For systems with 32GB+ VRAM

# Model identifiers for downloading
MAYA1_MODEL_ID = "maya-research/maya1"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


# ============================================================================
# TTS Engine Management
# ============================================================================

# Global TTS engine instances (loaded lazily)
_maya1_model = None
_maya1_tokenizer = None
_snac_model = None


def _get_project_audio_dir() -> Path:
    """Get the audio directory for the current project."""
    project_path = get_current_project_path()
    if not project_path:
        raise ValueError("No project is currently open. Use open_project first.")
    return Path(get_audiobook_dir(project_path))


@dataclass
class PackageManagerInfo:
    """Information about the Python package manager environment."""

    in_virtualenv: bool = False
    venv_path: Optional[str] = None
    has_uv: bool = False
    has_pip: bool = False
    python_version: str = ""
    install_command: str = "pip install"  # Best command to use
    extra_install_command: str = "pip install 'audiobook-mcp[maya1]'"


def detect_package_manager() -> PackageManagerInfo:
    """Detect the available package manager and virtual environment setup."""
    info = PackageManagerInfo()

    # Python version
    info.python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # Check if in virtual environment
    info.in_virtualenv = (
        hasattr(sys, "real_prefix")  # virtualenv
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)  # venv
    )
    if info.in_virtualenv:
        info.venv_path = sys.prefix

    # Check for uv
    info.has_uv = shutil.which("uv") is not None

    # Check for pip
    info.has_pip = shutil.which("pip") is not None or shutil.which("pip3") is not None

    # Determine best install command
    if info.has_uv:
        info.install_command = "uv pip install"
        info.extra_install_command = "uv pip install 'audiobook-mcp[maya1]'"
    elif info.has_pip:
        info.install_command = "pip install"
        info.extra_install_command = "pip install 'audiobook-mcp[maya1]'"
    else:
        info.install_command = "python -m pip install"
        info.extra_install_command = "python -m pip install 'audiobook-mcp[maya1]'"

    return info


@dataclass
class TTSCheckResult:
    status: str  # "ok" or "error"
    maya1_available: bool = False
    chatterbox_available: bool = False
    torch_installed: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    ffmpeg_available: bool = False
    device: Optional[str] = None
    device_name: Optional[str] = None
    vram_gb: Optional[float] = None
    system_memory_gb: Optional[float] = None
    chunk_size_chars: Optional[int] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    setup_instructions: dict = field(default_factory=dict)
    package_manager: Optional[PackageManagerInfo] = None


# Installation and setup instructions
MAYA1_SETUP_INSTRUCTIONS = """
## Maya1 Setup (Voice Design)

Maya1 requires PyTorch and related dependencies. Install with:

```bash
pip install "audiobook-mcp[maya1]"
```

Or manually:
```bash
pip install torch transformers snac
```

**Hardware Requirements:**
- NVIDIA GPU with CUDA: Best performance (16GB+ VRAM recommended)
- Apple Silicon (M1/M2/M3/M4): Supported via MPS (slower but works)
- CPU: Supported but slow (not recommended for batch generation)
"""

CHATTERBOX_SETUP_INSTRUCTIONS = """
## Chatterbox TTS Setup (Voice Cloning with Emotion Control)

Chatterbox is a high-quality TTS with zero-shot voice cloning and emotion control.
Supports paralinguistic tags like [laugh], [cough], [chuckle] for expressive speech.

### Installation

```bash
pip install chatterbox-tts
```

Or with uv:
```bash
uv pip install chatterbox-tts
```

### Hardware Requirements

- NVIDIA GPU with CUDA (recommended): Best performance
- Apple Silicon (MPS): Supported with some limitations
- CPU: Supported but slower

### Model Variants

- **ChatterboxTurboTTS**: Fastest, 350M parameters
- **ChatterboxTTS**: Standard English model
- **ChatterboxMultilingualTTS**: 23+ languages

### Key Features

- **Exaggeration parameter**: Controls speech expressiveness (0.0-1.0)
- **cfg_weight**: Controls pacing (lower = slower, more deliberate)
- **Paralinguistic tags**: [laugh], [cough], [chuckle], [sigh], etc.

### Usage Notes

- Works with 10+ seconds of reference audio
- Use exaggeration=0.7+ for dramatic characters
- Lower cfg_weight (~0.3) for slower, deliberate pacing
"""


@dataclass
class ModelStatus:
    """Status of a model's availability."""

    model_id: str
    downloaded: bool
    cache_path: Optional[str] = None
    size_gb: Optional[float] = None
    error: Optional[str] = None


def check_model_downloaded(model_id: str) -> ModelStatus:
    """Check if a HuggingFace model is downloaded to the cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir

        # Try to load config to see if model is cached
        try:
            config_path = try_to_load_from_cache(model_id, "config.json")
            if config_path is not None:
                # Model is cached, get more info
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == model_id:
                        size_gb = round(repo.size_on_disk / (1024**3), 2)
                        return ModelStatus(
                            model_id=model_id,
                            downloaded=True,
                            cache_path=str(repo.repo_path),
                            size_gb=size_gb,
                        )
                return ModelStatus(model_id=model_id, downloaded=True)
            else:
                return ModelStatus(model_id=model_id, downloaded=False)
        except Exception:
            return ModelStatus(model_id=model_id, downloaded=False)

    except ImportError:
        return ModelStatus(
            model_id=model_id,
            downloaded=False,
            error="huggingface_hub not installed",
        )


def download_maya1_models(force: bool = False) -> dict:
    """Download Maya1 model weights from HuggingFace.

    Downloads both the Maya1 language model and SNAC audio codec.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results for each model.
    """
    import sys

    results = {
        "maya1": {"status": "pending", "model_id": MAYA1_MODEL_ID},
        "snac": {"status": "pending", "model_id": SNAC_MODEL_ID},
    }

    # Check if already downloaded
    if not force:
        maya1_status = check_model_downloaded(MAYA1_MODEL_ID)
        snac_status = check_model_downloaded(SNAC_MODEL_ID)

        if maya1_status.downloaded and snac_status.downloaded:
            return {
                "status": "already_downloaded",
                "message": "All Maya1 models are already downloaded",
                "models": {
                    "maya1": {
                        "status": "cached",
                        "model_id": MAYA1_MODEL_ID,
                        "cache_path": maya1_status.cache_path,
                        "size_gb": maya1_status.size_gb,
                    },
                    "snac": {
                        "status": "cached",
                        "model_id": SNAC_MODEL_ID,
                        "cache_path": snac_status.cache_path,
                        "size_gb": snac_status.size_gb,
                    },
                },
            }

    # Download Maya1 model
    try:
        print(f"Downloading Maya1 model ({MAYA1_MODEL_ID})...", file=sys.stderr, flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # This will download and cache the model
        AutoTokenizer.from_pretrained(MAYA1_MODEL_ID)
        AutoModelForCausalLM.from_pretrained(MAYA1_MODEL_ID)

        results["maya1"]["status"] = "downloaded"
        print("Maya1 model downloaded successfully", file=sys.stderr, flush=True)

    except Exception as e:
        results["maya1"]["status"] = "error"
        results["maya1"]["error"] = str(e)
        print(f"Error downloading Maya1: {e}", file=sys.stderr, flush=True)

    # Download SNAC codec
    try:
        print(f"Downloading SNAC codec ({SNAC_MODEL_ID})...", file=sys.stderr, flush=True)
        import snac

        # This will download and cache the model
        snac.SNAC.from_pretrained(SNAC_MODEL_ID)

        results["snac"]["status"] = "downloaded"
        print("SNAC codec downloaded successfully", file=sys.stderr, flush=True)

    except Exception as e:
        results["snac"]["status"] = "error"
        results["snac"]["error"] = str(e)
        print(f"Error downloading SNAC: {e}", file=sys.stderr, flush=True)

    # Determine overall status
    all_success = all(r["status"] in ("downloaded", "cached") for r in results.values())
    any_error = any(r["status"] == "error" for r in results.values())

    return {
        "status": "success" if all_success else ("partial" if not any_error else "error"),
        "message": "Models downloaded successfully"
        if all_success
        else "Some models failed to download",
        "models": results,
    }


def get_model_status() -> dict:
    """Get the download status of all TTS models.

    Returns information about which models are downloaded and their cache locations.
    """
    maya1_status = check_model_downloaded(MAYA1_MODEL_ID)
    snac_status = check_model_downloaded(SNAC_MODEL_ID)

    models = {
        "maya1": {
            "model_id": MAYA1_MODEL_ID,
            "downloaded": maya1_status.downloaded,
            "cache_path": maya1_status.cache_path,
            "size_gb": maya1_status.size_gb,
            "required_for": "voice design (creating voices from descriptions)",
        },
        "snac": {
            "model_id": SNAC_MODEL_ID,
            "downloaded": snac_status.downloaded,
            "cache_path": snac_status.cache_path,
            "size_gb": snac_status.size_gb,
            "required_for": "audio decoding (used with Maya1)",
        },
    }

    all_downloaded = maya1_status.downloaded and snac_status.downloaded

    return {
        "all_downloaded": all_downloaded,
        "total_size_gb": sum(m.get("size_gb") or 0 for m in models.values()),
        "models": models,
        "download_instructions": None
        if all_downloaded
        else (
            "Use the download_tts_models tool to download missing models, "
            "or run: pip install audiobook-mcp[maya1] && python -c "
            "'from audiobook_mcp.tools.tts import download_maya1_models; download_maya1_models()'"
        ),
    }


def _get_best_device() -> tuple[str, Optional[str], Optional[float]]:
    """Detect the best available device for PyTorch.

    Returns (device_string, device_name, vram_gb)
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


def _get_system_memory_gb() -> float:
    """Get total system RAM in GB.

    For Apple Silicon, this is unified memory shared with GPU.
    """
    try:
        import subprocess
        import platform

        if platform.system() == "Darwin":
            # macOS - get total physical memory
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
            # Linux - use /proc/meminfo
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 2)
    except Exception:
        pass

    # Fallback: assume 16GB
    return 16.0


def _get_available_memory_gb() -> tuple[float, str]:
    """Get available GPU/unified memory and memory type.

    Returns (memory_gb, memory_type) where memory_type is 'cuda', 'mps', or 'cpu'.

    For Apple Silicon, returns total unified memory (available to both CPU and GPU).
    For CUDA, returns dedicated VRAM.
    For CPU-only, returns system RAM.
    """
    device, _, vram_gb = _get_best_device()

    if device == "cuda" and vram_gb:
        return (vram_gb, "cuda")

    if device == "mps":
        # Apple Silicon unified memory - GPU can use all of it
        system_mem = _get_system_memory_gb()
        return (system_mem, "mps")

    # CPU fallback
    system_mem = _get_system_memory_gb()
    return (system_mem, "cpu")


def _get_optimal_chunk_size() -> int:
    """Determine optimal chunk size (in characters) based on available memory.

    Returns recommended max characters per chunk.
    """
    memory_gb, memory_type = _get_available_memory_gb()

    if memory_gb >= 64:
        # Very high memory (64GB+ unified or VRAM) - can handle large chunks
        return CHUNK_SIZE_VERY_HIGH_VRAM
    elif memory_gb >= 32:
        # High memory (32GB+)
        return CHUNK_SIZE_VERY_HIGH_VRAM
    elif memory_gb >= 16:
        # Good memory (16-32GB)
        return CHUNK_SIZE_HIGH_VRAM
    elif memory_gb >= 8:
        # Medium memory (8-16GB)
        return CHUNK_SIZE_MEDIUM_VRAM
    else:
        # Low memory (< 8GB)
        return CHUNK_SIZE_LOW_VRAM


def _split_text_into_chunks(text: str, max_chars: int) -> list[str]:
    """Split text into chunks at sentence boundaries.

    Tries to keep chunks under max_chars while respecting sentence boundaries.
    Never splits mid-sentence.
    """
    import re

    # Split into sentences (handle various punctuation)
    # This regex matches sentences ending with . ! ? followed by space or end
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
                # Really long sentence - split at punctuation
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
            # Add to current chunk
            current_chunk = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        else:
            # Start new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def check_tts() -> TTSCheckResult:
    """Check if TTS engines are available and properly configured.

    Returns detailed status including setup instructions for any unavailable engines.
    """
    result = TTSCheckResult(status="ok")

    # Detect package manager environment
    pkg_info = detect_package_manager()
    result.package_manager = pkg_info

    # Check PyTorch and device
    try:
        import torch

        result.torch_installed = True
        result.cuda_available = torch.cuda.is_available()
        result.mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        device, device_name, vram_gb = _get_best_device()
        result.device = device
        result.device_name = device_name
        result.vram_gb = vram_gb

        # Get system memory and optimal chunk size
        result.system_memory_gb = _get_system_memory_gb()
        result.chunk_size_chars = _get_optimal_chunk_size()

        if device == "cpu":
            result.warnings.append(
                "No GPU detected - Maya1 will run on CPU (slower but functional)"
            )
    except ImportError:
        result.warnings.append(
            f"PyTorch not installed - Maya1 unavailable. "
            f"Install with: {pkg_info.install_command} torch"
        )
        result.setup_instructions["maya1"] = MAYA1_SETUP_INSTRUCTIONS

    # Check Maya1 dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        import snac  # noqa: F401

        result.maya1_available = True
    except ImportError as e:
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
        result.warnings.append(
            f"Maya1 dependency missing: {missing_pkg}. "
            f"Install with: {pkg_info.extra_install_command}"
        )
        result.setup_instructions["maya1"] = MAYA1_SETUP_INSTRUCTIONS

    # Check Chatterbox TTS (voice cloning with emotion control)
    try:
        from chatterbox.tts import ChatterboxTTS  # noqa: F401

        result.chatterbox_available = True
    except ImportError:
        result.warnings.append(
            f"chatterbox-tts not installed - high-quality voice cloning with emotion control. "
            f"Install with: {pkg_info.install_command} chatterbox-tts"
        )
        result.setup_instructions["chatterbox"] = CHATTERBOX_SETUP_INSTRUCTIONS

    # Add setup instructions for voice cloning if Chatterbox not available
    if not result.chatterbox_available:
        result.setup_instructions["chatterbox"] = CHATTERBOX_SETUP_INSTRUCTIONS

    # Check ffmpeg availability (needed for audio stitching)
    result.ffmpeg_available = shutil.which("ffmpeg") is not None
    if not result.ffmpeg_available:
        result.warnings.append(
            "ffmpeg not installed - required for audio stitching. "
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    # Determine overall status
    if not result.maya1_available and not result.chatterbox_available:
        result.status = "error"
        result.errors.append(
            "No TTS engines available. See setup_instructions for how to configure them."
        )
    elif not result.chatterbox_available:
        result.status = "partial"
        result.warnings.append(
            "Only Maya1 available. Chatterbox needed for voice cloning and long-form generation. See setup_instructions."
        )
    elif not result.maya1_available:
        result.status = "partial"
        result.warnings.append(
            "Only Chatterbox available. Maya1 needed for voice design. See setup_instructions."
        )

    return result


def list_tts_info() -> dict:
    """List available TTS engines, emotion tags, and voice presets."""
    return {
        "emotion_tags": EMOTION_TAGS,
        "emotion_usage": {
            "description": "Maya1 supports 20+ inline emotion tags. Insert them where you want emotional expression.",
            "supported": EMOTION_TAGS,
            "examples": [
                "I can't believe it! <laugh>",
                "<whisper> Don't tell anyone...",
                "NO! <angry> I won't do it!",
                "Our new update <laugh> finally ships with the feature you asked for.",
                "<sigh> I suppose we should get started.",
            ],
        },
        "voice_presets": VOICE_PRESETS,
        "description_format": {
            "description": "Maya1 understands natural language voice descriptions. Describe voices like briefing a voice actor.",
            "template": "Realistic {gender} voice in the {age} age with {accent} accent. {pitch} pitch, {timbre} timbre, {pacing} pacing, {tone} tone.",
            "examples": [
                "Female, in her 30s with an American accent and is an event host, energetic, clear diction",
                "Dark villain character, Male voice in their 40s with a British accent. low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
                "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
                "40-year-old, warm, low pitch, conversational",
                "Young enthusiastic female narrator with a bright, energetic delivery",
            ],
            "parameters": {
                "age": {
                    "suggestions": VOICE_AGES,
                    "examples": ["30s", "40-year-old", "late 20s", "elderly"],
                },
                "gender": {
                    "suggestions": VOICE_GENDERS,
                    "examples": ["male", "female", "gender-neutral"],
                },
                "accent": {
                    "suggestions": VOICE_ACCENTS,
                    "examples": ["American accent", "British accent", "Middle Eastern accent"],
                },
                "pitch": {
                    "suggestions": VOICE_PITCHES,
                    "examples": ["low pitch", "high pitch", "normal pitch"],
                },
                "timbre": {
                    "suggestions": VOICE_TIMBRES,
                    "examples": ["warm baritone", "gravelly", "clear diction", "bright"],
                },
                "pacing": {
                    "suggestions": VOICE_PACINGS,
                    "examples": ["conversational pacing", "slow pacing", "fast pacing"],
                },
                "tone": {
                    "suggestions": VOICE_TONES,
                    "examples": ["energetic", "calm", "menacing", "professional"],
                },
            },
        },
        "engines": {
            "maya1": {
                "name": "Maya1",
                "description": "Voice design via natural language descriptions with 20+ emotion tags",
                "use_case": "Creating unique voices from text descriptions",
                "requirements": "torch, transformers, snac",
                "capabilities": [
                    "Natural language voice descriptions",
                    "20+ inline emotion tags (laugh, whisper, angry, etc.)",
                    "Character voices (villain, narrator, etc.)",
                    "Accent and dialect support",
                ],
            },
            "chatterbox": {
                "name": "Chatterbox TTS",
                "description": "Voice cloning with emotion control and paralinguistic tags",
                "use_case": "Expressive voice cloning with emotion control",
                "requirements": "chatterbox-tts (pip install chatterbox-tts)",
                "capabilities": [
                    "Zero-shot voice cloning from reference audio",
                    "Paralinguistic tags: [laugh], [cough], [chuckle], [sigh]",
                    "Exaggeration control for expressiveness",
                    "cfg_weight for pacing control",
                    "23+ language support (multilingual model)",
                    "CUDA, MPS, and CPU support",
                ],
                "parameters": {
                    "exaggeration": "0.0-1.0, controls expressiveness (default 0.5)",
                    "cfg_weight": "Controls pacing, lower = slower (default 0.5)",
                },
            },
        },
    }


@dataclass
class VoiceDescriptionResult:
    """Result of building a voice description."""

    description: str
    gender: str
    age: str
    accent: str
    pitch: str
    timbre: str
    pacing: str
    tone: str
    warnings: list = field(default_factory=list)


def build_voice_description(
    gender: str = "female",
    age: str = "30s",
    accent: str = "american",
    pitch: str = "medium",
    timbre: str = "warm",
    pacing: str = "measured",
    tone: str = "professional",
) -> VoiceDescriptionResult:
    """Build a Maya1 voice description from individual parameters.

    Constructs a description string in the recommended format. Parameters
    are suggestions - Maya1 can understand any descriptive language, so
    custom values will work too.

    Args:
        gender: Voice gender (suggestions: male, female, or any descriptive term)
        age: Age range (suggestions: 10s, 20s, 30s, 40s, 50s, 60s, 70s, or descriptive like "elderly", "teenage")
        accent: Accent type (suggestions: american, british, australian, irish, scottish, indian, or any accent)
        pitch: Voice pitch (suggestions: low, medium-low, medium, medium-high, high, or descriptive)
        timbre: Voice timbre (suggestions: warm, cold, bright, gravelly, gentle, strong, smooth, husky, or descriptive)
        pacing: Speech pacing (suggestions: slow, measured, moderate, energetic, fast, or descriptive)
        tone: Voice tone (suggestions: professional, friendly, menacing, wise, enthusiastic, mysterious, or descriptive)

    Returns:
        VoiceDescriptionResult with the constructed description and all parameters.
    """
    warnings = []

    # Provide gentle suggestions for non-standard values (but don't error)
    if gender.lower() not in VOICE_GENDERS:
        warnings.append(f"Custom gender '{gender}' used. Common values: {VOICE_GENDERS}")

    if age.lower() not in VOICE_AGES:
        warnings.append(f"Custom age '{age}' used. Common values: {VOICE_AGES}")

    if accent.lower() not in VOICE_ACCENTS:
        warnings.append(f"Custom accent '{accent}' used. Common values: {VOICE_ACCENTS}")

    if pitch.lower() not in VOICE_PITCHES:
        warnings.append(f"Custom pitch '{pitch}' used. Common values: {VOICE_PITCHES}")

    if timbre.lower() not in VOICE_TIMBRES:
        warnings.append(f"Custom timbre '{timbre}' used. Common values: {VOICE_TIMBRES}")

    if pacing.lower() not in VOICE_PACINGS:
        warnings.append(f"Custom pacing '{pacing}' used. Common values: {VOICE_PACINGS}")

    if tone.lower() not in VOICE_TONES:
        warnings.append(f"Custom tone '{tone}' used. Common values: {VOICE_TONES}")

    # Build description
    description = (
        f"Realistic {gender} voice in the {age} age with {accent} accent. "
        f"{pitch.capitalize()} pitch, {timbre} timbre, {pacing} pacing, {tone} tone."
    )

    return VoiceDescriptionResult(
        description=description,
        gender=gender,
        age=age,
        accent=accent,
        pitch=pitch,
        timbre=timbre,
        pacing=pacing,
        tone=tone,
        warnings=warnings,
    )


# ============================================================================
# Maya1 TTS Engine
# ============================================================================


def _unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels.

    Maya1 outputs audio tokens in a packed 7-token-per-frame format.
    SNAC decoder expects 3 separate lists for different temporal resolutions:
    - L1: ~12 Hz (1 token per frame)
    - L2: ~23 Hz (2 tokens per frame)
    - L3: ~47 Hz (4 tokens per frame)
    """
    # Remove end token if present
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend(
            [
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )

    return [l1, l2, l3]


def _load_maya1():
    """Lazily load Maya1 model."""
    global _maya1_model, _maya1_tokenizer, _snac_model

    if _maya1_model is not None:
        return _maya1_model, _maya1_tokenizer, _snac_model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import snac
    import sys

    # Auto-detect best device
    device, device_name, _ = _get_best_device()

    # Select appropriate dtype for device
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        # MPS works best with float16 or float32
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading Maya1 model on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Load Maya1 model
    _maya1_tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")

    if device == "cuda":
        # Use device_map for CUDA
        _maya1_model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        # For MPS and CPU, load then move to device
        _maya1_model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            torch_dtype=dtype,
        ).to(device)

    # Load SNAC decoder
    _snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

    print(f"Maya1 model loaded successfully on {device}", file=sys.stderr, flush=True)
    return _maya1_model, _maya1_tokenizer, _snac_model


def _build_maya1_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1 using the correct XML-attribute format.

    Maya1 expects a specific prompt structure with special tokens:
    [SOH][BOS]<description="..."> text[EOT][EOH][SOA][SOS]
    """
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    # Format: <description="voice description"> text content
    formatted_text = f'<description="{description}"> {text}'

    prompt = soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token

    return prompt


def generate_with_maya1(
    text: str,
    description: str,
    output_path: Path,
) -> dict:
    """Generate audio using Maya1 TTS."""
    import torch
    import soundfile as sf

    model, tokenizer, snac_model = _load_maya1()
    device = next(model.parameters()).device

    # Build prompt using correct Maya1 format
    prompt = _build_maya1_prompt(tokenizer, description, text)

    # Generate with recommended parameters from Maya1 documentation
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # Allow longer generations
            min_new_tokens=28,  # At least 4 SNAC frames
            do_sample=True,
            temperature=0.4,  # Lower temperature for more stable output
            top_p=0.9,
            repetition_penalty=1.1,  # Prevent loops
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Extract audio tokens (remove the input prompt tokens)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :].tolist()

    # Unpack 7-token frames into 3 hierarchical levels for SNAC
    levels = _unpack_snac_from_7(generated_ids)

    if not levels[0]:
        raise ValueError("No audio tokens generated - model may have produced empty output")

    # Convert to tensors for SNAC decoder
    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) for level in levels
    ]

    # Decode with SNAC
    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)

    audio_np = audio[0, 0].cpu().numpy()

    # Trim warmup artifacts (first ~85ms at 24kHz)
    if len(audio_np) > 2048:
        audio_np = audio_np[2048:]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_np, SAMPLE_RATE)

    duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

    return {
        "status": "success",
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "sample_rate": SAMPLE_RATE,
    }


# ============================================================================
# Chatterbox TTS Engine (Voice cloning with emotion control)
# ============================================================================


def _get_absolute_sample_path(sample_path: str) -> str:
    """Convert relative sample path to absolute path."""
    if sample_path.startswith("http://") or sample_path.startswith("https://"):
        return sample_path

    if os.path.isabs(sample_path):
        return sample_path

    # Relative path - resolve from project directory
    audio_dir = _get_project_audio_dir()
    return str(audio_dir / sample_path)


# Global Chatterbox model instance (loaded lazily)
_chatterbox_model = None


def _get_chatterbox_device() -> str:
    """Get the best device for Chatterbox."""
    device, _, _ = _get_best_device()
    return device


def _load_chatterbox():
    """Lazily load Chatterbox model."""
    global _chatterbox_model

    if _chatterbox_model is not None:
        return _chatterbox_model

    import sys

    device = _get_chatterbox_device()
    print(f"Loading Chatterbox TTS model on {device}...", file=sys.stderr, flush=True)

    from chatterbox.tts import ChatterboxTTS

    _chatterbox_model = ChatterboxTTS.from_pretrained(device=device)

    print("Chatterbox TTS model loaded successfully", file=sys.stderr, flush=True)
    return _chatterbox_model


def _concatenate_audio_tensors(audio_tensors: list, sample_rate: int, silence_ms: int = 100):
    """Concatenate audio tensors with optional silence between them.

    Args:
        audio_tensors: List of torch audio tensors.
        sample_rate: Sample rate of the audio.
        silence_ms: Milliseconds of silence to insert between chunks.

    Returns:
        Combined audio tensor.
    """
    import torch

    if not audio_tensors:
        raise ValueError("No audio tensors to concatenate")

    if len(audio_tensors) == 1:
        return audio_tensors[0]

    # Create silence tensor
    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = torch.zeros(1, silence_samples)

    # Ensure all tensors are on CPU for concatenation
    tensors_cpu = [t.cpu() for t in audio_tensors]

    # Interleave audio with silence
    result_parts = []
    for i, audio in enumerate(tensors_cpu):
        result_parts.append(audio)
        if i < len(tensors_cpu) - 1:  # Don't add silence after last chunk
            result_parts.append(silence)

    return torch.cat(result_parts, dim=-1)


def generate_with_chatterbox(
    text: str,
    reference_audio_paths: list[str],
    reference_texts: list[str],
    output_path: Path,
    exaggeration: float = CHATTERBOX_DEFAULT_EXAGGERATION,
    cfg_weight: float = CHATTERBOX_DEFAULT_CFG_WEIGHT,
) -> dict:
    """Generate audio using Chatterbox TTS with voice cloning.

    Chatterbox supports paralinguistic tags like [laugh], [cough], [chuckle].
    The exaggeration parameter controls speech expressiveness.

    For long texts, automatically splits into chunks based on system memory
    and concatenates the results.

    Args:
        text: The text to synthesize. Can include tags like [laugh], [cough].
        reference_audio_paths: Paths to reference audio files for voice cloning.
        reference_texts: Transcripts of the reference audio (not used by Chatterbox).
        output_path: Where to save the generated audio.
        exaggeration: Controls expressiveness (0.0-1.0, default 0.5).
        cfg_weight: Controls pacing (lower = slower, default 0.5).

    Returns:
        Dict with status, output_path, duration_ms, and sample_rate.
    """
    import sys
    import torch
    import torchaudio as ta

    # Load model
    model = _load_chatterbox()

    # Get absolute path for reference audio
    abs_ref_paths = []
    for ref_path in reference_audio_paths:
        abs_path = _get_absolute_sample_path(ref_path)
        if not os.path.exists(abs_path):
            raise ValueError(f"Reference audio file not found: {abs_path}")
        abs_ref_paths.append(abs_path)

    # Use the first reference audio for voice cloning
    primary_ref_audio = abs_ref_paths[0]

    # Determine optimal chunk size based on system memory
    optimal_chunk_size = _get_optimal_chunk_size()
    memory_gb, memory_type = _get_available_memory_gb()

    # Check if text needs chunking
    if len(text) <= optimal_chunk_size:
        # Short text - generate directly
        wav = model.generate(
            text,
            audio_prompt_path=primary_ref_audio,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        chunks_used = 1
    else:
        # Long text - split into chunks and generate each
        chunks = _split_text_into_chunks(text, optimal_chunk_size)
        print(
            f"Text length ({len(text)} chars) exceeds chunk size ({optimal_chunk_size}). "
            f"Splitting into {len(chunks)} chunks. Memory: {memory_gb}GB {memory_type}",
            file=sys.stderr,
            flush=True,
        )

        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(
                f"  Generating chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...",
                file=sys.stderr,
                flush=True,
            )
            chunk_wav = model.generate(
                chunk,
                audio_prompt_path=primary_ref_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            audio_chunks.append(chunk_wav)

            # Clear CUDA cache between chunks if using CUDA
            if memory_type == "cuda":
                torch.cuda.empty_cache()

        # Concatenate all chunks with small silence between
        wav = _concatenate_audio_tensors(audio_chunks, model.sr, silence_ms=100)
        chunks_used = len(chunks)
        print("  All chunks generated and concatenated.", file=sys.stderr, flush=True)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), wav, model.sr)

    # Get duration
    duration_ms = int(wav.shape[-1] / model.sr * 1000)

    return {
        "status": "success",
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "sample_rate": model.sr,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "chunks_used": chunks_used,
        "memory_gb": memory_gb,
        "memory_type": memory_type,
    }


# ============================================================================
# High-Level TTS Functions
# ============================================================================


@dataclass
class GenerateResult:
    segment_id: str
    audio_path: str
    duration_ms: int
    description: Optional[str] = None
    engine: str = "maya1"


def generate_segment_audio(
    segment_id: str,
    description: Optional[str] = None,
    engine: str = "maya1",
) -> GenerateResult:
    """Generate audio for a single segment."""
    audio_dir = _get_project_audio_dir()

    # Get segment
    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")

    if not segment.text_content.strip():
        raise ValueError(f"Segment has no text content: {segment_id}")

    # Generate audio file path
    segments_dir = audio_dir / "audio" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{segment_id}.wav"
    output_path = segments_dir / filename

    if engine == "maya1":
        # Determine voice description from params, character voice config, or default
        voice_description = description

        if not voice_description and segment.character_id:
            character = get_character(segment.character_id)
            if character and character.voice_config:
                voice_config = json.loads(character.voice_config)
                if voice_config.get("provider") == "maya1":
                    voice_description = voice_config.get("voice_ref")

        voice_description = voice_description or DEFAULT_DESCRIPTION

        result = generate_with_maya1(segment.text_content, voice_description, output_path)
        duration_ms = result.get("duration_ms", 0)

    elif engine == "chatterbox":
        # Get voice samples for the character (voice cloning with emotion control)
        if not segment.character_id:
            raise ValueError(
                "Chatterbox requires a segment assigned to a character with voice samples"
            )

        samples = list_voice_samples(segment.character_id)
        if not samples:
            raise ValueError("No voice samples found for character. Generate or add samples first.")

        ref_paths = [s.sample_path for s in samples]
        ref_texts = [s.sample_text or "" for s in samples]

        # Get exaggeration and cfg_weight from character voice config if set
        exaggeration = CHATTERBOX_DEFAULT_EXAGGERATION
        cfg_weight = CHATTERBOX_DEFAULT_CFG_WEIGHT

        character = get_character(segment.character_id)
        if character and character.voice_config:
            voice_config = json.loads(character.voice_config)
            if voice_config.get("provider") == "chatterbox":
                settings = voice_config.get("settings", {})
                exaggeration = settings.get("exaggeration", CHATTERBOX_DEFAULT_EXAGGERATION)
                cfg_weight = settings.get("cfg_weight", CHATTERBOX_DEFAULT_CFG_WEIGHT)

        result = generate_with_chatterbox(
            segment.text_content,
            ref_paths,
            ref_texts,
            output_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        duration_ms = result.get("duration_ms", 0)
        voice_description = None

    else:
        raise ValueError(f"Unknown TTS engine: {engine}. Use 'maya1' or 'chatterbox'.")

    # Update segment with audio path (use lock for thread safety)
    relative_path = f"audio/segments/{filename}"
    with get_db_lock():
        db = get_database()
        cursor = db.cursor()
        cursor.execute(
            "UPDATE segments SET audio_path = ?, duration_ms = ? WHERE id = ?",
            (relative_path, duration_ms, segment_id),
        )
        db.commit()

    return GenerateResult(
        segment_id=segment_id,
        audio_path=relative_path,
        duration_ms=duration_ms,
        description=voice_description if engine == "maya1" else None,
        engine=engine,
    )


def generate_voice_sample(
    character_id: str,
    text: str,
    description: Optional[str] = None,
) -> dict:
    """Generate a voice sample for a character using Maya1.

    These samples can then be used with Chatterbox for voice cloning.
    """
    audio_dir = _get_project_audio_dir()

    # Get character
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    if not text.strip():
        raise ValueError("Sample text is required")

    # Determine voice description
    voice_description = description

    if not voice_description and character.voice_config:
        voice_config = json.loads(character.voice_config)
        if voice_config.get("provider") == "maya1":
            voice_description = voice_config.get("voice_ref")

    voice_description = voice_description or DEFAULT_DESCRIPTION

    # Generate audio file path for voice samples
    samples_dir = audio_dir / "audio" / "voice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Use UUID to allow multiple samples
    import uuid

    sample_id = str(uuid.uuid4())[:8]
    filename = f"{character_id}_{sample_id}.wav"
    output_path = samples_dir / filename

    # Generate audio with Maya1
    result = generate_with_maya1(text, voice_description, output_path)

    duration_ms = result.get("duration_ms", 0)

    # Store the voice sample in the database
    relative_path = f"audio/voice_samples/{filename}"
    sample = add_voice_sample(
        character_id=character_id,
        sample_path=relative_path,
        sample_text=text,
        duration_ms=duration_ms,
    )

    return {
        "character_id": character_id,
        "character_name": character.name,
        "sample_id": sample.id,
        "sample_path": relative_path,
        "sample_text": text,
        "duration_ms": duration_ms,
        "description": voice_description,
    }


def create_voice_candidates(
    character_id: str,
    sample_text: str,
    voice_descriptions: list[str],
) -> dict:
    """Generate multiple voice candidates with different descriptions.

    Creates one sample per description, all using the same sample_text.
    User can listen to each and pick their favorite.

    Args:
        character_id: The character to generate candidates for
        sample_text: The text to speak (same for all candidates, ~30 sec when spoken)
        voice_descriptions: List of voice descriptions to try (e.g., 3-4 variations)

    Returns:
        List of candidates with sample_id, description, and audio_path
    """
    audio_dir = _get_project_audio_dir()

    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    if not sample_text.strip():
        raise ValueError("Sample text is required")

    if not voice_descriptions or len(voice_descriptions) < 2:
        raise ValueError("At least 2 voice descriptions required for comparison")

    if len(voice_descriptions) > 5:
        raise ValueError("Maximum 5 voice candidates at a time")

    samples_dir = audio_dir / "audio" / "voice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    total_duration_ms = 0

    for i, description in enumerate(voice_descriptions):
        # Generate unique filename
        import uuid

        sample_id = str(uuid.uuid4())[:8]
        filename = f"{character_id}_candidate_{i}_{sample_id}.wav"
        output_path = samples_dir / filename

        # Generate audio with Maya1
        result = generate_with_maya1(sample_text, description, output_path)
        duration_ms = result.get("duration_ms", 0)
        total_duration_ms += duration_ms

        # Store in database
        relative_path = f"audio/voice_samples/{filename}"
        sample = add_voice_sample(
            character_id=character_id,
            sample_path=relative_path,
            sample_text=sample_text,
            duration_ms=duration_ms,
        )

        candidates.append(
            {
                "candidate_index": i,
                "sample_id": sample.id,
                "description": description,
                "sample_path": relative_path,
                "duration_ms": duration_ms,
            }
        )

    return {
        "character_id": character_id,
        "character_name": character.name,
        "sample_text": sample_text,
        "candidate_count": len(candidates),
        "total_duration_ms": total_duration_ms,
        "candidates": candidates,
        "next_step": "Use select_voice_candidate with the sample_id of your preferred voice",
    }


def select_voice_candidate(
    character_id: str,
    selected_sample_id: str,
    additional_sample_texts: Optional[list[str]] = None,
) -> dict:
    """Select a voice candidate and optionally generate more samples.

    This will:
    1. Set the character's voice_ref to the selected voice's description
    2. Delete all other candidate samples for this character
    3. Optionally generate additional samples with the selected voice

    Args:
        character_id: The character
        selected_sample_id: The sample_id of the winning candidate
        additional_sample_texts: Optional list of 2 more sample texts to generate
    """
    from .voice_samples import (
        get_voice_sample,
        list_voice_samples,
        delete_voice_sample,
    )

    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    # Get the selected sample
    selected = get_voice_sample(selected_sample_id)
    if not selected:
        raise ValueError(f"Voice sample not found: {selected_sample_id}")

    if selected.character_id != character_id:
        raise ValueError("Selected sample does not belong to this character")

    # We need to find the description used for this sample
    # Since we don't store it in the sample, we'll need the user to provide it
    # or we look it up from the character's current candidates
    # For now, we'll require the description to be passed or use character's voice_ref

    # Get all samples for this character
    all_samples = list_voice_samples(character_id)

    # Delete non-selected samples
    deleted_count = 0
    for sample in all_samples:
        if sample.id != selected_sample_id:
            delete_voice_sample(sample.id)
            deleted_count += 1

    result = {
        "character_id": character_id,
        "character_name": character.name,
        "selected_sample_id": selected_sample_id,
        "deleted_candidates": deleted_count,
        "remaining_samples": 1,
        "additional_samples_generated": 0,
    }

    # Generate additional samples if texts provided
    if additional_sample_texts:
        audio_dir = _get_project_audio_dir()
        samples_dir = audio_dir / "audio" / "voice_samples"

        # Get voice description from character
        voice_description = DEFAULT_DESCRIPTION
        if character.voice_config:
            voice_config = json.loads(character.voice_config)
            voice_description = voice_config.get("voice_ref") or voice_config.get(
                "voice_id", DEFAULT_DESCRIPTION
            )

        for text in additional_sample_texts[:2]:  # Max 2 additional
            import uuid

            sample_id = str(uuid.uuid4())[:8]
            filename = f"{character_id}_{sample_id}.wav"
            output_path = samples_dir / filename

            gen_result = generate_with_maya1(text, voice_description, output_path)
            duration_ms = gen_result.get("duration_ms", 0)

            relative_path = f"audio/voice_samples/{filename}"
            add_voice_sample(
                character_id=character_id,
                sample_path=relative_path,
                sample_text=text,
                duration_ms=duration_ms,
            )
            result["additional_samples_generated"] += 1

        result["remaining_samples"] += result["additional_samples_generated"]

    return result


def generate_batch_audio(
    segment_ids: Optional[list[str]] = None,
    chapter_id: Optional[str] = None,
    engine: str = "chatterbox",
) -> dict:
    """Generate audio for multiple segments.

    If segment_ids not provided, processes all pending segments (or all in chapter).
    """
    db = get_database()
    cursor = db.cursor()

    # Get segments to process
    if segment_ids:
        ids_to_process = segment_ids
    elif chapter_id:
        cursor.execute(
            "SELECT id FROM segments WHERE chapter_id = ? AND audio_path IS NULL ORDER BY sort_order",
            (chapter_id,),
        )
        ids_to_process = [row["id"] for row in cursor.fetchall()]
    else:
        cursor.execute("SELECT id FROM segments WHERE audio_path IS NULL ORDER BY sort_order")
        ids_to_process = [row["id"] for row in cursor.fetchall()]

    if not ids_to_process:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
        }

    results = []
    successful = 0
    failed = 0

    for segment_id in ids_to_process:
        try:
            result = generate_segment_audio(segment_id, engine=engine)
            results.append(
                {
                    "segment_id": segment_id,
                    "status": "success",
                    "audio_path": result.audio_path,
                    "duration_ms": result.duration_ms,
                }
            )
            successful += 1
        except Exception as e:
            results.append(
                {
                    "segment_id": segment_id,
                    "status": "error",
                    "error": str(e),
                }
            )
            failed += 1

    return {
        "total": len(ids_to_process),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
