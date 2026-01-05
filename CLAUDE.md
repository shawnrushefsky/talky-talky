# Talky Talky - Development Guide

This document provides context for Claude Code and other AI assistants working on this codebase.

## Project Overview

Talky Talky is a Model Context Protocol (MCP) server that provides Text-to-Speech capabilities for AI agents. It features a pluggable engine architecture supporting multiple TTS backends:

- **Maya1**: Text-prompted voice design - create unique voices from natural language descriptions
- **Chatterbox**: Audio-prompted voice cloning - clone voices from reference audio samples
- **MiraTTS**: Fast voice cloning with high-quality 48kHz output (CUDA only)
- **XTTS-v2**: Multilingual voice cloning supporting 17 languages

Plus audio utilities for format conversion, concatenation, and normalization.

## Architecture

### Technology Stack

- **Runtime**: Python 3.11+ (required for TTS library compatibility)
- **MCP SDK**: `mcp` with FastMCP for server implementation
- **Audio Processing**: ffmpeg for format conversion and concatenation
- **TTS Engines**:
  - Maya1 (local, requires GPU) - voice design from text descriptions
  - Chatterbox (local) - voice cloning from reference audio
  - MiraTTS (local, CUDA only) - fast voice cloning at 48kHz
  - XTTS-v2 (local) - multilingual voice cloning

### Python Version Requirement

This project requires Python 3.11+ due to numpy version conflicts between TTS libraries on Python 3.10. If a user has an older Python version, recommend:

1. **Use `uv` (recommended)** - It automatically manages Python versions:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv run --extra tts talky-talky
   ```

2. **Use `pyenv`** to install Python 3.11+:
   ```bash
   pyenv install 3.11
   pyenv local 3.11
   ```

### Directory Structure

```
talky_talky/
├── __init__.py
├── server.py             # MCP server entry point, tool registrations
├── tools/
│   ├── __init__.py
│   ├── audio.py          # Audio utilities (convert, concat, info)
│   └── tts/
│       ├── __init__.py   # Public interface, engine registry
│       ├── base.py       # Abstract engine interfaces
│       ├── utils.py      # Shared utilities (chunking, tag conversion)
│       ├── maya1.py      # Maya1 engine implementation
│       ├── chatterbox.py # Chatterbox engine implementation
│       ├── mira.py       # MiraTTS engine implementation
│       └── xtts.py       # XTTS-v2 engine implementation
└── utils/
    ├── __init__.py
    └── ffmpeg.py         # ffmpeg wrapper functions
```

### Engine Architecture

The TTS module uses a pluggable engine architecture:

```python
# Base classes in base.py
TTSEngine           # Abstract base for all engines
VoiceDesignEngine   # For text-prompted engines (Maya1)
VoiceCloningEngine  # For audio-prompted engines (Chatterbox)

# Registry in __init__.py
register_engine(MyEngine)  # Register new engines
get_engine("maya1")        # Get engine by ID
generate(text, output, engine="maya1", **kwargs)  # Unified generation
```

### Adding New TTS Engines

Follow these steps to add a new TTS engine. Use existing engines as reference (e.g., `xtts.py` for audio-prompted, `maya1.py` for text-prompted).

#### Step 1: Create the Engine File

Create `talky_talky/tools/tts/<engine_name>.py`:

```python
"""<EngineName> TTS Engine - Brief description."""

import os
import sys
from pathlib import Path

from .base import AudioPromptedEngine, TTSResult, EngineInfo, PromptingGuide
# Or use TextPromptedEngine for voice-description-based engines
from .utils import split_text_into_chunks, get_best_device, get_available_memory_gb

# Constants
SAMPLE_RATE = 24000  # Output sample rate
MAX_CHUNK_CHARS = 400  # Max chars per generation chunk

# Lazy-loaded model singleton
_model = None

def _load_model():
    """Lazily load the model."""
    global _model
    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading <EngineName> on {device}...", file=sys.stderr, flush=True)

    # Load model here
    # Handle device compatibility (CUDA, MPS, CPU)

    print("<EngineName> loaded successfully", file=sys.stderr, flush=True)
    return _model

class MyEngine(AudioPromptedEngine):  # or TextPromptedEngine
    @property
    def name(self) -> str:
        return "Engine Display Name"

    @property
    def engine_id(self) -> str:
        return "engine_id"  # lowercase, used in API

    def is_available(self) -> bool:
        """Check if dependencies are installed and device is compatible."""
        try:
            import required_package  # noqa: F401
            # Add device checks if needed (e.g., CUDA-only)
            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",  # or "text_prompted"
            description="Brief description",
            requirements="package-name (pip install package-name)",
            max_duration_secs=30,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,  # "[tag]" or "<tag>"
            emotion_tags=[],
            extra_info={...},
            prompting_guide=PromptingGuide(...),  # Optional but recommended
        )

    def get_setup_instructions(self) -> str:
        return """## Engine Setup Instructions..."""

    def generate(self, text, output_path, reference_audio_paths, **kwargs) -> TTSResult:
        """Generate audio. Always return TTSResult."""
        import soundfile as sf
        import numpy as np

        output_path = Path(output_path)

        # Validate inputs
        if not text.strip():
            return TTSResult(status="error", output_path=str(output_path),
                           duration_ms=0, sample_rate=SAMPLE_RATE, error="Empty text")

        try:
            model = _load_model()
            # Generate audio...
            audio = model.generate(text, ...)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, SAMPLE_RATE)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=int(len(audio) / SAMPLE_RATE * 1000),
                sample_rate=SAMPLE_RATE,
                metadata={...},
            )
        except Exception as e:
            return TTSResult(status="error", output_path=str(output_path),
                           duration_ms=0, sample_rate=SAMPLE_RATE, error=str(e))
```

#### Step 2: Register the Engine

In `talky_talky/tools/tts/__init__.py`:

```python
# Add import at top with other engines
from .myengine import MyEngine

# Add to register_engine calls
register_engine(MyEngine)

# Add to __all__
__all__ = [..., "MyEngine"]
```

#### Step 3: Add the MCP Tool

In `talky_talky/server.py`:

```python
@mcp.tool()
def speak_myengine(
    text: str,
    output_path: str,
    reference_audio_paths: list[str],  # For audio-prompted
    # voice_description: str,  # For text-prompted
    # Add engine-specific params with defaults
) -> dict:
    """Generate speech using MyEngine.

    Detailed docstring for AI agents...
    """
    result = generate(
        text=text,
        output_path=output_path,
        engine="myengine",
        reference_audio_paths=reference_audio_paths,
    )
    return to_dict(result)
```

#### Step 4: Add Dependencies

In `pyproject.toml`:

```toml
[project.optional-dependencies]
# Add new engine extra
myengine = [
    "required-package>=1.0.0",
]
# Update tts to include it
tts = [
    "talky-talky[maya1,chatterbox,mira,xtts,myengine]",
]
```

Then update the lock file:
```bash
uv lock
```

#### Step 5: Handle Device Compatibility

Important considerations:

1. **CUDA-only engines** (like MiraTTS): Check `torch.cuda.is_available()` in `is_available()`
2. **MPS support**: Use `get_best_device()` and handle MPS-specific loading
3. **PyTorch 2.6+ compatibility**: If loading pickled weights, patch `torch.load`:
   ```python
   import torch
   from functools import wraps

   _original = torch.load
   @wraps(_original)
   def _patched(*args, **kwargs):
       kwargs.setdefault("weights_only", False)
       return _original(*args, **kwargs)
   torch.load = _patched
   try:
       # Load model
   finally:
       torch.load = _original
   ```

### Verifying a New TTS Engine

After implementing, run these verification steps:

#### 1. Check Engine Registration

```bash
uv run python -c "
from talky_talky.tools.tts import list_engines, get_engine

engines = list_engines()
print('Registered engines:', list(engines.keys()))

engine = get_engine('myengine')
print(f'Name: {engine.name}')
print(f'Available: {engine.is_available()}')
"
```

#### 2. Install Dependencies

```bash
uv pip install -e ".[myengine]"
```

#### 3. Test Generation

```bash
uv run python -c "
from talky_talky.tools.tts import generate

result = generate(
    text='Hello, this is a test.',
    output_path='/tmp/test_output.wav',
    engine='myengine',
    reference_audio_paths=['/path/to/reference.wav'],  # if audio-prompted
)
print(f'Status: {result.status}')
print(f'Duration: {result.duration_ms}ms')
if result.error:
    print(f'Error: {result.error}')
"
```

#### 4. Verify Audio Output

```bash
# Check file exists and has content
ls -la /tmp/test_output.wav

# Get audio info
uv run python -c "
from talky_talky.tools.audio import get_audio_info
info = get_audio_info('/tmp/test_output.wav')
print(f'Duration: {info.duration_ms}ms')
print(f'Format: {info.format}')
"
```

#### 5. Run Linter

```bash
uvx ruff check talky_talky/tools/tts/myengine.py
uvx ruff format talky_talky/tools/tts/myengine.py
```

#### 6. Test MCP Tool

```bash
uv run python -c "
from talky_talky.server import speak_myengine

result = speak_myengine(
    text='Testing the MCP tool.',
    output_path='/tmp/mcp_test.wav',
    reference_audio_paths=['/path/to/reference.wav'],
)
print(result)
"
```

#### 7. Update Documentation

- Update `README.md` with engine description and examples
- Update `CLAUDE.md` TTS Engines section
- Check official model page for accurate feature descriptions

## MCP Tools

### TTS Engine Tools
- `check_tts_availability` - Check engine status and device info
- `get_tts_engines_info` - Get detailed info about all engines
- `list_available_engines` - List installed engines
- `get_tts_model_status` - Check Maya1 model download status
- `download_tts_models` - Download Maya1 models

### Speech Generation Tools
- `speak_maya1` - Generate speech with voice description
- `speak_chatterbox` - Generate speech with voice cloning
- `speak_mira` - Fast voice cloning with 48kHz output
- `speak_xtts` - Multilingual voice cloning (17 languages)

### Audio Utility Tools
- `get_audio_file_info` - Get audio file info (duration, format, size)
- `convert_audio_format` - Convert between formats (wav, mp3, m4a)
- `join_audio_files` - Concatenate multiple audio files
- `normalize_audio_levels` - Normalize to broadcast standard
- `check_ffmpeg_available` - Check ffmpeg installation

## TTS Engines

### Maya1 (Voice Design)

Creates unique voices from natural language descriptions with inline emotion tags.

**Requirements:**
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (best), or MPS (Apple Silicon), or CPU (slow)
- ~10GB disk space for model weights

**Emotion Tags:** `<laugh>`, `<sigh>`, `<gasp>`, `<whisper>`, `<angry>`, `<excited>`, etc.

**Voice Description Example:**
```
"Gruff male pirate, 50s, British accent, low pitch, gravelly, slow pacing"
```

### Chatterbox (Voice Cloning)

Clones voices from reference audio with emotion control.

**Installation:**
```bash
pip install chatterbox-tts
```

**Parameters:**
- `exaggeration`: 0.0-1.0+, controls expressiveness (default 0.5)
- `cfg_weight`: 0.0-1.0, controls pacing (default 0.5)

**Emotion Tags:** `[laugh]`, `[chuckle]`, `[cough]`, `[sigh]`

### MiraTTS (Fast Voice Cloning)

Fast voice cloning with high-quality 48kHz output.

**Requirements:**
- NVIDIA GPU with CUDA (6GB+ VRAM)
- Does NOT support MPS or CPU

**Features:**
- 48kHz output (higher quality than most TTS)
- Over 100x realtime performance
- Works with only 6GB VRAM

### XTTS-v2 (Multilingual Voice Cloning)

Multilingual voice cloning from Coqui supporting 17 languages.

**Installation:**
```bash
pip install TTS
```

**Features:**
- Only requires ~6 seconds of reference audio
- Cross-language cloning (clone voice in one language, output in another)
- Works on CUDA, MPS, and CPU

**Supported Languages:**
English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

**Parameters:**
- `language`: Target language code (default: "en")

## Installation & Setup

```bash
# Install base package
pip install -e .

# Install with Maya1 TTS support (requires CUDA GPU)
pip install -e ".[maya1]"

# Install with Chatterbox TTS support (voice cloning)
pip install -e ".[chatterbox]"

# Install with MiraTTS support (requires CUDA GPU)
pip install -e ".[mira]"

# Install with XTTS-v2 support (multilingual)
pip install -e ".[xtts]"

# Install all TTS engines
pip install -e ".[tts]"

# Install development dependencies
pip install -e ".[dev]"
```

## Running the Server

```bash
# Run the MCP server (communicates via stdio)
uv run talky-talky

# Or run directly
uv run python -m talky_talky.server
```

## Development Notes

- **This project uses `uv`** for package management and running Python
- Always use `uv run` to execute Python commands (e.g., `uv run python`, `uv run pytest`)
- Install dependencies with `uv pip install` or `uv sync`

## Debugging

- The server logs to stderr (stdout is reserved for MCP protocol)
- Use `print(..., file=sys.stderr)` for debug logging
