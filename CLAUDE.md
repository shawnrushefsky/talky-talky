# Talky Talky - Development Guide

This document provides context for Claude Code and other AI assistants working on this codebase.

## Project Overview

Talky Talky is a Model Context Protocol (MCP) server that provides Text-to-Speech and Speech-to-Text capabilities for AI agents. It features a pluggable engine architecture supporting multiple TTS and transcription backends:

- **Maya1**: Text-prompted voice design - create unique voices from natural language descriptions
- **Chatterbox**: Audio-prompted voice cloning - clone voices from reference audio samples
- **Chatterbox Turbo**: Fast voice cloning optimized for production (350M parameters)
- **MiraTTS**: Fast voice cloning with high-quality 48kHz output (CUDA only)
- **XTTS-v2**: Multilingual voice cloning supporting 17 languages
- **Kokoro**: Voice selection from 54 pre-built voices across 8 languages (82M, Apache 2.0)
- **Soprano**: Ultra-fast TTS at 2000x realtime with 32kHz output (CUDA only)
- **VibeVoice Realtime**: Real-time TTS with ~300ms latency, single speaker (Microsoft, 0.5B)
- **VibeVoice Long-form**: Long-form multi-speaker TTS up to 90 minutes (Microsoft, 1.5B)
- **CosyVoice3**: Zero-shot multilingual voice cloning with 9 languages (Alibaba, 0.5B)

**Transcription Engines (Speech-to-Text):**
- **Whisper**: OpenAI's robust speech recognition via transformers (99+ languages, MIT license)
- **Faster-Whisper**: CTranslate2-optimized Whisper (4x faster, same accuracy)

Plus audio utilities for format conversion, concatenation, and normalization.

## Architecture

### Technology Stack

- **Runtime**: Python 3.11+ (required for TTS library compatibility)
- **MCP SDK**: `mcp` with FastMCP for server implementation
- **Audio Processing**: ffmpeg for format conversion and concatenation
- **TTS Engines**:
  - Maya1 (local, requires GPU) - voice design from text descriptions
  - Chatterbox (local) - voice cloning from reference audio
  - Chatterbox Turbo (local) - fast voice cloning for production
  - MiraTTS (local, CUDA only) - fast voice cloning at 48kHz
  - XTTS-v2 (local) - multilingual voice cloning
  - Kokoro (local) - 54 pre-built voices, 8 languages
  - Soprano (local, CUDA only) - ultra-fast 2000x realtime
  - VibeVoice Realtime (local) - real-time TTS with ~300ms latency
  - VibeVoice Long-form (local) - multi-speaker long-form TTS
  - CosyVoice3 (local) - multilingual voice cloning with instruction control
- **Transcription Engines**:
  - Whisper (local) - OpenAI's speech recognition via transformers
  - Faster-Whisper (local) - CTranslate2-optimized, 4x faster

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
│   ├── tts/
│   │   ├── __init__.py   # Public interface, engine registry
│   │   ├── base.py       # Abstract engine interfaces
│   │   ├── utils.py      # Shared utilities (chunking, tag conversion)
│   │   ├── maya1.py      # Maya1 engine implementation
│   │   ├── chatterbox.py # Chatterbox engine implementation
│   │   ├── chatterbox_turbo.py # Chatterbox Turbo engine
│   │   ├── mira.py       # MiraTTS engine implementation
│   │   ├── xtts.py       # XTTS-v2 engine implementation
│   │   ├── kokoro.py     # Kokoro engine (voice selection)
│   │   ├── soprano.py    # Soprano engine (ultra-fast)
│   │   ├── vibevoice.py  # VibeVoice engines (realtime + long-form)
│   │   └── cosyvoice.py  # CosyVoice3 engine (multilingual)
│   └── transcription/
│       ├── __init__.py   # Public interface, engine registry
│       ├── base.py       # Abstract transcription interfaces
│       ├── whisper.py    # Whisper engine (transformers)
│       └── faster_whisper.py # Faster-Whisper engine (CTranslate2)
└── utils/
    ├── __init__.py
    └── ffmpeg.py         # ffmpeg wrapper functions
```

### TTS Engine Architecture

The TTS module uses a pluggable engine architecture:

```python
# Base classes in base.py
TTSEngine              # Abstract base for all engines
TextPromptedEngine     # For text-prompted engines (Maya1)
AudioPromptedEngine    # For audio-prompted engines (Chatterbox, MiraTTS, XTTS)
VoiceSelectionEngine   # For voice selection engines (Kokoro)

# Registry in __init__.py
register_engine(MyEngine)  # Register new engines
get_engine("maya1")        # Get engine by ID
generate(text, output, engine="maya1", **kwargs)  # Unified generation
```

### Transcription Engine Architecture

The transcription module mirrors the TTS architecture:

```python
# Base classes in transcription/base.py
TranscriptionEngine     # Abstract base for all transcription engines
TranscriptionResult     # Result dataclass with text, segments, metadata
TranscriptionSegment    # Individual segment with timestamps
WordSegment             # Word-level timestamps (when supported)
TranscriptionEngineInfo # Engine metadata and capabilities

# Registry in transcription/__init__.py
register_engine(MyEngine)  # Register new engines
get_engine("faster_whisper")  # Get engine by ID
transcribe(audio_path, engine="faster_whisper", **kwargs)  # Unified transcription
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
- `speak_chatterbox_turbo` - Fast voice cloning for production
- `speak_mira` - Fast voice cloning with 48kHz output
- `speak_xtts` - Multilingual voice cloning (17 languages)
- `speak_kokoro` - Use pre-built voices (54 voices, 8 languages)
- `speak_soprano` - Ultra-fast TTS at 2000x realtime (CUDA only)
- `speak_vibevoice_realtime` - Real-time TTS with ~300ms latency
- `speak_vibevoice_longform` - Long-form multi-speaker TTS (up to 90 min)
- `speak_cosyvoice` - Multilingual voice cloning with instruction control

### Audio Utility Tools
- `get_audio_file_info` - Get audio file info (duration, format, size)
- `convert_audio_format` - Convert between formats (wav, mp3, m4a)
- `join_audio_files` - Concatenate multiple audio files
- `normalize_audio_levels` - Normalize to broadcast standard
- `check_ffmpeg_available` - Check ffmpeg installation
- `play_audio` - Play audio file with system's default player
- `set_output_directory` - Set default directory for saving audio files
- `get_output_directory` - Get current default output directory

### Transcription Tools
- `check_transcription_availability` - Check transcription engine status and device info
- `get_transcription_engines_info` - Get detailed info about all transcription engines
- `list_available_transcription_engines` - List installed transcription engines
- `transcribe_audio` - Transcribe audio file to text
- `transcribe_with_timestamps` - Transcribe with word-level timestamps
- `verify_tts_output` - Verify TTS audio matches expected text (for agent verification)

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

### Chatterbox Turbo (Fast Voice Cloning)

Streamlined 350M parameter model optimized for low-latency voice cloning.

**Installation:**
```bash
pip install chatterbox-tts  # Same package as Chatterbox
```

**Features:**
- Faster inference than standard Chatterbox
- Simpler API (no exaggeration/cfg_weight parameters)
- <200ms production latency
- Works on CUDA, MPS, and CPU

**Emotion Tags:** `[laugh]`, `[chuckle]`, `[cough]`

### Kokoro (Voice Selection)

Lightweight 82M parameter TTS with 54 pre-built voices across 8 languages.

**Installation:**
```bash
pip install kokoro>=0.9.2
# System dependency required:
# Linux: apt-get install espeak-ng
# macOS: brew install espeak-ng
```

**Features:**
- 54 high-quality voices across 8 languages
- No voice cloning or description needed
- Very fast, runs on CPU/GPU/edge devices
- Apache 2.0 licensed

**Languages:** American English, British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, Portuguese

**Voice ID Format:** `[lang][gender]_[name]` (e.g., `af_heart`, `bm_george`)

**Parameters:**
- `voice`: Voice ID (default: "af_heart")
- `speed`: Speech rate multiplier (default: 1.0)

### Soprano (Ultra-Fast TTS)

Ultra-lightweight 80M parameter model with 2000x realtime speed.

**Installation:**
```bash
pip install soprano-tts
```

**Requirements:**
- NVIDIA GPU with CUDA (required)
- Does NOT support MPS or CPU

**Features:**
- 2000x realtime (10 hours audio in <20 seconds)
- High-fidelity 32kHz output
- <15ms streaming latency
- Single built-in voice

**Parameters:**
- `temperature`: Sampling randomness (default: 0.3)
- `top_p`: Nucleus sampling (default: 0.95)
- `repetition_penalty`: Prevents repetition (default: 1.2)

### VibeVoice Realtime (Real-time TTS)

Microsoft's real-time TTS with ~300ms first-audio latency.

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Features:**
- ~300ms to first audio (very low latency)
- Up to 10 minutes per generation
- Single speaker with multiple voice options
- Works on CUDA, MPS, and CPU

**Languages:** English (primary), other languages experimental

**Parameters:**
- `speaker_name`: Voice to use (default: "Carter")
- Available: Carter, Emily, Nova, Michael, Sarah

### VibeVoice Long-form (Multi-speaker TTS)

Microsoft's long-form TTS for podcasts and conversations.

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Features:**
- Up to 90 minutes per generation
- Multi-speaker support (up to 4 speakers)
- Natural turn-taking for dialogues
- Works on CUDA, MPS, and CPU

**Languages:** English and Chinese

**Parameters:**
- `speaker_name`: Primary speaker (default: "Carter")
- `speakers`: List of speaker names for multi-speaker (max 4)

**Note:** VibeVoice has dependency conflicts with Chatterbox. Install in a separate environment if needed.

### CosyVoice3 (Multilingual Voice Cloning)

Alibaba's zero-shot voice cloning with instruction-based control.

**Installation:**
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Install sox
# Ubuntu: sudo apt-get install sox libsox-dev
# macOS: brew install sox
```

**Features:**
- Zero-shot voice cloning from reference audio
- 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- 18+ Chinese dialects via instruction control
- Cross-lingual cloning
- Fine-grained control with `[breath]` tags

**Parameters:**
- `reference_audio_paths`: Reference audio for voice cloning
- `prompt_text`: Transcript of reference (improves quality)
- `instruction`: Natural language style control
- `language`: Target language code

**Instruction Examples:**
- `"请用广东话表达。"` - Speak in Cantonese
- `"请用尽可能快地语速说。"` - Speak as fast as possible

## Transcription Engines

### Whisper (via Transformers)

OpenAI's state-of-the-art speech recognition model via the transformers library.

**Installation:**
```bash
pip install transformers torch
# Or with talky-talky:
pip install talky-talky[whisper]
```

**Features:**
- 99+ languages with automatic detection
- Word-level timestamps
- Works on CUDA, MPS, and CPU
- Best accuracy among open-source models

**Model Sizes:**

| Model | Parameters | VRAM | Relative Speed |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |

**Parameters:**
- `model_size`: Model size (default: "base")
- `language`: Language code or None for auto-detection
- `return_timestamps`: True for segments, "word" for word-level

**Recommended Models:**
- **Development/Testing**: base (fast, decent accuracy)
- **Production**: large-v3-turbo (best speed/accuracy balance)
- **Maximum Accuracy**: large-v3

### Faster-Whisper (CTranslate2)

CTranslate2-optimized Whisper implementation - 4x faster with same accuracy.

**Installation:**
```bash
pip install faster-whisper
# Or with talky-talky:
pip install talky-talky[faster-whisper]
```

**Features:**
- 4x faster than original Whisper
- Lower memory usage through quantization
- Word-level timestamps with VAD filtering
- Batched inference support

**Model Sizes:**

| Model | Parameters | VRAM | Speed vs large |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |
| distil-large-v3 | 756M | ~4GB | 6x |

**Parameters:**
- `model_size`: Model size (default: "base")
- `language`: Language code or None for auto-detection
- `word_timestamps`: Enable word-level timestamps
- `vad_filter`: Filter silence using VAD (default: True)
- `beam_size`: Beam size for decoding (default: 5)

**English-Only Models:**
For English transcription, use .en models (e.g., "base.en") for slightly better accuracy.

**Hardware Support:**
- **NVIDIA GPU with CUDA**: Best performance (float16)
- **CPU**: Good performance with int8 quantization
- **Apple Silicon**: Uses CPU (MPS not directly supported by CTranslate2)

## Installation & Setup

```bash
# Install base package
pip install -e .

# Install with Maya1 TTS support (requires CUDA GPU)
pip install -e ".[maya1]"

# Install with Chatterbox TTS support (voice cloning, includes Turbo)
pip install -e ".[chatterbox]"

# Install with MiraTTS support (requires CUDA GPU)
pip install -e ".[mira]"

# Install with XTTS-v2 support (multilingual)
pip install -e ".[xtts]"

# Install with Kokoro support (54 pre-built voices, requires espeak-ng)
pip install -e ".[kokoro]"

# Install with Soprano support (ultra-fast, requires CUDA GPU)
pip install -e ".[soprano]"

# Install all TTS engines (core engines without dependency conflicts)
pip install -e ".[tts]"

# VibeVoice and CosyVoice require separate installation due to dependencies:
# VibeVoice:
git clone https://github.com/microsoft/VibeVoice.git && cd VibeVoice && pip install -e .
# CosyVoice:
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git && cd CosyVoice && pip install -r requirements.txt

# Install with Whisper transcription support (via transformers)
pip install -e ".[whisper]"

# Install with Faster-Whisper transcription support (4x faster)
pip install -e ".[faster-whisper]"

# Install all transcription engines
pip install -e ".[transcription]"

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

## Configuring MCP Clients

### Claude Desktop (macOS)

**Important:** GUI applications on macOS don't inherit the shell PATH, so you must use the full path to `uv`.

1. Find the full path to uv:
   ```bash
   which uv
   # Example output: /Users/username/.local/bin/uv
   ```

2. On macOS ARM64 (Apple Silicon), use individual extras instead of `--extra tts` to avoid CUDA-only dependencies (`mira`, `soprano`) that don't have macOS wheels:

3. Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "talky-talky": {
         "command": "/Users/username/.local/bin/uv",
         "args": [
           "run",
           "--directory", "/path/to/talky-talky",
           "--extra", "maya1",
           "--extra", "chatterbox",
           "--extra", "xtts",
           "--extra", "kokoro",
           "--extra", "vibevoice",
           "talky-talky"
         ]
       }
     }
   }
   ```

4. Replace paths with actual values and restart Claude Desktop.

### Claude Code / CLI Tools

CLI tools inherit the shell PATH, so you can use `uv` directly. Add to `.mcp.json` or `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "talky-talky"]
    }
  }
}
```

### Linux with CUDA

On Linux with CUDA GPU, you can use `--extra tts` to include all engines:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/talky-talky", "--extra", "tts", "talky-talky"]
    }
  }
}
```

## Development Notes

- **This project uses `uv`** for package management and running Python
- Always use `uv run` to execute Python commands (e.g., `uv run python`, `uv run pytest`)
- Install dependencies with `uv pip install` or `uv sync`

### MCP Protocol Compatibility

**Important:** MCP servers communicate via JSON-RPC over stdout. Many TTS libraries print progress messages, loading status, etc. to stdout, which breaks the protocol.

All TTS engines must use the `redirect_stdout_to_stderr()` context manager from `utils.py` when:
- Importing TTS libraries
- Loading models
- Running generation

```python
from .utils import redirect_stdout_to_stderr

def _load_model():
    with redirect_stdout_to_stderr():
        from some_tts_library import Model
        model = Model.from_pretrained("model-id")
    return model
```

This redirects any library output to stderr where it won't interfere with MCP communication.

## Debugging

- The server logs to stderr (stdout is reserved for MCP protocol)
- Use `print(..., file=sys.stderr)` for debug logging
