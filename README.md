# Talky Talky

A Text-to-Speech MCP (Model Context Protocol) server for AI agents. Generate speech with multiple TTS engines, convert audio formats, and process audio files—all through a standardized interface that works with any MCP-compatible client.

## Quick Setup with AI Agents

Copy and paste this prompt to your AI agent (Claude Code, Cursor, Windsurf, etc.) to have it automatically configure Talky Talky:

```
Install and configure the Talky Talky MCP server for text-to-speech capabilities.

1. Clone the repo: git clone https://github.com/shawnrushefsky/talky-talky.git
2. Add to my MCP configuration (.mcp.json or settings.json):
   {
     "mcpServers": {
       "talky-talky": {
         "command": "uv",
         "args": ["run", "--directory", "<path-to-talky-talky>", "--extra", "tts", "talky-talky"]
       }
     }
   }
3. Replace <path-to-talky-talky> with the actual clone path
4. Verify by checking TTS availability after restart

Requirements: Python 3.11+, ffmpeg, GPU recommended for TTS engines.
```

---

- [Talky Talky](#talky-talky)
  - [Quick Setup with AI Agents](#quick-setup-with-ai-agents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from Source](#install-from-source)
    - [Using uv (Recommended)](#using-uv-recommended)
    - [Using Docker](#using-docker)
      - [Basic Image (Audio Utilities Only)](#basic-image-audio-utilities-only)
      - [With CUDA (GPU-Accelerated TTS)](#with-cuda-gpu-accelerated-tts)
      - [Docker with MCP Clients](#docker-with-mcp-clients)
  - [Configuration](#configuration)
    - [Claude Desktop](#claude-desktop)
    - [Claude Code (CLI)](#claude-code-cli)
    - [Other MCP Clients](#other-mcp-clients)
  - [Available Tools](#available-tools)
    - [TTS Engine Tools](#tts-engine-tools)
    - [Speech Generation Tools](#speech-generation-tools)
    - [Audio Utility Tools](#audio-utility-tools)
  - [TTS Engine Guide](#tts-engine-guide)
    - [Maya1 (Voice Design)](#maya1-voice-design)
    - [Chatterbox (Voice Cloning)](#chatterbox-voice-cloning)
    - [Chatterbox Turbo (Fast Voice Cloning)](#chatterbox-turbo-fast-voice-cloning)
    - [MiraTTS (Fast Voice Cloning)](#miratts-fast-voice-cloning)
    - [XTTS-v2 (Multilingual)](#xtts-v2-multilingual)
    - [Kokoro (Voice Selection)](#kokoro-voice-selection)
    - [Soprano (Ultra-Fast TTS)](#soprano-ultra-fast-tts)
    - [VibeVoice Realtime (Real-time TTS)](#vibevoice-realtime-real-time-tts)
    - [VibeVoice Long-form (Multi-speaker TTS)](#vibevoice-long-form-multi-speaker-tts)
    - [CosyVoice3 (Multilingual Voice Cloning)](#cosyvoice3-multilingual-voice-cloning)
  - [Usage Examples](#usage-examples)
    - [Generate Speech with Maya1](#generate-speech-with-maya1)
    - [Clone a Voice with Chatterbox](#clone-a-voice-with-chatterbox)
    - [Use Kokoro Pre-Built Voices](#use-kokoro-pre-built-voices)
    - [Convert Audio Format](#convert-audio-format)
    - [Concatenate Audio Files](#concatenate-audio-files)
  - [Development](#development)
  - [License](#license)
  - [Contributing](#contributing)


## Features

- **Multiple TTS Engines** (10 engines):
  - **Maya1**: Natural language voice design with 20+ emotion tags (Apache 2.0)
  - **Chatterbox**: Zero-shot multilingual voice cloning with emotion control (23 languages)
  - **Chatterbox Turbo**: Fast voice cloning optimized for production (350M parameters)
  - **MiraTTS**: Ultra-fast voice cloning at 100x realtime, 48kHz output (CUDA only)
  - **XTTS-v2**: Cross-language voice cloning from just 6 seconds of audio (17 languages)
  - **Kokoro**: 54 high-quality pre-built voices across 8 languages (82M parameters, Apache 2.0)
  - **Soprano**: Ultra-fast TTS at 2000x realtime with 32kHz output (CUDA only)
  - **VibeVoice Realtime**: Real-time TTS with ~300ms latency, single speaker (Microsoft, 0.5B)
  - **VibeVoice Long-form**: Long-form multi-speaker TTS up to 90 minutes (Microsoft, 1.5B)
  - **CosyVoice3**: Zero-shot multilingual voice cloning with instruction control (Alibaba, 9 languages)
- **Audio Utilities**: Format conversion, concatenation, normalization
- **Cross-Platform**: Works on CUDA, MPS (Apple Silicon), and CPU

## Installation

### Prerequisites

- **Python 3.11 or later** (required due to TTS library dependencies)
- **ffmpeg** (required for audio conversion)
- **GPU** (recommended for TTS, but CPU also supported)

> **Don't have Python 3.11+?** We recommend using [uv](https://docs.astral.sh/uv/) which automatically manages Python versions:
> ```bash
> # Install uv (if not already installed)
> curl -LsSf https://astral.sh/uv/install.sh | sh
>
> # uv will automatically use Python 3.11+ when running talky-talky
> uv run --extra tts talky-talky
> ```
> Alternatively, use [pyenv](https://github.com/pyenv/pyenv) to install Python 3.11+.

### Install from Source

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# Basic installation (no TTS engines)
pip install -e .

# With specific TTS engines
pip install -e ".[maya1]"      # Voice design from descriptions
pip install -e ".[chatterbox]" # Voice cloning with emotion (includes Turbo)
pip install -e ".[mira]"       # Fast voice cloning (CUDA only)
pip install -e ".[xtts]"       # Multilingual voice cloning
pip install -e ".[kokoro]"     # 54 pre-built voices (requires espeak-ng)
pip install -e ".[soprano]"    # Ultra-fast TTS (CUDA only)

# All TTS engines
pip install -e ".[tts]"
```

### Using uv (Recommended)

```bash
git clone https://github.com/shawnrushefsky/talky-talky.git
cd talky-talky

# Install with uv
uv pip install -e ".[tts]"

# Or run directly without installing
uv run --extra tts talky-talky
```

### Using Docker

Pre-built images are available on GitHub Container Registry.

#### Basic Image (Audio Utilities Only)

The default image includes audio utilities but not TTS engines (smaller image size):

```bash
docker pull ghcr.io/shawnrushefsky/talky-talky:latest

# Run as MCP server (communicates via stdio)
docker run -i ghcr.io/shawnrushefsky/talky-talky:latest
```

#### With CUDA (GPU-Accelerated TTS)

For GPU-accelerated TTS, build a custom image with CUDA support:

```dockerfile
# Dockerfile.cuda
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3.11 -m pip install --no-cache-dir talky-talky[tts]

ENTRYPOINT ["python3.11", "-m", "talky_talky.server"]
```

Build and run with GPU access:

```bash
docker build -f Dockerfile.cuda -t talky-talky-cuda .
docker run -i --gpus all talky-talky-cuda
```

#### Docker with MCP Clients

For Claude Desktop or other MCP clients, configure Docker as the command:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "ghcr.io/shawnrushefsky/talky-talky:latest"]
    }
  }
}
```

For CUDA-enabled Docker:

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "--gpus", "all", "talky-talky-cuda"]
    }
  }
}
```

> **Note:** Mount volumes for persistent audio files: `-v /path/to/audio:/audio`

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "talky-talky": {
      "command": "talky-talky"
    }
  }
}
```

Or with uv (no install required):

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

### Claude Code (CLI)

Add to your project's `.mcp.json`:

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

Or add to `~/.claude/settings.json` for global access.

### Other MCP Clients

Talky Talky works with any MCP-compatible client including Cursor, Windsurf, Cline, Continue.dev, and Zed. Configure them to run `talky-talky` as the command.

## Available Tools

### TTS Engine Tools

| Tool | Description |
|------|-------------|
| `check_tts_availability` | Check which TTS engines are available and device info |
| `get_tts_engines_info` | Get detailed info about all engines including parameters |
| `list_available_engines` | List currently installed engines |
| `get_tts_model_status` | Check if Maya1 models are downloaded |
| `download_tts_models` | Download Maya1 model weights (~10GB) |

### Speech Generation Tools

| Tool | Description |
|------|-------------|
| `speak_maya1` | Generate speech with voice description (text-prompted) |
| `speak_chatterbox` | Generate speech with voice cloning and emotion control |
| `speak_chatterbox_turbo` | Fast voice cloning optimized for production |
| `speak_mira` | Fast voice cloning with 48kHz output (CUDA required) |
| `speak_xtts` | Multilingual voice cloning (17 languages) |
| `speak_kokoro` | Use pre-built voices (54 voices, 8 languages) |
| `speak_soprano` | Ultra-fast TTS at 2000x realtime (CUDA required) |
| `speak_vibevoice_realtime` | Real-time TTS with ~300ms latency |
| `speak_vibevoice_longform` | Long-form multi-speaker TTS (up to 90 min) |
| `speak_cosyvoice` | Multilingual voice cloning with instruction control |

### Audio Utility Tools

| Tool | Description |
|------|-------------|
| `get_audio_file_info` | Get audio file info (duration, format, size) |
| `convert_audio_format` | Convert between formats (wav, mp3, m4a) |
| `join_audio_files` | Concatenate multiple audio files |
| `normalize_audio_levels` | Normalize to broadcast standard (-16 LUFS) |
| `check_ffmpeg_available` | Check ffmpeg installation |

## TTS Engine Guide

### Engine License Summary

| Engine | License | Link |
|--------|---------|------|
| Maya1 | Apache-2.0 | [HuggingFace](https://huggingface.co/maya-research/maya1) |
| Chatterbox | MIT | [GitHub](https://github.com/resemble-ai/chatterbox) |
| Chatterbox Turbo | MIT | [GitHub](https://github.com/resemble-ai/chatterbox) |
| MiraTTS | MIT | [HuggingFace](https://huggingface.co/YatharthS/MiraTTS) |
| XTTS-v2 | CPML | [Coqui](https://coqui.ai/cpml) |
| Kokoro | Apache-2.0 | [GitHub](https://github.com/hexgrad/kokoro) |
| Soprano | Apache-2.0 | [HuggingFace](https://huggingface.co/SopranoAI/soprano-mlx) |
| VibeVoice | MIT | [GitHub](https://github.com/microsoft/VibeVoice) |
| CosyVoice3 | Apache-2.0 | [GitHub](https://github.com/FunAudioLLM/CosyVoice) |

### Maya1 (Voice Design)

A 3B parameter model that creates unique voices from natural language descriptions—like briefing an actor. No reference audio needed. Fully open-source under Apache 2.0.

**Voice Description Example:**
```
Realistic female voice in the 20s age with american accent.
Medium-high pitch, bright timbre, energetic pacing, enthusiastic tone.
```

**Emotion Tags (20+ supported):**
```
<laugh> <chuckle> <sigh> <gasp> <whisper> <angry> <yell> <cry> <cough>
```

**Example:**
```python
speak_maya1(
    text="I can't believe it worked! <laugh> We actually did it!",
    output_path="/tmp/output.wav",
    voice_description="Excited young woman, American accent, energetic"
)
```

**Requirements:** CUDA GPU with 16GB+ VRAM (best), MPS, or CPU. Outputs 24kHz audio.

### Chatterbox (Voice Cloning)

Zero-shot multilingual voice cloning with emotion/intensity control. Supports 23 languages including English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, and more. The first open-source TTS with exaggeration control. Outperforms ElevenLabs in side-by-side evaluations.

**Supported Languages:** Arabic, Chinese, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Turkish

**Emotion Tags:**
```
[laugh] [chuckle] [cough] [sigh] [gasp]
```

**Parameters:**
- `exaggeration`: 0.0-1.0+, controls expressiveness/intensity (default 0.5)
- `cfg_weight`: Controls pacing, lower = slower (default 0.5)

**Example:**
```python
speak_chatterbox(
    text="Hello there! [chuckle] Nice to meet you.",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"],
    exaggeration=0.6
)
```

**Requirements:** Works on CUDA, MPS, and CPU. Outputs 24kHz audio.

### Chatterbox Turbo (Fast Voice Cloning)

Streamlined 350M parameter model optimized for low-latency voice cloning in production voice agents. Faster than standard Chatterbox with a simpler API (no tuning parameters).

**Emotion Tags:**
```
[laugh] [chuckle] [cough]
```

**Example:**
```python
speak_chatterbox_turbo(
    text="Hi there! [chuckle] Thanks for calling.",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"]
)
```

**Requirements:** Works on CUDA, MPS, and CPU. Outputs 24kHz audio. <200ms production latency.

### MiraTTS (Fast Voice Cloning)

Ultra-fast voice cloning optimized for speed and efficiency. Over 100x realtime with batching, latency as low as 100ms. Outputs high-quality 48kHz audio (higher than most TTS models).

**Example:**
```python
speak_mira(
    text="The quick brown fox jumps over the lazy dog.",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"]
)
```

**Requirements:** NVIDIA GPU with CUDA (6GB+ VRAM). Does NOT support MPS or CPU. Outputs 48kHz audio.

### XTTS-v2 (Multilingual)

Voice cloning from just a 6-second audio clip with cross-language capabilities. Clone a voice in one language and generate speech in another while preserving voice characteristics. Powers Coqui Studio and API.

**Supported Languages (17):**
English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

**Example:**
```python
# Clone an English voice to speak Spanish
speak_xtts(
    text="Hola, ¿cómo estás?",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/english_reference.wav"],
    language="es"
)
```

**Requirements:** Works on CUDA, MPS, and CPU. Model downloads automatically (~6GB). Outputs 24kHz audio.

### Kokoro (Voice Selection)

Lightweight 82M parameter TTS model with 54 high-quality pre-built voices across 8 languages. No voice cloning or description needed—just select a voice ID. Very fast and runs on any hardware including edge devices. Apache 2.0 licensed.

**Supported Languages (8):**
American English, British English, Japanese, Mandarin Chinese, Spanish, French, Hindi, Italian, Portuguese

**Voice ID Format:** `[lang][gender]_[name]`
- `a` = American English, `b` = British English, `j` = Japanese, `z` = Mandarin
- `f` = female, `m` = male

**Popular Voices:**
- `af_heart`, `af_bella` - American English female (quality A)
- `am_fenrir`, `am_michael` - American English male (quality B)
- `bf_emma`, `bm_george` - British English
- `jf_alpha`, `jm_kumo` - Japanese

**Example:**
```python
speak_kokoro(
    text="Welcome to the future of text-to-speech.",
    output_path="/tmp/output.wav",
    voice="af_heart",
    speed=1.0
)
```

**Requirements:** Requires `espeak-ng` system dependency. Runs on CUDA, MPS, or CPU. Outputs 24kHz audio.

### Soprano (Ultra-Fast TTS)

Ultra-lightweight 80M parameter model with exceptional speed—2000x realtime (10 hours of audio in <20 seconds). Outputs high-fidelity 32kHz audio with <15ms streaming latency. Single built-in voice, no customization.

**Example:**
```python
speak_soprano(
    text="The quick brown fox jumps over the lazy dog.",
    output_path="/tmp/output.wav",
    temperature=0.3
)
```

**Parameters:**
- `temperature`: Sampling randomness (default 0.3)
- `top_p`: Nucleus sampling (default 0.95)
- `repetition_penalty`: Prevents repetition (default 1.2)

**Requirements:** NVIDIA GPU with CUDA required. Does NOT support MPS or CPU. Outputs 32kHz audio.

### VibeVoice Realtime (Real-time TTS)

Microsoft's real-time TTS with ~300ms first-audio latency. Single speaker with multiple voice options. Ideal for real-time applications.

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Available Speakers:** Carter, Emily, Nova, Michael, Sarah

**Example:**
```python
speak_vibevoice_realtime(
    text="Welcome to our service!",
    output_path="/tmp/output.wav",
    speaker_name="Carter"
)
```

**Requirements:** Works on CUDA, MPS, and CPU. English primary, other languages experimental. Outputs 24kHz audio.

> **Note:** VibeVoice has dependency conflicts with Chatterbox. Install in a separate environment if needed.

### VibeVoice Long-form (Multi-speaker TTS)

Microsoft's long-form TTS for podcasts and conversations. Supports up to 90 minutes and 4 speakers with natural turn-taking.

**Installation:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Example:**
```python
# Single speaker
speak_vibevoice_longform(
    text="This is a long-form narration...",
    output_path="/tmp/output.wav",
    speaker_name="Carter"
)

# Multi-speaker dialogue
speak_vibevoice_longform(
    text="Speaker1: Hello! Speaker2: Hi there!",
    output_path="/tmp/dialogue.wav",
    speakers=["Carter", "Emily"]
)
```

**Requirements:** GPU recommended (~6-8GB VRAM). Supports English and Chinese. Outputs 24kHz audio.

### CosyVoice3 (Multilingual Voice Cloning)

Alibaba's zero-shot voice cloning with instruction-based control for dialect, emotion, and speed. Supports 9 languages and 18+ Chinese dialects.

**Installation:**
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt

# Install sox (Ubuntu: apt-get install sox libsox-dev, macOS: brew install sox)
```

**Supported Languages (9):**
Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian

**Supported Chinese Dialects (18+):**
Cantonese, Sichuan, Shanghai, Taiwan, Dongbei, and more via instruction control

**Example:**
```python
# Zero-shot cloning
speak_cosyvoice(
    text="Hello, this is a test.",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"],
    prompt_text="Optional transcript of reference audio"
)

# With instruction control (Cantonese)
speak_cosyvoice(
    text="好少咯，一般系放嗰啲国庆啊",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"],
    instruction="请用广东话表达。"
)

# With breathing tags
speak_cosyvoice(
    text="[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"]
)
```

**Parameters:**
- `prompt_text`: Transcript of reference audio (improves quality)
- `instruction`: Natural language style control (dialect, speed, emotion)
- `language`: Target language code (auto-detected by default)

**Requirements:** GPU recommended. Works on CUDA and CPU. Outputs 22kHz audio.

## Usage Examples

### Generate Speech with Maya1

```
Generate speech saying "Welcome to the future of AI" with a deep male narrator voice, save to /tmp/welcome.wav
```

### Clone a Voice with Chatterbox

```
Use Chatterbox to clone the voice from /path/to/sample.wav and say "This is a test of voice cloning" with high expressiveness
```

### Use Kokoro Pre-Built Voices

```
Use Kokoro with the af_heart voice to say "Hello and welcome!" and save to /tmp/greeting.wav
```

### Convert Audio Format

```
Convert /tmp/speech.wav to MP3 format
```

### Concatenate Audio Files

```
Join these audio files into one: /tmp/intro.wav, /tmp/main.wav, /tmp/outro.wav
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check talky_talky
ruff format talky_talky
```

This project uses `uv` for package management. Always use `uv run` to execute Python commands.

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
