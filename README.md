# Talky Talky

A Text-to-Speech MCP (Model Context Protocol) server for AI agents. Generate speech with multiple TTS engines, convert audio formats, and process audio files—all through a standardized interface that works with any MCP-compatible client.

## Features

- **Multiple TTS Engines**:
  - **Maya1**: Natural language voice design with 20+ emotion tags (Apache 2.0)
  - **Chatterbox**: Zero-shot multilingual voice cloning with emotion control (23 languages)
  - **MiraTTS**: Ultra-fast voice cloning at 100x realtime, 48kHz output (CUDA only)
  - **XTTS-v2**: Cross-language voice cloning from just 6 seconds of audio (17 languages)
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
pip install -e ".[maya1]"      # Voice design
pip install -e ".[chatterbox]" # Voice cloning with emotion
pip install -e ".[mira]"       # Fast voice cloning (CUDA only)
pip install -e ".[xtts]"       # Multilingual voice cloning

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
| `speak_mira` | Fast voice cloning with 48kHz output (CUDA required) |
| `speak_xtts` | Multilingual voice cloning (17 languages) |

### Audio Utility Tools

| Tool | Description |
|------|-------------|
| `get_audio_file_info` | Get audio file info (duration, format, size) |
| `convert_audio_format` | Convert between formats (wav, mp3, m4a) |
| `join_audio_files` | Concatenate multiple audio files |
| `normalize_audio_levels` | Normalize to broadcast standard (-16 LUFS) |
| `check_ffmpeg_available` | Check ffmpeg installation |

## TTS Engine Guide

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

## Usage Examples

### Generate Speech with Maya1

```
Generate speech saying "Welcome to the future of AI" with a deep male narrator voice, save to /tmp/welcome.wav
```

### Clone a Voice with Chatterbox

```
Use Chatterbox to clone the voice from /path/to/sample.wav and say "This is a test of voice cloning" with high expressiveness
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
