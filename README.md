# Talky Talky

A Text-to-Speech MCP (Model Context Protocol) server for AI agents. Generate speech with multiple TTS engines, convert audio formats, and process audio files—all through a standardized interface that works with any MCP-compatible client.

## Features

- **Multiple TTS Engines**:
  - **Maya1**: Voice design via natural language descriptions with 20+ emotion tags
  - **Chatterbox**: High-quality voice cloning with emotion control
  - **MiraTTS**: Fast voice cloning with 48kHz output (CUDA only)
  - **XTTS-v2**: Multilingual voice cloning supporting 17 languages
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

Creates unique voices from natural language descriptions. No reference audio needed.

**Voice Description Example:**
```
Realistic female voice in the 20s age with american accent.
Medium-high pitch, bright timbre, energetic pacing, enthusiastic tone.
```

**Emotion Tags (inline):**
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

**Requirements:** CUDA GPU with 16GB+ VRAM (best), MPS, or CPU

### Chatterbox (Voice Cloning)

Clones voices from reference audio with emotion control.

**Emotion Tags:**
```
[laugh] [chuckle] [cough] [sigh] [gasp]
```

**Parameters:**
- `exaggeration`: 0.0-1.0+, controls expressiveness (default 0.5)
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

**Requirements:** Works on CUDA, MPS, and CPU

### MiraTTS (Fast Voice Cloning)

Fast voice cloning with high-quality 48kHz output.

**Example:**
```python
speak_mira(
    text="The quick brown fox jumps over the lazy dog.",
    output_path="/tmp/output.wav",
    reference_audio_paths=["/path/to/reference.wav"]
)
```

**Requirements:** NVIDIA GPU with CUDA (6GB+ VRAM). Does NOT support MPS or CPU.

### XTTS-v2 (Multilingual)

Multilingual voice cloning supporting 17 languages with cross-language cloning.

**Supported Languages:**
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

**Requirements:** Works on CUDA, MPS, and CPU. Model downloads automatically (~6GB).

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
