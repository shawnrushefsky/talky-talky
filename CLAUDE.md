# Audiobook MCP - Development Guide

This document provides context for Claude Code and other AI assistants working on this codebase.

## Project Overview

Audiobook MCP is a Model Context Protocol (MCP) server that orchestrates full-cast audiobook production. It manages the organization of voices, characters, chapters, and segments, with **native TTS capabilities**:

- **Maya1**: Text-guided voice design with 20+ emotion tags (creates unique voices from descriptions)
- **Fish Speech**: High-quality voice cloning from reference audio samples

## Architecture

### Technology Stack

- **Runtime**: Python 3.10+
- **MCP SDK**: `mcp` with FastMCP for server implementation
- **Database**: SQLite (per-project databases)
- **Audio Processing**: ffmpeg for audio stitching and format conversion
- **TTS Engines**:
  - Maya1 (local, requires GPU) - voice design from text descriptions
  - Fish Speech (local server or cloud API) - voice cloning from samples

### Directory Structure

```
audiobook_mcp/
├── __init__.py
├── server.py             # MCP server entry point, tool registrations
├── db/
│   ├── __init__.py
│   ├── connection.py     # SQLite connection management (singleton per project)
│   └── schema.py         # Table definitions and dataclasses
├── tools/
│   ├── __init__.py
│   ├── projects.py       # Project CRUD operations
│   ├── characters.py     # Character management
│   ├── chapters.py       # Chapter management
│   ├── segments.py       # Segment management
│   ├── voice_samples.py  # Voice sample storage for cloning
│   ├── import_tools.py   # Text import and dialogue detection
│   ├── audio.py          # Audio registration and stitching
│   └── tts.py            # TTS with Maya1 and Fish Speech
└── utils/
    ├── __init__.py
    ├── parser.py         # Text parsing utilities (dialogue detection)
    └── ffmpeg.py         # ffmpeg wrapper functions
```

### Data Model

```
Project (1) ←→ (N) Characters
Project (1) ←→ (N) Chapters
Chapter (1) ←→ (N) Segments
Segment (N) ←→ (1) Character (nullable)
Character (1) ←→ (N) VoiceSamples
```

- **Project**: Top-level container with metadata (title, author)
- **Character**: A speaking role with voice configuration
- **Chapter**: A chapter containing ordered segments
- **Segment**: A piece of text to be spoken, optionally assigned to a character
- **VoiceSample**: Reference audio for voice cloning (multiple per character)

### Key Design Decisions

1. **Per-Project Storage**: Each project has its own `.audiobook/` folder containing the SQLite database and audio files. This makes projects portable and self-contained.

2. **Voice Provider Agnostic**: Voice configurations are stored as JSON with `provider`, `voice_id`, and optional `settings`. This allows integration with any TTS service.

3. **Voice Cloning Workflow**: Use Maya1 to create voice samples from descriptions, then use Fish Speech to clone those voices for long-form generation.

4. **Single Active Project**: The server maintains one open project at a time via the connection singleton.

## Installation & Setup

```bash
# Install base package
pip install -e .

# Install with Maya1 TTS support (requires CUDA GPU)
pip install -e ".[maya1]"

# Install with Fish Speech support
pip install -e ".[fish-speech]"

# Install all TTS engines
pip install -e ".[tts]"

# Install development dependencies
pip install -e ".[dev]"
```

### Environment Variables

```bash
# Fish Speech local server (default: http://localhost:8080)
export FISH_SPEECH_API_URL=http://localhost:8080

# Fish Speech cloud API (alternative to local server)
export FISH_AUDIO_API_KEY=your_api_key_here
```

## Running the Server

```bash
# Run the MCP server (communicates via stdio)
audiobook-mcp

# Or run directly
python -m audiobook_mcp.server
```

## Adding New Tools

1. **Implement business logic** in appropriate `audiobook_mcp/tools/*.py` file:
   ```python
   from ..db.connection import get_database
   from dataclasses import dataclass

   @dataclass
   class MyResult:
       value: str

   def my_function(param: str) -> MyResult:
       db = get_database()
       cursor = db.cursor()
       # Implementation
       return MyResult(value="result")
   ```

2. **Register MCP tool** in `audiobook_mcp/server.py`:
   ```python
   from .tools.my_module import my_function

   @mcp.tool()
   def my_tool(param: str) -> dict:
       """Description of what this tool does."""
       result = my_function(param)
       return {"success": True, **to_dict(result)}
   ```

## Common Patterns

### Database Access

```python
from ..db.connection import get_database

def my_function():
    db = get_database()  # Throws if no project open
    cursor = db.cursor()
    cursor.execute("SELECT * FROM table WHERE id = ?", (id,))
    row = cursor.fetchone()
    return row
```

### Error Handling

FastMCP handles errors automatically. Raise exceptions and they'll be returned as error responses:

```python
@mcp.tool()
def my_tool(param: str) -> dict:
    if not param:
        raise ValueError("param is required")
    # ...
```

### File Paths

- Store relative paths in database (e.g., `audio/segments/abc.mp3`)
- Use `get_audiobook_dir()` to get absolute paths when needed
- Always validate file existence before operations

## Audio Processing

The `audiobook_mcp/utils/ffmpeg.py` module wraps ffmpeg commands:

- `check_ffmpeg()` - Verify ffmpeg is available
- `get_audio_duration()` - Get duration in milliseconds
- `validate_audio_file()` - Check file is valid audio
- `concatenate_audio_files()` - Join multiple audio files
- `create_audiobook_with_chapters()` - Create MP3 with ID3 chapter markers

## TTS Engines

### Maya1 (Voice Design)

Maya1 creates unique voices from natural language descriptions with 20+ inline emotion tags.

**Requirements:**
- Python 3.10+
- CUDA GPU with 16GB+ VRAM (RTX 4090, A100, H100)
- ~10GB disk space for model weights

**Emotion Tags:**
```
<laugh> <laugh_harder> <chuckle> <giggle> <snort>
<cry> <sob> <sigh> <gasp> <groan>
<whisper> <angry> <yell> <scream>
<cough> <clear_throat> <sniff> <hum> <mumble> <stutter>
```

**Voice Description Format:**
```
"Realistic [gender] voice in the [age]s age with [accent] accent. [pitch] pitch, [timbre] timbre, [pacing] pacing, [tone] tone."
```

**Options:**
- Gender: male, female
- Age: 10s, 20s, 30s, 40s, 50s, 60s, 70s
- Accent: american, british, australian, indian
- Pitch: low, medium-low, medium, medium-high, high
- Timbre: warm, bright, gravelly, smooth, cold, gentle, strong
- Pacing: slow, measured, moderate, energetic, fast
- Tone: professional, conversational, enthusiastic, wise, menacing, warm, determined

### Fish Speech (Voice Cloning)

Fish Speech clones voices from reference audio samples. Supports both local server and cloud API.

**Local Server Setup:**
```bash
# Run Fish Speech server (see Fish Speech docs)
# Set environment variable
export FISH_SPEECH_API_URL=http://localhost:8080
```

**Cloud API Setup:**
```bash
export FISH_AUDIO_API_KEY=your_api_key_here
```

### Recommended Workflow

1. **Create voice samples with Maya1**: Generate 10-30 seconds of reference audio for each character using voice descriptions
2. **Clone voices with Fish Speech**: Use the samples to clone the voice for long-form generation
3. **Generate segment audio**: Use Fish Speech to generate audio for all dialogue segments
4. **Stitch audiobook**: Combine all segments into chapters and the final audiobook

## Debugging

- The server logs to stderr (stdout is reserved for MCP protocol)
- Use `print(..., file=sys.stderr)` for debug logging
- Check `.audiobook/db.sqlite` directly for database state

## Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=audiobook_mcp
```
