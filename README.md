# Audiobook MCP

An MCP (Model Context Protocol) server for orchestrating full-cast audiobook production. Manage voice assignments, organize characters, import prose text, and stitch audio files into complete audiobooks—all through a standardized interface that works with any MCP-compatible client.

## Features

- **Project Management**: Create and manage audiobook projects with metadata
- **Character & Voice Management**: Define characters with voice configurations (provider-agnostic)
- **Chapter & Segment Organization**: Structure your book into chapters and individual speech segments
- **Prose Import**: Import plain text with automatic dialogue detection and paragraph splitting
- **Voice Provider Agnostic**: Works with any TTS service (ElevenLabs, OpenAI, Azure, etc.)
- **Audio Stitching**: Combine segment audio files into chapters and final audiobooks with chapter markers
- **Export for Voice Generation**: Export character lines for batch processing with external voice tools

## Installation

### Prerequisites

- **Node.js** 18 or later
- **ffmpeg** (required for audio stitching features)

### From Source

```bash
git clone https://github.com/shawnrushefsky/audiobook-mcp.git
cd audiobook-mcp
npm install
npm run build
```

### Using Docker

```bash
# Pull from GitHub Container Registry
docker pull ghcr.io/shawnrushefsky/audiobook-mcp:latest

# Or build locally
docker build -t audiobook-mcp .
```

## Configuration

### Claude Desktop

Add to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "node",
      "args": ["/path/to/audiobook-mcp/dist/index.js"]
    }
  }
}
```

### Claude Code (CLI)

Add to your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "node",
      "args": ["/path/to/audiobook-mcp/dist/index.js"]
    }
  }
}
```

Or use the project-level `.mcp.json`:

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "node",
      "args": ["./dist/index.js"]
    }
  }
}
```

### Docker Configuration

For Claude Desktop or Claude Code with Docker:

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-v", "/path/to/your/projects:/projects",
        "ghcr.io/shawnrushefsky/audiobook-mcp:latest"
      ]
    }
  }
}
```

### Cursor

Add to Cursor's MCP configuration (Settings → MCP Servers):

```json
{
  "audiobook": {
    "command": "node",
    "args": ["/path/to/audiobook-mcp/dist/index.js"]
  }
}
```

### Windsurf

Add to your Windsurf MCP configuration:

```json
{
  "mcpServers": {
    "audiobook": {
      "command": "node",
      "args": ["/path/to/audiobook-mcp/dist/index.js"]
    }
  }
}
```

### Cline (VS Code Extension)

Add to Cline's MCP server settings in VS Code:

```json
{
  "cline.mcpServers": {
    "audiobook": {
      "command": "node",
      "args": ["/path/to/audiobook-mcp/dist/index.js"]
    }
  }
}
```

### Continue.dev

Add to your Continue configuration (`~/.continue/config.json`):

```json
{
  "mcpServers": [
    {
      "name": "audiobook",
      "command": "node",
      "args": ["/path/to/audiobook-mcp/dist/index.js"]
    }
  ]
}
```

### Zed Editor

Add to your Zed settings (`~/.config/zed/settings.json`):

```json
{
  "language_models": {
    "mcp_servers": {
      "audiobook": {
        "command": "node",
        "args": ["/path/to/audiobook-mcp/dist/index.js"]
      }
    }
  }
}
```

### Generic MCP Client (TypeScript/JavaScript)

```typescript
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const transport = new StdioClientTransport({
  command: "node",
  args: ["/path/to/audiobook-mcp/dist/index.js"],
});

const client = new Client({
  name: "my-client",
  version: "1.0.0",
});

await client.connect(transport);

// List available tools
const tools = await client.listTools();

// Call a tool
const result = await client.callTool({
  name: "init_project",
  arguments: {
    path: "/path/to/my-audiobook",
    title: "My Audiobook",
    author: "Author Name",
  },
});
```

### Python Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="node",
    args=["/path/to/audiobook-mcp/dist/index.js"],
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # List tools
        tools = await session.list_tools()

        # Call a tool
        result = await session.call_tool(
            "init_project",
            arguments={
                "path": "/path/to/my-audiobook",
                "title": "My Audiobook",
                "author": "Author Name",
            }
        )
```

## Available Tools

### Project Management

| Tool | Description |
|------|-------------|
| `init_project` | Initialize a new audiobook project in a directory |
| `open_project` | Open an existing audiobook project |
| `get_project_info` | Get project details and statistics |
| `update_project` | Update project metadata (title, author, description) |

### Character Management

| Tool | Description |
|------|-------------|
| `add_character` | Add a new character with name and description |
| `list_characters` | List all characters with segment counts |
| `update_character` | Update character details |
| `delete_character` | Remove a character |
| `set_voice` | Assign voice configuration (provider, voice_id, settings) |
| `clear_voice` | Remove voice assignment from a character |

### Chapter Management

| Tool | Description |
|------|-------------|
| `add_chapter` | Add a new chapter |
| `list_chapters` | List all chapters with segment statistics |
| `update_chapter` | Update chapter title |
| `delete_chapter` | Remove a chapter and all its segments |
| `reorder_chapters` | Change chapter order |

### Segment Management

| Tool | Description |
|------|-------------|
| `add_segment` | Add a text segment to a chapter |
| `list_segments` | List segments in a chapter with character info |
| `update_segment` | Update segment text or assigned character |
| `delete_segment` | Remove a segment |
| `reorder_segments` | Change segment order within a chapter |
| `get_pending_segments` | Get all segments missing audio files |

### Content Import

| Tool | Description |
|------|-------------|
| `import_chapter_text` | Import prose text with automatic splitting |
| `assign_dialogue` | Bulk assign character to segments matching a pattern |
| `export_character_lines` | Export all lines for a character (for batch TTS) |
| `detect_dialogue` | Analyze text and suggest character assignments |
| `get_line_distribution` | Get character line count statistics |

### Audio Management

| Tool | Description |
|------|-------------|
| `register_segment_audio` | Link an audio file to a segment |
| `get_chapter_audio_status` | Check which segments have/need audio |
| `stitch_chapter` | Combine segment audio into chapter MP3 |
| `get_stitch_status` | Get overall book audio readiness status |
| `stitch_book` | Create final audiobook with chapter markers |
| `clear_segment_audio` | Remove audio association from a segment |

## Workflow Example

### 1. Initialize a Project

```
Use init_project to create a new audiobook project at /path/to/my-book with title "The Great Adventure" by "Jane Author"
```

### 2. Add Characters

```
Add a character named "Narrator" and mark them as the narrator.
Add a character named "Alice" with description "The protagonist, a young adventurer"
Add a character named "Bob" with description "Alice's mentor"
```

### 3. Set Up Voices

```
Set voice for Alice using provider "elevenlabs" with voice_id "voice_abc123"
Set voice for Bob using provider "elevenlabs" with voice_id "voice_def456"
```

### 4. Add Chapters and Import Text

```
Add a chapter titled "Chapter 1: The Beginning"
Import this text into Chapter 1, using the Narrator character as default:

Alice walked through the forest, her heart pounding.
"I can't believe I'm finally here," she said.
Bob appeared from behind a tree. "You made it," he replied with a smile.
```

### 5. Assign Dialogue

```
Use detect_dialogue on Chapter 1 to see unassigned segments.
Assign segments containing "I can't believe" to Alice.
Assign segments containing "You made it" to Bob.
```

### 6. Export for Voice Generation

```
Export all lines for Alice - this gives you the text to generate audio with your TTS tool.
```

### 7. Register Audio

After generating audio with your TTS tool (e.g., ElevenLabs):

```
Register audio file /path/to/segment1.mp3 for segment <segment-id>
```

### 8. Stitch and Export

```
Check the stitch status to see if all segments have audio.
Stitch Chapter 1 into a single audio file.
When all chapters are ready, stitch the complete book with chapter markers.
```

## Project Storage Structure

Each project creates a `.audiobook` folder in your specified directory:

```
your-project/
└── .audiobook/
    ├── db.sqlite              # Project database
    ├── audio/
    │   └── segments/          # Individual segment audio files
    └── exports/
        ├── chapters/          # Stitched chapter audio
        └── book/              # Final audiobook output
```

## Voice Configuration

Voice configurations are stored as JSON and are provider-agnostic:

```json
{
  "provider": "elevenlabs",
  "voice_id": "voice_abc123",
  "settings": {
    "stability": 0.5,
    "similarity_boost": 0.75
  }
}
```

This allows you to use any voice provider (ElevenLabs, OpenAI, Azure, Google, local models, etc.) and store provider-specific settings.

## Integration with Voice Services

This MCP server manages the audiobook structure but delegates voice generation to external tools. Common workflows:

### With ElevenLabs MCP

1. Use `export_character_lines` to get text for each character
2. Call ElevenLabs MCP to generate audio for each line
3. Use `register_segment_audio` to link generated audio back to segments

### With OpenAI TTS

1. Export character lines
2. Use OpenAI's TTS API to generate audio
3. Register the audio files with segments

### With Local TTS (Coqui, Bark, etc.)

1. Export character lines to a JSON file
2. Process with your local TTS pipeline
3. Register resulting audio files

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode for development
npm run dev
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
