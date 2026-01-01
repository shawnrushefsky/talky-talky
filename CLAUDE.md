# Audiobook MCP - Development Guide

This document provides context for Claude Code and other AI assistants working on this codebase.

## Project Overview

Audiobook MCP is a Model Context Protocol (MCP) server that orchestrates full-cast audiobook production. It manages the organization of voices, characters, chapters, and segments, while delegating actual voice synthesis to external tools like ElevenLabs MCP.

## Architecture

### Technology Stack

- **Runtime**: Node.js 18+ with TypeScript (ES modules)
- **MCP SDK**: `@modelcontextprotocol/sdk` for MCP server implementation
- **Database**: SQLite via `better-sqlite3` (per-project databases)
- **Validation**: Zod for schema validation
- **Audio Processing**: ffmpeg via child_process for audio stitching

### Directory Structure

```
src/
├── index.ts              # MCP server entry point, tool registrations
├── db/
│   ├── connection.ts     # SQLite connection management (singleton per project)
│   └── schema.ts         # Table definitions and TypeScript types
├── tools/
│   ├── projects.ts       # Project CRUD operations
│   ├── characters.ts     # Character management
│   ├── chapters.ts       # Chapter management
│   ├── segments.ts       # Segment management
│   ├── import.ts         # Text import and dialogue detection
│   └── audio.ts          # Audio registration and stitching
├── schemas/
│   └── index.ts          # Zod schemas for all tool inputs
└── utils/
    ├── parser.ts         # Text parsing utilities (dialogue detection)
    └── ffmpeg.ts         # ffmpeg wrapper functions
```

### Data Model

```
Project (1) ←→ (N) Characters
Project (1) ←→ (N) Chapters
Chapter (1) ←→ (N) Segments
Segment (N) ←→ (1) Character (nullable)
```

- **Project**: Top-level container with metadata (title, author)
- **Character**: A speaking role with voice configuration
- **Chapter**: A chapter containing ordered segments
- **Segment**: A piece of text to be spoken, optionally assigned to a character

### Key Design Decisions

1. **Per-Project Storage**: Each project has its own `.audiobook/` folder containing the SQLite database and audio files. This makes projects portable and self-contained.

2. **Voice Provider Agnostic**: Voice configurations are stored as JSON with `provider`, `voice_id`, and optional `settings`. This allows integration with any TTS service.

3. **Separation of Concerns**: The MCP server handles organization and stitching, but not voice synthesis. Voice generation is delegated to external tools/MCPs.

4. **Single Active Project**: The server maintains one open project at a time via the connection singleton.

## Build & Development

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Watch mode
npm run dev
```

## Testing the Server

The server communicates via stdio. For manual testing:

```bash
# Build first
npm run build

# Run directly (will wait for MCP protocol messages)
node dist/index.js
```

For integration testing, use an MCP client or the MCP inspector tool.

## Adding New Tools

1. **Add Zod schema** in `src/schemas/index.ts`:
   ```typescript
   export const myToolSchema = z.object({
     param: z.string().describe("Description shown in tool docs"),
   });
   ```

2. **Implement business logic** in appropriate `src/tools/*.ts` file:
   ```typescript
   export function myTool(params: { param: string }): Result {
     const db = getDatabase();
     // Implementation
     return result;
   }
   ```

3. **Register MCP tool** in `src/index.ts`:
   ```typescript
   server.tool(
     "my_tool",
     "Description of what this tool does",
     { param: myToolSchema.shape.param },
     async (params) => {
       try {
         const result = myTool(params);
         return {
           content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
         };
       } catch (error) {
         return {
           content: [{ type: "text", text: `Error: ${error.message}` }],
           isError: true,
         };
       }
     }
   );
   ```

## Common Patterns

### Database Access

```typescript
import { getDatabase } from "../db/connection.js";

export function myFunction() {
  const db = getDatabase(); // Throws if no project open
  const stmt = db.prepare(`SELECT * FROM table WHERE id = ?`);
  const row = stmt.get(id);
  return row;
}
```

### Error Handling

All tools should catch errors and return them in MCP format:
```typescript
return {
  content: [{ type: "text", text: `Error: ${error.message}` }],
  isError: true,
};
```

### File Paths

- Store relative paths in database (e.g., `audio/segments/abc.mp3`)
- Use `getAudiobookDir()` to get absolute paths when needed
- Always validate file existence before operations

## Audio Processing

The `src/utils/ffmpeg.ts` module wraps ffmpeg commands:

- `checkFfmpeg()` - Verify ffmpeg is available
- `getAudioDuration()` - Get duration in milliseconds
- `validateAudioFile()` - Check file is valid audio
- `concatenateAudioFiles()` - Join multiple audio files
- `createAudiobookWithChapters()` - Create MP3 with ID3 chapter markers

## MCP Protocol Notes

- Tools receive parameters as a flat object
- Tool responses must have `content` array with typed blocks
- Use `isError: true` for error responses
- Descriptions on Zod fields become tool parameter documentation

## Debugging

- The server logs to stderr (stdout is reserved for MCP protocol)
- Use `console.error()` for debug logging
- Check `.audiobook/db.sqlite` directly for database state

## Docker

The Dockerfile creates a multi-stage build:
1. **Builder stage**: Compiles TypeScript
2. **Production stage**: Minimal image with Node.js, ffmpeg, and compiled JS

Mount project directories as volumes when running in Docker.
