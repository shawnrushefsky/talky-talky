#!/usr/bin/env node
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { closeDatabase } from "./db/connection.js";
import {
  initProject,
  openProject,
  getProjectInfo,
  updateProject,
} from "./tools/projects.js";
import {
  initProjectSchema,
  openProjectSchema,
  updateProjectSchema,
  addCharacterSchema,
  updateCharacterSchema,
  deleteCharacterSchema,
  setVoiceSchema,
  addChapterSchema,
  updateChapterSchema,
  deleteChapterSchema,
  reorderChaptersSchema,
  addSegmentSchema,
  updateSegmentSchema,
  deleteSegmentSchema,
  listSegmentsSchema,
  reorderSegmentsSchema,
  importChapterTextSchema,
  assignDialogueSchema,
  exportCharacterLinesSchema,
  registerSegmentAudioSchema,
  stitchChapterSchema,
  stitchBookSchema,
} from "./schemas/index.js";
import {
  addCharacter,
  listCharacters,
  getCharacter,
  updateCharacter,
  deleteCharacter,
  setVoice,
  clearVoice,
  getCharactersWithStats,
} from "./tools/characters.js";
import {
  addChapter,
  listChapters,
  getChapter,
  updateChapter,
  deleteChapter,
  reorderChapters,
  getChaptersWithStats,
} from "./tools/chapters.js";
import {
  addSegment,
  listSegments,
  getSegment,
  updateSegment,
  deleteSegment,
  reorderSegments,
  getSegmentsWithCharacters,
  getPendingSegments,
} from "./tools/segments.js";
import {
  importChapterText,
  assignDialogue,
  exportCharacterLines,
  detectDialogue,
  getLineDistribution,
} from "./tools/import.js";
import {
  registerSegmentAudio,
  getChapterAudioStatus,
  stitchChapter,
  getStitchStatus,
  stitchBook,
  clearSegmentAudio,
} from "./tools/audio.js";

// Create MCP server
const server = new McpServer({
  name: "audiobook-mcp",
  version: "0.1.0",
});

// =============================================================================
// Project Management Tools
// =============================================================================

server.tool(
  "init_project",
  "Initialize a new audiobook project in a directory. Creates .audiobook folder with database and directory structure.",
  {
    path: initProjectSchema.shape.path,
    title: initProjectSchema.shape.title,
    author: initProjectSchema.shape.author,
    description: initProjectSchema.shape.description,
  },
  async (params) => {
    try {
      const project = initProject(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Project "${project.title}" initialized successfully`,
                project,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error initializing project: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "open_project",
  "Open an existing audiobook project. Required before using other project-specific tools.",
  {
    path: openProjectSchema.shape.path,
  },
  async (params) => {
    try {
      const project = openProject(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Project "${project.title}" opened successfully`,
                project,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error opening project: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "get_project_info",
  "Get information about the currently open audiobook project including statistics.",
  {},
  async () => {
    try {
      const info = getProjectInfo();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(info, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error getting project info: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "update_project",
  "Update the metadata of the currently open audiobook project.",
  {
    title: updateProjectSchema.shape.title,
    author: updateProjectSchema.shape.author,
    description: updateProjectSchema.shape.description,
  },
  async (params) => {
    try {
      const project = updateProject(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Project updated successfully",
                project,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error updating project: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Character Management Tools
// =============================================================================

server.tool(
  "add_character",
  "Add a new character to the audiobook project. Characters can be assigned voices and speak segments.",
  {
    name: addCharacterSchema.shape.name,
    description: addCharacterSchema.shape.description,
    is_narrator: addCharacterSchema.shape.is_narrator,
  },
  async (params) => {
    try {
      const character = addCharacter(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Character "${character.name}" added successfully`,
                character,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error adding character: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "list_characters",
  "List all characters in the project with their segment counts and voice configurations.",
  {},
  async () => {
    try {
      const characters = getCharactersWithStats();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                count: characters.length,
                characters: characters.map((c) => ({
                  ...c,
                  voice_config: c.voice_config ? JSON.parse(c.voice_config) : null,
                })),
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error listing characters: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "update_character",
  "Update an existing character's name, description, or narrator status.",
  {
    id: updateCharacterSchema.shape.id,
    name: updateCharacterSchema.shape.name,
    description: updateCharacterSchema.shape.description,
    is_narrator: updateCharacterSchema.shape.is_narrator,
  },
  async (params) => {
    try {
      const character = updateCharacter(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Character "${character.name}" updated successfully`,
                character,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error updating character: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "delete_character",
  "Delete a character from the project. Segments assigned to this character will become unassigned.",
  {
    id: deleteCharacterSchema.shape.id,
  },
  async (params) => {
    try {
      const character = getCharacter(params.id);
      if (!character) {
        throw new Error(`Character not found: ${params.id}`);
      }
      deleteCharacter(params.id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Character "${character.name}" deleted successfully`,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error deleting character: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "set_voice",
  "Assign a voice configuration to a character. Voice configs are provider-agnostic (works with 11Labs, OpenAI, etc).",
  {
    character_id: setVoiceSchema.shape.character_id,
    provider: setVoiceSchema.shape.provider,
    voice_id: setVoiceSchema.shape.voice_id,
    settings: setVoiceSchema.shape.settings,
  },
  async (params) => {
    try {
      const character = setVoice(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Voice assigned to "${character.name}" successfully`,
                character: {
                  ...character,
                  voice_config: character.voice_config
                    ? JSON.parse(character.voice_config)
                    : null,
                },
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error setting voice: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "clear_voice",
  "Remove the voice configuration from a character.",
  {
    character_id: setVoiceSchema.shape.character_id,
  },
  async (params) => {
    try {
      const character = clearVoice(params.character_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Voice cleared from "${character.name}" successfully`,
                character,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error clearing voice: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Chapter Management Tools
// =============================================================================

server.tool(
  "add_chapter",
  "Add a new chapter to the audiobook. Chapters contain segments of text to be narrated.",
  {
    title: addChapterSchema.shape.title,
    sort_order: addChapterSchema.shape.sort_order,
  },
  async (params) => {
    try {
      const chapter = addChapter(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Chapter "${chapter.title}" added successfully`,
                chapter,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error adding chapter: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "list_chapters",
  "List all chapters in the project with their segment statistics.",
  {},
  async () => {
    try {
      const chapters = getChaptersWithStats();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                count: chapters.length,
                chapters,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error listing chapters: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "update_chapter",
  "Update a chapter's title.",
  {
    id: updateChapterSchema.shape.id,
    title: updateChapterSchema.shape.title,
  },
  async (params) => {
    try {
      const chapter = updateChapter(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Chapter "${chapter.title}" updated successfully`,
                chapter,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error updating chapter: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "delete_chapter",
  "Delete a chapter and all its segments.",
  {
    id: deleteChapterSchema.shape.id,
  },
  async (params) => {
    try {
      const chapter = getChapter(params.id);
      if (!chapter) {
        throw new Error(`Chapter not found: ${params.id}`);
      }
      deleteChapter(params.id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Chapter "${chapter.title}" and all its segments deleted successfully`,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error deleting chapter: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "reorder_chapters",
  "Reorder chapters by providing an array of chapter IDs in the desired order.",
  {
    chapter_ids: reorderChaptersSchema.shape.chapter_ids,
  },
  async (params) => {
    try {
      const chapters = reorderChapters(params.chapter_ids);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Chapters reordered successfully",
                chapters,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error reordering chapters: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Segment Management Tools
// =============================================================================

server.tool(
  "add_segment",
  "Add a text segment to a chapter. Segments are individual pieces of narration assigned to characters.",
  {
    chapter_id: addSegmentSchema.shape.chapter_id,
    text_content: addSegmentSchema.shape.text_content,
    character_id: addSegmentSchema.shape.character_id,
    sort_order: addSegmentSchema.shape.sort_order,
  },
  async (params) => {
    try {
      const segment = addSegment(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Segment added successfully",
                segment,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error adding segment: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "list_segments",
  "List all segments in a chapter with character information.",
  {
    chapter_id: listSegmentsSchema.shape.chapter_id,
  },
  async (params) => {
    try {
      const segments = getSegmentsWithCharacters(params.chapter_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                count: segments.length,
                segments,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error listing segments: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "update_segment",
  "Update a segment's text content or assigned character.",
  {
    id: updateSegmentSchema.shape.id,
    text_content: updateSegmentSchema.shape.text_content,
    character_id: updateSegmentSchema.shape.character_id,
  },
  async (params) => {
    try {
      const segment = updateSegment(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Segment updated successfully",
                segment,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error updating segment: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "delete_segment",
  "Delete a segment from a chapter.",
  {
    id: deleteSegmentSchema.shape.id,
  },
  async (params) => {
    try {
      const segment = getSegment(params.id);
      if (!segment) {
        throw new Error(`Segment not found: ${params.id}`);
      }
      deleteSegment(params.id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Segment deleted successfully",
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error deleting segment: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "reorder_segments",
  "Reorder segments within a chapter by providing an array of segment IDs in the desired order.",
  {
    chapter_id: reorderSegmentsSchema.shape.chapter_id,
    segment_ids: reorderSegmentsSchema.shape.segment_ids,
  },
  async (params) => {
    try {
      const segments = reorderSegments(params.chapter_id, params.segment_ids);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Segments reordered successfully",
                count: segments.length,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error reordering segments: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "get_pending_segments",
  "Get all segments that are missing audio files, organized by chapter and character.",
  {},
  async () => {
    try {
      const segments = getPendingSegments();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                count: segments.length,
                segments,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error getting pending segments: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Content Import Tools
// =============================================================================

server.tool(
  "import_chapter_text",
  "Import prose text into a chapter, automatically splitting into segments at paragraph and dialogue boundaries.",
  {
    chapter_id: importChapterTextSchema.shape.chapter_id,
    text: importChapterTextSchema.shape.text,
    default_character_id: importChapterTextSchema.shape.default_character_id,
  },
  async (params) => {
    try {
      const result = importChapterText(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Imported ${result.segments_created} segments`,
                ...result,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error importing text: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "assign_dialogue",
  "Assign a character to all segments matching a text pattern (useful for bulk character assignment).",
  {
    chapter_id: assignDialogueSchema.shape.chapter_id,
    pattern: assignDialogueSchema.shape.pattern,
    character_id: assignDialogueSchema.shape.character_id,
  },
  async (params) => {
    try {
      const result = assignDialogue(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Updated ${result.updated_count} segments`,
                ...result,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error assigning dialogue: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "export_character_lines",
  "Export all lines for a specific character (useful for batch voice generation with external tools).",
  {
    character_id: exportCharacterLinesSchema.shape.character_id,
  },
  async (params) => {
    try {
      const result = exportCharacterLines(params.character_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error exporting lines: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "detect_dialogue",
  "Analyze a chapter to detect dialogue and suggest character assignments.",
  {
    chapter_id: listSegmentsSchema.shape.chapter_id,
  },
  async (params) => {
    try {
      const result = detectDialogue(params.chapter_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error detecting dialogue: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "get_line_distribution",
  "Get a summary of how lines are distributed across characters in the project.",
  {},
  async () => {
    try {
      const result = getLineDistribution();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error getting distribution: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Audio Management Tools
// =============================================================================

server.tool(
  "register_segment_audio",
  "Register an audio file for a segment. The audio will be copied to the project's audio directory.",
  {
    segment_id: registerSegmentAudioSchema.shape.segment_id,
    audio_path: registerSegmentAudioSchema.shape.audio_path,
    duration_ms: registerSegmentAudioSchema.shape.duration_ms,
  },
  async (params) => {
    try {
      const result = registerSegmentAudio(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Audio registered successfully",
                ...result,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error registering audio: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "get_chapter_audio_status",
  "Get the audio status for a chapter, including which segments are missing audio.",
  {
    chapter_id: stitchChapterSchema.shape.chapter_id,
  },
  async (params) => {
    try {
      const result = getChapterAudioStatus(params.chapter_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error getting status: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "stitch_chapter",
  "Combine all segment audio files in a chapter into a single MP3 file.",
  {
    chapter_id: stitchChapterSchema.shape.chapter_id,
    output_filename: stitchChapterSchema.shape.output_filename,
  },
  async (params) => {
    try {
      const result = stitchChapter(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: `Chapter "${result.chapter_title}" stitched successfully`,
                ...result,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error stitching chapter: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "get_stitch_status",
  "Get the overall audio status for the entire book, showing which chapters are ready to stitch.",
  {},
  async () => {
    try {
      const result = getStitchStatus();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(result, null, 2),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error getting status: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "stitch_book",
  "Combine all chapters into a final audiobook MP3 with chapter markers.",
  {
    output_filename: stitchBookSchema.shape.output_filename,
    include_chapter_markers: stitchBookSchema.shape.include_chapter_markers,
  },
  async (params) => {
    try {
      const result = stitchBook(params);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Audiobook created successfully",
                ...result,
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error stitching book: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

server.tool(
  "clear_segment_audio",
  "Remove the audio file association from a segment.",
  {
    segment_id: registerSegmentAudioSchema.shape.segment_id,
  },
  async (params) => {
    try {
      clearSegmentAudio(params.segment_id);
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              {
                success: true,
                message: "Audio cleared from segment",
              },
              null,
              2
            ),
          },
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error clearing audio: ${error instanceof Error ? error.message : String(error)}`,
          },
        ],
        isError: true,
      };
    }
  }
);

// =============================================================================
// Server Lifecycle
// =============================================================================

// Graceful shutdown
process.on("SIGINT", () => {
  closeDatabase();
  process.exit(0);
});

process.on("SIGTERM", () => {
  closeDatabase();
  process.exit(0);
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Audiobook MCP server started");
}

main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
