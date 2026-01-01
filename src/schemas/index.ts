import { z } from "zod";

// Project schemas
export const initProjectSchema = z.object({
  path: z.string().describe("Absolute path to the directory where the project will be created"),
  title: z.string().describe("Title of the audiobook"),
  author: z.string().optional().describe("Author of the audiobook"),
  description: z.string().optional().describe("Description of the audiobook project"),
});

export const openProjectSchema = z.object({
  path: z.string().describe("Absolute path to an existing audiobook project directory"),
});

export const updateProjectSchema = z.object({
  title: z.string().optional().describe("New title for the audiobook"),
  author: z.string().optional().describe("New author name"),
  description: z.string().optional().describe("New description"),
});

// Character schemas
export const addCharacterSchema = z.object({
  name: z.string().describe("Name of the character"),
  description: z.string().optional().describe("Description of the character"),
  is_narrator: z.boolean().optional().default(false).describe("Whether this character is a narrator"),
});

export const updateCharacterSchema = z.object({
  id: z.string().describe("ID of the character to update"),
  name: z.string().optional().describe("New name for the character"),
  description: z.string().optional().describe("New description"),
  is_narrator: z.boolean().optional().describe("Whether this character is a narrator"),
});

export const deleteCharacterSchema = z.object({
  id: z.string().describe("ID of the character to delete"),
});

export const setVoiceSchema = z.object({
  character_id: z.string().describe("ID of the character"),
  provider: z.string().describe("Voice provider (e.g., 'elevenlabs', 'openai')"),
  voice_id: z.string().describe("Voice ID from the provider"),
  settings: z.record(z.unknown()).optional().describe("Additional provider-specific settings"),
});

// Chapter schemas
export const addChapterSchema = z.object({
  title: z.string().describe("Title of the chapter"),
  sort_order: z.number().optional().describe("Position in the chapter order (defaults to end)"),
});

export const updateChapterSchema = z.object({
  id: z.string().describe("ID of the chapter to update"),
  title: z.string().optional().describe("New title for the chapter"),
});

export const deleteChapterSchema = z.object({
  id: z.string().describe("ID of the chapter to delete"),
});

export const reorderChaptersSchema = z.object({
  chapter_ids: z.array(z.string()).describe("Array of chapter IDs in the desired order"),
});

// Segment schemas
export const addSegmentSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter this segment belongs to"),
  text_content: z.string().describe("The text content to be spoken"),
  character_id: z.string().optional().describe("ID of the character speaking this segment"),
  sort_order: z.number().optional().describe("Position within the chapter (defaults to end)"),
});

export const updateSegmentSchema = z.object({
  id: z.string().describe("ID of the segment to update"),
  text_content: z.string().optional().describe("New text content"),
  character_id: z.string().optional().nullable().describe("New character ID (null to unassign)"),
});

export const deleteSegmentSchema = z.object({
  id: z.string().describe("ID of the segment to delete"),
});

export const listSegmentsSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter to list segments from"),
});

export const reorderSegmentsSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter"),
  segment_ids: z.array(z.string()).describe("Array of segment IDs in the desired order"),
});

// Import schemas
export const importChapterTextSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter to import into"),
  text: z.string().describe("The prose text to import"),
  default_character_id: z.string().optional().describe("Default character for non-dialogue text (narrator)"),
});

export const assignDialogueSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter"),
  pattern: z.string().describe("Text pattern to match (e.g., character name in dialogue tags)"),
  character_id: z.string().describe("Character to assign matching segments to"),
});

export const exportCharacterLinesSchema = z.object({
  character_id: z.string().describe("ID of the character to export lines for"),
});

// Audio schemas
export const registerSegmentAudioSchema = z.object({
  segment_id: z.string().describe("ID of the segment"),
  audio_path: z.string().describe("Path to the audio file"),
  duration_ms: z.number().optional().describe("Duration in milliseconds (auto-detected if not provided)"),
});

export const stitchChapterSchema = z.object({
  chapter_id: z.string().describe("ID of the chapter to stitch"),
  output_filename: z.string().optional().describe("Output filename (defaults to chapter title)"),
});

export const stitchBookSchema = z.object({
  output_filename: z.string().optional().describe("Output filename (defaults to project title)"),
  include_chapter_markers: z.boolean().optional().default(true).describe("Include chapter markers in the MP3"),
});
