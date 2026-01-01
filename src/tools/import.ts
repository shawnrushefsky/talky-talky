import { getDatabase } from "../db/connection.js";
import type { Segment } from "../db/schema.js";
import { getChapter } from "./chapters.js";
import { getCharacter, listCharacters } from "./characters.js";
import { bulkAddSegments, listSegments, updateSegment } from "./segments.js";
import {
  parseText,
  extractCharacterNames,
  cleanForTTS,
  type ParsedSegment,
} from "../utils/parser.js";

export interface ImportChapterTextParams {
  chapter_id: string;
  text: string;
  default_character_id?: string;
}

export interface AssignDialogueParams {
  chapter_id: string;
  pattern: string;
  character_id: string;
}

/**
 * Import prose text into a chapter, splitting into segments
 */
export function importChapterText(params: ImportChapterTextParams): {
  segments_created: number;
  dialogue_segments: number;
  narration_segments: number;
  detected_names: string[];
} {
  const { chapter_id, text, default_character_id } = params;

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  // Verify default character exists if provided
  if (default_character_id) {
    const character = getCharacter(default_character_id);
    if (!character) {
      throw new Error(`Character not found: ${default_character_id}`);
    }
  }

  // Parse the text
  const parsed = parseText(text);
  const detectedNames = extractCharacterNames(text);

  // Clean and prepare segments for insertion
  const segmentsToAdd = parsed.map((seg) => ({
    text_content: cleanForTTS(seg.text),
    // Assign default character to narration segments only
    character_id: !seg.is_dialogue ? default_character_id : undefined,
  }));

  // Bulk add segments
  bulkAddSegments(chapter_id, segmentsToAdd);

  // Count stats
  const dialogueCount = parsed.filter((s) => s.is_dialogue).length;
  const narrationCount = parsed.filter((s) => !s.is_dialogue).length;

  return {
    segments_created: parsed.length,
    dialogue_segments: dialogueCount,
    narration_segments: narrationCount,
    detected_names: detectedNames,
  };
}

/**
 * Assign a character to all dialogue segments matching a pattern
 */
export function assignDialogue(params: AssignDialogueParams): {
  updated_count: number;
  updated_segments: string[];
} {
  const { chapter_id, pattern, character_id } = params;

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  // Verify character exists
  const character = getCharacter(character_id);
  if (!character) {
    throw new Error(`Character not found: ${character_id}`);
  }

  // Get all segments in the chapter
  const segments = listSegments(chapter_id);

  // Find segments matching the pattern (case-insensitive)
  const regex = new RegExp(pattern, "i");
  const matchingSegments = segments.filter((s) =>
    regex.test(s.text_content)
  );

  // Update matching segments
  const updatedIds: string[] = [];
  for (const segment of matchingSegments) {
    updateSegment({ id: segment.id, character_id });
    updatedIds.push(segment.id);
  }

  return {
    updated_count: updatedIds.length,
    updated_segments: updatedIds,
  };
}

/**
 * Export all lines for a specific character (for batch voice generation)
 */
export function exportCharacterLines(character_id: string): {
  character_name: string;
  total_lines: number;
  total_characters: number;
  lines: Array<{
    segment_id: string;
    chapter_title: string;
    text: string;
    has_audio: boolean;
  }>;
} {
  const db = getDatabase();

  // Verify character exists
  const character = getCharacter(character_id);
  if (!character) {
    throw new Error(`Character not found: ${character_id}`);
  }

  // Get all segments for this character with chapter info
  const stmt = db.prepare(`
    SELECT s.*, ch.title as chapter_title
    FROM segments s
    JOIN chapters ch ON s.chapter_id = ch.id
    WHERE s.character_id = ?
    ORDER BY ch.sort_order ASC, s.sort_order ASC
  `);

  const rows = stmt.all(character_id) as Array<Record<string, unknown>>;

  const lines = rows.map((row) => ({
    segment_id: row.id as string,
    chapter_title: row.chapter_title as string,
    text: row.text_content as string,
    has_audio: row.audio_path !== null,
  }));

  const totalCharacters = lines.reduce((sum, l) => sum + l.text.length, 0);

  return {
    character_name: character.name,
    total_lines: lines.length,
    total_characters: totalCharacters,
    lines,
  };
}

/**
 * Detect potential dialogue and suggest character assignments
 */
export function detectDialogue(chapter_id: string): {
  total_segments: number;
  unassigned_segments: number;
  detected_names: string[];
  suggestions: Array<{
    segment_id: string;
    text_preview: string;
    potential_speaker?: string;
  }>;
} {
  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  const segments = listSegments(chapter_id);
  const fullText = segments.map((s) => s.text_content).join(" ");
  const detectedNames = extractCharacterNames(fullText);

  // Get existing characters for matching
  const characters = listCharacters();
  const characterNames = new Set(characters.map((c) => c.name.toLowerCase()));

  // Find unassigned segments and try to suggest speakers
  const unassigned = segments.filter((s) => !s.character_id);
  const suggestions = unassigned.slice(0, 20).map((seg) => {
    // Try to find a speaker from detected names or existing characters
    let potentialSpeaker: string | undefined;

    // Check if any detected name appears near this text
    for (const name of detectedNames) {
      if (characterNames.has(name.toLowerCase())) {
        potentialSpeaker = name;
        break;
      }
    }

    return {
      segment_id: seg.id,
      text_preview:
        seg.text_content.length > 80
          ? seg.text_content.slice(0, 80) + "..."
          : seg.text_content,
      potential_speaker: potentialSpeaker,
    };
  });

  return {
    total_segments: segments.length,
    unassigned_segments: unassigned.length,
    detected_names: detectedNames,
    suggestions,
  };
}

/**
 * Get a summary of character line distribution across the project
 */
export function getLineDistribution(): {
  total_segments: number;
  assigned_segments: number;
  unassigned_segments: number;
  by_character: Array<{
    character_id: string;
    character_name: string;
    is_narrator: boolean;
    segment_count: number;
    total_characters: number;
    has_voice: boolean;
  }>;
} {
  const db = getDatabase();

  // Total counts
  const totalResult = db
    .prepare(`SELECT COUNT(*) as total FROM segments`)
    .get() as { total: number };
  const assignedResult = db
    .prepare(`SELECT COUNT(*) as assigned FROM segments WHERE character_id IS NOT NULL`)
    .get() as { assigned: number };

  // By character breakdown
  const stmt = db.prepare(`
    SELECT
      c.id as character_id,
      c.name as character_name,
      c.is_narrator,
      c.voice_config,
      COUNT(s.id) as segment_count,
      COALESCE(SUM(LENGTH(s.text_content)), 0) as total_characters
    FROM characters c
    LEFT JOIN segments s ON s.character_id = c.id
    GROUP BY c.id
    ORDER BY segment_count DESC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  const byCharacter = rows.map((row) => ({
    character_id: row.character_id as string,
    character_name: row.character_name as string,
    is_narrator: Boolean(row.is_narrator),
    segment_count: row.segment_count as number,
    total_characters: row.total_characters as number,
    has_voice: row.voice_config !== null,
  }));

  return {
    total_segments: totalResult.total,
    assigned_segments: assignedResult.assigned,
    unassigned_segments: totalResult.total - assignedResult.assigned,
    by_character: byCharacter,
  };
}
