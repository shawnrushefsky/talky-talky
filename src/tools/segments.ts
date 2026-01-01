import { v4 as uuidv4 } from "uuid";
import { getDatabase } from "../db/connection.js";
import type { Segment } from "../db/schema.js";
import { getChapter } from "./chapters.js";
import { getCharacter } from "./characters.js";

export interface AddSegmentParams {
  chapter_id: string;
  text_content: string;
  character_id?: string;
  sort_order?: number;
}

export interface UpdateSegmentParams {
  id: string;
  text_content?: string;
  character_id?: string | null;
}

/**
 * Add a new segment to a chapter
 */
export function addSegment(params: AddSegmentParams): Segment {
  const db = getDatabase();
  const { chapter_id, text_content, character_id, sort_order } = params;

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  // Verify character exists if provided
  if (character_id) {
    const character = getCharacter(character_id);
    if (!character) {
      throw new Error(`Character not found: ${character_id}`);
    }
  }

  // Get the next sort order if not specified
  let order = sort_order;
  if (order === undefined) {
    const maxOrder = db
      .prepare(
        `SELECT COALESCE(MAX(sort_order), -1) as max_order FROM segments WHERE chapter_id = ?`
      )
      .get(chapter_id) as { max_order: number };
    order = maxOrder.max_order + 1;
  }

  const id = uuidv4();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO segments (id, chapter_id, character_id, text_content, sort_order, audio_path, duration_ms, created_at)
    VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
  `);

  stmt.run(id, chapter_id, character_id ?? null, text_content, order, now);

  return {
    id,
    chapter_id,
    character_id: character_id ?? null,
    text_content,
    sort_order: order,
    audio_path: null,
    duration_ms: null,
    created_at: now,
  };
}

/**
 * List all segments in a chapter
 */
export function listSegments(chapter_id: string): Segment[] {
  const db = getDatabase();

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  const stmt = db.prepare(`
    SELECT * FROM segments WHERE chapter_id = ? ORDER BY sort_order ASC
  `);

  const rows = stmt.all(chapter_id) as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    chapter_id: row.chapter_id as string,
    character_id: row.character_id as string | null,
    text_content: row.text_content as string,
    sort_order: row.sort_order as number,
    audio_path: row.audio_path as string | null,
    duration_ms: row.duration_ms as number | null,
    created_at: row.created_at as string,
  }));
}

/**
 * Get a segment by ID
 */
export function getSegment(id: string): Segment | null {
  const db = getDatabase();

  const stmt = db.prepare(`SELECT * FROM segments WHERE id = ?`);
  const row = stmt.get(id) as Record<string, unknown> | undefined;

  if (!row) return null;

  return {
    id: row.id as string,
    chapter_id: row.chapter_id as string,
    character_id: row.character_id as string | null,
    text_content: row.text_content as string,
    sort_order: row.sort_order as number,
    audio_path: row.audio_path as string | null,
    duration_ms: row.duration_ms as number | null,
    created_at: row.created_at as string,
  };
}

/**
 * Update a segment
 */
export function updateSegment(params: UpdateSegmentParams): Segment {
  const db = getDatabase();
  const { id, text_content, character_id } = params;

  // Verify segment exists
  const existing = getSegment(id);
  if (!existing) {
    throw new Error(`Segment not found: ${id}`);
  }

  // Verify character exists if provided (and not null)
  if (character_id !== undefined && character_id !== null) {
    const character = getCharacter(character_id);
    if (!character) {
      throw new Error(`Character not found: ${character_id}`);
    }
  }

  // Build update query
  const updates: string[] = [];
  const values: (string | null)[] = [];

  if (text_content !== undefined) {
    updates.push("text_content = ?");
    values.push(text_content);
  }
  if (character_id !== undefined) {
    updates.push("character_id = ?");
    values.push(character_id);
  }

  if (updates.length === 0) {
    return existing;
  }

  values.push(id);
  const stmt = db.prepare(
    `UPDATE segments SET ${updates.join(", ")} WHERE id = ?`
  );
  stmt.run(...values);

  return getSegment(id)!;
}

/**
 * Delete a segment
 */
export function deleteSegment(id: string): void {
  const db = getDatabase();

  // Verify segment exists
  const existing = getSegment(id);
  if (!existing) {
    throw new Error(`Segment not found: ${id}`);
  }

  const stmt = db.prepare(`DELETE FROM segments WHERE id = ?`);
  stmt.run(id);
}

/**
 * Reorder segments within a chapter
 */
export function reorderSegments(chapter_id: string, segmentIds: string[]): Segment[] {
  const db = getDatabase();

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  // Verify all segment IDs exist and belong to this chapter
  const existingSegments = listSegments(chapter_id);
  const existingIds = new Set(existingSegments.map((s) => s.id));

  for (const id of segmentIds) {
    if (!existingIds.has(id)) {
      throw new Error(`Segment not found in chapter: ${id}`);
    }
  }

  // Check for missing segments
  if (segmentIds.length !== existingSegments.length) {
    const missingIds = existingSegments
      .filter((s) => !segmentIds.includes(s.id))
      .map((s) => s.id);
    throw new Error(`Missing segments in reorder list: ${missingIds.join(", ")}`);
  }

  // Update sort orders
  const stmt = db.prepare(`UPDATE segments SET sort_order = ? WHERE id = ?`);

  const updateMany = db.transaction(() => {
    for (let i = 0; i < segmentIds.length; i++) {
      stmt.run(i, segmentIds[i]);
    }
  });

  updateMany();

  return listSegments(chapter_id);
}

/**
 * Get segments with character information
 */
export function getSegmentsWithCharacters(
  chapter_id: string
): Array<Segment & { character_name: string | null }> {
  const db = getDatabase();

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  const stmt = db.prepare(`
    SELECT s.*, c.name as character_name
    FROM segments s
    LEFT JOIN characters c ON s.character_id = c.id
    WHERE s.chapter_id = ?
    ORDER BY s.sort_order ASC
  `);

  const rows = stmt.all(chapter_id) as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    chapter_id: row.chapter_id as string,
    character_id: row.character_id as string | null,
    text_content: row.text_content as string,
    sort_order: row.sort_order as number,
    audio_path: row.audio_path as string | null,
    duration_ms: row.duration_ms as number | null,
    created_at: row.created_at as string,
    character_name: row.character_name as string | null,
  }));
}

/**
 * Get all segments that are missing audio
 */
export function getPendingSegments(): Array<
  Segment & { chapter_title: string; character_name: string | null }
> {
  const db = getDatabase();

  const stmt = db.prepare(`
    SELECT s.*, ch.title as chapter_title, c.name as character_name
    FROM segments s
    JOIN chapters ch ON s.chapter_id = ch.id
    LEFT JOIN characters c ON s.character_id = c.id
    WHERE s.audio_path IS NULL
    ORDER BY ch.sort_order ASC, s.sort_order ASC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    chapter_id: row.chapter_id as string,
    character_id: row.character_id as string | null,
    text_content: row.text_content as string,
    sort_order: row.sort_order as number,
    audio_path: row.audio_path as string | null,
    duration_ms: row.duration_ms as number | null,
    created_at: row.created_at as string,
    chapter_title: row.chapter_title as string,
    character_name: row.character_name as string | null,
  }));
}

/**
 * Bulk add segments to a chapter
 */
export function bulkAddSegments(
  chapter_id: string,
  segments: Array<{ text_content: string; character_id?: string }>
): Segment[] {
  const db = getDatabase();

  // Verify chapter exists
  const chapter = getChapter(chapter_id);
  if (!chapter) {
    throw new Error(`Chapter not found: ${chapter_id}`);
  }

  // Get starting sort order
  const maxOrder = db
    .prepare(
      `SELECT COALESCE(MAX(sort_order), -1) as max_order FROM segments WHERE chapter_id = ?`
    )
    .get(chapter_id) as { max_order: number };

  let order = maxOrder.max_order + 1;
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO segments (id, chapter_id, character_id, text_content, sort_order, audio_path, duration_ms, created_at)
    VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
  `);

  const results: Segment[] = [];

  const insertMany = db.transaction(() => {
    for (const seg of segments) {
      const id = uuidv4();
      stmt.run(id, chapter_id, seg.character_id ?? null, seg.text_content, order, now);
      results.push({
        id,
        chapter_id,
        character_id: seg.character_id ?? null,
        text_content: seg.text_content,
        sort_order: order,
        audio_path: null,
        duration_ms: null,
        created_at: now,
      });
      order++;
    }
  });

  insertMany();

  return results;
}
