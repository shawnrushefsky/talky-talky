import { v4 as uuidv4 } from "uuid";
import { getDatabase } from "../db/connection.js";
import type { Chapter } from "../db/schema.js";

export interface AddChapterParams {
  title: string;
  sort_order?: number;
}

export interface UpdateChapterParams {
  id: string;
  title?: string;
}

/**
 * Add a new chapter to the project
 */
export function addChapter(params: AddChapterParams): Chapter {
  const db = getDatabase();
  const { title, sort_order } = params;

  // Get the next sort order if not specified
  let order = sort_order;
  if (order === undefined) {
    const maxOrder = db
      .prepare(`SELECT COALESCE(MAX(sort_order), -1) as max_order FROM chapters`)
      .get() as { max_order: number };
    order = maxOrder.max_order + 1;
  }

  const id = uuidv4();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO chapters (id, title, sort_order, created_at)
    VALUES (?, ?, ?, ?)
  `);

  stmt.run(id, title, order, now);

  return {
    id,
    title,
    sort_order: order,
    created_at: now,
  };
}

/**
 * List all chapters in order
 */
export function listChapters(): Chapter[] {
  const db = getDatabase();

  const stmt = db.prepare(`
    SELECT * FROM chapters ORDER BY sort_order ASC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    title: row.title as string,
    sort_order: row.sort_order as number,
    created_at: row.created_at as string,
  }));
}

/**
 * Get a chapter by ID
 */
export function getChapter(id: string): Chapter | null {
  const db = getDatabase();

  const stmt = db.prepare(`SELECT * FROM chapters WHERE id = ?`);
  const row = stmt.get(id) as Record<string, unknown> | undefined;

  if (!row) return null;

  return {
    id: row.id as string,
    title: row.title as string,
    sort_order: row.sort_order as number,
    created_at: row.created_at as string,
  };
}

/**
 * Update a chapter
 */
export function updateChapter(params: UpdateChapterParams): Chapter {
  const db = getDatabase();
  const { id, title } = params;

  // Verify chapter exists
  const existing = getChapter(id);
  if (!existing) {
    throw new Error(`Chapter not found: ${id}`);
  }

  if (title !== undefined) {
    const stmt = db.prepare(`UPDATE chapters SET title = ? WHERE id = ?`);
    stmt.run(title, id);
  }

  return getChapter(id)!;
}

/**
 * Delete a chapter and all its segments
 */
export function deleteChapter(id: string): void {
  const db = getDatabase();

  // Verify chapter exists
  const existing = getChapter(id);
  if (!existing) {
    throw new Error(`Chapter not found: ${id}`);
  }

  // Segments will be deleted via ON DELETE CASCADE
  const stmt = db.prepare(`DELETE FROM chapters WHERE id = ?`);
  stmt.run(id);
}

/**
 * Reorder chapters
 */
export function reorderChapters(chapterIds: string[]): Chapter[] {
  const db = getDatabase();

  // Verify all chapter IDs exist
  const existingChapters = listChapters();
  const existingIds = new Set(existingChapters.map((c) => c.id));

  for (const id of chapterIds) {
    if (!existingIds.has(id)) {
      throw new Error(`Chapter not found: ${id}`);
    }
  }

  // Check for missing chapters
  if (chapterIds.length !== existingChapters.length) {
    const missingIds = existingChapters
      .filter((c) => !chapterIds.includes(c.id))
      .map((c) => c.id);
    throw new Error(`Missing chapters in reorder list: ${missingIds.join(", ")}`);
  }

  // Update sort orders
  const stmt = db.prepare(`UPDATE chapters SET sort_order = ? WHERE id = ?`);

  const updateMany = db.transaction(() => {
    for (let i = 0; i < chapterIds.length; i++) {
      stmt.run(i, chapterIds[i]);
    }
  });

  updateMany();

  return listChapters();
}

/**
 * Get chapters with segment statistics
 */
export function getChaptersWithStats(): Array<
  Chapter & {
    segment_count: number;
    segments_with_audio: number;
    total_duration_ms: number;
  }
> {
  const db = getDatabase();

  const stmt = db.prepare(`
    SELECT
      c.*,
      COUNT(s.id) as segment_count,
      SUM(CASE WHEN s.audio_path IS NOT NULL THEN 1 ELSE 0 END) as segments_with_audio,
      COALESCE(SUM(s.duration_ms), 0) as total_duration_ms
    FROM chapters c
    LEFT JOIN segments s ON s.chapter_id = c.id
    GROUP BY c.id
    ORDER BY c.sort_order ASC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    title: row.title as string,
    sort_order: row.sort_order as number,
    created_at: row.created_at as string,
    segment_count: row.segment_count as number,
    segments_with_audio: row.segments_with_audio as number,
    total_duration_ms: row.total_duration_ms as number,
  }));
}
