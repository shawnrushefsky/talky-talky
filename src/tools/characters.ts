import { v4 as uuidv4 } from "uuid";
import { getDatabase } from "../db/connection.js";
import type { Character, VoiceConfig } from "../db/schema.js";

export interface AddCharacterParams {
  name: string;
  description?: string;
  is_narrator?: boolean;
}

export interface UpdateCharacterParams {
  id: string;
  name?: string;
  description?: string;
  is_narrator?: boolean;
}

export interface SetVoiceParams {
  character_id: string;
  provider: string;
  voice_id: string;
  settings?: Record<string, unknown>;
}

/**
 * Add a new character to the project
 */
export function addCharacter(params: AddCharacterParams): Character {
  const db = getDatabase();
  const { name, description, is_narrator = false } = params;

  // Check for duplicate name
  const existing = db
    .prepare(`SELECT id FROM characters WHERE name = ?`)
    .get(name);
  if (existing) {
    throw new Error(`Character with name "${name}" already exists`);
  }

  const id = uuidv4();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO characters (id, name, description, voice_config, is_narrator, created_at)
    VALUES (?, ?, ?, NULL, ?, ?)
  `);

  stmt.run(id, name, description ?? null, is_narrator ? 1 : 0, now);

  return {
    id,
    name,
    description: description ?? null,
    voice_config: null,
    is_narrator,
    created_at: now,
  };
}

/**
 * List all characters in the project
 */
export function listCharacters(): Character[] {
  const db = getDatabase();

  const stmt = db.prepare(`
    SELECT * FROM characters ORDER BY is_narrator DESC, name ASC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    name: row.name as string,
    description: row.description as string | null,
    voice_config: row.voice_config as string | null,
    is_narrator: Boolean(row.is_narrator),
    created_at: row.created_at as string,
  }));
}

/**
 * Get a character by ID
 */
export function getCharacter(id: string): Character | null {
  const db = getDatabase();

  const stmt = db.prepare(`SELECT * FROM characters WHERE id = ?`);
  const row = stmt.get(id) as Record<string, unknown> | undefined;

  if (!row) return null;

  return {
    id: row.id as string,
    name: row.name as string,
    description: row.description as string | null,
    voice_config: row.voice_config as string | null,
    is_narrator: Boolean(row.is_narrator),
    created_at: row.created_at as string,
  };
}

/**
 * Update a character
 */
export function updateCharacter(params: UpdateCharacterParams): Character {
  const db = getDatabase();
  const { id, name, description, is_narrator } = params;

  // Verify character exists
  const existing = getCharacter(id);
  if (!existing) {
    throw new Error(`Character not found: ${id}`);
  }

  // Check for name conflict if renaming
  if (name && name !== existing.name) {
    const nameConflict = db
      .prepare(`SELECT id FROM characters WHERE name = ? AND id != ?`)
      .get(name, id);
    if (nameConflict) {
      throw new Error(`Character with name "${name}" already exists`);
    }
  }

  // Build update query
  const updates: string[] = [];
  const values: (string | number | null)[] = [];

  if (name !== undefined) {
    updates.push("name = ?");
    values.push(name);
  }
  if (description !== undefined) {
    updates.push("description = ?");
    values.push(description);
  }
  if (is_narrator !== undefined) {
    updates.push("is_narrator = ?");
    values.push(is_narrator ? 1 : 0);
  }

  if (updates.length === 0) {
    return existing;
  }

  values.push(id);
  const stmt = db.prepare(
    `UPDATE characters SET ${updates.join(", ")} WHERE id = ?`
  );
  stmt.run(...values);

  return getCharacter(id)!;
}

/**
 * Delete a character
 */
export function deleteCharacter(id: string): void {
  const db = getDatabase();

  // Verify character exists
  const existing = getCharacter(id);
  if (!existing) {
    throw new Error(`Character not found: ${id}`);
  }

  const stmt = db.prepare(`DELETE FROM characters WHERE id = ?`);
  stmt.run(id);
}

/**
 * Set voice configuration for a character
 */
export function setVoice(params: SetVoiceParams): Character {
  const db = getDatabase();
  const { character_id, provider, voice_id, settings } = params;

  // Verify character exists
  const existing = getCharacter(character_id);
  if (!existing) {
    throw new Error(`Character not found: ${character_id}`);
  }

  const voiceConfig: VoiceConfig = {
    provider,
    voice_id,
    settings,
  };

  const stmt = db.prepare(`UPDATE characters SET voice_config = ? WHERE id = ?`);
  stmt.run(JSON.stringify(voiceConfig), character_id);

  return getCharacter(character_id)!;
}

/**
 * Clear voice configuration for a character
 */
export function clearVoice(character_id: string): Character {
  const db = getDatabase();

  // Verify character exists
  const existing = getCharacter(character_id);
  if (!existing) {
    throw new Error(`Character not found: ${character_id}`);
  }

  const stmt = db.prepare(`UPDATE characters SET voice_config = NULL WHERE id = ?`);
  stmt.run(character_id);

  return getCharacter(character_id)!;
}

/**
 * Get characters with their line counts
 */
export function getCharactersWithStats(): Array<
  Character & { segment_count: number; total_text_length: number }
> {
  const db = getDatabase();

  const stmt = db.prepare(`
    SELECT
      c.*,
      COUNT(s.id) as segment_count,
      COALESCE(SUM(LENGTH(s.text_content)), 0) as total_text_length
    FROM characters c
    LEFT JOIN segments s ON s.character_id = c.id
    GROUP BY c.id
    ORDER BY c.is_narrator DESC, c.name ASC
  `);

  const rows = stmt.all() as Array<Record<string, unknown>>;

  return rows.map((row) => ({
    id: row.id as string,
    name: row.name as string,
    description: row.description as string | null,
    voice_config: row.voice_config as string | null,
    is_narrator: Boolean(row.is_narrator),
    created_at: row.created_at as string,
    segment_count: row.segment_count as number,
    total_text_length: row.total_text_length as number,
  }));
}
