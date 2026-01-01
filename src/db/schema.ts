import type Database from "better-sqlite3";

/**
 * Initialize the database schema for an audiobook project
 */
export function initializeSchema(db: Database.Database): void {
  // Project metadata table (single row)
  db.exec(`
    CREATE TABLE IF NOT EXISTS project (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      author TEXT,
      description TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `);

  // Characters table
  db.exec(`
    CREATE TABLE IF NOT EXISTS characters (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL UNIQUE,
      description TEXT,
      voice_config TEXT,
      is_narrator INTEGER NOT NULL DEFAULT 0,
      created_at TEXT NOT NULL
    )
  `);

  // Chapters table
  db.exec(`
    CREATE TABLE IF NOT EXISTS chapters (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      sort_order INTEGER NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Create index for chapter ordering
  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_chapters_sort_order ON chapters(sort_order)
  `);

  // Segments table
  db.exec(`
    CREATE TABLE IF NOT EXISTS segments (
      id TEXT PRIMARY KEY,
      chapter_id TEXT NOT NULL,
      character_id TEXT,
      text_content TEXT NOT NULL,
      sort_order INTEGER NOT NULL,
      audio_path TEXT,
      duration_ms INTEGER,
      created_at TEXT NOT NULL,
      FOREIGN KEY (chapter_id) REFERENCES chapters(id) ON DELETE CASCADE,
      FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE SET NULL
    )
  `);

  // Create indexes for segment queries
  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_segments_chapter ON segments(chapter_id);
    CREATE INDEX IF NOT EXISTS idx_segments_sort ON segments(chapter_id, sort_order);
    CREATE INDEX IF NOT EXISTS idx_segments_character ON segments(character_id)
  `);
}

/**
 * TypeScript types matching the database schema
 */
export interface Project {
  id: string;
  title: string;
  author: string | null;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface Character {
  id: string;
  name: string;
  description: string | null;
  voice_config: string | null; // JSON string
  is_narrator: boolean;
  created_at: string;
}

export interface Chapter {
  id: string;
  title: string;
  sort_order: number;
  created_at: string;
}

export interface Segment {
  id: string;
  chapter_id: string;
  character_id: string | null;
  text_content: string;
  sort_order: number;
  audio_path: string | null;
  duration_ms: number | null;
  created_at: string;
}

/**
 * Voice configuration structure (stored as JSON in voice_config)
 */
export interface VoiceConfig {
  provider: string; // e.g., "elevenlabs", "openai", etc.
  voice_id: string;
  settings?: Record<string, unknown>;
}
