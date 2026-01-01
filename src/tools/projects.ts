import { v4 as uuidv4 } from "uuid";
import {
  openDatabase,
  initProjectDirectory,
  isAudiobookProject,
  getDatabase,
  getCurrentProjectPath,
  getAudiobookDir,
} from "../db/connection.js";
import { initializeSchema, type Project } from "../db/schema.js";

export interface InitProjectParams {
  path: string;
  title: string;
  author?: string;
  description?: string;
}

export interface OpenProjectParams {
  path: string;
}

/**
 * Initialize a new audiobook project in a directory
 */
export function initProject(params: InitProjectParams): Project {
  const { path, title, author, description } = params;

  // Create directory structure
  initProjectDirectory(path);

  // Open database and create schema
  const db = openDatabase(path);
  initializeSchema(db);

  // Create project record
  const id = uuidv4();
  const now = new Date().toISOString();

  const stmt = db.prepare(`
    INSERT INTO project (id, title, author, description, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `);

  stmt.run(id, title, author ?? null, description ?? null, now, now);

  return {
    id,
    title,
    author: author ?? null,
    description: description ?? null,
    created_at: now,
    updated_at: now,
  };
}

/**
 * Open an existing audiobook project
 */
export function openProject(params: OpenProjectParams): Project {
  const { path } = params;

  if (!isAudiobookProject(path)) {
    throw new Error(`Not an audiobook project: ${path}. Run init_project first.`);
  }

  const db = openDatabase(path);

  const stmt = db.prepare(`SELECT * FROM project LIMIT 1`);
  const row = stmt.get() as Record<string, unknown> | undefined;

  if (!row) {
    throw new Error(`Project database is corrupted: no project record found`);
  }

  return {
    id: row.id as string,
    title: row.title as string,
    author: row.author as string | null,
    description: row.description as string | null,
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
  };
}

/**
 * Get information about the currently open project
 */
export function getProjectInfo(): {
  project: Project;
  path: string;
  stats: {
    character_count: number;
    chapter_count: number;
    segment_count: number;
    segments_with_audio: number;
    total_duration_ms: number;
  };
} {
  const projectPath = getCurrentProjectPath();
  if (!projectPath) {
    throw new Error("No project is currently open. Use open_project first.");
  }

  const db = getDatabase();

  // Get project
  const projectStmt = db.prepare(`SELECT * FROM project LIMIT 1`);
  const projectRow = projectStmt.get() as Record<string, unknown> | undefined;

  if (!projectRow) {
    throw new Error("Project database is corrupted: no project record found");
  }

  const project: Project = {
    id: projectRow.id as string,
    title: projectRow.title as string,
    author: projectRow.author as string | null,
    description: projectRow.description as string | null,
    created_at: projectRow.created_at as string,
    updated_at: projectRow.updated_at as string,
  };

  // Get stats
  const characterCount = db.prepare(`SELECT COUNT(*) as count FROM characters`).get() as { count: number };
  const chapterCount = db.prepare(`SELECT COUNT(*) as count FROM chapters`).get() as { count: number };
  const segmentCount = db.prepare(`SELECT COUNT(*) as count FROM segments`).get() as { count: number };
  const segmentsWithAudio = db.prepare(`SELECT COUNT(*) as count FROM segments WHERE audio_path IS NOT NULL`).get() as { count: number };
  const totalDuration = db.prepare(`SELECT COALESCE(SUM(duration_ms), 0) as total FROM segments WHERE duration_ms IS NOT NULL`).get() as { total: number };

  return {
    project,
    path: projectPath,
    stats: {
      character_count: characterCount.count,
      chapter_count: chapterCount.count,
      segment_count: segmentCount.count,
      segments_with_audio: segmentsWithAudio.count,
      total_duration_ms: totalDuration.total,
    },
  };
}

/**
 * Update project metadata
 */
export function updateProject(params: {
  title?: string;
  author?: string;
  description?: string;
}): Project {
  const db = getDatabase();
  const now = new Date().toISOString();

  // Build update query dynamically based on provided fields
  const updates: string[] = ["updated_at = ?"];
  const values: (string | null)[] = [now];

  if (params.title !== undefined) {
    updates.push("title = ?");
    values.push(params.title);
  }
  if (params.author !== undefined) {
    updates.push("author = ?");
    values.push(params.author);
  }
  if (params.description !== undefined) {
    updates.push("description = ?");
    values.push(params.description);
  }

  const stmt = db.prepare(`UPDATE project SET ${updates.join(", ")}`);
  stmt.run(...values);

  // Return updated project
  const projectStmt = db.prepare(`SELECT * FROM project LIMIT 1`);
  const row = projectStmt.get() as Record<string, unknown>;

  return {
    id: row.id as string,
    title: row.title as string,
    author: row.author as string | null,
    description: row.description as string | null,
    created_at: row.created_at as string,
    updated_at: row.updated_at as string,
  };
}
