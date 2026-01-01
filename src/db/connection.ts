import Database from "better-sqlite3";
import { existsSync, mkdirSync } from "fs";
import { join } from "path";

const AUDIOBOOK_DIR = ".audiobook";
const DB_FILENAME = "db.sqlite";

let currentDb: Database.Database | null = null;
let currentProjectPath: string | null = null;

/**
 * Get the .audiobook directory path for a project
 */
export function getAudiobookDir(projectPath: string): string {
  return join(projectPath, AUDIOBOOK_DIR);
}

/**
 * Get the database path for a project
 */
export function getDbPath(projectPath: string): string {
  return join(getAudiobookDir(projectPath), DB_FILENAME);
}

/**
 * Check if a directory is an initialized audiobook project
 */
export function isAudiobookProject(projectPath: string): boolean {
  return existsSync(getDbPath(projectPath));
}

/**
 * Initialize the .audiobook directory structure for a new project
 */
export function initProjectDirectory(projectPath: string): void {
  const audiobookDir = getAudiobookDir(projectPath);

  if (!existsSync(projectPath)) {
    throw new Error(`Project directory does not exist: ${projectPath}`);
  }

  if (existsSync(audiobookDir)) {
    throw new Error(`Project already initialized: ${audiobookDir}`);
  }

  // Create directory structure
  mkdirSync(audiobookDir, { recursive: true });
  mkdirSync(join(audiobookDir, "audio", "segments"), { recursive: true });
  mkdirSync(join(audiobookDir, "exports", "chapters"), { recursive: true });
  mkdirSync(join(audiobookDir, "exports", "book"), { recursive: true });
}

/**
 * Open (or create) the database for a project
 */
export function openDatabase(projectPath: string): Database.Database {
  const dbPath = getDbPath(projectPath);

  // Close existing connection if switching projects
  if (currentDb && currentProjectPath !== projectPath) {
    closeDatabase();
  }

  if (!currentDb) {
    if (!existsSync(getAudiobookDir(projectPath))) {
      throw new Error(`Not an audiobook project. Run init_project first: ${projectPath}`);
    }

    currentDb = new Database(dbPath);
    currentDb.pragma("journal_mode = WAL");
    currentDb.pragma("foreign_keys = ON");
    currentProjectPath = projectPath;
  }

  return currentDb;
}

/**
 * Get the currently open database
 */
export function getDatabase(): Database.Database {
  if (!currentDb) {
    throw new Error("No project is currently open. Use open_project first.");
  }
  return currentDb;
}

/**
 * Get the current project path
 */
export function getCurrentProjectPath(): string | null {
  return currentProjectPath;
}

/**
 * Close the current database connection
 */
export function closeDatabase(): void {
  if (currentDb) {
    currentDb.close();
    currentDb = null;
    currentProjectPath = null;
  }
}
