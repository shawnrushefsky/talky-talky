"""Database connection management for audiobook projects."""

import sqlite3
from pathlib import Path
from typing import Optional

AUDIOBOOK_DIR = ".audiobook"
DB_FILENAME = "db.sqlite"

_current_db: Optional[sqlite3.Connection] = None
_current_project_path: Optional[str] = None


def get_audiobook_dir(project_path: str) -> Path:
    """Get the .audiobook directory path for a project."""
    return Path(project_path) / AUDIOBOOK_DIR


def get_db_path(project_path: str) -> Path:
    """Get the database path for a project."""
    return get_audiobook_dir(project_path) / DB_FILENAME


def is_audiobook_project(project_path: str) -> bool:
    """Check if a directory is an initialized audiobook project."""
    return get_db_path(project_path).exists()


def init_project_directory(project_path: str) -> None:
    """Initialize the .audiobook directory structure for a new project."""
    project = Path(project_path)
    audiobook_dir = get_audiobook_dir(project_path)

    if not project.exists():
        raise ValueError(f"Project directory does not exist: {project_path}")

    if audiobook_dir.exists():
        raise ValueError(f"Project already initialized: {audiobook_dir}")

    # Create directory structure
    audiobook_dir.mkdir(parents=True, exist_ok=True)
    (audiobook_dir / "audio" / "segments").mkdir(parents=True, exist_ok=True)
    (audiobook_dir / "audio" / "voice_samples").mkdir(parents=True, exist_ok=True)
    (audiobook_dir / "exports" / "chapters").mkdir(parents=True, exist_ok=True)
    (audiobook_dir / "exports" / "book").mkdir(parents=True, exist_ok=True)


def open_database(project_path: str) -> sqlite3.Connection:
    """Open (or create) the database for a project."""
    global _current_db, _current_project_path

    db_path = get_db_path(project_path)

    # Close existing connection if switching projects
    if _current_db and _current_project_path != project_path:
        close_database()

    if not _current_db:
        if not get_audiobook_dir(project_path).exists():
            raise ValueError(f"Not an audiobook project. Run init_project first: {project_path}")

        _current_db = sqlite3.connect(str(db_path))
        _current_db.row_factory = sqlite3.Row  # Enable dict-like access
        _current_db.execute("PRAGMA journal_mode = WAL")
        _current_db.execute("PRAGMA foreign_keys = ON")
        _current_project_path = project_path

    return _current_db


def get_database() -> sqlite3.Connection:
    """Get the currently open database."""
    if not _current_db:
        raise ValueError("No project is currently open. Use open_project first.")
    return _current_db


def get_current_project_path() -> Optional[str]:
    """Get the current project path."""
    return _current_project_path


def close_database() -> None:
    """Close the current database connection."""
    global _current_db, _current_project_path

    if _current_db:
        _current_db.close()
        _current_db = None
        _current_project_path = None
