"""Project management tools."""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..db.connection import (
    open_database,
    init_project_directory,
    is_audiobook_project,
    get_database,
    get_current_project_path,
)
from ..db.schema import initialize_schema, Project


@dataclass
class ProjectStats:
    character_count: int
    chapter_count: int
    segment_count: int
    segments_with_audio: int
    total_duration_ms: int


@dataclass
class ProjectInfo:
    project: Project
    path: str
    stats: ProjectStats


def init_project(
    path: str,
    title: str,
    author: Optional[str] = None,
    description: Optional[str] = None,
) -> Project:
    """Initialize a new audiobook project in a directory."""
    # Create directory structure
    init_project_directory(path)

    # Open database and create schema
    db = open_database(path)
    initialize_schema(db)

    # Create project record
    project_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    cursor = db.cursor()
    cursor.execute(
        """
        INSERT INTO project (id, title, author, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (project_id, title, author, description, now, now),
    )
    db.commit()

    return Project(
        id=project_id,
        title=title,
        author=author,
        description=description,
        created_at=now,
        updated_at=now,
    )


def open_project(path: str) -> Project:
    """Open an existing audiobook project."""
    if not is_audiobook_project(path):
        raise ValueError(f"Not an audiobook project: {path}. Run init_project first.")

    db = open_database(path)
    cursor = db.cursor()
    cursor.execute("SELECT * FROM project LIMIT 1")
    row = cursor.fetchone()

    if not row:
        raise ValueError("Project database is corrupted: no project record found")

    return Project(
        id=row["id"],
        title=row["title"],
        author=row["author"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def get_project_info() -> ProjectInfo:
    """Get information about the currently open project."""
    project_path = get_current_project_path()
    if not project_path:
        raise ValueError("No project is currently open. Use open_project first.")

    db = get_database()
    cursor = db.cursor()

    # Get project
    cursor.execute("SELECT * FROM project LIMIT 1")
    row = cursor.fetchone()

    if not row:
        raise ValueError("Project database is corrupted: no project record found")

    project = Project(
        id=row["id"],
        title=row["title"],
        author=row["author"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )

    # Get stats
    cursor.execute("SELECT COUNT(*) as count FROM characters")
    character_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM chapters")
    chapter_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM segments")
    segment_count = cursor.fetchone()["count"]

    cursor.execute("SELECT COUNT(*) as count FROM segments WHERE audio_path IS NOT NULL")
    segments_with_audio = cursor.fetchone()["count"]

    cursor.execute(
        "SELECT COALESCE(SUM(duration_ms), 0) as total FROM segments WHERE duration_ms IS NOT NULL"
    )
    total_duration = cursor.fetchone()["total"]

    return ProjectInfo(
        project=project,
        path=project_path,
        stats=ProjectStats(
            character_count=character_count,
            chapter_count=chapter_count,
            segment_count=segment_count,
            segments_with_audio=segments_with_audio,
            total_duration_ms=total_duration,
        ),
    )


def update_project(
    title: Optional[str] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
) -> Project:
    """Update project metadata."""
    db = get_database()
    now = datetime.utcnow().isoformat() + "Z"

    # Build update query dynamically based on provided fields
    updates = ["updated_at = ?"]
    values: list = [now]

    if title is not None:
        updates.append("title = ?")
        values.append(title)
    if author is not None:
        updates.append("author = ?")
        values.append(author)
    if description is not None:
        updates.append("description = ?")
        values.append(description)

    cursor = db.cursor()
    cursor.execute(f"UPDATE project SET {', '.join(updates)}", values)
    db.commit()

    # Return updated project
    cursor.execute("SELECT * FROM project LIMIT 1")
    row = cursor.fetchone()

    return Project(
        id=row["id"],
        title=row["title"],
        author=row["author"],
        description=row["description"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
