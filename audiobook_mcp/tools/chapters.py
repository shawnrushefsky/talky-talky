"""Chapter management tools."""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..db.connection import get_database
from ..db.schema import Chapter


@dataclass
class ChapterWithStats:
    id: str
    title: str
    sort_order: int
    created_at: str
    segment_count: int
    segments_with_audio: int
    total_duration_ms: int


def add_chapter(title: str, sort_order: Optional[int] = None) -> Chapter:
    """Add a new chapter to the project."""
    db = get_database()
    cursor = db.cursor()

    # Get the next sort order if not specified
    if sort_order is None:
        cursor.execute("SELECT COALESCE(MAX(sort_order), -1) as max_order FROM chapters")
        sort_order = cursor.fetchone()["max_order"] + 1

    chapter_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    cursor.execute(
        """
        INSERT INTO chapters (id, title, sort_order, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (chapter_id, title, sort_order, now),
    )
    db.commit()

    return Chapter(
        id=chapter_id,
        title=title,
        sort_order=sort_order,
        created_at=now,
    )


def list_chapters() -> list[Chapter]:
    """List all chapters in order."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM chapters ORDER BY sort_order ASC")
    rows = cursor.fetchall()

    return [
        Chapter(
            id=row["id"],
            title=row["title"],
            sort_order=row["sort_order"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_chapter(chapter_id: str) -> Optional[Chapter]:
    """Get a chapter by ID."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,))
    row = cursor.fetchone()

    if not row:
        return None

    return Chapter(
        id=row["id"],
        title=row["title"],
        sort_order=row["sort_order"],
        created_at=row["created_at"],
    )


def update_chapter(chapter_id: str, title: Optional[str] = None) -> Chapter:
    """Update a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    existing = get_chapter(chapter_id)
    if not existing:
        raise ValueError(f"Chapter not found: {chapter_id}")

    if title is not None:
        cursor.execute("UPDATE chapters SET title = ? WHERE id = ?", (title, chapter_id))
        db.commit()

    return get_chapter(chapter_id)  # type: ignore


def delete_chapter(chapter_id: str) -> None:
    """Delete a chapter and all its segments."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    existing = get_chapter(chapter_id)
    if not existing:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Segments will be deleted via ON DELETE CASCADE
    cursor.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))
    db.commit()


def reorder_chapters(chapter_ids: list[str]) -> list[Chapter]:
    """Reorder chapters."""
    db = get_database()
    cursor = db.cursor()

    # Verify all chapter IDs exist
    existing_chapters = list_chapters()
    existing_ids = {c.id for c in existing_chapters}

    for cid in chapter_ids:
        if cid not in existing_ids:
            raise ValueError(f"Chapter not found: {cid}")

    # Check for missing chapters
    if len(chapter_ids) != len(existing_chapters):
        missing_ids = [c.id for c in existing_chapters if c.id not in chapter_ids]
        raise ValueError(f"Missing chapters in reorder list: {', '.join(missing_ids)}")

    # Update sort orders
    for i, chapter_id in enumerate(chapter_ids):
        cursor.execute("UPDATE chapters SET sort_order = ? WHERE id = ?", (i, chapter_id))

    db.commit()
    return list_chapters()


def get_chapters_with_stats() -> list[ChapterWithStats]:
    """Get chapters with segment statistics."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("""
        SELECT
            c.*,
            COUNT(s.id) as segment_count,
            SUM(CASE WHEN s.audio_path IS NOT NULL THEN 1 ELSE 0 END) as segments_with_audio,
            COALESCE(SUM(s.duration_ms), 0) as total_duration_ms
        FROM chapters c
        LEFT JOIN segments s ON s.chapter_id = c.id
        GROUP BY c.id
        ORDER BY c.sort_order ASC
    """)
    rows = cursor.fetchall()

    return [
        ChapterWithStats(
            id=row["id"],
            title=row["title"],
            sort_order=row["sort_order"],
            created_at=row["created_at"],
            segment_count=row["segment_count"],
            segments_with_audio=row["segments_with_audio"] or 0,
            total_duration_ms=row["total_duration_ms"] or 0,
        )
        for row in rows
    ]
