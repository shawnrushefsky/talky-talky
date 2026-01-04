"""Segment management tools."""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..db.connection import get_database
from ..db.schema import Segment
from .chapters import get_chapter
from .characters import get_character


@dataclass
class SegmentWithCharacter:
    id: str
    chapter_id: str
    character_id: Optional[str]
    text_content: str
    sort_order: int
    audio_path: Optional[str]
    duration_ms: Optional[int]
    created_at: str
    character_name: Optional[str]


@dataclass
class PendingSegment:
    id: str
    chapter_id: str
    character_id: Optional[str]
    text_content: str
    sort_order: int
    audio_path: Optional[str]
    duration_ms: Optional[int]
    created_at: str
    chapter_title: str
    character_name: Optional[str]


def add_segment(
    chapter_id: str,
    text_content: str,
    character_id: Optional[str] = None,
    sort_order: Optional[int] = None,
) -> Segment:
    """Add a new segment to a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Verify character exists if provided
    if character_id:
        character = get_character(character_id)
        if not character:
            raise ValueError(f"Character not found: {character_id}")

    # Get the next sort order if not specified
    if sort_order is None:
        cursor.execute(
            "SELECT COALESCE(MAX(sort_order), -1) as max_order FROM segments WHERE chapter_id = ?",
            (chapter_id,),
        )
        sort_order = cursor.fetchone()["max_order"] + 1

    segment_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    cursor.execute(
        """
        INSERT INTO segments (id, chapter_id, character_id, text_content, sort_order, audio_path, duration_ms, created_at)
        VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
        """,
        (segment_id, chapter_id, character_id, text_content, sort_order, now),
    )
    db.commit()

    return Segment(
        id=segment_id,
        chapter_id=chapter_id,
        character_id=character_id,
        text_content=text_content,
        sort_order=sort_order,
        audio_path=None,
        duration_ms=None,
        created_at=now,
    )


def list_segments(chapter_id: str) -> list[Segment]:
    """List all segments in a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    cursor.execute(
        "SELECT * FROM segments WHERE chapter_id = ? ORDER BY sort_order ASC",
        (chapter_id,),
    )
    rows = cursor.fetchall()

    return [
        Segment(
            id=row["id"],
            chapter_id=row["chapter_id"],
            character_id=row["character_id"],
            text_content=row["text_content"],
            sort_order=row["sort_order"],
            audio_path=row["audio_path"],
            duration_ms=row["duration_ms"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_segment(segment_id: str) -> Optional[Segment]:
    """Get a segment by ID."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM segments WHERE id = ?", (segment_id,))
    row = cursor.fetchone()

    if not row:
        return None

    return Segment(
        id=row["id"],
        chapter_id=row["chapter_id"],
        character_id=row["character_id"],
        text_content=row["text_content"],
        sort_order=row["sort_order"],
        audio_path=row["audio_path"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
    )


def update_segment(
    segment_id: str,
    text_content: Optional[str] = None,
    character_id: Optional[str] = None,
) -> Segment:
    """Update a segment."""
    db = get_database()
    cursor = db.cursor()

    # Verify segment exists
    existing = get_segment(segment_id)
    if not existing:
        raise ValueError(f"Segment not found: {segment_id}")

    # Verify character exists if provided (and not clearing)
    if character_id is not None and character_id != "":
        character = get_character(character_id)
        if not character:
            raise ValueError(f"Character not found: {character_id}")

    # Build update query
    updates = []
    values: list = []

    if text_content is not None:
        updates.append("text_content = ?")
        values.append(text_content)
    if character_id is not None:
        updates.append("character_id = ?")
        values.append(character_id if character_id else None)

    if not updates:
        return existing

    values.append(segment_id)
    cursor.execute(
        f"UPDATE segments SET {', '.join(updates)} WHERE id = ?",
        values,
    )
    db.commit()

    return get_segment(segment_id)  # type: ignore


def delete_segment(segment_id: str) -> None:
    """Delete a segment."""
    db = get_database()
    cursor = db.cursor()

    # Verify segment exists
    existing = get_segment(segment_id)
    if not existing:
        raise ValueError(f"Segment not found: {segment_id}")

    cursor.execute("DELETE FROM segments WHERE id = ?", (segment_id,))
    db.commit()


def reorder_segments(chapter_id: str, segment_ids: list[str]) -> list[Segment]:
    """Reorder segments within a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Verify all segment IDs exist and belong to this chapter
    existing_segments = list_segments(chapter_id)
    existing_ids = {s.id for s in existing_segments}

    for sid in segment_ids:
        if sid not in existing_ids:
            raise ValueError(f"Segment not found in chapter: {sid}")

    # Check for missing segments
    if len(segment_ids) != len(existing_segments):
        missing_ids = [s.id for s in existing_segments if s.id not in segment_ids]
        raise ValueError(f"Missing segments in reorder list: {', '.join(missing_ids)}")

    # Update sort orders
    for i, segment_id in enumerate(segment_ids):
        cursor.execute("UPDATE segments SET sort_order = ? WHERE id = ?", (i, segment_id))

    db.commit()
    return list_segments(chapter_id)


def get_segments_with_characters(chapter_id: str) -> list[SegmentWithCharacter]:
    """Get segments with character information."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    cursor.execute(
        """
        SELECT s.*, c.name as character_name
        FROM segments s
        LEFT JOIN characters c ON s.character_id = c.id
        WHERE s.chapter_id = ?
        ORDER BY s.sort_order ASC
        """,
        (chapter_id,),
    )
    rows = cursor.fetchall()

    return [
        SegmentWithCharacter(
            id=row["id"],
            chapter_id=row["chapter_id"],
            character_id=row["character_id"],
            text_content=row["text_content"],
            sort_order=row["sort_order"],
            audio_path=row["audio_path"],
            duration_ms=row["duration_ms"],
            created_at=row["created_at"],
            character_name=row["character_name"],
        )
        for row in rows
    ]


def get_pending_segments() -> list[PendingSegment]:
    """Get all segments that are missing audio."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("""
        SELECT s.*, ch.title as chapter_title, c.name as character_name
        FROM segments s
        JOIN chapters ch ON s.chapter_id = ch.id
        LEFT JOIN characters c ON s.character_id = c.id
        WHERE s.audio_path IS NULL
        ORDER BY ch.sort_order ASC, s.sort_order ASC
    """)
    rows = cursor.fetchall()

    return [
        PendingSegment(
            id=row["id"],
            chapter_id=row["chapter_id"],
            character_id=row["character_id"],
            text_content=row["text_content"],
            sort_order=row["sort_order"],
            audio_path=row["audio_path"],
            duration_ms=row["duration_ms"],
            created_at=row["created_at"],
            chapter_title=row["chapter_title"],
            character_name=row["character_name"],
        )
        for row in rows
    ]


def bulk_add_segments(
    chapter_id: str,
    segments: list[dict],
) -> list[Segment]:
    """Bulk add segments to a chapter."""
    db = get_database()
    cursor = db.cursor()

    # Verify chapter exists
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")

    # Get starting sort order
    cursor.execute(
        "SELECT COALESCE(MAX(sort_order), -1) as max_order FROM segments WHERE chapter_id = ?",
        (chapter_id,),
    )
    order = cursor.fetchone()["max_order"] + 1
    now = datetime.utcnow().isoformat() + "Z"

    results = []
    for seg in segments:
        segment_id = str(uuid.uuid4())
        character_id = seg.get("character_id")
        text_content = seg["text_content"]

        cursor.execute(
            """
            INSERT INTO segments (id, chapter_id, character_id, text_content, sort_order, audio_path, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (segment_id, chapter_id, character_id, text_content, order, now),
        )

        results.append(
            Segment(
                id=segment_id,
                chapter_id=chapter_id,
                character_id=character_id,
                text_content=text_content,
                sort_order=order,
                audio_path=None,
                duration_ms=None,
                created_at=now,
            )
        )
        order += 1

    db.commit()
    return results
