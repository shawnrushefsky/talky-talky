"""Voice sample management tools for voice cloning."""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..db.connection import get_database
from ..db.schema import VoiceSample
from .characters import get_character


@dataclass
class VoiceSamplesInfo:
    character_id: str
    character_name: str
    sample_count: int
    total_duration_ms: int
    samples: list[VoiceSample]


def add_voice_sample(
    character_id: str,
    sample_path: str,
    sample_text: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> VoiceSample:
    """Add a voice sample for a character (from local path or URL)."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    # Determine if it's a URL
    is_url = sample_path.startswith("http://") or sample_path.startswith("https://")

    # Get next sort_order for this character
    cursor.execute(
        "SELECT MAX(sort_order) as max_order FROM voice_samples WHERE character_id = ?",
        (character_id,),
    )
    result = cursor.fetchone()
    sort_order = (result["max_order"] or -1) + 1

    sample_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    cursor.execute(
        """
        INSERT INTO voice_samples (id, character_id, sample_path, sample_text, duration_ms, is_url, sort_order, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sample_id, character_id, sample_path, sample_text, duration_ms, 1 if is_url else 0, sort_order, now),
    )
    db.commit()

    return VoiceSample(
        id=sample_id,
        character_id=character_id,
        sample_path=sample_path,
        sample_text=sample_text,
        duration_ms=duration_ms,
        is_url=is_url,
        sort_order=sort_order,
        created_at=now,
    )


def list_voice_samples(character_id: str) -> list[VoiceSample]:
    """List all voice samples for a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    cursor.execute(
        """
        SELECT * FROM voice_samples
        WHERE character_id = ?
        ORDER BY sort_order ASC
        """,
        (character_id,),
    )
    rows = cursor.fetchall()

    return [
        VoiceSample(
            id=row["id"],
            character_id=row["character_id"],
            sample_path=row["sample_path"],
            sample_text=row["sample_text"],
            duration_ms=row["duration_ms"],
            is_url=bool(row["is_url"]),
            sort_order=row["sort_order"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_voice_sample(sample_id: str) -> Optional[VoiceSample]:
    """Get a specific voice sample by ID."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM voice_samples WHERE id = ?", (sample_id,))
    row = cursor.fetchone()

    if not row:
        return None

    return VoiceSample(
        id=row["id"],
        character_id=row["character_id"],
        sample_path=row["sample_path"],
        sample_text=row["sample_text"],
        duration_ms=row["duration_ms"],
        is_url=bool(row["is_url"]),
        sort_order=row["sort_order"],
        created_at=row["created_at"],
    )


def update_voice_sample(
    sample_id: str,
    sample_text: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> VoiceSample:
    """Update a voice sample."""
    db = get_database()
    cursor = db.cursor()

    # Verify sample exists
    existing = get_voice_sample(sample_id)
    if not existing:
        raise ValueError(f"Voice sample not found: {sample_id}")

    # Build update query
    updates = []
    values: list = []

    if sample_text is not None:
        updates.append("sample_text = ?")
        values.append(sample_text)
    if duration_ms is not None:
        updates.append("duration_ms = ?")
        values.append(duration_ms)

    if not updates:
        return existing

    values.append(sample_id)
    cursor.execute(
        f"UPDATE voice_samples SET {', '.join(updates)} WHERE id = ?",
        values,
    )
    db.commit()

    return get_voice_sample(sample_id)  # type: ignore


def delete_voice_sample(sample_id: str) -> None:
    """Delete a voice sample."""
    db = get_database()
    cursor = db.cursor()

    # Verify sample exists
    existing = get_voice_sample(sample_id)
    if not existing:
        raise ValueError(f"Voice sample not found: {sample_id}")

    cursor.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
    db.commit()


def clear_voice_samples(character_id: str) -> dict:
    """Delete all voice samples for a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    cursor.execute(
        "SELECT COUNT(*) as count FROM voice_samples WHERE character_id = ?",
        (character_id,),
    )
    count = cursor.fetchone()["count"]

    cursor.execute("DELETE FROM voice_samples WHERE character_id = ?", (character_id,))
    db.commit()

    return {"deleted_count": count}


def reorder_voice_samples(character_id: str, sample_ids: list[str]) -> list[VoiceSample]:
    """Reorder voice samples for a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    # Verify all sample IDs belong to this character
    existing_samples = list_voice_samples(character_id)
    existing_ids = {s.id for s in existing_samples}

    for sample_id in sample_ids:
        if sample_id not in existing_ids:
            raise ValueError(f"Voice sample {sample_id} not found for character {character_id}")

    # Update sort_order for each sample
    for i, sample_id in enumerate(sample_ids):
        cursor.execute("UPDATE voice_samples SET sort_order = ? WHERE id = ?", (i, sample_id))

    db.commit()
    return list_voice_samples(character_id)


def get_voice_samples_info(character_id: str) -> VoiceSamplesInfo:
    """Get voice samples info for a character (summary)."""
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    samples = list_voice_samples(character_id)
    total_duration = sum(s.duration_ms or 0 for s in samples)

    return VoiceSamplesInfo(
        character_id=character_id,
        character_name=character.name,
        sample_count=len(samples),
        total_duration_ms=total_duration,
        samples=samples,
    )
