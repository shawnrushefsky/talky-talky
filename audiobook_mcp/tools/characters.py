"""Character management tools."""

import json
import uuid
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass

from ..db.connection import get_database
from ..db.schema import Character, VoiceConfig


@dataclass
class CharacterWithStats:
    id: str
    name: str
    description: Optional[str]
    voice_config: Optional[str]
    is_narrator: bool
    created_at: str
    segment_count: int
    total_text_length: int


def add_character(
    name: str,
    description: Optional[str] = None,
    is_narrator: bool = False,
) -> Character:
    """Add a new character to the project."""
    db = get_database()
    cursor = db.cursor()

    # Check for duplicate name
    cursor.execute("SELECT id FROM characters WHERE name = ?", (name,))
    if cursor.fetchone():
        raise ValueError(f'Character with name "{name}" already exists')

    character_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    cursor.execute(
        """
        INSERT INTO characters (id, name, description, voice_config, is_narrator, created_at)
        VALUES (?, ?, ?, NULL, ?, ?)
        """,
        (character_id, name, description, 1 if is_narrator else 0, now),
    )
    db.commit()

    return Character(
        id=character_id,
        name=name,
        description=description,
        voice_config=None,
        is_narrator=is_narrator,
        created_at=now,
    )


def list_characters() -> list[Character]:
    """List all characters in the project."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM characters ORDER BY is_narrator DESC, name ASC")
    rows = cursor.fetchall()

    return [
        Character(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            voice_config=row["voice_config"],
            is_narrator=bool(row["is_narrator"]),
            created_at=row["created_at"],
        )
        for row in rows
    ]


def get_character(character_id: str) -> Optional[Character]:
    """Get a character by ID."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("SELECT * FROM characters WHERE id = ?", (character_id,))
    row = cursor.fetchone()

    if not row:
        return None

    return Character(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        voice_config=row["voice_config"],
        is_narrator=bool(row["is_narrator"]),
        created_at=row["created_at"],
    )


def update_character(
    character_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_narrator: Optional[bool] = None,
) -> Character:
    """Update a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    existing = get_character(character_id)
    if not existing:
        raise ValueError(f"Character not found: {character_id}")

    # Check for name conflict if renaming
    if name and name != existing.name:
        cursor.execute(
            "SELECT id FROM characters WHERE name = ? AND id != ?",
            (name, character_id),
        )
        if cursor.fetchone():
            raise ValueError(f'Character with name "{name}" already exists')

    # Build update query
    updates = []
    values: list = []

    if name is not None:
        updates.append("name = ?")
        values.append(name)
    if description is not None:
        updates.append("description = ?")
        values.append(description)
    if is_narrator is not None:
        updates.append("is_narrator = ?")
        values.append(1 if is_narrator else 0)

    if not updates:
        return existing

    values.append(character_id)
    cursor.execute(
        f"UPDATE characters SET {', '.join(updates)} WHERE id = ?",
        values,
    )
    db.commit()

    return get_character(character_id)  # type: ignore


def delete_character(character_id: str) -> None:
    """Delete a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    existing = get_character(character_id)
    if not existing:
        raise ValueError(f"Character not found: {character_id}")

    cursor.execute("DELETE FROM characters WHERE id = ?", (character_id,))
    db.commit()


def set_voice(
    character_id: str,
    provider: str,
    voice_id: str,
    settings: Optional[dict[str, Any]] = None,
) -> Character:
    """Set voice configuration for a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    existing = get_character(character_id)
    if not existing:
        raise ValueError(f"Character not found: {character_id}")

    voice_config = VoiceConfig(provider=provider, voice_id=voice_id, settings=settings)
    voice_config_json = json.dumps({
        "provider": voice_config.provider,
        "voice_id": voice_config.voice_id,
        "settings": voice_config.settings,
    })

    cursor.execute(
        "UPDATE characters SET voice_config = ? WHERE id = ?",
        (voice_config_json, character_id),
    )
    db.commit()

    return get_character(character_id)  # type: ignore


def clear_voice(character_id: str) -> Character:
    """Clear voice configuration for a character."""
    db = get_database()
    cursor = db.cursor()

    # Verify character exists
    existing = get_character(character_id)
    if not existing:
        raise ValueError(f"Character not found: {character_id}")

    cursor.execute(
        "UPDATE characters SET voice_config = NULL WHERE id = ?",
        (character_id,),
    )
    db.commit()

    return get_character(character_id)  # type: ignore


def get_characters_with_stats() -> list[CharacterWithStats]:
    """Get characters with their line counts."""
    db = get_database()
    cursor = db.cursor()

    cursor.execute("""
        SELECT
            c.*,
            COUNT(s.id) as segment_count,
            COALESCE(SUM(LENGTH(s.text_content)), 0) as total_text_length
        FROM characters c
        LEFT JOIN segments s ON s.character_id = c.id
        GROUP BY c.id
        ORDER BY c.is_narrator DESC, c.name ASC
    """)
    rows = cursor.fetchall()

    return [
        CharacterWithStats(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            voice_config=row["voice_config"],
            is_narrator=bool(row["is_narrator"]),
            created_at=row["created_at"],
            segment_count=row["segment_count"],
            total_text_length=row["total_text_length"],
        )
        for row in rows
    ]
