"""Database schema and type definitions for audiobook projects."""

import sqlite3
from dataclasses import dataclass
from typing import Optional, Any


def initialize_schema(db: sqlite3.Connection) -> None:
    """Initialize the database schema for an audiobook project."""
    cursor = db.cursor()

    # Project metadata table (single row)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS project (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            description TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Characters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            voice_config TEXT,
            is_narrator INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)

    # Voice samples table (multiple samples per character for better voice cloning)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS voice_samples (
            id TEXT PRIMARY KEY,
            character_id TEXT NOT NULL,
            sample_path TEXT NOT NULL,
            sample_text TEXT,
            duration_ms INTEGER,
            is_url INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
        )
    """)

    # Create index for voice samples
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_voice_samples_character ON voice_samples(character_id)
    """)

    # Chapters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            sort_order INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Create index for chapter ordering
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_chapters_sort_order ON chapters(sort_order)
    """)

    # Segments table
    cursor.execute("""
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
    """)

    # Create indexes for segment queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_segments_chapter ON segments(chapter_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_segments_sort ON segments(chapter_id, sort_order)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_segments_character ON segments(character_id)
    """)

    db.commit()


# Type definitions matching the database schema

@dataclass
class Project:
    id: str
    title: str
    author: Optional[str]
    description: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class Character:
    id: str
    name: str
    description: Optional[str]
    voice_config: Optional[str]  # JSON string
    is_narrator: bool
    created_at: str


@dataclass
class VoiceSample:
    id: str
    character_id: str
    sample_path: str  # Local path or URL to voice sample audio
    sample_text: Optional[str]  # Text that was spoken in the sample
    duration_ms: Optional[int]  # Duration of the sample
    is_url: bool  # Whether sample_path is a URL
    sort_order: int
    created_at: str


@dataclass
class Chapter:
    id: str
    title: str
    sort_order: int
    created_at: str


@dataclass
class Segment:
    id: str
    chapter_id: str
    character_id: Optional[str]
    text_content: str
    sort_order: int
    audio_path: Optional[str]
    duration_ms: Optional[int]
    created_at: str


@dataclass
class VoiceConfig:
    """Voice configuration structure (stored as JSON in voice_config)."""
    provider: str  # e.g., "maya1", "fish_speech", "elevenlabs"
    voice_id: str  # For maya1: description, for fish_speech: model_id or description
    settings: Optional[dict[str, Any]] = None
