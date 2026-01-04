#!/usr/bin/env python3
"""Audiobook MCP Server - Full-cast audiobook production with AI voice synthesis.

This MCP server orchestrates audiobook production by managing:
- Projects, characters, chapters, and segments
- Voice configurations and samples
- TTS generation with Maya1 and Fish Speech
- Audio stitching and export
"""

import json
import atexit
from typing import Optional, Any
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from .db.connection import close_database
from .db.schema import Project, Character, Chapter, Segment, VoiceSample

# Import tool implementations
from .tools.projects import (
    init_project,
    open_project,
    get_project_info,
    update_project,
)
from .tools.characters import (
    add_character,
    list_characters,
    get_character,
    update_character,
    delete_character,
    set_voice,
    clear_voice,
    get_characters_with_stats,
)
from .tools.chapters import (
    add_chapter,
    list_chapters,
    get_chapter,
    update_chapter,
    delete_chapter,
    reorder_chapters,
    get_chapters_with_stats,
)
from .tools.segments import (
    add_segment,
    list_segments,
    get_segment,
    update_segment,
    delete_segment,
    reorder_segments,
    get_segments_with_characters,
    get_pending_segments,
    bulk_add_segments,
)
from .tools.voice_samples import (
    add_voice_sample,
    list_voice_samples,
    get_voice_sample,
    update_voice_sample,
    delete_voice_sample,
    clear_voice_samples,
    reorder_voice_samples,
    get_voice_samples_info,
)
from .tools.tts import (
    check_tts,
    list_tts_info,
    generate_segment_audio,
    generate_voice_sample,
    generate_batch_audio,
    download_maya1_models,
    get_model_status,
)
from .tools.import_tools import (
    import_chapter_text,
    assign_dialogue,
    export_character_lines,
    detect_dialogue,
    get_line_distribution,
)
from .tools.audio import (
    register_segment_audio,
    get_chapter_audio_status,
    stitch_chapter,
    get_stitch_status,
    stitch_book,
    clear_segment_audio,
)


# Create MCP server
mcp = FastMCP("Audiobook MCP")

# Register cleanup on exit
atexit.register(close_database)


# ============================================================================
# Helper functions
# ============================================================================

def to_dict(obj: Any) -> dict:
    """Convert dataclass or object to dict."""
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


# ============================================================================
# Project Management Tools
# ============================================================================

@mcp.tool()
def init_audiobook_project(
    path: str,
    title: str,
    author: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Initialize a new audiobook project in a directory.

    Creates .audiobook folder with database and directory structure.
    """
    project = init_project(path, title, author, description)
    return {"success": True, "message": f'Project "{project.title}" initialized', "project": to_dict(project)}


@mcp.tool()
def open_audiobook_project(path: str) -> dict:
    """Open an existing audiobook project.

    Required before using other project-specific tools.
    """
    project = open_project(path)
    return {"success": True, "message": f'Project "{project.title}" opened', "project": to_dict(project)}


@mcp.tool()
def get_project() -> dict:
    """Get information about the currently open audiobook project including statistics."""
    info = get_project_info()
    return {
        "project": to_dict(info.project),
        "path": info.path,
        "stats": to_dict(info.stats),
    }


@mcp.tool()
def update_audiobook_project(
    title: Optional[str] = None,
    author: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Update the metadata of the currently open audiobook project."""
    project = update_project(title, author, description)
    return {"success": True, "message": "Project updated", "project": to_dict(project)}


# ============================================================================
# Character Management Tools
# ============================================================================

@mcp.tool()
def create_character(
    name: str,
    description: Optional[str] = None,
    is_narrator: bool = False,
) -> dict:
    """Add a new character to the audiobook project.

    Characters can be assigned voices and speak segments.
    """
    character = add_character(name, description, is_narrator)
    return {"success": True, "message": f'Character "{character.name}" added', "character": to_dict(character)}


@mcp.tool()
def get_characters() -> dict:
    """List all characters in the project with their segment counts and voice configurations."""
    characters = get_characters_with_stats()
    return {
        "count": len(characters),
        "characters": [
            {
                **to_dict(c),
                "voice_config": json.loads(c.voice_config) if c.voice_config else None,
            }
            for c in characters
        ],
    }


@mcp.tool()
def modify_character(
    character_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_narrator: Optional[bool] = None,
) -> dict:
    """Update an existing character's name, description, or narrator status."""
    character = update_character(character_id, name, description, is_narrator)
    return {"success": True, "message": f'Character "{character.name}" updated', "character": to_dict(character)}


@mcp.tool()
def remove_character(character_id: str) -> dict:
    """Delete a character from the project.

    Segments assigned to this character will become unassigned.
    """
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")
    delete_character(character_id)
    return {"success": True, "message": f'Character "{character.name}" deleted'}


@mcp.tool()
def set_character_voice(
    character_id: str,
    provider: str,
    voice_id: str,
    settings: Optional[dict] = None,
) -> dict:
    """Assign a voice configuration to a character.

    For Maya1: provider='maya1', voice_id is the voice description.
    For Fish Speech: provider='fish_speech', voice_id can be a model ID.
    """
    character = set_voice(character_id, provider, voice_id, settings)
    return {
        "success": True,
        "message": f'Voice assigned to "{character.name}"',
        "character": {
            **to_dict(character),
            "voice_config": json.loads(character.voice_config) if character.voice_config else None,
        },
    }


@mcp.tool()
def clear_character_voice(character_id: str) -> dict:
    """Remove the voice configuration from a character."""
    character = clear_voice(character_id)
    return {"success": True, "message": f'Voice cleared from "{character.name}"', "character": to_dict(character)}


# ============================================================================
# Chapter Management Tools
# ============================================================================

@mcp.tool()
def create_chapter(title: str, sort_order: Optional[int] = None) -> dict:
    """Add a new chapter to the audiobook.

    Chapters contain segments of text to be narrated.
    """
    chapter = add_chapter(title, sort_order)
    return {"success": True, "message": f'Chapter "{chapter.title}" added', "chapter": to_dict(chapter)}


@mcp.tool()
def get_chapters() -> dict:
    """List all chapters in the project with their segment statistics."""
    chapters = get_chapters_with_stats()
    return {"count": len(chapters), "chapters": [to_dict(c) for c in chapters]}


@mcp.tool()
def modify_chapter(chapter_id: str, title: Optional[str] = None) -> dict:
    """Update a chapter's title."""
    chapter = update_chapter(chapter_id, title)
    return {"success": True, "message": f'Chapter "{chapter.title}" updated', "chapter": to_dict(chapter)}


@mcp.tool()
def remove_chapter(chapter_id: str) -> dict:
    """Delete a chapter and all its segments."""
    chapter = get_chapter(chapter_id)
    if not chapter:
        raise ValueError(f"Chapter not found: {chapter_id}")
    delete_chapter(chapter_id)
    return {"success": True, "message": f'Chapter "{chapter.title}" and all its segments deleted'}


@mcp.tool()
def reorder_book_chapters(chapter_ids: list[str]) -> dict:
    """Reorder chapters by providing an array of chapter IDs in the desired order."""
    chapters = reorder_chapters(chapter_ids)
    return {"success": True, "message": "Chapters reordered", "chapters": [to_dict(c) for c in chapters]}


# ============================================================================
# Segment Management Tools
# ============================================================================

@mcp.tool()
def create_segment(
    chapter_id: str,
    text_content: str,
    character_id: Optional[str] = None,
    sort_order: Optional[int] = None,
) -> dict:
    """Add a text segment to a chapter.

    Segments are individual pieces of narration assigned to characters.
    """
    segment = add_segment(chapter_id, text_content, character_id, sort_order)
    return {"success": True, "message": "Segment added", "segment": to_dict(segment)}


@mcp.tool()
def get_chapter_segments(chapter_id: str) -> dict:
    """List all segments in a chapter with character information."""
    segments = get_segments_with_characters(chapter_id)
    return {"count": len(segments), "segments": [to_dict(s) for s in segments]}


@mcp.tool()
def modify_segment(
    segment_id: str,
    text_content: Optional[str] = None,
    character_id: Optional[str] = None,
) -> dict:
    """Update a segment's text content or assigned character."""
    segment = update_segment(segment_id, text_content, character_id)
    return {"success": True, "message": "Segment updated", "segment": to_dict(segment)}


@mcp.tool()
def remove_segment(segment_id: str) -> dict:
    """Delete a segment from a chapter."""
    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")
    delete_segment(segment_id)
    return {"success": True, "message": "Segment deleted"}


@mcp.tool()
def reorder_chapter_segments(chapter_id: str, segment_ids: list[str]) -> dict:
    """Reorder segments within a chapter by providing segment IDs in the desired order."""
    segments = reorder_segments(chapter_id, segment_ids)
    return {"success": True, "message": "Segments reordered", "count": len(segments)}


@mcp.tool()
def get_segments_without_audio() -> dict:
    """Get all segments that are missing audio files, organized by chapter and character."""
    segments = get_pending_segments()
    return {"count": len(segments), "segments": [to_dict(s) for s in segments]}


# ============================================================================
# Voice Sample Management Tools
# ============================================================================

@mcp.tool()
def create_voice_sample(
    character_id: str,
    text: str,
    description: Optional[str] = None,
) -> dict:
    """Generate a voice sample for a character using Maya1 TTS.

    Creates a reference audio clip for voice cloning with Fish Speech.
    Recommended: 10-30 seconds of varied speech for best results.
    """
    result = generate_voice_sample(character_id, text, description)
    return {"success": True, "message": f'Voice sample generated for "{result["character_name"]}"', **result}


@mcp.tool()
def add_external_voice_sample(
    character_id: str,
    sample_path: str,
    sample_text: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> dict:
    """Add a voice sample from a local path or URL.

    Multiple samples improve voice cloning quality (e.g., three 10-second clips).
    """
    sample = add_voice_sample(character_id, sample_path, sample_text, duration_ms)
    return {"success": True, "message": "Voice sample added", "sample": to_dict(sample)}


@mcp.tool()
def get_character_voice_samples(character_id: str) -> dict:
    """List all voice samples for a character."""
    samples = list_voice_samples(character_id)
    return {"count": len(samples), "samples": [to_dict(s) for s in samples]}


@mcp.tool()
def modify_voice_sample(
    sample_id: str,
    sample_text: Optional[str] = None,
    duration_ms: Optional[int] = None,
) -> dict:
    """Update a voice sample's metadata."""
    sample = update_voice_sample(sample_id, sample_text, duration_ms)
    return {"success": True, "message": "Voice sample updated", "sample": to_dict(sample)}


@mcp.tool()
def remove_voice_sample(sample_id: str) -> dict:
    """Delete a specific voice sample."""
    delete_voice_sample(sample_id)
    return {"success": True, "message": "Voice sample deleted"}


@mcp.tool()
def remove_all_voice_samples(character_id: str) -> dict:
    """Delete all voice samples for a character."""
    result = clear_voice_samples(character_id)
    return {"success": True, "message": f'Deleted {result["deleted_count"]} voice samples', **result}


@mcp.tool()
def get_voice_samples_summary(character_id: str) -> dict:
    """Get a summary of voice samples for a character, including total count and duration."""
    info = get_voice_samples_info(character_id)
    return {
        "character_id": info.character_id,
        "character_name": info.character_name,
        "sample_count": info.sample_count,
        "total_duration_ms": info.total_duration_ms,
        "samples": [to_dict(s) for s in info.samples],
    }


# ============================================================================
# TTS Tools
# ============================================================================

@mcp.tool()
def check_tts_availability() -> dict:
    """Check if TTS engines (Maya1, Fish Speech) are available and properly configured."""
    result = check_tts()
    return to_dict(result)


@mcp.tool()
def get_tts_info() -> dict:
    """List available TTS engines, emotion tags, voice presets, and description format."""
    return list_tts_info()


@mcp.tool()
def get_tts_model_status() -> dict:
    """Get the download status of TTS models (Maya1 and SNAC).

    Returns information about which models are downloaded and their cache locations.
    Use this to check if models need to be downloaded before using Maya1.
    """
    return get_model_status()


@mcp.tool()
def download_tts_models(force: bool = False) -> dict:
    """Download Maya1 TTS model weights from HuggingFace.

    Downloads both the Maya1 language model and SNAC audio codec.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        force: If True, re-download even if models exist in cache.
    """
    return download_maya1_models(force)


@mcp.tool()
def generate_audio_for_segment(
    segment_id: str,
    description: Optional[str] = None,
    engine: str = "maya1",
) -> dict:
    """Generate audio for a single segment using TTS.

    Supports Maya1 (voice design) or Fish Speech (voice cloning).
    Maya1: Uses description or character's voice config.
    Fish Speech: Requires character to have voice samples.
    """
    result = generate_segment_audio(segment_id, description, engine)
    return {"success": True, "message": "Audio generated for segment", **to_dict(result)}


# ============================================================================
# Import Tools
# ============================================================================

@mcp.tool()
def import_text_to_chapter(
    chapter_id: str,
    text: str,
    default_character_id: Optional[str] = None,
) -> dict:
    """Import prose text into a chapter, automatically splitting into segments.

    Detects dialogue (quoted text) and narration, assigning the default character
    to narration segments. Returns detected character names for dialogue assignment.
    """
    result = import_chapter_text(chapter_id, text, default_character_id)
    return {
        "success": True,
        "message": f"Created {result.segments_created} segments",
        **to_dict(result),
    }


@mcp.tool()
def assign_dialogue_to_character(
    chapter_id: str,
    pattern: str,
    character_id: str,
) -> dict:
    """Assign a character to all dialogue segments matching a regex pattern.

    Useful for bulk-assigning dialogue after import.
    """
    result = assign_dialogue(chapter_id, pattern, character_id)
    return {
        "success": True,
        "message": f"Updated {result.updated_count} segments",
        **to_dict(result),
    }


@mcp.tool()
def get_character_lines(character_id: str) -> dict:
    """Export all lines for a specific character.

    Useful for reviewing a character's dialogue or preparing for batch voice generation.
    """
    result = export_character_lines(character_id)
    return {
        "character_name": result.character_name,
        "total_lines": result.total_lines,
        "total_characters": result.total_characters,
        "lines": [to_dict(line) for line in result.lines],
    }


@mcp.tool()
def detect_dialogue_in_chapter(chapter_id: str) -> dict:
    """Detect potential dialogue and suggest character assignments.

    Analyzes unassigned segments and suggests possible speakers based on
    detected names and existing characters.
    """
    result = detect_dialogue(chapter_id)
    return {
        "total_segments": result.total_segments,
        "unassigned_segments": result.unassigned_segments,
        "detected_names": result.detected_names,
        "suggestions": [to_dict(s) for s in result.suggestions],
    }


@mcp.tool()
def get_character_line_distribution() -> dict:
    """Get a summary of character line distribution across the project.

    Shows segment counts and character usage statistics.
    """
    result = get_line_distribution()
    return {
        "total_segments": result.total_segments,
        "assigned_segments": result.assigned_segments,
        "unassigned_segments": result.unassigned_segments,
        "by_character": [to_dict(c) for c in result.by_character],
    }


# ============================================================================
# Audio Registration & Stitching Tools
# ============================================================================

@mcp.tool()
def register_audio_for_segment(
    segment_id: str,
    audio_path: str,
    duration_ms: Optional[int] = None,
) -> dict:
    """Register an externally-generated audio file for a segment.

    Copies the audio to the project's audio directory and links it to the segment.
    Use this when audio is generated outside the MCP server (e.g., via ElevenLabs).
    """
    result = register_segment_audio(segment_id, audio_path, duration_ms)
    return {"success": True, "message": "Audio registered", **to_dict(result)}


@mcp.tool()
def get_audio_status_for_chapter(chapter_id: str) -> dict:
    """Get the audio status for a chapter.

    Shows which segments have audio and which are missing.
    """
    result = get_chapter_audio_status(chapter_id)
    return {
        "chapter_id": result.chapter_id,
        "chapter_title": result.chapter_title,
        "total_segments": result.total_segments,
        "segments_with_audio": result.segments_with_audio,
        "segments_missing_audio": result.segments_missing_audio,
        "total_duration_ms": result.total_duration_ms,
        "ready_to_stitch": result.ready_to_stitch,
        "missing_segments": [to_dict(s) for s in result.missing_segments],
    }


@mcp.tool()
def stitch_chapter_audio(
    chapter_id: str,
    output_filename: Optional[str] = None,
) -> dict:
    """Stitch all segment audio files in a chapter into a single file.

    All segments must have audio registered. Creates an MP3 in the exports folder.
    """
    result = stitch_chapter(chapter_id, output_filename)
    return {"success": True, "message": "Chapter stitched", **to_dict(result)}


@mcp.tool()
def get_book_stitch_status() -> dict:
    """Get the overall audio stitching status for the entire book.

    Shows chapter-by-chapter readiness and overall progress.
    """
    result = get_stitch_status()
    return {
        "total_chapters": result.total_chapters,
        "chapters_ready": result.chapters_ready,
        "total_segments": result.total_segments,
        "segments_with_audio": result.segments_with_audio,
        "total_duration_ms": result.total_duration_ms,
        "ready_to_stitch_book": result.ready_to_stitch_book,
        "chapters": [to_dict(ch) for ch in result.chapters],
    }


@mcp.tool()
def stitch_full_audiobook(
    output_filename: Optional[str] = None,
    include_chapter_markers: bool = True,
) -> dict:
    """Stitch all chapters into a complete audiobook.

    Creates a single MP3 with optional chapter markers (ID3v2 chapters).
    All chapters must have all segment audio registered.
    """
    result = stitch_book(output_filename, include_chapter_markers)
    return {
        "success": True,
        "message": "Audiobook created",
        "output_path": result.output_path,
        "chapter_count": result.chapter_count,
        "total_duration_ms": result.total_duration_ms,
        "chapters": [to_dict(ch) for ch in result.chapters],
    }


@mcp.tool()
def clear_audio_from_segment(segment_id: str) -> dict:
    """Remove the audio file association from a segment.

    Use this to re-generate audio for a segment.
    """
    clear_segment_audio(segment_id)
    return {"success": True, "message": "Audio cleared from segment"}


@mcp.tool()
def generate_batch_segment_audio(
    segment_ids: Optional[list[str]] = None,
    chapter_id: Optional[str] = None,
    engine: str = "fish_speech",
) -> dict:
    """Generate audio for multiple segments in batch.

    Either provide a list of segment_ids or a chapter_id to process all segments in a chapter.
    Fish Speech is recommended for batch generation (faster for long-form content).
    """
    result = generate_batch_audio(segment_ids, chapter_id, engine)
    return {
        "success": True,
        "message": f"Generated audio for {result['successful']} segments",
        **result,
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the Audiobook MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
