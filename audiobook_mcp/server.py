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
import uuid
import threading
from datetime import datetime
from typing import Optional, Any
from dataclasses import asdict, dataclass, field
from enum import Enum

from mcp.server.fastmcp import FastMCP

from .db.connection import close_database

# Import tool implementations
from .tools.projects import (
    init_project,
    open_project,
    get_project_info,
    update_project,
    get_default_project_path,
)
from .tools.characters import (
    add_character,
    get_character,
    update_character,
    delete_character,
    set_voice,
    clear_voice,
    get_characters_with_stats,
)
from .tools.chapters import (
    add_chapter,
    get_chapter,
    update_chapter,
    delete_chapter,
    reorder_chapters,
    get_chapters_with_stats,
)
from .tools.segments import (
    add_segment,
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
    update_voice_sample,
    delete_voice_sample,
    clear_voice_samples,
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
    build_voice_description,
    create_voice_candidates,
    select_voice_candidate,
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


# Server version (should match pyproject.toml)
__version__ = "0.2.0"

# Create MCP server
mcp = FastMCP("Audiobook MCP")

# Register cleanup on exit
atexit.register(close_database)


# ============================================================================
# Queue-Based Job System (single worker to prevent OOM from concurrent TTS)
# ============================================================================


class JobStatus(Enum):
    """Status of an async job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AsyncJob:
    """Represents an async job for long-running operations."""

    job_id: str
    job_type: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[dict] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    queue_position: Optional[int] = None


@dataclass
class QueuedJob:
    """A job waiting in the queue with its execution details."""

    job: AsyncJob
    func: Any
    args: tuple
    kwargs: dict


# Global job storage and queue
_jobs: dict[str, AsyncJob] = {}
_jobs_lock = threading.Lock()
_job_queue: list[QueuedJob] = []  # Simple list as queue (FIFO)
_queue_lock = threading.RLock()  # Use RLock to allow reentrant locking
_worker_thread: Optional[threading.Thread] = None
_worker_running = False


def _update_queue_positions():
    """Update queue_position for all queued jobs.

    Note: Caller should hold _queue_lock or call with lock=True.
    """
    # No lock acquisition here - caller must hold the lock
    for i, queued_job in enumerate(_job_queue):
        queued_job.job.queue_position = i + 1


def _worker_loop():
    """Worker loop that processes jobs one at a time."""
    global _worker_running
    import sys

    while _worker_running:
        # Get next job from queue
        queued_job = None
        with _queue_lock:
            if _job_queue:
                queued_job = _job_queue.pop(0)
                _update_queue_positions()  # Safe - we hold the lock

        if queued_job is None:
            # No jobs, sleep briefly and check again
            threading.Event().wait(0.1)
            continue

        job = queued_job.job
        try:
            update_job(job.job_id, status=JobStatus.RUNNING)
            job.queue_position = None
            print(f"Starting job {job.job_id} ({job.job_type})", file=sys.stderr, flush=True)

            result = queued_job.func(*queued_job.args, **queued_job.kwargs)

            # Convert dataclass result to dict if needed
            if hasattr(result, "__dataclass_fields__"):
                result = asdict(result)

            update_job(job.job_id, status=JobStatus.COMPLETED, progress=1.0, result=result)
            print(f"Completed job {job.job_id}", file=sys.stderr, flush=True)

        except Exception as e:
            update_job(job.job_id, status=JobStatus.FAILED, error=str(e))
            print(f"Failed job {job.job_id}: {e}", file=sys.stderr, flush=True)


def _ensure_worker_running():
    """Start the worker thread if not already running."""
    global _worker_thread, _worker_running

    if _worker_thread is not None and _worker_thread.is_alive():
        return

    _worker_running = True
    _worker_thread = threading.Thread(target=_worker_loop, daemon=True)
    _worker_thread.start()


def create_job(job_type: str, metadata: Optional[dict] = None) -> AsyncJob:
    """Create a new async job and return it."""
    job_id = str(uuid.uuid4())
    job = AsyncJob(
        job_id=job_id,
        job_type=job_type,
        status=JobStatus.PENDING,
        created_at=datetime.now(),
        metadata=metadata or {},
    )
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[AsyncJob]:
    """Get a job by ID."""
    with _jobs_lock:
        return _jobs.get(job_id)


def get_queue_length() -> int:
    """Get the number of jobs waiting in the queue."""
    with _queue_lock:
        return len(_job_queue)


def update_job(
    job_id: str,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> Optional[AsyncJob]:
    """Update a job's status."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return None

        if status:
            job.status = status
            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = datetime.now()

        if progress is not None:
            job.progress = progress

        if result is not None:
            job.result = result

        if error is not None:
            job.error = error

        return job


def enqueue_job(job: AsyncJob, func, *args, **kwargs):
    """Add a job to the queue for processing by the worker."""
    _ensure_worker_running()

    queued_job = QueuedJob(job=job, func=func, args=args, kwargs=kwargs)

    with _queue_lock:
        _job_queue.append(queued_job)
        job.status = JobStatus.QUEUED
        job.queue_position = len(_job_queue)


# Backwards compatibility alias
run_job_async = enqueue_job


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
# MCP Resources (read-only data exposure)
# ============================================================================


@mcp.resource("audiobook://project")
def resource_project() -> str:
    """Current audiobook project information and statistics.

    Returns project metadata, path, and segment/chapter counts.
    Requires a project to be open.
    """
    try:
        info = get_project_info()
        return json.dumps(
            {
                "project": to_dict(info.project),
                "path": info.path,
                "stats": to_dict(info.stats),
            },
            indent=2,
        )
    except ValueError as e:
        return json.dumps({"error": str(e), "hint": "Use open_audiobook_project first"})


@mcp.resource("audiobook://characters")
def resource_characters() -> str:
    """All characters in the current project with voice configurations.

    Returns character list with segment counts and voice settings.
    Requires a project to be open.
    """
    try:
        characters = get_characters_with_stats()
        return json.dumps(
            {
                "count": len(characters),
                "characters": [
                    {
                        **to_dict(c),
                        "voice_config": json.loads(c.voice_config) if c.voice_config else None,
                    }
                    for c in characters
                ],
            },
            indent=2,
        )
    except ValueError as e:
        return json.dumps({"error": str(e), "hint": "Use open_audiobook_project first"})


@mcp.resource("audiobook://chapters")
def resource_chapters() -> str:
    """All chapters in the current project with segment statistics.

    Returns chapter list with segment counts and audio status.
    Requires a project to be open.
    """
    try:
        chapters = get_chapters_with_stats()
        return json.dumps(
            {"count": len(chapters), "chapters": [to_dict(c) for c in chapters]},
            indent=2,
        )
    except ValueError as e:
        return json.dumps({"error": str(e), "hint": "Use open_audiobook_project first"})


@mcp.resource("audiobook://tts/status")
def resource_tts_status() -> str:
    """TTS engine availability and configuration status.

    Returns which TTS engines are available (Maya1, Chatterbox, Fish Speech),
    hardware info (CUDA, MPS, CPU), and setup instructions for unavailable engines.
    """
    from .tools.tts import check_tts

    result = check_tts()
    return json.dumps(to_dict(result), indent=2, default=str)


# ============================================================================
# MCP Prompts (reusable interaction templates)
# ============================================================================


@mcp.prompt()
def audiobook_setup(title: str, author: str = "", num_characters: int = 3) -> str:
    """Guided workflow for setting up a new audiobook project.

    Generates a step-by-step prompt for creating a complete audiobook project
    with characters, voice configurations, and chapter structure.
    """
    author_text = f' by "{author}"' if author else ""
    return f"""Help me set up a new audiobook project titled "{title}"{author_text} with {num_characters} main characters.

Please guide me through these steps:

1. **Initialize Project**: Create the project directory and database
2. **Create Characters**: Add {num_characters} characters with names and descriptions
3. **Design Voices**: For each character, design a unique voice using Maya1 descriptions
4. **Generate Voice Samples**: Create reference audio samples for voice cloning
5. **Add Chapters**: Set up the chapter structure

For each character, I'll need:
- A distinctive name
- A brief description of their role and personality
- A voice description including: gender, age, accent, pitch, timbre, pacing, and tone
- 3 sample texts (50-100 words each) that showcase their voice range

Let's start by initializing the project, then work through each character one by one."""


@mcp.prompt()
def voice_design(character_name: str, character_description: str = "") -> str:
    """Help designing a voice description for a character.

    Generates a prompt to create a Maya1-compatible voice description
    with all the necessary parameters.
    """
    desc_text = (
        f"\n\nCharacter description: {character_description}" if character_description else ""
    )
    return f"""Help me design a voice for the character "{character_name}".{desc_text}

I need a Maya1-compatible voice description with these parameters:
- **Gender**: male, female, or other descriptive term
- **Age**: 10s, 20s, 30s, 40s, 50s, 60s, 70s, or descriptive (elderly, teenage)
- **Accent**: american, british, australian, irish, scottish, indian, or any accent
- **Pitch**: low, medium-low, medium, medium-high, high
- **Timbre**: warm, cold, bright, gravelly, gentle, strong, smooth, husky
- **Pacing**: slow, measured, moderate, energetic, fast
- **Tone**: professional, friendly, menacing, wise, enthusiastic, mysterious, warm, determined

Format the final description as:
"Realistic [gender] voice in the [age] age with [accent] accent. [Pitch] pitch, [timbre] timbre, [pacing] pacing, [tone] tone."

Also suggest 3 sample texts (50-100 words each) that would:
1. Show calm, measured speech (narration or reflection)
2. Show emotional speech (excitement, anger, urgency) with emotion tags like <laugh>, <angry>, <whisper>
3. Show conversational speech (casual dialogue)

These samples will be used to generate reference audio for voice cloning."""


@mcp.prompt()
def voice_workflow() -> str:
    """Complete voice setup and audio generation workflow.

    Documents the intended two-stage TTS workflow:
    1. Maya1 creates voice samples from descriptions
    2. Chatterbox clones those samples for segment audio
    """
    return """# Audiobook Voice Workflow

This documents the intended workflow for creating character voices and generating audio.

## Overview

The system uses a **two-stage TTS workflow**:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Voice Sample Creation (Maya1)                        │
│                                                                 │
│  Voice Description + Sample Scripts → Reference Audio Samples  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Audio Generation (Chatterbox)                        │
│                                                                 │
│  Voice Samples + Segment Text → Final Audio                    │
└─────────────────────────────────────────────────────────────────┘
```

## Stage 1: Create Voice Samples with Maya1

Maya1 is a **voice design** engine - it creates unique voices from natural language descriptions.

### Option A: Voice Candidate Selection (Recommended)

Use this when you want to try multiple voice variations and pick the best one:

**Step 1: Generate Voice Candidates**

Create 3-4 variations of the voice description:

```
generate_voice_candidates(
    character_id="...",
    sample_text="A longer piece of text (~50-100 words) that showcases the voice...",
    voice_descriptions=[
        "Realistic male voice in the 30s with american accent. Low pitch, warm timbre, measured pacing.",
        "Realistic male voice in the 30s with american accent. Medium-low pitch, gravelly timbre, slow pacing.",
        "Realistic male voice in the 40s with american accent. Low pitch, warm timbre, conversational pacing."
    ]
)
```

**Step 2: Listen to Candidates**

Each candidate generates an audio file. Listen to each and pick your favorite.

**Step 3: Select the Winner**

```
choose_voice_candidate(
    character_id="...",
    selected_sample_id="<sample_id of preferred voice>",
    additional_sample_texts=[
        "Additional sample showing emotional range...",
        "Another sample with conversational dialogue..."
    ]
)
```

This will:
- Delete all other candidates
- Optionally generate 2 more samples with the selected voice
- Result in 3 high-quality voice samples for cloning

### Option B: Direct Sample Creation

Use this when you're confident about the voice description:

```
set_character_voice(
    character_id="...",
    provider="maya1",
    voice_ref="Realistic female voice in the 30s age with american accent. Medium pitch, warm timbre, measured pacing, professional tone."
)

create_voice_samples(
    character_id="...",
    voice_description="...",
    sample_texts=[
        "Sample 1: Calm narration or reflection...",
        "Sample 2: Emotional speech with <laugh> or <angry> tags...",
        "Sample 3: Casual conversational dialogue..."
    ]
)
```

## Stage 2: Generate Segment Audio with Chatterbox

Chatterbox is a **voice cloning** engine - it uses the Maya1-generated samples to clone the voice.

### Prerequisites
- Character must have voice samples (from Stage 1)
- Segment must be assigned to a character

### Generate Audio

```
# Single segment
generate_audio_for_segment(
    segment_id="...",
    engine="chatterbox"  # Uses voice samples for cloning
)

# Batch generation (recommended)
generate_batch_segment_audio(
    chapter_id="...",
    engine="chatterbox"
)
```

### Emotion Control in Chatterbox

Add paralinguistic tags to segment text:
- `[laugh]` - laughter
- `[chuckle]` - light laugh
- `[cough]` - cough
- `[sigh]` - sigh

## Common Mistakes to Avoid

1. **DON'T** use Maya1 for segment audio directly - it's for sample creation only
2. **DON'T** try to use Chatterbox without voice samples - create samples first
3. **DON'T** forget to assign segments to characters before generating audio

## Quick Reference

| Task | Tool | Engine |
|------|------|--------|
| Generate voice candidates | `generate_voice_candidates` | Maya1 |
| Select preferred voice | `choose_voice_candidate` | Maya1 |
| Set voice description | `set_character_voice` | - |
| Create voice samples | `create_voice_samples` | Maya1 |
| Generate segment audio | `generate_audio_for_segment` | Chatterbox |
| Batch generate audio | `generate_batch_segment_audio` | Chatterbox |
"""


@mcp.prompt()
def chapter_workflow(chapter_title: str = "the current chapter") -> str:
    """Guide for processing a chapter end-to-end.

    Generates a prompt for the complete workflow from text import
    through audio generation and stitching.
    """
    return f"""Help me process {chapter_title} from start to finish.

**Workflow Steps:**

1. **Import Text**
   - Use `import_text_to_chapter` with screenplay-format text
   - Format: `CHARACTER NAME: Dialogue text here.`
   - Use NARRATOR for non-dialogue narration

2. **Assign Characters**
   - Use `detect_dialogue_in_chapter` to find unassigned segments
   - Use `assign_dialogue_to_character` to bulk-assign by pattern
   - Or use `modify_segment` for individual assignments

3. **Review Segments**
   - Use `get_chapter_segments` to see all segments
   - Verify each segment has the correct character assigned
   - Check that all characters have voice samples

4. **Generate Audio**
   - Use `generate_batch_segment_audio` with engine="chatterbox"
   - Or use `generate_audio_for_segment` for individual control
   - Check progress with `get_job_status` and `list_jobs`

5. **Verify and Stitch**
   - Use `get_audio_status_for_chapter` to check completion
   - Use `stitch_chapter_audio` to combine into single file

Let me know which step you'd like to start with, or if you want me to walk through the entire process."""


# ============================================================================
# Project Management Tools
# ============================================================================


@mcp.tool()
def get_suggested_project_path(title: str) -> dict:
    """Get a suggested project path based on the title.

    Returns a platform-appropriate path in the user's Documents folder.
    Use this before init_audiobook_project to suggest a default location.
    """
    suggested_path = get_default_project_path(title)
    return {
        "suggested_path": suggested_path,
        "title": title,
    }


@mcp.tool()
def init_audiobook_project(
    path: str,
    title: str,
    author: Optional[str] = None,
    description: Optional[str] = None,
) -> dict:
    """Initialize a new audiobook project in a directory.

    Creates .audiobook folder with database and directory structure.
    The directory will be created if it doesn't exist.

    Tip: Use get_suggested_project_path() first to get a recommended location
    in the user's Documents folder.
    """
    project = init_project(path, title, author, description)
    return {
        "success": True,
        "message": f'Project "{project.title}" initialized',
        "project": to_dict(project),
        "path": path,
    }


@mcp.tool()
def open_audiobook_project(path: str) -> dict:
    """Open an existing audiobook project.

    Required before using other project-specific tools.
    """
    project = open_project(path)
    return {
        "success": True,
        "message": f'Project "{project.title}" opened',
        "project": to_dict(project),
    }


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
    # Input validation
    if not name or not name.strip():
        raise ValueError("Character name cannot be empty")
    name = name.strip()
    if len(name) > 200:
        raise ValueError("Character name cannot exceed 200 characters")

    character = add_character(name, description, is_narrator)
    return {
        "success": True,
        "message": f'Character "{character.name}" added',
        "character": to_dict(character),
    }


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
    return {
        "success": True,
        "message": f'Character "{character.name}" updated',
        "character": to_dict(character),
    }


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
    voice_ref: str,
    settings: Optional[dict] = None,
) -> dict:
    """Assign a voice configuration to a character.

    For Maya1: provider='maya1', voice_ref is the voice description.
    For Chatterbox: provider='chatterbox', voice_ref can be any identifier.
    """
    character = set_voice(character_id, provider, voice_ref, settings)
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
    return {
        "success": True,
        "message": f'Voice cleared from "{character.name}"',
        "character": to_dict(character),
    }


# ============================================================================
# Chapter Management Tools
# ============================================================================


@mcp.tool()
def create_chapter(title: str, sort_order: Optional[int] = None) -> dict:
    """Add a new chapter to the audiobook.

    Chapters contain segments of text to be narrated.
    """
    # Input validation
    if not title or not title.strip():
        raise ValueError("Chapter title cannot be empty")
    title = title.strip()
    if len(title) > 500:
        raise ValueError("Chapter title cannot exceed 500 characters")

    chapter = add_chapter(title, sort_order)
    return {
        "success": True,
        "message": f'Chapter "{chapter.title}" added',
        "chapter": to_dict(chapter),
    }


@mcp.tool()
def get_chapters() -> dict:
    """List all chapters in the project with their segment statistics."""
    chapters = get_chapters_with_stats()
    return {"count": len(chapters), "chapters": [to_dict(c) for c in chapters]}


@mcp.tool()
def modify_chapter(chapter_id: str, title: Optional[str] = None) -> dict:
    """Update a chapter's title."""
    chapter = update_chapter(chapter_id, title)
    return {
        "success": True,
        "message": f'Chapter "{chapter.title}" updated',
        "chapter": to_dict(chapter),
    }


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
    return {
        "success": True,
        "message": "Chapters reordered",
        "chapters": [to_dict(c) for c in chapters],
    }


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
    # Input validation
    if not chapter_id or not chapter_id.strip():
        raise ValueError("chapter_id is required")
    if not text_content or not text_content.strip():
        raise ValueError("text_content cannot be empty")
    if len(text_content) > 50000:
        raise ValueError("text_content cannot exceed 50,000 characters")

    segment = add_segment(chapter_id, text_content.strip(), character_id, sort_order)
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


@mcp.tool()
def bulk_create_segments(chapter_id: str, segments: list[dict]) -> dict:
    """Add multiple segments to a chapter in one operation.

    Each segment dict should have:
    - text_content (required): The text for the segment
    - character_id (optional): ID of the character speaking this segment

    Example:
    [
        {"text_content": "Hello, world!", "character_id": "char-123"},
        {"text_content": "Goodbye, world!"}
    ]
    """
    # Input validation
    if not chapter_id or not chapter_id.strip():
        raise ValueError("chapter_id is required")
    if not segments:
        raise ValueError("segments list cannot be empty")
    if len(segments) > 1000:
        raise ValueError("Cannot add more than 1000 segments at once")

    # Validate each segment
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"Segment at index {i} must be a dictionary")
        if "text_content" not in seg or not seg["text_content"]:
            raise ValueError(f"Segment at index {i} is missing required 'text_content'")
        if len(seg["text_content"]) > 50000:
            raise ValueError(f"Segment at index {i} text_content exceeds 50,000 characters")

    result = bulk_add_segments(chapter_id, segments)
    return {
        "success": True,
        "message": f"Added {len(result)} segments",
        "count": len(result),
        "segments": [to_dict(s) for s in result],
    }


# ============================================================================
# Voice Sample Management Tools
# ============================================================================


@mcp.tool()
def create_voice_samples(
    character_id: str,
    voice_description: str,
    sample_texts: list[str],
) -> dict:
    """Generate voice samples for a character using Maya1 TTS.

    Creates reference audio clips for voice cloning with Fish Speech.

    Args:
        character_id: The character to generate samples for.
        voice_description: Detailed description of the voice. Be specific about
            age, gender, accent, pitch, timbre, pacing, and emotional quality.
            Example: "A gruff male pirate in his 50s with a thick British accent.
            Deep, gravelly voice with slow deliberate pacing. Speaks with authority
            and a hint of menace, but can show warmth to trusted crew."
        sample_texts: List of 3 in-character speech samples (50-100 words each).
            Each should show a different emotional range:
            - Sample 1: Calm, measured speech (narration or reflection)
            - Sample 2: Emotional speech (excitement, anger, urgency)
            - Sample 3: Conversational speech (casual dialogue)

    Maya1 emotion tags can be included: <laugh>, <sigh>, <gasp>, <angry>, <whisper>

    Example for a pirate captain:
        voice_description: "A gruff male pirate captain in his 50s with a thick
            British accent. Deep gravelly voice, slow pacing, commanding presence."
        sample_texts: [
            "The sea has been my home for forty years now...",
            "All hands on deck! <angry> The enemy approaches!...",
            "You know lad, being a captain is not just about giving orders..."
        ]
    """
    if len(sample_texts) != 3:
        raise ValueError("Exactly 3 sample texts are required for optimal voice cloning")

    results = []
    for i, text in enumerate(sample_texts):
        result = generate_voice_sample(character_id, text, voice_description)
        results.append(result)

    total_duration = sum(r["duration_ms"] for r in results)

    return {
        "success": True,
        "message": f'Generated 3 voice samples for "{results[0]["character_name"]}" ({total_duration}ms total)',
        "character_id": character_id,
        "character_name": results[0]["character_name"],
        "voice_description": voice_description,
        "total_duration_ms": total_duration,
        "samples": [
            {
                "sample_id": r["sample_id"],
                "sample_path": r["sample_path"],
                "duration_ms": r["duration_ms"],
                "text_preview": r["sample_text"][:80] + "..."
                if len(r["sample_text"]) > 80
                else r["sample_text"],
            }
            for r in results
        ],
    }


@mcp.tool()
def generate_voice_candidates(
    character_id: str,
    sample_text: str,
    voice_descriptions: list[str],
) -> dict:
    """Generate multiple voice candidates with slightly different descriptions.

    Creates one sample per description, all using the same sample_text.
    User can listen to each candidate and pick their favorite voice.

    Workflow:
    1. Call this with 3-4 voice description variations
    2. Listen to each candidate audio file
    3. Call select_voice_candidate with the sample_id of your preferred voice
    4. Optionally generate 2 more samples with the selected voice

    Args:
        character_id: The character to generate candidates for.
        sample_text: The text to speak (same for all candidates, ~50-100 words).
        voice_descriptions: List of 2-5 voice descriptions to try (variations on a theme).

    Example:
        voice_descriptions: [
            "Realistic male voice in the 30s with american accent. Low pitch, warm timbre, measured pacing.",
            "Realistic male voice in the 30s with american accent. Medium-low pitch, gravelly timbre, slow pacing.",
            "Realistic male voice in the 40s with american accent. Low pitch, warm timbre, conversational pacing."
        ]
    """
    result = create_voice_candidates(character_id, sample_text, voice_descriptions)
    return {
        "success": True,
        "message": f'Generated {result["candidate_count"]} voice candidates for "{result["character_name"]}"',
        **result,
    }


@mcp.tool()
def choose_voice_candidate(
    character_id: str,
    selected_sample_id: str,
    additional_sample_texts: Optional[list[str]] = None,
) -> dict:
    """Select a voice candidate and optionally generate more samples.

    After using generate_voice_candidates and listening to the options:
    1. Call this with the sample_id of your preferred voice
    2. All other candidates will be deleted
    3. Optionally provide 2 additional sample texts to generate more reference audio

    Args:
        character_id: The character.
        selected_sample_id: The sample_id of the winning candidate.
        additional_sample_texts: Optional list of 2 more sample texts to generate
            (for better voice cloning quality).
    """
    result = select_voice_candidate(character_id, selected_sample_id, additional_sample_texts)
    return {
        "success": True,
        "message": f'Voice selected for "{result["character_name"]}". '
        f"Deleted {result['deleted_candidates']} other candidates. "
        f"{result['remaining_samples']} sample(s) ready for voice cloning.",
        **result,
    }


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
    return {
        "success": True,
        "message": f"Deleted {result['deleted_count']} voice samples",
        **result,
    }


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
def create_voice_description(
    gender: str = "female",
    age: str = "30s",
    accent: str = "american",
    pitch: str = "medium",
    timbre: str = "warm",
    pacing: str = "measured",
    tone: str = "professional",
) -> dict:
    """Build a Maya1 voice description from individual parameters.

    Creates a properly formatted voice description string. Maya1 understands
    natural language, so any descriptive terms work - the suggestions below
    are just common values.

    Suggestions (not strict - use any descriptive terms):
    - gender: male, female (or any descriptive term)
    - age: 10s, 20s, 30s, 40s, 50s, 60s, 70s (or "elderly", "teenage", etc.)
    - accent: american, british, australian, irish, scottish, indian (or any accent)
    - pitch: low, medium-low, medium, medium-high, high
    - timbre: warm, cold, bright, gravelly, gentle, strong, smooth, husky
    - pacing: slow, measured, moderate, energetic, fast
    - tone: professional, friendly, menacing, wise, enthusiastic, mysterious, warm, determined, calm, excited
    """
    result = build_voice_description(gender, age, accent, pitch, timbre, pacing, tone)
    return to_dict(result)


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
    engine: str = "chatterbox",
    run_async: bool = True,
) -> dict:
    """Generate audio for a single segment using TTS.

    Recommended workflow:
    1. Use Maya1 to create voice samples for each character (create_voice_samples)
    2. Use Chatterbox for all segment audio generation (this tool with engine="chatterbox")

    Engines:
    - chatterbox (default, recommended): Voice cloning with emotion control ([laugh], [cough], etc.)
    - maya1: Direct voice design from description (useful for one-off samples)

    Args:
        segment_id: The segment to generate audio for.
        description: Voice description (for maya1 engine).
        engine: TTS engine to use.
        run_async: If True (default), runs generation in background and returns job_id.
                   Use get_job_status to check progress. If False, blocks until complete.

    Note: Jobs are queued and processed one at a time to prevent memory issues.
    """
    if run_async:
        # Create async job and add to queue
        job = create_job(
            job_type="generate_audio",
            metadata={"segment_id": segment_id, "engine": engine},
        )
        enqueue_job(job, generate_segment_audio, segment_id, description, engine)
        queue_len = get_queue_length()
        return {
            "success": True,
            "async": True,
            "job_id": job.job_id,
            "queue_position": job.queue_position,
            "queue_length": queue_len,
            "message": f"Audio generation queued (position {job.queue_position} of {queue_len}). Use get_job_status to check progress.",
        }
    else:
        # Run synchronously (may timeout for long audio)
        result = generate_segment_audio(segment_id, description, engine)
        return {
            "success": True,
            "async": False,
            "message": "Audio generated for segment",
            **to_dict(result),
        }


@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Get the status of an async job.

    Returns job status, progress, and result (if completed).
    Use this to check on audio generation jobs.
    """
    job = get_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")

    response = {
        "job_id": job.job_id,
        "job_type": job.job_type,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at.isoformat(),
        "metadata": job.metadata,
    }

    # Include queue position for queued jobs
    if job.status == JobStatus.QUEUED and job.queue_position is not None:
        response["queue_position"] = job.queue_position
        response["queue_length"] = get_queue_length()

    if job.started_at:
        response["started_at"] = job.started_at.isoformat()

    if job.completed_at:
        response["completed_at"] = job.completed_at.isoformat()

    if job.status == JobStatus.COMPLETED and job.result:
        response["result"] = job.result

    if job.status == JobStatus.FAILED and job.error:
        response["error"] = job.error

    return response


@mcp.tool()
def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """List async jobs, optionally filtered by status or type.

    Args:
        status: Filter by status ('pending', 'queued', 'running', 'completed', 'failed').
        job_type: Filter by job type (e.g., 'generate_audio').
        limit: Maximum number of jobs to return (default 20).
    """
    with _jobs_lock:
        jobs = list(_jobs.values())

    # Apply filters
    if status:
        try:
            filter_status = JobStatus(status)
            jobs = [j for j in jobs if j.status == filter_status]
        except ValueError:
            raise ValueError(
                f"Invalid status: {status}. Use: pending, queued, running, completed, failed"
            )

    if job_type:
        jobs = [j for j in jobs if j.job_type == job_type]

    # Sort by created_at descending (newest first)
    jobs.sort(key=lambda j: j.created_at, reverse=True)

    # Apply limit
    jobs = jobs[:limit]

    def job_to_dict(j: AsyncJob) -> dict:
        d = {
            "job_id": j.job_id,
            "job_type": j.job_type,
            "status": j.status.value,
            "progress": j.progress,
            "created_at": j.created_at.isoformat(),
            "metadata": j.metadata,
        }
        if j.status == JobStatus.QUEUED and j.queue_position is not None:
            d["queue_position"] = j.queue_position
        return d

    return {
        "count": len(jobs),
        "queue_length": get_queue_length(),
        "jobs": [job_to_dict(j) for j in jobs],
    }


# ============================================================================
# Import Tools
# ============================================================================


@mcp.tool()
def import_text_to_chapter(
    chapter_id: str,
    text: str,
    default_character_id: Optional[str] = None,
) -> dict:
    """Import screenplay-format text into a chapter.

    Expected format:
        CHARACTER NAME: Dialogue or narration text here.

        NARRATOR: The scene description.

        CAPTAIN BLACKBEARD: Arr, me hearties!

    Character names in the script are matched (case-insensitive) to existing
    characters in the project. Unmatched names result in unassigned segments.
    """
    # Input validation
    if not chapter_id or not chapter_id.strip():
        raise ValueError("chapter_id is required")
    if not text or not text.strip():
        raise ValueError("text cannot be empty")
    if len(text) > 500000:
        raise ValueError("text cannot exceed 500,000 characters")

    result = import_chapter_text(chapter_id, text, default_character_id)
    return {
        "success": True,
        "message": f"Created {result.segments_created} segments ({result.assigned_segments} assigned, {result.unassigned_segments} unassigned)",
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
    engine: str = "chatterbox",
) -> dict:
    """Generate audio for multiple segments in batch.

    Either provide a list of segment_ids or a chapter_id to process all segments in a chapter.

    Engines:
    - chatterbox (default): Voice cloning with emotion control
    - maya1: Direct voice design from description
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
