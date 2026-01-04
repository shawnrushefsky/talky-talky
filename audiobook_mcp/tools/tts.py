"""Text-to-Speech tools with Maya1 and Fish Speech integration.

This module provides TTS capabilities using:
- Maya1: Voice design via natural language descriptions + emotion tags
- Fish Speech: Voice cloning from reference audio samples (local or cloud API)
"""

import json
import os
import tempfile
import requests
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

from ..db.connection import get_database, get_audiobook_dir, get_current_project_path
from .segments import get_segment
from .characters import get_character
from .voice_samples import add_voice_sample, list_voice_samples


# ============================================================================
# Voice Presets and Constants
# ============================================================================

EMOTION_TAGS = [
    "laugh", "laugh_harder", "chuckle", "giggle", "snort",
    "cry", "sob", "sigh", "gasp", "groan",
    "whisper", "angry", "yell", "scream",
    "cough", "clear_throat", "sniff", "hum",
    "mumble", "stutter",
]

VOICE_PRESETS = {
    "narrator_male": "Realistic male voice in the 40s age with american accent. Low pitch, warm timbre, measured pacing, professional tone.",
    "narrator_female": "Realistic female voice in the 30s age with american accent. Medium pitch, warm timbre, measured pacing, professional tone.",
    "young_male": "Realistic male voice in the 20s age with american accent. Medium-high pitch, bright timbre, energetic pacing, enthusiastic tone.",
    "young_female": "Realistic female voice in the 20s age with american accent. High pitch, bright timbre, energetic pacing, enthusiastic tone.",
    "old_male": "Realistic male voice in the 60s age with american accent. Low pitch, gravelly timbre, slow pacing, wise tone.",
    "old_female": "Realistic female voice in the 60s age with american accent. Medium pitch, gentle timbre, measured pacing, warm tone.",
    "child_male": "Realistic male voice in the 10s age with american accent. High pitch, bright timbre, fast pacing, excited tone.",
    "child_female": "Realistic female voice in the 10s age with american accent. High pitch, bright timbre, fast pacing, excited tone.",
    "villain": "Realistic male voice in the 40s age with british accent. Low pitch, cold timbre, slow pacing, menacing tone.",
    "hero": "Realistic male voice in the 30s age with american accent. Medium-low pitch, strong timbre, confident pacing, determined tone.",
    "mysterious": "Realistic voice in the 30s age. Low pitch, hushed timbre, slow pacing, enigmatic tone.",
}

DEFAULT_DESCRIPTION = VOICE_PRESETS["narrator_female"]
SAMPLE_RATE = 24000

# Fish Speech settings
FISH_SPEECH_API_URL = os.environ.get("FISH_SPEECH_API_URL", "http://localhost:8080")
FISH_AUDIO_API_KEY = os.environ.get("FISH_AUDIO_API_KEY", "")


# ============================================================================
# TTS Engine Management
# ============================================================================

# Global TTS engine instances (loaded lazily)
_maya1_model = None
_maya1_tokenizer = None
_snac_model = None


def _get_project_audio_dir() -> Path:
    """Get the audio directory for the current project."""
    project_path = get_current_project_path()
    if not project_path:
        raise ValueError("No project is currently open. Use open_project first.")
    return Path(get_audiobook_dir(project_path))


@dataclass
class TTSCheckResult:
    status: str  # "ok" or "error"
    maya1_available: bool = False
    fish_speech_local_available: bool = False
    fish_speech_cloud_available: bool = False
    torch_installed: bool = False
    cuda_available: bool = False
    mps_available: bool = False
    device: Optional[str] = None
    device_name: Optional[str] = None
    vram_gb: Optional[float] = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _get_best_device() -> tuple[str, Optional[str], Optional[float]]:
    """Detect the best available device for PyTorch.

    Returns (device_string, device_name, vram_gb)
    Priority: CUDA > MPS > CPU
    """
    try:
        import torch
    except ImportError:
        return ("cpu", "CPU (PyTorch not installed)", None)

    # Check CUDA first (NVIDIA GPUs)
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        return ("cuda", device_name, vram_gb)

    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ("mps", "Apple Silicon (MPS)", None)

    # Fall back to CPU
    return ("cpu", "CPU", None)


def check_tts() -> TTSCheckResult:
    """Check if TTS engines are available and properly configured."""
    result = TTSCheckResult(status="ok")

    # Check PyTorch and device
    try:
        import torch
        result.torch_installed = True
        result.cuda_available = torch.cuda.is_available()
        result.mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        device, device_name, vram_gb = _get_best_device()
        result.device = device
        result.device_name = device_name
        result.vram_gb = vram_gb

        if device == "cpu":
            result.warnings.append("No GPU detected - Maya1 will run on CPU (slower)")
    except ImportError:
        result.warnings.append("PyTorch not installed - Maya1 unavailable")

    # Check Maya1 dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac
        result.maya1_available = True
    except ImportError as e:
        result.warnings.append(f"Maya1 dependencies missing: {e}")

    # Check Fish Speech local server
    try:
        response = requests.get(f"{FISH_SPEECH_API_URL}/", timeout=2)
        result.fish_speech_local_available = response.status_code == 200
    except Exception:
        result.warnings.append(f"Fish Speech local server not available at {FISH_SPEECH_API_URL}")

    # Check Fish Speech cloud API
    if FISH_AUDIO_API_KEY:
        result.fish_speech_cloud_available = True
    else:
        result.warnings.append("FISH_AUDIO_API_KEY not set - cloud API unavailable")

    if not result.maya1_available and not result.fish_speech_local_available and not result.fish_speech_cloud_available:
        result.status = "error"
        result.errors.append("No TTS engines available")

    return result


def list_tts_info() -> dict:
    """List available TTS engines, emotion tags, and voice presets."""
    return {
        "emotion_tags": EMOTION_TAGS,
        "emotion_usage": {
            "inline": "Embed tags in text like: Hello <laugh> how are you?",
            "supported": EMOTION_TAGS,
            "examples": [
                "I can't believe it! <laugh>",
                "<whisper> Don't tell anyone...",
                "NO! <angry> I won't do it!",
            ],
        },
        "voice_presets": VOICE_PRESETS,
        "description_format": {
            "template": "Realistic {gender} voice in the {age} age with {accent} accent. {pitch} pitch, {timbre} timbre, {pacing} pacing, {tone} tone.",
            "genders": ["male", "female"],
            "ages": ["10s", "20s", "30s", "40s", "50s", "60s", "70s"],
            "accents": ["american", "british", "australian", "irish", "scottish"],
            "pitches": ["low", "medium-low", "medium", "medium-high", "high"],
            "timbres": ["warm", "cold", "bright", "gravelly", "gentle", "strong"],
            "pacings": ["slow", "measured", "moderate", "energetic", "fast"],
            "tones": ["professional", "friendly", "menacing", "wise", "enthusiastic", "mysterious"],
        },
        "engines": {
            "maya1": {
                "name": "Maya1",
                "description": "Voice design via natural language descriptions with emotion tags",
                "use_case": "Creating unique voices from text descriptions",
                "requirements": "torch, transformers, snac",
            },
            "fish_speech": {
                "name": "Fish Speech",
                "description": "Voice cloning from reference audio samples",
                "use_case": "Cloning voices for long-form generation",
                "requirements": "Local server or FISH_AUDIO_API_KEY for cloud",
            },
        },
        "configuration": {
            "FISH_SPEECH_API_URL": FISH_SPEECH_API_URL,
            "FISH_AUDIO_API_KEY": "***" if FISH_AUDIO_API_KEY else "(not set)",
        },
    }


# ============================================================================
# Maya1 TTS Engine
# ============================================================================

def _load_maya1():
    """Lazily load Maya1 model."""
    global _maya1_model, _maya1_tokenizer, _snac_model

    if _maya1_model is not None:
        return _maya1_model, _maya1_tokenizer, _snac_model

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import snac
    import sys

    # Auto-detect best device
    device, device_name, _ = _get_best_device()

    # Select appropriate dtype for device
    if device == "cuda":
        dtype = torch.bfloat16
    elif device == "mps":
        # MPS works best with float16 or float32
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading Maya1 model on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Load Maya1 model
    _maya1_tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1")

    if device == "cuda":
        # Use device_map for CUDA
        _maya1_model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        # For MPS and CPU, load then move to device
        _maya1_model = AutoModelForCausalLM.from_pretrained(
            "maya-research/maya1",
            torch_dtype=dtype,
        ).to(device)

    # Load SNAC decoder
    _snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

    print(f"Maya1 model loaded successfully on {device}", file=sys.stderr, flush=True)
    return _maya1_model, _maya1_tokenizer, _snac_model


def generate_with_maya1(
    text: str,
    description: str,
    output_path: Path,
) -> dict:
    """Generate audio using Maya1 TTS."""
    import torch
    import soundfile as sf

    model, tokenizer, snac_model = _load_maya1()
    device = next(model.parameters()).device

    # Build prompt
    prompt = f"<|voice_description|>{description}<|text|>{text}"

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # Extract audio tokens and decode with SNAC
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    audio_codes = generated_ids.unsqueeze(0)
    audio = snac_model.decode(audio_codes)
    audio_np = audio.squeeze().cpu().numpy()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_np, SAMPLE_RATE)

    duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

    return {
        "status": "success",
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "sample_rate": SAMPLE_RATE,
    }


# ============================================================================
# Fish Speech TTS Engine
# ============================================================================

def _get_absolute_sample_path(sample_path: str) -> str:
    """Convert relative sample path to absolute path."""
    if sample_path.startswith("http://") or sample_path.startswith("https://"):
        return sample_path

    if os.path.isabs(sample_path):
        return sample_path

    # Relative path - resolve from project directory
    audio_dir = _get_project_audio_dir()
    return str(audio_dir / sample_path)


def generate_with_fish_speech_local(
    text: str,
    reference_audio_paths: list[str],
    reference_texts: list[str],
    output_path: Path,
) -> dict:
    """Generate audio using local Fish Speech server with voice cloning."""
    # Read reference audio files
    references = []
    for audio_path, ref_text in zip(reference_audio_paths, reference_texts):
        abs_path = _get_absolute_sample_path(audio_path)

        if abs_path.startswith("http"):
            # Download URL to temp file
            response = requests.get(abs_path)
            response.raise_for_status()
            audio_data = response.content
        else:
            with open(abs_path, "rb") as f:
                audio_data = f.read()

        references.append({
            "audio": audio_data,
            "text": ref_text or "",
        })

    # Call Fish Speech API
    # The API expects multipart form data with references
    files = []
    data = {"text": text}

    for i, ref in enumerate(references):
        files.append(("references", (f"ref_{i}.wav", ref["audio"], "audio/wav")))
        data[f"reference_text_{i}"] = ref["text"]

    response = requests.post(
        f"{FISH_SPEECH_API_URL}/v1/tts",
        data=data,
        files=files,
        timeout=300,  # 5 minute timeout for long generations
    )
    response.raise_for_status()

    # Save audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(response.content)

    # Get duration
    import soundfile as sf
    info = sf.info(str(output_path))
    duration_ms = int(info.duration * 1000)

    return {
        "status": "success",
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "sample_rate": info.samplerate,
    }


def generate_with_fish_speech_cloud(
    text: str,
    reference_audio_paths: list[str],
    reference_texts: list[str],
    output_path: Path,
) -> dict:
    """Generate audio using Fish Audio cloud API with voice cloning."""
    from fish_audio_sdk import Session
    from fish_audio_sdk.schemas import TTSRequest, ReferenceAudio

    if not FISH_AUDIO_API_KEY:
        raise ValueError("FISH_AUDIO_API_KEY environment variable not set")

    session = Session(FISH_AUDIO_API_KEY)

    # Prepare reference audio
    references = []
    for audio_path, ref_text in zip(reference_audio_paths, reference_texts):
        abs_path = _get_absolute_sample_path(audio_path)

        if abs_path.startswith("http"):
            response = requests.get(abs_path)
            response.raise_for_status()
            audio_data = response.content
        else:
            with open(abs_path, "rb") as f:
                audio_data = f.read()

        references.append(ReferenceAudio(
            audio=audio_data,
            text=ref_text or "",
        ))

    # Generate audio
    audio_data = b""
    for chunk in session.tts(TTSRequest(
        text=text,
        references=references,
        format="wav",
    )):
        audio_data += chunk

    # Save audio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_data)

    # Get duration
    import soundfile as sf
    info = sf.info(str(output_path))
    duration_ms = int(info.duration * 1000)

    return {
        "status": "success",
        "output_path": str(output_path),
        "duration_ms": duration_ms,
        "sample_rate": info.samplerate,
    }


def generate_with_fish_speech(
    text: str,
    reference_audio_paths: list[str],
    reference_texts: list[str],
    output_path: Path,
    use_cloud: bool = False,
) -> dict:
    """Generate audio using Fish Speech with voice cloning.

    Automatically selects local or cloud API based on availability and preference.
    """
    if use_cloud or FISH_AUDIO_API_KEY:
        try:
            return generate_with_fish_speech_cloud(
                text, reference_audio_paths, reference_texts, output_path
            )
        except Exception as e:
            if use_cloud:
                raise
            # Fall through to local if cloud fails and wasn't explicitly requested
            print(f"Cloud API failed, trying local: {e}", flush=True)

    return generate_with_fish_speech_local(
        text, reference_audio_paths, reference_texts, output_path
    )


# ============================================================================
# High-Level TTS Functions
# ============================================================================

@dataclass
class GenerateResult:
    segment_id: str
    audio_path: str
    duration_ms: int
    description: Optional[str] = None
    engine: str = "maya1"


def generate_segment_audio(
    segment_id: str,
    description: Optional[str] = None,
    engine: str = "maya1",
) -> GenerateResult:
    """Generate audio for a single segment."""
    audio_dir = _get_project_audio_dir()

    # Get segment
    segment = get_segment(segment_id)
    if not segment:
        raise ValueError(f"Segment not found: {segment_id}")

    if not segment.text_content.strip():
        raise ValueError(f"Segment has no text content: {segment_id}")

    # Generate audio file path
    segments_dir = audio_dir / "audio" / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{segment_id}.wav"
    output_path = segments_dir / filename

    if engine == "maya1":
        # Determine voice description from params, character voice config, or default
        voice_description = description

        if not voice_description and segment.character_id:
            character = get_character(segment.character_id)
            if character and character.voice_config:
                voice_config = json.loads(character.voice_config)
                if voice_config.get("provider") == "maya1":
                    voice_description = voice_config.get("voice_id")

        voice_description = voice_description or DEFAULT_DESCRIPTION

        result = generate_with_maya1(segment.text_content, voice_description, output_path)
        duration_ms = result.get("duration_ms", 0)

    elif engine == "fish_speech":
        # Get voice samples for the character
        if not segment.character_id:
            raise ValueError("Fish Speech requires a segment assigned to a character with voice samples")

        samples = list_voice_samples(segment.character_id)
        if not samples:
            raise ValueError(f"No voice samples found for character. Generate or add samples first.")

        ref_paths = [s.sample_path for s in samples]
        ref_texts = [s.sample_text or "" for s in samples]

        result = generate_with_fish_speech(
            segment.text_content, ref_paths, ref_texts, output_path
        )
        duration_ms = result.get("duration_ms", 0)
        voice_description = None

    else:
        raise ValueError(f"Unknown TTS engine: {engine}. Use 'maya1' or 'fish_speech'.")

    # Update segment with audio path
    db = get_database()
    cursor = db.cursor()
    relative_path = f"audio/segments/{filename}"
    cursor.execute(
        "UPDATE segments SET audio_path = ?, duration_ms = ? WHERE id = ?",
        (relative_path, duration_ms, segment_id),
    )
    db.commit()

    return GenerateResult(
        segment_id=segment_id,
        audio_path=relative_path,
        duration_ms=duration_ms,
        description=voice_description if engine == "maya1" else None,
        engine=engine,
    )


def generate_voice_sample(
    character_id: str,
    text: str,
    description: Optional[str] = None,
) -> dict:
    """Generate a voice sample for a character using Maya1.

    These samples can then be used with Fish Speech for voice cloning.
    """
    audio_dir = _get_project_audio_dir()

    # Get character
    character = get_character(character_id)
    if not character:
        raise ValueError(f"Character not found: {character_id}")

    if not text.strip():
        raise ValueError("Sample text is required")

    # Determine voice description
    voice_description = description

    if not voice_description and character.voice_config:
        voice_config = json.loads(character.voice_config)
        if voice_config.get("provider") == "maya1":
            voice_description = voice_config.get("voice_id")

    voice_description = voice_description or DEFAULT_DESCRIPTION

    # Generate audio file path for voice samples
    samples_dir = audio_dir / "audio" / "voice_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Use UUID to allow multiple samples
    import uuid
    sample_id = str(uuid.uuid4())[:8]
    filename = f"{character_id}_{sample_id}.wav"
    output_path = samples_dir / filename

    # Generate audio with Maya1
    result = generate_with_maya1(text, voice_description, output_path)

    duration_ms = result.get("duration_ms", 0)

    # Store the voice sample in the database
    relative_path = f"audio/voice_samples/{filename}"
    sample = add_voice_sample(
        character_id=character_id,
        sample_path=relative_path,
        sample_text=text,
        duration_ms=duration_ms,
    )

    return {
        "character_id": character_id,
        "character_name": character.name,
        "sample_id": sample.id,
        "sample_path": relative_path,
        "sample_text": text,
        "duration_ms": duration_ms,
        "description": voice_description,
    }


def generate_batch_audio(
    segment_ids: Optional[list[str]] = None,
    chapter_id: Optional[str] = None,
    engine: str = "fish_speech",
) -> dict:
    """Generate audio for multiple segments.

    If segment_ids not provided, processes all pending segments (or all in chapter).
    """
    db = get_database()
    cursor = db.cursor()

    # Get segments to process
    if segment_ids:
        ids_to_process = segment_ids
    elif chapter_id:
        cursor.execute(
            "SELECT id FROM segments WHERE chapter_id = ? AND audio_path IS NULL ORDER BY sort_order",
            (chapter_id,),
        )
        ids_to_process = [row["id"] for row in cursor.fetchall()]
    else:
        cursor.execute(
            "SELECT id FROM segments WHERE audio_path IS NULL ORDER BY sort_order"
        )
        ids_to_process = [row["id"] for row in cursor.fetchall()]

    if not ids_to_process:
        return {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
        }

    results = []
    successful = 0
    failed = 0

    for segment_id in ids_to_process:
        try:
            result = generate_segment_audio(segment_id, engine=engine)
            results.append({
                "segment_id": segment_id,
                "status": "success",
                "audio_path": result.audio_path,
                "duration_ms": result.duration_ms,
            })
            successful += 1
        except Exception as e:
            results.append({
                "segment_id": segment_id,
                "status": "error",
                "error": str(e),
            })
            failed += 1

    return {
        "total": len(ids_to_process),
        "successful": successful,
        "failed": failed,
        "results": results,
    }
