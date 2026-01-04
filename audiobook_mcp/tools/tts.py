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

# Model identifiers for downloading
MAYA1_MODEL_ID = "maya-research/maya1"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


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
    setup_instructions: dict = field(default_factory=dict)


# Installation and setup instructions
MAYA1_SETUP_INSTRUCTIONS = """
## Maya1 Setup (Voice Design)

Maya1 requires PyTorch and related dependencies. Install with:

```bash
pip install "audiobook-mcp[maya1]"
```

Or manually:
```bash
pip install torch transformers snac
```

**Hardware Requirements:**
- NVIDIA GPU with CUDA: Best performance (16GB+ VRAM recommended)
- Apple Silicon (M1/M2/M3/M4): Supported via MPS (slower but works)
- CPU: Supported but slow (not recommended for batch generation)
"""

FISH_SPEECH_LOCAL_SETUP_INSTRUCTIONS = """
## Fish Speech Local Server Setup (Voice Cloning)

Fish Speech requires running a separate inference server. Choose one method:

### Option 1: Docker (Recommended)

**For NVIDIA GPU:**
```bash
docker run -d \\
  --name fish-speech \\
  --gpus all \\
  -p 8080:8080 \\
  fishaudio/fish-speech:latest-server-cuda
```

**For CPU only:**
```bash
docker run -d \\
  --name fish-speech \\
  -p 8080:8080 \\
  fishaudio/fish-speech:latest-server
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Install dependencies
pip install -e .

# Download model checkpoints
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# Start the server
python -m tools.api_server \\
  --listen 0.0.0.0:8080 \\
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \\
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \\
  --decoder-config-name modded_dac_vq
```

### Configuration

Set the server URL (default is http://localhost:8080):
```bash
export FISH_SPEECH_API_URL=http://localhost:8080
```
"""

FISH_SPEECH_CLOUD_SETUP_INSTRUCTIONS = """
## Fish Speech Cloud API Setup (Voice Cloning)

The Fish Audio cloud API is the easiest option - no local server required.

### Steps:

1. Create an account at https://fish.audio
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export FISH_AUDIO_API_KEY=your_api_key_here
```

### Pricing

Fish Audio offers pay-as-you-go pricing. Check https://fish.audio for current rates.
"""


@dataclass
class ModelStatus:
    """Status of a model's availability."""
    model_id: str
    downloaded: bool
    cache_path: Optional[str] = None
    size_gb: Optional[float] = None
    error: Optional[str] = None


def check_model_downloaded(model_id: str) -> ModelStatus:
    """Check if a HuggingFace model is downloaded to the cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir
        from transformers import AutoConfig

        # Try to load config to see if model is cached
        try:
            config_path = try_to_load_from_cache(model_id, "config.json")
            if config_path is not None:
                # Model is cached, get more info
                cache_info = scan_cache_dir()
                for repo in cache_info.repos:
                    if repo.repo_id == model_id:
                        size_gb = round(repo.size_on_disk / (1024**3), 2)
                        return ModelStatus(
                            model_id=model_id,
                            downloaded=True,
                            cache_path=str(repo.repo_path),
                            size_gb=size_gb,
                        )
                return ModelStatus(model_id=model_id, downloaded=True)
            else:
                return ModelStatus(model_id=model_id, downloaded=False)
        except Exception:
            return ModelStatus(model_id=model_id, downloaded=False)

    except ImportError:
        return ModelStatus(
            model_id=model_id,
            downloaded=False,
            error="huggingface_hub not installed",
        )


def download_maya1_models(force: bool = False) -> dict:
    """Download Maya1 model weights from HuggingFace.

    Downloads both the Maya1 language model and SNAC audio codec.
    This may take a while depending on your internet connection (~10GB total).

    Args:
        force: If True, re-download even if models exist in cache.

    Returns:
        Status dict with download results for each model.
    """
    import sys

    results = {
        "maya1": {"status": "pending", "model_id": MAYA1_MODEL_ID},
        "snac": {"status": "pending", "model_id": SNAC_MODEL_ID},
    }

    # Check if already downloaded
    if not force:
        maya1_status = check_model_downloaded(MAYA1_MODEL_ID)
        snac_status = check_model_downloaded(SNAC_MODEL_ID)

        if maya1_status.downloaded and snac_status.downloaded:
            return {
                "status": "already_downloaded",
                "message": "All Maya1 models are already downloaded",
                "models": {
                    "maya1": {
                        "status": "cached",
                        "model_id": MAYA1_MODEL_ID,
                        "cache_path": maya1_status.cache_path,
                        "size_gb": maya1_status.size_gb,
                    },
                    "snac": {
                        "status": "cached",
                        "model_id": SNAC_MODEL_ID,
                        "cache_path": snac_status.cache_path,
                        "size_gb": snac_status.size_gb,
                    },
                },
            }

    # Download Maya1 model
    try:
        print(f"Downloading Maya1 model ({MAYA1_MODEL_ID})...", file=sys.stderr, flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # This will download and cache the model
        AutoTokenizer.from_pretrained(MAYA1_MODEL_ID)
        AutoModelForCausalLM.from_pretrained(MAYA1_MODEL_ID)

        results["maya1"]["status"] = "downloaded"
        print(f"Maya1 model downloaded successfully", file=sys.stderr, flush=True)

    except Exception as e:
        results["maya1"]["status"] = "error"
        results["maya1"]["error"] = str(e)
        print(f"Error downloading Maya1: {e}", file=sys.stderr, flush=True)

    # Download SNAC codec
    try:
        print(f"Downloading SNAC codec ({SNAC_MODEL_ID})...", file=sys.stderr, flush=True)
        import snac

        # This will download and cache the model
        snac.SNAC.from_pretrained(SNAC_MODEL_ID)

        results["snac"]["status"] = "downloaded"
        print(f"SNAC codec downloaded successfully", file=sys.stderr, flush=True)

    except Exception as e:
        results["snac"]["status"] = "error"
        results["snac"]["error"] = str(e)
        print(f"Error downloading SNAC: {e}", file=sys.stderr, flush=True)

    # Determine overall status
    all_success = all(r["status"] in ("downloaded", "cached") for r in results.values())
    any_error = any(r["status"] == "error" for r in results.values())

    return {
        "status": "success" if all_success else ("partial" if not any_error else "error"),
        "message": "Models downloaded successfully" if all_success else "Some models failed to download",
        "models": results,
    }


def get_model_status() -> dict:
    """Get the download status of all TTS models.

    Returns information about which models are downloaded and their cache locations.
    """
    maya1_status = check_model_downloaded(MAYA1_MODEL_ID)
    snac_status = check_model_downloaded(SNAC_MODEL_ID)

    models = {
        "maya1": {
            "model_id": MAYA1_MODEL_ID,
            "downloaded": maya1_status.downloaded,
            "cache_path": maya1_status.cache_path,
            "size_gb": maya1_status.size_gb,
            "required_for": "voice design (creating voices from descriptions)",
        },
        "snac": {
            "model_id": SNAC_MODEL_ID,
            "downloaded": snac_status.downloaded,
            "cache_path": snac_status.cache_path,
            "size_gb": snac_status.size_gb,
            "required_for": "audio decoding (used with Maya1)",
        },
    }

    all_downloaded = maya1_status.downloaded and snac_status.downloaded

    return {
        "all_downloaded": all_downloaded,
        "total_size_gb": sum(m.get("size_gb") or 0 for m in models.values()),
        "models": models,
        "download_instructions": None if all_downloaded else (
            "Use the download_tts_models tool to download missing models, "
            "or run: pip install audiobook-mcp[maya1] && python -c "
            "'from audiobook_mcp.tools.tts import download_maya1_models; download_maya1_models()'"
        ),
    }


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
    """Check if TTS engines are available and properly configured.

    Returns detailed status including setup instructions for any unavailable engines.
    """
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
            result.warnings.append("No GPU detected - Maya1 will run on CPU (slower but functional)")
    except ImportError:
        result.warnings.append("PyTorch not installed - Maya1 unavailable. Install with: pip install torch")
        result.setup_instructions["maya1"] = MAYA1_SETUP_INSTRUCTIONS

    # Check Maya1 dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac
        result.maya1_available = True
    except ImportError as e:
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else str(e)
        result.warnings.append(f"Maya1 dependency missing: {missing_pkg}")
        result.setup_instructions["maya1"] = MAYA1_SETUP_INSTRUCTIONS

    # Check Fish Speech local server
    fish_speech_local_error = None
    try:
        response = requests.get(f"{FISH_SPEECH_API_URL}/", timeout=2)
        if response.status_code == 200:
            result.fish_speech_local_available = True
        else:
            fish_speech_local_error = f"Server returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        fish_speech_local_error = "Connection refused - server not running"
    except requests.exceptions.Timeout:
        fish_speech_local_error = "Connection timeout - server not responding"
    except Exception as e:
        fish_speech_local_error = str(e)

    if fish_speech_local_error:
        result.warnings.append(f"Fish Speech local server ({FISH_SPEECH_API_URL}): {fish_speech_local_error}")

    # Check Fish Speech cloud API
    if FISH_AUDIO_API_KEY:
        result.fish_speech_cloud_available = True
    else:
        result.warnings.append("FISH_AUDIO_API_KEY environment variable not set")

    # Add setup instructions for Fish Speech if neither option is available
    if not result.fish_speech_local_available and not result.fish_speech_cloud_available:
        result.setup_instructions["fish_speech_local"] = FISH_SPEECH_LOCAL_SETUP_INSTRUCTIONS
        result.setup_instructions["fish_speech_cloud"] = FISH_SPEECH_CLOUD_SETUP_INSTRUCTIONS

    # Determine overall status
    if not result.maya1_available and not result.fish_speech_local_available and not result.fish_speech_cloud_available:
        result.status = "error"
        result.errors.append("No TTS engines available. See setup_instructions for how to configure them.")
    elif not result.fish_speech_local_available and not result.fish_speech_cloud_available:
        result.status = "partial"
        result.warnings.append("Only Maya1 available. Fish Speech needed for voice cloning. See setup_instructions.")
    elif not result.maya1_available:
        result.status = "partial"
        result.warnings.append("Only Fish Speech available. Maya1 needed for voice design. See setup_instructions.")

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
