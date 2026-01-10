"""Maya1 TTS Engine - Text-prompted TTS.

Maya1 generates speech using natural language voice descriptions.
Supports inline emotion tags like <laugh>, <sigh>, <angry>.
"""

import sys
from pathlib import Path

from .base import TextPromptedEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import (
    convert_bracket_to_angle_tags,
    split_text_into_chunks,
    get_best_device,
    redirect_stdout_to_stderr,
)


# ============================================================================
# Constants
# ============================================================================

# Maya1 emotion tags (complete list from maya-research/maya1)
EMOTION_TAGS = [
    "laugh",
    "laugh_harder",
    "chuckle",
    "giggle",
    "snort",
    "sigh",
    "gasp",
    "exhale",
    "gulp",
    "cry",
    "scream",
    "angry",
    "whisper",
    "excited",
    "curious",
    "sarcastic",
    "sing",
]

# Default voice description
DEFAULT_DESCRIPTION = (
    "Female narrator in her 30s with American accent, warm timbre, measured pacing"
)

# Audio settings
SAMPLE_RATE = 24000

# SNAC token format constants
SNAC_TOKENS_PER_FRAME = 7
CODE_TOKEN_OFFSET = 128266
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258

# Special tokens for prompt construction
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
TEXT_EOT_ID = 128009

# Generation limits
MAX_TOKENS = 4096
MAX_DURATION_SECS = 48
MAX_CHUNK_CHARS = 600

# Model identifiers
MODEL_ID = "maya-research/maya1"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"


# ============================================================================
# Model Management
# ============================================================================

# Global model instances (loaded lazily)
_model = None
_tokenizer = None
_snac_model = None


def _load_models():
    """Lazily load Maya1 model and SNAC decoder."""
    global _model, _tokenizer, _snac_model

    if _model is not None:
        return _model, _tokenizer, _snac_model

    device, device_name, _ = get_best_device()

    print(f"Loading Maya1 model on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import snac

        # Select appropriate dtype for device
        if device == "cuda":
            dtype = torch.bfloat16
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = torch.float32

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        if device == "cuda":
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
            ).to(device)

        _snac_model = snac.SNAC.from_pretrained(SNAC_MODEL_ID).to(device)

    print(f"Maya1 model loaded successfully on {device}", file=sys.stderr, flush=True)
    return _model, _tokenizer, _snac_model


def _unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]

    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]

    if frames == 0:
        return [[], [], []]

    l1, l2, l3 = [], [], []

    for i in range(frames):
        slots = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend(
            [
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )

    return [l1, l2, l3]


def _build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token

    formatted_text = f'<description="{description}"> {text}'
    return soh_token + bos_token + formatted_text + eot_token + eoh_token + soa_token + sos_token


def _generate_chunk(
    text: str,
    description: str,
    model,
    tokenizer,
    snac_model,
    device,
    temperature: float = 0.4,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
):
    """Generate a single chunk of audio with Maya1."""
    import torch

    prompt = _build_prompt(tokenizer, description, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            min_new_tokens=28,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :].tolist()
    levels = _unpack_snac_from_7(generated_ids)

    if not levels[0]:
        raise ValueError("No audio tokens generated")

    codes_tensor = [
        torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) for level in levels
    ]

    with torch.inference_mode():
        z_q = snac_model.quantizer.from_codes(codes_tensor)
        audio = snac_model.decoder(z_q)

    audio_np = audio[0, 0].cpu().numpy()

    # Trim warmup artifacts
    if len(audio_np) > 2048:
        audio_np = audio_np[2048:]

    return audio_np


# ============================================================================
# Engine Implementation
# ============================================================================


class Maya1Engine(TextPromptedEngine):
    """Maya1 TTS Engine - Text-prompted voice design.

    Generates audio from text using natural language voice descriptions.
    Supports inline emotion tags like <laugh>, <sigh>, <angry>.

    Parameters:
        voice_description (str): Natural language description of the voice.
            Example: "Gruff male pirate, 50s, British accent, slow pacing"
        temperature (float): Sampling temperature (default 0.4). Lower = more stable.
        top_p (float): Nucleus sampling parameter (default 0.9).
        repetition_penalty (float): Penalty for repetition (default 1.1).
    """

    @property
    def name(self) -> str:
        return "Maya1"

    @property
    def engine_id(self) -> str:
        return "maya1"

    def is_available(self) -> bool:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
            import snac  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="text_prompted",
            description="Creates unique voices from natural language descriptions",
            requirements="torch, transformers, snac (~8GB VRAM)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=True,
            emotion_format="<tag>",
            emotion_tags=EMOTION_TAGS,
            prompting_guide=PromptingGuide(
                overview=(
                    "Maya1 creates unique voices from natural language descriptions. "
                    "Describe the voice like you're briefing a voice actor - be specific "
                    "about age, gender, accent, and character. Supports inline emotion tags."
                ),
                text_formatting=[
                    "Write naturally - Maya1 handles punctuation and pacing automatically",
                    "Use ... for pauses: 'Well... I suppose you're right'",
                    "Emotion tags go inline at the exact moment: 'The treasure! <gasp> It's real!'",
                    "Don't overload with tags - one per sentence is usually enough",
                    "Tags work best at natural pause points (after punctuation)",
                ],
                emotion_tags={
                    "format": "<tag>",
                    "available": EMOTION_TAGS,
                    "categories": {
                        "laughter": ["laugh", "laugh_harder", "chuckle", "giggle", "snort"],
                        "breath": ["sigh", "gasp", "exhale", "gulp"],
                        "distress": ["cry", "scream"],
                        "vocal_style": ["angry", "whisper", "excited", "curious", "sarcastic"],
                        "other": ["sing"],
                    },
                    "examples": [
                        {
                            "text": "The treasure! <gasp> After all these years!",
                            "note": "Gasp at moment of discovery",
                        },
                        {
                            "text": "<whisper> Don't tell anyone about this...",
                            "note": "Whisper at start for secretive tone",
                        },
                        {
                            "text": "NO! <angry> I won't do it!",
                            "note": "Angry emphasis after exclamation",
                        },
                        {
                            "text": "<laugh> I can hardly believe it worked!",
                            "note": "Laugh before joyful statement",
                        },
                        {
                            "text": "And then he said... <chuckle> you won't believe this...",
                            "note": "Chuckle mid-sentence",
                        },
                    ],
                },
                voice_guidance={
                    "type": "description",
                    "structure": "Describe like briefing a voice actor. Include: gender, age, accent, pitch, timbre, pacing, character.",
                    "elements": {
                        "gender": ["male", "female"],
                        "age": ["child", "teenage", "20s", "30s", "40s", "50s", "60s", "elderly"],
                        "accent": [
                            "American",
                            "British",
                            "Australian",
                            "Irish",
                            "Scottish",
                            "Southern US",
                            "New York",
                        ],
                        "pitch": ["low", "medium-low", "medium", "medium-high", "high"],
                        "timbre": [
                            "warm",
                            "bright",
                            "gravelly",
                            "smooth",
                            "husky",
                            "nasal",
                            "resonant",
                        ],
                        "pacing": ["slow", "measured", "moderate", "energetic", "fast"],
                        "character": [
                            "authoritative",
                            "gentle",
                            "menacing",
                            "cheerful",
                            "wise",
                            "nervous",
                        ],
                    },
                    "examples": [
                        "Female narrator in her 30s with American accent, warm timbre, measured pacing",
                        "Gruff male pirate, 50s, British accent, low pitch, gravelly, slow deliberate speech",
                        "Elderly woman, warm and grandmotherly, slight Irish lilt, gentle and wise",
                        "Young male reporter, 20s, energetic, clear American accent, fast pacing",
                        "Dark villain, male, 40s, British accent, low pitch, cold timbre, menacing",
                    ],
                    "tips": [
                        "Short, specific descriptions work better than verbose ones",
                        "Focus on 3-4 distinctive qualities rather than listing everything",
                        "Character archetypes help: 'pirate captain', 'news anchor', 'fairy godmother'",
                        "Consistency matters - use the same description for the same character",
                    ],
                },
                parameters={
                    "voice_description": {
                        "type": "str",
                        "required": True,
                        "description": "Natural language description of the voice",
                    },
                    "temperature": {
                        "type": "float",
                        "default": 0.4,
                        "range": [0.1, 1.0],
                        "description": "Sampling randomness. Lower = more consistent, higher = more varied",
                        "recommended": "0.4 for narration, 0.5-0.6 for dialogue with more variation",
                    },
                    "top_p": {
                        "type": "float",
                        "default": 0.9,
                        "range": [0.1, 1.0],
                        "description": "Nucleus sampling. Usually leave at default.",
                    },
                    "repetition_penalty": {
                        "type": "float",
                        "default": 1.1,
                        "range": [1.0, 2.0],
                        "description": "Prevents repetitive patterns. Increase if output loops.",
                    },
                },
                tips=[
                    "Generate a short sample first to verify the voice before long passages",
                    "For consistent characters, save the exact voice_description and reuse it",
                    "If output sounds robotic, try slightly higher temperature (0.5-0.6)",
                    "If output is unstable/garbled, lower temperature to 0.3-0.4",
                    "Long texts are automatically chunked - each chunk uses the same voice",
                ],
                examples=[
                    {
                        "name": "Pirate Captain",
                        "voice_description": "Gruff male pirate captain, 50s, British accent, low gravelly voice, slow deliberate speech, commanding presence",
                        "text": "Arr, me hearties! <laugh> The treasure be ours at last! <whisper> But tell no one where we found it...",
                        "params": {"temperature": 0.4},
                    },
                    {
                        "name": "News Anchor",
                        "voice_description": "Professional female news anchor, 30s, American accent, clear authoritative voice, measured pacing",
                        "text": "Breaking news tonight. Officials confirm the discovery. <sigh> More details as they become available.",
                        "params": {"temperature": 0.3},
                    },
                    {
                        "name": "Excited Child",
                        "voice_description": "Young girl, 8 years old, American accent, high pitch, energetic and excited",
                        "text": "<excited> Mom, mom, mom! <gasp> Look what I found! <giggle> It's so cool!",
                        "params": {"temperature": 0.5},
                    },
                ],
            ),
            extra_info={
                "model_id": MODEL_ID,
                "snac_model_id": SNAC_MODEL_ID,
                "default_description": DEFAULT_DESCRIPTION,
                "license": "Apache-2.0",
                "license_url": "https://huggingface.co/maya-research/maya1",
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=3.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~8GB VRAM used. First generation slower due to model loading.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=1.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Approximately realtime. May vary with Metal Performance Shaders load.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.1,
                    device_type="cpu",
                    reference_hardware="AMD Ryzen 9 5900X",
                    notes="Very slow. GPU strongly recommended.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Maya1 Setup (Voice Design)

Maya1 requires PyTorch and related dependencies. Install with:

```bash
pip install "talky-talky[maya1]"
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

    def generate(
        self,
        text: str,
        output_path: Path,
        voice_description: str = DEFAULT_DESCRIPTION,
        temperature: float = 0.4,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with Maya1.

        Args:
            text: Text to synthesize. Can include emotion tags like <laugh>.
            output_path: Where to save the generated audio.
            voice_description: Natural language voice description.
            temperature: Sampling temperature (0.1-1.0, default 0.4).
            top_p: Nucleus sampling (0.1-1.0, default 0.9).
            repetition_penalty: Repetition penalty (1.0-2.0, default 1.1).

        Returns:
            TTSResult with status and metadata.
        """
        import numpy as np
        import soundfile as sf

        output_path = Path(output_path)

        # Validate inputs
        if not text or not text.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Text cannot be empty",
            )

        if not voice_description or not voice_description.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Voice description cannot be empty",
            )

        try:
            model, tokenizer, snac_model = _load_models()
            device = next(model.parameters()).device
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load Maya1 models: {e}",
            )

        # Convert bracket-style tags to angle-style for Maya1
        text = convert_bracket_to_angle_tags(text)

        try:
            # Check if text needs chunking
            if len(text) <= MAX_CHUNK_CHARS:
                audio_np = _generate_chunk(
                    text,
                    voice_description,
                    model,
                    tokenizer,
                    snac_model,
                    device,
                    temperature,
                    top_p,
                    repetition_penalty,
                )
                chunks_used = 1
            else:
                chunks = split_text_into_chunks(text, MAX_CHUNK_CHARS)
                print(
                    f"Splitting into {len(chunks)} chunks ({len(text)} chars)",
                    file=sys.stderr,
                    flush=True,
                )

                audio_chunks = []
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i + 1}/{len(chunks)}...", file=sys.stderr, flush=True)
                    chunk_audio = _generate_chunk(
                        chunk,
                        voice_description,
                        model,
                        tokenizer,
                        snac_model,
                        device,
                        temperature,
                        top_p,
                        repetition_penalty,
                    )
                    audio_chunks.append(chunk_audio)

                # Concatenate with 100ms silence between chunks
                silence = np.zeros(2400, dtype=np.float32)
                result_parts = []
                for i, chunk_audio in enumerate(audio_chunks):
                    result_parts.append(chunk_audio)
                    if i < len(audio_chunks) - 1:
                        result_parts.append(silence)

                audio_np = np.concatenate(result_parts)
                chunks_used = len(chunks)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_np, SAMPLE_RATE)

            duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
                chunks_used=chunks_used,
                metadata={
                    "voice_description": voice_description,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                },
            )

        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Generation failed: {e}",
            )


# ============================================================================
# Model Management Functions
# ============================================================================


def check_models_downloaded() -> dict:
    """Check if Maya1 models are downloaded."""
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir

        def check_model(model_id: str) -> dict:
            try:
                config_path = try_to_load_from_cache(model_id, "config.json")
                if config_path is not None:
                    cache_info = scan_cache_dir()
                    for repo in cache_info.repos:
                        if repo.repo_id == model_id:
                            return {
                                "downloaded": True,
                                "cache_path": str(repo.repo_path),
                                "size_gb": round(repo.size_on_disk / (1024**3), 2),
                            }
                    return {"downloaded": True}
                return {"downloaded": False}
            except Exception:
                return {"downloaded": False}

        maya1_status = check_model(MODEL_ID)
        snac_status = check_model(SNAC_MODEL_ID)

        return {
            "maya1": {**maya1_status, "model_id": MODEL_ID},
            "snac": {**snac_status, "model_id": SNAC_MODEL_ID},
            "all_downloaded": maya1_status.get("downloaded") and snac_status.get("downloaded"),
        }

    except ImportError:
        return {
            "maya1": {"downloaded": False, "error": "huggingface_hub not installed"},
            "snac": {"downloaded": False, "error": "huggingface_hub not installed"},
            "all_downloaded": False,
        }


def download_models(force: bool = False) -> dict:
    """Download Maya1 models from HuggingFace."""
    if not force:
        status = check_models_downloaded()
        if status["all_downloaded"]:
            return {
                "status": "already_downloaded",
                "message": "All Maya1 models are already downloaded",
                "models": status,
            }

    results = {"maya1": {"status": "pending"}, "snac": {"status": "pending"}}

    try:
        print(f"Downloading Maya1 model ({MODEL_ID})...", file=sys.stderr, flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL_ID)
        AutoModelForCausalLM.from_pretrained(MODEL_ID)
        results["maya1"]["status"] = "downloaded"
    except Exception as e:
        results["maya1"]["status"] = "error"
        results["maya1"]["error"] = str(e)

    try:
        print(f"Downloading SNAC codec ({SNAC_MODEL_ID})...", file=sys.stderr, flush=True)
        import snac

        snac.SNAC.from_pretrained(SNAC_MODEL_ID)
        results["snac"]["status"] = "downloaded"
    except Exception as e:
        results["snac"]["status"] = "error"
        results["snac"]["error"] = str(e)

    all_success = all(r["status"] in ("downloaded", "cached") for r in results.values())
    return {
        "status": "success" if all_success else "error",
        "message": "Models downloaded" if all_success else "Some models failed",
        "models": results,
    }
