"""Chatterbox TTS Engine - Audio-prompted TTS.

Chatterbox generates speech using reference audio samples for voice matching.
Supports paralinguistic tags like [laugh], [cough], [chuckle].
"""

import os
import sys
from pathlib import Path

from .base import AudioPromptedEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import (
    convert_angle_to_bracket_tags,
    split_text_into_chunks,
    get_best_device,
    get_available_memory_gb,
    redirect_stdout_to_stderr,
)


# ============================================================================
# Constants
# ============================================================================

# Chatterbox paralinguistic tags
EMOTION_TAGS = [
    "laugh",
    "chuckle",
    "cough",
    "sigh",
    "gasp",
]

# Default parameters
DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.5

# Generation limits
MAX_DURATION_SECS = 40
MAX_CHUNK_CHARS = 500


# ============================================================================
# Model Management
# ============================================================================

_model = None


def _load_model():
    """Lazily load Chatterbox model."""
    global _model

    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading Chatterbox TTS model on {device}...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        from chatterbox.tts import ChatterboxTTS

        _model = ChatterboxTTS.from_pretrained(device=device)

    print("Chatterbox TTS model loaded successfully", file=sys.stderr, flush=True)
    return _model


def _concatenate_audio_tensors(audio_tensors: list, sample_rate: int, silence_ms: int = 100):
    """Concatenate audio tensors with silence between them."""
    import torch

    if not audio_tensors:
        raise ValueError("No audio tensors to concatenate")

    if len(audio_tensors) == 1:
        return audio_tensors[0]

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = torch.zeros(1, silence_samples)
    tensors_cpu = [t.cpu() for t in audio_tensors]

    result_parts = []
    for i, audio in enumerate(tensors_cpu):
        result_parts.append(audio)
        if i < len(tensors_cpu) - 1:
            result_parts.append(silence)

    return torch.cat(result_parts, dim=-1)


# ============================================================================
# Engine Implementation
# ============================================================================


class ChatterboxEngine(AudioPromptedEngine):
    """Chatterbox TTS Engine - Audio-prompted voice cloning.

    Clones voices from reference audio samples with emotion control.
    Supports paralinguistic tags like [laugh], [cough], [chuckle].

    Parameters:
        reference_audio_paths (list[str]): Paths to reference audio files.
            At least one required. 10+ seconds of clear speech recommended.
        exaggeration (float): Controls speech expressiveness (0.0-1.0+, default 0.5).
            0.0 = flat/monotone, 0.5 = natural, 0.7+ = dramatic.
        cfg_weight (float): Controls pacing/adherence to reference (0.0-1.0, default 0.5).
            Lower values = slower, more deliberate speech.
    """

    @property
    def name(self) -> str:
        return "Chatterbox TTS"

    @property
    def engine_id(self) -> str:
        return "chatterbox"

    def is_available(self) -> bool:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",
            description="Clones voices from reference audio with emotion control",
            requirements="chatterbox-tts (pip install chatterbox-tts)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=24000,  # Chatterbox uses 24kHz
            supports_emotions=True,
            emotion_format="[tag]",
            emotion_tags=EMOTION_TAGS,
            extra_info={
                "license": "MIT",
                "license_url": "https://github.com/resemble-ai/chatterbox",
                "reference_audio": "10+ seconds of clear speech recommended",
                "parameters": {
                    "reference_audio_paths": {
                        "type": "list[str]",
                        "required": True,
                        "description": "Paths to reference audio files for voice cloning",
                    },
                    "exaggeration": {
                        "type": "float",
                        "default": DEFAULT_EXAGGERATION,
                        "range": [0.0, 1.5],
                        "description": "Controls speech expressiveness",
                        "values": {
                            "0.0": "Flat, monotone",
                            "0.5": "Natural, conversational",
                            "0.7+": "Dramatic, theatrical",
                        },
                    },
                    "cfg_weight": {
                        "type": "float",
                        "default": DEFAULT_CFG_WEIGHT,
                        "range": [0.0, 1.0],
                        "description": "Controls pacing and adherence to reference",
                        "values": {
                            "0.3": "Slower, more deliberate",
                            "0.5": "Balanced (default)",
                        },
                    },
                },
                "recommended_combinations": [
                    {"style": "Natural conversation", "exaggeration": 0.5, "cfg_weight": 0.5},
                    {"style": "Dramatic/emotional", "exaggeration": 0.7, "cfg_weight": 0.3},
                    {"style": "Calm narration", "exaggeration": 0.4, "cfg_weight": 0.5},
                ],
                "emotion_examples": [
                    "Hi there [chuckle], have you got a minute?",
                    "That's hilarious! [laugh] I love it.",
                    "Sorry [cough] excuse me, where was I?",
                ],
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=5.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~6GB VRAM used. First generation slower due to model loading.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=2.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Good performance on Apple Silicon. May use unified memory.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.2,
                    device_type="cpu",
                    reference_hardware="AMD Ryzen 9 5900X",
                    notes="Slow but functional. GPU strongly recommended.",
                ),
            },
            prompting_guide=PromptingGuide(
                overview=(
                    "Chatterbox clones voices from reference audio samples. The quality of "
                    "your output depends heavily on your reference audio quality. Use the "
                    "exaggeration and cfg_weight parameters to fine-tune expressiveness "
                    "and pacing. Add paralinguistic tags like [laugh] for natural reactions."
                ),
                text_formatting=[
                    "Write naturally - Chatterbox handles punctuation well",
                    "Use ellipses (...) for trailing off or hesitation",
                    "Use dashes (--) for interruptions or abrupt stops",
                    "Spell out numbers for clearer pronunciation",
                    "Use phonetic spelling for unusual words if needed",
                ],
                emotion_tags={
                    "format": "[tag]",
                    "placement": "Inline where the sound should occur",
                    "available_tags": {
                        "[laugh]": "Full laugh, use for genuine amusement",
                        "[chuckle]": "Soft, brief laugh - more subtle than [laugh]",
                        "[cough]": "Clearing throat or actual cough",
                        "[sigh]": "Exhale expressing emotion (relief, frustration, etc.)",
                        "[gasp]": "Sharp intake of breath (surprise, shock)",
                    },
                    "notes": [
                        "Tags produce actual sounds, not just inflection",
                        "Place tags where the sound naturally occurs in speech",
                        "Don't overuse - one or two per paragraph maximum",
                        "Tags work best at sentence boundaries or natural pauses",
                    ],
                },
                voice_guidance={
                    "reference_audio_requirements": {
                        "duration": "10-30 seconds of clear speech (longer is better)",
                        "quality": "Clean audio without background noise or music",
                        "content": "Natural speech, not singing or whispering",
                        "format": "WAV or MP3, any sample rate (will be converted)",
                    },
                    "best_practices": [
                        "Use audio that matches the emotional range you need",
                        "Multiple short clips work, but one longer clip is often better",
                        "Avoid reference audio with music, effects, or multiple speakers",
                        "The reference sets the baseline voice; parameters adjust from there",
                    ],
                    "source_ideas": [
                        "Generate reference audio with Maya1 using voice descriptions",
                        "Use existing voice recordings (with permission)",
                        "Record your own voice samples",
                    ],
                },
                parameters={
                    "exaggeration": {
                        "description": "Controls speech expressiveness and emotional intensity",
                        "range": "0.0 to 1.5 (can go higher for extreme effect)",
                        "default": 0.5,
                        "guide": {
                            "0.0-0.2": "Flat, robotic - rarely useful",
                            "0.3-0.4": "Subdued, calm - good for meditation, ASMR",
                            "0.5": "Natural conversation - balanced default",
                            "0.6-0.7": "Animated, engaged - good for storytelling",
                            "0.8-1.0": "Theatrical, dramatic - characters, performances",
                            "1.0+": "Exaggerated - use sparingly for comedic effect",
                        },
                    },
                    "cfg_weight": {
                        "description": "Controls pacing and adherence to reference voice",
                        "range": "0.0 to 1.0",
                        "default": 0.5,
                        "guide": {
                            "0.2-0.3": "Slower, more deliberate pacing",
                            "0.4-0.5": "Balanced, natural pacing",
                            "0.6-0.7": "Slightly faster, more energetic",
                            "0.8-1.0": "Strict adherence to reference (less variation)",
                        },
                    },
                },
                tips=[
                    "Start with defaults (0.5, 0.5) and adjust one parameter at a time",
                    "For dramatic scenes: increase exaggeration to 0.7, lower cfg_weight to 0.3",
                    "For calm narration: lower exaggeration to 0.4, keep cfg_weight at 0.5",
                    "Generate a short test clip before committing to long passages",
                    "The reference audio's emotional tone influences output even with low exaggeration",
                    "Consistent reference audio across generations maintains voice consistency",
                ],
                examples=[
                    {
                        "use_case": "Natural conversation",
                        "text": "Hi there [chuckle], how's it going? I haven't seen you in ages!",
                        "exaggeration": 0.5,
                        "cfg_weight": 0.5,
                        "notes": "Default parameters for everyday dialogue",
                    },
                    {
                        "use_case": "Dramatic moment",
                        "text": "No... [gasp] It can't be. After all these years, you've returned.",
                        "exaggeration": 0.75,
                        "cfg_weight": 0.3,
                        "notes": "Higher exaggeration for emotion, lower cfg_weight for dramatic pacing",
                    },
                    {
                        "use_case": "Calm narration",
                        "text": (
                            "The forest was quiet that morning. A gentle mist hung between "
                            "the trees, softening the edges of the world."
                        ),
                        "exaggeration": 0.4,
                        "cfg_weight": 0.5,
                        "notes": "Lower exaggeration for steady, even narration",
                    },
                    {
                        "use_case": "Comedy/exaggerated",
                        "text": "You want ME to do WHAT?! [laugh] That's the craziest thing I've ever heard!",
                        "exaggeration": 0.9,
                        "cfg_weight": 0.4,
                        "notes": "High exaggeration for comedic emphasis",
                    },
                ],
            ),
        )

    def get_setup_instructions(self) -> str:
        return """
## Chatterbox TTS Setup (Voice Cloning with Emotion Control)

Chatterbox is a high-quality TTS with zero-shot voice cloning and emotion control.
Supports paralinguistic tags like [laugh], [cough], [chuckle] for expressive speech.

### Installation

```bash
pip install chatterbox-tts
```

Or with uv:
```bash
uv pip install chatterbox-tts
```

### Hardware Requirements

- NVIDIA GPU with CUDA (recommended): Best performance
- Apple Silicon (MPS): Supported with some limitations
- CPU: Supported but slower

### Model Variants

- **ChatterboxTurboTTS**: Fastest, 350M parameters
- **ChatterboxTTS**: Standard English model
- **ChatterboxMultilingualTTS**: 23+ languages

### Usage Notes

- Works with 10+ seconds of reference audio
- Use exaggeration=0.7+ for dramatic characters
- Lower cfg_weight (~0.3) for slower, deliberate pacing
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        exaggeration: float = DEFAULT_EXAGGERATION,
        cfg_weight: float = DEFAULT_CFG_WEIGHT,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with Chatterbox voice cloning.

        Args:
            text: Text to synthesize. Can include tags like [laugh].
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio files.
            exaggeration: Expressiveness control (0.0-1.5, default 0.5).
            cfg_weight: Pacing control (0.0-1.0, default 0.5).

        Returns:
            TTSResult with status and metadata.
        """
        import torch
        import torchaudio as ta

        output_path = Path(output_path)

        # Validate inputs
        if not text or not text.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=24000,
                error="Text cannot be empty",
            )

        if not reference_audio_paths:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=24000,
                error="At least one reference audio path is required",
            )

        # Validate reference audio paths
        for ref_path in reference_audio_paths:
            if not os.path.exists(ref_path):
                return TTSResult(
                    status="error",
                    output_path=str(output_path),
                    duration_ms=0,
                    sample_rate=24000,
                    error=f"Reference audio file not found: {ref_path}",
                )

        try:
            # Load model
            model = _load_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=24000,
                error=f"Failed to load Chatterbox model: {e}",
            )

        primary_ref_audio = reference_audio_paths[0]

        # Convert angle-style tags to bracket-style for Chatterbox
        text = convert_angle_to_bracket_tags(text)

        memory_gb, memory_type = get_available_memory_gb()

        try:
            # Redirect stdout to stderr during generation
            # to prevent library output from breaking MCP JSON protocol
            with redirect_stdout_to_stderr():
                # Check if text needs chunking
                if len(text) <= MAX_CHUNK_CHARS:
                    wav = model.generate(
                        text,
                        audio_prompt_path=primary_ref_audio,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight,
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
                        chunk_wav = model.generate(
                            chunk,
                            audio_prompt_path=primary_ref_audio,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )
                        audio_chunks.append(chunk_wav)

                        if memory_type == "cuda":
                            torch.cuda.empty_cache()

                    wav = _concatenate_audio_tensors(audio_chunks, model.sr, silence_ms=100)
                    chunks_used = len(chunks)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ta.save(str(output_path), wav, model.sr)

            duration_ms = int(wav.shape[-1] / model.sr * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=model.sr,
                chunks_used=chunks_used,
                metadata={
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "reference_audio": primary_ref_audio,
                    "memory_gb": memory_gb,
                    "memory_type": memory_type,
                },
            )

        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=24000,
                error=f"Generation failed: {e}",
            )
