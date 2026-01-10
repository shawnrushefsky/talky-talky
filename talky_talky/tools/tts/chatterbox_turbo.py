"""Chatterbox Turbo TTS Engine - Fast audio-prompted TTS.

Chatterbox Turbo is a streamlined 350M parameter model optimized for
low-latency voice cloning in production voice agents.
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

# Chatterbox Turbo paralinguistic tags
EMOTION_TAGS = [
    "laugh",
    "chuckle",
    "cough",
]

# Generation limits
MAX_DURATION_SECS = 40
MAX_CHUNK_CHARS = 500

# Sample rate is determined by the model
SAMPLE_RATE = 24000  # Default, actual rate from model.sr


# ============================================================================
# Model Management
# ============================================================================

_model = None


def _load_model():
    """Lazily load Chatterbox Turbo model."""
    global _model

    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading Chatterbox Turbo TTS model on {device}...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        _model = ChatterboxTurboTTS.from_pretrained(device=device)

    print("Chatterbox Turbo TTS model loaded successfully", file=sys.stderr, flush=True)
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


class ChatterboxTurboEngine(AudioPromptedEngine):
    """Chatterbox Turbo TTS Engine - Fast audio-prompted voice cloning.

    Chatterbox Turbo is a streamlined 350M parameter model optimized for
    low-latency voice cloning. Simpler than the full Chatterbox model
    with fewer parameters to tune.

    Parameters:
        reference_audio_paths (list[str]): Paths to reference audio files.
            At least one required. 10+ seconds of clear speech recommended.

    Supports paralinguistic tags: [laugh], [chuckle], [cough]
    """

    @property
    def name(self) -> str:
        return "Chatterbox Turbo"

    @property
    def engine_id(self) -> str:
        return "chatterbox_turbo"

    def is_available(self) -> bool:
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",
            description="Fast voice cloning optimized for low-latency production use",
            requirements="chatterbox-tts (pip install chatterbox-tts)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=True,
            emotion_format="[tag]",
            emotion_tags=EMOTION_TAGS,
            extra_info={
                "license": "MIT",
                "license_url": "https://github.com/resemble-ai/chatterbox",
                "model_size": "350M parameters",
                "reference_audio": "10+ seconds of clear speech recommended",
                "latency": "<200ms production latency",
                "parameters": {
                    "reference_audio_paths": {
                        "type": "list[str]",
                        "required": True,
                        "description": "Paths to reference audio files for voice cloning",
                    },
                },
                "emotion_examples": [
                    "Hi there [chuckle], have you got a minute?",
                    "That's hilarious! [laugh] I love it.",
                    "Sorry [cough] excuse me, where was I?",
                ],
                "differences_from_standard": {
                    "parameters": "No cfg_weight or exaggeration tuning",
                    "speed": "Faster inference, lower latency",
                    "size": "350M vs 500M parameters",
                    "use_case": "Production voice agents",
                },
            },
            prompting_guide=PromptingGuide(
                overview=(
                    "Chatterbox Turbo is a streamlined voice cloning model optimized for "
                    "speed. It clones voices from reference audio with minimal configuration. "
                    "Perfect for production voice agents where latency matters."
                ),
                text_formatting=[
                    "Write naturally - handles punctuation well",
                    "Use ellipses (...) for trailing off or hesitation",
                    "Use dashes (--) for interruptions or abrupt stops",
                    "Spell out numbers for clearer pronunciation",
                ],
                emotion_tags={
                    "format": "[tag]",
                    "placement": "Inline where the sound should occur",
                    "available_tags": {
                        "[laugh]": "Full laugh, use for genuine amusement",
                        "[chuckle]": "Soft, brief laugh - more subtle",
                        "[cough]": "Clearing throat or actual cough",
                    },
                    "notes": [
                        "Tags produce actual sounds, not just inflection",
                        "Place tags where the sound naturally occurs",
                        "Don't overuse - one or two per paragraph maximum",
                    ],
                },
                voice_guidance={
                    "reference_audio_requirements": {
                        "duration": "10-30 seconds of clear speech",
                        "quality": "Clean audio without background noise",
                        "content": "Natural speech, not singing or whispering",
                        "format": "WAV or MP3, any sample rate",
                    },
                    "best_practices": [
                        "Use clear reference audio for best results",
                        "Avoid references with music or multiple speakers",
                        "One longer clip is often better than multiple short ones",
                    ],
                },
                tips=[
                    "Turbo is faster but has fewer tuning options than standard Chatterbox",
                    "Use standard Chatterbox if you need exaggeration/cfg_weight control",
                    "Ideal for real-time voice agent applications",
                    "Consistent reference audio maintains voice consistency",
                ],
                examples=[
                    {
                        "use_case": "Voice agent greeting",
                        "text": "Hi there! [chuckle] Thanks for calling. How can I help you today?",
                        "notes": "Simple, natural dialogue for voice agents",
                    },
                    {
                        "use_case": "Quick response",
                        "text": "Got it! I'll look that up for you right now.",
                        "notes": "Short, snappy responses work well",
                    },
                ],
            ),
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=8.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~4GB VRAM used. Faster than standard Chatterbox due to smaller model.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=3.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Good performance on Apple Silicon. Faster than standard Chatterbox.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.3,
                    device_type="cpu",
                    reference_hardware="AMD Ryzen 9 5900X",
                    notes="Slow but functional. GPU strongly recommended.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Chatterbox Turbo Setup (Fast Voice Cloning)

Chatterbox Turbo is a streamlined voice cloning model optimized for speed.

### Installation

```bash
pip install chatterbox-tts
```

Or with uv:
```bash
uv pip install chatterbox-tts
```

### Hardware Requirements

- **NVIDIA GPU with CUDA**: Best performance, recommended for production
- **Apple Silicon (MPS)**: Supported with some limitations
- **CPU**: Supported but slower

### Differences from Standard Chatterbox

| Feature | Turbo | Standard |
|---------|-------|----------|
| Parameters | 350M | 500M |
| Tuning | No cfg_weight/exaggeration | Full control |
| Speed | Faster | Slower |
| Use case | Production agents | Creative control |

### Usage Notes

- Simpler API - just provide reference audio and text
- No need to tune exaggeration or cfg_weight parameters
- Ideal for real-time voice agent applications
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        **kwargs,
    ) -> TTSResult:
        """Generate audio with Chatterbox Turbo voice cloning.

        Args:
            text: Text to synthesize. Can include tags like [laugh].
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio files.

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
                sample_rate=SAMPLE_RATE,
                error="Text cannot be empty",
            )

        if not reference_audio_paths:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="At least one reference audio path is required",
            )

        # Validate reference audio paths
        for ref_path in reference_audio_paths:
            if not os.path.exists(ref_path):
                return TTSResult(
                    status="error",
                    output_path=str(output_path),
                    duration_ms=0,
                    sample_rate=SAMPLE_RATE,
                    error=f"Reference audio file not found: {ref_path}",
                )

        try:
            model = _load_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load Chatterbox Turbo model: {e}",
            )

        primary_ref_audio = reference_audio_paths[0]

        # Convert angle-style tags to bracket-style for Chatterbox
        text = convert_angle_to_bracket_tags(text)

        memory_gb, memory_type = get_available_memory_gb()
        sample_rate = model.sr if hasattr(model, "sr") else SAMPLE_RATE

        try:
            # Redirect stdout to stderr during generation
            # to prevent library output from breaking MCP JSON protocol
            with redirect_stdout_to_stderr():
                # Check if text needs chunking
                if len(text) <= MAX_CHUNK_CHARS:
                    wav = model.generate(
                        text,
                        audio_prompt_path=primary_ref_audio,
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
                        )
                        audio_chunks.append(chunk_wav)

                        if memory_type == "cuda":
                            torch.cuda.empty_cache()

                    wav = _concatenate_audio_tensors(audio_chunks, sample_rate, silence_ms=100)
                    chunks_used = len(chunks)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ta.save(str(output_path), wav, sample_rate)

            duration_ms = int(wav.shape[-1] / sample_rate * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                chunks_used=chunks_used,
                metadata={
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
                sample_rate=sample_rate if "sample_rate" in dir() else SAMPLE_RATE,
                error=f"Generation failed: {e}",
            )
