"""MiraTTS Engine - Fast audio-prompted TTS.

MiraTTS generates high-quality 48kHz speech using reference audio for voice cloning.
Uses transformers for cross-platform compatibility (CUDA, MPS, CPU).
"""

import os
import sys
from pathlib import Path

from .base import AudioPromptedEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import (
    split_text_into_chunks,
    get_best_device,
    get_available_memory_gb,
    redirect_stdout_to_stderr,
)


# ============================================================================
# Constants
# ============================================================================

# Audio settings
SAMPLE_RATE = 48000  # MiraTTS outputs high-quality 48kHz audio

# Generation limits
MAX_DURATION_SECS = 60
MAX_CHUNK_CHARS = 500

# Model identifier
MODEL_ID = "YatharthS/MiraTTS"


# ============================================================================
# Model Management
# ============================================================================

_model = None
_codec = None


class MiraTTSTransformers:
    """MiraTTS wrapper using transformers instead of lmdeploy.

    This provides cross-platform compatibility (CUDA, MPS, CPU) at the cost
    of some speed compared to the lmdeploy-based implementation.
    """

    def __init__(self, model_id: str = MODEL_ID):
        device, device_name, _ = get_best_device()
        self.device = device

        print(f"Loading MiraTTS model on {device} ({device_name})...", file=sys.stderr, flush=True)

        # Redirect stdout to stderr during import and model loading
        # to prevent library output from breaking MCP JSON protocol
        with redirect_stdout_to_stderr():
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from ncodec.codec import TTSCodec

            # Load model with appropriate dtype
            if device == "cuda":
                dtype = torch.bfloat16
            elif device == "mps":
                dtype = torch.float16  # MPS works better with float16
            else:
                dtype = torch.float32

            # Load model - device_map only works for CUDA
            if device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # For MPS and CPU, load without device_map then move
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                self.model = self.model.to(device)

            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.codec = TTSCodec()

        # Generation config matching original MiraTTS
        self.gen_config = {
            "top_p": 0.95,
            "top_k": 50,
            "temperature": 0.8,
            "max_new_tokens": 1024,
            "repetition_penalty": 1.2,
            "do_sample": True,
        }

        print("MiraTTS model loaded successfully", file=sys.stderr, flush=True)

    def encode_audio(self, audio_file: str):
        """Encodes audio into context tokens."""
        return self.codec.encode(audio_file)

    def generate(self, text: str, context_tokens):
        """Generates speech from input text."""
        import torch

        # Format prompt using codec
        formatted_prompt = self.codec.format_prompt(text, context_tokens, None)

        # Tokenize - the prompt is already formatted as a string
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.gen_config,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the generated tokens (skip input tokens)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Convert to audio using codec
        audio = self.codec.decode(generated_text, context_tokens)
        return audio


def _load_model():
    """Lazily load MiraTTS model."""
    global _model

    if _model is not None:
        return _model

    # Try lmdeploy first (faster), fall back to transformers
    try:
        from mira.model import MiraTTS

        device, device_name, _ = get_best_device()
        print(f"Loading MiraTTS (lmdeploy) on {device}...", file=sys.stderr, flush=True)
        _model = MiraTTS(MODEL_ID)
        print("MiraTTS model loaded successfully (lmdeploy)", file=sys.stderr, flush=True)
    except ImportError:
        # Fall back to transformers-based implementation
        _model = MiraTTSTransformers(MODEL_ID)

    return _model


def _concatenate_audio_arrays(audio_arrays: list, sample_rate: int, silence_ms: int = 100):
    """Concatenate audio arrays with silence between them."""
    import numpy as np

    if not audio_arrays:
        raise ValueError("No audio arrays to concatenate")

    if len(audio_arrays) == 1:
        return audio_arrays[0]

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    result_parts = []
    for i, audio in enumerate(audio_arrays):
        result_parts.append(audio)
        if i < len(audio_arrays) - 1:
            result_parts.append(silence)

    return np.concatenate(result_parts)


# ============================================================================
# Engine Implementation
# ============================================================================


class MiraEngine(AudioPromptedEngine):
    """MiraTTS Engine - Fast audio-prompted voice cloning.

    Clones voices from reference audio with high quality 48kHz output.
    Optimized for speed: over 100x realtime with only 6GB VRAM.

    Parameters:
        reference_audio_paths (list[str]): Paths to reference audio files.
            At least one required. Clear speech samples work best.
    """

    @property
    def name(self) -> str:
        return "MiraTTS"

    @property
    def engine_id(self) -> str:
        return "mira"

    def is_available(self) -> bool:
        try:
            import torch

            # ncodec (FastBiCodec) hardcodes CUDA - it won't work without it
            if not torch.cuda.is_available():
                return False
            # Check for codec (required)
            from ncodec.codec import TTSCodec  # noqa: F401

            # Check for transformers (fallback) or lmdeploy (fast)
            try:
                from mira.model import MiraTTS  # noqa: F401
            except ImportError:
                from transformers import AutoModelForCausalLM  # noqa: F401
            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",
            description="Fast voice cloning with high-quality 48kHz output",
            requirements="mira-tts (pip install git+https://github.com/ysharma3501/MiraTTS.git)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            extra_info={
                "license": "MIT",
                "license_url": "https://huggingface.co/YatharthS/MiraTTS",
                "model_id": MODEL_ID,
                "reference_audio": "Clear speech samples, any duration",
                "performance": "100x+ realtime with batching",
                "vram_requirement": "6GB NVIDIA GPU (CUDA required)",
                "cuda_required": True,
                "mps_supported": False,
                "cpu_supported": False,
                "parameters": {
                    "reference_audio_paths": {
                        "type": "list[str]",
                        "required": True,
                        "description": "Paths to reference audio files for voice cloning",
                    },
                },
            },
            prompting_guide=PromptingGuide(
                overview=(
                    "MiraTTS is a fast, high-quality voice cloning model. It generates "
                    "48kHz audio (higher quality than most TTS models) from reference "
                    "audio samples. Optimized for speed with over 100x realtime performance."
                ),
                text_formatting=[
                    "Write naturally - MiraTTS handles punctuation well",
                    "Use ellipses (...) for pauses or trailing off",
                    "Spell out numbers for clearer pronunciation",
                    "MiraTTS does not support emotion tags",
                ],
                emotion_tags={},
                voice_guidance={
                    "reference_audio_requirements": {
                        "duration": "Any duration, but 10-30 seconds recommended",
                        "quality": "Clean audio without background noise",
                        "content": "Natural speech works best",
                        "format": "WAV, MP3, OGG, or any librosa-compatible format",
                    },
                    "best_practices": [
                        "Use clear, clean reference audio for best results",
                        "Reference audio quality directly affects output quality",
                        "Multiple reference files can be used for consistency",
                    ],
                },
                parameters={
                    "reference_audio_paths": {
                        "description": "Paths to reference audio files for voice cloning",
                        "type": "list[str]",
                        "required": True,
                    },
                },
                tips=[
                    "MiraTTS is optimized for speed - great for batch processing",
                    "48kHz output is higher quality than most TTS models (typically 24kHz)",
                    "Works well with only 6GB VRAM",
                    "No emotion tags supported - use voice quality from reference audio",
                ],
                examples=[
                    {
                        "use_case": "Standard narration",
                        "text": "The quick brown fox jumps over the lazy dog.",
                        "notes": "Simple, clean text works best",
                    },
                    {
                        "use_case": "Dialogue",
                        "text": "Well, I suppose you're right about that. Let me think...",
                        "notes": "Ellipses create natural pauses",
                    },
                ],
            ),
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=100.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~6GB VRAM used. 100x+ realtime with batching. CUDA required.",
                ),
                # MPS and CPU not supported - ncodec requires CUDA
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## MiraTTS Setup (Fast Voice Cloning)

MiraTTS is a fast, high-quality voice cloning model with 48kHz output.

**IMPORTANT: Requires NVIDIA GPU with CUDA.** The audio codec (ncodec) hardcodes
CUDA device. MPS/CPU are not supported.

### Installation

Install with the mira extra:
```bash
pip install talky-talky[mira]
```

Or with uv:
```bash
uv pip install talky-talky[mira]
```

### Hardware Requirements

- **NVIDIA GPU with CUDA: Required** (6GB+ VRAM)
- Apple Silicon (MPS): NOT supported (ncodec requires CUDA)
- CPU: NOT supported (ncodec requires CUDA)

### Key Features

- 48kHz high-quality audio output
- Over 100x realtime with batching
- Works with only 6GB VRAM
- Low latency (~100ms)
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        **kwargs,
    ) -> TTSResult:
        """Generate audio with MiraTTS voice cloning.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio files.

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
                error=f"Failed to load MiraTTS model: {e}",
            )

        primary_ref_audio = reference_audio_paths[0]

        try:
            # Encode reference audio to context tokens
            context_tokens = model.encode_audio(primary_ref_audio)

            # Check if text needs chunking
            if len(text) <= MAX_CHUNK_CHARS:
                audio = model.generate(text, context_tokens)
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
                    chunk_audio = model.generate(chunk, context_tokens)
                    audio_chunks.append(chunk_audio)

                audio = _concatenate_audio_arrays(audio_chunks, SAMPLE_RATE, silence_ms=100)
                chunks_used = len(chunks)

            # Ensure audio is numpy array
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Handle shape - MiraTTS may return (samples,) or (1, samples)
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, SAMPLE_RATE)

            duration_ms = int(len(audio) / SAMPLE_RATE * 1000)

            memory_gb, memory_type = get_available_memory_gb()

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
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
                sample_rate=SAMPLE_RATE,
                error=f"Generation failed: {e}",
            )
