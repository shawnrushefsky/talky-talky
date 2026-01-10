"""XTTS-v2 Engine - Multilingual audio-prompted TTS.

XTTS-v2 from Coqui generates multilingual speech using reference audio for voice cloning.
Supports 17 languages with just 6 seconds of reference audio.
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
SAMPLE_RATE = 24000  # XTTS-v2 outputs 24kHz audio

# Generation limits
MAX_DURATION_SECS = 30  # XTTS-v2 generates up to ~30 seconds per chunk
MAX_CHUNK_CHARS = 400  # Conservative limit for reliable generation

# Model identifier
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
    "hi": "Hindi",
}


# ============================================================================
# Model Management
# ============================================================================

_model = None


def _load_model():
    """Lazily load XTTS-v2 model."""
    global _model

    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading XTTS-v2 model on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        # Workaround for PyTorch 2.6+ weights_only security change
        # TTS library uses pickle-based weights - temporarily patch torch.load
        import torch
        from functools import wraps

        _original_torch_load = torch.load

        @wraps(_original_torch_load)
        def _patched_load(*args, **kwargs):
            # Force weights_only=False for TTS model loading
            kwargs.setdefault("weights_only", False)
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_load

        try:
            from TTS.api import TTS

            # Initialize TTS with GPU flag
            # TTS uses "gpu=True" for CUDA, but we need to handle MPS separately
            use_gpu = device == "cuda"
            _model = TTS(MODEL_NAME, gpu=use_gpu)
        finally:
            # Restore original torch.load
            torch.load = _original_torch_load

        # For MPS, manually move the model to the device after loading
        if device == "mps":
            try:
                if hasattr(_model, "synthesizer") and _model.synthesizer is not None:
                    if hasattr(_model.synthesizer, "tts_model"):
                        _model.synthesizer.tts_model = _model.synthesizer.tts_model.to(device)
                        print("Moved XTTS-v2 model to MPS device", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Warning: Could not move model to MPS: {e}", file=sys.stderr, flush=True)

    print("XTTS-v2 model loaded successfully", file=sys.stderr, flush=True)
    return _model


def _concatenate_audio_lists(audio_lists: list, sample_rate: int, silence_ms: int = 100):
    """Concatenate audio lists with silence between them."""
    import numpy as np

    if not audio_lists:
        raise ValueError("No audio lists to concatenate")

    if len(audio_lists) == 1:
        return audio_lists[0]

    silence_samples = int(sample_rate * silence_ms / 1000)
    silence = [0.0] * silence_samples

    result_parts = []
    for i, audio in enumerate(audio_lists):
        # Convert to list if numpy array
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        result_parts.extend(audio)
        if i < len(audio_lists) - 1:
            result_parts.extend(silence)

    return np.array(result_parts, dtype=np.float32)


# ============================================================================
# Engine Implementation
# ============================================================================


class XTTSEngine(AudioPromptedEngine):
    """XTTS-v2 Engine - Multilingual audio-prompted voice cloning.

    Clones voices from reference audio samples with support for 17 languages.
    Only needs ~6 seconds of reference audio for good quality cloning.

    Parameters:
        reference_audio_paths (list[str]): Paths to reference audio files.
            At least one required. 6+ seconds of clear speech recommended.
        language (str): Target language code (default: "en").
            Supported: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi
    """

    @property
    def name(self) -> str:
        return "XTTS-v2"

    @property
    def engine_id(self) -> str:
        return "xtts"

    def is_available(self) -> bool:
        try:
            from TTS.api import TTS  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",
            description="Multilingual voice cloning with 17 language support",
            requirements="TTS (pip install TTS)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            extra_info={
                "license": "CPML (Coqui Public Model License)",
                "license_url": "https://coqui.ai/cpml",
                "model_name": MODEL_NAME,
                "reference_audio": "6+ seconds of clear speech recommended",
                "multilingual": True,
                "supported_languages": SUPPORTED_LANGUAGES,
                "cross_language_cloning": True,
                "parameters": {
                    "reference_audio_paths": {
                        "type": "list[str]",
                        "required": True,
                        "description": "Paths to reference audio files for voice cloning",
                    },
                    "language": {
                        "type": "str",
                        "default": "en",
                        "description": "Target language code",
                        "options": list(SUPPORTED_LANGUAGES.keys()),
                    },
                },
            },
            prompting_guide=PromptingGuide(
                overview=(
                    "XTTS-v2 is a multilingual voice cloning model that supports 17 languages. "
                    "It only requires about 6 seconds of reference audio to clone a voice. "
                    "Unique feature: cross-language voice cloning - clone a voice in one language "
                    "and generate speech in another language while preserving the voice."
                ),
                text_formatting=[
                    "Write naturally - XTTS-v2 handles punctuation well",
                    "Use ellipses (...) for pauses",
                    "Spell out numbers for clearer pronunciation",
                    "For non-English text, ensure proper character encoding",
                    "XTTS-v2 does not support emotion tags",
                ],
                emotion_tags={},
                voice_guidance={
                    "reference_audio_requirements": {
                        "duration": "6-30 seconds of clear speech",
                        "quality": "Clean audio without background noise",
                        "content": "Natural speech, single speaker",
                        "format": "WAV, MP3, or any common audio format",
                    },
                    "best_practices": [
                        "Use clear, clean reference audio for best results",
                        "6 seconds is the minimum, but more can improve quality",
                        "Reference audio language doesn't have to match output language",
                        "Avoid reference audio with music or multiple speakers",
                    ],
                    "cross_language_notes": [
                        "You can clone a voice from English audio and output in Japanese",
                        "Voice characteristics transfer across languages",
                        "Some accents may carry over into other languages",
                    ],
                },
                parameters={
                    "reference_audio_paths": {
                        "description": "Paths to reference audio files for voice cloning",
                        "type": "list[str]",
                        "required": True,
                    },
                    "language": {
                        "description": "Target language code for output speech",
                        "type": "str",
                        "default": "en",
                        "options": SUPPORTED_LANGUAGES,
                    },
                },
                tips=[
                    "XTTS-v2 is great for multilingual projects",
                    "Cross-language cloning is a unique feature - use it!",
                    "Quality improves with longer reference audio (up to ~30 seconds)",
                    "For best results, use reference audio in a quiet environment",
                    "The model handles accents well across languages",
                ],
                examples=[
                    {
                        "use_case": "English narration",
                        "text": "The quick brown fox jumps over the lazy dog.",
                        "language": "en",
                        "notes": "Standard English generation",
                    },
                    {
                        "use_case": "Spanish dialogue",
                        "text": "Buenos días, ¿cómo estás hoy?",
                        "language": "es",
                        "notes": "Spanish with proper accents",
                    },
                    {
                        "use_case": "Cross-language cloning",
                        "text": "こんにちは、元気ですか?",
                        "language": "ja",
                        "notes": "Clone English voice to speak Japanese",
                    },
                ],
            ),
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=3.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~6GB VRAM used. First generation slower due to model loading.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=1.5,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="MPS support may require manual device handling.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.3,
                    device_type="cpu",
                    reference_hardware="AMD Ryzen 9 5900X",
                    notes="Slow but functional. GPU recommended for production use.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## XTTS-v2 Setup (Multilingual Voice Cloning)

XTTS-v2 from Coqui is a multilingual voice cloning model supporting 17 languages.
Only requires ~6 seconds of reference audio.

### Installation

```bash
pip install TTS
```

Or with uv:
```bash
uv pip install TTS
```

Or install with talky-talky:
```bash
pip install talky-talky[xtts]
```

### Hardware Requirements

- **NVIDIA GPU with CUDA**: Best performance
- **Apple Silicon (MPS)**: Supported (may need manual device handling)
- **CPU**: Supported but slower

### Supported Languages (17)

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish,
Russian, Dutch, Czech, Arabic, Chinese, Japanese, Hungarian, Korean, Hindi

### Key Features

- Voice cloning with only 6 seconds of audio
- Cross-language voice cloning (clone voice in one language, output in another)
- High quality multilingual synthesis

### First Run

The model (~6GB) will be downloaded automatically on first use.
This may take a few minutes depending on your connection.
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        language: str = "en",
        **kwargs,
    ) -> TTSResult:
        """Generate audio with XTTS-v2 voice cloning.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio files.
            language: Target language code (default: "en").

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

        # Validate language
        if language not in SUPPORTED_LANGUAGES:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}",
            )

        try:
            model = _load_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load XTTS-v2 model: {e}",
            )

        primary_ref_audio = reference_audio_paths[0]

        try:
            # Check if text needs chunking
            if len(text) <= MAX_CHUNK_CHARS:
                audio = model.tts(
                    text=text,
                    speaker_wav=primary_ref_audio,
                    language=language,
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
                    chunk_audio = model.tts(
                        text=chunk,
                        speaker_wav=primary_ref_audio,
                        language=language,
                    )
                    audio_chunks.append(chunk_audio)

                audio = _concatenate_audio_lists(audio_chunks, SAMPLE_RATE, silence_ms=100)
                chunks_used = len(chunks)

            # Ensure audio is numpy array
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            elif hasattr(audio, "numpy"):
                audio = audio.numpy()
            audio = np.asarray(audio, dtype=np.float32)

            # Handle shape
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
                    "language": language,
                    "language_name": SUPPORTED_LANGUAGES[language],
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
