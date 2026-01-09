"""Faster-Whisper Transcription Engine - CTranslate2-optimized Whisper.

Faster-Whisper is a reimplementation of Whisper using CTranslate2, which is
a fast inference engine for Transformer models. It's up to 4x faster than
the original Whisper implementation with the same accuracy.

Key advantages:
- 4x faster inference than original Whisper
- Lower memory usage
- Batched inference support
- Word-level timestamps with VAD filtering
"""

import sys
import time
from pathlib import Path
from typing import Optional

from .base import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
    TranscriptionEngineInfo,
)

# Reuse utilities from TTS module
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tts.utils import get_best_device, redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Model sizes and their characteristics
MODEL_SIZES = {
    "tiny": {"params": "39M", "vram_gb": 1, "relative_speed": 32},
    "tiny.en": {"params": "39M", "vram_gb": 1, "relative_speed": 32, "english_only": True},
    "base": {"params": "74M", "vram_gb": 1, "relative_speed": 16},
    "base.en": {"params": "74M", "vram_gb": 1, "relative_speed": 16, "english_only": True},
    "small": {"params": "244M", "vram_gb": 2, "relative_speed": 6},
    "small.en": {"params": "244M", "vram_gb": 2, "relative_speed": 6, "english_only": True},
    "medium": {"params": "769M", "vram_gb": 5, "relative_speed": 2},
    "medium.en": {"params": "769M", "vram_gb": 5, "relative_speed": 2, "english_only": True},
    "large-v1": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v2": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v3": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v3-turbo": {"params": "809M", "vram_gb": 6, "relative_speed": 8},
    "distil-large-v2": {"params": "756M", "vram_gb": 4, "relative_speed": 6},
    "distil-large-v3": {"params": "756M", "vram_gb": 4, "relative_speed": 6},
}

# Default model size
DEFAULT_MODEL_SIZE = "base"

# Supported languages (same as Whisper)
SUPPORTED_LANGUAGES = [
    "en",
    "zh",
    "de",
    "es",
    "ru",
    "ko",
    "fr",
    "ja",
    "pt",
    "tr",
    "pl",
    "ca",
    "nl",
    "ar",
    "sv",
    "it",
    "id",
    "hi",
    "fi",
    "vi",
    "he",
    "uk",
    "el",
    "ms",
    "cs",
    "ro",
    "da",
    "hu",
    "ta",
    "no",
    "th",
    "ur",
    "hr",
    "bg",
    "lt",
    "la",
    "mi",
    "ml",
    "cy",
    "sk",
    "te",
    "fa",
    "lv",
    "bn",
    "sr",
    "az",
    "sl",
    "kn",
    "et",
    "mk",
    "br",
    "eu",
    "is",
    "hy",
    "ne",
    "mn",
    "bs",
    "kk",
    "sq",
    "sw",
    "gl",
    "mr",
    "pa",
    "si",
    "km",
    "sn",
    "yo",
    "so",
    "af",
    "oc",
    "ka",
    "be",
    "tg",
    "sd",
    "gu",
    "am",
    "yi",
    "lo",
    "uz",
    "fo",
    "ht",
    "ps",
    "tk",
    "nn",
    "mt",
    "sa",
    "lb",
    "my",
    "bo",
    "tl",
    "mg",
    "as",
    "tt",
    "haw",
    "ln",
    "ha",
    "ba",
    "jw",
    "su",
]


# ============================================================================
# Model Management
# ============================================================================

_models: dict = {}  # Cache loaded models by size


def _get_compute_type(device: str) -> str:
    """Determine optimal compute type for device."""
    if device == "cuda":
        return "float16"
    elif device == "mps":
        # MPS doesn't support int8 quantization well
        return "float32"
    else:
        return "int8"  # CPU benefits from int8 quantization


def _load_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Lazily load Faster-Whisper model."""
    global _models

    if model_size in _models:
        return _models[model_size]

    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_SIZES.keys())}")

    device, device_name, _ = get_best_device()
    compute_type = _get_compute_type(device)

    print(
        f"Loading Faster-Whisper {model_size} on {device} ({device_name}) "
        f"with compute_type={compute_type}...",
        file=sys.stderr,
        flush=True,
    )

    with redirect_stdout_to_stderr():
        from faster_whisper import WhisperModel

        # For MPS, we need to use CPU as faster-whisper doesn't support MPS directly
        # But it's still faster than transformers due to CTranslate2 optimizations
        if device == "mps":
            actual_device = "cpu"
            print(
                "Note: Faster-Whisper doesn't support MPS directly, using CPU with optimizations",
                file=sys.stderr,
                flush=True,
            )
        else:
            actual_device = device

        model = WhisperModel(
            model_size,
            device=actual_device,
            compute_type=compute_type,
        )

        _models[model_size] = model

    print(f"Faster-Whisper {model_size} model loaded successfully", file=sys.stderr, flush=True)
    return model


# ============================================================================
# Engine Implementation
# ============================================================================


class FasterWhisperEngine(TranscriptionEngine):
    """Faster-Whisper Transcription Engine - CTranslate2-optimized Whisper.

    Up to 4x faster than the original Whisper implementation with the same accuracy.
    Uses CTranslate2 for efficient inference with support for quantization.

    Features:
    - 4x faster than original Whisper
    - Word-level timestamps
    - Voice Activity Detection (VAD) filtering
    - Batched inference support
    - Lower memory usage through quantization
    """

    @property
    def name(self) -> str:
        return "Faster-Whisper"

    @property
    def engine_id(self) -> str:
        return "faster_whisper"

    def is_available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> TranscriptionEngineInfo:
        return TranscriptionEngineInfo(
            name=self.name,
            engine_id=self.engine_id,
            description=(
                "CTranslate2-optimized Whisper implementation. "
                "Up to 4x faster than original Whisper with the same accuracy. "
                "Supports word-level timestamps and VAD filtering."
            ),
            requirements="faster-whisper (pip install faster-whisper)",
            supported_languages=SUPPORTED_LANGUAGES,
            supports_word_timestamps=True,
            supports_language_detection=True,
            model_sizes=list(MODEL_SIZES.keys()),
            default_model_size=DEFAULT_MODEL_SIZE,
            extra_info={
                "license": "MIT",
                "models": MODEL_SIZES,
                "speed_improvement": "4x faster than original Whisper",
                "parameters": {
                    "model_size": {
                        "type": "str",
                        "default": DEFAULT_MODEL_SIZE,
                        "options": list(MODEL_SIZES.keys()),
                        "description": "Model size - larger = more accurate but slower",
                    },
                    "language": {
                        "type": "str",
                        "default": None,
                        "description": "Language code (e.g., 'en'). Auto-detected if not specified.",
                    },
                    "word_timestamps": {
                        "type": "bool",
                        "default": False,
                        "description": "Enable word-level timestamps",
                    },
                    "vad_filter": {
                        "type": "bool",
                        "default": True,
                        "description": "Filter out silence using VAD",
                    },
                    "beam_size": {
                        "type": "int",
                        "default": 5,
                        "description": "Beam size for decoding",
                    },
                },
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Faster-Whisper Setup

Faster-Whisper is a CTranslate2-optimized implementation of Whisper.
It's up to 4x faster than the original with the same accuracy.

### Installation

```bash
pip install faster-whisper
```

Or install with talky-talky:
```bash
pip install talky-talky[faster-whisper]
```

### Hardware Requirements

- **NVIDIA GPU with CUDA**: Best performance (float16 compute)
- **CPU**: Good performance with int8 quantization
- **Apple Silicon**: Uses CPU (MPS not directly supported by CTranslate2)

### Model Sizes

| Model | Parameters | VRAM | Speed vs large |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |
| distil-large-v3 | 756M | ~4GB | 6x |

### English-Only Models

For English transcription, use the .en models (e.g., 'base.en') for
slightly better accuracy and speed.

### Key Features

- **VAD Filtering**: Automatically removes silence
- **Word Timestamps**: Get timing for each word
- **Batched Inference**: Process multiple audio files efficiently
- **Quantization**: Lower memory usage on CPU
"""

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        model_size: str = DEFAULT_MODEL_SIZE,
        word_timestamps: bool = False,
        vad_filter: bool = True,
        beam_size: int = 5,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using Faster-Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en'). Auto-detect if None.
            model_size: Model size to use.
            word_timestamps: Enable word-level timestamps.
            vad_filter: Filter out silence using Voice Activity Detection.
            beam_size: Beam size for decoding (default: 5).

        Returns:
            TranscriptionResult with transcribed text and metadata.
        """
        audio_path = Path(audio_path)
        start_time = time.time()

        # Validate audio file
        if not audio_path.exists():
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Audio file not found: {audio_path}",
            )

        # Validate language if provided
        if language and language not in SUPPORTED_LANGUAGES:
            # Check for English-only model
            model_info = MODEL_SIZES.get(model_size, {})
            if model_info.get("english_only") and language != "en":
                return TranscriptionResult(
                    status="error",
                    text="",
                    error=f"Model {model_size} only supports English, but language '{language}' was specified",
                )
            if language not in SUPPORTED_LANGUAGES:
                return TranscriptionResult(
                    status="error",
                    text="",
                    error=f"Unsupported language: {language}",
                )

        try:
            model = _load_model(model_size)
        except Exception as e:
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Failed to load Faster-Whisper model: {e}",
            )

        try:
            # Run transcription
            with redirect_stdout_to_stderr():
                segments_iter, info = model.transcribe(
                    str(audio_path),
                    language=language,
                    beam_size=beam_size,
                    word_timestamps=word_timestamps,
                    vad_filter=vad_filter,
                )

                # Collect all segments (generator)
                raw_segments = list(segments_iter)

            processing_time = int((time.time() - start_time) * 1000)

            # Build result
            full_text_parts = []
            segments = []

            for seg in raw_segments:
                full_text_parts.append(seg.text.strip())

                # Build word segments if available
                words = []
                if word_timestamps and seg.words:
                    for word in seg.words:
                        words.append(
                            WordSegment(
                                word=word.word,
                                start=word.start,
                                end=word.end,
                                confidence=word.probability
                                if hasattr(word, "probability")
                                else None,
                            )
                        )

                segment = TranscriptionSegment(
                    text=seg.text.strip(),
                    start=seg.start,
                    end=seg.end,
                    words=words,
                    confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else None,
                )
                segments.append(segment)

            full_text = " ".join(full_text_parts)

            return TranscriptionResult(
                status="success",
                text=full_text,
                segments=segments,
                language=info.language,
                language_probability=info.language_probability,
                duration_seconds=info.duration,
                processing_time_ms=processing_time,
                metadata={
                    "model_size": model_size,
                    "vad_filter": vad_filter,
                    "word_timestamps": word_timestamps,
                    "beam_size": beam_size,
                },
            )

        except Exception as e:
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Transcription failed: {e}",
            )
