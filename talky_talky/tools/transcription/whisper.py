"""Whisper Transcription Engine - OpenAI's robust speech recognition.

Whisper is a general-purpose speech recognition model trained on 680,000 hours of
multilingual and multitask supervised data. It supports transcription and translation
for 99+ languages.

This implementation uses the transformers library for excellent cross-platform support.
"""

import sys
import time
from pathlib import Path
from typing import Optional

from .base import (
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionEngineInfo,
)

# Reuse utilities from TTS module
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tts.utils import get_best_device, redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Model sizes and their approximate VRAM requirements
MODEL_SIZES = {
    "tiny": {"params": "39M", "vram_gb": 1, "relative_speed": 32},
    "base": {"params": "74M", "vram_gb": 1, "relative_speed": 16},
    "small": {"params": "244M", "vram_gb": 2, "relative_speed": 6},
    "medium": {"params": "769M", "vram_gb": 5, "relative_speed": 2},
    "large": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v2": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v3": {"params": "1550M", "vram_gb": 10, "relative_speed": 1},
    "large-v3-turbo": {"params": "809M", "vram_gb": 6, "relative_speed": 8},
}

# HuggingFace model IDs
MODEL_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}

# Default model size
DEFAULT_MODEL_SIZE = "base"

# Supported languages (ISO 639-1 codes)
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


def _load_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Lazily load Whisper model."""
    global _models

    if model_size in _models:
        return _models[model_size]

    if model_size not in MODEL_IDS:
        raise ValueError(f"Unknown model size: {model_size}. Available: {list(MODEL_IDS.keys())}")

    model_id = MODEL_IDS[model_size]
    device, device_name, _ = get_best_device()

    print(
        f"Loading Whisper {model_size} model on {device} ({device_name})...",
        file=sys.stderr,
        flush=True,
    )

    with redirect_stdout_to_stderr():
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        # Determine torch dtype based on device
        if device == "cuda":
            torch_dtype = torch.float16
        elif device == "mps":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        # Load model and processor
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )

        _models[model_size] = pipe

    print(f"Whisper {model_size} model loaded successfully", file=sys.stderr, flush=True)
    return pipe


# ============================================================================
# Engine Implementation
# ============================================================================


class WhisperEngine(TranscriptionEngine):
    """Whisper Transcription Engine - OpenAI's robust speech recognition.

    Uses the transformers library for excellent cross-platform support (CUDA, MPS, CPU).
    Supports 99+ languages with automatic language detection.

    Model sizes (trade-off between speed and accuracy):
    - tiny: Fastest, least accurate
    - base: Good balance for most uses (default)
    - small: Better accuracy
    - medium: High accuracy
    - large-v3: Best accuracy
    - large-v3-turbo: Best balance of speed and accuracy (recommended for production)
    """

    @property
    def name(self) -> str:
        return "Whisper"

    @property
    def engine_id(self) -> str:
        return "whisper"

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> TranscriptionEngineInfo:
        return TranscriptionEngineInfo(
            name=self.name,
            engine_id=self.engine_id,
            description=(
                "OpenAI's Whisper speech recognition model via transformers. "
                "Supports 99+ languages with automatic detection. "
                "Best accuracy among open-source models."
            ),
            requirements="transformers, torch (pip install transformers torch)",
            supported_languages=SUPPORTED_LANGUAGES,
            supports_word_timestamps=True,
            supports_language_detection=True,
            model_sizes=list(MODEL_SIZES.keys()),
            default_model_size=DEFAULT_MODEL_SIZE,
            extra_info={
                "license": "MIT",
                "models": MODEL_SIZES,
                "model_ids": MODEL_IDS,
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
                    "return_timestamps": {
                        "type": "bool or str",
                        "default": True,
                        "description": "Return timestamps. Use 'word' for word-level.",
                    },
                },
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Whisper Setup (via Transformers)

Whisper is OpenAI's state-of-the-art speech recognition model.
This implementation uses the transformers library for broad compatibility.

### Installation

```bash
pip install transformers torch
```

Or install with talky-talky:
```bash
pip install talky-talky[whisper]
```

### Hardware Requirements

- **NVIDIA GPU with CUDA**: Best performance
- **Apple Silicon (MPS)**: Good performance
- **CPU**: Supported but slower

### Model Sizes

| Model | Parameters | VRAM | Relative Speed |
|-------|-----------|------|----------------|
| tiny | 39M | ~1GB | 32x |
| base | 74M | ~1GB | 16x |
| small | 244M | ~2GB | 6x |
| medium | 769M | ~5GB | 2x |
| large-v3 | 1550M | ~10GB | 1x |
| large-v3-turbo | 809M | ~6GB | 8x |

### Recommended Models

- **Development/Testing**: base (fast, decent accuracy)
- **Production**: large-v3-turbo (best speed/accuracy balance)
- **Maximum Accuracy**: large-v3

### First Run

Models are downloaded automatically on first use from HuggingFace.
"""

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        model_size: str = DEFAULT_MODEL_SIZE,
        return_timestamps: bool | str = True,
        **kwargs,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper.

        Args:
            audio_path: Path to the audio file.
            language: Language code (e.g., 'en'). Auto-detect if None.
            model_size: Model size to use (tiny, base, small, medium, large-v3, etc.)
            return_timestamps: True for segment timestamps, 'word' for word-level.

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
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES[:20]}...",
            )

        try:
            pipe = _load_model(model_size)
        except Exception as e:
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Failed to load Whisper model: {e}",
            )

        try:
            # Prepare generation kwargs
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language

            # Run transcription
            with redirect_stdout_to_stderr():
                result = pipe(
                    str(audio_path),
                    return_timestamps=return_timestamps,
                    generate_kwargs=generate_kwargs,
                )

            processing_time = int((time.time() - start_time) * 1000)

            # Parse result
            text = result.get("text", "").strip()
            chunks = result.get("chunks", [])

            # Build segments
            segments = []
            for chunk in chunks:
                timestamp = chunk.get("timestamp", (None, None))
                start = timestamp[0] if timestamp[0] is not None else 0.0
                end = timestamp[1] if timestamp[1] is not None else start

                segment = TranscriptionSegment(
                    text=chunk.get("text", "").strip(),
                    start=start,
                    end=end,
                )
                segments.append(segment)

            # Get audio duration from the last segment
            duration = segments[-1].end if segments else None

            return TranscriptionResult(
                status="success",
                text=text,
                segments=segments,
                language=language,  # Note: transformers pipeline doesn't return detected language
                duration_seconds=duration,
                processing_time_ms=processing_time,
                metadata={
                    "model_size": model_size,
                    "model_id": MODEL_IDS[model_size],
                    "return_timestamps": return_timestamps,
                },
            )

        except Exception as e:
            return TranscriptionResult(
                status="error",
                text="",
                error=f"Transcription failed: {e}",
            )
