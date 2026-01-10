"""VibeVoice TTS Engine - Real-time and long-form speech synthesis.

VibeVoice is Microsoft's open-source TTS system with two variants:
- Realtime-0.5B: Single speaker, ~300ms latency, up to 10 minutes
- 1.5B: Multi-speaker (up to 4), up to 90 minutes, long-form conversations

Both models support English; 1.5B also supports Chinese.
"""

import sys
from pathlib import Path
from typing import Optional

from .base import TextPromptedEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import split_text_into_chunks, get_best_device, redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Model identifiers
MODEL_ID_REALTIME = "microsoft/VibeVoice-Realtime-0.5B"
MODEL_ID_LONGFORM = "microsoft/VibeVoice-1.5B"

# Audio settings
SAMPLE_RATE = 24000  # Based on tokenizer description

# Generation limits
MAX_CHUNK_CHARS_REALTIME = 500
MAX_CHUNK_CHARS_LONGFORM = 2000
MAX_DURATION_SECS_REALTIME = 600  # 10 minutes
MAX_DURATION_SECS_LONGFORM = 5400  # 90 minutes

# Default speaker
DEFAULT_SPEAKER = "Carter"

# Available experimental speakers (downloaded via download_experimental_voices.sh)
EXPERIMENTAL_SPEAKERS = [
    "Carter",
    "Emily",
    "Nova",
    "Michael",
    "Sarah",
]

# ============================================================================
# Model Management
# ============================================================================

_realtime_model = None
_longform_model = None


def _load_realtime_model():
    """Lazily load VibeVoice Realtime 0.5B model."""
    global _realtime_model

    if _realtime_model is not None:
        return _realtime_model

    device, device_name, _ = get_best_device()
    print(f"Loading VibeVoice Realtime on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        try:
            from vibevoice import VibeVoiceRealtime
        except ImportError:
            raise ImportError(
                "VibeVoice not installed. Install with: pip install vibevoice\n"
                "Or clone: git clone https://github.com/microsoft/VibeVoice.git && pip install -e ."
            )

        _realtime_model = VibeVoiceRealtime(model_path=MODEL_ID_REALTIME, device=device)

    print("VibeVoice Realtime loaded successfully", file=sys.stderr, flush=True)
    return _realtime_model


def _load_longform_model():
    """Lazily load VibeVoice 1.5B model for long-form generation."""
    global _longform_model

    if _longform_model is not None:
        return _longform_model

    device, device_name, _ = get_best_device()
    print(f"Loading VibeVoice 1.5B on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        try:
            from vibevoice import VibeVoice
        except ImportError:
            raise ImportError(
                "VibeVoice not installed. Install with: pip install vibevoice\n"
                "Or clone: git clone https://github.com/microsoft/VibeVoice.git && pip install -e ."
            )

        _longform_model = VibeVoice(model_path=MODEL_ID_LONGFORM, device=device)

    print("VibeVoice 1.5B loaded successfully", file=sys.stderr, flush=True)
    return _longform_model


# ============================================================================
# Engine Implementation - Realtime 0.5B
# ============================================================================


class VibeVoiceRealtimeEngine(TextPromptedEngine):
    """VibeVoice Realtime 0.5B Engine - Fast single-speaker TTS.

    Generates speech with ~300ms first-audio latency.
    Supports streaming text input and up to 10 minutes of audio.

    Parameters:
        speaker_name (str): Name of the speaker voice to use (default: "Carter")
    """

    @property
    def name(self) -> str:
        return "VibeVoice Realtime"

    @property
    def engine_id(self) -> str:
        return "vibevoice_realtime"

    def is_available(self) -> bool:
        try:
            import vibevoice  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="text_prompted",
            description="Real-time TTS with ~300ms latency, single speaker, up to 10 min",
            requirements="vibevoice (pip install vibevoice or git clone)",
            max_duration_secs=MAX_DURATION_SECS_REALTIME,
            chunk_size_chars=MAX_CHUNK_CHARS_REALTIME,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            prompting_guide=PromptingGuide(
                overview=(
                    "VibeVoice Realtime generates high-quality speech with very low latency. "
                    "It uses a simple speaker name to select the voice. English is the primary "
                    "language; other languages may work but are experimental."
                ),
                text_formatting=[
                    "Write naturally - the model handles punctuation and pacing",
                    "Use proper punctuation for natural pauses",
                    "Avoid special characters, code, or formulas",
                    "Long texts are automatically chunked",
                ],
                voice_guidance={
                    "type": "speaker_name",
                    "description": "Select from available speaker voices",
                    "available_speakers": EXPERIMENTAL_SPEAKERS,
                    "default": DEFAULT_SPEAKER,
                },
                parameters={
                    "speaker_name": {
                        "type": "str",
                        "default": DEFAULT_SPEAKER,
                        "description": "Name of the speaker voice to use",
                    },
                },
                tips=[
                    "Best for English content",
                    "Single speaker only - use VibeVoice 1.5B for multi-speaker",
                    "Ideal for real-time applications needing low latency",
                    "Max ~10 minutes per generation",
                ],
            ),
            extra_info={
                "license": "MIT",
                "license_url": "https://github.com/microsoft/VibeVoice",
                "model_id": MODEL_ID_REALTIME,
                "parameters": "0.5B",
                "latency": "~300ms to first audio",
                "languages": ["en"],
                "experimental_languages": ["de", "fr", "it", "ja", "ko", "nl", "pl", "pt", "es"],
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=15.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~2-4GB VRAM used. ~300ms first-audio latency, then fast streaming.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=8.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Good performance on Apple Silicon for real-time use.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=1.5,
                    device_type="cpu",
                    reference_hardware="Intel i9-12900K",
                    notes="0.5B model runs reasonably well on CPU.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## VibeVoice Realtime Setup

VibeVoice requires the vibevoice package from Microsoft.

**Option 1: Install from PyPI (when available):**
```bash
pip install vibevoice
```

**Option 2: Install from source:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Hardware Requirements:**
- GPU recommended for best performance
- Works on CUDA, MPS, and CPU
- ~2-4GB VRAM for 0.5B model
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        voice_description: str = DEFAULT_SPEAKER,
        speaker_name: Optional[str] = None,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with VibeVoice Realtime.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            voice_description: Speaker name (for compatibility with TextPromptedEngine).
            speaker_name: Explicit speaker name (overrides voice_description).

        Returns:
            TTSResult with status and metadata.
        """
        import numpy as np
        import soundfile as sf

        output_path = Path(output_path)

        # Use speaker_name if provided, otherwise use voice_description as speaker name
        speaker = speaker_name or voice_description or DEFAULT_SPEAKER

        # Validate inputs
        if not text or not text.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Text cannot be empty",
            )

        try:
            model = _load_realtime_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load VibeVoice Realtime: {e}",
            )

        try:
            # Check if text needs chunking
            if len(text) <= MAX_CHUNK_CHARS_REALTIME:
                audio_np = model.generate(text=text, speaker_name=speaker)
                chunks_used = 1
            else:
                chunks = split_text_into_chunks(text, MAX_CHUNK_CHARS_REALTIME)
                print(
                    f"Splitting into {len(chunks)} chunks ({len(text)} chars)",
                    file=sys.stderr,
                    flush=True,
                )

                audio_chunks = []
                for i, chunk in enumerate(chunks):
                    print(f"  Chunk {i + 1}/{len(chunks)}...", file=sys.stderr, flush=True)
                    chunk_audio = model.generate(text=chunk, speaker_name=speaker)
                    audio_chunks.append(chunk_audio)

                # Concatenate with 100ms silence between chunks
                silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
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
                    "speaker_name": speaker,
                    "model": "realtime-0.5b",
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
# Engine Implementation - Long-form 1.5B
# ============================================================================


class VibeVoiceLongformEngine(TextPromptedEngine):
    """VibeVoice 1.5B Engine - Long-form multi-speaker TTS.

    Generates long-form speech (up to 90 minutes) with multi-speaker support.
    Supports up to 4 distinct speakers for conversations and podcasts.

    Parameters:
        speakers (list[str]): List of speaker names for multi-speaker generation.
        speaker_name (str): Primary speaker for single-speaker generation.
    """

    @property
    def name(self) -> str:
        return "VibeVoice Long-form"

    @property
    def engine_id(self) -> str:
        return "vibevoice_longform"

    def is_available(self) -> bool:
        try:
            import vibevoice  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="text_prompted",
            description="Long-form TTS up to 90 min, multi-speaker (4), conversations",
            requirements="vibevoice (pip install vibevoice or git clone)",
            max_duration_secs=MAX_DURATION_SECS_LONGFORM,
            chunk_size_chars=MAX_CHUNK_CHARS_LONGFORM,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            prompting_guide=PromptingGuide(
                overview=(
                    "VibeVoice 1.5B generates long-form audio content like podcasts and "
                    "conversations. Supports up to 4 speakers with natural turn-taking. "
                    "Best for English and Chinese content."
                ),
                text_formatting=[
                    "Use speaker labels like 'Speaker1:' or names for multi-speaker",
                    "Natural dialogue formatting works well",
                    "Can handle very long content (up to 90 minutes)",
                    "Use proper punctuation for natural pacing",
                ],
                voice_guidance={
                    "type": "speaker_names",
                    "description": "Assign speaker names for multi-speaker generation",
                    "max_speakers": 4,
                    "default": DEFAULT_SPEAKER,
                },
                parameters={
                    "speaker_name": {
                        "type": "str",
                        "default": DEFAULT_SPEAKER,
                        "description": "Primary speaker name for single-speaker generation",
                    },
                    "speakers": {
                        "type": "list[str]",
                        "default": None,
                        "description": "List of speaker names for multi-speaker (max 4)",
                    },
                },
                tips=[
                    "Ideal for podcasts, audiobooks, and long conversations",
                    "Multi-speaker works best with clear speaker attribution",
                    "English and Chinese are fully supported",
                    "Uses ~6-8GB VRAM with 1.5B parameters",
                ],
            ),
            extra_info={
                "license": "MIT",
                "license_url": "https://github.com/microsoft/VibeVoice",
                "model_id": MODEL_ID_LONGFORM,
                "parameters": "1.5B (3B total with tokenizers)",
                "max_speakers": 4,
                "languages": ["en", "zh"],
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=10.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~6-8GB VRAM used. Optimized for long-form content up to 90 minutes.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=5.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Good performance on Apple Silicon. May use significant unified memory.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.5,
                    device_type="cpu",
                    reference_hardware="Intel i9-12900K",
                    notes="1.5B model is slow on CPU. GPU strongly recommended.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## VibeVoice Long-form Setup

VibeVoice 1.5B is for long-form and multi-speaker generation.

**Option 1: Install from PyPI (when available):**
```bash
pip install vibevoice
```

**Option 2: Install from source:**
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .
```

**Hardware Requirements:**
- GPU strongly recommended for 1.5B model
- ~6-8GB VRAM
- Works on CUDA, MPS, and CPU (CPU will be slow)
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        voice_description: str = DEFAULT_SPEAKER,
        speaker_name: Optional[str] = None,
        speakers: Optional[list[str]] = None,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with VibeVoice 1.5B.

        Args:
            text: Text to synthesize. Can include speaker labels for multi-speaker.
            output_path: Where to save the generated audio.
            voice_description: Primary speaker name (for TextPromptedEngine compatibility).
            speaker_name: Explicit primary speaker name.
            speakers: List of speaker names for multi-speaker generation.

        Returns:
            TTSResult with status and metadata.
        """
        import soundfile as sf

        output_path = Path(output_path)

        # Determine speaker(s)
        speaker = speaker_name or voice_description or DEFAULT_SPEAKER
        speaker_list = speakers if speakers else [speaker]

        # Validate inputs
        if not text or not text.strip():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Text cannot be empty",
            )

        if len(speaker_list) > 4:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error="Maximum 4 speakers supported",
            )

        try:
            model = _load_longform_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load VibeVoice 1.5B: {e}",
            )

        try:
            # Generate audio
            if len(speaker_list) == 1:
                audio_np = model.generate(text=text, speaker_name=speaker_list[0])
            else:
                audio_np = model.generate(text=text, speakers=speaker_list)

            # Save
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_np, SAMPLE_RATE)

            duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
                metadata={
                    "speakers": speaker_list,
                    "num_speakers": len(speaker_list),
                    "model": "1.5b",
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
