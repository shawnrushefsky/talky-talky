"""Soprano TTS Engine - Ultra-fast CUDA TTS.

Soprano-80M is an ultra-lightweight, extremely fast TTS model with
2000x realtime performance. Requires CUDA GPU.
"""

import sys
from pathlib import Path

from .base import TTSEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import get_best_device, redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 32000  # High-fidelity 32kHz output
MAX_CHUNK_CHARS = 1000  # Soprano handles long text well
MAX_DURATION_SECS = 120

# Default sampling parameters
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.2


# ============================================================================
# Model Management
# ============================================================================

_model = None


def _load_model():
    """Lazily load Soprano model."""
    global _model

    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()

    if device != "cuda":
        raise RuntimeError(
            f"Soprano requires CUDA GPU. Detected: {device} ({device_name}). "
            "CPU and MPS are not currently supported."
        )

    print("Loading Soprano TTS model on CUDA...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        from soprano import SopranoTTS

        _model = SopranoTTS()

    print("Soprano TTS model loaded successfully", file=sys.stderr, flush=True)

    return _model


# ============================================================================
# Engine Implementation
# ============================================================================


class SopranoEngine(TTSEngine):
    """Soprano TTS Engine - Ultra-fast CUDA TTS.

    Soprano-80M is an ultra-lightweight model with exceptional speed:
    - 2000x realtime (10 hours of audio in <20 seconds)
    - High-fidelity 32kHz output
    - <15ms streaming latency

    Requires NVIDIA GPU with CUDA.

    Parameters:
        temperature (float): Sampling temperature (default 0.3).
        top_p (float): Nucleus sampling parameter (default 0.95).
        repetition_penalty (float): Penalty for repetition (default 1.2).
    """

    @property
    def name(self) -> str:
        return "Soprano"

    @property
    def engine_id(self) -> str:
        return "soprano"

    def is_available(self) -> bool:
        try:
            from soprano import SopranoTTS  # noqa: F401
            import torch

            # Soprano requires CUDA
            if not torch.cuda.is_available():
                return False

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="simple",
            description="Ultra-fast TTS with 2000x realtime speed and 32kHz output (CUDA only)",
            requirements="soprano-tts, NVIDIA GPU with CUDA",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            prompting_guide=PromptingGuide(
                overview=(
                    "Soprano is an ultra-fast TTS engine optimized for speed. It produces "
                    "high-fidelity 32kHz audio at over 2000x realtime. Requires NVIDIA GPU. "
                    "Uses a single voice - no voice selection or cloning."
                ),
                text_formatting=[
                    "Write naturally with standard punctuation",
                    "Soprano handles long texts efficiently",
                    "Use sentence-level pacing for best results",
                ],
                parameters={
                    "temperature": {
                        "type": "float",
                        "default": DEFAULT_TEMPERATURE,
                        "range": [0.1, 1.0],
                        "description": "Sampling temperature. Lower = more consistent.",
                    },
                    "top_p": {
                        "type": "float",
                        "default": DEFAULT_TOP_P,
                        "range": [0.1, 1.0],
                        "description": "Nucleus sampling parameter.",
                    },
                    "repetition_penalty": {
                        "type": "float",
                        "default": DEFAULT_REPETITION_PENALTY,
                        "range": [1.0, 2.0],
                        "description": "Penalty for repetition in output.",
                    },
                },
                tips=[
                    "Soprano is optimized for speed - ideal for batch processing",
                    "32kHz output is perceptually close to 44.1kHz quality",
                    "Use for applications where speed is critical",
                    "Does not support voice cloning or custom voices",
                ],
                examples=[
                    {
                        "use_case": "Fast narration",
                        "text": "Welcome to the future of ultra-fast text-to-speech.",
                        "temperature": 0.3,
                        "notes": "Default parameters work well for most cases",
                    },
                ],
            ),
            extra_info={
                "model_size": "80M parameters",
                "architecture": "Qwen3 LLM + Vocos decoder",
                "license": "Apache-2.0",
                "license_url": "https://huggingface.co/SopranoAI/soprano-mlx",
                "realtime_factor": "2000x",
                "streaming_latency": "<15ms",
                "device_requirement": "CUDA only",
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=2000.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="Ultra-fast 80M model. Generates 10 hours of audio in <20 seconds.",
                ),
                # MPS and CPU not supported
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Soprano TTS Setup (Ultra-Fast CUDA TTS)

Soprano-80M is an ultra-fast TTS model with 2000x realtime performance.

### Installation

```bash
pip install soprano-tts
```

Or from source:
```bash
git clone https://github.com/ekwek1/soprano.git
cd soprano
pip install -e .
```

### Hardware Requirements

- **NVIDIA GPU with CUDA** (required)
- CPU and MPS are NOT supported
- Uses LMDeploy by default for optimal performance

### Notes

- 32kHz high-fidelity output
- <15ms streaming latency
- No voice selection - uses a single built-in voice
- Best for batch processing where speed is critical
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with Soprano.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            temperature: Sampling temperature (default 0.3).
            top_p: Nucleus sampling parameter (default 0.95).
            repetition_penalty: Penalty for repetition (default 1.2).

        Returns:
            TTSResult with status and metadata.
        """
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

        # Check CUDA availability
        device, device_name, _ = get_best_device()
        if device != "cuda":
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Soprano requires CUDA GPU. Detected: {device} ({device_name})",
            )

        try:
            model = _load_model()
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load Soprano model: {e}",
            )

        try:
            # Generate audio with output path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Soprano's infer method can save directly to file
            audio = model.infer(
                text,
                str(output_path),
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Calculate duration from the returned audio tensor
            if hasattr(audio, "shape"):
                if len(audio.shape) == 1:
                    num_samples = audio.shape[0]
                else:
                    num_samples = audio.shape[-1]
                duration_ms = int(num_samples / SAMPLE_RATE * 1000)
            else:
                # If audio is not returned, try to get duration from file
                import soundfile as sf

                info = sf.info(str(output_path))
                duration_ms = int(info.duration * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
                metadata={
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
