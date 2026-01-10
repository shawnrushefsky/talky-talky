"""CosyVoice3 TTS Engine - Zero-shot multilingual voice cloning.

CosyVoice 3.0 is Alibaba's open-source TTS system that generates natural speech
with superior content consistency, speaker similarity, and prosody naturalness.

Features:
- Zero-shot voice cloning from reference audio
- 9 languages: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- 18+ Chinese dialects/accents
- Instruction-based control for dialect, emotion, speed
- Fine-grained control with [breath] tags
"""

import sys
from pathlib import Path
from typing import Optional

from .base import AudioPromptedEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import get_best_device, redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

# Model identifier
MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
TTSFRD_MODEL_ID = "FunAudioLLM/CosyVoice-ttsfrd"

# Audio settings
SAMPLE_RATE = 22050  # CosyVoice default

# Generation limits
MAX_CHUNK_CHARS = 500
MAX_DURATION_SECS = 300  # 5 minutes per generation

# Supported languages
SUPPORTED_LANGUAGES = [
    "zh",  # Chinese
    "en",  # English
    "ja",  # Japanese
    "ko",  # Korean
    "de",  # German
    "es",  # Spanish
    "fr",  # French
    "it",  # Italian
    "ru",  # Russian
]

# Chinese dialects supported via instruction control
CHINESE_DIALECTS = [
    "Guangdong (Cantonese)",
    "Minnan",
    "Sichuan",
    "Dongbei",
    "Shan3xi (Shaanxi)",
    "Shan1xi (Shanxi)",
    "Shanghai",
    "Tianjin",
    "Shandong",
    "Ningxia",
    "Gansu",
    "Qinghai",
    "Hubei",
    "Taiwan",
    "Chongqing",
    "Henan",
    "Anhui",
    "Jiangxi",
]

# ============================================================================
# Model Management
# ============================================================================

_model = None


def _load_model(model_dir: Optional[str] = None):
    """Lazily load CosyVoice3 model."""
    global _model

    if _model is not None:
        return _model

    device, device_name, _ = get_best_device()
    print(f"Loading CosyVoice3 on {device} ({device_name})...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        try:
            # CosyVoice uses a specific import path
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError:
            raise ImportError(
                "CosyVoice not installed. Install with:\n"
                "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
                "cd CosyVoice && pip install -r requirements.txt"
            )

        # Use provided model_dir or download from HuggingFace
        if model_dir:
            _model = AutoModel(model_dir=model_dir)
        else:
            # Default: download to pretrained_models directory
            from huggingface_hub import snapshot_download

            local_dir = Path.home() / ".cache" / "cosyvoice" / "Fun-CosyVoice3-0.5B"
            if not local_dir.exists():
                print(
                    f"Downloading CosyVoice3 model to {local_dir}...", file=sys.stderr, flush=True
                )
                snapshot_download(MODEL_ID, local_dir=str(local_dir))

            _model = AutoModel(model_dir=str(local_dir))

    print("CosyVoice3 loaded successfully", file=sys.stderr, flush=True)
    return _model


# ============================================================================
# Engine Implementation
# ============================================================================


class CosyVoice3Engine(AudioPromptedEngine):
    """CosyVoice3 Engine - Zero-shot multilingual voice cloning.

    Generates speech by cloning a voice from reference audio.
    Supports 9 languages and instruction-based control.

    Parameters:
        reference_audio_paths (list[str]): Reference audio for voice cloning.
        prompt_text (str): Optional transcript of the reference audio.
        instruction (str): Optional instruction for style control.
        language (str): Target language code.
    """

    @property
    def name(self) -> str:
        return "CosyVoice3"

    @property
    def engine_id(self) -> str:
        return "cosyvoice3"

    def is_available(self) -> bool:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401

            return True
        except ImportError:
            return False

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="audio_prompted",
            description="Zero-shot voice cloning with 9 languages and instruction control",
            requirements="CosyVoice (git clone + pip install)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=True,
            emotion_format="instruction",
            emotion_tags=["[breath]"],
            prompting_guide=PromptingGuide(
                overview=(
                    "CosyVoice3 clones voices from reference audio with excellent quality. "
                    "It supports 9 languages and can be controlled via natural language "
                    "instructions for dialect, emotion, speed, and more."
                ),
                text_formatting=[
                    "Use [breath] tags for breathing sounds at natural pause points",
                    "Proper punctuation helps with natural pacing",
                    "Can handle code-switching between languages",
                    "For Chinese, Pinyin can be used for pronunciation control",
                ],
                emotion_tags={
                    "format": "[tag]",
                    "available": ["breath"],
                    "usage": "Insert [breath] at natural pause points for breathing sounds",
                    "examples": [
                        {
                            "text": "[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点",
                            "note": "Breathing at natural pauses",
                        },
                    ],
                },
                voice_guidance={
                    "type": "reference_audio",
                    "description": "Provide reference audio for voice cloning",
                    "tips": [
                        "Use clear audio with minimal background noise",
                        "5-10 seconds of reference is usually sufficient",
                        "Optionally provide transcript of reference audio",
                    ],
                },
                parameters={
                    "reference_audio_paths": {
                        "type": "list[str]",
                        "required": True,
                        "description": "Paths to reference audio files for voice cloning",
                    },
                    "prompt_text": {
                        "type": "str",
                        "default": None,
                        "description": "Transcript of reference audio (improves quality)",
                    },
                    "instruction": {
                        "type": "str",
                        "default": None,
                        "description": "Natural language instruction for style control",
                        "examples": [
                            "请用广东话表达。 (Speak in Cantonese)",
                            "请用尽可能快地语速说一句话。 (Speak as fast as possible)",
                            "请用温柔的语气说。 (Speak in a gentle tone)",
                        ],
                    },
                    "language": {
                        "type": "str",
                        "default": "auto",
                        "description": "Target language code (auto-detected if not specified)",
                        "options": SUPPORTED_LANGUAGES,
                    },
                },
                tips=[
                    "For best results, provide transcript of reference audio",
                    "Instruction control works best for Chinese content",
                    "Cross-lingual cloning is supported (clone voice in one language, output in another)",
                    "The model handles text normalization automatically",
                ],
                examples=[
                    {
                        "name": "Zero-shot English",
                        "text": "CosyVoice provides accurate, stable, and fast voice generation.",
                        "reference_audio": "speaker_sample.wav",
                        "prompt_text": "You are a helpful assistant.",
                    },
                    {
                        "name": "Cantonese dialect",
                        "text": "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。",
                        "instruction": "请用广东话表达。",
                    },
                    {
                        "name": "With breathing",
                        "text": "[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点",
                        "note": "Fine-grained control with breath tags",
                    },
                ],
            ),
            extra_info={
                "license": "Apache-2.0",
                "license_url": "https://github.com/FunAudioLLM/CosyVoice",
                "model_id": MODEL_ID,
                "parameters": "0.5B",
                "languages": SUPPORTED_LANGUAGES,
                "chinese_dialects": CHINESE_DIALECTS,
                "features": [
                    "zero-shot voice cloning",
                    "cross-lingual cloning",
                    "instruction-based control",
                    "text normalization",
                    "bi-streaming (text-in, audio-out)",
                ],
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=8.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="~4-6GB VRAM used. Zero-shot cloning with 9 language support.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=3.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Reasonable performance on Apple Silicon.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=0.4,
                    device_type="cpu",
                    reference_hardware="Intel i9-12900K",
                    notes="0.5B model is slow on CPU. GPU recommended.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## CosyVoice3 Setup

CosyVoice3 requires cloning the repository and installing dependencies.

**Installation:**
```bash
# Clone repository
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Create conda environment (recommended)
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice

# Install dependencies
pip install -r requirements.txt

# Install sox for audio processing
# Ubuntu/Debian:
sudo apt-get install sox libsox-dev
# macOS:
brew install sox
```

**Download models:**
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                 local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

**Hardware Requirements:**
- GPU recommended for best performance
- Works on CUDA and CPU
- ~4-6GB VRAM for 0.5B model
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        reference_audio_paths: list[str],
        prompt_text: Optional[str] = None,
        instruction: Optional[str] = None,
        language: str = "auto",
        stream: bool = False,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with CosyVoice3.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            reference_audio_paths: Paths to reference audio for voice cloning.
            prompt_text: Transcript of reference audio (improves quality).
            instruction: Natural language instruction for style control.
            language: Target language code.
            stream: Whether to use streaming generation.

        Returns:
            TTSResult with status and metadata.
        """

        try:
            import torchaudio
        except ImportError:
            import soundfile as sf

            torchaudio = None

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

        # Validate reference audio exists
        ref_path = Path(reference_audio_paths[0])
        if not ref_path.exists():
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Reference audio not found: {ref_path}",
            )

        try:
            model = _load_model()
            actual_sample_rate = getattr(model, "sample_rate", SAMPLE_RATE)
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load CosyVoice3: {e}",
            )

        try:
            # Build prompt text if provided
            if prompt_text:
                full_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
            else:
                full_prompt = "You are a helpful assistant.<|endofprompt|>"

            # Choose generation method based on parameters
            if instruction:
                # Instruction-based generation (for dialect, emotion, speed control)
                generator = model.inference_instruct2(
                    text,
                    full_prompt,
                    str(ref_path),
                    stream=stream,
                )
            else:
                # Zero-shot generation
                generator = model.inference_zero_shot(
                    text,
                    full_prompt,
                    str(ref_path),
                    stream=stream,
                )

            # Collect audio chunks
            audio_chunks = []
            for i, result in enumerate(generator):
                if "tts_speech" in result:
                    audio_chunks.append(result["tts_speech"])

            if not audio_chunks:
                return TTSResult(
                    status="error",
                    output_path=str(output_path),
                    duration_ms=0,
                    sample_rate=actual_sample_rate,
                    error="No audio generated",
                )

            # Concatenate all chunks
            import torch

            if len(audio_chunks) == 1:
                audio_tensor = audio_chunks[0]
            else:
                audio_tensor = torch.cat(audio_chunks, dim=1)

            # Save audio
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if torchaudio:
                torchaudio.save(str(output_path), audio_tensor, actual_sample_rate)
            else:
                # Fallback to soundfile
                audio_np = audio_tensor.squeeze().cpu().numpy()
                sf.write(str(output_path), audio_np, actual_sample_rate)

            # Calculate duration
            num_samples = audio_tensor.shape[1] if audio_tensor.dim() > 1 else audio_tensor.shape[0]
            duration_ms = int(num_samples / actual_sample_rate * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=actual_sample_rate,
                chunks_used=len(audio_chunks),
                metadata={
                    "reference_audio": str(ref_path),
                    "prompt_text": prompt_text,
                    "instruction": instruction,
                    "language": language,
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
