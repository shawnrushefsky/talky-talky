"""Kokoro TTS Engine - Voice selection from pre-built voices.

Kokoro-82M is a lightweight, high-quality TTS model with 54 pre-built voices
across 8 languages. Uses voice IDs to select from the available voices.
"""

import sys
from pathlib import Path

from .base import VoiceSelectionEngine, TTSResult, EngineInfo, PromptingGuide, SpeedEstimate
from .utils import redirect_stdout_to_stderr


# ============================================================================
# Constants
# ============================================================================

SAMPLE_RATE = 24000
MAX_CHUNK_CHARS = 500
MAX_DURATION_SECS = 60

# Default voice
DEFAULT_VOICE = "af_heart"

# Language codes and their descriptions
LANGUAGES = {
    "a": {"name": "American English", "espeak": "en-us"},
    "b": {"name": "British English", "espeak": "en-gb"},
    "j": {"name": "Japanese", "espeak": "ja"},
    "z": {"name": "Mandarin Chinese", "espeak": "zh"},
    "e": {"name": "Spanish", "espeak": "es"},
    "f": {"name": "French", "espeak": "fr-fr"},
    "h": {"name": "Hindi", "espeak": "hi"},
    "i": {"name": "Italian", "espeak": "it"},
    "p": {"name": "Brazilian Portuguese", "espeak": "pt-br"},
}

# Complete voice catalog with metadata
# Format: voice_id -> {gender, language, name, quality}
VOICES = {
    # American English - Female
    "af_heart": {"gender": "female", "lang": "a", "name": "Heart", "quality": "A"},
    "af_alloy": {"gender": "female", "lang": "a", "name": "Alloy", "quality": "B"},
    "af_aoede": {"gender": "female", "lang": "a", "name": "Aoede", "quality": "B"},
    "af_bella": {"gender": "female", "lang": "a", "name": "Bella", "quality": "A"},
    "af_jessica": {"gender": "female", "lang": "a", "name": "Jessica", "quality": "C"},
    "af_kore": {"gender": "female", "lang": "a", "name": "Kore", "quality": "B"},
    "af_nicole": {"gender": "female", "lang": "a", "name": "Nicole", "quality": "B"},
    "af_nova": {"gender": "female", "lang": "a", "name": "Nova", "quality": "B"},
    "af_river": {"gender": "female", "lang": "a", "name": "River", "quality": "C"},
    "af_sarah": {"gender": "female", "lang": "a", "name": "Sarah", "quality": "B"},
    "af_sky": {"gender": "female", "lang": "a", "name": "Sky", "quality": "B"},
    # American English - Male
    "am_adam": {"gender": "male", "lang": "a", "name": "Adam", "quality": "D"},
    "am_echo": {"gender": "male", "lang": "a", "name": "Echo", "quality": "C"},
    "am_eric": {"gender": "male", "lang": "a", "name": "Eric", "quality": "C"},
    "am_fenrir": {"gender": "male", "lang": "a", "name": "Fenrir", "quality": "B"},
    "am_liam": {"gender": "male", "lang": "a", "name": "Liam", "quality": "C"},
    "am_michael": {"gender": "male", "lang": "a", "name": "Michael", "quality": "B"},
    "am_onyx": {"gender": "male", "lang": "a", "name": "Onyx", "quality": "C"},
    "am_puck": {"gender": "male", "lang": "a", "name": "Puck", "quality": "B"},
    "am_santa": {"gender": "male", "lang": "a", "name": "Santa", "quality": "C"},
    # British English - Female
    "bf_alice": {"gender": "female", "lang": "b", "name": "Alice", "quality": "C"},
    "bf_emma": {"gender": "female", "lang": "b", "name": "Emma", "quality": "B"},
    "bf_isabella": {"gender": "female", "lang": "b", "name": "Isabella", "quality": "B"},
    "bf_lily": {"gender": "female", "lang": "b", "name": "Lily", "quality": "C"},
    # British English - Male
    "bm_daniel": {"gender": "male", "lang": "b", "name": "Daniel", "quality": "C"},
    "bm_fable": {"gender": "male", "lang": "b", "name": "Fable", "quality": "B"},
    "bm_george": {"gender": "male", "lang": "b", "name": "George", "quality": "B"},
    "bm_lewis": {"gender": "male", "lang": "b", "name": "Lewis", "quality": "C"},
    # Japanese - Female
    "jf_alpha": {"gender": "female", "lang": "j", "name": "Alpha", "quality": "B"},
    "jf_gongitsune": {"gender": "female", "lang": "j", "name": "Gongitsune", "quality": "B"},
    "jf_nezumi": {"gender": "female", "lang": "j", "name": "Nezumi", "quality": "B"},
    "jf_tebukuro": {"gender": "female", "lang": "j", "name": "Tebukuro", "quality": "B"},
    # Japanese - Male
    "jm_kumo": {"gender": "male", "lang": "j", "name": "Kumo", "quality": "B"},
    # Mandarin Chinese - Female
    "zf_xiaobei": {"gender": "female", "lang": "z", "name": "Xiaobei", "quality": "C"},
    "zf_xiaoni": {"gender": "female", "lang": "z", "name": "Xiaoni", "quality": "C"},
    "zf_xiaoxiao": {"gender": "female", "lang": "z", "name": "Xiaoxiao", "quality": "C"},
    "zf_xiaoyi": {"gender": "female", "lang": "z", "name": "Xiaoyi", "quality": "C"},
    # Mandarin Chinese - Male
    "zm_yunjian": {"gender": "male", "lang": "z", "name": "Yunjian", "quality": "C"},
    "zm_yunxi": {"gender": "male", "lang": "z", "name": "Yunxi", "quality": "C"},
    "zm_yunxia": {"gender": "male", "lang": "z", "name": "Yunxia", "quality": "C"},
    "zm_yunyang": {"gender": "male", "lang": "z", "name": "Yunyang", "quality": "C"},
    # Spanish
    "ef_dora": {"gender": "female", "lang": "e", "name": "Dora", "quality": "B"},
    "em_alex": {"gender": "male", "lang": "e", "name": "Alex", "quality": "B"},
    "em_santa": {"gender": "male", "lang": "e", "name": "Santa", "quality": "C"},
    # French
    "ff_siwis": {"gender": "female", "lang": "f", "name": "Siwis", "quality": "B"},
    # Hindi
    "hf_alpha": {"gender": "female", "lang": "h", "name": "Alpha", "quality": "B"},
    "hf_beta": {"gender": "female", "lang": "h", "name": "Beta", "quality": "B"},
    "hm_omega": {"gender": "male", "lang": "h", "name": "Omega", "quality": "B"},
    "hm_psi": {"gender": "male", "lang": "h", "name": "Psi", "quality": "B"},
    # Italian
    "if_sara": {"gender": "female", "lang": "i", "name": "Sara", "quality": "B"},
    "im_nicola": {"gender": "male", "lang": "i", "name": "Nicola", "quality": "B"},
    # Brazilian Portuguese
    "pf_dora": {"gender": "female", "lang": "p", "name": "Dora", "quality": "B"},
    "pm_alex": {"gender": "male", "lang": "p", "name": "Alex", "quality": "B"},
    "pm_santa": {"gender": "male", "lang": "p", "name": "Santa", "quality": "C"},
}


# ============================================================================
# Model Management
# ============================================================================

_pipeline_cache: dict = {}


def _get_pipeline(lang_code: str):
    """Get or create a KPipeline for the given language."""
    global _pipeline_cache

    if lang_code in _pipeline_cache:
        return _pipeline_cache[lang_code]

    print(f"Loading Kokoro pipeline for language '{lang_code}'...", file=sys.stderr, flush=True)

    # Redirect stdout to stderr during import and model loading
    # to prevent library output from breaking MCP JSON protocol
    with redirect_stdout_to_stderr():
        from kokoro import KPipeline

        pipeline = KPipeline(lang_code=lang_code)

    _pipeline_cache[lang_code] = pipeline
    print(f"Kokoro pipeline loaded for '{lang_code}'", file=sys.stderr, flush=True)

    return pipeline


# ============================================================================
# Engine Implementation
# ============================================================================


class KokoroEngine(VoiceSelectionEngine):
    """Kokoro TTS Engine - Voice selection from 54 pre-built voices.

    Kokoro-82M is a lightweight, fast TTS model with high-quality voices
    across 8 languages. Select from predefined voice IDs.

    Parameters:
        voice (str): Voice ID to use (e.g., 'af_heart', 'bm_george').
            See get_available_voices() for the full list.
        speed (float): Speech rate multiplier (default 1.0).
            0.5 = half speed, 2.0 = double speed.
    """

    @property
    def name(self) -> str:
        return "Kokoro"

    @property
    def engine_id(self) -> str:
        return "kokoro"

    def is_available(self) -> bool:
        try:
            from kokoro import KPipeline  # noqa: F401

            return True
        except ImportError:
            return False

    def get_available_voices(self) -> dict[str, dict]:
        """Get all available Kokoro voices.

        Returns:
            Dict mapping voice_id to metadata including:
            - gender: 'male' or 'female'
            - lang: language code
            - language: full language name
            - name: voice name
            - quality: quality grade (A-D)
        """
        result = {}
        for voice_id, info in VOICES.items():
            lang_info = LANGUAGES.get(info["lang"], {})
            result[voice_id] = {
                "gender": info["gender"],
                "lang": info["lang"],
                "language": lang_info.get("name", "Unknown"),
                "name": info["name"],
                "quality": info["quality"],
            }
        return result

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            engine_type="voice_selection",
            description="Lightweight, fast TTS with 54 high-quality pre-built voices across 8 languages",
            requirements="kokoro>=0.9.2, espeak-ng (system)",
            max_duration_secs=MAX_DURATION_SECS,
            chunk_size_chars=MAX_CHUNK_CHARS,
            sample_rate=SAMPLE_RATE,
            supports_emotions=False,
            emotion_format=None,
            emotion_tags=[],
            prompting_guide=PromptingGuide(
                overview=(
                    "Kokoro uses pre-built voices selected by ID. Choose a voice based on "
                    "language, gender, and quality grade. Voice IDs follow the pattern: "
                    "[lang][gender]_[name] (e.g., 'af_heart' = American Female Heart)."
                ),
                text_formatting=[
                    "Write naturally - Kokoro handles punctuation and pacing well",
                    "Use standard punctuation for natural pauses",
                    "Numbers are pronounced as-is; spell out if needed",
                    "Long text is automatically chunked at sentence boundaries",
                ],
                voice_guidance={
                    "type": "selection",
                    "format": "[lang][gender]_[name]",
                    "lang_codes": {
                        "a": "American English",
                        "b": "British English",
                        "j": "Japanese",
                        "z": "Mandarin Chinese",
                        "e": "Spanish",
                        "f": "French",
                        "h": "Hindi",
                        "i": "Italian",
                        "p": "Brazilian Portuguese",
                    },
                    "gender_codes": {"f": "Female", "m": "Male"},
                    "quality_grades": {
                        "A": "Excellent quality, highly recommended",
                        "B": "Good quality, reliable",
                        "C": "Acceptable quality",
                        "D": "Lower quality, use with caution",
                    },
                    "recommended_voices": {
                        "American English": ["af_heart", "af_bella", "am_fenrir", "am_michael"],
                        "British English": ["bf_emma", "bm_george", "bm_fable"],
                        "Japanese": ["jf_alpha", "jm_kumo"],
                    },
                },
                parameters={
                    "voice": {
                        "type": "str",
                        "required": True,
                        "description": "Voice ID from the available voices list",
                        "default": DEFAULT_VOICE,
                    },
                    "speed": {
                        "type": "float",
                        "default": 1.0,
                        "range": [0.5, 2.0],
                        "description": "Speech rate multiplier",
                    },
                },
                tips=[
                    "Use quality grade A or B voices for best results",
                    "af_heart and af_bella are the highest quality American English voices",
                    "For Japanese, install misaki[ja]: pip install misaki[ja]",
                    "For Mandarin, install misaki[zh]: pip install misaki[zh]",
                    "Kokoro is very fast and lightweight - great for batch processing",
                ],
                examples=[
                    {
                        "use_case": "American English narration",
                        "voice": "af_heart",
                        "text": "Welcome to the future of text-to-speech technology.",
                        "notes": "af_heart is the highest quality voice",
                    },
                    {
                        "use_case": "British male narrator",
                        "voice": "bm_george",
                        "text": "Good evening, and welcome to the programme.",
                        "notes": "Clear British accent",
                    },
                    {
                        "use_case": "Japanese female voice",
                        "voice": "jf_alpha",
                        "text": "こんにちは、お元気ですか?",
                        "notes": "Requires misaki[ja] for best results",
                    },
                ],
            ),
            extra_info={
                "model_size": "82M parameters",
                "architecture": "StyleTTS 2 + ISTFTNet",
                "license": "Apache-2.0",
                "license_url": "https://github.com/hexgrad/kokoro",
                "languages": list(LANGUAGES.keys()),
                "total_voices": len(VOICES),
                "default_voice": DEFAULT_VOICE,
            },
            speed_estimates={
                "cuda": SpeedEstimate(
                    realtime_factor=50.0,
                    device_type="cuda",
                    reference_hardware="RTX 4090 (24GB)",
                    notes="Very fast due to 82M lightweight model. Minimal VRAM required.",
                ),
                "mps": SpeedEstimate(
                    realtime_factor=30.0,
                    device_type="mps",
                    reference_hardware="Apple M1 Max (32GB)",
                    notes="Excellent performance on Apple Silicon due to small model size.",
                ),
                "cpu": SpeedEstimate(
                    realtime_factor=5.0,
                    device_type="cpu",
                    reference_hardware="Intel i9-12900K",
                    notes="Lightweight model runs efficiently on CPU.",
                ),
            },
        )

    def get_setup_instructions(self) -> str:
        return """
## Kokoro TTS Setup (Voice Selection)

Kokoro-82M is a lightweight, fast TTS model with 54 pre-built voices.

### Installation

```bash
pip install kokoro>=0.9.2 soundfile
```

**System dependency (required):**
- Linux/WSL: `apt-get install espeak-ng`
- macOS: `brew install espeak-ng`
- Windows: Download from https://github.com/espeak-ng/espeak-ng/releases

### Language-specific extras

For best results with non-English languages:
```bash
pip install misaki[ja]  # Japanese
pip install misaki[zh]  # Mandarin Chinese
```

### Hardware Requirements

- Lightweight model runs on CPU, GPU, or edge devices
- No VRAM requirements - works anywhere
- Very fast inference (~1000 chars = 1 minute audio)
"""

    def generate(
        self,
        text: str,
        output_path: Path,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        **kwargs,
    ) -> TTSResult:
        """Generate audio with Kokoro.

        Args:
            text: Text to synthesize.
            output_path: Where to save the generated audio.
            voice: Voice ID to use (e.g., 'af_heart').
            speed: Speech rate multiplier (default 1.0).

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

        if voice not in VOICES:
            available = list(VOICES.keys())
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Unknown voice '{voice}'. Available: {available[:10]}... (use get_available_voices())",
            )

        voice_info = VOICES[voice]
        lang_code = voice_info["lang"]

        try:
            pipeline = _get_pipeline(lang_code)
        except Exception as e:
            return TTSResult(
                status="error",
                output_path=str(output_path),
                duration_ms=0,
                sample_rate=SAMPLE_RATE,
                error=f"Failed to load Kokoro pipeline: {e}",
            )

        try:
            # Generate audio - pipeline yields chunks
            audio_chunks = []
            generator = pipeline(text, voice=voice, speed=speed)

            for i, (gs, ps, audio) in enumerate(generator):
                if audio is not None and len(audio) > 0:
                    audio_chunks.append(audio)

            if not audio_chunks:
                return TTSResult(
                    status="error",
                    output_path=str(output_path),
                    duration_ms=0,
                    sample_rate=SAMPLE_RATE,
                    error="No audio generated",
                )

            # Concatenate chunks
            audio_np = np.concatenate(audio_chunks)

            # Save output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio_np, SAMPLE_RATE)

            duration_ms = int(len(audio_np) / SAMPLE_RATE * 1000)

            return TTSResult(
                status="success",
                output_path=str(output_path),
                duration_ms=duration_ms,
                sample_rate=SAMPLE_RATE,
                chunks_used=len(audio_chunks),
                metadata={
                    "voice": voice,
                    "voice_name": voice_info["name"],
                    "language": LANGUAGES[lang_code]["name"],
                    "speed": speed,
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
