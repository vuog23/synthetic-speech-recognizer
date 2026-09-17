"""Edit model paths and the inference contract here, then restart the server."""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    whisper_model: str = "openai/whisper-base"
    kokoro_model: str = "hexgrad/Kokoro-82M"
    cache_dir: Path = ROOT / ".cache"
    device: str = "auto"  # "auto", "cpu", or "cuda"
    classifier_path: Path = ROOT / "models/convnext_base/best_model_int8.onnx"
    classifier_name: str = "ConvNeXt Base · INT8"
    labels: tuple[str, ...] = ("Human", "Synthetic")  # real=0, fake=1
    output_type: str = "logits"  # or "probabilities"
    preprocessing: dict = field(default_factory=lambda: {
        "sr": 16000, "target_length": 48000, "n_fft": 1024,
        "hop_length": 256, "n_mels": 128, "normalize_wav": False,
        "fmin": 0, "fmax": 8000,
    })
    max_upload_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: int = 180  # Also accommodates generated speech.
    max_text_characters: int = 1000


settings = Settings()
