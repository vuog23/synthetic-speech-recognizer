"""Lazy Python model adapters; no weights are loaded during server startup."""
import os
from threading import Lock

import numpy as np

from .audio import encode_wav
from .config import Settings


class ModelUnavailable(RuntimeError):
    pass


class InferenceService:
    def __init__(self, config: Settings):
        self.config = config
        self.lock = Lock()
        self.whisper = None
        self.kokoro = None
        self.classifier = None
        self.preprocessor = None
        # Set before importing Hugging Face libraries. Keep downloads in project.
        os.environ.setdefault("HF_HOME", str(config.cache_dir))
        os.environ.setdefault("HF_HUB_CACHE", str(config.cache_dir))

    def transcribe(self, samples, sample_rate):
        if self.whisper is None:
            from src.speech2text.service import WhisperTranscriber
            self.whisper = WhisperTranscriber(
                self.config.whisper_model, self.config.cache_dir, self.config.device
            )
        return {"text": self.whisper.transcribe(samples, sample_rate)}

    def synthesize(self, text, voice, speed):
        if self.kokoro is None:
            from src.text2speech.service import KokoroSynthesizer
            self.kokoro = KokoroSynthesizer(self.config.kokoro_model, self.config.device)
        samples, rate = self.kokoro.synthesize(text, voice, speed)
        if len(samples) / rate > self.config.max_audio_seconds:
            raise ValueError("Generated audio is too long. Please shorten the text.")
        return encode_wav(samples, rate)

    def classify(self, samples, sample_rate):
        if not self.config.classifier_path.is_file():
            raise ModelUnavailable("Classifier weights were not found. Update classifier_path in backend/config.py and restart the server.")
        if self.classifier is None:
            from src.classification.inference_onnx import ONNXInference
            from src.classification.features.mel_spectrogram import AudioPreprocessor
            self.classifier = ONNXInference(str(self.config.classifier_path))
            self.preprocessor = AudioPreprocessor(**self.config.preprocessing)
        import librosa
        rate = self.preprocessor.sr
        wav = librosa.resample(samples, orig_sr=sample_rate, target_sr=rate) if sample_rate != rate else samples
        features = self.preprocessor.process(wav, training=False)
        _, output = self.classifier.predict(features)
        values = np.asarray(output, dtype=np.float64)
        if values.shape != (1, len(self.config.labels)) or not np.isfinite(values).all():
            raise ModelUnavailable("Classifier outputs do not match labels in backend/config.py.")
        values = values[0]
        if self.config.output_type == "logits":
            scores = np.exp(values - np.max(values))
            scores /= scores.sum()
        elif self.config.output_type == "probabilities":
            if np.any(values < 0) or np.any(values > 1) or not np.isclose(values.sum(), 1, atol=0.01):
                raise ModelUnavailable("The classifier returned invalid probabilities.")
            scores = values / values.sum()
        else:
            raise ModelUnavailable("Unknown classifier output_type in backend/config.py.")
        index = int(scores.argmax())
        return {
            "label": self.config.labels[index], "probability": float(scores[index]),
            "scores": scores.tolist(), "labels": list(self.config.labels),
            "model": self.config.classifier_name,
            "windowSeconds": self.preprocessor.target_length / rate,
        }
