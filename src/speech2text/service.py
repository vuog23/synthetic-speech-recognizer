"""Reusable version of the Whisper base inference demonstrated in inference.py."""
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


class WhisperTranscriber:
    def __init__(self, model_id, cache_dir, device="auto"):
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        model = WhisperForConditionalGeneration.from_pretrained(model_id, cache_dir=str(cache_dir))
        model.config.forced_decoder_ids = None
        model.generation_config.forced_decoder_ids = None
        self.pipeline = pipeline(
            "automatic-speech-recognition", model=model,
            tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor,
            device=device,
        )

    def transcribe(self, samples, sample_rate):
        if sample_rate != 16000:
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=16000)
        with torch.inference_mode():
            # Overlapping chunks also transcribe recordings longer than 30 seconds.
            result = self.pipeline(
                {"raw": samples, "sampling_rate": 16000},
                chunk_length_s=30, stride_length_s=5,
                generate_kwargs={"task": "transcribe"},
            )
        return result["text"].strip()
