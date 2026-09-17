"""Reusable Kokoro 82M generation using the project's KPipeline approach."""
import numpy as np
import torch
from kokoro import KPipeline


class KokoroSynthesizer:
    def __init__(self, model_id="hexgrad/Kokoro-82M", device="auto"):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model_id = model_id
        self.pipelines = {}

    def synthesize(self, text, voice="af_heart", speed=1.0):
        language = voice[0]
        if language not in self.pipelines:
            shared_model = next(iter(self.pipelines.values())).model if self.pipelines else True
            self.pipelines[language] = KPipeline(
                lang_code=language, repo_id=self.model_id,
                model=shared_model, device=self.device,
            )
        with torch.inference_mode():
            chunks = [audio.detach().cpu().numpy() for _, _, audio in
                      self.pipelines[language](text, voice=voice, speed=speed) if audio is not None]
        if not chunks:
            raise ValueError("Kokoro returned no speech. Try different text.")
        return np.concatenate(chunks).astype(np.float32), 24000
