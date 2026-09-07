import torch
import torchaudio

class SpecAugment:
    def __init__(
        self,
        freq_mask_param=20,
        time_mask_param=30
    ):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param
        )

        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param
        )

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)

        return spec