import torchaudio

class SpecAugment:
    def __init__(self):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(20)
        self.time_mask = torchaudio.transforms.TimeMasking(30)

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec