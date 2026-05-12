import librosa
import numpy as np
import torch.nn.functional as F
import torch

class AudioPreprocessor:
    def __init__(self, sr=16000, target_length=48000,
                 n_fft=1024, hop_length=256, n_mels=128,
                 normalize=True):
        self.sr = sr
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize

    def normalize_wav(self, wav):
        if not self.normalize:
            return wav
        return wav / (np.max(np.abs(wav)) + 1e-6)

    def pad_or_trim(self, wav):
        if len(wav) >= self.target_length:
            return wav[:self.target_length]

        total_pad = self.target_length - len(wav)
        left = total_pad // 2
        right = total_pad - left
        return np.pad(wav, (left, right), mode='constant')

    def wav2spec(self, wav):
        spec = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        spec = librosa.power_to_db(spec, ref=np.max)
        spec = np.clip(spec, -80, 0)
        spec = (spec + 80.) / 80.

        return spec

    def process(self, wav):
        wav = self.normalize_wav(wav)
        wav = self.pad_or_trim(wav)

        spec = self.wav2spec(wav)

        spec = torch.tensor(spec, dtype=torch.float32)

        spec = spec.unsqueeze(0).unsqueeze(0)

        spec = F.interpolate(
            spec,
            size=(128, 128),
            mode='bilinear',
            align_corners=False
        )

        spec = spec.squeeze(0)

        return spec