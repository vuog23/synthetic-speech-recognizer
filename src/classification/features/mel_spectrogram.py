import librosa
import numpy as np
import torch.nn.functional as F
import torch

class AudioPreprocessor:
    def __init__(self,
                sr=16000,
                target_length=48000,
                n_fft=1024,
                hop_length=256,
                n_mels=128,
                normalize_wav=False,
                fmin=0,
                fmax=None):
        
        self.sr = sr
        self.target_length = target_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize_wav_enabled = normalize_wav

        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sr // 2


    def normalize_wav(self, wav):
        if not self.normalize_wav_enabled:
            return wav
        peak = np.max(np.abs(wav))
        if peak < 1e-8:
            return wav
        return wav / peak


    def crop_or_pad(self, wav, training=False):
        length = len(wav)
        if length > self.target_length:
            if training:
                max_start = length - self.target_length
                start = np.random.randint(0, max_start + 1)
            else:
                start = (length - self.target_length) // 2
            return wav[start:start + self.target_length]

        if length == self.target_length:
            return wav

        total_pad = self.target_length - length
        left = total_pad // 2
        right = total_pad - left
        return np.pad(
            wav,
            (left, right),
            mode="constant"
        )


    def wav2mel(self, wav):
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0
        )

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        mel = np.clip(
            mel,
            -80,
            0.0
        )

        mel = (mel + 80.0) / 80.0

        return mel.astype(np.float32)


    def process(self, wav, training=False):
        wav = self.normalize_wav(wav)

        wav = self.crop_or_pad(
            wav,
            training=training
        )

        mel = self.wav2mel(wav)
        spec = torch.from_numpy(mel)
        spec = spec.unsqueeze(0)

        return spec