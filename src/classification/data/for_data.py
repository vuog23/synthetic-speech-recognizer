from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from src.classification.features.mel_spectrogram import AudioPreprocessor
from src.classification.features.spec_augment import SpecAugment

class FoRDataset(Dataset):
    def __init__(
        self,
        root_path,
        split,
        preprocessor=None,
        augment=False,
        classes=("real", "fake")
    ):

        self.root = Path(root_path) / split

        self.preprocessor = (
            preprocessor
            if preprocessor is not None
            else AudioPreprocessor()
        )

        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None

        self.class_to_idx = {
            class_name: idx
            for idx, class_name in enumerate(classes)
        }

        self.samples = []

        for class_name in classes:
            class_dir = self.root / class_name

            label = self.class_to_idx[class_name]

            # Final contains ASVspoof/LibriSpeech WAV files and VCTK FLAC files.
            # librosa supports both formats, so keep the merged dataset intact.
            files = sorted(
                path
                for suffix in ("*.wav", "*.flac")
                for path in class_dir.glob(suffix)
            )

            for file_path in files:
                self.samples.append(
                    (file_path, label)
                )


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        file_path, label = self.samples[index]

        wav, _ = librosa.load(
            file_path,
            sr=self.preprocessor.sr,
            mono=True
        )

        if not np.isfinite(wav).all():
            raise FloatingPointError(
                f"Audio contains NaN or Inf values: {file_path}"
            )

        spec = self.preprocessor.process(
            wav,
            training=self.augment
        )

        if self.augment:
            spec = self.spec_augment(spec)

        if not torch.isfinite(spec).all():
            raise FloatingPointError(
                f"Spectrogram contains NaN or Inf values: {file_path}"
            )

        return spec, torch.tensor(
            label,
            dtype=torch.long
        )
