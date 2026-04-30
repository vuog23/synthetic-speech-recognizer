import torch
import torch.nn.functional as F
import librosa
from pathlib import Path
from tqdm import tqdm
import numpy as np

def wav2melspec(root_path, output_path, sr=16000, target_length=48000, n_mels=224):
    root = Path(root_path)
    specs, labels = [], []

    for label_idx, label_name in enumerate(["fake", "real"]):  # fake=0, real=1
        input_folder = root / label_name
        for file_path in tqdm(list(input_folder.iterdir()), desc=label_name):
            if not file_path.is_file():
                continue
            try:
                wav, _ = librosa.load(str(file_path), sr=sr, mono=True)
                wav = wav / (np.max(np.abs(wav)) + 1e-6)

                if len(wav) < target_length:
                    pad_total = target_length - len(wav)
                    wav = np.pad(wav, (pad_total // 2, pad_total - pad_total // 2))
                else:
                    start = (len(wav) - target_length) // 2
                    wav = wav[start:start + target_length]

                spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
                spec = librosa.power_to_db(spec, ref=np.max, top_db=80.0)
                spec = (spec + 80) / 80

                spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                spec = F.interpolate(spec, size=(224, 224), mode='bilinear', align_corners=False)
                spec = spec.squeeze(0)  # (1, 224, 224)

                specs.append(spec)  # (1, H, W)
                labels.append(label_idx)

            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    torch.save({"specs": torch.stack(specs), "labels": torch.tensor(labels, dtype=torch.long)}, output_path)
    print(f"Saved {len(specs)} samples to {output_path}")