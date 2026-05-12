from pathlib import Path
import librosa
import torch
from tqdm import tqdm

def wav2melspec(root_path, output_path, preprocessor, classes=("real", "fake")):
    root = Path(root_path)
    output_path = Path(output_path)

    datas = []
    labels = []

    for label_idx, label_name in enumerate(classes):
        folder = root / label_name
        files = list(folder.glob("*.wav"))

        for file_path in tqdm(files, desc=f"Processing {label_name}"):
            try:
                wav, _ = librosa.load(file_path, sr=16000, mono=True)

                spec = preprocessor.process(wav)

                tensor = torch.tensor(
                    spec,
                    dtype=torch.float32
                ).unsqueeze(0)

                datas.append(tensor)
                labels.append(label_idx)

            except Exception as e:
                print(f"Error {file_path.name}: {e}")

    datas = torch.stack(datas)
    labels = torch.tensor(labels, dtype=torch.long)

    torch.save({
        "data": datas,
        "label": labels
    }, output_path)

    print(f"Saved {len(datas)} samples to {output_path}")