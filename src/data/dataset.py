import torch
from torch.utils.data import Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, pt_path, transform=None):
        data = torch.load(pt_path)
        self.specs  = data["specs"]
        self.labels = data["labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        if self.transform:
            spec = self.transform(spec)
        return spec, self.labels[idx]