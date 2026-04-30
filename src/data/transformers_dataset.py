import torch
from torch.utils.data import Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, pt_path: str, image_processor):
        data        = torch.load(pt_path, map_location="cpu")
        self.specs  = data["specs"].float()   # (N, 1, H, W)
        self.labels = data["labels"].long()
        self.proc   = image_processor
        print(f"  Loaded {pt_path}: {len(self.labels)} samples")
 
    def __len__(self):
        return len(self.labels)
 
    def __getitem__(self, idx):
        spec   = self.specs[idx]                                  # (1, 224, 224) already in [0,1]
        spec   = spec.repeat(3, 1, 1)                             # (3, 224, 224) — Swin expects RGB
        np_img = spec.permute(1, 2, 0).numpy()                   # (224, 224, 3)
        processed = self.proc(
            images         = np_img,
            return_tensors = "pt",
            do_rescale     = False,
        )
        return {
            "pixel_values": processed["pixel_values"].squeeze(0),
            "labels":       self.labels[idx],
        }