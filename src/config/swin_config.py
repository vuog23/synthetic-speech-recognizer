import torch
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class SwinConfig:
    MODEL_ID: str = "microsoft/swin-tiny-patch4-window7-224"

    TRAIN_PT: str = "/kaggle/input/datasets/trieuvuongnguyen/for-preprocessed/processed/train.pt"
    VAL_PT: str = "/kaggle/input/datasets/trieuvuongnguyen/for-preprocessed/processed/val.pt"

    OUTPUT_DIR: str = "./swin-fakeaudio"
    FINAL_DIR: str = "./swin-fakeaudio-final"

    LABEL2ID: dict = None
    ID2LABEL: dict = None
    NUM_CLASSES: int = 2

    BATCH_SIZE: int = 16
    EPOCHS: int = 20
    LR: float = 1e-4
    SEED: int = 42

    def __post_init__(self):
        if self.LABEL2ID is None:
            self.LABEL2ID = {"fake": 0, "real": 1}
        if self.ID2LABEL is None:
            self.ID2LABEL = {0: "fake", 1: "real"}

    def set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed_all(self.SEED)

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")