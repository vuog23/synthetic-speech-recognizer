import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.architecture.blocks.cbam_block import CBAM
from src.model.architecture.blocks.dense_block import DenseBlock
from src.model.architecture.blocks.mbconv_block import MBConvSE
from src.model.architecture.blocks.transition_block import TransitionBlock
from src.model.architecture.blocks.multi_head_self_attention import MHSABlock

class MADNet(nn.Module):
    def __init__(self, num_classes: int = 2, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        # Input: (B, 1, 224, 224)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )  # → (B, 32, 112, 112)

        self.stage1 = MBConvSE(32, 64, expand_ratio=2, stride=2)
        # → (B, 64, 56, 56)

        self.dense_block = DenseBlock(64, num_layers=2, growth_rate=16)
        # → (B, 96, 56, 56)  [64 + 2×16 = 96]

        self.dense_drop = nn.Dropout2d(p=0.2)

        self.transition = TransitionBlock(self.dense_block.out_channels, 64)
        # → (B, 64, 28, 28)

        self.cbam = CBAM(64)
        # → (B, 64, 28, 28)

        self.stage2 = MBConvSE(64, 128, expand_ratio=4, stride=2)
        # → (B, 128, 14, 14)

        self.pre_mhsa_drop = nn.Dropout2d(p=0.1)

        self.mhsa = MHSABlock(128, num_heads=4, dropout=dropout)
        # → (B, 128, 14, 14)

        self.pool = nn.AdaptiveAvgPool2d(1)
        # → (B, 128, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.dense_drop(self.dense_block(x))
        x = self.transition(x)
        x = self.cbam(x)
        x = self.stage2(x)
        x = self.pre_mhsa_drop(x)
        x = self.mhsa(x)
        x = self.pool(x)
        return self.classifier(x)
