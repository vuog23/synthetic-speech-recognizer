import torch
import torch.nn as nn

class TransitionBlock(nn.Module):
    """BN → Conv 1×1 (compress) → AvgPool 2×2 (halve resolution)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.trans = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trans(x)