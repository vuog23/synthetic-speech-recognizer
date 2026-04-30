import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP → FC(reduce) → ReLU → FC(restore) → Sigmoid."""
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        reduced = max(channels // ratio, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)