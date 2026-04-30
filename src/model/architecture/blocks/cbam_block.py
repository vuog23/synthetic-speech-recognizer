import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """CBAM channel branch: shared MLP on GAP + GMP → sigmoid gate."""
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        reduced  = max(channels // ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg   = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx    = F.adaptive_max_pool2d(x, 1).flatten(1)
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * scale.view(x.size(0), x.size(1), 1, 1)


class SpatialAttention(nn.Module):
    """CBAM spatial branch: channel avg+max → 7×7 conv → sigmoid mask."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.max(dim=1, keepdim=True).values
        mask = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * mask


class CBAM(nn.Module):
    """Sequential CBAM: channel attention → spatial attention."""
    def __init__(self, channels: int, ratio: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, ratio)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))