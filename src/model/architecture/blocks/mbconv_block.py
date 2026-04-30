import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.architecture.blocks.squeeze_excitation import SEBlock

class MBConvSE(nn.Module):
    """
    Inverted Bottleneck + SE with optional residual.
    expand 1×1 → depthwise k×k → SE → project 1×1
    Residual added only when stride=1 AND in_ch==out_ch.
    """
    def __init__(self, in_ch: int, out_ch: int, expand_ratio: int = 4,
                 stride: int = 1, se_ratio: int = 4, kernel_size: int = 3):
        super().__init__()
        mid_ch             = in_ch * expand_ratio
        self.use_residual  = (stride == 1 and in_ch == out_ch)

        self.expand  = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
        )
        self.se      = SEBlock(mid_ch, se_ratio)   # SEBlock defined above ✓
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw_conv(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out