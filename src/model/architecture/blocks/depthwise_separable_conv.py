import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise conv + pointwise conv, BN, ReLU6."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                             padding=padding, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))