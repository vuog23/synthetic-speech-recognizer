import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """
    BN→ReLU6→bottleneck 1×1→BN→ReLU6→DWS conv.
    Concatenates output with input (dense connection).
    """
    def __init__(self, in_ch: int, growth_rate: int):
        super().__init__()
        bottleneck = growth_rate * 4
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, bottleneck, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(bottleneck)
        self.dw    = nn.Conv2d(bottleneck, bottleneck, 3, padding=1,
                               groups=bottleneck, bias=False)
        self.pw    = nn.Conv2d(bottleneck, growth_rate, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(growth_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu6(self.bn1(x), inplace=True)
        out = F.relu6(self.bn2(self.conv1(out)), inplace=True)
        out = self.bn3(self.pw(self.dw(out)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    """Stack of DenseLayers; channels grow by growth_rate each layer."""
    def __init__(self, in_ch: int, num_layers: int, growth_rate: int):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers       = nn.ModuleList(layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x