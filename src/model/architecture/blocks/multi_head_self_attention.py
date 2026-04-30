import torch
import torch.nn as nn

class MHSABlock(nn.Module):
    """
    2-D Multi-Head Self-Attention:
    (B,C,H,W) → flatten → (B,H*W,C) → LayerNorm → MHA → residual → reshape
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm    = nn.LayerNorm(channels)
        self.attn    = nn.MultiheadAttention(channels, num_heads,
                                             dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W  = x.shape
        tokens      = x.flatten(2).transpose(1, 2)           # (B, H*W, C)
        normed      = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens      = tokens + self.dropout(attn_out)
        return tokens.transpose(1, 2).view(B, C, H, W)