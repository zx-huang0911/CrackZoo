import torch
import torch.nn as nn


class MSCFM(nn.Module):
    """Multi-scale context feature module defined in the MCCA spec."""

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.branch_r2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.branch_r3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.branch_r4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=False)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.fuse = nn.Conv2d(out_channels * 3 + in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b2 = self.branch_r2(x)
        b3 = self.branch_r3(x)
        b4 = self.branch_r4(x)
        gap_proj = self.gap_proj(self.gap(x))

        concat = torch.cat([b2, b3, b4, x], dim=1)
        fused = self.fuse(concat)
        out = gap_proj * fused
        return out
