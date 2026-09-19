import torch
import torch.nn as nn
import torch.nn.functional as F


class CAM(nn.Module):
    """Cross-attention module used in top-down MCCA fusion."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.high_conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

        self.low_mid = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.low_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.high_gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        h, w = low.shape[-2:]

        high_up = F.interpolate(high, size=(h, w), mode="bilinear", align_corners=False)
        high_feat = self.high_conv3x3(high_up)

        low_gap = F.adaptive_avg_pool2d(low, output_size=1)
        low_mid = self.low_mid(low_gap)
        low_gate = self.sigmoid(self.low_gate(self.relu(low_mid)))

        high_mid = self.relu(high_feat)
        high_gate = self.sigmoid(self.high_gate(high_mid))

        term_l2h = low_gate * high_feat
        term_h2l = high_gate * low
        fused = term_l2h + term_h2l
        return self.out_proj(fused)
