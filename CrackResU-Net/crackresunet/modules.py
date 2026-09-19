from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafeBatchNorm2d(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] == 1 and input.shape[2] * input.shape[3] == 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int | None = None,
    dilation: int = 1,
) -> nn.Sequential:
    if padding is None:
        padding = ((kernel_size - 1) // 2) * dilation
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        SafeBatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ChannelAttentionBranch(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.bn = SafeBatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        out = self.fc1(pooled)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn(out)
        return out


class SpatialAttentionBranch(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, dilation: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.reduce = conv_bn_relu(channels, hidden, kernel_size=1)
        self.dilated1 = conv_bn_relu(hidden, hidden, kernel_size=3, dilation=dilation)
        self.dilated2 = conv_bn_relu(hidden, hidden, kernel_size=3, dilation=dilation)
        self.proj = nn.Conv2d(hidden, 1, kernel_size=1, bias=False)
        self.bn = SafeBatchNorm2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x)
        out = self.dilated1(out)
        out = self.dilated2(out)
        out = self.proj(out)
        out = self.bn(out)
        return out


class BAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, dilation: int = 4) -> None:
        super().__init__()
        self.channel_branch = ChannelAttentionBranch(channels, reduction=reduction)
        self.spatial_branch = SpatialAttentionBranch(channels, reduction=reduction, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.channel_branch(x)
        ms = self.spatial_branch(x)
        mask = torch.sigmoid(mc + ms)
        return x + x * mask


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        merged = torch.cat([avg_map, max_map], dim=1)
        mask = torch.sigmoid(self.conv(merged))
        return x + x * mask


class CPC(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        reduced = max(1, in_channels // 2)
        self.reduce = conv_bn_relu(in_channels, reduced, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        ch_max, _ = torch.max(reduced, dim=1, keepdim=True)
        ch_avg = torch.mean(reduced, dim=1, keepdim=True)
        return torch.cat([ch_max, ch_avg], dim=1)


class SimNL(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.query_cpc = CPC(channels)
        self.key_cpc = CPC(channels)
        value_channels = max(1, channels // 2)
        self.value_proj = conv_bn_relu(channels, value_channels, kernel_size=1)
        self.out_proj = conv_bn_relu(value_channels, channels, kernel_size=1)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        query = self.query_cpc(x).flatten(start_dim=2).transpose(1, 2)  # B, HW, 2
        key = self.key_cpc(x).flatten(start_dim=2)  # B, 2, HW
        affinity = torch.bmm(query, key)  # B, HW, HW
        attention = F.softmax(affinity, dim=-1)

        value = self.value_proj(x).flatten(start_dim=2).transpose(1, 2)  # B, HW, C/2
        aggregated = torch.bmm(attention, value).transpose(1, 2).reshape(b, -1, h, w)
        return self.out_proj(aggregated)

    def forward(self, x: torch.Tensor, residual: bool = True) -> torch.Tensor:
        transformed = self._transform(x)
        if residual:
            return x + transformed
        return transformed


class PRAM(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.branch_1 = conv_bn_relu(channels, channels, kernel_size=1)
        self.simnl_2 = SimNL(channels)
        self.simnl_3 = SimNL(channels)
        self.simnl_6 = SimNL(channels)
        self.final_proj = conv_bn_relu(channels * 5, channels, kernel_size=1)

    def _pool_upsample(self, x: torch.Tensor, out_size: Tuple[int, int], pool_size: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
        if pool_size == 1:
            branch = self.branch_1(pooled)
        elif pool_size == 2:
            # PRAM explicitly uses SimNL without residual inside pooled branches.
            branch = self.simnl_2(pooled, residual=False)
        elif pool_size == 3:
            # PRAM explicitly uses SimNL without residual inside pooled branches.
            branch = self.simnl_3(pooled, residual=False)
        else:
            # PRAM explicitly uses SimNL without residual inside pooled branches.
            branch = self.simnl_6(pooled, residual=False)
        return F.interpolate(branch, size=out_size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        b1 = self._pool_upsample(x, (h, w), 1)
        b2 = self._pool_upsample(x, (h, w), 2)
        b3 = self._pool_upsample(x, (h, w), 3)
        b6 = self._pool_upsample(x, (h, w), 6)
        merged = torch.cat([x, b1, b2, b3, b6], dim=1)
        return self.final_proj(merged)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = conv_bn_relu(in_channels, out_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.conv(x)


class UpconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            conv_bn_relu(in_channels, out_channels, kernel_size=3),
            conv_bn_relu(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpconvLastBlock(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(x)


class AuxiliaryHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.reduce = conv_bn_relu(in_channels, 64, kernel_size=1)
        self.out = UpconvLastBlock(in_channels=64, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.out(x)


@dataclass(frozen=True)
class ShapeContract:
    name: str
    shape: Tuple[int, ...]
