from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torchvision.models import ResNet34_Weights, resnet34

from .modules import (
    AuxiliaryHead,
    BAM,
    PRAM,
    SpatialAttention,
    UpconvBlock,
    UpconvLastBlock,
    UpsampleBlock,
)


@dataclass
class CrackResUNetConfig:
    num_classes: int = 2
    pretrained_encoder: bool = True
    bam_reduction: int = 16
    bam_dilation: int = 4
    variant: str = "main"  # main,no_sa,no_aux,no_pram,all_pram,all_sa


class CrackResUNet(nn.Module):
    """CrackResU-Net with configurable ablation variants.

    Variants:
    - main: SA on shallow skips, PRAM on deep skips, aux enabled
    - no_sa: shallow identity, deep PRAM, aux enabled
    - no_aux: SA + PRAM, aux disabled
    - no_pram: shallow SA, deep identity, aux enabled
    - all_pram: PRAM on all skip levels, aux enabled
    - all_sa: SA on all skip levels, aux enabled
    """

    def __init__(self, config: CrackResUNetConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        weights = ResNet34_Weights.DEFAULT if config.pretrained_encoder else None
        encoder = resnet34(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.bam1 = BAM(64, reduction=config.bam_reduction, dilation=config.bam_dilation)
        self.bam2 = BAM(128, reduction=config.bam_reduction, dilation=config.bam_dilation)
        self.bam3 = BAM(256, reduction=config.bam_reduction, dilation=config.bam_dilation)
        self.bam4 = BAM(512, reduction=config.bam_reduction, dilation=config.bam_dilation)

        self.sa64 = SpatialAttention()
        self.pram64 = PRAM(64)
        self.pram128 = PRAM(128)
        self.pram256 = PRAM(256)

        self.up4 = UpsampleBlock(512, 256)
        self.fuse4 = UpconvBlock(256 + 256, 256)

        self.up3 = UpsampleBlock(256, 128)
        self.fuse3 = UpconvBlock(128 + 128, 128)

        self.up2 = UpsampleBlock(128, 64)
        self.fuse2 = UpconvBlock(64 + 64, 64)

        self.up1 = UpsampleBlock(64, 64)
        self.fuse1 = UpconvBlock(64 + 64, 64)

        self.final_head = UpconvLastBlock(64, config.num_classes)
        self.aux_head = AuxiliaryHead(128, config.num_classes)

        supported = {"main", "no_sa", "no_aux", "no_pram", "all_pram", "all_sa"}
        if config.variant not in supported:
            raise ValueError(f"Unsupported variant '{config.variant}'. Supported: {sorted(supported)}")

    def _refine_shallow(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.variant in {"no_sa"}:
            return x
        if self.config.variant in {"all_pram"}:
            return self.pram64(x)
        return self.sa64(x)

    def _refine_deep_128(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.variant in {"no_pram"}:
            return x
        if self.config.variant in {"all_sa"}:
            return self.sa64(x)
        return self.pram128(x)

    def _refine_deep_256(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.variant in {"no_pram"}:
            return x
        if self.config.variant in {"all_sa"}:
            return self.sa64(x)
        return self.pram256(x)

    def forward(self, x: torch.Tensor, return_shapes: bool = False) -> Dict[str, torch.Tensor] | Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[int, ...]]]:
        shapes: Dict[str, Tuple[int, ...]] = {}

        e1 = self.stem(x)  # /2,64
        shapes["E1"] = tuple(e1.shape)

        b1 = self.layer1(self.maxpool(e1))  # /4,64
        b1 = self.bam1(b1)
        shapes["B1"] = tuple(b1.shape)

        b2 = self.layer2(b1)  # /8,128
        b2 = self.bam2(b2)
        shapes["B2"] = tuple(b2.shape)

        b3 = self.layer3(b2)  # /16,256
        b3 = self.bam3(b3)
        shapes["B3"] = tuple(b3.shape)

        b4 = self.layer4(b3)  # /32,512
        b4 = self.bam4(b4)
        shapes["B4"] = tuple(b4.shape)

        s1 = self._refine_shallow(e1)
        s2 = self._refine_shallow(b1)
        s3 = self._refine_deep_128(b2)
        s4 = self._refine_deep_256(b3)
        shapes["S1"] = tuple(s1.shape)
        shapes["S2"] = tuple(s2.shape)
        shapes["S3"] = tuple(s3.shape)
        shapes["S4"] = tuple(s4.shape)

        d4u = self.up4(b4)
        d4 = self.fuse4(torch.cat([d4u, s4], dim=1))
        shapes["D4"] = tuple(d4.shape)

        d3u = self.up3(d4)
        d3 = self.fuse3(torch.cat([d3u, s3], dim=1))
        shapes["D3"] = tuple(d3.shape)

        d2u = self.up2(d3)
        d2 = self.fuse2(torch.cat([d2u, s2], dim=1))
        shapes["D2"] = tuple(d2.shape)

        d1u = self.up1(d2)
        d1 = self.fuse1(torch.cat([d1u, s1], dim=1))
        shapes["D1"] = tuple(d1.shape)

        main_logits = self.final_head(d1)
        shapes["MainLogits"] = tuple(main_logits.shape)

        aux_logits = None
        if self.config.variant != "no_aux":
            aux_logits = self.aux_head(b2)
            shapes["AuxLogits"] = tuple(aux_logits.shape)

        out = {"main_logits": main_logits, "aux_logits": aux_logits}
        if return_shapes:
            return out, shapes
        return out
