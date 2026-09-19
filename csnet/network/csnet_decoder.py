"""
CSNet decoder and associated modules.

This file contains implementations of the decoder and auxiliary blocks
for the CSNet segmentation model.  The decoder improves upon the
DeepLabV3+ decoder by introducing a DenseASPP module to better
aggregate multi‑scale context features, a feature fusion module with
learnable weighting, and an attention mechanism inspired by CBAM to
selectively emphasize informative regions.

The modules are annotated with task markers indicating which task
they correspond to:

* Task 2 – Decoder modification: defines the CSNetHead class that
  combines DenseASPP output with intermediate encoder features and
  applies feature fusion and attention before final classification.
* Task 3 – FFAB module: defines the FeatureFusionModule and
  AttentionModule classes implementing the feature fusion and
  attention mechanisms.
* Task 4 – DenseASPP: defines the DenseASPP module implementing a
  densely connected atrous spatial pyramid pooling as described in
  the paper.

"""

import torch
from torch import nn
import torch.nn.functional as F


########### Task 4 DenseASPP Module start ###########
class DenseASPP(nn.Module):
    """Densely connected Atrous Spatial Pyramid Pooling.

    This module implements the DenseASPP described in the CSNet
    paper.  Instead of parallel atrous convolutions, dilated
    convolutions are connected sequentially so that features from all
    preceding convolutions are concatenated and reused.  A final 1×1
    convolution compresses the concatenated features to a desired
    number of channels.

    Args:
        in_channels: Number of input channels from the encoder.
        out_channels: Number of channels of the output context
            feature map.
        dilations: Iterable of dilation rates used in each stage.
        inter_channels: Number of channels used inside each atrous
            convolution.  If None, defaults to out_channels // len(dilations).
    """

    def __init__(self, in_channels: int, out_channels: int = 256,
                 dilations=(3, 6, 12, 18, 24), inter_channels: int = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = out_channels // len(dilations)
        self.dilations = dilations
        self.stages = nn.ModuleList()
        in_ch = in_channels
        for dilation in dilations:
            stage = nn.Sequential(
                nn.Conv2d(in_ch, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            )
            self.stages.append(stage)
            in_ch += inter_channels
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for stage in self.stages:
            inp = torch.cat(features, dim=1)
            out = stage(inp)
            features.append(out)
        concat = torch.cat(features, dim=1)
        return self.project(concat)
########### Task 4 DenseASPP Module end ###########


########### Task 3 FFAB and Attention Module start ###########
class FeatureFusionModule(nn.Module):
    """Feature fusion with learnable weighting.

    Given a high‑resolution feature map (e.g., upsampled context) and
    a low‑level feature map, this module projects both to the same
    number of channels, concatenates them, computes a spatial gating
    weight and produces a weighted combination.  This design follows
    the description of the feature fusion module in the paper where
    high‑level features are upsampled and fused with low‑level features
    through an element‑wise weighted summation.

    Args:
        in_channels_high: Number of channels of the high‑resolution
            feature map.
        in_channels_low: Number of channels of the low‑level feature
            map.
        out_channels: Desired number of output channels.
    """

    def __init__(self, in_channels_high: int, in_channels_low: int,
                 out_channels: int = 64):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels_high, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels_low, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, high_feat: torch.Tensor, low_feat: torch.Tensor) -> torch.Tensor:
        high_proj = self.conv_high(high_feat)
        low_proj = self.conv_low(low_feat)
        x = torch.cat([high_proj, low_proj], dim=1)
        w = self.gate(x)
        fused = w * high_proj + (1 - w) * low_proj
        return fused


class ChannelAttention(nn.Module):
    """Channel attention module from CBAM."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention module from CBAM."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        scale = self.sigmoid(out)
        return x * scale


class AttentionModule(nn.Module):
    """Combined channel and spatial attention module."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out
########### Task 3 FFAB and Attention Module end ###########


########### Task 2 Decoder Enhancement start ###########
class CSNetHead(nn.Module):
    """Decoder head for CSNet.

    This decoder implements a two-branch fusion strategy as described
    in the CSNet architecture: one branch applies a DenseASPP to the
    deepest encoder features and fuses them with the 1/16 feature map,
    while the other branch fuses an upsampled 1/16 feature map with
    the 1/4 feature map via an element‑wise weighted summation.  Both
    branches are later merged and upsampled to produce the final
    segmentation output.

    Args:
        in_channels: Number of channels of the deepest encoder
            features (``out`` key).
        low16_channels: Number of channels of the encoder feature map at
            1/16 resolution (``low_level16`` key).
        low4_channels: Number of channels of the encoder feature map at
            1/4 resolution (``low_level`` key).
        num_classes: Number of output segmentation classes.
    """

    def __init__(self, in_channels: int, low16_channels: int,
                 low4_channels: int, num_classes: int):
        super().__init__()
        # number of channels used inside DenseASPP and refinement
        aspp_out_channels = 256
        # number of channels for fusion outputs
        fusion_out_channels = 64
        # ----------------------------------------------------------------------
        # Branch 1: Fuse upsampled 1/16 features (to 1/4) with 1/4 features via
        # element‑wise weighted summation and attention.
        # FeatureFusionModule expects high_feat and low_feat projections; here
        # the high resolution branch comes from the upsampled 1/16 features,
        # and the low resolution branch comes from the 1/4 features.  After
        # weighted summation we apply an attention module to refine the fused
        # output.  The result is a 1/4 resolution feature map ``A``.
        self.fusion_low = FeatureFusionModule(
            in_channels_high=low16_channels,
            in_channels_low=low4_channels,
            out_channels=fusion_out_channels
        )
        self.attention_low = AttentionModule(fusion_out_channels)

        # ----------------------------------------------------------------------
        # Branch 2: Apply DenseASPP to deepest features and fuse with 1/16
        # features.  The output of DenseASPP (context) is concatenated with a
        # projection of the 1/16 feature map.  A convolution refines this
        # concatenation to produce a feature map which is later upsampled to
        # 1/4 resolution (``G``).
        self.dense_aspp = DenseASPP(in_channels, out_channels=aspp_out_channels)
        # project low16 channels to a unified dimension
        self.low16_project = nn.Sequential(
            nn.Conv2d(low16_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # refine concatenated context and projected low16 features
        self.concat_refine = nn.Sequential(
            nn.Conv2d(aspp_out_channels + 128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------------
        # Final combination: concatenate the upsampled refined context (G) with
        # the fused low‑level output (A) and reduce channels.  A small
        # classifier then produces the per‑pixel logits.
        self.final_conv = nn.Sequential(
            nn.Conv2d(256 + fusion_out_channels, fusion_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(fusion_out_channels, fusion_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_out_channels, num_classes, kernel_size=1)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: dict) -> torch.Tensor:
        # Unpack multi-level features
        x = features['out']
        low16 = features['low_level16']
        low4 = features['low_level']

        # Branch 1: fuse upsampled 1/16 feature map with 1/4 feature map via
        # element‑wise weighted summation.  We first upsample the 1/16
        # feature to the spatial size of the 1/4 feature.
        low16_up = F.interpolate(low16, size=low4.shape[2:], mode='bilinear', align_corners=False)
        fused_low = self.fusion_low(low16_up, low4)
        att_low = self.attention_low(fused_low)

        # Branch 2: DenseASPP on deepest features and fusion with 1/16 features.
        context = self.dense_aspp(x)
        low16_proj = self.low16_project(low16)
        concat = torch.cat([context, low16_proj], dim=1)
        refined = self.concat_refine(concat)
        refined_up = F.interpolate(refined, size=low4.shape[2:], mode='bilinear', align_corners=False)

        # Final combination: concatenate upsampled refined context (G) and
        # refined low‑level fusion output (A), reduce channels, and classify.
        merged = torch.cat([refined_up, att_low], dim=1)
        final_feat = self.final_conv(merged)
        out = self.classifier(final_feat)
        return out
########### Task 2 Decoder Enhancement end ###########
