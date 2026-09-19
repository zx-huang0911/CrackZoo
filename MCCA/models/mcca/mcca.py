from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ResNet18Backbone
from .cam import CAM
from .mscfm import MSCFM


class MCCA(nn.Module):
    VARIANTS = {
        "baseline_plain",
        "baseline_deepsup",
        "mcca_no_mscfm",
        "mcca_no_cam",
        "full_mcca",
    }

    def __init__(
        self,
        input_size: int = 640,
        pretrained_backbone: bool = False,
        model_variant: str = "full_mcca",
    ):
        super().__init__()
        self.input_size = input_size
        self.model_variant = model_variant.lower()
        if self.model_variant not in self.VARIANTS:
            raise ValueError(f"Unknown model_variant '{model_variant}'. Available: {sorted(self.VARIANTS)}")

        self.use_mscfm = self.model_variant in {"mcca_no_cam", "full_mcca"}
        self.use_cam = self.model_variant in {"mcca_no_mscfm", "full_mcca"}
        self.use_deepsup = self.model_variant in {"baseline_deepsup", "mcca_no_mscfm", "mcca_no_cam", "full_mcca"}

        self.backbone = ResNet18Backbone(pretrained=pretrained_backbone)

        in_channels = [64, 128, 256, 512]
        self.proj_blocks = nn.ModuleList([nn.Conv2d(c, 64, kernel_size=1) for c in in_channels])
        self.mscfm_blocks = nn.ModuleList([MSCFM(c, out_channels=64) for c in in_channels])

        self.cam_34 = CAM(channels=64)
        self.cam_23 = CAM(channels=64)
        self.cam_12 = CAM(channels=64)

        self.topdown_34 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.topdown_23 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.topdown_12 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)

        self.side_heads = nn.ModuleList([nn.Conv2d(64, 1, kernel_size=1) for _ in range(4)])
        self.final_fuse = nn.Conv2d(4, 1, kernel_size=1)
        self.final_plain = nn.Conv2d(64, 1, kernel_size=1)

    def _upsample_to_input(self, x: torch.Tensor, input_hw: List[int]) -> torch.Tensor:
        return F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> Dict[str, Union[List[torch.Tensor], torch.Tensor]]:
        input_hw = list(x.shape[-2:])

        e1, e2, e3, e4 = self.backbone(x)
        backbone_feats = [e1, e2, e3, e4]

        if self.use_mscfm:
            m1 = self.mscfm_blocks[0](e1)
            m2 = self.mscfm_blocks[1](e2)
            m3 = self.mscfm_blocks[2](e3)
            m4 = self.mscfm_blocks[3](e4)
        else:
            m1 = self.proj_blocks[0](e1)
            m2 = self.proj_blocks[1](e2)
            m3 = self.proj_blocks[2](e3)
            m4 = self.proj_blocks[3](e4)
        mscfm_feats = [m1, m2, m3, m4]

        if self.use_cam:
            a4 = m4
            a3 = self.cam_34(m3, a4)
            a2 = self.cam_23(m2, a3)
            a1 = self.cam_12(m1, a2)
        else:
            a4 = m4
            a3 = m3 + self.topdown_34(F.interpolate(a4, size=m3.shape[-2:], mode="bilinear", align_corners=False))
            a2 = m2 + self.topdown_23(F.interpolate(a3, size=m2.shape[-2:], mode="bilinear", align_corners=False))
            a1 = m1 + self.topdown_12(F.interpolate(a2, size=m1.shape[-2:], mode="bilinear", align_corners=False))
        cam_feats = [a1, a2, a3, a4]

        if self.model_variant == "baseline_plain":
            side_logits = []
            final_logit = self._upsample_to_input(self.final_plain(a1), input_hw)
        else:
            s1 = self._upsample_to_input(self.side_heads[0](a1), input_hw)
            s2 = self._upsample_to_input(self.side_heads[1](a2), input_hw)
            s3 = self._upsample_to_input(self.side_heads[2](a3), input_hw)
            s4 = self._upsample_to_input(self.side_heads[3](a4), input_hw)
            side_logits = [s1, s2, s3, s4]

            if self.model_variant == "baseline_deepsup":
                final_logit = s1
            else:
                s_cat = torch.cat(side_logits, dim=1)
                final_logit = self.final_fuse(s_cat)

        return {
            "model_variant": self.model_variant,
            "backbone_feats": backbone_feats,
            "mscfm_feats": mscfm_feats,
            "cam_feats": cam_feats,
            "side_logits": side_logits,
            "final_logit": final_logit,
        }
