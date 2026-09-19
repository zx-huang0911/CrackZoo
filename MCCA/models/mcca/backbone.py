from typing import List

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Backbone(nn.Module):
    """ResNet-18 feature extractor returning E1-E4."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return [e1, e2, e3, e4]
