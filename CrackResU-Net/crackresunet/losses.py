from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_oh, dims)
        union = torch.sum(probs, dims) + torch.sum(targets_oh, dims)
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CrackLoss(nn.Module):
    def __init__(self, aux_weight: float = 1.0) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        main_logits = outputs["main_logits"]
        main_ce = self.ce(main_logits, targets)
        main_dice = self.dice(main_logits, targets)
        main_loss = main_ce + main_dice

        aux_logits = outputs.get("aux_logits")
        aux_loss = torch.tensor(0.0, device=targets.device)
        if aux_logits is not None:
            aux_loss = self.dice(aux_logits, targets)

        total_loss = main_loss + self.aux_weight * aux_loss
        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "aux_loss": aux_loss,
            "main_ce": main_ce,
            "main_dice": main_dice,
        }
