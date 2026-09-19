from __future__ import annotations

from typing import Dict

import torch


EPS = 1e-9


def mask_iou_f1(pred: torch.Tensor, gt: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    """Compute batch-aggregated IoU and F1 for binary masks."""
    p = (pred > float(thr)).to(torch.float32)
    g = (gt > float(thr)).to(torch.float32)

    inter = (p * g).sum(dim=(1, 2, 3))
    union = ((p + g) > 0).to(torch.float32).sum(dim=(1, 2, 3))
    iou = inter / (union + EPS)

    tp = inter
    fp = (p * (1.0 - g)).sum(dim=(1, 2, 3))
    fn = ((1.0 - p) * g).sum(dim=(1, 2, 3))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + EPS)

    return {
        "iou": float(iou.mean().item()),
        "f1": float(f1.mean().item()),
    }


def dice_loss_from_probs(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    p = pred.to(torch.float32)
    g = gt.to(torch.float32)
    inter = (p * g).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + EPS) / (den + EPS)
    return 1.0 - dice.mean()
