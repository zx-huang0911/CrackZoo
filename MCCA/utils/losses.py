import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def binary_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()

    dims = (1, 2, 3)
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_batch_pos_weight(target: torch.Tensor, mode: str, fixed_pos_weight: float, max_pos_weight: float) -> Optional[float]:
    mode = mode.lower()
    if mode == "fixed":
        return float(fixed_pos_weight)
    if mode not in {"auto", "capped_auto"}:
        return None

    # target is [N, 1, H, W] float in {0,1}
    pos = float(target.sum().item())
    total = float(target.numel())
    neg = max(total - pos, 1.0)
    pos = max(pos, 1.0)
    pw = neg / pos
    if mode == "capped_auto":
        pw = min(pw, float(max_pos_weight))
    return float(pw)


class MCCABinaryLoss(nn.Module):
    MODES = {"bce", "weighted_bce", "weighted_bce_dice", "weighted_bce_tversky"}

    def __init__(
        self,
        loss_name: str = "weighted_bce_dice",
        pos_weight_mode: str = "capped_auto",
        fixed_pos_weight: float = 16.0,
        max_pos_weight: float = 24.0,
        wbce_weight: float = 1.0,
        dice_weight: float = 1.0,
        side_loss_weight: float = 0.10,
        use_deepsup: bool = True,
    ):
        super().__init__()
        self.loss_name = loss_name.lower()
        if self.loss_name not in self.MODES:
            raise ValueError(f"Unknown loss_name '{loss_name}'. Available: {sorted(self.MODES)}")

        self.pos_weight_mode = pos_weight_mode.lower()
        self.fixed_pos_weight = fixed_pos_weight
        self.max_pos_weight = max_pos_weight
        self.wbce_weight = wbce_weight
        self.dice_weight = dice_weight
        self.side_loss_weight = side_loss_weight
        self.use_deepsup = use_deepsup

    def _bce(self, logits: torch.Tensor, target: torch.Tensor, pos_weight_value: Optional[float]) -> torch.Tensor:
        if pos_weight_value is None:
            return F.binary_cross_entropy_with_logits(logits, target)
        pos_weight = torch.tensor([float(pos_weight_value)], dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    def _single_logit_loss(self, logits: torch.Tensor, target: torch.Tensor, pos_weight_value: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self.loss_name
        if mode == "bce":
            bce = self._bce(logits, target, None)
            dice = torch.zeros((), dtype=bce.dtype, device=bce.device)
            return bce, bce, dice

        bce = self._bce(logits, target, pos_weight_value)
        if mode == "weighted_bce":
            dice = torch.zeros((), dtype=bce.dtype, device=bce.device)
            return bce, bce, dice

        # weighted_bce_dice and weighted_bce_tversky fallback to dice scaffold
        dice = binary_dice_loss_from_logits(logits, target)
        combined = self.wbce_weight * bce + self.dice_weight * dice
        return combined, bce, dice

    def forward(self, outputs, target: torch.Tensor):
        target = target.float().unsqueeze(1)
        final_logit = outputs["final_logit"]
        side_logits = outputs.get("side_logits", [])

        if self.loss_name == "bce":
            pos_weight_value = None
        else:
            pos_weight_value = compute_batch_pos_weight(
                target=target,
                mode=self.pos_weight_mode,
                fixed_pos_weight=self.fixed_pos_weight,
                max_pos_weight=self.max_pos_weight,
            )

        final_loss, final_bce, final_dice = self._single_logit_loss(final_logit, target, pos_weight_value)

        if self.use_deepsup and len(side_logits) > 0:
            side_losses = []
            side_bces = []
            side_dices = []
            for side in side_logits:
                one_side_loss, one_side_bce, one_side_dice = self._single_logit_loss(side, target, pos_weight_value)
                side_losses.append(one_side_loss)
                side_bces.append(one_side_bce)
                side_dices.append(one_side_dice)
            aux_loss = torch.stack(side_losses).sum()
            aux_bce = torch.stack(side_bces).sum()
            aux_dice = torch.stack(side_dices).sum()
        else:
            aux_loss = torch.zeros((), dtype=final_loss.dtype, device=final_loss.device)
            aux_bce = torch.zeros((), dtype=final_loss.dtype, device=final_loss.device)
            aux_dice = torch.zeros((), dtype=final_loss.dtype, device=final_loss.device)

        total_loss = final_loss + self.side_loss_weight * aux_loss

        return {
            "total_loss": total_loss,
            "final_loss": final_loss.detach(),
            "final_bce": final_bce.detach(),
            "final_dice": final_dice.detach(),
            "aux_loss": aux_loss.detach(),
            "aux_bce": aux_bce.detach(),
            "aux_dice": aux_dice.detach(),
            "pos_weight": float(pos_weight_value) if pos_weight_value is not None else 1.0,
            "loss_name": self.loss_name,
        }
