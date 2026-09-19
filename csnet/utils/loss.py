import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


########### Task 5 Loss Function Enhancement start ###########
class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss.

    Implements the weighted generalized dice loss described in the CSNet
    paper.  This loss mitigates class imbalance by weighting each class
    inversely proportional to the squared area of the class in the ground
    truth.  It is suitable for segmentation tasks with highly
    imbalanced object and background pixels, such as pavement crack
    segmentation.

    Args:
        epsilon: Small constant added to denominators to avoid division
            by zero.
        ignore_index: Label index to ignore when computing loss (e.g.,
            255 in Pascal VOC).  Ignored pixels do not contribute to
            the loss.
    """

    def __init__(self, epsilon: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Inputs: (N, C, H, W), targets: (N, H, W)
        N, C, H, W = inputs.size()
        device = inputs.device
        # create one‑hot encoding of targets ignoring ignore_index
        mask = (targets != self.ignore_index)
        # flatten spatial dims and remove ignored pixels
        targets_masked = targets.clone()
        targets_masked[~mask] = 0
        # one‑hot encode
        targets_onehot = torch.zeros((N, C, H, W), dtype=inputs.dtype, device=device)
        targets_onehot.scatter_(1, targets_masked.unsqueeze(1), 1)
        # zero out ignored pixels in both predictions and targets
        inputs = inputs * mask.unsqueeze(1)
        targets_onehot = targets_onehot * mask.unsqueeze(1)
        # compute prediction probabilities
        probs = F.softmax(inputs, dim=1)
        # compute class weights
        class_sums = targets_onehot.sum(dim=(0, 2, 3))
        weights = 1.0 / (class_sums.clamp(min=self.epsilon) ** 2)
        # compute numerator and denominator
        intersection = (probs * targets_onehot).sum(dim=(0, 2, 3))
        union = (probs + targets_onehot).sum(dim=(0, 2, 3))
        # dice score per class
        dice_score = (2 * weights * intersection + self.epsilon) / (weights * union + self.epsilon)
        loss = 1.0 - dice_score.mean()
        return loss
########### Task 5 Loss Function Enhancement end ###########
