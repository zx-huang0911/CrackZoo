import os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class Denormalize:
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean / std
        self._std = 1 / std

    def __call__(self, tensor: np.ndarray):
        return (tensor - self._mean.reshape(-1, 1, 1)) / self._std.reshape(-1, 1, 1)


def generate_grid_view(mask_tensor: torch.Tensor, grid_size: int = 16) -> np.ndarray:
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0)

    mask_tensor = mask_tensor.float()
    grid_map = F.max_pool2d(mask_tensor, kernel_size=grid_size, stride=grid_size)
    h, w = mask_tensor.shape[-2:]
    grid_up = F.interpolate(grid_map, size=(h, w), mode="nearest")
    return grid_up.squeeze().cpu().numpy().astype(np.uint8)


def sigmoid_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).to(torch.uint8)


def save_prediction_canvas(save_path, image_np, gt_rgb, pred_rgb, grid_gt_rgb, grid_pred_rgb, grid_size: int):
    h, w, _ = image_np.shape
    canvas = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)

    canvas[0:h, 0:w, :] = image_np
    canvas[0:h, w:2 * w, :] = gt_rgb
    canvas[0:h, 2 * w:3 * w, :] = pred_rgb

    canvas[h:2 * h, 0:w, :] = grid_gt_rgb
    canvas[h:2 * h, w:2 * w, :] = grid_pred_rgb

    labels = [
        ("Original", (10, 30)),
        ("Ground Truth", (w + 10, 30)),
        ("Prediction", (2 * w + 10, 30)),
        (f"Grid GT (s={grid_size})", (10, h + 30)),
        (f"Grid Pred (s={grid_size})", (w + 10, h + 30)),
    ]
    for txt, pos in labels:
        cv2.putText(canvas, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(canvas).save(save_path)


def _prob_to_rgb(prob_map: np.ndarray) -> np.ndarray:
    prob_255 = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    return np.stack([prob_255, prob_255, prob_255], axis=-1)


def save_diagnostic_pack(
    save_path: str,
    image_np: np.ndarray,
    gt_mask: np.ndarray,
    side_probs: list,
    final_prob: np.ndarray,
    threshold: float = 0.5,
):
    """Save a stable 2x5 diagnostic canvas.

    Includes input, GT, side1-4, final raw probability, final binary mask, overlay, and error map.
    """
    h, w, _ = image_np.shape
    canvas = np.zeros((h * 2, w * 5, 3), dtype=np.uint8)

    pred_mask = (final_prob >= threshold).astype(np.uint8)
    gt_rgb = np.stack([gt_mask * 255, gt_mask * 255, gt_mask * 255], axis=-1).astype(np.uint8)
    pred_bin_rgb = np.stack([pred_mask * 255, pred_mask * 255, pred_mask * 255], axis=-1).astype(np.uint8)
    final_raw_rgb = _prob_to_rgb(final_prob)

    side_rgbs = []
    for i in range(4):
        if i < len(side_probs):
            side_rgbs.append(_prob_to_rgb(side_probs[i]))
        else:
            side_rgbs.append(np.zeros((h, w, 3), dtype=np.uint8))

    overlay = image_np.copy()
    overlay[..., 2] = np.clip(overlay[..., 2] * 0.5 + pred_mask * 255 * 0.5, 0, 255).astype(np.uint8)

    error = np.zeros((h, w, 3), dtype=np.uint8)
    fp = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn = np.logical_and(pred_mask == 0, gt_mask == 1)
    error[fp] = [255, 0, 0]
    error[fn] = [0, 255, 0]

    tiles = [
        image_np,
        gt_rgb,
        side_rgbs[0],
        side_rgbs[1],
        side_rgbs[2],
        side_rgbs[3],
        final_raw_rgb,
        pred_bin_rgb,
        overlay,
        error,
    ]

    labels = [
        "Input",
        "GT",
        "Side-1",
        "Side-2",
        "Side-3",
        "Side-4",
        "Final-Prob",
        "Final-Binary",
        "Overlay",
        "Error(FP-red/FN-green)",
    ]

    for idx, tile in enumerate(tiles):
        r = idx // 5
        c = idx % 5
        y0, y1 = r * h, (r + 1) * h
        x0, x1 = c * w, (c + 1) * w
        canvas[y0:y1, x0:x1] = tile
        cv2.putText(canvas, labels[idx], (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(canvas).save(save_path)
