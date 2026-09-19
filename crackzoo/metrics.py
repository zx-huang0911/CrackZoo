from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


EPS = 1e-7


def binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2 or not arr.size:
        raise ValueError("Mask must be a non-empty 2D array")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 255:
        raise ValueError("Mask values must be finite in [0,1] or [0,255]")
    if arr.dtype == np.bool_:
        return arr
    if arr.max(initial=0) > 1:
        arr = arr.astype(np.float32) / 255.0
    return arr >= threshold


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, int]:
    gt = binarize(y_true, threshold)
    pred = binarize(y_pred, threshold)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred, threshold)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    iou = tp / (tp + fp + fn + EPS)
    accuracy = (tp + tn) / (tp + fp + fn + tn + EPS)
    specificity = tn / (tn + fp + EPS)
    dice = 2 * tp / (2 * tp + fp + fn + EPS)
    return {
        **{k: float(v) for k, v in c.items()},
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dice": float(dice),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
    }


def grid_mask(mask: np.ndarray, grid_size: int = 16, positive_threshold: float = 0.05) -> np.ndarray:
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")
    if not 0 < positive_threshold <= 1:
        raise ValueError("positive_threshold must be in (0, 1]")
    binary = binarize(mask)
    h, w = binary.shape[-2:]
    rows = int(np.ceil(h / grid_size))
    cols = int(np.ceil(w / grid_size))
    out = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        y1, y2 = r * grid_size, min((r + 1) * grid_size, h)
        for c in range(cols):
            x1, x2 = c * grid_size, min((c + 1) * grid_size, w)
            cell = binary[y1:y2, x1:x2]
            out[r, c] = bool(cell.size and cell.mean() >= positive_threshold)
    return out


def grid_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    grid_size: int = 16,
    positive_threshold: float = 0.05,
) -> Dict[str, float]:
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError("Original mask dimensions must match before gridding")
    gt_grid = grid_mask(y_true, grid_size=grid_size, positive_threshold=positive_threshold)
    pred_grid = grid_mask(y_pred, grid_size=grid_size, positive_threshold=positive_threshold)
    metrics = binary_metrics(gt_grid, pred_grid, threshold=0.5)
    gt_ratio = float(gt_grid.mean())
    pred_ratio = float(pred_grid.mean())
    return {
        "grid_precision": metrics["precision"],
        "grid_recall": metrics["recall"],
        "grid_f1": metrics["f1"],
        "grid_iou": metrics["iou"],
        "gt_grid_ratio": gt_ratio,
        "pred_grid_ratio": pred_ratio,
        "crack_grid_ratio_error": abs(pred_ratio - gt_ratio),
    }


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    # A square Chebyshev neighbourhood, identical on every installation.
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy+h, dx:dx+w]
    return out


def boundary_f1(y_true: np.ndarray, y_pred: np.ndarray, tolerance_px: int = 2) -> float:
    if isinstance(tolerance_px, bool) or not isinstance(tolerance_px, int) or tolerance_px < 0:
        raise ValueError("tolerance_px must be a non-negative integer")
    gt = binarize(y_true)
    pred = binarize(y_pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0
    gt_dil = _dilate(gt, tolerance_px)
    pred_dil = _dilate(pred, tolerance_px)
    precision = np.logical_and(pred, gt_dil).sum() / (pred.sum() + EPS)
    recall = np.logical_and(gt, pred_dil).sum() / (gt.sum() + EPS)
    return float(2 * precision * recall / (precision + recall + EPS))


def cldice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return float("nan")
    gt = binarize(y_true)
    pred = binarize(y_pred)
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, pred={pred.shape}")
    if not gt.any() and not pred.any():
        return 1.0
    if not gt.any() or not pred.any():
        return 0.0
    skel_gt = skeletonize(gt)
    skel_pred = skeletonize(pred)
    tprec = np.logical_and(skel_pred, gt).sum() / (skel_pred.sum() + EPS)
    tsens = np.logical_and(skel_gt, pred).sum() / (skel_gt.sum() + EPS)
    return float(2 * tprec * tsens / (tprec + tsens + EPS))


def aggregate(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row if isinstance(row[key], (int, float))})
    result = {}
    for key in keys:
        values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
        result[key] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return result
