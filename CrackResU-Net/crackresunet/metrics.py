from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BinaryStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    # Tolerance-aware counters for crack matching (default tolerance = 2 px).
    tol_matched_pred: int = 0
    tol_total_pred: int = 0
    tol_matched_gt: int = 0
    tol_total_gt: int = 0

    @staticmethod
    def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return binary.astype(bool)

        src = binary.astype(bool)
        h, w = src.shape
        out = np.zeros_like(src, dtype=bool)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue

                if dy >= 0:
                    ys_src = slice(0, h - dy)
                    ys_dst = slice(dy, h)
                else:
                    ys_src = slice(-dy, h)
                    ys_dst = slice(0, h + dy)

                if dx >= 0:
                    xs_src = slice(0, w - dx)
                    xs_dst = slice(dx, w)
                else:
                    xs_src = slice(-dx, w)
                    xs_dst = slice(0, w + dx)

                out[ys_dst, xs_dst] |= src[ys_src, xs_src]

        return out

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        p = preds.astype(bool)
        t = targets.astype(bool)
        self.tp += int(np.logical_and(p, t).sum())
        self.fp += int(np.logical_and(p, np.logical_not(t)).sum())
        self.fn += int(np.logical_and(np.logical_not(p), t).sum())
        self.tn += int(np.logical_and(np.logical_not(p), np.logical_not(t)).sum())

    def update_tolerant(self, preds: np.ndarray, targets: np.ndarray, radius: int = 2) -> None:
        p = preds.astype(bool)
        t = targets.astype(bool)

        dil_t = self._dilate(t, radius)
        dil_p = self._dilate(p, radius)

        matched_pred = np.logical_and(p, dil_t).sum()
        matched_gt = np.logical_and(t, dil_p).sum()

        self.tol_matched_pred += int(matched_pred)
        self.tol_total_pred += int(p.sum())
        self.tol_matched_gt += int(matched_gt)
        self.tol_total_gt += int(t.sum())

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2.0 * self.tp) / (2.0 * self.tp + self.fp + self.fn + eps)

        tol_precision = self.tol_matched_pred / (self.tol_total_pred + eps)
        tol_recall = self.tol_matched_gt / (self.tol_total_gt + eps)
        tol_f1 = (2.0 * tol_precision * tol_recall) / (tol_precision + tol_recall + eps)
        return {
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1),
            "IoU": float(iou),
            "Dice": float(dice),
            "TolPrecision": float(tol_precision),
            "TolRecall": float(tol_recall),
            "TolF1": float(tol_f1),
        }
