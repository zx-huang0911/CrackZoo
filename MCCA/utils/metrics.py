import numpy as np


class BinarySegMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, target: np.ndarray, pred: np.ndarray):
        t = target.astype(np.bool_)
        p = pred.astype(np.bool_)
        self.tp += np.logical_and(p, t).sum()
        self.fp += np.logical_and(p, ~t).sum()
        self.fn += np.logical_and(~p, t).sum()
        self.tn += np.logical_and(~p, ~t).sum()

    def get_results(self):
        eps = 1e-7
        total = self.tp + self.tn + self.fp + self.fn
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = (2 * self.tp) / (2 * self.tp + self.fp + self.fn + eps)
        acc = (self.tp + self.tn) / (total + eps)
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        # Threshold-dependent pixel MAE for binary predictions.
        bin_mae = (self.fp + self.fn) / (total + eps)
        return {
            "IoU": float(iou),
            "Dice": float(dice),
            "Accuracy": float(acc),
            "Precision": float(precision),
            "Recall": float(recall),
            "MAE": float(bin_mae),
        }

    @staticmethod
    def to_str(results):
        return "\n" + "\n".join([f"{k}: {v:.6f}" for k, v in results.items()])


class BinaryPaperMetrics:
    """Paper-aligned optional metrics computed from probability maps."""

    def __init__(self, num_thresholds: int = 101, beta: float = 0.3):
        self.thresholds = np.linspace(0.0, 1.0, num_thresholds)
        self.beta = beta
        self.reset()

    def reset(self):
        n = len(self.thresholds)
        self.tp = np.zeros(n, dtype=np.float64)
        self.fp = np.zeros(n, dtype=np.float64)
        self.fn = np.zeros(n, dtype=np.float64)
        self.mae_sum = 0.0
        self.mae_count = 0

    def update(self, target: np.ndarray, prob: np.ndarray):
        t = target.astype(np.bool_)
        p = np.clip(prob.astype(np.float64), 0.0, 1.0)

        self.mae_sum += np.abs(p - t.astype(np.float64)).sum()
        self.mae_count += t.size

        flat_t = t.reshape(-1)
        flat_p = p.reshape(-1)
        pred_all = flat_p[:, None] >= self.thresholds[None, :]
        target_all = flat_t[:, None]

        self.tp += np.logical_and(pred_all, target_all).sum(axis=0)
        self.fp += np.logical_and(pred_all, ~target_all).sum(axis=0)
        self.fn += np.logical_and(~pred_all, target_all).sum(axis=0)

    def get_results(self):
        eps = 1e-12
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)

        beta2 = self.beta * self.beta
        f_beta = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + eps)

        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        pr_auc = np.trapz(precision_sorted, recall_sorted)

        return {
            "AIU": float(np.mean(iou)),
            "MaxFbeta": float(np.max(f_beta)),
            "MeanFbeta": float(np.mean(f_beta)),
            "MAE": float(self.mae_sum / (self.mae_count + eps)),
            "PRAUC": float(pr_auc),
        }
