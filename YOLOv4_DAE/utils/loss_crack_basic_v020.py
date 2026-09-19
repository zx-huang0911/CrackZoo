from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import bbox_iou
from utils.loss import FocalLoss, smooth_BCE
from utils.targets_threshold_v020 import build_targets_with_threshold_v020


class CrackBasicLossV020:
    """v0.2.0 basic-model loss: loc + obj + cls + alpha * threshold(MSE)."""

    def __init__(
        self,
        num_classes: int,
        alpha_threshold_loss: float = 50.0,
        anchor_t: float = 4.0,
        box_weight: float = 0.05,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
        cls_pw: float = 1.0,
        obj_pw: float = 1.0,
        fl_gamma: float = 0.0,
        gr: float = 1.0,
    ) -> None:
        self.num_classes = int(num_classes)
        self.alpha_threshold_loss = float(alpha_threshold_loss)
        self.anchor_t = float(anchor_t)
        self.box_weight = float(box_weight)
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)
        self.gr = float(gr)

        self.cp, self.cn = smooth_BCE(eps=0.0)
        self.mse = nn.MSELoss(reduction="mean")

        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_pw]))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_pw]))

        if fl_gamma > 0:
            self.bce_cls = FocalLoss(self.bce_cls, fl_gamma)
            self.bce_obj = FocalLoss(self.bce_obj, fl_gamma)

    def __call__(
        self,
        preds: Sequence[torch.Tensor],
        targets: torch.Tensor,
        anchor_vecs: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = preds[0].device
        if targets.device != device:
            targets = targets.to(device)
        self.bce_cls = self.bce_cls.to(device)
        self.bce_obj = self.bce_obj.to(device)

        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lthr = torch.zeros(1, device=device)

        # Diagnostics for threshold supervision on responsible positives only.
        num_gt_boxes = int(targets.shape[0])
        num_positive_anchors = 0
        thr_pred_sum = torch.zeros(1, device=device)
        thr_tgt_sum = torch.zeros(1, device=device)
        thr_abs_err_sum = torch.zeros(1, device=device)
        lthr_sum = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors, tthr = build_targets_with_threshold_v020(
            preds=preds,
            targets=targets,
            anchor_vecs=anchor_vecs,
            anchor_t=self.anchor_t,
        )

        no = len(preds)
        balance = [4.0, 1.0, 0.4] if no == 3 else [1.0] * no
        if no == 2:
            balance = [1.0, 1.0]

        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                num_positive_anchors += int(n)
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                if self.num_classes > 1:
                    pcls = ps[:, 5 : 5 + self.num_classes]
                    t = torch.full_like(pcls, self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.bce_cls(pcls, t)

                # Threshold supervision only on responsible positive anchors.
                pthr = ps[:, 5 + self.num_classes].sigmoid()
                tthr_i = tthr[i].to(device).clamp(0.0, 1.0)

                # Use explicit sum reduction, then normalize globally by positive anchors.
                lthr_sum += F.mse_loss(pthr, tthr_i, reduction="sum")
                thr_pred_sum += pthr.sum()
                thr_tgt_sum += tthr_i.sum()
                thr_abs_err_sum += torch.abs(pthr - tthr_i).sum()

            lobj += self.bce_obj(pi[..., 4], tobj) * balance[i]

        s = 3.0 / no
        lbox *= self.box_weight * s
        lobj *= self.obj_weight * s
        lcls *= self.cls_weight * s
        if num_positive_anchors > 0:
            lthr = lthr_sum / float(num_positive_anchors)
        else:
            lthr = lthr_sum * 0.0

        total = lbox + lobj + lcls + self.alpha_threshold_loss * lthr
        bs = preds[0].shape[0]
        total = total * bs

        if num_positive_anchors > 0:
            mean_thr_pred = (thr_pred_sum / float(num_positive_anchors)).detach()
            mean_thr_tgt = (thr_tgt_sum / float(num_positive_anchors)).detach()
            mean_thr_abs_err = (thr_abs_err_sum / float(num_positive_anchors)).detach()
        else:
            mean_thr_pred = torch.zeros(1, device=device)
            mean_thr_tgt = torch.zeros(1, device=device)
            mean_thr_abs_err = torch.zeros(1, device=device)

        items = {
            "loss_total": total.detach(),
            "loss_loc": lbox.detach(),
            "loss_conf": lobj.detach(),
            "loss_cls": lcls.detach(),
            "loss_thr": lthr.detach(),
            "num_gt_boxes": torch.tensor(float(num_gt_boxes), device=device),
            "num_positive_anchors": torch.tensor(float(num_positive_anchors), device=device),
            "thr_pred_mean_pos": mean_thr_pred,
            "thr_target_mean_pos": mean_thr_tgt,
            "thr_abs_err_mean_pos": mean_thr_abs_err,
            "alpha_threshold_loss": torch.tensor(float(self.alpha_threshold_loss), device=device),
        }
        return total, items
