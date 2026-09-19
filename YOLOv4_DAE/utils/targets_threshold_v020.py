from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def build_targets_with_threshold_v020(
    preds: Sequence[torch.Tensor],
    targets: torch.Tensor,
    anchor_vecs: Sequence[torch.Tensor],
    anchor_t: float = 4.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[Tuple[torch.Tensor, ...]], List[torch.Tensor], List[torch.Tensor]]:
    """YOLO target assignment extended with threshold targets on positive anchors only.

    Inputs:
      preds: list of [B, na, ny, nx, no] logits per scale.
      targets: [N, 7] with columns [batch_idx, cls, x, y, w, h, thr].
      anchor_vecs: list of [na, 2] anchors in grid units per scale.
    """

    nt = targets.shape[0]
    tcls: List[torch.Tensor] = []
    tbox: List[torch.Tensor] = []
    indices: List[Tuple[torch.Tensor, ...]] = []
    anch: List[torch.Tensor] = []
    tthr: List[torch.Tensor] = []

    gain = torch.ones(7, device=targets.device)
    off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()

    g = 0.5
    for i, pi in enumerate(preds):
        anchors = anchor_vecs[i].to(targets.device)
        gain[2:6] = torch.tensor(pi.shape, device=targets.device)[[3, 2, 3, 2]]

        a, t, offsets = [], targets * gain, 0
        if nt:
            na = anchors.shape[0]
            at = torch.arange(na, device=targets.device).view(na, 1).repeat(1, nt)
            r = t[None, :, 4:6] / anchors[:, None]
            j = torch.max(r, 1.0 / r).max(2)[0] < anchor_t
            a, t = at[j], t.repeat(na, 1, 1)[j]

            gxy = t[:, 2:4]
            z = torch.zeros_like(gxy)
            j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
            l, m = ((gxy % 1.0 > (1.0 - g)) & (gxy < (gain[[2, 3]] - 1.0))).T
            a = torch.cat((a, a[j], a[k], a[l], a[m]), 0)
            t = torch.cat((t, t[j], t[k], t[l], t[m]), 0)
            offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

        if t.numel() == 0:
            empty_long = torch.zeros((0,), dtype=torch.long, device=targets.device)
            empty_float = torch.zeros((0,), dtype=torch.float32, device=targets.device)
            indices.append((empty_long, empty_long, empty_long, empty_long))
            tbox.append(torch.zeros((0, 4), dtype=torch.float32, device=targets.device))
            anch.append(torch.zeros((0, 2), dtype=torch.float32, device=targets.device))
            tcls.append(empty_long)
            tthr.append(empty_float)
            continue

        b, c = t[:, :2].long().T
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        thr = t[:, 6].float().clamp(0.0, 1.0)

        gij = (gxy - offsets).long()
        gi, gj = gij.T

        max_gj = int(gain[3].item()) - 1
        max_gi = int(gain[2].item()) - 1
        indices.append((
            b,
            a,
            gj.clamp_(0, max_gj),
            gi.clamp_(0, max_gi),
        ))
        tbox.append(torch.cat((gxy - gij, gwh), 1))
        anch.append(anchors[a])
        tcls.append(c)
        tthr.append(thr)

    return tcls, tbox, indices, anch, tthr
