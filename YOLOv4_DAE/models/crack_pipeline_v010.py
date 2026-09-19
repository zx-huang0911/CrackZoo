from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.models import Darknet
from utils.general import box_iou, non_max_suppression, xywh2xyxy


_ROUTE_LIKE = {
    "WeightedFeatureFusion",
    "FeatureConcat",
    "FeatureConcat2",
    "FeatureConcat3",
    "FeatureConcat_l",
    "ScaleChannel",
    "ScaleSpatial",
}


def _conv_bn_leaky(in_ch: int, out_ch: int, k: int = 3, s: int = 1) -> nn.Sequential:
    pad = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        attn = self.sigmoid(avg + mx)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.cam = ChannelAttention(channels, reduction=reduction)
        self.sam = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sam(self.cam(x))


class BackboneAFPN(nn.Module):
    """Reuses yolov4-tiny CSPDarknet53-tiny backbone and adds AFPN head."""

    def __init__(self, cfg_path: str = "cfg/yolov4-tiny.cfg") -> None:
        super().__init__()
        self.backbone = Darknet(cfg_path, img_size=(416, 416))

        # f26 and f13 are tapped from original tiny backbone path.
        self.f26_index = 24
        self.f13_index = 26
        self.f26_channels = 512
        self.f13_channels = 512

        self.cbam13_in = CBAM(self.f13_channels)
        self.cbam26_in = CBAM(self.f26_channels)

        self.cbl13_1 = _conv_bn_leaky(self.f13_channels, 256, k=1, s=1)
        self.cbl13_2 = _conv_bn_leaky(256, 512, k=3, s=1)

        self.conv_up = _conv_bn_leaky(256, 128, k=1, s=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cbam_fuse = CBAM(128)

        self.cbl26 = _conv_bn_leaky(self.f26_channels + 128, 256, k=3, s=1)

        self.out13_channels = 512
        self.out26_channels = 256

    def _forward_backbone_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out: List[torch.Tensor] = []
        f13: Optional[torch.Tensor] = None
        f26: Optional[torch.Tensor] = None

        for i, module in enumerate(self.backbone.module_list):
            if i > self.f13_index:
                break

            name = module.__class__.__name__
            if name in _ROUTE_LIKE:
                x = module(x, out)
            else:
                x = module(x)

            out.append(x if self.backbone.routs[i] else [])

            if i == self.f26_index:
                f26 = x
            elif i == self.f13_index:
                f13 = x

        if f13 is None or f26 is None:
            raise RuntimeError("Failed to capture f13/f26 from yolov4-tiny backbone")
        return f13, f26

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f13, f26 = self._forward_backbone_features(x)

        a13 = self.cbam13_in(f13)
        a26 = self.cbam26_in(f26)

        d13 = self.cbl13_1(a13)
        y13 = self.cbl13_2(d13)

        u26_pre = self.upsample(self.conv_up(d13))
        u26 = self.cbam_fuse(u26_pre)

        cat26 = torch.cat([a26, u26], dim=1)
        y26 = self.cbl26(cat26)

        return y13, y26


class ThresholdYOLODecode(nn.Module):
    """YOLO decode with one extra threshold channel per anchor."""

    def __init__(self, anchors: Sequence[Sequence[float]], nc: int, stride: int) -> None:
        super().__init__()
        self.anchors = torch.tensor(anchors, dtype=torch.float32)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 6
        self.stride = stride
        self.nx, self.ny = 0, 0
        self.register_buffer("grid", torch.zeros(1))
        self.register_buffer("anchor_vec", self.anchors / float(stride))
        self.register_buffer("anchor_wh", self.anchor_vec.view(1, self.na, 1, 1, 2))

    def _create_grids(self, ng: Tuple[int, int], device: torch.device) -> None:
        nx, ny = ng
        self.nx, self.ny = nx, ny
        yv, xv = torch.meshgrid(
            [torch.arange(ny, device=device), torch.arange(nx, device=device)],
            indexing="ij",
        )
        self.grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, ny, nx = p.shape
        if (self.nx, self.ny) != (nx, ny):
            self._create_grids((nx, ny), p.device)

        p = p.view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        io = p.sigmoid()
        io[..., :2] = (io[..., :2] * 2.0 - 0.5 + self.grid)
        io[..., 2:4] = (io[..., 2:4] * 2.0) ** 2 * self.anchor_wh
        io[..., :4] *= self.stride
        return io.view(bs, -1, self.no), p


@dataclass
class DetectionRecord:
    box: torch.Tensor
    score: torch.Tensor
    class_id: torch.Tensor
    threshold: torch.Tensor


class CrackPredictionHead(nn.Module):
    def __init__(
        self,
        in13_channels: int,
        in26_channels: int,
        num_classes: int = 4,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 3
        self.no = num_classes + 6
        self.pred_channels = self.num_anchors * self.no

        self.pred_conv13 = nn.Conv2d(in13_channels, self.pred_channels, kernel_size=1, stride=1, padding=0)
        self.pred_conv26 = nn.Conv2d(in26_channels, self.pred_channels, kernel_size=1, stride=1, padding=0)

        all_anchors = [
            (10.0, 14.0),
            (23.0, 27.0),
            (37.0, 58.0),
            (81.0, 82.0),
            (135.0, 169.0),
            (344.0, 319.0),
        ]
        self.decode13 = ThresholdYOLODecode([all_anchors[3], all_anchors[4], all_anchors[5]], num_classes, stride=32)
        self.decode26 = ThresholdYOLODecode([all_anchors[1], all_anchors[2], all_anchors[3]], num_classes, stride=16)

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = int(max_det)

    def _decode(self, pred13_raw: torch.Tensor, pred26_raw: torch.Tensor) -> torch.Tensor:
        d13, _ = self.decode13(pred13_raw)
        d26, _ = self.decode26(pred26_raw)
        return torch.cat([d13, d26], dim=1)

    def forward_train(
        self,
        y13: torch.Tensor,
        y26: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Training path that skips NMS/mask generation and returns raw per-scale tensors."""
        pred13_raw = self.pred_conv13(y13)
        pred26_raw = self.pred_conv26(y26)

        bs13, _, ny13, nx13 = pred13_raw.shape
        bs26, _, ny26, nx26 = pred26_raw.shape
        p13 = pred13_raw.view(bs13, self.num_anchors, self.no, ny13, nx13).permute(0, 1, 3, 4, 2).contiguous()
        p26 = pred26_raw.view(bs26, self.num_anchors, self.no, ny26, nx26).permute(0, 1, 3, 4, 2).contiguous()
        return pred13_raw, pred26_raw, [p13, p26]

    def decode_for_eval(self, pred13_raw: torch.Tensor, pred26_raw: torch.Tensor) -> torch.Tensor:
        return self._decode(pred13_raw, pred26_raw)

    def _match_thresholds(self, decoded: torch.Tensor, nms_boxes: torch.Tensor) -> List[DetectionRecord]:
        if nms_boxes.numel() == 0:
            return []

        det_records: List[DetectionRecord] = []

        src_xyxy = xywh2xyxy(decoded[:, :4])
        src_obj = decoded[:, 4]
        src_cls = decoded[:, 5 : 5 + self.num_classes]
        src_thr = decoded[:, 5 + self.num_classes]
        src_cls_conf, src_cls_id = src_cls.max(1)
        src_score = src_obj * src_cls_conf

        for det in nms_boxes:
            box = det[:4]
            score = det[4]
            cls_id = det[5]

            cls_mask = src_cls_id == cls_id.long()
            idxs = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
            if idxs.numel() == 0:
                thr = torch.tensor(0.5, device=decoded.device, dtype=decoded.dtype)
            else:
                cand_boxes = src_xyxy[idxs]
                ious = box_iou(box.unsqueeze(0), cand_boxes).squeeze(0)
                cand_scores = src_score[idxs]
                # Favor overlap first, then score.
                pick = torch.argmax(ious + 1e-3 * cand_scores)
                thr = src_thr[idxs[pick]]

            det_records.append(
                DetectionRecord(
                    box=box,
                    score=score,
                    class_id=cls_id,
                    threshold=thr,
                )
            )

        return det_records

    def run_nms(
        self,
        decoded: torch.Tensor,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        max_det: Optional[int] = None,
        nms_out: Optional[List[torch.Tensor]] = None,
    ) -> List[List[DetectionRecord]]:
        if nms_out is None:
            resolved_conf = float(self.conf_thres if conf_thres is None else conf_thres)
            resolved_iou = float(self.iou_thres if iou_thres is None else iou_thres)
            resolved_max_det = int(self.max_det if max_det is None else max_det)
            pred_for_nms = decoded[:, :, : 5 + self.num_classes]
            nms_out = non_max_suppression(
                pred_for_nms,
                conf_thres=resolved_conf,
                iou_thres=resolved_iou,
                classes=None,
                agnostic=False,
                max_det=resolved_max_det,
            )

        kept_detections: List[List[DetectionRecord]] = []
        for b in range(decoded.shape[0]):
            kept = self._match_thresholds(decoded[b], nms_out[b])
            kept_detections.append(kept)
        return kept_detections

    def build_raw_mask_from_detections(
        self,
        x_rgb: torch.Tensor,
        kept_detections: List[List[DetectionRecord]],
    ) -> torch.Tensor:
        bsz, _, h, w = x_rgb.shape
        raw_mask = torch.zeros((bsz, 1, h, w), dtype=x_rgb.dtype, device=x_rgb.device)

        # Convert to grayscale in segmentation stage only (not an input preprocess module).
        x_norm = x_rgb / 255.0 if x_rgb.max() > 1.0 else x_rgb
        gray = 0.2989 * x_norm[:, 0:1] + 0.5870 * x_norm[:, 1:2] + 0.1140 * x_norm[:, 2:3]

        for b in range(bsz):
            for det in kept_detections[b]:
                box = det.box
                thr = det.threshold.clamp(0.0, 1.0)

                x1 = int(torch.floor(box[0]).item())
                y1 = int(torch.floor(box[1]).item())
                x2 = int(torch.ceil(box[2]).item())
                y2 = int(torch.ceil(box[3]).item())

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                patch = gray[b, 0, y1:y2, x1:x2]
                local_mask = (patch <= thr).to(raw_mask.dtype)
                raw_mask[b, 0, y1:y2, x1:x2] = torch.maximum(raw_mask[b, 0, y1:y2, x1:x2], local_mask)

        return raw_mask

    def forward(
        self,
        x_rgb: torch.Tensor,
        y13: torch.Tensor,
        y26: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[DetectionRecord]], torch.Tensor]:
        pred13_raw = self.pred_conv13(y13)
        pred26_raw = self.pred_conv26(y26)

        decoded = self._decode(pred13_raw, pred26_raw)
        kept_detections = self.run_nms(decoded)
        raw_mask = self.build_raw_mask_from_detections(x_rgb, kept_detections)

        return pred13_raw, pred26_raw, kept_detections, raw_mask


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, op: int = 0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=op, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CrackDAE(nn.Module):
    """DAE structure follows paper table for v0.1.0 forward implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(1, 32)
        self.conv2 = ConvBNReLU(32, 32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ConvBNReLU(32, 64)
        self.conv5 = ConvBNReLU(64, 64)
        self.conv6 = ConvBNReLU(64, 128)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv8 = ConvTransposeBNReLU(128, 64, k=3, s=1, p=1)
        self.deconv9 = ConvTransposeBNReLU(64, 64, k=2, s=2, p=0)
        self.deconv10 = ConvTransposeBNReLU(64, 32, k=3, s=1, p=1)
        self.deconv11 = ConvTransposeBNReLU(32, 16, k=3, s=1, p=1)
        self.deconv12 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.deconv13 = ConvTransposeBNReLU(16, 8, k=2, s=2, p=0)
        self.deconv14 = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid15 = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool7(x)

        x = self.deconv8(x)
        x = self.deconv9(x)
        x = self.deconv10(x)
        x = self.deconv11(x)
        x = self.deconv12(x)

        x = self.deconv13(x)
        x = self.deconv14(x)
        x = self.sigmoid15(x)
        return x


class BasicCrackModel(nn.Module):
    def __init__(self, cfg_path: str = "cfg/yolov4-tiny.cfg", num_classes: int = 4) -> None:
        super().__init__()
        self.backbone_afpn = BackboneAFPN(cfg_path=cfg_path)
        self.pred_head = CrackPredictionHead(
            in13_channels=self.backbone_afpn.out13_channels,
            in26_channels=self.backbone_afpn.out26_channels,
            num_classes=num_classes,
        )

    def forward(
        self, x_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[DetectionRecord]], torch.Tensor]:
        y13, y26 = self.backbone_afpn(x_rgb)
        pred13_raw, pred26_raw, kept_detections, raw_mask = self.pred_head(x_rgb, y13, y26)
        return pred13_raw, pred26_raw, kept_detections, raw_mask

    def forward_train(self, x_rgb: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Training helper that keeps gradients on raw outputs and avoids NMS/mask ops."""
        y13, y26 = self.backbone_afpn(x_rgb)
        pred13_raw, pred26_raw, preds = self.pred_head.forward_train(y13, y26)
        return {
            "pred13_raw": pred13_raw,
            "pred26_raw": pred26_raw,
            "preds": preds,
        }


class FullCrackPipeline(nn.Module):
    def __init__(self, cfg_path: str = "cfg/yolov4-tiny.cfg", num_classes: int = 4) -> None:
        super().__init__()
        self.basic_model = BasicCrackModel(cfg_path=cfg_path, num_classes=num_classes)
        self.dae = CrackDAE()

    def forward(
        self, x_rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[DetectionRecord]], torch.Tensor, torch.Tensor]:
        pred13_raw, pred26_raw, kept_detections, raw_mask = self.basic_model(x_rgb)
        clean_mask = self.dae(raw_mask)
        return pred13_raw, pred26_raw, kept_detections, raw_mask, clean_mask
