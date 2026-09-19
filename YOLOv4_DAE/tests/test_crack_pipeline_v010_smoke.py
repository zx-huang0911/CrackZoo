from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.crack_pipeline_v010 import BasicCrackModel, FullCrackPipeline


def test_shape_smoke() -> None:
    x = torch.randn(1, 3, 416, 416)

    basic = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=4).eval()
    y13, y26 = basic.backbone_afpn(x)

    assert tuple(y13.shape[:2]) == (1, basic.backbone_afpn.out13_channels)
    assert tuple(y13.shape[-2:]) == (13, 13)
    assert tuple(y26.shape[:2]) == (1, basic.backbone_afpn.out26_channels)
    assert tuple(y26.shape[-2:]) == (26, 26)

    pred13_raw, pred26_raw, kept_detections, raw_mask = basic(x)
    expected_channels = 3 * (4 + 6)
    assert tuple(pred13_raw.shape) == (1, expected_channels, 13, 13)
    assert tuple(pred26_raw.shape) == (1, expected_channels, 26, 26)
    assert tuple(raw_mask.shape) == (1, 1, 416, 416)
    assert isinstance(kept_detections, list)


def test_forward_smoke() -> None:
    x = torch.randn(1, 3, 416, 416)

    basic = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=4).eval()
    pred13_raw, pred26_raw, kept_detections, raw_mask = basic(x)
    assert pred13_raw.ndim == 4
    assert pred26_raw.ndim == 4
    assert isinstance(kept_detections, list)
    assert tuple(raw_mask.shape) == (1, 1, 416, 416)

    full = FullCrackPipeline(cfg_path="cfg/yolov4-tiny.cfg", num_classes=4).eval()
    pred13_raw, pred26_raw, kept_detections, raw_mask, clean_mask = full(x)
    assert pred13_raw.ndim == 4
    assert pred26_raw.ndim == 4
    assert isinstance(kept_detections, list)
    assert tuple(raw_mask.shape) == (1, 1, 416, 416)
    assert tuple(clean_mask.shape) == (1, 1, 416, 416)


def test_decode_nms_threshold_smoke() -> None:
    x = torch.zeros(1, 3, 416, 416)
    basic = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=4).eval()

    with torch.no_grad():
        y13, y26 = basic.backbone_afpn(x)

        pred13_raw = torch.full((1, basic.pred_head.pred_channels, 13, 13), -20.0)
        pred26_raw = torch.full((1, basic.pred_head.pred_channels, 26, 26), -20.0)

        # Force one strong detection candidate in 26x26 scale, anchor 0, class 0, threshold channel.
        no = basic.pred_head.no
        anchor_idx = 0
        gx, gy = 5, 6
        base = anchor_idx * no

        pred26_raw[0, base + 0, gy, gx] = 0.0
        pred26_raw[0, base + 1, gy, gx] = 0.0
        pred26_raw[0, base + 2, gy, gx] = 0.0
        pred26_raw[0, base + 3, gy, gx] = 0.0
        pred26_raw[0, base + 4, gy, gx] = 8.0
        pred26_raw[0, base + 5, gy, gx] = 8.0
        pred26_raw[0, base + 6, gy, gx] = -8.0
        pred26_raw[0, base + 7, gy, gx] = -8.0
        pred26_raw[0, base + 8, gy, gx] = -8.0
        pred26_raw[0, base + 9, gy, gx] = 0.2

        decoded = basic.pred_head._decode(pred13_raw, pred26_raw)
        kept = basic.pred_head.run_nms(decoded)
        raw_mask = basic.pred_head.build_raw_mask_from_detections(x, kept)

    assert len(kept) == 1
    assert tuple(raw_mask.shape) == (1, 1, 416, 416)

    # Confirm threshold field survives decode+nms flow when there is a kept detection.
    if len(kept[0]) > 0:
        thr = kept[0][0].threshold
        assert 0.0 <= float(thr.item()) <= 1.0


if __name__ == "__main__":
    test_shape_smoke()
    test_forward_smoke()
    test_decode_nms_threshold_smoke()
    print("All v0.1.0 crack pipeline smoke tests passed.")
