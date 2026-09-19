from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from models.crack_pipeline_v010 import DetectionRecord


def _to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3:
        arr = image.detach().cpu().float().permute(1, 2, 0).numpy()
    else:
        raise ValueError("Expected image tensor with shape [3, H, W]")

    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_uint8_mask(mask: torch.Tensor) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("Expected mask tensor with shape [H, W] or [1, H, W]")
    arr = mask.detach().cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def draw_kept_detections(
    image_rgb: torch.Tensor,
    kept_detections: Sequence[DetectionRecord],
    class_names: Optional[Sequence[str]] = None,
    show_score: bool = True,
    show_threshold: bool = True,
) -> np.ndarray:
    canvas = _to_uint8_rgb(image_rgb)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for det in kept_detections:
        box = det.box.detach().cpu().float().numpy()
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        score = float(det.score.detach().cpu().item())
        class_id = int(det.class_id.detach().cpu().item())
        threshold = float(det.threshold.detach().cpu().item())

        color = (40, 220, 40)
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), color, 2)

        class_text = str(class_id)
        if class_names is not None and 0 <= class_id < len(class_names):
            class_text = class_names[class_id]

        text_parts = [f"cls:{class_text}"]
        if show_score:
            text_parts.append(f"s:{score:.3f}")
        if show_threshold:
            text_parts.append(f"thr:{threshold:.3f}")
        text = " ".join(text_parts)
        cv2.putText(canvas_bgr, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def draw_gt_boxes(
    image_rgb: torch.Tensor,
    gt_boxes_xyxy: Sequence[Sequence[float]],
    gt_classes: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    canvas = _to_uint8_rgb(image_rgb)
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(gt_boxes_xyxy):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), (255, 180, 20), 2)
        if gt_classes is not None and i < len(gt_classes):
            cls_id = int(gt_classes[i])
            cls_text = str(cls_id)
            if class_names is not None and 0 <= cls_id < len(class_names):
                cls_text = class_names[cls_id]
            cv2.putText(canvas_bgr, f"gt:{cls_text}", (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 20), 1, cv2.LINE_AA)

    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def _add_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = img_rgb.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w - 1, min(28, h - 1)), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _mask_overlay_rgb(img_rgb: np.ndarray, raw_mask_u8: np.ndarray, gt_mask_u8: Optional[np.ndarray]) -> np.ndarray:
    out = img_rgb.copy()
    raw_m = raw_mask_u8 > 127
    if gt_mask_u8 is not None:
        gt_m = gt_mask_u8 > 127
    else:
        gt_m = np.zeros_like(raw_m)

    # GT in green, prediction in red, overlap tends to yellow.
    out[gt_m, 1] = 255
    out[raw_m, 2] = 255
    out = cv2.addWeighted(img_rgb, 0.55, out, 0.45, 0.0)
    return out


def save_debug_outputs(
    image_rgb: torch.Tensor,
    kept_detections: Sequence[DetectionRecord],
    save_dir: str,
    stem: str,
    raw_mask: Optional[torch.Tensor] = None,
    clean_mask: Optional[torch.Tensor] = None,
    class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_files: List[str] = []

    det_img = draw_kept_detections(image_rgb, kept_detections, class_names=class_names)
    det_path = out_dir / f"{stem}_detections.png"
    cv2.imwrite(str(det_path), cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
    saved_files.append(str(det_path))

    if raw_mask is not None:
        raw_path = out_dir / f"{stem}_raw_mask.png"
        cv2.imwrite(str(raw_path), _to_uint8_mask(raw_mask))
        saved_files.append(str(raw_path))

    if clean_mask is not None:
        clean_path = out_dir / f"{stem}_clean_mask.png"
        cv2.imwrite(str(clean_path), _to_uint8_mask(clean_mask))
        saved_files.append(str(clean_path))

    return saved_files


def save_debug_panel(
    image_rgb: torch.Tensor,
    kept_detections: Sequence[DetectionRecord],
    raw_mask: torch.Tensor,
    save_path: str,
    gt_mask: Optional[torch.Tensor] = None,
    gt_boxes_xyxy: Optional[Sequence[Sequence[float]]] = None,
    gt_classes: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """Saves an 8-view debug panel for end-to-end detection/threshold/mask inspection."""
    img = _to_uint8_rgb(image_rgb)
    raw_u8 = _to_uint8_mask(raw_mask)
    gt_u8 = _to_uint8_mask(gt_mask) if gt_mask is not None else None

    gt_boxes_xyxy = gt_boxes_xyxy or []
    gt_classes = gt_classes or []

    pane1 = _add_title(img, "1) RGB")
    pane2 = _add_title(
        draw_gt_boxes(image_rgb, gt_boxes_xyxy, gt_classes=gt_classes, class_names=class_names),
        "2) GT boxes",
    )
    pane3 = _add_title(
        draw_kept_detections(image_rgb, kept_detections, class_names=class_names, show_score=True, show_threshold=True),
        "3) Pred boxes",
    )
    pane4 = _add_title(
        draw_kept_detections(image_rgb, kept_detections, class_names=class_names, show_score=True, show_threshold=False),
        "4) Pred scores",
    )
    pane5 = _add_title(
        draw_kept_detections(image_rgb, kept_detections, class_names=class_names, show_score=False, show_threshold=True),
        "5) Pred thresholds",
    )
    pane6 = _add_title(cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB), "6) Raw synth mask")
    if gt_u8 is not None:
        pane7_base = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB)
    else:
        pane7_base = np.zeros_like(pane6)
    pane7 = _add_title(pane7_base, "7) GT mask")
    pane8 = _add_title(_mask_overlay_rgb(img, raw_u8, gt_u8), f"8) Overlay (NMS kept={len(kept_detections)})")

    top = np.concatenate([pane1, pane2, pane3, pane4], axis=1)
    bot = np.concatenate([pane5, pane6, pane7, pane8], axis=1)
    panel = np.concatenate([top, bot], axis=0)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return str(out)
