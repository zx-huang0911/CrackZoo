from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_metrics_csv(csv_path: Path, row: Dict[str, float]) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    m = (mask.astype(np.uint8) * 255)
    return np.stack([m, m, m], axis=-1)


def _error_map_rgb(mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    # FP in red, FN in green, TP in white.
    fp = np.logical_and(pred == 1, mask == 0)
    fn = np.logical_and(pred == 0, mask == 1)
    tp = np.logical_and(pred == 1, mask == 1)

    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[fp] = np.array([255, 0, 0], dtype=np.uint8)
    out[fn] = np.array([0, 255, 0], dtype=np.uint8)
    out[tp] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _any_positive_grid(mask: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = mask.shape
    pad_h = (block - (h % block)) % block
    pad_w = (block - (w % block)) % block
    padded = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    hh, ww = padded.shape
    grid_h = hh // block
    grid_w = ww // block
    grouped = padded.reshape(grid_h, block, grid_w, block)
    # OR rule: if any crack pixel exists in a 16x16 block, mark block as crack.
    vote = grouped.sum(axis=(1, 3))
    pooled = (vote > 0).astype(np.uint8)

    up = np.repeat(np.repeat(pooled, block, axis=0), block, axis=1)
    up = up[:h, :w]
    return up


def _grid_visual(mask: np.ndarray, block: int = 16) -> np.ndarray:
    up = _any_positive_grid(mask, block=block)
    rgb = _mask_to_rgb(up)
    rgb[::block, :, :] = 100
    rgb[:, ::block, :] = 100
    return rgb


def _label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), text, fill=(255, 140, 0))
    return np.array(img)


def save_val_composite(out_dir: Path, name: str, image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> None:
    ensure_dir(out_dir)
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gt_rgb = _mask_to_rgb(mask)
    pred_rgb = _mask_to_rgb(pred)
    err_rgb = _error_map_rgb(mask, pred)
    gt_grid_rgb = _grid_visual(mask, block=16)
    pred_grid_rgb = _grid_visual(pred, block=16)

    p1 = _label_panel(image_u8, "Input")
    p2 = _label_panel(gt_rgb, "GT")
    p3 = _label_panel(pred_rgb, "Pred")
    p4 = _label_panel(err_rgb, "Error")
    p5 = _label_panel(gt_grid_rgb, "GT Grid x16")
    p6 = _label_panel(pred_grid_rgb, "Pred Grid x16")

    h, w = image_u8.shape[:2]
    canvas = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = p1
    canvas[0:h, w : 2 * w] = p2
    canvas[0:h, 2 * w : 3 * w] = p3
    canvas[h : 2 * h, 0:w] = p4
    canvas[h : 2 * h, w : 2 * w] = p5
    canvas[h : 2 * h, 2 * w : 3 * w] = p6

    Image.fromarray(canvas).save(out_dir / f"{name}_panel6.png")
