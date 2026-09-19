from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dataset_v020 import create_crack_dataloader_v020
from models.crack_pipeline_v010 import BasicCrackModel, DetectionRecord
from utils.crack_debug_v010 import save_debug_panel


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _verify_label_source(name: str, p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")
    if "labels_threshold_auto_v033" not in str(p).replace('\\', '/'):
        raise RuntimeError(f"{name} must point to labels_threshold_auto_v033, got {p}")


def _load_compatible_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, List[str]]:
    own_state = model.state_dict()
    compatible: Dict[str, torch.Tensor] = {}
    skipped: List[str] = []
    for k, v in state_dict.items():
        if k not in own_state:
            skipped.append(k)
            continue
        if own_state[k].shape != v.shape:
            skipped.append(k)
            continue
        compatible[k] = v
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped.extend(list(missing))
    skipped.extend(list(unexpected))
    return len(compatible), len(state_dict), skipped


def _to_u8_mask(mask01: torch.Tensor) -> np.ndarray:
    arr = mask01.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _sanitize_id(idx: int, path_str: str) -> str:
    stem = Path(path_str).stem
    stem = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in stem)
    return f"{idx:06d}_{stem}"


def _xywhn_to_xyxy_px(rows: torch.Tensor, h: int, w: int) -> Tuple[List[List[float]], List[int]]:
    boxes: List[List[float]] = []
    classes: List[int] = []
    if rows.numel() == 0:
        return boxes, classes
    for r in rows:
        cls = int(r[0].item())
        xc, yc, bw, bh = [float(x.item()) for x in r[1:5]]
        x1 = (xc - bw / 2.0) * w
        y1 = (yc - bh / 2.0) * h
        x2 = (xc + bw / 2.0) * w
        y2 = (yc + bh / 2.0) * h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls)
    return boxes, classes


def _topk_kept_for_vis(kept: Sequence[DetectionRecord], k: int) -> List[DetectionRecord]:
    if not kept:
        return []
    sorted_kept = sorted(kept, key=lambda d: float(d.score.detach().cpu().item()), reverse=True)
    return list(sorted_kept[: max(1, int(k))])


def build_split_cache(model: BasicCrackModel, cfg: Dict, split_name: str, split_cfg: Dict, device: torch.device) -> Dict[str, object]:
    image_root = str(split_cfg["image_root"])
    label_root = Path(str(split_cfg["label_root"]))
    mask_root = str(split_cfg["mask_root"])
    _verify_label_source(f"{split_name}.label_root", label_root)

    cache_root = Path(cfg["cache"]["cache_root"])
    split_dir = cache_root / split_name
    raw_dir = split_dir / "raw_masks"
    gt_dir = split_dir / "gt_masks"
    meta_dir = split_dir / "metadata"
    preview_dir = split_dir / "preview_panels"

    for d in (raw_dir, gt_dir, meta_dir, preview_dir):
        d.mkdir(parents=True, exist_ok=True)

    dl, ds = create_crack_dataloader_v020(
        image_root=image_root,
        label_root=str(label_root),
        mask_root=mask_root,
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(cfg["cache"]["batch_size"]),
        shuffle=False,
        augment=False,
        workers=int(cfg["cache"].get("workers", 0)),
    )

    conf = float(cfg["cache"]["detector_conf_thres"])
    iou = float(cfg["cache"]["detector_iou_thres"])
    max_det = int(cfg["cache"].get("max_det", 300))
    vis_topk = int(cfg["cache"].get("vis_topk", 5))
    max_preview = int(cfg["cache"].get("max_preview_panels", 8))

    manifest_path = split_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()

    saved = 0
    preview_saved = 0
    kept_sum = 0.0
    score_sum = 0.0
    thr_sum = 0.0

    model.eval()
    with torch.no_grad(), manifest_path.open("a", encoding="utf-8") as mf:
        for imgs, targets, masks, paths, _shapes in tqdm(dl, desc=f"build_cache_{split_name}"):
            if masks is None:
                raise RuntimeError(f"Split {split_name} has no GT masks from mask_root={mask_root}")

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            masks = masks.to(device)

            out = model.forward_train(imgs)
            decoded = model.pred_head.decode_for_eval(out["pred13_raw"], out["pred26_raw"])
            pred_for_nms = decoded[:, :, : 5 + int(cfg["model"]["num_classes"])]
            nms_out = model.pred_head.run_nms(
                decoded,
                conf_thres=conf,
                iou_thres=iou,
                max_det=max_det,
                nms_out=None,
            )
            raw_masks = model.pred_head.build_raw_mask_from_detections(imgs, nms_out)

            bsz, _, h, w = imgs.shape
            for i in range(bsz):
                image_path = str(paths[i])
                image_id = _sanitize_id(saved, image_path)

                raw_u8 = _to_u8_mask(raw_masks[i])
                gt_u8 = _to_u8_mask(masks[i])

                cv2.imwrite(str(raw_dir / f"{image_id}.png"), raw_u8)
                cv2.imwrite(str(gt_dir / f"{image_id}.png"), gt_u8)

                kept = nms_out[i]
                kept_n = int(len(kept))
                mean_score = float(torch.stack([d.score for d in kept]).mean().item()) if kept_n > 0 else 0.0
                mean_thr = float(torch.stack([d.threshold for d in kept]).mean().item()) if kept_n > 0 else 0.0

                kept_sum += float(kept_n)
                score_sum += mean_score
                thr_sum += mean_thr

                row = {
                    "image_id": image_id,
                    "split": split_name,
                    "image_path": image_path,
                    "raw_mask_path": str(raw_dir / f"{image_id}.png"),
                    "gt_mask_path": str(gt_dir / f"{image_id}.png"),
                    "detector_checkpoint": str(cfg["cache"]["frozen_basic_checkpoint"]),
                    "detector_conf_thres": conf,
                    "detector_iou_thres": iou,
                    "max_det": max_det,
                    "kept_detections": kept_n,
                    "mean_kept_score": mean_score,
                    "mean_kept_threshold": mean_thr,
                }
                mf.write(json.dumps(row) + "\n")
                (meta_dir / f"{image_id}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

                if preview_saved < max_preview:
                    gt_rows = targets[targets[:, 0] == i, 1:6]
                    gt_boxes, gt_classes = _xywhn_to_xyxy_px(gt_rows, h=int(h), w=int(w))
                    save_debug_panel(
                        image_rgb=imgs[i].detach().cpu(),
                        kept_detections=_topk_kept_for_vis(kept, vis_topk),
                        raw_mask=raw_masks[i].detach().cpu(),
                        save_path=str(preview_dir / f"{image_id}_panel.png"),
                        gt_mask=masks[i].detach().cpu(),
                        gt_boxes_xyxy=gt_boxes,
                        gt_classes=gt_classes,
                        class_names=cfg["model"].get("class_names", ["crack"]),
                    )
                    preview_saved += 1

                saved += 1

    return {
        "split": split_name,
        "samples": int(saved),
        "mean_kept_detections": float(kept_sum / max(saved, 1)),
        "mean_kept_score": float(score_sum / max(saved, 1)),
        "mean_kept_threshold": float(thr_sum / max(saved, 1)),
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline raw-mask DAE cache for v0.5.0")
    parser.add_argument("--config", type=str, default="configs/crack_v050_dae_rawmask.yaml")
    parser.add_argument("--splits", type=str, default="train,val,test")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _set_seed(int(cfg["cache"].get("seed", 50)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicCrackModel(
        cfg_path=str(cfg["model"]["cfg_path"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)

    ckpt_path = str(cfg["cache"]["frozen_basic_checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device)
    loaded_n, total_n, skipped = _load_compatible_state_dict(model, ckpt["model"])
    print({
        "frozen_basic_checkpoint": ckpt_path,
        "loaded_keys": int(loaded_n),
        "total_ckpt_keys": int(total_n),
        "skipped_keys_count": int(len(skipped)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "checkpoint_best_map50": float(ckpt.get("best_map50", -1.0)),
        "checkpoint_best_miou_raw": float(ckpt.get("best_miou_raw", -1.0)),
    })

    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    results = []
    for split in split_names:
        if split not in cfg["splits"]:
            raise KeyError(f"Split not found in config: {split}")
        res = build_split_cache(model=model, cfg=cfg, split_name=split, split_cfg=cfg["splits"][split], device=device)
        print(res)
        results.append(res)

    summary = {
        "version": "v0.5.0-local-dae-rawmask",
        "config": args.config,
        "frozen_basic_checkpoint": ckpt_path,
        "splits": results,
    }
    cache_root = Path(cfg["cache"]["cache_root"])
    summary_path = cache_root / "cache_build_summary_v050.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved_cache_summary={summary_path}")


if __name__ == "__main__":
    main()
