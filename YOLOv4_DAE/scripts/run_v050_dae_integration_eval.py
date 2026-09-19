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
from models.crack_pipeline_v010 import BasicCrackModel, CrackDAE, DetectionRecord
from utils.dae_metrics_v050 import mask_iou_f1


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


def _to_u8_rgb(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().permute(1, 2, 0).numpy()
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _add_cap(img: np.ndarray, txt: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 26), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _topk_kept_for_vis(kept: Sequence[DetectionRecord], k: int) -> List[DetectionRecord]:
    if not kept:
        return []
    sorted_kept = sorted(kept, key=lambda d: float(d.score.detach().cpu().item()), reverse=True)
    return list(sorted_kept[: max(1, int(k))])


def _draw_kept_boxes(img_rgb: np.ndarray, kept: Sequence[DetectionRecord]) -> np.ndarray:
    out = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for d in kept:
        box = d.box.detach().cpu().numpy()
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        score = float(d.score.detach().cpu().item())
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(out, f"s:{score:.3f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 220, 40), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def _save_panel(rgb: torch.Tensor, raw: torch.Tensor, clean: torch.Tensor, gt: torch.Tensor, kept: Sequence[DetectionRecord], save_path: Path) -> None:
    rgb_u8 = _to_u8_rgb(rgb)
    raw_u8 = _to_u8_mask(raw)
    clean_u8 = _to_u8_mask(clean)
    gt_u8 = _to_u8_mask(gt)

    p1 = _add_cap(rgb_u8, "1) RGB")
    p2 = _add_cap(cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB), "2) Raw synthesized")
    p3 = _add_cap(cv2.cvtColor(clean_u8, cv2.COLOR_GRAY2RGB), "3) Clean after DAE")
    p4 = _add_cap(cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB), "4) GT mask")

    overlay = rgb_u8.copy()
    overlay[gt_u8 > 127, 1] = 255
    overlay[clean_u8 > 127, 2] = 255
    overlay = cv2.addWeighted(rgb_u8, 0.55, overlay, 0.45, 0.0)
    overlay = _draw_kept_boxes(overlay, kept)
    p5 = _add_cap(overlay, "5) Overlay + kept boxes")

    panel = np.concatenate([p1, p2, p3, p4, p5], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.5.0 integration eval: fixed basic -> raw_mask -> DAE -> clean")
    parser.add_argument("--config", type=str, default="configs/crack_v050_dae_rawmask.yaml")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--dae-checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _set_seed(int(cfg["cache"].get("seed", 50)))

    split_name = args.split.strip() or str(cfg["integration_eval"].get("split", "test"))
    if split_name not in cfg["splits"]:
        raise KeyError(f"Unknown split: {split_name}")
    split_cfg = cfg["splits"][split_name]

    label_root = Path(str(split_cfg["label_root"]))
    _verify_label_source(f"{split_name}.label_root", label_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basic = BasicCrackModel(
        cfg_path=str(cfg["model"]["cfg_path"]),
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    basic_ckpt_path = str(cfg["cache"]["frozen_basic_checkpoint"])
    basic_ckpt = torch.load(basic_ckpt_path, map_location=device)
    loaded_n, total_n, skipped = _load_compatible_state_dict(basic, basic_ckpt["model"])
    print({
        "frozen_basic_checkpoint": basic_ckpt_path,
        "loaded_keys": int(loaded_n),
        "total_ckpt_keys": int(total_n),
        "skipped_keys_count": int(len(skipped)),
    })

    dae_ckpt_path = args.dae_checkpoint.strip() or str(Path(cfg["dae_train"]["save_dir"]) / "weights" / "best_clean_iou.pt")
    dae_ckpt = torch.load(dae_ckpt_path, map_location=device)
    dae = CrackDAE().to(device)
    dae.load_state_dict(dae_ckpt["model"], strict=True)

    dl, ds = create_crack_dataloader_v020(
        image_root=str(split_cfg["image_root"]),
        label_root=str(split_cfg["label_root"]),
        mask_root=str(split_cfg["mask_root"]),
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(cfg["cache"]["batch_size"]),
        shuffle=False,
        augment=False,
        workers=int(cfg["cache"].get("workers", 0)),
    )

    conf = float(cfg["cache"]["detector_conf_thres"])
    iou = float(cfg["cache"]["detector_iou_thres"])
    max_det = int(cfg["cache"].get("max_det", 300))
    max_vis = int(cfg["integration_eval"].get("max_visualizations", 8))
    vis_topk = int(cfg["integration_eval"].get("vis_topk", 5))

    save_dir = Path(str(cfg["integration_eval"]["save_dir"]))
    panel_dir = save_dir / split_name / "panels"
    save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 0
    raw_iou_sum = 0.0
    raw_f1_sum = 0.0
    clean_iou_sum = 0.0
    clean_f1_sum = 0.0
    vis_saved = 0

    basic.eval()
    dae.eval()
    with torch.no_grad():
        for imgs, _targets, masks, paths, _shapes in tqdm(dl, desc=f"integration_{split_name}"):
            if masks is None:
                raise RuntimeError("Integration eval requires GT masks")

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            out = basic.forward_train(imgs)
            decoded = basic.pred_head.decode_for_eval(out["pred13_raw"], out["pred26_raw"])
            kept = basic.pred_head.run_nms(decoded, conf_thres=conf, iou_thres=iou, max_det=max_det, nms_out=None)
            raw = basic.pred_head.build_raw_mask_from_detections(imgs, kept)
            clean = dae(raw)

            bs = imgs.shape[0]
            raw_m = mask_iou_f1(raw, masks)
            clean_m = mask_iou_f1(clean, masks)
            raw_iou_sum += raw_m["iou"] * bs
            raw_f1_sum += raw_m["f1"] * bs
            clean_iou_sum += clean_m["iou"] * bs
            clean_f1_sum += clean_m["f1"] * bs
            n_samples += bs

            for i in range(bs):
                if vis_saved >= max_vis:
                    break
                sid = f"{vis_saved:03d}_{Path(paths[i]).stem}"
                _save_panel(
                    rgb=imgs[i].detach().cpu(),
                    raw=raw[i].detach().cpu(),
                    clean=clean[i].detach().cpu(),
                    gt=masks[i].detach().cpu(),
                    kept=_topk_kept_for_vis(kept[i], vis_topk),
                    save_path=panel_dir / f"{sid}_panel.png",
                )
                vis_saved += 1

    out = {
        "version": "v0.5.0-local-dae-rawmask",
        "config": args.config,
        "split": split_name,
        "samples": len(ds),
        "frozen_basic_checkpoint": basic_ckpt_path,
        "dae_checkpoint": dae_ckpt_path,
        "detector_conf_thres": conf,
        "detector_iou_thres": iou,
        "raw_vs_gt": {
            "iou": float(raw_iou_sum / max(n_samples, 1)),
            "f1": float(raw_f1_sum / max(n_samples, 1)),
        },
        "clean_vs_gt": {
            "iou": float(clean_iou_sum / max(n_samples, 1)),
            "f1": float(clean_f1_sum / max(n_samples, 1)),
        },
        "delta": {
            "iou": float((clean_iou_sum - raw_iou_sum) / max(n_samples, 1)),
            "f1": float((clean_f1_sum - raw_f1_sum) / max(n_samples, 1)),
        },
        "saved_visualizations": int(vis_saved),
    }

    summary_path = save_dir / f"summary_{split_name}.json"
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
