from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from datasets.crack_dataset_v020 import create_crack_dataloader_v020
from models.crack_pipeline_v010 import BasicCrackModel, CrackDAE
from utils.general import non_max_suppression


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _mask_iou_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Dict[str, float]:
    pred = (pred_mask > 0.5).to(torch.float32)
    gt = (gt_mask > 0.5).to(torch.float32)

    inter = float((pred * gt).sum().item())
    union = float(((pred + gt) > 0).sum().item())
    iou = _safe_div(inter, union)

    tp = inter
    fp = float((pred * (1.0 - gt)).sum().item())
    fn = float(((1.0 - pred) * gt).sum().item())
    f1 = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    return {"iou": iou, "f1": f1}


def _overlay_mask_on_image(img_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = img_rgb.copy()
    m = mask > 0
    if np.any(m):
        overlay = np.zeros_like(out)
        overlay[:, :] = np.array(color, dtype=np.uint8)
        out[m] = ((1.0 - alpha) * out[m] + alpha * overlay[m]).astype(np.uint8)
    return out


def _tile_title(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = img_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _save_panel(
    img_rgb: np.ndarray,
    raw_mask: np.ndarray,
    clean_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    out_path: Path,
) -> None:
    raw_u8 = (raw_mask > 0).astype(np.uint8) * 255
    clean_u8 = (clean_mask > 0).astype(np.uint8) * 255

    raw_bgr = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB)
    clean_bgr = cv2.cvtColor(clean_u8, cv2.COLOR_GRAY2RGB)

    if gt_mask is None:
        gt_u8 = np.zeros_like(raw_u8)
    else:
        gt_u8 = (gt_mask > 0).astype(np.uint8) * 255
    gt_bgr = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB)

    raw_overlay = _overlay_mask_on_image(img_rgb, raw_u8, color=(255, 0, 0), alpha=0.35)
    clean_overlay = _overlay_mask_on_image(img_rgb, clean_u8, color=(0, 255, 0), alpha=0.35)
    gt_overlay = _overlay_mask_on_image(img_rgb, gt_u8, color=(255, 255, 0), alpha=0.35)

    t1 = _tile_title(img_rgb, "1) RGB")
    t2 = _tile_title(raw_bgr, "2) Raw mask")
    t3 = _tile_title(clean_bgr, "3) DAE clean mask")
    t4 = _tile_title(gt_bgr, "4) GT mask")
    t5 = _tile_title(raw_overlay, "5) RGB + raw")
    t6 = _tile_title(clean_overlay, "6) RGB + clean")
    t7 = _tile_title(gt_overlay, "7) RGB + GT")

    err = np.zeros_like(img_rgb)
    if gt_mask is not None:
        pred = (clean_u8 > 0)
        gt = (gt_u8 > 0)
        err[np.logical_and(pred, ~gt)] = (255, 0, 0)
        err[np.logical_and(~pred, gt)] = (255, 255, 0)
        err[np.logical_and(pred, gt)] = (0, 255, 0)
    t8 = _tile_title(err, "8) Clean vs GT error map")

    top = np.hstack([t1, t2, t3, t4])
    bottom = np.hstack([t5, t6, t7, t8])
    panel = np.vstack([top, bottom])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _load_model_state(path: Path) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(str(path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"Invalid checkpoint state format: {path}")
    return state


def _infer_num_classes_from_basic_state(state: Dict[str, torch.Tensor], default_nc: int) -> int:
    w = state.get("pred_head.pred_conv13.weight", None)
    if w is None or w.ndim < 1:
        return int(default_nc)
    out_ch = int(w.shape[0])
    if out_ch % 3 != 0:
        return int(default_nc)
    no = out_ch // 3
    nc = no - 6
    if nc <= 0:
        return int(default_nc)
    return int(nc)


def _load_state_shape_matched(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    model_sd = model.state_dict()
    loadable = {}
    skipped = 0
    for k, v in state.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            loadable[k] = v
        else:
            skipped += 1
    model_sd.update(loadable)
    model.load_state_dict(model_sd, strict=False)
    return {"loaded": len(loadable), "skipped": skipped}


def _detect_checkpoint_type(state: Dict[str, torch.Tensor]) -> str:
    keys = list(state.keys())
    if any(k.startswith("backbone_afpn") for k in keys):
        return "basic"
    if any(k.startswith("conv1.block") for k in keys):
        return "dae"
    return "unknown"


def _resolve_checkpoints(basic_ckpt: Path, dae_ckpt: Path) -> Tuple[Path, Path]:
    basic_state = _load_model_state(basic_ckpt)
    dae_state = _load_model_state(dae_ckpt)

    t1 = _detect_checkpoint_type(basic_state)
    t2 = _detect_checkpoint_type(dae_state)

    if t1 == "basic" and t2 == "dae":
        return basic_ckpt, dae_ckpt
    if t1 == "dae" and t2 == "basic":
        return dae_ckpt, basic_ckpt

    raise RuntimeError(
        "Unable to resolve checkpoint types. "
        f"basic_ckpt={basic_ckpt} type={t1}, dae_ckpt={dae_ckpt} type={t2}"
    )


def run_chain_val(args: argparse.Namespace) -> Dict[str, object]:
    cfg = yaml.safe_load(Path(args.config_basic).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basic_ckpt, dae_ckpt = _resolve_checkpoints(Path(args.basic_ckpt), Path(args.dae_ckpt))
    basic_state = _load_model_state(basic_ckpt)
    dae_state = _load_model_state(dae_ckpt)

    image_root = args.image_root or cfg["data"]["val_image_root"]
    label_root = args.label_root or cfg["data"].get("val_label_root")
    mask_root = args.mask_root or cfg["data"].get("val_mask_root")

    dl, ds = create_crack_dataloader_v020(
        image_root=image_root,
        label_root=label_root,
        mask_root=mask_root,
        image_size=int(args.image_size or cfg["model"]["image_size"]),
        batch_size=int(args.batch_size),
        shuffle=False,
        augment=False,
        workers=int(args.workers),
    )

    inferred_nc = _infer_num_classes_from_basic_state(basic_state, int(cfg["model"]["num_classes"]))
    basic_model = BasicCrackModel(
        cfg_path=str(args.model_cfg or cfg["model"]["cfg_path"]),
        num_classes=int(inferred_nc),
    ).to(device)
    dae_model = CrackDAE().to(device)

    basic_load_info = _load_state_shape_matched(basic_model, basic_state)
    dae_load_info = _load_state_shape_matched(dae_model, dae_state)

    basic_model.eval()
    dae_model.eval()

    out_dir = Path(args.save_dir)
    panel_dir = out_dir / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        panel_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    raw_iou_sum = 0.0
    raw_f1_sum = 0.0
    clean_iou_sum = 0.0
    clean_f1_sum = 0.0
    vis_saved = 0

    det_count_sum = 0
    empty_det_images = 0

    t_basic = 0.0
    t_dae = 0.0

    limit = int(args.max_samples)

    with torch.no_grad():
        for imgs, _targets, masks, paths, _shapes in tqdm(dl, desc="chain_val", dynamic_ncols=True):
            if n_images >= limit:
                break

            imgs = imgs.to(device, non_blocking=True)
            if masks is not None:
                masks = masks.to(device, non_blocking=True)

            t0 = time.perf_counter()
            train_out = basic_model.forward_train(imgs)
            pred13_raw = train_out["pred13_raw"]
            pred26_raw = train_out["pred26_raw"]
            decoded = basic_model.pred_head.decode_for_eval(pred13_raw, pred26_raw)
            pred_for_nms = decoded[:, :, : 5 + int(inferred_nc)]
            nms_out = non_max_suppression(
                pred_for_nms,
                conf_thres=float(args.conf_thres),
                iou_thres=float(args.iou_thres),
                max_det=int(args.max_det),
            )
            kept = basic_model.pred_head.run_nms(
                decoded,
                conf_thres=float(args.conf_thres),
                iou_thres=float(args.iou_thres),
                max_det=int(args.max_det),
                nms_out=nms_out,
            )
            raw_masks = basic_model.pred_head.build_raw_mask_from_detections(imgs, kept)
            t1 = time.perf_counter()

            clean_masks = dae_model(raw_masks)
            t2 = time.perf_counter()

            t_basic += (t1 - t0)
            t_dae += (t2 - t1)

            bsz = imgs.shape[0]
            for i in range(bsz):
                if n_images >= limit:
                    break

                n_images += 1
                det_n = len(kept[i])
                det_count_sum += det_n
                if det_n == 0:
                    empty_det_images += 1

                if masks is not None:
                    m_raw = _mask_iou_f1(raw_masks[i, 0], masks[i, 0])
                    m_clean = _mask_iou_f1(clean_masks[i, 0], masks[i, 0])

                    raw_iou_sum += m_raw["iou"]
                    raw_f1_sum += m_raw["f1"]
                    clean_iou_sum += m_clean["iou"]
                    clean_f1_sum += m_clean["f1"]

                if args.save_vis and vis_saved < int(args.max_vis):
                    img_u8 = (imgs[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    raw_u8 = (raw_masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    clean_u8 = (clean_masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
                    gt_u8 = None
                    if masks is not None:
                        gt_u8 = (masks[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)

                    stem = Path(paths[i]).stem
                    _save_panel(
                        img_rgb=img_u8,
                        raw_mask=raw_u8,
                        clean_mask=clean_u8,
                        gt_mask=gt_u8,
                        out_path=panel_dir / f"{n_images:04d}_{stem}.jpg",
                    )
                    vis_saved += 1

    summary = {
        "version": "v0.5.1-chain-infer",
        "basic_checkpoint": str(basic_ckpt),
        "dae_checkpoint": str(dae_ckpt),
        "inferred_num_classes": int(inferred_nc),
        "basic_load_info": basic_load_info,
        "dae_load_info": dae_load_info,
        "image_root": str(image_root),
        "label_root": str(label_root) if label_root else "",
        "mask_root": str(mask_root) if mask_root else "",
        "num_images": int(n_images),
        "max_samples": int(limit),
        "mean_detections_per_image": _safe_div(det_count_sum, n_images),
        "empty_detection_images": int(empty_det_images),
        "empty_detection_fraction": _safe_div(empty_det_images, n_images),
        "raw_iou": _safe_div(raw_iou_sum, n_images),
        "raw_f1": _safe_div(raw_f1_sum, n_images),
        "clean_iou": _safe_div(clean_iou_sum, n_images),
        "clean_f1": _safe_div(clean_f1_sum, n_images),
        "improved_iou": _safe_div(clean_iou_sum - raw_iou_sum, n_images),
        "improved_f1": _safe_div(clean_f1_sum - raw_f1_sum, n_images),
        "avg_basic_infer_ms_per_image": _safe_div(t_basic * 1000.0, n_images),
        "avg_dae_infer_ms_per_image": _safe_div(t_dae * 1000.0, n_images),
        "saved_visualizations": int(vis_saved),
        "save_dir": str(out_dir),
        "panel_dir": str(panel_dir),
    }

    if summary["mean_detections_per_image"] < 0.2:
        summary["warning"] = (
            "Detector produced very few boxes. Consider lowering --conf-thres "
            "(e.g. 0.02) and reducing --iou-thres (e.g. 0.30)."
        )

    out_json = out_dir / "summary_chain_val.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved_summary={out_json}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chain inference: basic model -> DAE and val on subset")
    parser.add_argument("--config-basic", type=str, default="configs/crack_v020_basic.yaml")
    parser.add_argument("--model-cfg", type=str, default="")

    parser.add_argument("--basic-ckpt", type=str, default="weights/epoch_200.pt")
    parser.add_argument("--dae-ckpt", type=str, default="weights/best_clean_iou.pt")

    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--label-root", type=str, default="")
    parser.add_argument("--mask-root", type=str, default="")

    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)

    parser.add_argument("--conf-thres", type=float, default=0.005)
    parser.add_argument("--iou-thres", type=float, default=0.28)
    parser.add_argument("--max-det", type=int, default=1000)

    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--save-dir", type=str, default="runs/chain_infer_v051")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--max-vis", type=int, default=24)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_chain_val(args)


if __name__ == "__main__":
    main()
