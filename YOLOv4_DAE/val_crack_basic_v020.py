from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

from datasets.crack_dataset_v020 import create_crack_dataloader_v020
from models.crack_pipeline_v010 import BasicCrackModel, DetectionRecord
from utils.crack_debug_v010 import save_debug_outputs, save_debug_panel
from utils.general import box_iou, non_max_suppression, xywh2xyxy
from utils.metrics import ap_per_class


def _process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.tensor(matches, device=iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def _safe_iou_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> Tuple[float, float]:
    pred = (pred_mask > 0.5).to(torch.float32)
    gt = (gt_mask > 0.5).to(torch.float32)

    inter = float((pred * gt).sum().item())
    union = float(((pred + gt) > 0).sum().item())
    iou = inter / (union + 1e-9)

    tp = inter
    fp = float((pred * (1.0 - gt)).sum().item())
    fn = float(((1.0 - pred) * gt).sum().item())
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-9)
    return iou, f1


def evaluate_basic_model(
    model: BasicCrackModel,
    dataloader,
    device: torch.device,
    num_classes: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int = 300,
    class_names: Optional[Sequence[str]] = None,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    max_vis: int = 0,
) -> Dict[str, float]:
    model.eval()
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    miou_vals: List[float] = []
    f1_vals: List[float] = []

    vis_saved = 0
    vis_root = Path(vis_dir) if vis_dir else None
    if vis_root is not None and save_vis:
        vis_root.mkdir(parents=True, exist_ok=True)

    for imgs, targets, masks, paths, _shapes in tqdm(dataloader, desc="val_v020", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)

        with torch.no_grad():
            train_out = model.forward_train(imgs)
            pred13_raw = train_out["pred13_raw"]
            pred26_raw = train_out["pred26_raw"]
            decoded = model.pred_head.decode_for_eval(pred13_raw, pred26_raw)

            pred_for_nms = decoded[:, :, : 5 + num_classes]
            nms_out = non_max_suppression(
                pred_for_nms,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=int(max_det),
            )

            kept_detections = model.pred_head.run_nms(
                decoded,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=int(max_det),
                nms_out=nms_out,
            )
            raw_masks = model.pred_head.build_raw_mask_from_detections(imgs, kept_detections)

        bsz, _, h, w = imgs.shape
        whwh = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)

        for si in range(bsz):
            pred = nms_out[si]
            labels = targets[targets[:, 0] == si, 1:6]
            nl = labels.shape[0]
            tcls = labels[:, 0].tolist() if nl else []

            if pred is None or pred.shape[0] == 0:
                if nl:
                    stats.append((
                        torch.zeros(0, niou, dtype=torch.bool).cpu().numpy(),
                        np.array([], dtype=np.float32),
                        np.array([], dtype=np.float32),
                        np.asarray(tcls, dtype=np.float32),
                    ))
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5]) * whwh
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = _process_batch(pred, labelsn, iouv)

                stats.append((
                    correct.cpu().numpy(),
                    pred[:, 4].detach().cpu().numpy(),
                    pred[:, 5].detach().cpu().numpy(),
                    np.asarray(tcls, dtype=np.float32),
                ))

            if masks is not None:
                gt_mask = masks[si].to(device)
                iou, f1 = _safe_iou_f1(raw_masks[si, 0], gt_mask[0])
                miou_vals.append(iou)
                f1_vals.append(f1)

            if save_vis and vis_root is not None and vis_saved < max_vis:
                stem = Path(paths[si]).stem
                gt_mask = masks[si] if masks is not None else None

                gt_rows = targets[targets[:, 0] == si, 1:6]
                gt_boxes_xyxy = []
                gt_classes = []
                if gt_rows.shape[0] > 0:
                    gt_xyxy = (xywh2xyxy(gt_rows[:, 1:5]) * whwh).detach().cpu().numpy()
                    gt_boxes_xyxy = gt_xyxy.tolist()
                    gt_classes = gt_rows[:, 0].detach().cpu().numpy().astype(np.int32).tolist()

                save_debug_outputs(
                    image_rgb=imgs[si].detach().cpu(),
                    kept_detections=kept_detections[si],
                    save_dir=str(vis_root),
                    stem=stem,
                    raw_mask=raw_masks[si].detach().cpu(),
                    clean_mask=None,
                    class_names=class_names,
                )
                save_debug_panel(
                    image_rgb=imgs[si].detach().cpu(),
                    kept_detections=kept_detections[si],
                    raw_mask=raw_masks[si].detach().cpu(),
                    save_path=str(vis_root / f"{stem}_panel.png"),
                    gt_mask=gt_mask.detach().cpu() if gt_mask is not None else None,
                    gt_boxes_xyxy=gt_boxes_xyxy,
                    gt_classes=gt_classes,
                    class_names=class_names,
                )
                vis_saved += 1

    if stats:
        stats_np = [np.concatenate(x, 0) if len(x) else np.array([]) for x in zip(*stats)]
    else:
        stats_np = [np.array([]), np.array([]), np.array([]), np.array([])]

    if len(stats_np[0]) and stats_np[0].any():
        p, r, ap, _f1, _ap_class = ap_per_class(*stats_np)
        ap50 = ap[:, 0]
        map50 = float(ap50.mean()) if len(ap50) else 0.0
        map5095 = float(ap.mean()) if len(ap) else 0.0
        precision = float(p[:, 0].mean()) if len(p) else 0.0
        recall = float(r[:, 0].mean()) if len(r) else 0.0
    else:
        precision = 0.0
        recall = 0.0
        map50 = 0.0
        map5095 = 0.0

    miou = float(np.mean(miou_vals)) if miou_vals else 0.0
    f1_mask = float(np.mean(f1_vals)) if f1_vals else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map5095": map5095,
        "miou_raw": miou,
        "f1_raw": f1_mask,
    }


def _load_checkpoint(model: BasicCrackModel, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    current = model.state_dict()
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    loadable = {k: v for k, v in cleaned.items() if k in current and current[k].shape == v.shape}
    current.update(loadable)
    model.load_state_dict(current, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.2.0 basic-model validation")
    parser.add_argument("--config", type=str, default="configs/crack_v020_basic.yaml")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--save-vis", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicCrackModel(
        cfg_path=cfg["model"]["cfg_path"],
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)
    _load_checkpoint(model, args.weights, device)

    val_bs = args.batch_size or int(cfg["train"]["batch_size"])
    val_workers = args.workers if args.workers is not None else int(cfg["train"]["workers"])

    val_loader, _ = create_crack_dataloader_v020(
        image_root=cfg["data"]["val_image_root"],
        label_root=cfg["data"].get("val_label_root"),
        mask_root=cfg["data"].get("val_mask_root"),
        image_size=int(cfg["model"]["image_size"]),
        batch_size=val_bs,
        shuffle=False,
        augment=False,
        workers=val_workers,
    )

    metrics = evaluate_basic_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=int(cfg["model"]["num_classes"]),
        conf_thres=float(cfg["val"]["conf_thres"]),
        iou_thres=float(cfg["val"]["iou_thres"]),
        max_det=int(cfg["val"].get("max_det", 300)),
        class_names=cfg["model"].get("class_names"),
        save_vis=args.save_vis or bool(cfg["val"].get("save_visualization", False)),
        vis_dir=str(cfg["val"].get("vis_dir", "runs/crack_v020_basic/val_vis")),
        max_vis=int(cfg["val"].get("max_visualizations", 0)),
    )

    print("v0.2.0 validation metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
