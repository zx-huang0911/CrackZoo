from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dae_dataset_v050 import create_crack_dae_dataloader_v050
from models.crack_pipeline_v010 import CrackDAE
from utils.dae_metrics_v050 import dice_loss_from_probs, mask_iou_f1


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_mask_triplet_panel(raw: torch.Tensor, clean: torch.Tensor, gt: torch.Tensor, save_path: Path, title: str) -> None:
    def to_u8(x: torch.Tensor) -> np.ndarray:
        arr = x.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = arr[0]
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255.0).astype(np.uint8)

    raw_u8 = to_u8(raw)
    clean_u8 = to_u8(clean)
    gt_u8 = to_u8(gt)

    raw_rgb = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2RGB)
    clean_rgb = cv2.cvtColor(clean_u8, cv2.COLOR_GRAY2RGB)
    gt_rgb = cv2.cvtColor(gt_u8, cv2.COLOR_GRAY2RGB)

    overlay = np.zeros_like(gt_rgb)
    overlay[gt_u8 > 127, 1] = 255
    overlay[clean_u8 > 127, 2] = 255

    def add_cap(img: np.ndarray, txt: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 26), (0, 0, 0), -1)
        cv2.putText(out, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    p1 = add_cap(raw_rgb, "1) Raw mask")
    p2 = add_cap(clean_rgb, "2) Pred clean mask")
    p3 = add_cap(gt_rgb, "3) GT mask")
    p4 = add_cap(overlay, title)

    panel = np.concatenate([p1, p2, p3, p4], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _validate(
    model: CrackDAE,
    val_loader,
    device: torch.device,
    bce_w: float,
    dice_w: float,
    vis_dir: Path | None,
    max_vis: int,
) -> Dict[str, float]:
    model.eval()

    loss_sum = 0.0
    n_batches = 0

    raw_iou_sum = 0.0
    raw_f1_sum = 0.0
    clean_iou_sum = 0.0
    clean_f1_sum = 0.0
    n_samples = 0

    vis_saved = 0
    with torch.no_grad():
        for raw, gt, ids, _meta in tqdm(val_loader, desc="dae_val", leave=False):
            raw = raw.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            clean = model(raw)
            bce = F.binary_cross_entropy(clean, gt)
            dloss = dice_loss_from_probs(clean, gt)
            loss = float(bce_w * bce + dice_w * dloss)

            bs = raw.shape[0]
            n_batches += 1
            loss_sum += loss
            n_samples += bs

            raw_m = mask_iou_f1(raw, gt)
            clean_m = mask_iou_f1(clean, gt)
            raw_iou_sum += raw_m["iou"] * bs
            raw_f1_sum += raw_m["f1"] * bs
            clean_iou_sum += clean_m["iou"] * bs
            clean_f1_sum += clean_m["f1"] * bs

            if vis_dir is not None and vis_saved < max_vis:
                for i in range(bs):
                    if vis_saved >= max_vis:
                        break
                    sid = str(ids[i])
                    _save_mask_triplet_panel(
                        raw=raw[i].detach().cpu(),
                        clean=clean[i].detach().cpu(),
                        gt=gt[i].detach().cpu(),
                        save_path=vis_dir / f"{sid}_panel.png",
                        title="4) Overlay (clean:red, gt:green)",
                    )
                    vis_saved += 1

    return {
        "val_loss": float(loss_sum / max(n_batches, 1)),
        "raw_iou": float(raw_iou_sum / max(n_samples, 1)),
        "raw_f1": float(raw_f1_sum / max(n_samples, 1)),
        "clean_iou": float(clean_iou_sum / max(n_samples, 1)),
        "clean_f1": float(clean_f1_sum / max(n_samples, 1)),
        "improved_iou": float((clean_iou_sum - raw_iou_sum) / max(n_samples, 1)),
        "improved_f1": float((clean_f1_sum - raw_f1_sum) / max(n_samples, 1)),
        "saved_visualizations": int(vis_saved),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone DAE on offline raw-mask cache (v0.5.0)")
    parser.add_argument("--config", type=str, default="configs/crack_v050_dae_rawmask.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["dae_train"]

    _set_seed(int(tcfg.get("seed", 50)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_root = str(cfg["cache"]["cache_root"])
    train_loader, train_ds = create_crack_dae_dataloader_v050(
        cache_root=cache_root,
        split="train",
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        workers=int(tcfg.get("workers", 0)),
        include_meta=False,
    )
    val_loader, val_ds = create_crack_dae_dataloader_v050(
        cache_root=cache_root,
        split="val",
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        workers=int(tcfg.get("workers", 0)),
        include_meta=False,
    )

    model = CrackDAE().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(tcfg["epochs"]), 1))

    bce_w = float(tcfg.get("bce_weight", 1.0))
    dice_w = float(tcfg.get("dice_weight", 1.0))

    save_dir = Path(str(tcfg["save_dir"]))
    weights_dir = save_dir / "weights"
    vis_dir = save_dir / "vis"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    wb_cfg = cfg.get("wandb", {})
    use_wandb = bool(wb_cfg.get("enabled", False))
    wb_run = None
    if use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is enabled but not installed")
        wb_run = wandb.init(
            project=str(wb_cfg.get("project", "crack-paper-repro")),
            name=str(wb_cfg.get("run_name", "dae_v050_rawmask")),
            config=cfg,
            dir=str(save_dir),
        )

    epochs = int(tcfg["epochs"])
    val_interval = max(int(tcfg.get("val_interval", 2)), 1)
    vis_interval = max(int(tcfg.get("vis_interval", 5)), 1)
    ckpt_interval = max(int(tcfg.get("checkpoint_interval", 5)), 1)
    max_vis = int(tcfg.get("max_vis", 6))

    history: List[Dict[str, float | int]] = []
    best_clean_iou = -1.0
    best_epoch = -1

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_total = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"dae_v050 epoch {epoch}/{epochs}", leave=False)
            for raw, gt, _ids, _meta in pbar:
                raw = raw.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)

                clean = model(raw)
                bce = F.binary_cross_entropy(clean, gt)
                dloss = dice_loss_from_probs(clean, gt)
                loss = bce_w * bce + dice_w * dloss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                n_batches += 1
                loss_total += float(loss.item())
                pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            scheduler.step()
            row: Dict[str, float | int] = {
                "epoch": int(epoch),
                "train_loss": float(loss_total / max(n_batches, 1)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }

            eval_row: Dict[str, float] = {}
            if epoch % val_interval == 0 or epoch == epochs:
                do_vis = (epoch % vis_interval == 0) or (epoch == epochs)
                eval_row = _validate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    bce_w=bce_w,
                    dice_w=dice_w,
                    vis_dir=(vis_dir / f"epoch_{epoch:03d}") if do_vis else None,
                    max_vis=max_vis,
                )
                row.update(eval_row)

                if eval_row.get("clean_iou", -1.0) >= best_clean_iou:
                    best_clean_iou = float(eval_row["clean_iou"])
                    best_epoch = int(epoch)
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "best_clean_iou": float(best_clean_iou),
                            "config": args.config,
                        },
                        str(weights_dir / "best_clean_iou.pt"),
                    )

            torch.save(
                {
                    "epoch": int(epoch),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_clean_iou": float(best_clean_iou),
                    "config": args.config,
                },
                str(weights_dir / "last.pt"),
            )
            if epoch % ckpt_interval == 0 or epoch == epochs:
                torch.save(
                    {
                        "epoch": int(epoch),
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_clean_iou": float(best_clean_iou),
                        "config": args.config,
                    },
                    str(weights_dir / f"epoch_{epoch:03d}.pt"),
                )

            history.append(row)
            with (save_dir / "history_v050.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(row)
            if wb_run is not None:
                wb_payload = {k: v for k, v in row.items() if isinstance(v, (int, float))}
                wb_run.log(wb_payload, step=int(epoch))

        summary = {
            "version": "v0.5.0-local-dae-rawmask",
            "config": args.config,
            "save_dir": str(save_dir),
            "cache_root": cache_root,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "epochs": int(epochs),
            "loss": {
                "bce_weight": bce_w,
                "dice_weight": dice_w,
            },
            "best_clean_iou": float(best_clean_iou),
            "best_epoch": int(best_epoch),
            "weights_best": str(weights_dir / "best_clean_iou.pt"),
            "weights_last": str(weights_dir / "last.pt"),
            "history": history,
        }
        summary_path = save_dir / "summary_v050_dae_train.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_summary={summary_path}")
    finally:
        if wb_run is not None:
            wb_run.finish()


if __name__ == "__main__":
    main()
