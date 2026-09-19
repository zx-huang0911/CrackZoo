from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from datasets.crack_dataset_v020 import create_crack_dataloader_v020
from models.crack_pipeline_v010 import BasicCrackModel
from utils.loss_crack_basic_v020 import CrackBasicLossV020
from val_crack_basic_v020 import evaluate_basic_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _anchor_vecs_from_model(model: BasicCrackModel):
    return [
        model.pred_head.decode13.anchor_vec.detach(),
        model.pred_head.decode26.anchor_vec.detach(),
    ]


def _save_checkpoint(
    ckpt_path: Path,
    model: BasicCrackModel,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
    }
    torch.save(ckpt, str(ckpt_path))


def _load_checkpoint(
    ckpt_path: str,
    model: BasicCrackModel,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> Dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "start_epoch": int(ckpt.get("epoch", -1)) + 1,
        "best_metric": float(ckpt.get("best_metric", 0.0)),
    }


def _load_pretrained_model(ckpt_path: str, model: BasicCrackModel, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    if not isinstance(state, dict):
        state = state.state_dict()
    current = model.state_dict()
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    loadable = {k: v for k, v in cleaned.items() if k in current and current[k].shape == v.shape}
    current.update(loadable)
    model.load_state_dict(current, strict=False)
    return len(loadable)


def train_v020(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    seed = int(cfg["train"].get("seed", 42))
    _set_seed(seed)

    save_dir = Path(cfg["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = create_crack_dataloader_v020(
        image_root=cfg["data"]["train_image_root"],
        label_root=cfg["data"].get("train_label_root"),
        mask_root=cfg["data"].get("train_mask_root"),
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        augment=True,
        workers=int(cfg["train"]["workers"]),
    )

    val_loader, _ = create_crack_dataloader_v020(
        image_root=cfg["data"]["val_image_root"],
        label_root=cfg["data"].get("val_label_root"),
        mask_root=cfg["data"].get("val_mask_root"),
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        augment=False,
        workers=int(cfg["train"]["workers"]),
    )

    model = BasicCrackModel(
        cfg_path=cfg["model"]["cfg_path"],
        num_classes=int(cfg["model"]["num_classes"]),
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(int(cfg["train"]["epochs"]), 1))

    criterion = CrackBasicLossV020(
        num_classes=int(cfg["model"]["num_classes"]),
        alpha_threshold_loss=float(cfg["loss"]["alpha_threshold_loss"]),
        anchor_t=float(cfg["loss"]["anchor_t"]),
        box_weight=float(cfg["loss"]["box_weight"]),
        obj_weight=float(cfg["loss"]["obj_weight"]),
        cls_weight=float(cfg["loss"]["cls_weight"]),
        cls_pw=float(cfg["loss"]["cls_pw"]),
        obj_pw=float(cfg["loss"]["obj_pw"]),
        fl_gamma=float(cfg["loss"].get("fl_gamma", 0.0)),
    )

    start_epoch = 0
    best_metric = 0.0
    resume_path = str(cfg["train"].get("resume_checkpoint", "") or "")
    pretrained_path = str(cfg["train"].get("pretrained_checkpoint", "") or "")
    if pretrained_path and not resume_path:
        loaded = _load_pretrained_model(pretrained_path, model, device)
        print(f"Loaded {loaded} shape-matched tensors from pretrained checkpoint: {pretrained_path}")
    if resume_path:
        state = _load_checkpoint(resume_path, model, optimizer, scheduler, device)
        start_epoch = int(state["start_epoch"])
        best_metric = float(state["best_metric"])
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    anchor_vecs = _anchor_vecs_from_model(model)
    epochs = int(cfg["train"]["epochs"])
    val_interval = max(int(cfg["train"].get("val_interval", 1)), 1)

    log_file = save_dir / "train_log_v020.txt"

    for epoch in range(start_epoch, epochs):
        model.train()

        running = {
            "loss_total": 0.0,
            "loss_loc": 0.0,
            "loss_conf": 0.0,
            "loss_cls": 0.0,
            "loss_thr": 0.0,
            "num_gt_boxes": 0.0,
            "num_positive_anchors": 0.0,
            "thr_pred_mean_pos": 0.0,
            "thr_target_mean_pos": 0.0,
            "thr_abs_err_mean_pos": 0.0,
        }
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"train_v020 epoch {epoch + 1}/{epochs}")
        for imgs, targets, _masks, _paths, _shapes in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            out = model.forward_train(imgs)
            preds = out["preds"]
            loss, items = criterion(preds=preds, targets=targets, anchor_vecs=anchor_vecs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n_batches += 1
            for k in running:
                running[k] += float(items[k].item())

            pbar.set_postfix({
                "lt": f"{items['loss_total'].item():.4f}",
                "lb": f"{items['loss_loc'].item():.4f}",
                "lo": f"{items['loss_conf'].item():.4f}",
                "lc": f"{items['loss_cls'].item():.4f}",
                "lthr": f"{items['loss_thr'].item():.4f}",
                "gt": int(items["num_gt_boxes"].item()),
                "pos": int(items["num_positive_anchors"].item()),
                "thre": f"{items['thr_abs_err_mean_pos'].item():.4f}",
            })

        scheduler.step()

        if n_batches > 0:
            for k in running:
                running[k] /= n_batches

        val_metrics = {}
        if ((epoch + 1) % val_interval) == 0:
            val_metrics = evaluate_basic_model(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=int(cfg["model"]["num_classes"]),
                conf_thres=float(cfg["val"]["conf_thres"]),
                iou_thres=float(cfg["val"]["iou_thres"]),
                class_names=cfg["model"].get("class_names"),
                save_vis=bool(cfg["val"].get("save_visualization", False)),
                vis_dir=str(cfg["val"].get("vis_dir", save_dir / "val_vis")),
                max_vis=int(cfg["val"].get("max_visualizations", 0)),
            )

        metric_for_best = float(val_metrics.get("map50", 0.0))
        if metric_for_best >= best_metric:
            best_metric = metric_for_best
            _save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, epoch, best_metric)

        _save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, epoch, best_metric)

        line = {
            "epoch": epoch + 1,
            **running,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "alpha_threshold_loss": float(cfg["loss"]["alpha_threshold_loss"]),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(str(line) + "\n")

        print(line)

    print(f"Training finished. Best map50={best_metric:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v0.2.0 basic-model trainer")
    parser.add_argument("--config", type=str, default="configs/crack_v020_basic.yaml")
    args = parser.parse_args()
    train_v020(args.config)


if __name__ == "__main__":
    main()
