from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CrackLoss
from .metrics import BinaryStats
from .utils import append_metrics_csv, ensure_dir, save_val_composite, write_json

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class TrainConfig:
    run_name: str
    epochs: int = 20
    batch_size: int = 4
    val_batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 0
    val_interval_epochs: int = 5
    aux_weight: float = 1.0
    save_top_k_vis: int = 8
    tolerance_px: int = 2
    use_wandb: bool = False
    wandb_project: str = "crackresunet-repro"
    wandb_entity: str = ""
    wandb_group: str = "formal-main"
    wandb_job_type: str = "train"
    wandb_tags: str = ""
    wandb_mode: str = "online"
    wandb_run_name: str = ""
    wandb_config: Dict[str, object] | None = None


def _to_image_np(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = x * std + mean
    x = torch.clamp(x, 0.0, 1.0)
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: CrackLoss,
    device: torch.device,
    epoch: int,
    tolerance_px: int,
) -> Dict[str, float]:
    model.train()
    running_total = 0.0
    running_main = 0.0
    running_aux = 0.0
    stats = BinaryStats()

    pbar = tqdm(loader, desc=f"train epoch {epoch:03d}", leave=True)
    for batch in pbar:
        images, masks, _ = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss_dict = criterion(outputs, masks)
        loss = loss_dict["total_loss"]
        loss.backward()
        optimizer.step()

        running_total += float(loss.item())
        running_main += float(loss_dict["main_loss"].item())
        running_aux += float(loss_dict["aux_loss"].item())

        probs = torch.softmax(outputs["main_logits"], dim=1)[:, 1]
        preds = (probs > 0.5).long().detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()
        for b in range(preds.shape[0]):
            stats.update(preds[b], targets[b])
            stats.update_tolerant(preds[b], targets[b], radius=tolerance_px)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            main=f"{loss_dict['main_loss'].item():.4f}",
            aux=f"{loss_dict['aux_loss'].item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    out = stats.compute()
    out.update(
        {
            "train_loss_total": running_total / max(1, len(loader)),
            "train_loss_main": running_main / max(1, len(loader)),
            "train_loss_aux": running_aux / max(1, len(loader)),
            "train_lr": float(optimizer.param_groups[0]["lr"]),
        }
    )
    return out


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: CrackLoss,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    save_top_k_vis: int,
    tolerance_px: int,
) -> Dict[str, float]:
    model.eval()
    stats = BinaryStats()
    val_losses: List[float] = []
    vis_paths: List[Path] = []

    vis_dir = out_dir / "val_vis" / f"epoch_{epoch:04d}"
    ensure_dir(vis_dir)

    pbar = tqdm(loader, desc=f"val epoch {epoch:03d}", leave=True)
    for idx, batch in enumerate(pbar):
        images, masks, names = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        outputs = model(images)
        loss_dict = criterion(outputs, masks)
        val_losses.append(float(loss_dict["total_loss"].item()))

        probs = torch.softmax(outputs["main_logits"], dim=1)[:, 1]
        preds = (probs > 0.5).long().detach().cpu().numpy()
        targets = masks.detach().cpu().numpy()

        for b in range(preds.shape[0]):
            stats.update(preds[b], targets[b])
            stats.update_tolerant(preds[b], targets[b], radius=tolerance_px)

        if idx < save_top_k_vis:
            for b in range(min(images.shape[0], 1)):
                image_np = _to_image_np(images[b])
                save_val_composite(vis_dir, names[b], image_np, targets[b], preds[b])
                vis_paths.append(vis_dir / f"{names[b]}_panel6.png")

        pbar.set_postfix(val_loss=f"{np.mean(val_losses):.4f}")

    score = stats.compute()
    score["ValLoss"] = float(np.mean(val_losses) if val_losses else 0.0)
    score["_vis_paths"] = vis_paths
    return score


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_cfg: TrainConfig,
    out_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    ensure_dir(out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = CrackLoss(aux_weight=train_cfg.aux_weight)

    wb_run = None
    if train_cfg.use_wandb:
        if wandb is None:
            print("[WARN] wandb not installed. Continue without wandb logging.")
        else:
            tags = [t.strip() for t in train_cfg.wandb_tags.split(",") if t.strip()]
            wb_run = wandb.init(
                project=train_cfg.wandb_project,
                entity=(train_cfg.wandb_entity or None),
                group=train_cfg.wandb_group,
                job_type=train_cfg.wandb_job_type,
                name=(train_cfg.wandb_run_name or train_cfg.run_name),
                mode=train_cfg.wandb_mode,
                tags=tags,
                config=train_cfg.wandb_config or {},
            )

    best_f1 = -1.0
    best_dice = -1.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch,
            tolerance_px=train_cfg.tolerance_px,
        )
        train_row = {"epoch": epoch, **train_metrics}
        append_metrics_csv(out_dir / "train_metrics_epoch.csv", train_row)

        if wb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_total": train_metrics["train_loss_total"],
                    "train/loss_main": train_metrics["train_loss_main"],
                    "train/loss_aux": train_metrics["train_loss_aux"],
                    "train/lr": train_metrics["train_lr"],
                    "train/precision": train_metrics["Precision"],
                    "train/recall": train_metrics["Recall"],
                    "train/f1": train_metrics["F1"],
                    "train/dice": train_metrics["Dice"],
                    "train/iou": train_metrics["IoU"],
                    "train/tol_precision": train_metrics["TolPrecision"],
                    "train/tol_recall": train_metrics["TolRecall"],
                    "train/tol_f1": train_metrics["TolF1"],
                }
            )

        if epoch % train_cfg.val_interval_epochs == 0 or epoch == train_cfg.epochs:
            val_metrics = validate(
                model,
                val_loader,
                criterion,
                device,
                out_dir,
                epoch,
                save_top_k_vis=train_cfg.save_top_k_vis,
                tolerance_px=train_cfg.tolerance_px,
            )
            vis_paths = val_metrics.pop("_vis_paths", [])
            prev_best_dice = best_dice
            best_f1 = max(best_f1, val_metrics["F1"])
            best_dice = max(best_dice, val_metrics["Dice"])
            val_metrics["BestF1SoFar"] = best_f1
            val_metrics["BestDiceSoFar"] = best_dice
            val_row = {"epoch": epoch, **val_metrics}
            append_metrics_csv(out_dir / "metrics_epoch.csv", val_row)

            if wb_run is not None:
                payload = {
                    "epoch": epoch,
                    "val/loss": val_metrics["ValLoss"],
                    "val/precision": val_metrics["Precision"],
                    "val/recall": val_metrics["Recall"],
                    "val/f1": val_metrics["F1"],
                    "val/dice": val_metrics["Dice"],
                    "val/iou": val_metrics["IoU"],
                    "val/tol_precision": val_metrics["TolPrecision"],
                    "val/tol_recall": val_metrics["TolRecall"],
                    "val/tol_f1": val_metrics["TolF1"],
                    "val/best_dice_so_far": val_metrics["BestDiceSoFar"],
                    "val/best_f1_so_far": val_metrics["BestF1SoFar"],
                }
                if vis_paths:
                    payload["val/samples"] = [wandb.Image(str(p), caption=p.name) for p in vis_paths[:4] if p.exists()]
                wandb.log(payload)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_dir / f"epoch_{epoch:04d}.pth",
            )

            if val_metrics["Dice"] > prev_best_dice:
                best_metrics = {"best_epoch": epoch, **val_metrics}
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                write_json(out_dir / "best_metrics.json", best_metrics)

    if wb_run is not None:
        artifact = wandb.Artifact(f"{train_cfg.run_name}-artifacts", type="run_artifacts")
        for rel in [
            "train_metrics_epoch.csv",
            "metrics_epoch.csv",
            "best_metrics.json",
            "summary.md",
            "formal_main_summary.md",
            "checkpoints/best_model.pth",
        ]:
            p = out_dir / rel
            if p.exists():
                artifact.add_file(str(p), name=rel)
        wb_run.log_artifact(artifact)
        wb_run.finish()

    return best_metrics
