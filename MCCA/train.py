import argparse
import csv
import math
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CrackSegmentation, build_crack_splits
from models import available_models, create_model
from utils import (
    BinaryPaperMetrics,
    BinarySegMetrics,
    Denormalize,
    MCCABinaryLoss,
    PolyLR,
    load_checkpoint,
    save_diagnostic_pack,
    save_checkpoint,
)
from utils import ext_transforms as et


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _append_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _append_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def get_argparser(defaults=None):
    defaults = defaults or {}

    def d(name, value):
        return defaults.get(name, value)

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--config", type=str, default=d("config", None))
    parser.add_argument("--data_root", type=str, default=d("data_root", "../data/private"))
    parser.add_argument("--model", type=str, default=d("model", "full_mcca"), choices=available_models())
    parser.add_argument("--model_variant", type=str, default=d("model_variant", ""), choices=["", "baseline_plain", "baseline_deepsup", "mcca_no_mscfm", "mcca_no_cam", "full_mcca"])
    parser.add_argument("--input_size", type=int, default=d("input_size", 640))
    parser.add_argument("--batch_size", type=int, default=d("batch_size", 2))
    parser.add_argument("--val_batch_size", type=int, default=d("val_batch_size", 1))
    parser.add_argument("--total_itrs", type=int, default=d("total_itrs", 2000))
    parser.add_argument("--lr", type=float, default=d("lr", 1e-3))
    parser.add_argument("--weight_decay", type=float, default=d("weight_decay", 1e-4))
    parser.add_argument("--gpu_id", type=str, default=d("gpu_id", "0"))
    parser.add_argument("--random_seed", type=int, default=d("random_seed", 1))
    parser.add_argument("--print_interval", type=int, default=d("print_interval", 10))
    parser.add_argument("--val_interval", type=int, default=d("val_interval", 100))
    parser.add_argument("--val_log_interval", type=int, default=d("val_log_interval", 100))
    parser.add_argument("--val_max_batches", type=int, default=d("val_max_batches", 0))
    parser.add_argument("--val_max_samples", type=int, default=d("val_max_samples", 0))
    parser.add_argument("--train_max_samples", type=int, default=d("train_max_samples", 0))
    parser.add_argument("--progress_mode", type=str, default=d("progress_mode", "auto"), choices=["auto", "tqdm", "plain"])
    parser.add_argument("--num_workers", type=int, default=d("num_workers", 2))
    parser.add_argument("--ckpt", type=str, default=d("ckpt", None))
    parser.add_argument("--continue_training", action=argparse.BooleanOptionalAction, default=d("continue_training", False))
    parser.add_argument("--test_only", action=argparse.BooleanOptionalAction, default=d("test_only", False))
    parser.add_argument("--save_dir", type=str, default=d("save_dir", "outputs"))
    parser.add_argument("--threshold", type=float, default=d("threshold", 0.55))

    parser.add_argument("--loss_name", type=str, default=d("loss_name", "weighted_bce_dice"), choices=["bce", "weighted_bce", "weighted_bce_dice", "weighted_bce_tversky"])
    parser.add_argument("--loss_pos_weight_mode", type=str, default=d("loss_pos_weight_mode", "capped_auto"), choices=["auto", "fixed", "capped_auto"])
    parser.add_argument("--loss_fixed_pos_weight", type=float, default=d("loss_fixed_pos_weight", 16.0))
    parser.add_argument("--loss_max_pos_weight", type=float, default=d("loss_max_pos_weight", 24.0))
    parser.add_argument("--loss_wbce_weight", type=float, default=d("loss_wbce_weight", 1.0))
    parser.add_argument("--loss_dice_weight", type=float, default=d("loss_dice_weight", 1.0))
    parser.add_argument("--loss_side_loss_weight", type=float, default=d("loss_side_loss_weight", 0.10))

    parser.add_argument("--pretrained_backbone", action=argparse.BooleanOptionalAction, default=d("pretrained_backbone", False))
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=d("dry_run", False))

    parser.add_argument("--tiny_overfit", action=argparse.BooleanOptionalAction, default=d("tiny_overfit", False))
    parser.add_argument("--tiny_overfit_num_samples", type=int, default=d("tiny_overfit_num_samples", 8))
    parser.add_argument("--tiny_overfit_disable_heavy_aug", action=argparse.BooleanOptionalAction, default=d("tiny_overfit_disable_heavy_aug", True))
    parser.add_argument("--tiny_overfit_eval_on_train", action=argparse.BooleanOptionalAction, default=d("tiny_overfit_eval_on_train", True))
    parser.add_argument("--tiny_overfit_list_path", type=str, default=d("tiny_overfit_list_path", None))

    parser.add_argument("--enable_paper_metrics", action=argparse.BooleanOptionalAction, default=d("enable_paper_metrics", True))
    parser.add_argument("--paper_metric_thresholds", type=int, default=d("paper_metric_thresholds", 101))
    parser.add_argument("--fbeta_beta", type=float, default=d("fbeta_beta", 0.3))

    parser.add_argument("--save_vis", action=argparse.BooleanOptionalAction, default=d("save_vis", True))
    parser.add_argument("--num_vis_samples", type=int, default=d("num_vis_samples", 2))
    parser.add_argument("--save_train_vis", action=argparse.BooleanOptionalAction, default=d("save_train_vis", True))

    parser.add_argument("--analysis_postprocess", action=argparse.BooleanOptionalAction, default=d("analysis_postprocess", False))
    parser.add_argument("--analysis_min_component_size", type=int, default=d("analysis_min_component_size", 0))
    parser.add_argument("--analysis_morph_open_kernel", type=int, default=d("analysis_morph_open_kernel", 0))
    return parser


def _select_tiny_subset(train_files, opts):
    num = min(opts.tiny_overfit_num_samples, len(train_files))
    list_path = opts.tiny_overfit_list_path
    if list_path is None:
        list_path = os.path.join(opts.save_dir, f"tiny_overfit_{num}_seed{opts.random_seed}.txt")

    if os.path.isfile(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            selected = [line.strip() for line in f.readlines() if line.strip()]
        selected = [s for s in selected if s in set(train_files)]
        selected = selected[:num]
    else:
        selected = sorted(train_files)[:num]
        list_dir = os.path.dirname(list_path)
        if list_dir:
            os.makedirs(list_dir, exist_ok=True)
        with open(list_path, "w", encoding="utf-8") as f:
            for name in selected:
                f.write(name + "\n")

    print(f"[tiny_overfit] using {len(selected)} samples from {list_path}")
    print("[tiny_overfit] sample list:", ", ".join(selected))
    return selected


def build_datasets(opts):
    train_root = val_root = opts.data_root
    if os.path.isdir(os.path.join(opts.data_root, "train", "images")):
        train_root = os.path.join(opts.data_root, "train")
        val_root = os.path.join(opts.data_root, opts.eval_split if opts.test_only else "val")
        train_files = sorted(f for f in os.listdir(os.path.join(train_root, "images")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        val_files = sorted(f for f in os.listdir(os.path.join(val_root, "images")) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    else:
        train_files, val_files = build_crack_splits(opts.data_root, random_seed=opts.random_seed)

    if opts.tiny_overfit:
        tiny_files = _select_tiny_subset(train_files, opts)
        train_files = tiny_files
        if opts.tiny_overfit_eval_on_train:
            val_files = tiny_files
            val_root = train_root

    if opts.tiny_overfit and opts.tiny_overfit_disable_heavy_aug:
        train_transform = et.ExtCompose([
            et.ExtResize((opts.input_size, opts.input_size)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        train_transform = et.ExtCompose([
            et.ExtRandomRotation(degrees=180),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop((opts.input_size, opts.input_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtRandomVerticalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = et.ExtCompose([
        et.ExtResize((opts.input_size, opts.input_size)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dst = CrackSegmentation(root=train_root, transform=train_transform, file_list=train_files)
    val_dst = CrackSegmentation(root=val_root, transform=val_transform, file_list=val_files)
    return train_dst, val_dst


def validate(model, loader, device, threshold=0.5, paper_metrics=None, vis_cfg=None, val_log_interval=0, max_batches=0, max_samples=0, use_tqdm=True):
    model.eval()
    metrics_raw = BinarySegMetrics()
    metrics_raw.reset()

    metrics_post = BinarySegMetrics()
    metrics_post.reset()

    paper_metrics_post = None
    use_post = False

    denorm = None
    vis_saved = 0
    vis_max = 0
    vis_dir = None
    vis_prefix = "val"
    vis_itr = 0
    if vis_cfg is not None:
        denorm = vis_cfg["denorm"]
        vis_max = vis_cfg["num_vis_samples"]
        vis_dir = vis_cfg["vis_dir"]
        vis_prefix = vis_cfg.get("prefix", "val")
        vis_itr = vis_cfg.get("itr", 0)
        use_post = vis_cfg.get("use_postprocess", False)
        if use_post and paper_metrics is not None:
            paper_metrics_post = BinaryPaperMetrics(
                num_thresholds=vis_cfg.get("paper_metric_thresholds", 101),
                beta=vis_cfg.get("fbeta_beta", 0.3),
            )

    def postprocess_mask(mask_2d: np.ndarray, min_component_size: int = 0, morph_open_kernel: int = 0) -> np.ndarray:
        out = mask_2d.astype(np.uint8).copy()
        if min_component_size > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
            filtered = np.zeros_like(out)
            for lab in range(1, num):
                area = stats[lab, cv2.CC_STAT_AREA]
                if area >= min_component_size:
                    filtered[labels == lab] = 1
            out = filtered
        if morph_open_kernel and morph_open_kernel > 1:
            k = int(morph_open_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
            out = (out > 0).astype(np.uint8)
        return out

    total_batches = len(loader)
    effective_total = min(total_batches, max_batches) if max_batches and max_batches > 0 else total_batches
    batch_size_hint = getattr(loader, "batch_size", None) or 1
    if max_samples and max_samples > 0:
        sample_limited_batches = int(math.ceil(max_samples / float(batch_size_hint)))
        effective_total = min(effective_total, sample_limited_batches)
    if effective_total < total_batches:
        tqdm.write(f"[VAL_INFO] using subset {effective_total}/{total_batches} batches for this validation")
    if max_samples and max_samples > 0:
        tqdm.write(f"[VAL_INFO] max_samples={max_samples}")

    val_iter = loader
    val_pbar = None
    if use_tqdm:
        val_pbar = tqdm(
            loader,
            total=effective_total,
            desc="val",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=False,
            mininterval=0.5,
            ascii=True,
        )
        val_iter = val_pbar
    processed_samples = 0
    with torch.no_grad():
        for bi, (images, masks) in enumerate(val_iter, start=1):
            if bi > effective_total:
                break
            if max_samples and max_samples > 0:
                remaining = max_samples - processed_samples
                if remaining <= 0:
                    break
                if images.size(0) > remaining:
                    images = images[:remaining]
                    masks = masks[:remaining]
            if val_log_interval:
                if bi == 1 or bi % val_log_interval == 0 or bi == effective_total:
                    if use_tqdm and val_pbar is not None:
                        val_pbar.set_postfix_str(f"{bi}/{effective_total}")
                    else:
                        print(f"[VAL] {bi}/{effective_total}")
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            outputs = model(images)
            probs = torch.sigmoid(outputs["final_logit"])
            pred_mask = (probs >= threshold).to(torch.uint8).squeeze(1)
            processed_samples += images.size(0)

            targets = masks.cpu().numpy()
            preds = pred_mask.cpu().numpy()
            metrics_raw.update(targets, preds)

            if paper_metrics is not None:
                paper_metrics.update(targets, probs.squeeze(1).cpu().numpy())

            if use_post:
                preds_post = np.stack(
                    [
                        postprocess_mask(
                            p,
                            min_component_size=vis_cfg.get("analysis_min_component_size", 0),
                            morph_open_kernel=vis_cfg.get("analysis_morph_open_kernel", 0),
                        )
                        for p in preds
                    ],
                    axis=0,
                )
                metrics_post.update(targets, preds_post)
                if paper_metrics_post is not None:
                    paper_metrics_post.update(targets, preds_post.astype(np.float32))

            if vis_dir is not None and vis_saved < vis_max:
                bs = images.size(0)
                for b in range(bs):
                    if vis_saved >= vis_max:
                        break
                    img_np = (denorm(images[b].cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                    gt_mask = masks[b].cpu().numpy().astype(np.uint8)
                    side_probs = [
                        torch.sigmoid(s[b, 0]).cpu().numpy().astype(np.float32)
                        for s in outputs.get("side_logits", [])
                    ]
                    final_prob = probs[b, 0].cpu().numpy().astype(np.float32)
                    save_path = os.path.join(vis_dir, f"{vis_prefix}_itr{vis_itr:06d}_idx{vis_saved:03d}.png")
                    save_diagnostic_pack(save_path, img_np, gt_mask, side_probs, final_prob, threshold=threshold)
                    vis_saved += 1
    if val_pbar is not None:
        val_pbar.close()

    result = metrics_raw.get_results()
    if paper_metrics is not None:
        paper_result = paper_metrics.get_results()
        for k, v in paper_result.items():
            if k == "MAE" and "MAE" in result:
                result["ProbMAE"] = v
            else:
                result[k] = v

    if use_post:
        post_result = metrics_post.get_results()
        for k, v in post_result.items():
            result[f"post_{k}"] = v
        if paper_metrics_post is not None:
            post_paper = paper_metrics_post.get_results()
            for k, v in post_paper.items():
                result[f"post_{k}"] = v
    return result


def run_dry_step(model, device, criterion):
    model.train()
    x = torch.randn(1, 3, 640, 640, device=device)
    y = torch.randint(0, 2, (1, 640, 640), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer.zero_grad()
    outputs = model(x)
    losses = criterion(outputs, y)
    losses["total_loss"].backward()
    optimizer.step()
    return losses


def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default=None)
    initial_opts, _ = initial_parser.parse_known_args()

    cfg_defaults = {}
    if initial_opts.config is not None:
        with open(initial_opts.config, "r", encoding="utf-8") as f:
            cfg_defaults = json.load(f)

    opts = get_argparser(defaults=cfg_defaults).parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.progress_mode == "tqdm":
        use_tqdm = True
    elif opts.progress_mode == "plain":
        use_tqdm = False
    else:
        use_tqdm = bool(sys.stdout.isatty())

    if opts.model_variant:
        model = create_model("mcca", input_size=opts.input_size, pretrained_backbone=opts.pretrained_backbone, model_variant=opts.model_variant)
        exp_name = opts.model_variant
    else:
        model = create_model(opts.model, input_size=opts.input_size, pretrained_backbone=opts.pretrained_backbone)
        exp_name = opts.model
    model = model.to(device)

    criterion = MCCABinaryLoss(
        loss_name=opts.loss_name,
        pos_weight_mode=opts.loss_pos_weight_mode,
        fixed_pos_weight=opts.loss_fixed_pos_weight,
        max_pos_weight=opts.loss_max_pos_weight,
        wbce_weight=opts.loss_wbce_weight,
        dice_weight=opts.loss_dice_weight,
        side_loss_weight=opts.loss_side_loss_weight,
        use_deepsup=getattr(model, "use_deepsup", True),
    ).to(device)

    if opts.dry_run:
        losses = run_dry_step(model, device, criterion)
        print(f"dry_run_total_loss={losses['total_loss'].item():.6f}")
        return

    train_dst, val_dst = build_datasets(opts)

    train_loader = DataLoader(
        train_dst,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dst,
        batch_size=opts.val_batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
    )

    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    scheduler = PolyLR(optimizer, max_iters=opts.total_itrs, power=0.9)

    cur_itrs = 0
    best_iou = 0.0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        cur_itrs, best_iou = load_checkpoint(
            opts.ckpt,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            continue_training=opts.continue_training,
            map_location="cpu",
        )

    if opts.test_only:
        paper_metrics = BinaryPaperMetrics(num_thresholds=opts.paper_metric_thresholds, beta=opts.fbeta_beta) if opts.enable_paper_metrics else None
        vis_dir = os.path.join(opts.save_dir, "vis") if opts.save_vis else None
        val_score = validate(
            model,
            val_loader,
            device,
            threshold=opts.threshold,
            paper_metrics=paper_metrics,
            val_log_interval=opts.val_log_interval,
            max_batches=0,
            max_samples=opts.val_max_samples,
            use_tqdm=use_tqdm,
            vis_cfg={
                "denorm": denorm,
                "num_vis_samples": opts.num_vis_samples,
                "vis_dir": vis_dir,
                "prefix": "test",
                "itr": cur_itrs,
                "use_postprocess": opts.analysis_postprocess,
                "analysis_min_component_size": opts.analysis_min_component_size,
                "analysis_morph_open_kernel": opts.analysis_morph_open_kernel,
                "paper_metric_thresholds": opts.paper_metric_thresholds,
                "fbeta_beta": opts.fbeta_beta,
            } if opts.save_vis else None,
        )
        metrics_dir = os.path.join(opts.save_dir, "metrics")
        _write_json(
            os.path.join(metrics_dir, "test_metrics.json"),
            {
                "mode": "test_only",
                "threshold": opts.threshold,
                "ckpt": opts.ckpt,
                **val_score,
            },
        )
        print(BinarySegMetrics.to_str(val_score))
        return

    os.makedirs(opts.save_dir, exist_ok=True)
    with open(os.path.join(opts.save_dir, "effective_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(opts), f, indent=2)
    print(f"Train samples={len(train_dst)} Val samples={len(val_dst)}")
    print(f"Experiment={exp_name}")
    print(f"iters_per_epoch={len(train_loader)} (ceil({len(train_dst)}/{opts.batch_size}))")
    if opts.train_max_samples and opts.train_max_samples > 0:
        print(f"train_max_samples={opts.train_max_samples}")
    if opts.val_max_samples and opts.val_max_samples > 0:
        print(f"val_max_samples={opts.val_max_samples}")
    print(f"progress_mode={opts.progress_mode} resolved_use_tqdm={use_tqdm}")
    print(
        f"Loss={opts.loss_name} pos_mode={opts.loss_pos_weight_mode} fixed_pw={opts.loss_fixed_pos_weight} "
        f"max_pw={opts.loss_max_pos_weight} wbce_w={opts.loss_wbce_weight} dice_w={opts.loss_dice_weight} "
        f"side_w={opts.loss_side_loss_weight} threshold={opts.threshold}"
    )

    metrics_dir = os.path.join(opts.save_dir, "metrics")
    val_jsonl = os.path.join(metrics_dir, "val_metrics.jsonl")
    val_csv = os.path.join(metrics_dir, "val_metrics.csv")

    interval_loss = 0.0
    effective_train_batches = len(train_loader)
    if opts.train_max_samples and opts.train_max_samples > 0:
        effective_train_batches = min(len(train_loader), int(math.ceil(opts.train_max_samples / float(opts.batch_size))))

    total_epochs_est = int(math.ceil(opts.total_itrs / float(effective_train_batches))) if effective_train_batches > 0 else 0
    current_epoch = int(math.ceil(cur_itrs / float(effective_train_batches))) if effective_train_batches > 0 else 0
    best_itr = cur_itrs
    last_val_score = None

    while cur_itrs < opts.total_itrs:
        current_epoch += 1
        epoch_pbar = None
        if use_tqdm:
            epoch_pbar = tqdm(
                total=effective_train_batches,
                desc=f"train e{current_epoch}/{total_epochs_est}",
                file=sys.stdout,
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,
                ascii=True,
            )
        model.train()
        for batch_idx, (images, masks) in enumerate(train_loader, start=1):
            if cur_itrs >= opts.total_itrs:
                break
            if batch_idx > effective_train_batches:
                break

            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            losses = criterion(outputs, masks)
            loss = losses["total_loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            if epoch_pbar is not None:
                epoch_pbar.update(1)

            interval_loss += loss.item()
            if cur_itrs % opts.print_interval == 0:
                avg_loss = interval_loss / opts.print_interval
                if epoch_pbar is not None:
                    epoch_pbar.set_postfix(
                        {
                            "itr": f"{cur_itrs}/{opts.total_itrs}",
                            "loss": f"{avg_loss:.4f}",
                            "final": f"{losses['final_loss'].item():.4f}",
                            "aux": f"{losses['aux_loss'].item():.4f}",
                            "pw": f"{losses['pos_weight']:.2f}",
                        }
                    )
                else:
                    print(
                        f"Itr {cur_itrs}/{opts.total_itrs} Loss={avg_loss:.6f} "
                        f"Final={losses['final_loss'].item():.6f} Aux={losses['aux_loss'].item():.6f} "
                        f"FinalBCE={losses['final_bce'].item():.6f} FinalDice={losses['final_dice'].item():.6f} "
                        f"PosW={losses['pos_weight']:.4f}"
                    )
                interval_loss = 0.0

            if opts.save_train_vis and opts.tiny_overfit and cur_itrs % opts.print_interval == 0:
                with torch.no_grad():
                    img_np = (denorm(images[0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                    gt_mask = masks[0].detach().cpu().numpy().astype(np.uint8)
                    side_probs = [
                        torch.sigmoid(s[0, 0]).detach().cpu().numpy().astype(np.float32)
                        for s in outputs.get("side_logits", [])
                    ]
                    final_prob = torch.sigmoid(outputs["final_logit"][0, 0]).detach().cpu().numpy().astype(np.float32)
                    train_vis_dir = os.path.join(opts.save_dir, "vis")
                    train_vis_path = os.path.join(train_vis_dir, f"train_itr{cur_itrs:06d}_idx000.png")
                    save_diagnostic_pack(train_vis_path, img_np, gt_mask, side_probs, final_prob, threshold=opts.threshold)

            if cur_itrs % opts.val_interval == 0:
                latest_path = os.path.join(opts.save_dir, f"latest_{exp_name}.pth")
                save_checkpoint(latest_path, model, optimizer, scheduler, cur_itrs, best_iou)
                if use_tqdm:
                    tqdm.write(f"[VAL_TRIGGER] epoch={current_epoch} itr={cur_itrs} interval={opts.val_interval}")
                else:
                    print(f"[VAL_TRIGGER] epoch={current_epoch} itr={cur_itrs} interval={opts.val_interval}")

                paper_metrics = BinaryPaperMetrics(num_thresholds=opts.paper_metric_thresholds, beta=opts.fbeta_beta) if opts.enable_paper_metrics else None
                vis_dir = os.path.join(opts.save_dir, "vis") if opts.save_vis else None
                val_score = validate(
                    model,
                    val_loader,
                    device,
                    threshold=opts.threshold,
                    paper_metrics=paper_metrics,
                    val_log_interval=opts.val_log_interval,
                    max_batches=opts.val_max_batches,
                    max_samples=opts.val_max_samples,
                    use_tqdm=use_tqdm,
                    vis_cfg={
                        "denorm": denorm,
                        "num_vis_samples": opts.num_vis_samples,
                        "vis_dir": vis_dir,
                        "prefix": f"val_ep{current_epoch:03d}",
                        "itr": cur_itrs,
                        "use_postprocess": opts.analysis_postprocess,
                        "analysis_min_component_size": opts.analysis_min_component_size,
                        "analysis_morph_open_kernel": opts.analysis_morph_open_kernel,
                        "paper_metric_thresholds": opts.paper_metric_thresholds,
                        "fbeta_beta": opts.fbeta_beta,
                    } if opts.save_vis else None,
                )

                val_row = {
                    "epoch": current_epoch,
                    "itr": cur_itrs,
                    "threshold": opts.threshold,
                    **val_score,
                }
                _write_json(os.path.join(metrics_dir, f"val_ep{current_epoch:03d}_itr{cur_itrs:06d}.json"), val_row)
                _append_jsonl(val_jsonl, val_row)
                _append_csv(val_csv, val_row)

                if use_tqdm:
                    tqdm.write(BinarySegMetrics.to_str(val_score))
                else:
                    print(BinarySegMetrics.to_str(val_score))

                last_val_score = val_score

                if val_score["IoU"] > best_iou:
                    best_iou = val_score["IoU"]
                    best_itr = cur_itrs
                    best_path = os.path.join(opts.save_dir, f"best_{exp_name}.pth")
                    save_checkpoint(best_path, model, optimizer, scheduler, cur_itrs, best_iou)

        if epoch_pbar is not None:
            epoch_pbar.close()

    summary = {
        "experiment": exp_name,
        "total_itrs": opts.total_itrs,
        "effective_train_batches": effective_train_batches,
        "estimated_total_epochs": total_epochs_est,
        "final_itr": cur_itrs,
        "best_iou": best_iou,
        "best_itr": best_itr,
        "best_checkpoint": os.path.join(opts.save_dir, f"best_{exp_name}.pth"),
        "latest_checkpoint": os.path.join(opts.save_dir, f"latest_{exp_name}.pth"),
        "last_val": last_val_score,
    }
    _write_json(os.path.join(metrics_dir, "final_summary.json"), summary)


if __name__ == "__main__":
    main()
