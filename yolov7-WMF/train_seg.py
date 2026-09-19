import argparse
import json
import random
import numpy as np
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo import Model
from utils.general import increment_path, set_logging
from utils.segmentation import (
    SegmentationDataset,
    SegLoss,
    compute_seg_stats,
    load_seg_data_config,
    save_seg_visual,
)
from utils.torch_utils import select_device


def load_weights(model, weight_path, device):
    if not weight_path:
        return
    try:
        ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        src = ckpt['model']
        if isinstance(src, dict):
            state = src
        else:
            state = src.float().state_dict()
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = ckpt.state_dict()
    dst = model.state_dict()
    keep = {k: v for k, v in state.items() if k in dst and v.shape == dst[k].shape}
    model.load_state_dict(keep, strict=False)
    print(f'Loaded {len(keep)} tensors from {weight_path}')


@torch.no_grad()
def run_val(model, loader, device, save_dir=None, max_save=8, max_batches=-1, threshold=0.5):
    model.eval()
    tot = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
    saved = 0
    for bi, (imgs, masks, paths) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        stats = compute_seg_stats(logits, masks, threshold=threshold)
        tot['tp'] += stats['tp']
        tot['fp'] += stats['fp']
        tot['fn'] += stats['fn']
        if save_dir is not None and saved < max_save:
            probs = torch.sigmoid(logits)
            for b in range(imgs.size(0)):
                if saved >= max_save:
                    break
                save_path = save_dir / f'val_{saved:03d}_{Path(paths[b]).stem}.jpg'
                save_seg_visual(imgs[b], masks[b], probs[b], save_path, threshold=threshold)
                saved += 1
        if max_batches > 0 and (bi + 1) >= max_batches:
            break
    eps = 1e-6
    tp, fp, fn = tot['tp'], tot['fp'], tot['fn']
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/seg')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--lambda-bce', type=float, default=1.0)
    parser.add_argument('--lambda-dice', type=float, default=1.0)
    parser.add_argument('--pos-weight', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-train-steps', type=int, default=-1)
    parser.add_argument('--max-val-batches', type=int, default=-1)
    parser.add_argument('--no-augment', action='store_true', default=False)
    return parser.parse_args()


def main(opt):
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch)
    data_cfg = load_seg_data_config(opt.data)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    save_dir.mkdir(parents=True, exist_ok=True)
    val_vis_dir = save_dir / 'val_vis'
    val_vis_dir.mkdir(parents=True, exist_ok=True)
    train_vis_dir = save_dir / 'train_vis'
    train_vis_dir.mkdir(parents=True, exist_ok=True)

    train_set = SegmentationDataset(
        images_dir=data_cfg['train_images'],
        masks_dir=data_cfg['train_masks'],
        img_size=opt.img,
        augment=not opt.no_augment
    )
    val_set = SegmentationDataset(
        images_dir=data_cfg['val_images'],
        masks_dir=data_cfg['val_masks'],
        img_size=opt.img,
        augment=False
    )
    train_loader = DataLoader(
        train_set,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=opt.batch,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=False
    )

    model = Model(opt.cfg, ch=3, nc=1).to(device)
    load_weights(model, opt.weights, device)
    criterion = SegLoss(lambda_bce=opt.lambda_bce, lambda_dice=opt.lambda_dice, pos_weight=opt.pos_weight)
    optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(opt.epochs, 1), eta_min=opt.lr * 0.05)

    best_f1 = -1.0
    metrics_log = []

    for epoch in range(opt.epochs):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f'epoch {epoch + 1}/{opt.epochs}')
        loss_sum = 0.0
        bce_sum = 0.0
        dice_sum = 0.0
        n_step = 0
        for imgs, masks, paths in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss, bce, dice = criterion(logits, masks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            bce_sum += bce.item()
            dice_sum += dice.item()
            n_step += 1
            pbar.set_postfix(loss=f'{loss_sum / n_step:.4f}', bce=f'{bce_sum / n_step:.4f}', dice=f'{dice_sum / n_step:.4f}')
            if n_step == 1:
                probs = torch.sigmoid(logits.detach())
                save_seg_visual(imgs[0], masks[0], probs[0], train_vis_dir / f'epoch_{epoch + 1:03d}.jpg', threshold=opt.threshold)
            if opt.max_train_steps > 0 and n_step >= opt.max_train_steps:
                break

        scheduler.step()
        val_stats = run_val(
            model,
            val_loader,
            device,
            save_dir=val_vis_dir if epoch % 5 == 0 else None,
            max_save=6,
            max_batches=opt.max_val_batches,
            threshold=opt.threshold,
        )
        record = {
            'epoch': epoch + 1,
            'loss': loss_sum / max(n_step, 1),
            'bce': bce_sum / max(n_step, 1),
            'dice_loss': dice_sum / max(n_step, 1),
            'threshold': opt.threshold,
            'pos_weight': opt.pos_weight,
            **val_stats
        }
        metrics_log.append(record)
        print(json.dumps(record, ensure_ascii=False))
        ckpt = {
            'epoch': epoch + 1,
            'model': deepcopy(model).half(),
            'optimizer': optimizer.state_dict(),
            'metrics': record,
            'cfg': opt.cfg,
            'data': opt.data
        }
        torch.save(ckpt, save_dir / 'last.pt')
        if val_stats['f1'] > best_f1:
            best_f1 = val_stats['f1']
            torch.save(ckpt, save_dir / 'best.pt')

    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_log, f, indent=2, ensure_ascii=False)
    print(f'Saved results to {save_dir}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
