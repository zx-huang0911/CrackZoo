import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo import Model
from utils.general import increment_path, set_logging
from utils.segmentation import SegmentationDataset, compute_seg_stats, load_seg_data_config, save_seg_visual
from utils.torch_utils import select_device


def load_ckpt(model, weight_path, device):
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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--img', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/seg_val')
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--save-samples', type=int, default=20)
    parser.add_argument('--max-batches', type=int, default=-1)
    return parser.parse_args()


@torch.no_grad()
def main(opt):
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch)
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = save_dir / 'vis'
    vis_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = load_seg_data_config(opt.data)
    val_set = SegmentationDataset(
        images_dir=data_cfg['val_images'],
        masks_dir=data_cfg['val_masks'],
        img_size=opt.img,
        augment=False
    )
    loader = DataLoader(
        val_set,
        batch_size=opt.batch,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True
    )

    model = Model(opt.cfg, ch=3, nc=1).to(device)
    load_ckpt(model, opt.weights, device)
    model.eval()

    tot = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
    saved = 0
    pbar = tqdm(loader, total=len(loader), desc='val')
    for bi, (imgs, masks, paths) in enumerate(pbar):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
        stats = compute_seg_stats(logits, masks)
        tot['tp'] += stats['tp']
        tot['fp'] += stats['fp']
        tot['fn'] += stats['fn']
        probs = torch.sigmoid(logits)
        for b in range(imgs.size(0)):
            if saved >= opt.save_samples:
                break
            save_seg_visual(imgs[b], masks[b], probs[b], vis_dir / f'{saved:03d}_{Path(paths[b]).stem}.jpg')
            saved += 1
        if opt.max_batches > 0 and (bi + 1) >= opt.max_batches:
            break

    eps = 1e-6
    tp, fp, fn = tot['tp'], tot['fp'], tot['fn']
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    result = {'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou, 'dice': dice}
    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, ensure_ascii=False))
    print(f'Saved results to {save_dir}')


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
