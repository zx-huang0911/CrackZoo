from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_seg_data_config(data_path):
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    if all(k in data for k in ['train_images', 'train_masks', 'val_images', 'val_masks']):
        return data
    if 'path' in data:
        root = Path(data['path'])
        a_train_images = root / 'images' / 'train'
        a_train_masks = root / 'masks' / 'train'
        a_val_images = root / 'images' / 'val'
        a_val_masks = root / 'masks' / 'val'
        b_train_images = root / 'train' / 'images'
        b_train_masks = root / 'train' / 'masks'
        b_val_images = root / 'val' / 'images'
        b_val_masks = root / 'val' / 'masks'
        if a_train_images.is_dir() and a_train_masks.is_dir():
            data['train_images'] = str(a_train_images)
            data['train_masks'] = str(a_train_masks)
            data['val_images'] = str(a_val_images if a_val_images.is_dir() else (root / 'images' / 'test'))
            data['val_masks'] = str(a_val_masks if a_val_masks.is_dir() else (root / 'masks' / 'test'))
            return data
        if b_train_images.is_dir() and b_train_masks.is_dir():
            data['train_images'] = str(b_train_images)
            data['train_masks'] = str(b_train_masks)
            data['val_images'] = str(b_val_images if b_val_images.is_dir() else (root / 'test' / 'images'))
            data['val_masks'] = str(b_val_masks if b_val_masks.is_dir() else (root / 'test' / 'masks'))
            return data
    raise ValueError('Invalid segmentation data config')


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=640, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = int(img_size)
        self.augment = augment
        if not self.images_dir.is_dir() or not self.masks_dir.is_dir():
            raise FileNotFoundError(f'Invalid dataset dirs: {self.images_dir} {self.masks_dir}')
        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
        if len(self.images) == 0:
            raise RuntimeError(f'No images in {self.images_dir}')
        mask_files = [p for p in self.masks_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        self.mask_by_stem = {p.stem: p for p in mask_files}
        missing = [p.name for p in self.images if p.stem not in self.mask_by_stem]
        if missing:
            raise RuntimeError(f'Missing masks for {len(missing)} images, examples: {missing[:5]}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_by_stem[img_path.stem]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f'Failed to read {img_path} or {mask_path}')
        if self.augment:
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            if random.random() < 0.2:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        mask = torch.from_numpy(mask).unsqueeze(0).contiguous()
        return img, mask, str(img_path)


def dice_loss(logits, target, eps=1e-6):
    pred = torch.sigmoid(logits)
    pred = pred.reshape(pred.size(0), -1)
    target = target.reshape(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    den = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


class SegLoss(torch.nn.Module):
    def __init__(self, lambda_bce=1.0, lambda_dice=1.0, pos_weight=1.0):
        super().__init__()
        self.lambda_bce = float(lambda_bce)
        self.lambda_dice = float(lambda_dice)
        self.pos_weight = float(pos_weight)

    def forward(self, logits, target):
        pos_weight = target.new_tensor([self.pos_weight])
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        dice = dice_loss(logits, target)
        loss = self.lambda_bce * bce + self.lambda_dice * dice
        return loss, bce.detach(), dice.detach()


def compute_seg_stats(logits, target, threshold=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    target = (target >= 0.5).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'dice': dice
    }


def save_seg_visual(image_tensor, gt_mask_tensor, pred_mask_tensor, save_path, threshold=0.5):
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    gt = (gt_mask_tensor.detach().cpu().numpy().squeeze() > 0.5).astype(np.uint8)
    pred = (pred_mask_tensor.detach().cpu().numpy().squeeze() >= threshold).astype(np.uint8)
    overlay = img.copy()
    overlay[gt == 1] = [0, 255, 0]
    overlay[pred == 1] = [255, 0, 0]
    blend = cv2.addWeighted(img, 0.55, overlay, 0.45, 0)
    panel = np.concatenate([img, np.stack([gt * 255] * 3, axis=2), np.stack([pred * 255] * 3, axis=2), blend], axis=1)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), panel)
