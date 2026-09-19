from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _list_images(path_or_txt: str) -> List[Path]:
    src = Path(path_or_txt)
    if src.is_file() and src.suffix.lower() == ".txt":
        lines = [x.strip() for x in src.read_text(encoding="utf-8").splitlines() if x.strip()]
        return [Path(x) for x in lines]
    if src.is_dir():
        files: List[Path] = []
        for ext in _IMG_EXTS:
            files.extend(Path(src).rglob(f"*{ext}"))
            files.extend(Path(src).rglob(f"*{ext.upper()}"))
        return sorted(files)
    raise FileNotFoundError(f"Image source not found: {path_or_txt}")


def _replace_dir_token(p: Path, old: str, new: str) -> Path:
    parts = list(p.parts)
    for i, token in enumerate(parts):
        if token == old:
            parts[i] = new
            return Path(*parts)
    return p


def _resolve_label_path(img_path: Path, label_root: Optional[str]) -> Path:
    if label_root:
        return Path(label_root) / f"{img_path.stem}.txt"
    a = _replace_dir_token(img_path, "images", "labels")
    if a != img_path:
        return a.with_suffix(".txt")
    return img_path.parent / "labels" / f"{img_path.stem}.txt"


def _resolve_mask_path(img_path: Path, mask_root: Optional[str]) -> Optional[Path]:
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    candidates: List[Path] = []
    if mask_root:
        for e in exts:
            candidates.append(Path(mask_root) / f"{img_path.stem}{e}")
    else:
        m = _replace_dir_token(img_path, "images", "masks")
        if m != img_path:
            for e in exts:
                candidates.append(m.with_suffix(e))
        else:
            for e in exts:
                candidates.append(img_path.parent / "masks" / f"{img_path.stem}{e}")
    for c in candidates:
        if c.exists():
            return c
    return None


def _parse_label_file(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.zeros((0, 6), dtype=np.float32)

    rows: List[List[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        vals = [float(x) for x in line.split()]
        if len(vals) < 5:
            continue
        cls_id, xc, yc, w, h = vals[:5]
        thr = vals[5] if len(vals) >= 6 else 0.5
        rows.append([cls_id, xc, yc, w, h, thr])

    if not rows:
        return np.zeros((0, 6), dtype=np.float32)

    arr = np.asarray(rows, dtype=np.float32)
    arr[:, 1:5] = np.clip(arr[:, 1:5], 0.0, 1.0)
    arr[:, 5] = np.clip(arr[:, 5], 0.0, 1.0)
    return arr


class CrackBasicDatasetV020(Dataset):
    """v0.2.0 basic-model dataset: image + bbox/class/threshold + optional mask."""

    def __init__(
        self,
        image_root: str,
        label_root: Optional[str] = None,
        mask_root: Optional[str] = None,
        image_size: int = 416,
        augment: bool = False,
        hflip_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_paths = _list_images(image_root)
        if not self.image_paths:
            raise RuntimeError(f"No images found for {image_root}")
        self.label_root = label_root
        self.mask_root = mask_root
        self.image_size = int(image_size)
        self.augment = augment
        self.hflip_prob = float(hflip_prob)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        img_path = self.image_paths[index]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)

        label_path = _resolve_label_path(img_path, self.label_root)
        labels = _parse_label_file(label_path)

        mask_tensor: Optional[torch.Tensor] = None
        mask_path = _resolve_mask_path(img_path, self.mask_root)
        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        if self.augment and np.random.rand() < self.hflip_prob:
            img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))
            if labels.shape[0] > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
            if mask_tensor is not None:
                mask_tensor = torch.flip(mask_tensor, dims=[2])

        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        targets = torch.zeros((labels.shape[0], 7), dtype=torch.float32)
        if labels.shape[0] > 0:
            targets[:, 1:6] = torch.from_numpy(labels[:, :5])
            targets[:, 6] = torch.from_numpy(labels[:, 5])

        # Keep ratio metadata for potential future scaling hooks.
        shapes = ((h0, w0), (self.image_size / max(h0, 1), self.image_size / max(w0, 1)))
        return img, targets, mask_tensor, str(img_path), shapes

    @staticmethod
    def collate_fn(batch):
        imgs, targets, masks, paths, shapes = zip(*batch)
        out_targets = []
        for i, t in enumerate(targets):
            if t.numel() == 0:
                continue
            t = t.clone()
            t[:, 0] = i
            out_targets.append(t)
        merged_targets = torch.cat(out_targets, 0) if out_targets else torch.zeros((0, 7), dtype=torch.float32)

        all_masks_present = all(m is not None for m in masks)
        if all_masks_present:
            merged_masks = torch.stack([m for m in masks], 0)
        else:
            merged_masks = None

        return torch.stack(imgs, 0), merged_targets, merged_masks, list(paths), list(shapes)


def create_crack_dataloader_v020(
    image_root: str,
    label_root: Optional[str],
    mask_root: Optional[str],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    workers: int,
) -> Tuple[DataLoader, CrackBasicDatasetV020]:
    ds = CrackBasicDatasetV020(
        image_root=image_root,
        label_root=label_root,
        mask_root=mask_root,
        image_size=image_size,
        augment=augment,
        hflip_prob=0.5 if augment else 0.0,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=CrackBasicDatasetV020.collate_fn,
    )
    return dl, ds
