from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CrackDAEDatasetV050(Dataset):
    """Offline cached dataset for v0.5.0 DAE training and evaluation."""

    def __init__(
        self,
        cache_root: str,
        split: str,
        image_size: int = 416,
        include_meta: bool = False,
    ) -> None:
        super().__init__()
        self.cache_root = Path(cache_root)
        self.split = split
        self.image_size = int(image_size)
        self.include_meta = bool(include_meta)

        self.split_dir = self.cache_root / split
        self.raw_dir = self.split_dir / "raw_masks"
        self.gt_dir = self.split_dir / "gt_masks"
        self.meta_dir = self.split_dir / "metadata"

        manifest = self.split_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest}")

        self.samples: List[Dict] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            self.samples.append(json.loads(line))

        if not self.samples:
            raise RuntimeError(f"No samples found in manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_mask(self, path: Path) -> torch.Tensor:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if arr.shape[0] != self.image_size or arr.shape[1] != self.image_size:
            arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        arr = (arr > 127).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, Optional[Dict]]:
        row = self.samples[idx]
        image_id = str(row["image_id"])

        raw_path = self.raw_dir / f"{image_id}.png"
        gt_path = self.gt_dir / f"{image_id}.png"

        raw = self._read_mask(raw_path)
        gt = self._read_mask(gt_path)

        meta: Optional[Dict] = None
        if self.include_meta:
            meta_path = self.meta_dir / f"{image_id}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = dict(row)
        return raw, gt, image_id, meta

    @staticmethod
    def collate_fn(batch):
        raws, gts, ids, metas = zip(*batch)
        return torch.stack(raws, dim=0), torch.stack(gts, dim=0), list(ids), list(metas)


def create_crack_dae_dataloader_v050(
    cache_root: str,
    split: str,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    workers: int,
    include_meta: bool = False,
) -> Tuple[DataLoader, CrackDAEDatasetV050]:
    ds = CrackDAEDatasetV050(
        cache_root=cache_root,
        split=split,
        image_size=image_size,
        include_meta=include_meta,
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=True,
        collate_fn=CrackDAEDatasetV050.collate_fn,
    )
    return dl, ds
