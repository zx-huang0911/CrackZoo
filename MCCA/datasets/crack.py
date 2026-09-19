import glob
import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CrackSegmentation(Dataset):
    def __init__(self, root: str, transform=None, file_list: Optional[Sequence[str]] = None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.images_dir = os.path.join(self.root, "images")
        self.masks_dir = self._find_masks_dir(self.root)

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise RuntimeError(f"Masks directory not found: {self.masks_dir}")

        if file_list is None:
            file_list = [
                os.path.basename(f)
                for f in glob.glob(os.path.join(self.images_dir, "*.*"))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        self.images: List[str] = []
        self.masks: List[str] = []

        for file_name in sorted(file_list):
            img_path = os.path.join(self.images_dir, file_name)
            mask_path = self._find_mask_path(file_name)
            if mask_path is None:
                continue
            self.images.append(img_path)
            self.masks.append(mask_path)

        if len(self.images) == 0:
            raise RuntimeError(f"No image-mask pairs found under {self.root}")

    @staticmethod
    def _find_masks_dir(root: str) -> str:
        candidates = [
            os.path.join(root, "masks"),
            os.path.join(root, "train", "masks"),
            os.path.join(root, "test", "masks"),
        ]
        for d in candidates:
            if os.path.isdir(d):
                return d
        return candidates[0]

    def _find_mask_path(self, file_name: str) -> Optional[str]:
        same_name = os.path.join(self.masks_dir, file_name)
        if os.path.isfile(same_name):
            return same_name
        name, _ = os.path.splitext(file_name)
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(self.masks_dir, name + ext)
            if os.path.isfile(p):
                return p
        return None

    def __getitem__(self, index: int):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        threshold = 0 if mask.getextrema()[1] <= 1 else 127
        mask = mask.point(lambda p: 1 if p > threshold else 0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask

    def __len__(self) -> int:
        return len(self.images)

    @classmethod
    def decode_target(cls, mask: np.ndarray) -> np.ndarray:
        cmap = np.zeros((256, 3), dtype=np.uint8)
        cmap[0] = [0, 0, 0]
        cmap[1] = [255, 255, 255]
        return cmap[mask]


def build_crack_splits(data_root: str, random_seed: int = 1, train_ratio: float = 0.85) -> Tuple[List[str], List[str]]:
    img_dir = os.path.join(data_root, "images")
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"Images directory not found: {img_dir}")

    all_files = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(img_dir, "*.*"))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    all_files = sorted(all_files)

    rng = random.Random(random_seed)
    rng.shuffle(all_files)

    split_idx = int(train_ratio * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    return train_files, val_files
