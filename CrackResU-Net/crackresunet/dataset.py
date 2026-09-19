from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class CrackSegDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 320,
        max_samples: int = 0,
        random_seed: int = 1,
        train_ratio: float = 0.85,
        manifest_path: str = "",
        use_formal_augs: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.use_formal_augs = use_formal_augs

        if manifest_path:
            self.samples = self._load_samples_from_manifest(self.data_root, Path(manifest_path))
        else:
            self.samples = self._collect_samples(
                split=split,
                random_seed=random_seed,
                train_ratio=train_ratio,
            )

        if max_samples > 0:
            self.samples = self.samples[:max_samples]

        self.img_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _image_candidates(directory: Path) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return sorted([p for p in directory.glob("*") if p.suffix.lower() in exts])

    @staticmethod
    def _find_mask(mask_dir: Path, stem: str) -> Path | None:
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            p = mask_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    @staticmethod
    def _load_samples_from_manifest(data_root: Path, manifest_path: Path) -> List[Tuple[Path, Path]]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        pairs: List[Tuple[Path, Path]] = []
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            s = line.strip()
            if not s:
                continue
            parts = s.split("\t")
            if len(parts) < 2:
                raise ValueError(f"Invalid manifest line: {s}")
            img_rel, mask_rel = parts[0], parts[1]
            img_path = (data_root / img_rel).resolve()
            mask_path = (data_root / mask_rel).resolve()
            if not img_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Manifest pair missing: {img_path} | {mask_path}")
            pairs.append((img_path, mask_path))

        return pairs

    def _paired_samples_from_dirs(self, img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
        image_paths = self._image_candidates(img_dir)
        pairs: List[Tuple[Path, Path]] = []
        for img_path in image_paths:
            mask_path = self._find_mask(mask_dir, img_path.stem)
            if mask_path is not None:
                pairs.append((img_path, mask_path))
        return pairs

    def _collect_samples(self, split: str, random_seed: int, train_ratio: float) -> List[Tuple[Path, Path]]:
        # Mode A: split folders: <root>/train/images, <root>/val/images ...
        split_img_dir = self.data_root / split / "images"
        split_mask_dir = self.data_root / split / "masks"
        if split_img_dir.exists() and split_mask_dir.exists():
            return self._paired_samples_from_dirs(split_img_dir, split_mask_dir)

        # Mode B: flat folders: <root>/images and <root>/masks, then deterministic split.
        flat_img_dir = self.data_root / "images"
        flat_mask_dir = self.data_root / "masks"
        if not flat_img_dir.exists() or not flat_mask_dir.exists():
            raise FileNotFoundError(
                "Expected either split folders '<root>/<split>/images,masks' or flat folders '<root>/images,masks'."
            )

        all_pairs = self._paired_samples_from_dirs(flat_img_dir, flat_mask_dir)
        rng = random.Random(random_seed)
        rng.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * train_ratio)

        if split == "train":
            return all_pairs[:split_idx]
        if split == "val":
            return all_pairs[split_idx:]
        return all_pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.split == "train" and self.use_formal_augs:
            # Formal paper-aligned augmentations: color jitter + random translation.
            image = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05)(image)

            max_dx = int(0.1 * image.width)
            max_dy = int(0.1 * image.height)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            image = TF.affine(
                image,
                angle=0.0,
                translate=[dx, dy],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=0.0,
                translate=[dx, dy],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_t = self.img_tf(image)

        if self.split == "train" and self.use_formal_augs:
            # Gaussian noise in tensor space, clipped to valid range.
            noise = torch.randn_like(image_t) * 0.03
            image_t = torch.clamp(image_t + noise, -3.0, 3.0)

        mask_np = np.array(mask)
        mask_np = (mask_np > (0 if mask_np.max() <= 1 else 127)).astype(np.int64)
        mask_t = torch.from_numpy(mask_np)
        return image_t, mask_t, img_path.stem


class TinyRandomDataset(Dataset):
    def __init__(self, n: int = 8, image_size: int = 320) -> None:
        self.n = n
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image = torch.rand(3, self.image_size, self.image_size)
        mask = torch.randint(0, 2, size=(self.image_size, self.image_size), dtype=torch.long)
        return image, mask, f"rand_{index:04d}"
