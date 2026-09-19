"""Validate explicit splits without redistributing or rewriting source data."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def indexed(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Missing directory: {directory}")
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate sample stem: {path.stem}")
        result[path.stem] = path
    if not result:
        raise ValueError(f"No images in {directory}")
    return result


def audit_dataset(root, splits=("train", "val", "test")):
    root = Path(root)
    rows, seen_pixels, seen_stems = [], {}, {}
    errors = []
    for split in splits:
        images, masks = indexed(root / split / "images"), indexed(root / split / "masks")
        if images.keys() != masks.keys():
            raise ValueError(f"{split}: image/mask pairing mismatch")
        for stem, path in images.items():
            with Image.open(path) as im:
                pixels = np.asarray(im.convert("RGB"))
            with Image.open(masks[stem]) as im:
                mask = np.asarray(im.convert("L"))
            if pixels.shape[:2] != mask.shape:
                errors.append(f"{split}/{stem}: image/mask dimensions differ")
            values = set(np.unique(mask).tolist())
            if not (values <= {0, 1} or values <= {0, 255}):
                errors.append(f"{split}/{stem}: mask is not binary")
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            for seen, key, label in ((seen_pixels, digest, "decoded image"), (seen_stems, stem, "sample stem")):
                if key in seen and seen[key] != split:
                    errors.append(f"{split}/{stem}: {label} overlaps {seen[key]}")
                seen[key] = split
            rows.append(dict(split=split, sample=stem, image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             mask_sha256=hashlib.sha256(masks[stem].read_bytes()).hexdigest()))
    return dict(ok=not errors, errors=errors, samples=rows,
                limitation="Exact decoded-image and name checks; not a near-duplicate or source-scene leakage detector")
