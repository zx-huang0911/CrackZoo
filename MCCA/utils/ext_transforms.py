import random
from collections.abc import Iterable
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


class ExtCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ExtRandomRotation:
    def __init__(self, degrees: float):
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees

    def __call__(self, image, mask):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return F.rotate(image, angle), F.rotate(mask, angle)


class ExtRandomScale:
    def __init__(self, scale_range):
        self.scale_range = scale_range

    def __call__(self, image, mask):
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        h, w = image.size[1], image.size[0]
        target_size = (int(h * scale), int(w * scale))
        return F.resize(image, target_size, Image.BILINEAR), F.resize(mask, target_size, Image.NEAREST)


class ExtRandomCrop:
    def __init__(self, size, pad_if_needed=False):
        self.size = (size, size) if isinstance(size, int) else size
        self.pad_if_needed = pad_if_needed

    def __call__(self, image, mask):
        if self.pad_if_needed:
            pad_w = max(0, self.size[1] - image.size[0])
            pad_h = max(0, self.size[0] - image.size[1])
            if pad_w > 0 or pad_h > 0:
                left = pad_w // 2
                right = pad_w - left
                top = pad_h // 2
                bottom = pad_h - top
                image = F.pad(image, (left, top, right, bottom))
                mask = F.pad(mask, (left, top, right, bottom))

        i, j, h, w = self._get_params(image)
        return F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

    def _get_params(self, image):
        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw


class ExtRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(mask)
        return image, mask


class ExtRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            return F.vflip(image), F.vflip(mask)
        return image, mask


class ExtResize:
    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, image, mask):
        return F.resize(image, self.size, Image.BILINEAR), F.resize(mask, self.size, Image.NEAREST)


class ExtCenterCrop:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        return F.center_crop(image, self.size), F.center_crop(mask, self.size)


class ExtToTensor:
    def __call__(self, image, mask):
        image = F.to_tensor(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.uint8))
        return image, mask


class ExtNormalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        return F.normalize(image, self.mean, self.std), mask
