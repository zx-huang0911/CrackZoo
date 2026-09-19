import os
import torch.utils.data as data
from PIL import Image
import numpy as np

class CrackSegmentation(data.Dataset):
    def __init__(self, root, image_set='train', transform=None, file_list=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.image_set = image_set

        # Determine paths based on observed structure
        # c:\...\crack_segmentation_dataset\images
        # c:\...\crack_segmentation_dataset\masks OR c:\...\crack_segmentation_dataset\train\masks

        self.images_dir = os.path.join(self.root, 'images')

        # Try to find masks dir
        # Priority 1: direct 'masks' folder
        # Priority 2: 'train/masks' folder
        possible_mask_dirs = [
            os.path.join(self.root, 'masks'),
            os.path.join(self.root, 'train', 'masks'),
            os.path.join(self.root, 'test', 'masks') # Just in case
        ]

        self.masks_dir = None
        for d in possible_mask_dirs:
            if os.path.exists(d):
                self.masks_dir = d
                break

        if self.masks_dir is None:
             raise RuntimeError(f"Masks directory not found in {self.root}. Searched: {possible_mask_dirs}")

        if not os.path.isdir(self.images_dir):
             raise RuntimeError(f"Images directory not found in {self.root}")

        self.images = []
        self.masks = []

        # List images
        if not os.path.exists(self.images_dir):
             print(f"Warning: {self.images_dir} does not exist")
             return

        if file_list is None:
            file_list = [f for f in os.listdir(self.images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for file_name in file_list:
            img_path = os.path.join(self.images_dir, file_name)

            # Construct mask path. Assumption: same filename (extensions might differ)
            # Try same extension first
            mask_path = os.path.join(self.masks_dir, file_name)

            if not os.path.exists(mask_path):
                 # Try replacing extension
                 name, _ = os.path.splitext(file_name)
                 found = False
                 for ext in ['.png', '.jpg', '.jpeg']:
                     mp = os.path.join(self.masks_dir, name + ext)
                     if os.path.exists(mp):
                         mask_path = mp
                         found = True
                         break
                 if not found:
                     # Skip if no mask found
                     continue

            self.images.append(img_path)
            self.masks.append(mask_path)

        print(f"Loaded {len(self.images)} pairs from {self.root} ({image_set})")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('L')

        # Ensure binary mask (0/1)
        # Threshold at 128 (assuming 0=background, 255=crack)
        threshold = 0 if target.getextrema()[1] <= 1 else 127
        target = target.point(lambda p: 1 if p > threshold else 0)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        # mask shape: (H, W)
        # return shape: (H, W, 3)
        cmap = np.zeros((256, 3), dtype='uint8')
        cmap[0] = [0, 0, 0]      # Background: Black
        cmap[1] = [255, 255, 255] # Crack: White

        return cmap[mask]
