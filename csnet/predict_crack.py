
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import glob
import cv2

from torch.utils import data
from datasets import CrackSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='../data/private',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='crack', choices=['crack'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2, help="num classes (default: 2 for crack)")

    # Model Options
    parser.add_argument("--model", type=str, default='csnetv3plus_csnet_encoder',
                        help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Predict Options
    parser.add_argument("--ckpt", default='checkpoints/best_csnetv3plus_csnet_encoder_crack_os16.pth', type=str,
                        help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--grid_size", type=int, default=16, help="grid size for downsampling (default: 16)")
    parser.add_argument("--max_samples", type=int, default=0, help="limit val samples for quick check (0=all)")

    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'crack':
        root = opts.data_root
        img_dir = os.path.join(root, 'images')
        if not os.path.exists(img_dir):
            raise RuntimeError(f"Images directory not found: {img_dir}")

        all_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(img_dir, '*.*'))
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

        # Shuffle with seed to match training split
        random.seed(opts.random_seed)
        random.shuffle(all_files)

        # 85% train, 15% val
        split_idx = int(0.85 * len(all_files))
        val_files = all_files[split_idx:]
        if opts.max_samples > 0:
            val_files = val_files[:opts.max_samples]

        # Val transform
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_dst = CrackSegmentation(root=opts.data_root, image_set='val', transform=val_transform, file_list=val_files)
        return val_dst
    return None

def generate_grid_view(mask_tensor, grid_size=16):
    """
    Generate grid classification view.
    mask_tensor: (1, H, W) or (H, W) tensor, values 0 or 1.
    Returns: (H, W) numpy array with 0 or 1, blocky.
    """
    if mask_tensor.dim() == 2:
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    elif mask_tensor.dim() == 3:
        mask_tensor = mask_tensor.unsqueeze(0) # (1, 1, H, W)

    # MaxPool to find if block has crack
    # kernel_size=grid_size, stride=grid_size
    # We need to handle padding if H/W not divisible by grid_size, but usually we just crop or pad.
    # To keep it simple and aligned, we use ceil_mode=True or adapt.
    # But for visualization, let's just use max_pool2d

    # Ensure input is float for interpolation later, but max_pool works on float/int
    mask_tensor = mask_tensor.float()

    # Downsample: if any pixel in grid is 1, max_pool will be 1.
    grid_map = F.max_pool2d(mask_tensor, kernel_size=grid_size, stride=grid_size)

    # Upsample back to original size nearest neighbor
    H, W = mask_tensor.shape[2], mask_tensor.shape[3]
    grid_up = F.interpolate(grid_map, size=(H, W), mode='nearest')

    return grid_up.squeeze().cpu().numpy().astype(np.uint8)

def main():
    opts = get_argparser().parse_args()

    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataset
    val_dst = get_dataset(opts)
    val_loader = data.DataLoader(val_dst, batch_size=1, shuffle=False, num_workers=1)
    print("Dataset: %s, Val set: %d" % (opts.dataset, len(val_dst)))

    # Setup model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)

    # Load checkpoint
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
        # Handle DataParallel wrapping
        if 'model_state' in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        # If saved with DataParallel, keys have 'module.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print("Model restored from %s" % opts.ckpt)
    else:
        print("Error: Checkpoint not found at %s" % opts.ckpt)
        return

    model.to(device)
    model.eval()

    # Create results directory
    save_dir = 'results_prediction'
    os.makedirs(save_dir, exist_ok=True)

    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("Starting prediction...")

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(val_loader), total=len(val_dst)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            # Predict
            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1] # (B, H, W)

            # Process first image in batch (batch_size=1)
            img_tensor = images[0].cpu()
            target_tensor = labels[0].cpu() # (H, W)
            pred_tensor = preds[0].cpu()    # (H, W)

            # 1. Original Image
            img_np = (denorm(img_tensor.numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)

            # 2. GT Crack Segmentation (RGB)
            gt_rgb = val_dst.decode_target(target_tensor.numpy()).astype(np.uint8)

            # 3. Pred Crack Segmentation (RGB)
            pred_rgb = val_dst.decode_target(pred_tensor.numpy()).astype(np.uint8)

            # 4. Grid GT (Downsampled -> Upsampled)
            grid_gt_mask = generate_grid_view(target_tensor, grid_size=opts.grid_size) # (H, W) 0/1
            grid_gt_rgb = val_dst.decode_target(grid_gt_mask).astype(np.uint8)

            # 5. Grid Pred (Downsampled -> Upsampled)
            grid_pred_mask = generate_grid_view(pred_tensor, grid_size=opts.grid_size) # (H, W) 0/1
            grid_pred_rgb = val_dst.decode_target(grid_pred_mask).astype(np.uint8)

            # 6. Composite Layout
            # Layout: 2 rows.
            # Row 1: [Original, GT, Pred]
            # Row 2: [Grid GT, Grid Pred, Blank]

            H, W, C = img_np.shape

            # Create canvas
            canvas = np.zeros((H * 2, W * 3, 3), dtype=np.uint8)

            # Row 1
            canvas[0:H, 0:W, :] = img_np
            canvas[0:H, W:W*2, :] = gt_rgb
            canvas[0:H, W*2:W*3, :] = pred_rgb

            # Row 2
            canvas[H:H*2, 0:W, :] = grid_gt_rgb
            canvas[H:H*2, W:W*2, :] = grid_pred_rgb
            # Last slot empty (black)

            # Add text labels (Optional but helpful)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            color = (255, 0, 0) # Blue

            # Helper to put text
            def put_text(img, text, pos):
                cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

            # We write on the canvas directly
            # Row 1 Labels
            put_text(canvas, "Original", (10, 30))
            put_text(canvas, "Ground Truth", (W + 10, 30))
            put_text(canvas, "Prediction", (W*2 + 10, 30))

            # Row 2 Labels
            put_text(canvas, f"Grid GT (s={opts.grid_size})", (10, H + 30))
            put_text(canvas, f"Grid Pred (s={opts.grid_size})", (W + 10, H + 30))

            # Save single composite image
            save_path = os.path.join(save_dir, f'result_{i}.png')
            Image.fromarray(canvas).save(save_path)

            if i % 10 == 0:
                print(f"Processed {i}/{len(val_dst)}")

    print(f"Prediction complete. Results saved to {save_dir}")

if __name__ == '__main__':
    main()
