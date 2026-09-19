import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CrackSegmentation, build_crack_splits
from models import available_models, create_model
from utils import Denormalize, generate_grid_view, load_checkpoint, save_prediction_canvas, sigmoid_to_mask
from utils import ext_transforms as et


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="../data/private")
    parser.add_argument("--model", type=str, default="full_mcca", choices=available_models())
    parser.add_argument("--model_variant", type=str, default="", choices=["", "baseline_plain", "baseline_deepsup", "mcca_no_mscfm", "mcca_no_cam", "full_mcca"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--input_size", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--save_dir", type=str, default="outputs/prediction")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    return parser


def build_val_dataset(opts):
    _, val_files = build_crack_splits(opts.data_root, random_seed=opts.random_seed)
    if opts.max_samples > 0:
        val_files = val_files[: opts.max_samples]

    val_transform = et.ExtCompose([
        et.ExtResize((opts.input_size, opts.input_size)),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return CrackSegmentation(root=opts.data_root, transform=val_transform, file_list=val_files)


def main():
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default=None)
    initial_opts, _ = initial_parser.parse_known_args()

    if initial_opts.config is not None:
        with open(initial_opts.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    parser = get_argparser()
    parser.set_defaults(**cfg)
    opts = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    if opts.model_variant:
        model = create_model("mcca", input_size=opts.input_size, model_variant=opts.model_variant)
    else:
        model = create_model(opts.model, input_size=opts.input_size)
    model.to(device)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        load_checkpoint(opts.ckpt, model, continue_training=False, map_location="cpu")

    model.eval()

    if opts.dry_run:
        x = torch.randn(1, 3, opts.input_size, opts.input_size, device=device)
        with torch.no_grad():
            out = model(x)
        print(f"dry_run_pred_shape={tuple(out['final_logit'].shape)}")
        return

    dataset = build_val_dataset(opts)
    loader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    denorm = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    os.makedirs(opts.save_dir, exist_ok=True)

    with torch.no_grad():
        idx = 0
        for images, masks in tqdm(loader, total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            outputs = model(images)
            preds = sigmoid_to_mask(outputs["final_logit"], threshold=opts.threshold).squeeze(1)

            for b in range(images.size(0)):
                img_np = (denorm(images[b].cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                gt = masks[b].cpu().numpy().astype(np.uint8)
                pred = preds[b].cpu().numpy().astype(np.uint8)

                gt_rgb = dataset.decode_target(gt)
                pred_rgb = dataset.decode_target(pred)
                grid_gt_rgb = dataset.decode_target(generate_grid_view(masks[b], opts.grid_size))
                grid_pred_rgb = dataset.decode_target(generate_grid_view(preds[b], opts.grid_size))

                save_path = os.path.join(opts.save_dir, f"result_{idx}.png")
                save_prediction_canvas(save_path, img_np, gt_rgb, pred_rgb, grid_gt_rgb, grid_pred_rgb, opts.grid_size)
                idx += 1


if __name__ == "__main__":
    main()
