"""Evaluate a compatible CrackResU-Net state dict on an explicit held-out split."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from crackresunet.dataset import CrackSegDataset
from crackresunet.model import CrackResUNet, CrackResUNetConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crackzoo.data_audit import indexed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--variant", default="main")
    a = p.parse_args()
    if a.image_size < 32 or a.image_size % 32:
        p.error("--image-size must be a positive multiple of 32")
    images = indexed(a.data_root / a.split / "images")
    masks = indexed(a.data_root / a.split / "masks")
    if images.keys() != masks.keys():
        p.error("Image/mask sample names differ")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output directory must be empty")
    model = CrackResUNet(CrackResUNetConfig(pretrained_encoder=False, variant=a.variant))
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.set_num_threads(2)
    dataset = CrackSegDataset(str(a.data_root), split=a.split, image_size=a.image_size)
    target = a.output / "predictions"
    target.mkdir(parents=True)
    with torch.inference_mode():
        for image, _, stems in DataLoader(dataset, batch_size=1, shuffle=False):
            logits = model(image)["main_logits"]
            stem = stems[0]
            with Image.open(masks[stem]) as mask:
                size = (mask.height, mask.width)
            logits = torch.nn.functional.interpolate(logits, size=size, mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].numpy().astype(np.uint8) * 255
            Image.fromarray(pred).save(target / (stem + ".png"))
    (a.output / "inference.json").write_text(json.dumps({"model": "crackresunet", "variant": a.variant,
        "split": a.split, "samples": len(dataset), "image_size": a.image_size, "output": "PNG binary masks at original dimensions",
        "checkpoint_sha256": __import__("hashlib").sha256(a.checkpoint.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(f"Wrote {len(dataset)} predictions to {target}; evaluate with scripts/evaluate_masks.py")


if __name__ == "__main__":
    main()
