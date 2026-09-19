"""Generate artificial line masks and inspect the evaluation pipeline (no trained model)."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "reports" / "demo")
    a = p.parse_args()
    out = a.output.resolve()
    # Never mix a new demonstration with an old or user-provided dataset.
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Output directory must be empty; choose a new --output.")
    rng = np.random.default_rng(17)
    for split in ("train", "val", "test"):
        for folder in ("images", "masks"):
            (out / "dataset" / split / folder).mkdir(parents=True, exist_ok=True)
        for i in range(4):
            mask = Image.new("L", (128, 128))
            draw = ImageDraw.Draw(mask)
            points = [(int(rng.integers(20, 108)), y) for y in range(8, 125, 16)]
            draw.line(points, fill=255, width=3)
            pixels = np.clip(rng.normal(175, 10, (128, 128)), 0, 255).astype("uint8")
            pixels[np.asarray(mask) > 0] = 40
            name = f"synthetic_{split}_{i:02d}.png"
            Image.fromarray(pixels).convert("RGB").save(out / "dataset" / split / "images" / name)
            mask.save(out / "dataset" / split / "masks" / name)
            if split == "test":
                (out / "predictions").mkdir(exist_ok=True)
                # Deliberate false positives and misses, not model inference.
                pred = np.asarray(mask).copy()
                pred[40:56] = 0
                pred[80:83, 10:70] = 255
                Image.fromarray(pred).save(out / "predictions" / name)
                if i == 0:
                    panel = Image.new("RGB", (128 * 3, 152), "white")
                    for col, (label, im) in enumerate((("Synthetic input", Image.fromarray(pixels)), ("Ground truth", mask), ("Perturbed mask", Image.fromarray(pred)))):
                        panel.paste(im.convert("RGB"), (col * 128, 24))
                        ImageDraw.Draw(panel).text((col * 128 + 4, 6), label, fill="black")
                    panel.save(out / "synthetic_example.png")
    subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_masks.py"), "--gt-dir", str(out / "dataset/test/masks"), "--pred-dir", str(out / "predictions"), "--out-csv", str(out / "metrics.csv")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_dataset.py"), "--root", str(out / "dataset"), "--output", str(out / "dataset_audit.json")], check=True)
    (out / "README.txt").write_text("Generated from scratch with seed 17. Artificial line masks and deliberately perturbed labels; no private data, no model accuracy results.\n")


if __name__ == "__main__":
    main()
