from __future__ import annotations

import argparse
import csv
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackzoo.data_audit import indexed
from crackzoo.metrics import aggregate, binarize, binary_metrics, boundary_f1, cldice, grid_metrics


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted binary masks against ground-truth masks.")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-csv", default="reports/mask_metrics.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--grid-positive-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    rows = []
    gt_files = indexed(gt_dir)
    pred_files = indexed(pred_dir)
    if gt_files.keys() != pred_files.keys():
        raise SystemExit(f"Mask pairing mismatch: missing={sorted(gt_files.keys()-pred_files.keys())}, extra={sorted(pred_files.keys()-gt_files.keys())}")
    inputs = []
    for stem, gt_path in sorted(gt_files.items()):
        pred_path = pred_files[stem]
        gt = binarize(read_mask(gt_path), threshold=0.5)
        pred = binarize(read_mask(pred_path), threshold=args.threshold)
        inputs.append({"sample": gt_path.name, "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(), "pred_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest()})
        row = {"sample": gt_path.name}
        row.update(binary_metrics(gt, pred, threshold=args.threshold))
        row.update(grid_metrics(gt, pred, args.grid_size, args.grid_positive_threshold))
        row["tolerance_f1"] = boundary_f1(gt, pred)
        row["cldice"] = cldice(gt, pred)
        rows.append(row)
    if not rows:
        raise SystemExit("No matching masks found.")
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {"sample": "__mean__", **aggregate(rows)}
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({k: summary.get(k, "") for k in fieldnames})
    out_csv.with_suffix(".json").write_text(json.dumps({
        "samples": len(rows), "aggregation": "per-image macro mean", "threshold": args.threshold,
        "grid_size": args.grid_size, "grid_positive_threshold": args.grid_positive_threshold,
        "tolerance_px": 2, "tolerance_geometry": "square (Chebyshev); foreground tolerance, not extracted boundaries",
        "empty_foreground_overlap": 0, "empty_foreground_tolerance_f1": 1,
        "cldice_available": bool(np.isfinite(rows[0]["cldice"])), "inputs": inputs,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
