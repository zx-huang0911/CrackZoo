from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import yaml
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dae_dataset_v050 import create_crack_dae_dataloader_v050
from models.crack_pipeline_v010 import CrackDAE
from train_crack_dae_v050 import _validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate standalone DAE on offline cache (v0.5.0)")
    parser.add_argument("--config", type=str, default="configs/crack_v050_dae_rawmask.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--max-vis", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tcfg = cfg["dae_train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint.strip() or str(Path(tcfg["save_dir"]) / "weights" / "best_clean_iou.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    model = CrackDAE().to(device)
    model.load_state_dict(ckpt["model"], strict=True)

    loader, ds = create_crack_dae_dataloader_v050(
        cache_root=str(cfg["cache"]["cache_root"]),
        split=str(args.split),
        image_size=int(cfg["model"]["image_size"]),
        batch_size=int(tcfg["batch_size"]),
        shuffle=False,
        workers=int(tcfg.get("workers", 0)),
        include_meta=False,
    )

    bce_w = float(tcfg.get("bce_weight", 1.0))
    dice_w = float(tcfg.get("dice_weight", 1.0))

    save_dir = Path(tcfg["save_dir"]) / f"val_{args.split}"
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, float] = _validate(
        model=model,
        val_loader=loader,
        device=device,
        bce_w=bce_w,
        dice_w=dice_w,
        vis_dir=save_dir / "panels",
        max_vis=int(args.max_vis),
    )

    out = {
        "version": "v0.5.0-local-dae-rawmask",
        "config": args.config,
        "split": args.split,
        "checkpoint": ckpt_path,
        "samples": len(ds),
        "metrics": metrics,
    }

    out_path = save_dir / f"summary_{args.split}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved_summary={out_path}")


if __name__ == "__main__":
    main()
