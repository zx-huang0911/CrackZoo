"""Write local absolute-path configs after auditing disjoint dataset splits."""
from pathlib import Path
import argparse
import json
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crackzoo.data_audit import audit_dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["yolov7_wmf", "yolov4_dae"], required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frozen-detector", type=Path)
    a = p.parse_args()
    data = a.dataset_root.expanduser().resolve()
    result = audit_dataset(data)
    if not result["ok"]:
        raise SystemExit("Dataset audit failed: " + "; ".join(result["errors"][:8]))
    out = a.output.resolve()
    if out.exists():
        raise SystemExit("Config already exists; select a new --output.")
    if a.model == "yolov7_wmf":
        config = {f"{split}_{kind}": str(data / split / kind) for split in ("train", "val") for kind in ("images", "masks")}
        config.update(nc=1, names=["crack"])
    else:
        if not a.frozen_detector or not a.frozen_detector.is_file():
            p.error("DAE cache generation requires --frozen-detector pointing to a trained local checkpoint")
        config = yaml.safe_load((ROOT / "YOLOv4_DAE/configs/crack_v050_dae_rawmask.yaml").read_text())
        config["dataset"]["dataset_root"] = str(data)
        config["model"]["cfg_path"] = str(ROOT / "YOLOv4_DAE/cfg/yolov4-tiny.cfg")
        config["cache"]["cache_root"] = str(out.parent / "dae_cache")
        config["cache"]["frozen_basic_checkpoint"] = str(a.frozen_detector.resolve())
        for split in ("train", "val", "test"):
            labels = data / split / "labels_threshold_auto_v033"
            if not labels.is_dir():
                p.error(f"Missing threshold-channel detector labels: {labels}")
            config["splits"][split] = dict(image_root=str(data / split / "images"), mask_root=str(data / split / "masks"), label_root=str(labels))
        config["dae_train"]["save_dir"] = str(out.parent / "dae_train")
        config["integration_eval"]["save_dir"] = str(out.parent / "dae_eval")
        config["report"] = dict(docs_markdown=str(out.parent / "dae_report.md"), logs_markdown=str(out.parent / "dae_log.md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    out.with_suffix(".audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
