from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackzoo.registry import DEFAULTS, model_keys
from crackzoo.runners import build_suite, run_command, write_command_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or execute a multi-model CrackZoo benchmark suite.")
    parser.add_argument("--models", nargs="+", default=model_keys())
    parser.add_argument("--mode", default="train", choices=["train", "eval", "ablation"])
    parser.add_argument("--dataset-root", default=DEFAULTS["dataset"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    parser.add_argument("--seed", default=DEFAULTS["seed"])
    parser.add_argument("--epochs", default=DEFAULTS["epochs"])
    parser.add_argument("--total-itrs", default=DEFAULTS["total_itrs"])
    parser.add_argument("--batch-size", default=DEFAULTS["batch_size"])
    parser.add_argument("--val-batch-size", default=DEFAULTS["val_batch_size"])
    parser.add_argument("--manifest", default="reports/command_manifest.json")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and any(key.startswith("yolov") for key in args.models):
        raise SystemExit("Execute YOLO individually with run_experiment.py --config; this suite only previews default YOLO commands")
    commands = build_suite(
        args.models,
        args.mode,
        dataset=args.dataset_root,
        output=args.output,
        seed=args.seed,
        epochs=args.epochs,
        total_itrs=args.total_itrs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
    )
    write_command_manifest(args.manifest, commands)
    status = 0
    for command in commands:
        status = run_command(command, execute=args.execute) or status
        print("")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
