from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackzoo.registry import command_for, model_keys
from crackzoo.runners import run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or execute a unified CrackZoo model command.")
    parser.add_argument("--model", required=True, choices=model_keys())
    parser.add_argument("--mode", default="train", choices=["train", "eval", "ablation"])
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--epochs", default=None)
    parser.add_argument("--total-itrs", default=None)
    parser.add_argument("--batch-size", default=None)
    parser.add_argument("--val-batch-size", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--execute", action="store_true", help="Run the command. By default only prints it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments appended to the generated command.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.execute and args.model.startswith("yolov") and not args.config:
        raise SystemExit("YOLO execution needs an explicit --config; generate it with scripts/prepare_yolo_config.py")
    if args.dataset_root and args.model.startswith("yolov"):
        raise SystemExit("For YOLO, set the dataset via scripts/prepare_yolo_config.py and pass --config")
    spec = command_for(
        args.model,
        args.mode,
        dataset=args.dataset_root,
        output=args.output,
        seed=args.seed,
        epochs=args.epochs,
        total_itrs=args.total_itrs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        **{k:v for k,v in {"checkpoint":args.checkpoint,"config":args.config}.items() if v is not None},
    )
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if args.extra:
        spec = type(spec)(cwd=spec.cwd, argv=[*spec.argv, *args.extra], env_name=spec.env_name, note=spec.note)
    return run_command(spec, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
