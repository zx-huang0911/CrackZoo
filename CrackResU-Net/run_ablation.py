from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run six CrackResU-Net ablation groups with same protocol")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_interval_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--protocol", type=str, default="formal_cfd_622")
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="crackresunet-repro")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--only_group", type=str, default="", help="Run only one config group name from configs/ablations.json")
    parser.add_argument("--print_commands_only", action="store_true", default=False, help="Print per-variant commands and exit")
    parser.add_argument("--use_tiny_random", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--run_prefix", type=str, default="v0.2.1_ablation")
    return parser.parse_args()


def _build_cmd(args: argparse.Namespace, run_name: str, variant: str, group_name: str) -> list[str]:
    cmd = [
        sys.executable,
        "train.py",
        "--run_name",
        run_name,
        "--variant",
        variant,
        "--protocol",
        args.protocol,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--val_batch_size",
        str(args.val_batch_size),
        "--val_interval_epochs",
        str(args.val_interval_epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--image_size",
        str(args.image_size),
        "--num_workers",
        str(args.num_workers),
        "--random_seed",
        str(args.random_seed),
        "--use_wandb",
        str(args.use_wandb),
        "--wandb_project",
        args.wandb_project,
        "--wandb_group",
        "formal-ablation",
        "--wandb_job_type",
        "train",
        "--wandb_tags",
        f"formal,ablation,{group_name},seed{args.random_seed},epoch{args.epochs},protocol-{args.protocol}",
        "--wandb_mode",
        args.wandb_mode,
    ]

    if args.data_root:
        cmd.extend(["--data_root", args.data_root])
    if args.wandb_entity:
        cmd.extend(["--wandb_entity", args.wandb_entity])
    if args.use_tiny_random:
        cmd.append("--use_tiny_random")
    if args.cpu:
        cmd.append("--cpu")
    return cmd


def main() -> None:
    args = parse_args()
    cfg_path = Path("configs") / "ablations.json"
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    if args.only_group:
        if args.only_group not in config:
            raise ValueError(f"Unknown --only_group '{args.only_group}'. Available: {', '.join(config.keys())}")
        config = {args.only_group: config[args.only_group]}

    results = []
    for name, group in config.items():
        run_name = f"{args.run_prefix}_{name}"
        cmd = _build_cmd(args, run_name=run_name, variant=group["variant"], group_name=name)

        print("RUN:", " ".join(cmd))
        if args.print_commands_only:
            continue
        subprocess.run(cmd, check=True)

        best_json = Path("logs") / run_name / "best_metrics.json"
        if best_json.exists():
            payload = json.loads(best_json.read_text(encoding="utf-8"))
            payload["run_name"] = run_name
            payload["group_name"] = name
            payload["variant"] = group["variant"]
            results.append(payload)

    if args.print_commands_only:
        return

    summary_dir = Path("logs") / args.run_prefix
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "ablation_summary.md"
    summary_text = "# Formal Ablation Pilot Summary\n\n"
    summary_text += "Protocol:\n"
    summary_text += f"- protocol: {args.protocol}\n"
    summary_text += f"- epochs: {args.epochs}\n"
    summary_text += f"- batch_size: {args.batch_size}\n"
    summary_text += f"- val_batch_size: {args.val_batch_size}\n"
    summary_text += f"- lr: {args.lr}\n"
    summary_text += f"- weight_decay: {args.weight_decay}\n"
    summary_text += f"- seed: {args.random_seed}\n\n"

    if results:
        by_f1 = sorted(results, key=lambda r: float(r.get("F1", 0.0)), reverse=True)
        by_dice = sorted(results, key=lambda r: float(r.get("Dice", 0.0)), reverse=True)

        summary_text += "## Best Metrics Table\n\n"
        summary_text += "| group | variant | best_epoch | F1 | Dice | IoU | TolF1 | ValLoss |\n"
        summary_text += "|---|---|---:|---:|---:|---:|---:|---:|\n"
        for r in results:
            summary_text += (
                f"| {r.get('group_name', '')} | {r.get('variant', '')} | {int(r.get('best_epoch', 0))} "
                f"| {float(r.get('F1', 0.0)):.6f} | {float(r.get('Dice', 0.0)):.6f} "
                f"| {float(r.get('IoU', 0.0)):.6f} | {float(r.get('TolF1', 0.0)):.6f} "
                f"| {float(r.get('ValLoss', 0.0)):.6f} |\n"
            )

        summary_text += "\n## Ranking by F1\n\n"
        for i, r in enumerate(by_f1, start=1):
            summary_text += f"{i}. {r.get('group_name')} ({float(r.get('F1', 0.0)):.6f})\n"

        summary_text += "\n## Ranking by Dice\n\n"
        for i, r in enumerate(by_dice, start=1):
            summary_text += f"{i}. {r.get('group_name')} ({float(r.get('Dice', 0.0)):.6f})\n"

    summary_text += "\n## Run Folders\n\n"
    for name in config.keys():
        summary_text += f"- logs/{args.run_prefix}_{name}\n"
    summary_path.write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
