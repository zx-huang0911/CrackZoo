from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from crackresunet.dataset import CrackSegDataset, TinyRandomDataset
from crackresunet.engine import TrainConfig, fit
from crackresunet.model import CrackResUNet, CrackResUNetConfig
from crackresunet.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CrackResU-Net")
    parser.add_argument("--data_root", type=str, default="", help="Dataset root")
    parser.add_argument("--run_name", type=str, default="sample_run")
    parser.add_argument(
        "--variant",
        type=str,
        default="main",
        choices=[
            "main",
            "no_sa",
            "no_aux",
            "no_pram",
            "all_pram",
            "all_sa",
            "without_sa",
            "without_aux",
            "without_pram",
        ],
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--val_interval_epochs", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--aux_weight", type=float, default=1.0)
    parser.add_argument("--save_top_k_vis", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="", help="Optional checkpoint for shape-matched warm start")

    # Protocol freeze / formal run settings.
    parser.add_argument("--protocol", type=str, default="auto", choices=["auto", "formal_cfd_622"])
    parser.add_argument("--tolerance_px", type=int, default=2)
    parser.add_argument("--formal_augs", type=int, default=1)

    # v0.1.2 tiny overfit workflow (kept for compatibility, no longer primary stage).
    parser.add_argument("--tiny_overfit", type=int, default=0, help="Enable deterministic tiny real-data overfit workflow")
    parser.add_argument("--tiny_overfit_samples", type=int, default=16, help="Number of samples for tiny overfit mode")
    parser.add_argument("--tiny_overfit_use_same_val", type=int, default=1, help="Use same tiny subset for val when tiny overfit is enabled")
    parser.add_argument("--tiny_overfit_epochs", type=int, default=50, help="Epochs to use in tiny overfit mode")

    # Weights & Biases.
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="crackresunet-repro")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_group", type=str, default="formal-main")
    parser.add_argument("--wandb_job_type", type=str, default="train")
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--use_tiny_random", action="store_true", default=False)
    return parser.parse_args()


def _normalize_variant(variant: str) -> str:
    alias = {
        "without_sa": "no_sa",
        "without_aux": "no_aux",
        "without_pram": "no_pram",
    }
    return alias.get(variant, variant)


def _stems_from_dataset(ds: CrackSegDataset | TinyRandomDataset) -> List[str]:
    if isinstance(ds, CrackSegDataset):
        return [img_path.stem for img_path, _ in ds.samples]
    return [f"rand_{i:04d}" for i in range(len(ds))]


def _image_candidates(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in directory.glob("*") if p.suffix.lower() in exts])


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _collect_pairs_from_dirs(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    image_paths = _image_candidates(img_dir)
    pairs: List[Tuple[Path, Path]] = []
    for img_path in image_paths:
        mask_path = _find_mask(mask_dir, img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))
    return pairs


def _collect_all_pairs_for_formal(data_root: Path) -> List[Tuple[Path, Path]]:
    # Prefer CFD train/test folders when available.
    train_img = data_root / "train" / "images"
    train_mask = data_root / "train" / "masks"
    test_img = data_root / "test" / "images"
    test_mask = data_root / "test" / "masks"

    all_pairs: List[Tuple[Path, Path]] = []
    if train_img.exists() and train_mask.exists():
        all_pairs.extend(_collect_pairs_from_dirs(train_img, train_mask))
    if test_img.exists() and test_mask.exists():
        all_pairs.extend(_collect_pairs_from_dirs(test_img, test_mask))

    if all_pairs:
        return sorted(all_pairs, key=lambda x: x[0].name)

    # Fallback to flat root/images + root/masks.
    flat_img = data_root / "images"
    flat_mask = data_root / "masks"
    if flat_img.exists() and flat_mask.exists():
        return _collect_pairs_from_dirs(flat_img, flat_mask)

    raise FileNotFoundError("Cannot collect samples for formal protocol from train/test or flat layout.")


def _write_manifest(path: Path, data_root: Path, pairs: List[Tuple[Path, Path]]) -> None:
    ensure_dir(path.parent)
    lines: List[str] = []
    for img, mask in pairs:
        img_rel = img.resolve().relative_to(data_root.resolve()).as_posix()
        mask_rel = mask.resolve().relative_to(data_root.resolve()).as_posix()
        lines.append(f"{img_rel}\t{mask_rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_formal_manifests(args: argparse.Namespace, out_dir: Path) -> Dict[str, Path]:
    data_root = Path(args.data_root)
    all_pairs = _collect_all_pairs_for_formal(data_root)

    rng = random.Random(args.random_seed)
    rng.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train : n_train + n_val]
    test_pairs = all_pairs[n_train + n_val :]

    proto_dir = Path("logs") / "protocols" / f"formal_cfd_622_seed{args.random_seed}"
    ensure_dir(proto_dir)

    train_manifest = proto_dir / "train_manifest.txt"
    val_manifest = proto_dir / "val_manifest.txt"
    test_manifest = proto_dir / "test_manifest.txt"

    _write_manifest(train_manifest, data_root, train_pairs)
    _write_manifest(val_manifest, data_root, val_pairs)
    _write_manifest(test_manifest, data_root, test_pairs)

    # Copy frozen manifests into run folder for full reproducibility.
    ensure_dir(out_dir)
    (out_dir / "train_manifest.txt").write_text(train_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "val_manifest.txt").write_text(val_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    (out_dir / "test_manifest.txt").write_text(test_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "train": train_manifest,
        "val": val_manifest,
        "test": test_manifest,
    }


def make_dataloaders(args: argparse.Namespace, manifests: Dict[str, Path] | None = None):
    selected_train_stems: List[str] = []
    selected_val_stems: List[str] = []

    if args.use_tiny_random:
        train_ds = TinyRandomDataset(n=max(4, args.batch_size * 2), image_size=args.image_size)
        val_ds = TinyRandomDataset(n=max(4, args.val_batch_size * 2), image_size=args.image_size)
    else:
        if not args.data_root:
            raise ValueError("--data_root is required when not using --use_tiny_random")

        if manifests is not None:
            train_ds = CrackSegDataset(
                data_root=args.data_root,
                split="train",
                image_size=args.image_size,
                max_samples=args.max_train_samples,
                random_seed=args.random_seed,
                manifest_path=str(manifests["train"]),
                use_formal_augs=bool(args.formal_augs),
            )
            val_ds = CrackSegDataset(
                data_root=args.data_root,
                split="val",
                image_size=args.image_size,
                max_samples=args.max_val_samples,
                random_seed=args.random_seed,
                manifest_path=str(manifests["val"]),
                use_formal_augs=False,
            )
        else:
            train_ds = CrackSegDataset(
                data_root=args.data_root,
                split="train",
                image_size=args.image_size,
                max_samples=args.max_train_samples,
                random_seed=args.random_seed,
            )

            if args.tiny_overfit:
                n = max(1, int(args.tiny_overfit_samples))
                train_ds.samples = train_ds.samples[:n]
                if len(train_ds.samples) == 0:
                    raise RuntimeError("No train samples found for tiny overfit selection.")

                if args.tiny_overfit_use_same_val:
                    val_ds = CrackSegDataset(
                        data_root=args.data_root,
                        split="train",
                        image_size=args.image_size,
                        max_samples=0,
                        random_seed=args.random_seed,
                    )
                    val_ds.samples = list(train_ds.samples)
                else:
                    val_ds = CrackSegDataset(
                        data_root=args.data_root,
                        split="val",
                        image_size=args.image_size,
                        max_samples=0,
                        random_seed=args.random_seed,
                    )
                    val_ds.samples = val_ds.samples[:n]
            else:
                val_ds = CrackSegDataset(
                    data_root=args.data_root,
                    split="val",
                    image_size=args.image_size,
                    max_samples=args.max_val_samples,
                    random_seed=args.random_seed,
                )

    if len(train_ds) == 0:
        raise RuntimeError("Train dataset is empty after loading/splitting. Please check data_root and masks.")
    if len(val_ds) == 0:
        raise RuntimeError("Val dataset is empty after loading/splitting. Please check data_root and masks.")

    selected_train_stems = _stems_from_dataset(train_ds)
    selected_val_stems = _stems_from_dataset(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, selected_train_stems, selected_val_stems


def _write_lines(path: Path, lines: List[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_shape_matched_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> int:
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    if not isinstance(state, dict):
        state = state.state_dict()
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    current = model.state_dict()
    loadable = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    current.update(loadable)
    model.load_state_dict(current, strict=False)
    return len(loadable)


def write_overfit_summary(out_dir: Path, args: argparse.Namespace, best: Dict[str, float]) -> None:
    val_rows = _read_csv_rows(out_dir / "metrics_epoch.csv")

    first_f1 = float(val_rows[0]["F1"]) if val_rows else 0.0
    last_f1 = float(val_rows[-1]["F1"]) if val_rows else 0.0
    best_f1 = float(best.get("F1", 0.0))
    best_epoch = int(best.get("best_epoch", 0)) if best else 0

    overfit_signal = "YES" if best_f1 > max(0.9, first_f1 + 0.3) else "PARTIAL"
    trend_hint = "improving" if last_f1 >= first_f1 else "unstable"

    text = (
        "# Tiny Overfit Summary\n\n"
        f"- run_name: {args.run_name}\n"
        f"- variant: {args.variant}\n"
        f"- tiny_overfit: {args.tiny_overfit}\n"
        f"- tiny_overfit_samples: {args.tiny_overfit_samples}\n"
        f"- tiny_overfit_use_same_val: {args.tiny_overfit_use_same_val}\n"
        f"- epochs: {args.epochs}\n"
        f"- batch_size: {args.batch_size}\n"
        f"- val_batch_size: {args.val_batch_size}\n"
        f"- val_interval_epochs: {args.val_interval_epochs}\n"
        f"- random_seed: {args.random_seed}\n"
        "\n"
        "## Metrics Snapshot\n"
        f"- first_epoch_f1: {first_f1:.6f}\n"
        f"- last_epoch_f1: {last_f1:.6f}\n"
        f"- best_epoch: {best_epoch}\n"
        f"- best_f1: {best_f1:.6f}\n"
        f"- overfit_signal: {overfit_signal}\n"
        f"- trend_hint: {trend_hint}\n"
    )
    (out_dir / "overfit_summary.md").write_text(text, encoding="utf-8")


def write_run_note(out_dir: Path, args: argparse.Namespace, manifests: Dict[str, Path] | None) -> None:
    ensure_dir(out_dir)
    metric_note = (
        "- strict metrics: Precision/Recall/F1/IoU/Dice use exact pixel match\n"
        f"- tolerance-aware metrics: TolPrecision/TolRecall/TolF1 use {args.tolerance_px}px crack matching radius\n"
        "- ValLoss uses strict training loss (CE+Dice main + aux-weighted aux dice)\n"
    )

    text = (
        "# CrackResU-Net Run Note\n\n"
        f"- run_name: {args.run_name}\n"
        f"- variant: {args.variant}\n"
        f"- protocol: {args.protocol}\n"
        f"- epochs: {args.epochs}\n"
        f"- batch_size: {args.batch_size}\n"
        f"- val_batch_size: {args.val_batch_size}\n"
        f"- val_interval_epochs: {args.val_interval_epochs}\n"
        f"- image_size: {args.image_size}\n"
        f"- random_seed: {args.random_seed}\n"
        f"- lr: {args.lr}\n"
        f"- weight_decay: {args.weight_decay}\n"
        f"- aux_weight: {args.aux_weight}\n"
        f"- tiny_overfit: {args.tiny_overfit}\n"
        f"- output_dir: logs/{args.run_name}\n"
        "\n"
        "## Visualization Protocol\n"
        "- 6-panel composite unchanged\n"
        "- grid panel uses 16x16 ANY-POSITIVE (OR) pooling for visualization only\n"
        "\n"
        "## Metric Protocol\n"
        f"{metric_note}"
        "\n"
        "## Manifest Protocol\n"
    )

    if manifests:
        text += (
            f"- train_manifest: {manifests['train'].as_posix()}\n"
            f"- val_manifest: {manifests['val'].as_posix()}\n"
            f"- test_manifest: {manifests['test'].as_posix()}\n"
            "- split ratio: 6:2:2 (deterministic by seed)\n"
        )
    else:
        text += "- using dataset native split behavior (no frozen formal manifests)\n"

    (out_dir / "summary.md").write_text(text, encoding="utf-8")


def write_formal_main_summary(out_dir: Path, args: argparse.Namespace, best: Dict[str, float]) -> None:
    text = (
        "# Formal Main Summary\n\n"
        f"- run_name: {args.run_name}\n"
        f"- variant: {args.variant}\n"
        f"- protocol: {args.protocol}\n"
        f"- epochs: {args.epochs}\n"
        f"- batch_size: {args.batch_size}\n"
        f"- val_batch_size: {args.val_batch_size}\n"
        f"- val_interval_epochs: {args.val_interval_epochs}\n"
        f"- best_epoch: {int(best.get('best_epoch', 0))}\n"
        "\n"
        "## Best Metrics\n"
        f"- Precision: {float(best.get('Precision', 0.0)):.6f}\n"
        f"- Recall: {float(best.get('Recall', 0.0)):.6f}\n"
        f"- F1: {float(best.get('F1', 0.0)):.6f}\n"
        f"- Dice: {float(best.get('Dice', 0.0)):.6f}\n"
        f"- IoU: {float(best.get('IoU', 0.0)):.6f}\n"
        f"- TolPrecision: {float(best.get('TolPrecision', 0.0)):.6f}\n"
        f"- TolRecall: {float(best.get('TolRecall', 0.0)):.6f}\n"
        f"- TolF1: {float(best.get('TolF1', 0.0)):.6f}\n"
        f"- ValLoss: {float(best.get('ValLoss', 0.0)):.6f}\n"
    )
    (out_dir / "formal_main_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.variant = _normalize_variant(args.variant)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    if args.tiny_overfit:
        args.epochs = int(args.tiny_overfit_epochs)
        args.batch_size = 1
        args.val_batch_size = 1
        args.val_interval_epochs = 1
        args.save_top_k_vis = max(args.save_top_k_vis, int(args.tiny_overfit_samples))

    # Formal protocol defaults (paper-aligned) when requested.
    manifests: Dict[str, Path] | None = None
    out_dir = Path("logs") / args.run_name
    if args.protocol == "formal_cfd_622":
        args.image_size = 320
        if args.lr == 1e-3:
            args.lr = 1e-4
        args.weight_decay = 1e-5
        manifests = freeze_formal_manifests(args, out_dir)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    cfg = CrackResUNetConfig(
        num_classes=2,
        pretrained_encoder=not args.use_tiny_random,
        variant=args.variant,
    )
    model = CrackResUNet(cfg).to(device)
    if args.ckpt:
        loaded = load_shape_matched_checkpoint(model, args.ckpt, device)
        print(f"Loaded {loaded} shape-matched tensors from {args.ckpt}")

    train_loader, val_loader, selected_train_stems, selected_val_stems = make_dataloaders(args, manifests=manifests)

    train_cfg = TrainConfig(
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        val_interval_epochs=args.val_interval_epochs,
        aux_weight=args.aux_weight,
        save_top_k_vis=args.save_top_k_vis,
        tolerance_px=args.tolerance_px,
        use_wandb=bool(args.use_wandb),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_job_type=args.wandb_job_type,
        wandb_tags=args.wandb_tags,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.run_name,
        wandb_config={
            "run_name": args.run_name,
            "variant": args.variant,
            "protocol": args.protocol,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "image_size": args.image_size,
            "val_interval_epochs": args.val_interval_epochs,
            "aux_weight": args.aux_weight,
            "random_seed": args.random_seed,
            "tolerance_px": args.tolerance_px,
            "formal_augs": args.formal_augs,
        },
    )

    write_run_note(out_dir, args, manifests)

    _write_lines(out_dir / "selected_train_samples.txt", selected_train_stems)
    _write_lines(out_dir / "selected_val_samples.txt", selected_val_stems)

    best = fit(model, train_loader, val_loader, train_cfg, out_dir, device)

    if args.protocol == "formal_cfd_622":
        write_formal_main_summary(out_dir, args, best)

    if args.tiny_overfit:
        write_overfit_summary(out_dir, args, best)

    print("Best metrics:", best)


if __name__ == "__main__":
    main()
