from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

from .paths import DEFAULT_DATASET_ROOT, REPO_ROOT


@dataclass(frozen=True)
class CommandSpec:
    cwd: str
    argv: List[str]
    env_name: str
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    family: str
    source_dir: str
    checkpoint: str
    train: CommandSpec
    eval: CommandSpec
    ablation: CommandSpec | None = None


PYTHON = sys.executable


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "csnet": ModelSpec(
        key="csnet",
        name="CSNet / DeepLabV3Plus",
        family="segmentation",
        source_dir="csnet",
        checkpoint="models/CSNet/0/csnet_crack.pth",
        train=CommandSpec(
            cwd="csnet",
            env_name="seg25",
            argv=[
                PYTHON,
                "main_crack.py",
                "--dataset",
                "crack",
                "--data_root",
                "{dataset}",
                "--model",
                "csnetv3plus_csnet_encoder",
                "--total_itrs",
                "{total_itrs}",
                "--batch_size",
                "{batch_size}",
                "--val_batch_size",
                "{val_batch_size}",
                "--crop_size",
                "513",
                "--val_interval",
                "{val_interval}",
                "--loss_type",
                "generalized_dice",
                "--random_seed",
                "{seed}",
            ],
        ),
        eval=CommandSpec(
            cwd="csnet",
            env_name="seg25",
            argv=[
                PYTHON,
                "main_crack.py",
                "--dataset",
                "crack",
                "--data_root",
                "{dataset}",
                "--model",
                "csnetv3plus_csnet_encoder",
                "--test_only", "--eval_split", "test",
                "--ckpt",
                "{checkpoint}",
                "--val_batch_size",
                "{val_batch_size}",
                "--crop_size",
                "513",
            ],
        ),
    ),
    "mcca": ModelSpec(
        key="mcca",
        name="MCCA",
        family="segmentation",
        source_dir="MCCA",
        checkpoint="models/MCCA/0/mcca_phase3e_sw0025_best.pth",
        train=CommandSpec(
            cwd="MCCA",
            env_name="seg25",
            argv=[
                PYTHON,
                "train.py",
                "--data_root",
                "{dataset}",
                "--model",
                "full_mcca",
                "--model_variant",
                "full_mcca",
                "--input_size",
                "640",
                "--batch_size",
                "{batch_size}",
                "--val_batch_size",
                "{val_batch_size}",
                "--total_itrs",
                "{total_itrs}",
                "--val_interval",
                "{val_interval}",
                "--random_seed",
                "{seed}",
                "--save_dir",
                "{output}/mcca_seed{seed}",
            ],
        ),
        eval=CommandSpec(
            cwd="MCCA",
            env_name="seg25",
            argv=[
                PYTHON,
                "train.py",
                "--test_only", "--eval_split", "test",
                "--data_root",
                "{dataset}",
                "--model",
                "full_mcca",
                "--model_variant",
                "full_mcca",
                "--ckpt",
                "{checkpoint}",
                "--input_size",
                "640",
                "--val_batch_size",
                "{val_batch_size}",
            ],
        ),

    ),
    "crackresunet": ModelSpec(
        key="crackresunet",
        name="CrackResU-Net",
        family="segmentation",
        source_dir="CrackResU-Net",
        checkpoint="models/CrackResU-Net/0/crackresunet_v0_4_0_best.pth",
        train=CommandSpec(
            cwd="CrackResU-Net",
            env_name="seg25",
            argv=[
                PYTHON,
                "train.py",
                "--data_root",
                "{dataset}",
                "--run_name",
                "crackres_main_seed{seed}",
                "--variant",
                "main",
                "--protocol",
                "auto",
                "--epochs",
                "{epochs}",
                "--batch_size",
                "{batch_size}",
                "--val_batch_size",
                "{val_batch_size}",
                "--val_interval_epochs",
                "{val_interval_epochs}",
                "--random_seed",
                "{seed}",
                "--wandb_mode",
                "disabled",
            ],
        ),
        eval=CommandSpec(
            cwd="CrackResU-Net",
            env_name="seg25",
            argv=[
                PYTHON,
                "evaluate.py", "--data-root", "{dataset}",
                "--checkpoint", "{checkpoint}", "--output", "{output}/crackresunet_eval",
            ],
            note="Evaluate a supplied compatible checkpoint on the explicit test split.",
        ),
        ablation=CommandSpec(
            cwd="CrackResU-Net",
            env_name="seg25",
            argv=[
                PYTHON,
                "run_ablation.py",
                "--data_root",
                "{dataset}",
                "--protocol",
                "auto",
                "--epochs",
                "{epochs}",
                "--batch_size",
                "{batch_size}",
                "--val_batch_size",
                "{val_batch_size}",
                "--random_seed",
                "{seed}",
                "--wandb_mode",
                "disabled",
            ],
        ),
    ),
    "yolov7_wmf": ModelSpec(
        key="yolov7_wmf",
        name="YOLOv7-WMF",
        family="detection_segmentation",
        source_dir="yolov7-WMF",
        checkpoint="models/YOLOv7-WMF/0/yolov7-WMF.pt",
        train=CommandSpec(
            cwd="yolov7-WMF",
            env_name="yolov7",
            argv=[
                PYTHON,
                "train_seg.py",
                "--cfg",
                "cfg/training/yolov7-wmf-step3-semseg.yaml",
                "--data",
                "{config}",
                "--weights",
                "",
                "--epochs",
                "{epochs}",
                "--batch",
                "{batch_size}",
                "--img",
                "640",
                "--project",
                "{output}/yolov7_wmf",
                "--name",
                "step3_seed{seed}",
                "--seed", "{seed}",
            ],
        ),
        eval=CommandSpec(
            cwd="yolov7-WMF",
            env_name="yolov7",
            argv=[
                PYTHON,
                "val_seg.py",
                "--cfg",
                "cfg/training/yolov7-wmf-step3-semseg.yaml",
                "--data",
                "{config}",
                "--weights",
                "{checkpoint}",
                "--img",
                "640",
                "--batch",
                "{val_batch_size}",
                "--project",
                "{output}/yolov7_wmf_val",
                "--name",
                "step3_eval",
            ],
        ),
    ),
    "yolov4_dae": ModelSpec(
        key="yolov4_dae",
        name="YOLOv4-DAE",
        family="detection_grid",
        source_dir="YOLOv4_DAE",
        checkpoint="models/YOLOv4_DAE/0/epoch_200.pt",
        train=CommandSpec(
            cwd="YOLOv4_DAE",
            env_name="seg25",
            argv=[
                PYTHON,
                "train_crack_dae_v050.py",
                "--config",
                "{config}",
            ],
        ),
        eval=CommandSpec(
            cwd="YOLOv4_DAE",
            env_name="seg25",
            argv=[
                PYTHON,
                "scripts/run_v050_dae_integration_eval.py",
                "--config",
                "{config}",
            ],
        ),
    ),
}


DEFAULTS = {
    "dataset": str(DEFAULT_DATASET_ROOT),
    "repo": str(REPO_ROOT),
    "output": str(REPO_ROOT / "runs"),
    "seed": "1",
    "epochs": "50",
    "total_itrs": "10000",
    "batch_size": "8",
    "val_batch_size": "4",
    "val_interval": "500",
    "val_interval_epochs": "5",
}


def model_keys() -> List[str]:
    return sorted(MODEL_REGISTRY)


def get_model(key: str) -> ModelSpec:
    normalized = key.lower().replace("-", "_")
    aliases = {
        "yolov7-wmf": "yolov7_wmf",
        "yolov4-dae": "yolov4_dae",
        "crackresu_net": "crackresunet",
        "crackresu-net": "crackresunet",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(model_keys())}")
    return MODEL_REGISTRY[normalized]


def format_argv(argv: List[str], **overrides: str) -> List[str]:
    values = dict(DEFAULTS)
    values.update({k: str(v) for k, v in overrides.items() if v is not None})
    return [part.format(**values) for part in argv]


def command_for(model_key: str, mode: str, **overrides: str) -> CommandSpec:
    spec = get_model(model_key)
    if mode not in {"train", "eval", "ablation"}:
        raise ValueError(f"Unknown mode: {mode}")
    command = getattr(spec, mode)
    if command is None:
        raise ValueError(f"Model '{model_key}' does not define mode '{mode}'")
    overrides = dict(overrides)
    overrides.setdefault("checkpoint", str(REPO_ROOT / spec.checkpoint))
    overrides.setdefault("config", str(REPO_ROOT / spec.source_dir / ("data/crack_seg.yaml" if model_key == "yolov7_wmf" else "configs/crack_v050_dae_rawmask.yaml")))
    for key in ("dataset", "output", "checkpoint", "config"):
        if overrides.get(key) is not None:
            overrides[key] = str(Path(overrides[key]).expanduser().resolve())
    return CommandSpec(
        cwd=str((REPO_ROOT / command.cwd).resolve()),
        argv=format_argv(command.argv, **overrides),
        env_name=command.env_name,
        note=command.note,
    )
