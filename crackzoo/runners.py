from __future__ import annotations

import subprocess
import os
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from .registry import CommandSpec, command_for


def command_line(argv: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(argv)) if os.name == "nt" else shlex.join(argv)


def run_command(spec: CommandSpec, execute: bool = False) -> int:
    print(f"Legacy environment hint (not activated): {spec.env_name}")
    print(f"cwd: {spec.cwd}")
    print(command_line(spec.argv))
    if spec.note:
        print(f"note: {spec.note}")
    if not execute:
        return 0
    # Fail before launching an expensive run when a required local asset is absent.
    for flag in ("--ckpt", "--checkpoint", "--config", "--data", "--data_root", "--data-root"):
        if flag in spec.argv:
            value = spec.argv[spec.argv.index(flag) + 1]
            path = Path(value)
            if not path.is_absolute():
                path = Path(spec.cwd) / path
            if not path.exists():
                raise FileNotFoundError(f"{flag}: {path}")
    for flag in ("--data_root", "--data-root"):
        if flag in spec.argv:
            from .data_audit import audit_dataset
            result = audit_dataset(spec.argv[spec.argv.index(flag) + 1])
            if not result["ok"]:
                raise ValueError("Dataset split audit failed: " + "; ".join(result["errors"][:8]))
    return subprocess.run(spec.argv, cwd=spec.cwd, check=False).returncode


def build_suite(model_keys: List[str], mode: str, **kwargs: str) -> List[CommandSpec]:
    return [command_for(key, mode, **kwargs) for key in model_keys]


def write_command_manifest(path: str | Path, commands: List[CommandSpec]) -> None:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps([asdict(command) for command in commands], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
