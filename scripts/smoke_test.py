from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackzoo.metrics import binary_metrics, grid_metrics
from crackzoo.registry import MODEL_REGISTRY, REPO_ROOT, command_for


def check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"[OK] {message}")
    else:
        print(f"[FAIL] {message}")
        failures.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast structural smoke test for CrackZoo.")
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--compile-third-party", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports" / "smoke_test_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []

    for key, spec in MODEL_REGISTRY.items():
        check((REPO_ROOT / spec.source_dir).exists(), f"{key} source exists", failures)
        if args.require_checkpoints:
            check((REPO_ROOT / spec.checkpoint).exists(), f"{key} checkpoint exists locally", failures)
        for mode in ("train", "eval"):
            command = command_for(key, mode)
            check(Path(command.cwd).exists(), f"{key} {mode} cwd exists", failures)
            if len(command.argv) > 1 and command.argv[1].endswith(".py"):
                check((Path(command.cwd) / command.argv[1]).exists(), f"{key} {mode} entry exists", failures)

    metric = binary_metrics([[0, 1], [1, 1]], [[0, 1], [0, 1]])
    check(metric["tp"] == 2 and metric["fn"] == 1, "binary metrics sanity", failures)
    grid = grid_metrics([[0, 1], [1, 1]], [[0, 1], [0, 1]], grid_size=1)
    check(grid["grid_recall"] > 0.6, "grid metrics sanity", failures)

    for path in [*(ROOT / "crackzoo").glob("*.py"), *(ROOT / "scripts").glob("*.py")]:
        py_compile.compile(str(path), doraise=True)
        print(f"[OK] compiled {path}")

    if args.compile_third_party:
        for path in [
            Path("csnet/main_crack.py"),
            Path("MCCA/train.py"),
            Path("CrackResU-Net/train.py"),
            Path("YOLOv4_DAE/train_crack_dae_v050.py"),
            Path("yolov7-WMF/train_seg.py"),
        ]:
            py_compile.compile(str(ROOT / path), doraise=True)
            print(f"[OK] compiled {path}")

    report = {"ok": not failures, "failures": failures}
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
