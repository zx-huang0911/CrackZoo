from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _polygon_to_bbox(tokens: List[float]) -> Tuple[float, float, float, float]:
    # tokens: [cls, x1, y1, x2, y2, ...] in normalized coords.
    coords = tokens[1:]
    if len(coords) < 4 or (len(coords) % 2) != 0:
        raise ValueError("Invalid polygon token length")

    xs = coords[0::2]
    ys = coords[1::2]

    x_min = _clamp01(min(xs))
    x_max = _clamp01(max(xs))
    y_min = _clamp01(min(ys))
    y_max = _clamp01(max(ys))

    xc = (x_min + x_max) * 0.5
    yc = (y_min + y_max) * 0.5
    w = max(0.0, x_max - x_min)
    h = max(0.0, y_max - y_min)

    return _clamp01(xc), _clamp01(yc), _clamp01(w), _clamp01(h)


def _iter_label_files(label_dir: Path) -> Iterable[Path]:
    for p in sorted(label_dir.glob("*.txt")):
        if p.is_file():
            yield p


def convert_split(
    label_in_dir: Path,
    label_out_dir: Path,
    default_threshold: float,
) -> Tuple[int, int, int, int]:
    label_out_dir.mkdir(parents=True, exist_ok=True)

    file_count = 0
    line_in_count = 0
    line_out_count = 0
    malformed_line_count = 0

    for src in _iter_label_files(label_in_dir):
        file_count += 1
        out_lines: List[str] = []

        for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
            raw = raw.strip()
            if not raw:
                continue

            line_in_count += 1
            parts = raw.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                malformed_line_count += 1
                continue

            # Already in YOLO bbox (5 or 6 fields): keep bbox, override/append threshold.
            if len(vals) == 5 or len(vals) == 6:
                cls_id = int(vals[0])
                xc, yc, w, h = [_clamp01(v) for v in vals[1:5]]
                out_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {default_threshold:.6f}")
                line_out_count += 1
                continue

            # Polygon-style labels (class + 2*k coords): convert to bbox.
            if len(vals) > 6:
                cls_id = int(vals[0])
                try:
                    xc, yc, w, h = _polygon_to_bbox(vals)
                except ValueError:
                    malformed_line_count += 1
                    continue
                out_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f} {default_threshold:.6f}")
                line_out_count += 1
                continue

            # Ignore malformed lines silently to keep conversion robust.
            malformed_line_count += 1

        dst = label_out_dir / src.name
        dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    return file_count, line_in_count, line_out_count, malformed_line_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert crack polygon labels to v0.2.0 bbox+threshold labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to crack_segmentation_dataset root",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Threshold value appended to each converted label row",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    default_threshold = _clamp01(float(args.default_threshold))
    print(
        "[bridge-note] default threshold is a temporary placeholder for local"
        " v0.2.0 bridge, not paper-grade threshold ground truth."
    )

    for split in ["train", "test"]:
        label_in_dir = dataset_root / split / "labels"
        if not label_in_dir.exists():
            raise FileNotFoundError(f"Missing label dir: {label_in_dir}")

        label_out_dir = dataset_root / split / "labels_threshold"
        files, lines_in, lines_out, malformed = convert_split(
            label_in_dir=label_in_dir,
            label_out_dir=label_out_dir,
            default_threshold=default_threshold,
        )
        print(
            f"[{split}] files={files} lines_in={lines_in} lines_out={lines_out} malformed={malformed} "
            f"output={label_out_dir}"
        )


if __name__ == "__main__":
    main()
