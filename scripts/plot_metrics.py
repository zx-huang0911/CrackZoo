from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crackzoo.plotting import plot_metric_bars, plot_pareto


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CrackZoo metric tables.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", default="bar", choices=["bar", "pareto"])
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--x", default="fps")
    parser.add_argument("--y", default="dice")
    parser.add_argument("--label", default="model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.kind == "pareto":
        plot_pareto(args.csv, args.out, x=args.x, y=args.y, label=args.label)
    else:
        plot_metric_bars(args.csv, args.out, metric=args.metric, label=args.label)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
