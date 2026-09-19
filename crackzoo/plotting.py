from __future__ import annotations

import csv
from pathlib import Path
from typing import List


def read_rows(path: str | Path) -> List[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def plot_pareto(
    csv_path: str | Path,
    out_path: str | Path,
    x: str = "fps",
    y: str = "dice",
    label: str = "model",
) -> None:
    import matplotlib.pyplot as plt

    rows = read_rows(csv_path)
    xs = [float(row[x]) for row in rows]
    ys = [float(row[y]) for row in rows]
    labels = [row.get(label, "") for row in rows]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(xs, ys, s=48)
    for x_val, y_val, text in zip(xs, ys, labels):
        ax.annotate(text, (x_val, y_val), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_bars(
    csv_path: str | Path,
    out_path: str | Path,
    metric: str = "dice",
    label: str = "model",
) -> None:
    import matplotlib.pyplot as plt

    rows = read_rows(csv_path)
    names = [row.get(label, "") for row in rows]
    values = [float(row[metric]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 0.8), 4.5), dpi=160)
    ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
