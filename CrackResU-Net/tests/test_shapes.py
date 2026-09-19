from __future__ import annotations

from pathlib import Path

import torch

from crackresunet.model import CrackResUNet, CrackResUNetConfig


def run_shape_check() -> str:
    model = CrackResUNet(CrackResUNetConfig(pretrained_encoder=False, variant="main"))
    model.eval()

    x = torch.randn(1, 3, 320, 320)
    with torch.no_grad():
        _, shapes = model(x, return_shapes=True)

    lines = ["| Stage | Tensor Shape (N,C,H,W) |", "|---|---|"]
    order = [
        "E1",
        "B1",
        "B2",
        "B3",
        "B4",
        "S1",
        "S2",
        "S3",
        "S4",
        "D4",
        "D3",
        "D2",
        "D1",
        "AuxLogits",
        "MainLogits",
    ]
    for key in order:
        if key in shapes:
            lines.append(f"| {key} | {list(shapes[key])} |")

    table = "\n".join(lines)
    print(table)
    return table


def main() -> None:
    table = run_shape_check()
    out = Path("logs") / "sample_run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "shape_check.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
