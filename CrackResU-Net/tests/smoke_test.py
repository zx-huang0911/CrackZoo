from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crackresunet.dataset import TinyRandomDataset
from crackresunet.engine import TrainConfig, fit
from crackresunet.model import CrackResUNet, CrackResUNetConfig


def main() -> None:
    device = torch.device("cpu")
    model = CrackResUNet(CrackResUNetConfig(pretrained_encoder=False, variant="main")).to(device)

    train_ds = TinyRandomDataset(n=4, image_size=320)
    val_ds = TinyRandomDataset(n=4, image_size=320)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    cfg = TrainConfig(
        run_name="sample_run",
        epochs=1,
        batch_size=1,
        val_batch_size=1,
        lr=1e-3,
        num_workers=0,
        val_interval_epochs=1,
        aux_weight=1.0,
        save_top_k_vis=2,
    )
    best = fit(model, train_loader, val_loader, cfg, out_dir=Path("logs") / "sample_run", device=device)
    print("SMOKE_TEST_OK", best)


if __name__ == "__main__":
    main()
