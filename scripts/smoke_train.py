"""Train on generated masks, validate, and round-trip a checkpoint on CPU or CUDA.

Runs one model per process to isolate upstream imports. No external data or weights.
This small training harness exercises native models/losses, not the full experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"csnet": "csnet", "mcca": "MCCA", "crackresunet": "CrackResU-Net",
           "yolov7_wmf": "yolov7-WMF", "yolov4_dae": "YOLOv4_DAE"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=FOLDERS)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--keep-checkpoint", action="store_true")
    a = p.parse_args()
    if a.steps < 1 or a.size < 64 or a.size % 32:
        p.error("steps must be positive; size must be >=64 and a multiple of 32")
    out = a.output.resolve()
    if not out.is_relative_to(ROOT / ".local"):
        p.error("output must be inside the repository's .local directory")
    out.mkdir(parents=True, exist_ok=False)

    import torch
    import torchvision
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no automatic CPU fallback")
    device = torch.device(a.device)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
        torch.cuda.reset_peak_memory_stats()
    os.chdir(ROOT / FOLDERS[a.model])
    sys.path.insert(0, str(Path.cwd()))
    started = time.perf_counter()

    def samples(seed):
        generator = torch.Generator().manual_seed(seed)
        mask = torch.zeros(4, a.size, a.size, dtype=torch.long)
        for i in range(4):
            for row in range(a.size):
                col = (row // 3 + i * 7 + seed) % (a.size - 4)
                mask[i, row, col:col + 3] = 1
        rgb = (0.7 - 0.5 * mask[:, None].float()).expand(-1, 3, -1, -1)
        rgb = (rgb + 0.03 * torch.randn(rgb.shape, generator=generator)).clamp(0, 1)
        return rgb, mask

    train_x, train_y = samples(17)
    val_x, val_y = samples(29)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=2, shuffle=False, num_workers=0)
    val_x, val_y = val_x[:2].to(device), val_y[:2].to(device)

    if a.model == "csnet":
        from network.modeling import csnetv3plus_csnet_encoder
        from utils.loss import GeneralizedDiceLoss
        model = csnetv3plus_csnet_encoder(num_classes=2, pretrained_backbone=False)
        criterion = GeneralizedDiceLoss(ignore_index=255)
        loss_fn = lambda x, y: criterion(model(x), y)
        predict = lambda x: model(x)
        loss_name = "GeneralizedDiceLoss"
    elif a.model == "mcca":
        from models.mcca import MCCA
        from utils.losses import MCCABinaryLoss
        model = MCCA(input_size=a.size, pretrained_backbone=False)
        criterion = MCCABinaryLoss()
        loss_fn = lambda x, y: criterion(model(x), y)["total_loss"]
        predict = lambda x: model(x)["final_logit"]
        loss_name = "weighted BCE + Dice + side supervision"
    elif a.model == "crackresunet":
        from crackresunet.model import CrackResUNet, CrackResUNetConfig
        from crackresunet.losses import CrackLoss
        model = CrackResUNet(CrackResUNetConfig(pretrained_encoder=False))
        criterion = CrackLoss()
        loss_fn = lambda x, y: criterion(model(x), y)["total_loss"]
        predict = lambda x: model(x)["main_logits"]
        loss_name = "CE + Dice + auxiliary Dice"
    elif a.model == "yolov7_wmf":
        from models.yolo import Model
        from utils.segmentation import SegLoss
        model = Model("cfg/training/yolov7-wmf-step3-semseg.yaml", ch=3, nc=1)
        criterion = SegLoss()
        loss_fn = lambda x, y: criterion(model(x), y[:, None].float())[0]
        predict = lambda x: model(x)
        loss_name = "BCE + Dice"
    else:
        from models.crack_pipeline_v010 import FullCrackPipeline
        from utils.loss_crack_basic_v020 import CrackBasicLossV020
        from utils.dae_metrics_v050 import dice_loss_from_probs
        model = FullCrackPipeline(num_classes=1)
        criterion = CrackBasicLossV020(num_classes=1)

        def loss_fn(x, y):
            # Synthetic boxes exercise threshold target assignment, not mask-derived boxes.
            targets = torch.tensor([[i, 0, .5, .5, .5, .5, .35] for i in range(x.shape[0])], device=device)
            raw = model.basic_model.forward_train(x)
            head = model.basic_model.pred_head
            det_loss, items = criterion(raw["preds"], targets, [head.decode13.anchor_vec, head.decode26.anchor_vec])
            if int(items["num_positive_anchors"]) <= 0:
                raise AssertionError("No positive detector targets assigned")
            gt = y[:, None].float()
            corrupted = (0.8 * gt + 0.1).clamp(0, 1)
            clean = model.dae(corrupted)
            return det_loss.sum() + F.binary_cross_entropy(clean, gt) + dice_loss_from_probs(clean, gt)

        predict = lambda x: model(x)[-1]
        loss_name = "detector box/objectness/threshold + separate DAE BCE/Dice"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    tracked = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    # Small probes in each parameter verify updates without a second GPU model copy.
    before = {n: v.detach().flatten()[:32].cpu().clone() for n, v in tracked}
    losses = []
    batches = iter(loader)
    model.train()
    for step in range(a.steps):
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y = next(batches)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(x, y)
        if not torch.isfinite(loss).all():
            raise AssertionError("Non-finite training loss")
        loss.backward()
        grads = [v.grad for _, v in tracked if v.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise AssertionError("Missing or non-finite gradients")
        optimizer.step()
        if not all(torch.isfinite(v).all() for _, v in tracked):
            raise AssertionError("Non-finite model parameters")
        losses.append(float(loss.detach()))

    changed = [n for n, v in tracked if not torch.equal(before[n], v.detach().flatten()[:32].cpu())]
    if not changed:
        raise AssertionError("Optimizer did not update any probed parameter")
    if a.model == "yolov4_dae":
        for prefix in ("basic_model.", "dae."):
            if not any(n.startswith(prefix) for n in changed):
                raise AssertionError(f"No parameter update for {prefix}")
    model.eval()
    with torch.no_grad():
        validation_loss = float(loss_fn(val_x, val_y))
        expected = predict(val_x).detach().clone()
    if not torch.isfinite(torch.tensor(validation_loss)) or not torch.isfinite(expected).all():
        raise AssertionError("Non-finite validation result")
    if expected.shape[0] != 2 or expected.shape[-2:] != (a.size, a.size):
        raise AssertionError(f"Unexpected prediction shape: {expected.shape}")

    checkpoint = out / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": a.steps}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Mutate both weights and optimizer before restoring so reload is actually exercised.
    with torch.no_grad():
        for _, parameter in tracked:
            parameter.zero_()
    optimizer.state.clear()
    loaded = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(loaded["model"], strict=True)
    optimizer.load_state_dict(loaded["optimizer"])
    if loaded["steps"] != a.steps or not optimizer.state:
        raise AssertionError("Checkpoint metadata or optimizer state missing")
    with torch.no_grad():
        actual = predict(val_x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = dict(model=a.model, status="passed", device=str(device),
                  gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
                  python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
                  cuda_runtime=torch.version.cuda, seed=17, steps=a.steps, batch_size=2, image_size=a.size,
                  train_data="4 synthetic line masks", validation_data="2 separately generated line masks (seed 29)",
                  loss=loss_name, training_losses=losses, validation_loss=validation_loss,
                  finite_gradients=True, parameters_updated=len(changed),
                  checkpoint_strict_reload=True, optimizer_state_reloaded=True, prediction_reload_matches=True,
                  checkpoint_sha256=digest, checkpoint_retained=a.keep_checkpoint,
                  output_shape=list(expected.shape), pretrained_weights=False, precision="float32",
                  peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
                  seconds=round(time.perf_counter() - started, 3),
                  scope="synthetic-data training harness; not native CLI or research accuracy reproduction")
    if a.model == "yolov4_dae":
        report["yolov4_scope"] = "Separate detector and DAE losses; combined inference. No gradient through rasterization/NMS; no real cache generation."
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if not a.keep_checkpoint:
        checkpoint.unlink()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
