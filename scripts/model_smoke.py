"""CPU/CUDA forward/backward checks with random inputs; never a trained-model benchmark."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["csnet", "mcca", "crackresunet", "yolov7_wmf", "yolov4_dae"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-pipeline", action="store_true", help="Also check YOLOv4 detector backward and combined inference")
    args = parser.parse_args()
    if args.full_pipeline and args.model != "yolov4_dae":
        parser.error("--full-pipeline is only for yolov4_dae")
    output = args.output.resolve()
    import torch
    import torchvision
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; no automatic CPU fallback")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(17)
    folder = {"csnet": "csnet", "mcca": "MCCA", "crackresunet": "CrackResU-Net", "yolov7_wmf": "yolov7-WMF", "yolov4_dae": "YOLOv4_DAE"}[args.model]
    os.chdir(ROOT / folder)
    sys.path.insert(0, str(ROOT / folder))
    started = time.perf_counter()
    x = torch.randn(2, 3, 64, 64, device=device)
    scope = "full model"
    if args.model == "csnet":
        from network.modeling import csnetv3plus_csnet_encoder
        model = csnetv3plus_csnet_encoder(num_classes=2, pretrained_backbone=False)
        select = lambda y: y
    elif args.model == "mcca":
        from models.mcca import MCCA
        model = MCCA(input_size=64, pretrained_backbone=False)
        select = lambda y: y["final_logit"]
    elif args.model == "crackresunet":
        from crackresunet.model import CrackResUNet, CrackResUNetConfig
        model = CrackResUNet(CrackResUNetConfig(pretrained_encoder=False))
        select = lambda y: y["main_logits"]
    elif args.model == "yolov7_wmf":
        from models.yolo import Model
        model = Model("cfg/training/yolov7-wmf-step3-semseg.yaml", ch=3, nc=1)
        select = lambda y: y
    else:
        from models.crack_pipeline_v010 import CrackDAE
        model = CrackDAE()
        scope = "DAE only; detector and combined inference not tested"
        if args.full_pipeline:
            from models.crack_pipeline_v010 import FullCrackPipeline
            pipeline = FullCrackPipeline(num_classes=1).to(device)
            pipeline.train()
            raw = pipeline.basic_model.forward_train(x)
            sum(raw[k].square().mean() for k in ("pred13_raw", "pred26_raw")).backward()
            detector_grads = [p.grad for p in pipeline.basic_model.parameters() if p.grad is not None]
            assert detector_grads and all(torch.isfinite(g).all() for g in detector_grads)
            pipeline.eval()
            with torch.inference_mode():
                clean = pipeline(x)[-1]
            assert clean.shape == (2, 1, 64, 64) and torch.isfinite(clean).all()
            model = pipeline.dae
            scope = "full pipeline inference; separate detector raw-head and DAE backward"
        x = torch.rand(2, 1, 64, 64, device=device)
        select = lambda y: y
    model = model.to(device)
    model.train()
    y = select(model(x))
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"Unexpected output: {type(y).__name__}")
    assert y.shape[0] == x.shape[0] and y.shape[-2:] == x.shape[-2:], y.shape
    assert torch.isfinite(y).all(), "non-finite output"
    loss = y.square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "invalid gradients"
    if device.type == "cuda":
        torch.cuda.synchronize()
    report = dict(model=args.model, scope=scope, input_shape=list(x.shape), output_shape=list(y.shape),
                  finite_output=True, finite_gradients=True, parameters=sum(p.numel() for p in model.parameters()),
                  seed=17, device=str(device), cuda_runtime=torch.version.cuda, torch=torch.__version__, torchvision=torchvision.__version__,
                  seconds=round(time.perf_counter() - started, 3), pretrained_weights=False,
                  validation="random-input forward/backward; no accuracy claim")
    if args.full_pipeline:
        report["pipeline_input_shape"] = [2, 3, 64, 64]
        report["parameter_scope"] = "DAE only; parameter count is not the full detector-plus-DAE"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
