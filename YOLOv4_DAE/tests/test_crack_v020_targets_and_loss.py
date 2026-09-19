from __future__ import annotations

import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.crack_dataset_v020 import CrackBasicDatasetV020, create_crack_dataloader_v020
from models.crack_pipeline_v010 import BasicCrackModel
from utils.loss_crack_basic_v020 import CrackBasicLossV020
from utils.targets_threshold_v020 import build_targets_with_threshold_v020
from val_crack_basic_v020 import evaluate_basic_model


def _build_tiny_dataset(base: Path, image_size: int = 128) -> dict:
    if base.exists():
        shutil.rmtree(base)
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels_threshold").mkdir(parents=True, exist_ok=True)
    (base / "masks").mkdir(parents=True, exist_ok=True)

    for i in range(2):
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.rectangle(img, (32, 32), (96, 96), (220, 220, 220), -1)
        cv2.imwrite(str(base / "images" / f"sample_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        (base / "labels_threshold" / f"sample_{i}.txt").write_text("0 0.5 0.5 0.5 0.5 0.35\n", encoding="utf-8")

        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        mask[32:96, 32:96] = 255
        cv2.imwrite(str(base / "masks" / f"sample_{i}.png"), mask)

    return {
        "image_root": str(base / "images"),
        "label_root": str(base / "labels_threshold"),
        "mask_root": str(base / "masks"),
    }


def test_dataset_smoke() -> None:
    ds_info = _build_tiny_dataset(ROOT / "tests" / "_tmp_v020_dataset")
    ds = CrackBasicDatasetV020(
        image_root=ds_info["image_root"],
        label_root=ds_info["label_root"],
        mask_root=ds_info["mask_root"],
        image_size=128,
        augment=False,
    )

    img, targets, mask, path, _shapes = ds[0]
    assert tuple(img.shape) == (3, 128, 128)
    assert targets.shape[1] == 7
    assert targets.shape[0] == 1
    assert 0.0 <= float(targets[0, 6].item()) <= 1.0
    assert mask is not None and tuple(mask.shape) == (1, 128, 128)
    assert path.endswith(".png")


def test_target_assignment_smoke() -> None:
    ds_info = _build_tiny_dataset(ROOT / "tests" / "_tmp_v020_targets")
    dl, _ = create_crack_dataloader_v020(
        image_root=ds_info["image_root"],
        label_root=ds_info["label_root"],
        mask_root=ds_info["mask_root"],
        image_size=128,
        batch_size=1,
        shuffle=False,
        augment=False,
        workers=0,
    )
    imgs, targets, _masks, _paths, _shapes = next(iter(dl))

    model = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=1).eval()
    with torch.no_grad():
        out = model.forward_train(imgs)

    anchor_vecs = [
        model.pred_head.decode13.anchor_vec.detach(),
        model.pred_head.decode26.anchor_vec.detach(),
    ]
    tcls, tbox, indices, anch, tthr = build_targets_with_threshold_v020(
        preds=out["preds"],
        targets=targets,
        anchor_vecs=anchor_vecs,
        anchor_t=4.0,
    )

    total_pos = sum(x.shape[0] for x in tthr)
    assert total_pos > 0
    for i in range(len(tthr)):
        assert tthr[i].shape[0] == tbox[i].shape[0]
        assert tthr[i].shape[0] == indices[i][0].shape[0]
        if tthr[i].numel() > 0:
            assert float(tthr[i].min().item()) >= 0.0
            assert float(tthr[i].max().item()) <= 1.0


def test_loss_smoke() -> None:
    ds_info = _build_tiny_dataset(ROOT / "tests" / "_tmp_v020_loss")
    dl, _ = create_crack_dataloader_v020(
        image_root=ds_info["image_root"],
        label_root=ds_info["label_root"],
        mask_root=ds_info["mask_root"],
        image_size=128,
        batch_size=1,
        shuffle=False,
        augment=False,
        workers=0,
    )
    imgs, targets, _masks, _paths, _shapes = next(iter(dl))

    model = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=1).train()
    out = model.forward_train(imgs)
    criterion = CrackBasicLossV020(num_classes=1, alpha_threshold_loss=50.0)
    anchor_vecs = [model.pred_head.decode13.anchor_vec, model.pred_head.decode26.anchor_vec]
    loss, items = criterion(preds=out["preds"], targets=targets, anchor_vecs=anchor_vecs)

    assert torch.isfinite(loss).item()
    for k in ["loss_total", "loss_loc", "loss_conf", "loss_cls", "loss_thr"]:
        assert k in items
        assert torch.isfinite(items[k]).item()


def test_train_step_smoke() -> None:
    ds_info = _build_tiny_dataset(ROOT / "tests" / "_tmp_v020_train")
    dl, _ = create_crack_dataloader_v020(
        image_root=ds_info["image_root"],
        label_root=ds_info["label_root"],
        mask_root=ds_info["mask_root"],
        image_size=128,
        batch_size=1,
        shuffle=False,
        augment=False,
        workers=0,
    )
    imgs, targets, _masks, _paths, _shapes = next(iter(dl))

    model = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=1).train()
    criterion = CrackBasicLossV020(num_classes=1, alpha_threshold_loss=50.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    out = model.forward_train(imgs)
    anchor_vecs = [model.pred_head.decode13.anchor_vec, model.pred_head.decode26.anchor_vec]
    loss, _items = criterion(preds=out["preds"], targets=targets, anchor_vecs=anchor_vecs)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def test_val_step_smoke() -> None:
    ds_info = _build_tiny_dataset(ROOT / "tests" / "_tmp_v020_val")
    dl, _ = create_crack_dataloader_v020(
        image_root=ds_info["image_root"],
        label_root=ds_info["label_root"],
        mask_root=ds_info["mask_root"],
        image_size=128,
        batch_size=1,
        shuffle=False,
        augment=False,
        workers=0,
    )

    model = BasicCrackModel(cfg_path="cfg/yolov4-tiny.cfg", num_classes=1).eval()
    metrics = evaluate_basic_model(
        model=model,
        dataloader=dl,
        device=torch.device("cpu"),
        num_classes=1,
        conf_thres=0.25,
        iou_thres=0.45,
        class_names=["crack"],
        save_vis=False,
        vis_dir=None,
        max_vis=0,
    )
    for k in ["precision", "recall", "map50", "map5095", "miou_raw", "f1_raw"]:
        assert k in metrics


if __name__ == "__main__":
    test_dataset_smoke()
    test_target_assignment_smoke()
    test_loss_smoke()
    test_train_step_smoke()
    test_val_step_smoke()
    print("All v0.2.0 dataset/targets/loss/step smoke tests passed.")
