import os
import torch


def save_checkpoint(path, model, optimizer, scheduler, cur_itrs, best_score):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "cur_itrs": cur_itrs,
        "best_score": best_score,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, continue_training=False, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("model_state", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(cleaned)

    cur_itrs = 0
    best_score = 0.0
    if continue_training:
        if optimizer is not None and ckpt.get("optimizer_state") is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        cur_itrs = ckpt.get("cur_itrs", 0)
        best_score = ckpt.get("best_score", 0.0)

    return cur_itrs, best_score
