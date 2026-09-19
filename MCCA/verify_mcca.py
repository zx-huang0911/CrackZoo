import torch

from models import create_model
from utils.losses import MCCABinaryLoss


EXPECTED = {
    "E1": (1, 64, 160, 160),
    "E2": (1, 128, 80, 80),
    "E3": (1, 256, 40, 40),
    "E4": (1, 512, 20, 20),
    "M1": (1, 64, 160, 160),
    "M2": (1, 64, 80, 80),
    "M3": (1, 64, 40, 40),
    "M4": (1, 64, 20, 20),
    "A1": (1, 64, 160, 160),
    "A2": (1, 64, 80, 80),
    "A3": (1, 64, 40, 40),
    "A4": (1, 64, 20, 20),
    "S1": (1, 1, 640, 640),
    "S2": (1, 1, 640, 640),
    "S3": (1, 1, 640, 640),
    "S4": (1, 1, 640, 640),
    "Y": (1, 1, 640, 640),
}


def check_shapes(outputs):
    e1, e2, e3, e4 = outputs["backbone_feats"]
    m1, m2, m3, m4 = outputs["mscfm_feats"]
    a1, a2, a3, a4 = outputs["cam_feats"]
    s1, s2, s3, s4 = outputs["side_logits"]
    y = outputs["final_logit"]

    observed = {
        "E1": tuple(e1.shape),
        "E2": tuple(e2.shape),
        "E3": tuple(e3.shape),
        "E4": tuple(e4.shape),
        "M1": tuple(m1.shape),
        "M2": tuple(m2.shape),
        "M3": tuple(m3.shape),
        "M4": tuple(m4.shape),
        "A1": tuple(a1.shape),
        "A2": tuple(a2.shape),
        "A3": tuple(a3.shape),
        "A4": tuple(a4.shape),
        "S1": tuple(s1.shape),
        "S2": tuple(s2.shape),
        "S3": tuple(s3.shape),
        "S4": tuple(s4.shape),
        "Y": tuple(y.shape),
    }

    for k, v in EXPECTED.items():
        if observed[k] != v:
            raise RuntimeError(f"Shape mismatch: {k} expected={v} observed={observed[k]}")
    return observed


def dry_train_step(model, device):
    model.train()
    criterion = MCCABinaryLoss(loss_name="weighted_bce_dice")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    x = torch.randn(1, 3, 640, 640, device=device)
    target = torch.randint(0, 2, (1, 640, 640), device=device)

    optimizer.zero_grad()
    out = model(x)
    losses = criterion(out, target)
    losses["total_loss"].backward()
    optimizer.step()

    return losses["total_loss"].item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = ["baseline_plain", "baseline_deepsup", "mcca_no_mscfm", "mcca_no_cam", "full_mcca"]

    for variant in variants:
        print(f"=== verify variant: {variant} ===")
        model = create_model("mcca", input_size=640, model_variant=variant).to(device)

        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 640, 640, device=device)
            outputs = model(x)

        y_shape = tuple(outputs["final_logit"].shape)
        if y_shape != EXPECTED["Y"]:
            raise RuntimeError(f"Final logit shape mismatch for {variant}: {y_shape}")

        if variant == "full_mcca":
            observed = check_shapes(outputs)
            for k in ["E1", "E2", "E3", "E4", "M1", "M2", "M3", "M4", "A1", "A2", "A3", "A4", "S1", "S2", "S3", "S4", "Y"]:
                print(f"{k}: {observed[k]}")
        else:
            print(f"Y: {y_shape}; sides={len(outputs.get('side_logits', []))}")

        criterion = MCCABinaryLoss(loss_name="weighted_bce_dice", use_deepsup=getattr(model, "use_deepsup", True))
        model.train()
        x = torch.randn(1, 3, 640, 640, device=device)
        target = torch.randint(0, 2, (1, 640, 640), device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        out = model(x)
        losses = criterion(out, target)
        losses["total_loss"].backward()
        optimizer.step()
        print(f"dry_train_total_loss={losses['total_loss'].item():.6f}")


if __name__ == "__main__":
    main()
