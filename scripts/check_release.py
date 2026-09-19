"""Check the tracked release snapshot, not ignored local research material."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "docs/assets/synthetic_example.png",
    "docs/assets/same-domain-iou.png",
    "docs/assets/pixel-iou-grid-f1.png",
    "docs/assets/module-ablation.png",
    "docs/assets/pixel-predictions.png",
    "docs/assets/grid-predictions.png",
}
FORBIDDEN = {".pt", ".pth", ".ckpt", ".onnx", ".npy", ".npz", ".pkl", ".pdf", ".docx", ".pptx", ".zip", ".pyc"}
PRIVATE_DIRS = {".local", "reports", "runs", "wandb", "checkpoints", "backend", "Agent_Logs"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
SECRETS = re.compile(rb"gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--for-publication", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "reports/release_check.json")
    a = p.parse_args()
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    files = [name for name in files if name]
    if not files:
        raise SystemExit("No tracked files: stage the intended source snapshot first.")
    errors = []
    for name in files:
        path = ROOT / name
        parts = Path(name).parts
        if path.is_symlink() or any(x in PRIVATE_DIRS for x in parts):
            errors.append({"path": name, "reason": "private directory or symlink"})
        if path.suffix.lower() in FORBIDDEN or (path.suffix.lower() in IMAGE_SUFFIXES and name not in ASSETS):
            errors.append({"path": name, "reason": "research artifact or unreviewed image"})
        if path.stat().st_size > 2_000_000:
            errors.append({"path": name, "reason": "oversized file"})
        if SECRETS.search(path.read_bytes()):
            errors.append({"path": name, "reason": "credential-like content (value omitted)"})
    review = json.loads((ROOT / "configs/release_review.json").read_text())
    blockers = {k: v for k, v in review.items() if v["status"] != "resolved"}
    report = dict(tracked_files=len(files), content_errors=errors, review_blockers=blockers,
                  public_ready=not errors and not blockers,
                  limitation="Pattern and file-policy checks do not establish legal clearance or scan old remote history")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"tracked={len(files)}, content_errors={len(errors)}, unresolved_review_items={len(blockers)}")
    return 1 if errors or (a.for_publication and blockers) else 0


if __name__ == "__main__":
    raise SystemExit(main())
