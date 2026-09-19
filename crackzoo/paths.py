from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(os.environ.get("CRACKZOO_DATASET", REPO_ROOT / "data" / "private")).expanduser().resolve()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (REPO_ROOT / value).resolve()


def model_path(relative_path: str | Path) -> Path:
    return resolve_repo_path(Path("models") / relative_path)
