"""Config loading with the checks that catch the usual mistakes early."""
from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["lamah_root"])
    if "<you>" in str(root):
        raise SystemExit(f"Edit `lamah_root` in {path}: it still holds the placeholder path.")
    if not root.is_dir():
        raise SystemExit(f"`lamah_root` = {root} is not a folder. Check the path in {path}.")
    ids = [b["id"] for b in cfg["basins"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate basin ids in config.")
    return cfg
