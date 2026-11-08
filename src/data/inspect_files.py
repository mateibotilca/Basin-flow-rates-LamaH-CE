"""Step 1: check that every file the config needs exists and print its header.

Usage:
    python -m src.data.inspect_files --config configs/basins.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.data.lamah_io import ATTRIBUTE_FILES, meteo_path, runoff_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/basins.yaml")
    cfg = load_config(ap.parse_args().config)
    root = cfg["lamah_root"]

    paths = []
    for b in cfg["basins"]:
        paths.append(("runoff target", runoff_path(root, b["id"])))
        paths += [("runoff upstream", runoff_path(root, u)) for u in b["upstream"]]
        paths.append(("meteo (A)", meteo_path(root, b["id"])))
    paths += [("attributes", Path(root) / rel) for rel in ATTRIBUTE_FILES.values()]

    ok = True
    shown = set()
    for kind, p in paths:
        if not p.exists():
            print(f"[MISSING] {kind:16s} {p}")
            ok = False
            continue
        print(f"[ok]      {kind:16s} {p}")
        if kind not in shown:  # show one header per file type
            head = pd.read_csv(p, sep=";", nrows=3)
            print("          columns:", list(head.columns))
            print(head.to_string(index=False).replace("\n", "\n          "))
            shown.add(kind)
    print("\nAll files found." if ok else "\nSome files are missing - fix lamah_root or unzip them.")


if __name__ == "__main__":
    main()
