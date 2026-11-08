"""CP2 evaluation protocol: hydrological-year folds, purge, and the hold-out guard.

- Development samples: target_hy in `development_hy`; hold-out: target_hy in `holdout_hy`.
- Hold-out samples are never used at CP2, not even for training. Every function that
  could hand them out raises `HoldoutAccessError` unless `allow_holdout=True` (CP4 only).
- Blocked folds (`folds.blocks`): test = development samples whose target_hy is in the
  block; train = the other development samples minus the purge (targets within
  `purge_days` before the block's first day or after its last day).
- LOYO: one fold per development HY, same purge (RF robustness check only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build_features import hy_bounds


class HoldoutAccessError(RuntimeError):
    """Raised when hold-out samples would be used before CP4."""


@dataclass
class Fold:
    fold_id: int
    scheme: str
    test_hy: tuple[int, int]
    first_day: pd.Timestamp
    last_day: pd.Timestamp
    train: np.ndarray          # boolean mask over the sample table
    test: np.ndarray
    purged: np.ndarray

    @property
    def label(self) -> str:
        a, b = self.test_hy
        return f"HY{a}" if a == b else f"HY{a}-{b}"


def _hy_range(pair) -> tuple[int, int]:
    a, b = int(pair[0]), int(pair[1])
    if a > b:
        raise ValueError(f"bad HY range {pair}")
    return a, b


def is_development(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    a, b = _hy_range(cfg["development_hy"])
    return ((meta["target_hy"] >= a) & (meta["target_hy"] <= b)).to_numpy()


def is_holdout(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    a, b = _hy_range(cfg["holdout_hy"])
    return ((meta["target_hy"] >= a) & (meta["target_hy"] <= b)).to_numpy()


def guard_holdout(meta: pd.DataFrame, cfg: dict, allow_holdout: bool = False) -> None:
    """Raise if the sample table contains hold-out samples and access was not allowed."""
    n = int(is_holdout(meta, cfg).sum())
    if n and not allow_holdout:
        raise HoldoutAccessError(
            f"{n} hold-out samples (HY{cfg['holdout_hy'][0]}-{cfg['holdout_hy'][1]}) present; "
            "hold-out data is reserved for CP4 (pass allow_holdout=True only there)")


def holdout_samples(meta: pd.DataFrame, cfg: dict, allow_holdout: bool = False) -> pd.DataFrame:
    """Return the hold-out samples. Raises unless `allow_holdout=True` (CP4 only)."""
    if not allow_holdout:
        raise HoldoutAccessError("hold-out samples are reserved for CP4: pass allow_holdout=True")
    return meta.loc[is_holdout(meta, cfg)]


def development_samples(meta: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return meta.loc[is_development(meta, cfg)]


def fold_blocks(cfg: dict, scheme: str) -> list[tuple[int, int]]:
    dev0, dev1 = _hy_range(cfg["development_hy"])
    if scheme == "loyo":
        return [(h, h) for h in range(dev0, dev1 + 1)]
    if scheme != "blocked":
        raise ValueError(f"unknown fold scheme {scheme!r}")
    blocks = [_hy_range(b) for b in cfg["folds"]["blocks"]]
    covered = [h for a, b in blocks for h in range(a, b + 1)]
    if sorted(covered) != list(range(dev0, dev1 + 1)):
        raise ValueError("fold blocks must cover every development HY exactly once")
    return blocks


def make_folds(meta: pd.DataFrame, cfg: dict, scheme: str | None = None,
               allow_holdout: bool = False) -> list[Fold]:
    """Folds over the sample table `meta` (needs columns target_date, target_hy)."""
    guard_holdout(meta, cfg, allow_holdout)
    scheme = scheme or cfg["folds"]["scheme"]
    start_month = int(cfg["hydro_year_start_month"])
    purge = pd.Timedelta(days=int(cfg["folds"]["purge_days"]))
    dev = is_development(meta, cfg)
    t_date = pd.DatetimeIndex(meta["target_date"])
    t_hy = meta["target_hy"].to_numpy()

    folds = []
    for i, (a, b) in enumerate(fold_blocks(cfg, scheme)):
        first, last = hy_bounds(a, b, start_month)
        test = dev & (t_hy >= a) & (t_hy <= b)
        near = ((t_date >= first - purge) & (t_date < first)) | ((t_date > last) & (t_date <= last + purge))
        purged = dev & ~test & np.asarray(near)
        train = dev & ~test & ~purged
        folds.append(Fold(fold_id=i, scheme=scheme, test_hy=(a, b), first_day=first, last_day=last,
                          train=train, test=test, purged=purged))
    return folds
