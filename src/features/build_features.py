"""CP2 representation layer: samples, targets and the feature representations A and B.

One sample per issue date t (end of day t, 24:00 UTC):
    target_date = t + lead_days,   y = q_target(target_date)   [m3/s]
    target_hy   = hydrological year of the target date (Oct-Sep, labelled by its end year)

Features are built exactly as listed under `representations` in configs/cp2.yaml:
    lags:  [k] -> value(t - k)                      `<var>_lag<k>`
    diffs: [k] -> value(t) - value(t - k)           `d<var>_<k>`
    sums:  [k] -> sum over days t-k+1 .. t          `<var>_sum<k>`
    means: [k] -> mean over days t-k+1 .. t         `<var>_mean<k>`
    calendar: doy_sin / doy_cos of the issue date (period 365.25 d)

Causality: inputs only use `shift(k)` with k >= 0 and trailing `rolling(k, min_periods=k)`
windows. The target is looked up by date (t + lead), never by shifting the inputs.

Usage (prints the sample counts per basin):
    python -m src.features.build_features --config configs/cp2.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

OPS = ("lags", "diffs", "sums", "means")
CALENDAR = ("doy_sin", "doy_cos")


# ----------------------------------------------------------------------------- config / io
def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(path: str | Path) -> tuple[dict, dict]:
    """Return (cp2 config, basins config). The basins config is read as plain YAML:
    CP2 only needs the basin list and the meteo variable names, not the LamaH folder."""
    cfg = load_yaml(path)
    basins_cfg = load_yaml(cfg["basins_config"])
    return cfg, basins_cfg


def load_basin_table(processed_dir: str | Path, basin_id: int) -> pd.DataFrame:
    path = Path(processed_dir) / f"basin_{basin_id}.csv"
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{path}: duplicate dates")
    # Row shifts must equal day shifts: enforce an explicit daily calendar.
    full = pd.date_range(df.index.min(), df.index.max(), freq="D", name="date")
    if len(full) != len(df):
        df = df.reindex(full)
    return df


# ----------------------------------------------------------------------------- calendar
def hydro_year(dates, start_month: int) -> np.ndarray:
    """HY label: year + 1 if month >= start_month else year (HY1982 = 1981-10-01 .. 1982-09-30)."""
    d = pd.DatetimeIndex(dates)
    return np.where(d.month >= start_month, d.year + 1, d.year).astype(int)


def hy_bounds(hy_first: int, hy_last: int, start_month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last calendar day of the HY range [hy_first, hy_last]."""
    first = pd.Timestamp(year=hy_first - 1, month=start_month, day=1)
    last = pd.Timestamp(year=hy_last, month=start_month, day=1) - pd.Timedelta(days=1)
    return first, last


# ----------------------------------------------------------------------------- features
def _variables(key: str, df: pd.DataFrame, meteo_vars: list[str]) -> list[str]:
    q_up = sorted(c for c in df.columns if c.startswith("q_up_"))
    if key == "q_upstream":
        if not q_up:
            raise ValueError("representation asks for q_upstream but the table has no q_up_* column")
        return q_up
    if key == "all_dynamic":
        return ["q_target", *q_up, *meteo_vars]
    return [key]


def build_features(df: pd.DataFrame, rep_spec: dict, meteo_vars: list[str]) -> pd.DataFrame:
    """Feature table indexed by issue date t, built from the representation spec."""
    cols: dict[str, pd.Series] = {}

    def add(name: str, s: pd.Series) -> None:
        if name in cols:
            raise ValueError(f"duplicate feature name {name}")
        cols[name] = s

    for key, spec in rep_spec.items():
        if key == "calendar":
            unknown = set(spec) - set(CALENDAR)
            if unknown:
                raise ValueError(f"unknown calendar features {sorted(unknown)}")
            doy = df.index.dayofyear.to_numpy()
            ang = 2 * np.pi * doy / 365.25
            if "doy_sin" in spec:
                add("doy_sin", pd.Series(np.sin(ang), index=df.index))
            if "doy_cos" in spec:
                add("doy_cos", pd.Series(np.cos(ang), index=df.index))
            continue
        unknown = set(spec) - set(OPS)
        if unknown:
            raise ValueError(f"{key}: unknown operations {sorted(unknown)}")
        for var in _variables(key, df, meteo_vars):
            if var not in df.columns:
                raise KeyError(f"column {var!r} (from representation key {key!r}) not in the table")
            s = df[var]
            for k in spec.get("lags", []):
                _check_k(k, 0, var)
                add(f"{var}_lag{k}", s.shift(k))
            for k in spec.get("diffs", []):
                _check_k(k, 1, var)
                add(f"d{var}_{k}", s - s.shift(k))
            for k in spec.get("sums", []):
                _check_k(k, 1, var)
                add(f"{var}_sum{k}", s.rolling(k, min_periods=k).sum())
            for k in spec.get("means", []):
                _check_k(k, 1, var)
                add(f"{var}_mean{k}", s.rolling(k, min_periods=k).mean())
    return pd.DataFrame(cols, index=df.index)


def _check_k(k, kmin: int, var: str) -> None:
    if not isinstance(k, int) or k < kmin:
        raise ValueError(f"{var}: window/lag {k!r} must be an integer >= {kmin}")


# ----------------------------------------------------------------------------- samples
@dataclass
class SampleSet:
    """Samples of one basin. `meta` and every `X[rep]` share the same index (issue date t)."""
    basin_id: int
    meta: pd.DataFrame                # target_date, target_hy, y, q_issue
    X: dict[str, pd.DataFrame]
    counts: dict = field(default_factory=dict)


def build_samples(df: pd.DataFrame, cfg: dict, meteo_vars: list[str], basin_id: int = -1,
                  representations: list[str] | None = None, allow_holdout: bool = False) -> SampleSet:
    """Build the sample table of one basin.

    A sample is kept only if its target date lies in a development HY (or a hold-out HY
    when `allow_holdout=True`, reserved for CP4), is not in the warm-up period, and has
    no NaN in the target or in the features of ANY configured representation. Using one
    common sample set makes every model (baselines, RF-A, RF-B) score on identical days.
    """
    lead = int(cfg["lead_days"])
    start_month = int(cfg["hydro_year_start_month"])
    reps = list(cfg["representations"]) if representations is None else representations
    # The validity mask always uses every configured representation, so `--quick` runs
    # (representation A only) score on the same days as the full run.
    feats_all = {r: build_features(df, cfg["representations"][r], meteo_vars)
                 for r in cfg["representations"]}

    issue = df.index
    target_date = issue + pd.Timedelta(days=lead)
    y = df["q_target"].reindex(target_date).to_numpy()
    t_hy = hydro_year(target_date, start_month)

    dev0, dev1 = cfg["development_hy"]
    ho0, ho1 = cfg["holdout_hy"]
    w0, w1 = pd.Timestamp(cfg["warmup"]["start"]), pd.Timestamp(cfg["warmup"]["end"])
    in_warmup = (target_date >= w0) & (target_date <= w1)
    is_dev = (t_hy >= dev0) & (t_hy <= dev1) & ~in_warmup
    is_ho = (t_hy >= ho0) & (t_hy <= ho1) & ~in_warmup
    in_period = is_dev | is_ho                       # excludes warm-up, HY1981 and HY2018
    keep_period = is_dev | (is_ho & allow_holdout)

    finite = np.isfinite(y)
    for f in feats_all.values():
        finite &= f.notna().all(axis=1).to_numpy()
    keep = keep_period & finite

    counts = {
        "basin_id": basin_id,
        "n_issue_days": int(len(df)),
        "n_target_outside_period": int((~in_period).sum()),   # warm-up, HY1981, HY2018, beyond record
        "n_target_in_warmup": int(in_warmup.sum()),
        "n_holdout_excluded": int((is_ho & ~allow_holdout).sum()),
        "n_dropped_nan_dev": int((is_dev & ~finite).sum()),
        "n_dropped_nan_holdout": int((is_ho & ~finite).sum()) if allow_holdout else 0,
        "n_samples": int(keep.sum()),
        "n_samples_dev": int((keep & is_dev).sum()),
    }
    meta = pd.DataFrame({
        "target_date": target_date[keep],
        "target_hy": t_hy[keep],
        "y": y[keep],
        "q_issue": df["q_target"].to_numpy()[keep],      # persistence forecast q_target(t)
    }, index=issue[keep])
    meta.index.name = "issue_date"
    X = {r: feats_all[r].loc[keep] for r in reps}
    return SampleSet(basin_id=basin_id, meta=meta, X=X, counts=counts)


# ----------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/cp2.yaml")
    args = ap.parse_args()
    cfg, bcfg = load_configs(args.config)
    for b in bcfg["basins"]:
        df = load_basin_table(cfg["processed_dir"], b["id"])
        s = build_samples(df, cfg, bcfg["meteo_vars"], basin_id=b["id"])
        shapes = {r: x.shape[1] for r, x in s.X.items()}
        print(f"basin {b['id']}: {s.counts} | n_features {shapes}")


if __name__ == "__main__":
    main()
