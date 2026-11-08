"""Readers for the LamaH-CE daily files.

Facts used here come from the LamaH-CE documentation (Klingler et al., 2021, ESSD;
Info/2_Timeseries.txt):
  * all files are semicolon-separated;
  * all time series are in UTC (no daylight saving); a daily value aggregates
    00:00-23:59 UTC;
  * discharge is in m3/s; remaining runoff gaps are coded as -999;
  * runoff files carry the per-time-step flags ckhs (checked by the hydrological
    service), qceq (value constant for >= 10 steps) and qcol (outlier);
  * meteorological files have no gaps.

Column names that the documentation does not state explicitly (the discharge
column, the date columns) are detected and validated at read time instead of
being assumed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MISSING_CODE = -999
DATE_COLS = ("YYYY", "MM", "DD")
RUNOFF_FLAG_COLS = ("ckhs", "qceq", "qcol")
# Columns that are calendar helpers, never data.
CALENDAR_COLS = set(DATE_COLS) | {"DOY", "HOD", "hh"}

RUNOFF_DIR = Path("D_gauges") / "2_timeseries" / "daily"
METEO_DIRS = {
    "A": Path("A_basins_total_upstrm") / "2_timeseries" / "daily",
    "B": Path("B_basins_intermediate_all") / "2_timeseries" / "daily",
    "C": Path("C_basins_intermediate_lowimp") / "2_timeseries" / "daily",
}
ATTRIBUTE_FILES = {
    "gauge": Path("D_gauges") / "1_attributes" / "Gauge_attributes.csv",
    "hydro": Path("D_gauges") / "1_attributes" / "Hydro_indices_1981_2017.csv",
    "catchment_A": Path("A_basins_total_upstrm") / "1_attributes" / "Catchment_attributes.csv",
    "hierarchy_B": Path("B_basins_intermediate_all") / "1_attributes" / "Gauge_hierarchy.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Check `lamah_root` in the config and that the "
            "daily LamaH-CE files were unzipped."
        )
    df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_daily_index(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [c for c in DATE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name}: expected date columns {DATE_COLS}, found {list(df.columns)}"
        )
    idx = pd.to_datetime(
        pd.DataFrame({"year": df["YYYY"], "month": df["MM"], "day": df["DD"]})
    )
    out = df.drop(columns=[c for c in df.columns if c in CALENDAR_COLS])
    out.index = pd.DatetimeIndex(idx, name="date")
    if not out.index.is_unique:
        raise ValueError(f"{path.name}: duplicated dates - is this really a daily file?")
    return out.sort_index()


def runoff_path(root: str | Path, gauge_id: int) -> Path:
    return Path(root) / RUNOFF_DIR / f"ID_{gauge_id}.csv"


def meteo_path(root: str | Path, basin_id: int, delineation: str = "A") -> Path:
    return Path(root) / METEO_DIRS[delineation] / f"ID_{basin_id}.csv"


def read_runoff(root: str | Path, gauge_id: int) -> pd.DataFrame:
    """Daily discharge of one gauge.

    Returns columns: q [m3/s, NaN where coded -999] plus whichever of
    ckhs / qceq / qcol are present in the file.
    """
    path = runoff_path(root, gauge_id)
    df = _to_daily_index(_read_csv(path), path)
    value_cols = [c for c in df.columns if c not in RUNOFF_FLAG_COLS]
    if len(value_cols) != 1:
        raise ValueError(
            f"{path.name}: cannot identify a single discharge column among "
            f"{list(df.columns)}; set it explicitly in lamah_io.read_runoff."
        )
    raw = pd.to_numeric(df[value_cols[0]], errors="coerce")
    n_other_negative = int(((raw < 0) & (raw != MISSING_CODE)).sum())
    if n_other_negative:
        raise ValueError(
            f"{path.name}: {n_other_negative} negative values that are not the -999 gap code"
        )
    out = pd.DataFrame({"q": raw.mask(raw == MISSING_CODE)}, index=df.index)
    for flag in RUNOFF_FLAG_COLS:
        if flag in df.columns:
            out[flag] = df[flag]
    out.attrs["discharge_column"] = value_cols[0]
    out.attrs["path"] = str(path)
    return out


def read_meteo(
    root: str | Path, basin_id: int, variables: list[str], delineation: str = "A"
) -> pd.DataFrame:
    """Daily ERA5-Land forcings aggregated over one basin polygon."""
    path = meteo_path(root, basin_id, delineation)
    df = _to_daily_index(_read_csv(path), path)
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise KeyError(
            f"{path.name}: variables {missing} not found. Available: {list(df.columns)}. "
            "Fix `meteo_vars` in the config."
        )
    return df[variables].apply(pd.to_numeric, errors="coerce")


def read_attributes(root: str | Path) -> pd.DataFrame:
    """Static gauge + catchment attributes, one row per ID (files that are absent are skipped)."""
    frames = []
    for key, rel in ATTRIBUTE_FILES.items():
        path = Path(root) / rel
        if not path.exists():
            print(f"[warn] attribute file not found, skipped: {path}")
            continue
        frames.append(_read_csv(path).set_index("ID"))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f[[c for c in f.columns if c not in out.columns]], how="outer")
    return out
