"""Step 2: ingest, align and quality-report the CP1 observation layer.

For every basin in the config this script
  1. reads the target discharge, the upstream discharges and the ERA5-Land forcings;
  2. aligns them on one explicit daily UTC calendar (missing days become NaN,
     never silently dropped or interpolated);
  3. writes one tidy table per basin to data/processed/basin_<id>.csv;
  4. writes the numbers the data card needs to results/cp1/:
       basin_summary.csv  - one row per basin (record, gaps, Q99, events)
       gap_report.csv     - one row per series (coverage, longest gap, QC flags)
       events_Q99.csv     - every independent exceedance event of the target
     and the CP1 figures (see src/data/figures.py).

Nothing here creates model features or targets: the lead-time shift (t -> t+2)
and the training-period threshold are defined in CP2, so this layer cannot leak.

Usage:
    python -m src.data.build_dataset --config configs/basins.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import load_config
from src.data.figures import plot_record_overview, plot_regimes
from src.data.lamah_io import read_attributes, read_meteo, read_runoff


# ----------------------------------------------------------------------------- build
def build_basin(root: str, basin: dict, meteo_vars: list[str], start: str, end: str) -> pd.DataFrame:
    target = read_runoff(root, basin["id"])
    cols = {"q_target": target["q"]}
    for flag in ("ckhs", "qceq", "qcol"):
        if flag in target:
            cols[f"target_{flag}"] = target[flag]
    for up in basin["upstream"]:
        cols[f"q_up_{up}"] = read_runoff(root, up)["q"]
    df = pd.DataFrame(cols)
    df = df.join(read_meteo(root, basin["id"], meteo_vars), how="outer")

    calendar = pd.date_range(start, end, freq="D", name="date")
    outside = df.index.difference(calendar)
    df = df.reindex(calendar)  # explicit daily calendar: absent days -> NaN
    df.attrs["n_rows_outside_period"] = len(outside)
    return df


# ----------------------------------------------------------------------------- QC
def longest_nan_run(s: pd.Series) -> int:
    isna = s.isna().to_numpy()
    if not isna.any():
        return 0
    edges = np.diff(np.concatenate([[0], isna.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return int((ends - starts).max())


def gap_report(df: pd.DataFrame, basin_id: int) -> pd.DataFrame:
    rows = []
    series_cols = [c for c in df.columns if not c.startswith("target_")]
    for col in series_cols:
        s = df[col]
        row = {
            "basin_id": basin_id,
            "series": col,
            "first_valid": s.first_valid_index().date() if s.notna().any() else None,
            "last_valid": s.last_valid_index().date() if s.notna().any() else None,
            "n_days": len(s),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 3),
            "longest_gap_days": longest_nan_run(s),
        }
        if col == "q_target":
            for flag in ("ckhs", "qceq", "qcol"):
                f = df.get(f"target_{flag}")
                if f is not None:
                    key = "pct_checked" if flag == "ckhs" else f"n_{flag}_flagged"
                    row[key] = round(100 * (f == 1).mean(), 2) if flag == "ckhs" else int((f == 1).sum())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- events
def exceedance_events(q: pd.Series, threshold: float, merge_gap_days: int) -> pd.DataFrame:
    """Independent events: runs of days with q > threshold, merging runs that are
    separated by <= merge_gap_days days below the threshold."""
    above = q > threshold
    days = q.index[above.fillna(False)]
    if len(days) == 0:
        return pd.DataFrame(columns=["start", "end", "duration_days", "peak_date", "peak_q"])
    groups, current = [], [days[0]]
    for d in days[1:]:
        if (d - current[-1]).days <= merge_gap_days + 1:
            current.append(d)
        else:
            groups.append(current)
            current = [d]
    groups.append(current)
    rows = []
    for g in groups:
        seg = q.loc[g[0]: g[-1]]
        rows.append({
            "start": g[0].date(),
            "end": g[-1].date(),
            "duration_days": (g[-1] - g[0]).days + 1,
            "peak_date": seg.idxmax().date(),
            "peak_q": round(float(seg.max()), 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/basins.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg["lamah_root"]
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    proc_dir = Path(cfg["output"]["processed_dir"])
    res_dir = Path(cfg["output"]["results_dir"])
    proc_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    attrs = read_attributes(root)
    summaries, gaps, events_all, tables = [], [], [], {}

    for b in cfg["basins"]:
        bid = b["id"]
        print(f"--- basin {bid} ({b['name']})")
        df = build_basin(root, b, cfg["meteo_vars"], start, end)
        df.to_csv(proc_dir / f"basin_{bid}.csv", float_format="%.4f")
        tables[bid] = df

        q = df["q_target"]
        thr = float(q.quantile(cfg["event_quantile"]))
        ev = exceedance_events(q, thr, cfg["event_merge_gap_days"])
        ev.insert(0, "basin_id", bid)
        events_all.append(ev)
        gaps.append(gap_report(df, bid))

        n_years = q.notna().sum() / 365.25
        a = attrs.loc[bid] if bid in attrs.index else pd.Series(dtype=object)
        up_area = sum(attrs.loc[u, "area_gov"] for u in b["upstream"] if u in attrs.index) \
            if "area_gov" in attrs else np.nan
        peak_months = pd.to_datetime(ev["peak_date"]).dt.month if len(ev) else pd.Series(dtype=int)
        summaries.append({
            "basin_id": bid,
            "name": b["name"],
            "regime": b["regime"],
            "river": a.get("river"),
            "area_gov_km2": a.get("area_gov"),
            "elev_mean_m": a.get("elev_mean"),
            "glac_fra": a.get("glac_fra"),
            "frac_snow": a.get("frac_snow"),
            "hfd_mean_days": a.get("hfd_mean"),
            "degimpact": a.get("degimpact"),
            "upstream_ids": ",".join(map(str, b["upstream"])),
            "upstream_area_share": round(up_area / a["area_gov"], 3) if "area_gov" in a else np.nan,
            "record_start": q.first_valid_index().date(),
            "record_end": q.last_valid_index().date(),
            "n_days_valid": int(q.notna().sum()),
            "pct_missing_target": round(100 * q.isna().mean(), 3),
            "q_mean_m3s": round(float(q.mean()), 3),
            f"q{int(cfg['event_quantile']*100)}_m3s": round(thr, 2),
            "q_max_m3s": round(float(q.max()), 2),
            "n_events": len(ev),
            "events_per_year": round(len(ev) / n_years, 2),
            "pct_events_MJJA": round(100 * peak_months.isin([5, 6, 7, 8]).mean(), 1) if len(ev) else np.nan,
            "discharge_column_in_file": read_runoff(root, bid).attrs["discharge_column"],
        })
        print(f"    {int(q.notna().sum())} valid days, Q99 = {thr:.2f} m3/s, {len(ev)} events")

    summary = pd.DataFrame(summaries)
    summary.to_csv(res_dir / "basin_summary.csv", index=False)
    pd.concat(gaps).to_csv(res_dir / "gap_report.csv", index=False)
    pd.concat(events_all).to_csv(res_dir / "events_Q99.csv", index=False)

    plot_record_overview(tables, summary, cfg, res_dir)
    plot_regimes(tables, summary, res_dir)

    pd.set_option("display.width", 200)
    print("\n" + summary.drop(columns=["discharge_column_in_file"]).to_string(index=False))
    print(f"\nOutputs written to {proc_dir}/ and {res_dir}/")


if __name__ == "__main__":
    main()
