"""Reproducible basin selection for CP1 (documents how configs/basins.yaml was chosen).

Filters, applied to all LamaH-CE gauges:
  * country == AUT (HZB data are quality-checked for the full record);
  * degimpact in {u, l} (no or low human influence);
  * daily record starts in 1981 and is still running (obsend == 0);
  * gaps_post <= 1 permille after LamaH's own interpolation;
  * HIERARCHY >= 2 and every upstream gauge (NEXTUPID) passes the same record filters;
  * upstream gauges cover 30-90 % of the target area (upstream signal matters,
    but there is still an intermediate catchment to forecast);
  * 150 km2 <= area_gov <= 5000 km2 (daily resolution is meaningful).
Regime label (heuristic, documented in the data card):
  glac_fra > 0.03 or elev_mean >= 1800 m -> alpine; >= 900 m -> mountain;
  < 600 m -> lowland; otherwise hill. hfd_mean is reported as the independent check.

Usage:
    python -m src.data.select_basins --config configs/basins.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.config import load_config
from src.data.lamah_io import read_attributes


def regime(r: pd.Series) -> str:
    if r.glac_fra > 0.03 or r.elev_mean >= 1800:
        return "alpine (nival-glacial)"
    if r.elev_mean >= 900:
        return "mountain (nivo-pluvial)"
    if r.elev_mean < 600:
        return "lowland (pluvial)"
    return "hill (pluvio-nival)"


def record_ok(a: pd.DataFrame) -> pd.Series:
    return (a.obsbeg_day <= 1981) & (a.obsend == 0) & (a.gaps_post <= 1) & (a.degimpact != "x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/basins.yaml")
    cfg = load_config(ap.parse_args().config)
    a = read_attributes(cfg["lamah_root"])
    ok = record_ok(a)

    rows = []
    for gid, r in a.iterrows():
        if str(r.NEXTUPID) in ("0", "nan"):
            continue
        ups = [int(x) for x in str(r.NEXTUPID).split(",")]
        ups = [u for u in ups if u in a.index]
        rows.append({
            "ID": gid,
            "upstream_ids": ",".join(map(str, ups)),
            "upstream_degimpact": "".join(a.loc[ups, "degimpact"]),
            "upstream_ok": bool(ok.loc[ups].all()),
            "upstream_area_share": a.loc[ups, "area_gov"].sum() / r.area_gov,
        })
    up = pd.DataFrame(rows).set_index("ID")
    df = a.join(up, how="inner")
    cand = df[(df.country == "AUT") & df.degimpact.isin(["u", "l"]) & ok.loc[df.index]
              & (df.HIERARCHY >= 2) & df.upstream_ok
              & df.upstream_area_share.between(0.3, 0.9) & df.area_gov.between(150, 5000)].copy()
    cand["regime"] = cand.apply(regime, axis=1)
    cols = ["name", "river", "fedstate", "area_gov", "elev_mean", "frac_snow", "glac_fra",
            "hfd_mean", "Q95", "typimpact", "degimpact", "upstream_ids", "upstream_degimpact",
            "upstream_area_share", "regime"]
    cand = cand.sort_values(["regime", "elev_mean"])[cols].round(3)
    out = Path(cfg["output"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    cand.to_csv(out / "basin_candidates.csv")
    pd.set_option("display.width", 250)
    print(f"{len(a)} gauges -> {len(cand)} candidates\n{cand.regime.value_counts()}\n")
    print(cand.to_string())


if __name__ == "__main__":
    main()
