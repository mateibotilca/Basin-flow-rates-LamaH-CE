"""Smoke test: build a small fake LamaH-CE folder and run the full CP1 pipeline on it.

It checks that your Python environment works before you point the config at the
real data. The fake files imitate the documented format (semicolon separator,
-999 gap code, ckhs/qceq/qcol flags); the real files are validated separately by
`python -m src.data.inspect_files`.

Usage:
    python -m tests.smoke_test
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASINS = {533: [530, 532], 332: [328, 331], 195: [193, 194]}
METEO = ["prec", "2m_temp_mean", "2m_temp_min", "2m_temp_max", "swe", "volsw_123"]


def _calendar(start="1981-01-01", end="2017-12-31") -> pd.DataFrame:
    d = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"YYYY": d.year, "MM": d.month, "DD": d.day}), d


def make_fake_lamah(root: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    cal, d = _calendar()
    doy = d.dayofyear.to_numpy()
    run_dir = root / "D_gauges" / "2_timeseries" / "daily"
    met_dir = root / "A_basins_total_upstrm" / "2_timeseries" / "daily"
    run_dir.mkdir(parents=True)
    met_dir.mkdir(parents=True)

    ids = sorted({i for b, ups in BASINS.items() for i in [b, *ups]})
    for gid in ids:
        base = 10 + 8 * np.sin(2 * np.pi * (doy - 90) / 365.25)
        spikes = rng.gamma(0.3, 15, len(d))
        q = np.round(np.convolve(base + spikes, np.ones(3) / 3, mode="same"), 3)
        q[np.arange(40) * 100 + 7] = -999          # 40 isolated one-day gaps
        q[5000:5012] = -999                              # one 12-day gap
        df = cal.copy()
        df["qobs"] = q
        df["ckhs"], df["qceq"], df["qcol"] = 1, 0, 0
        df.to_csv(run_dir / f"ID_{gid}.csv", sep=";", index=False)

    for bid in BASINS:
        df = cal.copy()
        df["DOY"] = doy
        df["prec"] = np.round(rng.gamma(0.6, 5, len(d)), 2)
        t = 8 + 10 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        df["2m_temp_mean"], df["2m_temp_min"], df["2m_temp_max"] = t, t - 5, t + 5
        df["swe"] = np.clip(-t, 0, None) * 10
        df["volsw_123"] = 0.3
        df.to_csv(met_dir / f"ID_{bid}.csv", sep=";", index=False)

    attr = root / "D_gauges" / "1_attributes"
    attr.mkdir(parents=True)
    pd.DataFrame({"ID": ids, "river": "fake", "area_gov": [500 if i in BASINS else 150 for i in ids],
                  "degimpact": "l"}).to_csv(attr / "Gauge_attributes.csv", sep=";", index=False)


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        make_fake_lamah(tmp / "LamaH-CE")
        cfg = yaml.safe_load(open("configs/basins.yaml", encoding="utf-8"))
        cfg["lamah_root"] = str(tmp / "LamaH-CE")
        cfg["output"] = {"processed_dir": str(tmp / "processed"), "results_dir": str(tmp / "results")}
        cfg_path = tmp / "cfg.yaml"
        yaml.safe_dump(cfg, open(cfg_path, "w", encoding="utf-8"))
        r = subprocess.run([sys.executable, "-m", "src.data.build_dataset", "--config", str(cfg_path)],
                           capture_output=True, text=True)
        print(r.stdout[-1500:])
        if r.returncode != 0:
            print(r.stderr)
            raise SystemExit("SMOKE TEST FAILED")
        res = tmp / "results"
        expected = ["basin_summary.csv", "gap_report.csv", "events_Q99.csv",
                    "fig_cp1_record_overview.png", "fig_cp1_regimes.png"]
        missing = [f for f in expected if not (res / f).exists()]
        if missing:
            raise SystemExit(f"SMOKE TEST FAILED, missing outputs: {missing}")
        gaps = pd.read_csv(res / "gap_report.csv")
        tgt = gaps[(gaps.series == "q_target")]
        assert (tgt.n_missing == 52).all(), tgt  # 40 isolated + 12 consecutive
        assert (tgt.longest_gap_days == 12).all(), tgt
        print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
