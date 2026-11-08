"""CP2 entry point: samples -> folds -> baselines (persistence, climatology, RF) -> metrics.

Usage:
    python -m src.baselines.run_cp2 --config configs/cp2.yaml            # main run (blocked folds)
    python -m src.baselines.run_cp2 --config configs/cp2.yaml --folds loyo   # RF robustness check
    python -m src.baselines.run_cp2 --config configs/cp2.yaml --quick    # 1 seed, repr. A, -> <results_dir>/quick/

Main-run outputs (results_dir, default results/cp2/):
    folds.csv, predictions.csv.gz, metrics_by_fold.csv, metrics_pooled.csv, table_cp2_main.csv / .md,
    run_info.json, SUMMARY.md, fig_cp2_protocol.png/.pdf, fig_cp2_skill.png/.pdf,
    isel_qceq_days.csv
LOYO-run outputs: table_cp2_loyo.csv, folds_loyo.csv, run_info_loyo.json (SUMMARY.md is refreshed).

Hold-out hydrological years are never loaded into the sample tables here (CP4 only).
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.models import Climatology, fit_predict_rf, fold_thresholds, persistence_predict
from src.evaluation import metrics as M
from src.evaluation.folds import fold_blocks, make_folds
from src.features.build_features import build_samples, hydro_year, load_basin_table, load_configs

RF = "random_forest"
NO_REP = "none"
NO_SEED = -1
QC_CHECK_BASIN = 533      # COULD 1: Isel QC check (target_qceq == 1 days)
DISPLAY = {("persistence", NO_REP): "Persistence", ("climatology", NO_REP): "Climatology",
           (RF, "A_engineered"): "RF-A (engineered)", (RF, "B_raw_lags"): "RF-B (raw lags)"}


def display_name(model: str, rep: str) -> str:
    return DISPLAY.get((model, rep), f"{model} {rep}" if rep != NO_REP else model)


class Timer:
    def __init__(self):
        self.steps: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        self.steps[name] = round(self.steps.get(name, 0.0) + seconds, 2)


# ----------------------------------------------------------------------------- fit / predict
def run_basin(samples, folds, cfg: dict, reps: list[str], seeds: list[int], timer: Timer,
              with_baselines: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    meta = samples.meta
    quantiles = cfg["threshold_quantiles"]
    thr_names = list(quantiles)
    rf_cfg = cfg["baselines"]["random_forest"]
    clim_w = cfg["baselines"]["climatology"]["smooth_window_days"]
    preds, fold_rows = [], []

    for fold in folds:
        tr, te = fold.train, fold.test
        y_tr = meta["y"].to_numpy()[tr]
        thr = fold_thresholds(y_tr, quantiles)
        fold_rows.append({"scheme": fold.scheme, "fold": fold.fold_id, "test_hy_first": fold.test_hy[0],
                          "test_hy_last": fold.test_hy[1], "test_first_day": fold.first_day.date(),
                          "test_last_day": fold.last_day.date(), "basin_id": samples.basin_id,
                          "n_train": int(tr.sum()), "n_test": int(te.sum()), "n_purged": int(fold.purged.sum()),
                          **{f"thr_{k}": v for k, v in thr.items()}})
        base = pd.DataFrame({
            "basin_id": samples.basin_id, "fold": fold.fold_id,
            "issue_date": meta.index[te], "target_date": meta["target_date"].to_numpy()[te],
            "target_hy": meta["target_hy"].to_numpy()[te], "y_obs": meta["y"].to_numpy()[te],
            **{f"thr_{k}": thr[k] for k in thr_names},
        })

        def emit(model, rep, seed, y_pred):
            d = base.copy()
            d.insert(1, "model", model)
            d.insert(2, "representation", rep)
            d.insert(3, "seed", seed)
            d["y_pred"] = y_pred
            preds.append(d)

        if with_baselines:
            t0 = time.perf_counter()
            emit("persistence", NO_REP, NO_SEED, persistence_predict(meta["q_issue"].to_numpy()[te]))
            clim = Climatology(clim_w).fit(meta["target_date"].to_numpy()[tr], y_tr)
            emit("climatology", NO_REP, NO_SEED, clim.predict(meta["target_date"].to_numpy()[te]))
            timer.add("baselines", time.perf_counter() - t0)

        for rep in reps:
            X = samples.X[rep].to_numpy()
            for seed in seeds:
                t0 = time.perf_counter()
                emit(RF, rep, seed, fit_predict_rf(rf_cfg, seed, X[tr], y_tr, X[te]))
                timer.add(f"rf_{rep}", time.perf_counter() - t0)

    cols = ["basin_id", "model", "representation", "seed", "fold", "issue_date", "target_date",
            "target_hy", "y_obs", "y_pred", *[f"thr_{k}" for k in thr_names]]
    return pd.concat(preds, ignore_index=True)[cols], fold_rows


# ----------------------------------------------------------------------------- evaluation
def evaluate(pred: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pooled metrics + HY-bootstrap draws per basin x model x representation x seed,
    and the same metrics per fold. Returns (pooled rows, by-fold rows, draws)."""
    thr_names = list(cfg["threshold_quantiles"])
    dev_hy = list(range(cfg["development_hy"][0], cfg["development_hy"][1] + 1))
    bs = cfg["metrics"]["bootstrap"]
    W = M.bootstrap_weights(len(dev_hy), int(bs["n_resamples"]), int(bs["seed"]))  # shared: paired draws
    names_out = M.metric_names(thr_names)
    pooled, by_fold, draws = [], [], {}

    for key, g in pred.groupby(["basin_id", "model", "representation", "seed"], sort=False):
        if g["target_date"].duplicated().any():
            raise AssertionError(f"{key}: a target date appears in more than one test fold")
        thr = {t: g[f"thr_{t}"].to_numpy() for t in thr_names}
        Gs, names = M.group_stats(g["y_obs"], g["y_pred"], thr, g["target_hy"], dev_hy)
        point = M.metrics_from_stats(Gs.sum(0), names, thr_names)
        d = M.bootstrap_metrics(Gs, names, thr_names, W)
        draws[key] = {m: d[m] for m in names_out}
        pooled.append({**dict(zip(["basin_id", "model", "representation", "seed"], key)),
                       "n_days": len(g), **{k: float(v) for k, v in point.items()}})
        for f, gf in g.groupby("fold"):
            thr_f = {t: gf[f"thr_{t}"].to_numpy() for t in thr_names}
            Sf, _ = M.group_stats(gf["y_obs"], gf["y_pred"], thr_f, gf["target_hy"], dev_hy)
            pf = M.metrics_from_stats(Sf.sum(0), names, thr_names)
            by_fold.append({**dict(zip(["basin_id", "model", "representation", "seed"], key)),
                            "fold": f, "test_hy_first": int(gf["target_hy"].min()),
                            "test_hy_last": int(gf["target_hy"].max()), "n_days": len(gf),
                            **{k: float(v) for k, v in pf.items()}})
    return pd.DataFrame(pooled), pd.DataFrame(by_fold), draws


def main_table(pooled: pd.DataFrame, draws: dict, cfg: dict, basin_names: dict) -> pd.DataFrame:
    """One row per basin x model (RF: per representation). RF point = mean over seeds;
    interval = percentiles of the bootstrap draws pooled over seeds; plus min-max over seeds."""
    thr_names = list(cfg["threshold_quantiles"])
    ci = cfg["metrics"]["bootstrap"]["ci"]
    names = M.metric_names(thr_names)
    rows = []
    for (b, model, rep), g in pooled.groupby(["basin_id", "model", "representation"], sort=False):
        seeds = list(g["seed"])
        row = {"basin_id": b, "basin_name": basin_names.get(b, ""), "model": model, "representation": rep,
               "label": display_name(model, rep), "n_seeds": len(seeds) if model == RF else 0,
               "n_days": int(g["n_days"].iloc[0])}
        for m in names:
            pooled_draws = np.concatenate([draws[(b, model, rep, s)][m] for s in seeds])
            row[m] = float(g[m].mean())
            row[f"{m}_lo"], row[f"{m}_hi"] = M.percentile_interval(pooled_draws, ci)
        for t in thr_names:
            row[f"n_exceed_days_{t}"] = int(g[f"n_exceed_days_{t}"].iloc[0])
        for m in names:
            row[f"{m}_seedmin"] = float(g[m].min()) if model == RF else np.nan
            row[f"{m}_seedmax"] = float(g[m].max()) if model == RF else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    # Column order of the brief first, extras (KGE/RMSE intervals, seed ranges) after.
    lead = ["basin_id", "basin_name", "model", "representation", "label", "n_seeds", "n_days",
            "NSE", "NSE_lo", "NSE_hi", "KGE", "RMSE"]
    for t in thr_names:
        for m in M.ALERT_METRICS:
            lead += [f"{m}_{t}", f"{m}_{t}_lo", f"{m}_{t}_hi"]
        lead += [f"n_exceed_days_{t}"]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


# ----------------------------------------------------------------------------- reporting
def _fmt(v, lo=None, hi=None, nd=2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    s = f"{v:.{nd}f}"
    if lo is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


def table_markdown(tab: pd.DataFrame, cfg: dict) -> str:
    thr_names = list(cfg["threshold_quantiles"])
    ci = cfg["metrics"]["bootstrap"]["ci"]
    pct = int(round(100 * (ci[1] - ci[0])))
    out = [f"Values: pooled over the {len(cfg['folds']['blocks'])} blocked folds "
           f"(development HY{cfg['development_hy'][0]}-{cfg['development_hy'][1]}), "
           f"[{pct} % HY-bootstrap interval]. RF: mean of {len(cfg['baselines']['random_forest']['seeds'])} seeds.",
           "", "**Continuous skill** (RMSE in m3/s)", "",
           "| Basin | Model | NSE | KGE | RMSE |", "|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        out.append(f"| {r.basin_id} | {r.label} | {_fmt(r.NSE, r.NSE_lo, r.NSE_hi)} | {_fmt(r.KGE)} | {_fmt(r.RMSE)} |")
    for t in thr_names:
        q = cfg["threshold_quantiles"][t]
        out += ["", f"**Alerts at {t}** (threshold = training quantile {q} of each fold; day-based)", "",
                "| Basin | Model | POD | FAR | CSI | obs. days |", "|---|---|---|---|---|---|"]
        for _, r in tab.iterrows():
            cells = [_fmt(r[f"{m}_{t}"], r[f"{m}_{t}_lo"], r[f"{m}_{t}_hi"]) for m in M.ALERT_METRICS]
            out.append(f"| {r.basin_id} | {r.label} | " + " | ".join(cells) + f" | {int(r[f'n_exceed_days_{t}'])} |")
    out += ["", "n/a = no alert issued, so FAR is undefined. obs. days = observed exceedance days "
            "(each fold's own threshold)."]
    return "\n".join(out) + "\n"


def observations(tab: pd.DataFrame, cfg: dict, loyo: pd.DataFrame | None) -> list[str]:
    """3-5 factual statements read off the table (no interpretation)."""
    thr_names = list(cfg["threshold_quantiles"])
    n_hy = cfg["development_hy"][1] - cfg["development_hy"][0] + 1
    obs = []
    best = []
    def top(g, col):
        """Labels tied for the best value at the 2 decimals shown in the tables."""
        r = g[col].round(2)
        return " = ".join(g.label[r == r.max()]) + f" {r.max():.2f}"

    for b, g in tab.groupby("basin_id", sort=False):
        best.append(f"{b} {top(g, 'NSE')}")
    obs.append("Highest NSE per basin (ties at 2 decimals joined by '='): " + "; ".join(best) + ".")
    parts = []
    for t in thr_names:
        per = []
        for b, g in tab.groupby("basin_id", sort=False):
            per.append(f"{b} {top(g.dropna(subset=[f'CSI_{t}']), f'CSI_{t}')}")
        parts.append(f"CSI@{t}: " + "; ".join(per))
    silent = tab[tab[[f"FAR_{t}" for t in thr_names]].isna().all(axis=1)]
    line = "Highest CSI per basin - " + " | ".join(parts) + "."
    if len(silent):
        names = sorted(set(silent.label))
        where = "all basins" if silent.basin_id.nunique() == tab.basin_id.nunique() else \
            "basins " + ", ".join(str(x) for x in silent.basin_id.unique())
        line += (f" {', '.join(names)} issues no alert at any threshold in {where} "
                 "(POD = CSI = 0, FAR undefined).")
    obs.append(line)
    t = thr_names[-1]
    sub = tab.drop(silent.index)
    w = sub[f"CSI_{t}_hi"] - sub[f"CSI_{t}_lo"]
    n = tab[f"n_exceed_days_{t}"]
    obs.append(f"CSI@{t} 90 % intervals span {w.min():.2f}-{w.max():.2f} (persistence and RF); "
               f"{int(n.min())}-{int(n.max())} observed {t} exceedance days per basin over {n_hy} HY.")
    rf = tab[tab.model == RF]
    reps = list(dict.fromkeys(rf.representation))
    if len(reps) == 2:
        a, bb = (rf[rf.representation == r].set_index("basin_id") for r in reps)
        d = (a["NSE"] - bb["NSE"])
        s = f"{display_name(RF, reps[0])} minus {display_name(RF, reps[1])}: NSE " + ", ".join(f"{k} {v:+.3f}" for k, v in d.items())
        for tt in thr_names:
            dc = (a[f"CSI_{tt}"] - bb[f"CSI_{tt}"])
            s += f"; CSI@{tt} " + ", ".join(f"{k} {v:+.2f}" for k, v in dc.items())
        obs.append(s + ".")
    if loyo is not None and len(loyo):
        r = loyo[loyo.model == RF].pivot(index="basin_id", columns="folds", values="NSE")
        if {"blocked", "loyo"} <= set(r.columns):
            obs.append("RF-A (seed 0) NSE, blocked vs LOYO folds (`table_cp2_loyo.csv`): "
                       + ", ".join(f"{k} {v.blocked:.3f} vs {v.loyo:.3f}" for k, v in r.iterrows()) + ".")
    return obs


def write_summary(res: Path, cfg: dict) -> None:
    tab = pd.read_csv(res / "table_cp2_main.csv")
    loyo = pd.read_csv(res / "table_cp2_loyo.csv") if (res / "table_cp2_loyo.csv").exists() else None
    thr_names = list(cfg["threshold_quantiles"])
    f = cfg["folds"]
    rf = cfg["baselines"]["random_forest"]
    blocks = ", ".join(f"{a}-{b}" for a, b in f["blocks"])
    protocol = (
        f"**Protocol.** One forecast per issue date t (end of day t, inputs <= t), target = mean discharge "
        f"of day t+{cfg['lead_days']}; samples belong to the hydrological year (Oct-Sep) of the target. "
        f"Development HY{cfg['development_hy'][0]}-{cfg['development_hy'][1]} is split into "
        f"{len(f['blocks'])} blocked leave-years-out folds ({blocks}); training targets within "
        f"{f['purge_days']} days of the test block are purged. Hold-out HY{cfg['holdout_hy'][0]}-"
        f"{cfg['holdout_hy'][1]} is not used. Alert thresholds ({', '.join(thr_names)}), climatology and RF "
        f"are fitted on each fold's training targets only. Test predictions of all folds are pooled; "
        f"[intervals] are 90 % percentile intervals from {cfg['metrics']['bootstrap']['n_resamples']} "
        f"bootstrap resamples of whole HY. RF: {rf['n_estimators']} trees, min_samples_leaf "
        f"{rf['min_samples_leaf']}, max_features {rf['max_features']}, mean of seeds {rf['seeds']}, no tuning.")
    labels = list(dict.fromkeys(tab.label))
    lines = ["# CP2 summary - baselines on the development period", "", protocol, "",
             "| Basin | Metric | " + " | ".join(labels) + " |", "|---|---|" + "---|" * len(labels)]
    for b, g in tab.groupby("basin_id", sort=False):
        g = g.set_index("label")
        for m in ["NSE"] + [f"CSI_{t}" for t in thr_names]:
            cells = [_fmt(g.at[l, m], g.at[l, f"{m}_lo"], g.at[l, f"{m}_hi"]) if l in g.index else "-" for l in labels]
            lines.append(f"| {b} | {m.replace('_', '@')} | " + " | ".join(cells) + " |")
    lines += ["", "**Observations** (full table: `table_cp2_main.md`)", ""]
    lines += [f"- {o}" for o in observations(tab, cfg, loyo)]
    (res / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_info() -> dict:
    def run(*args):
        try:
            r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:  # noqa: BLE001 - git missing is not fatal
            return None
    status = run("status", "--porcelain")
    # git_dirty: true when `git status --porcelain` lists anything (modified or untracked files).
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_dirty": None if status is None else bool(status)}


def package_versions() -> dict:
    import matplotlib
    import sklearn
    import yaml
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__, "platform": platform.platform()}


def qceq_check(df: pd.DataFrame, fold_rows: list[dict], cfg: dict) -> pd.DataFrame:
    """COULD 1: list the target_qceq == 1 days with their run length and, for development
    days, whether q_target exceeds the Q95 of the training years of the fold that tests it."""
    if "target_qceq" not in df:
        return pd.DataFrame()
    flag = df["target_qceq"].fillna(0).to_numpy() == 1
    run_id = np.cumsum(np.r_[True, flag[1:] != flag[:-1]])
    run_len = pd.Series(run_id).map(pd.Series(run_id[flag]).value_counts()).to_numpy()
    d = df.index[flag]
    hy = hydro_year(d, cfg["hydro_year_start_month"])
    out = pd.DataFrame({"date": d.date, "q_target": df["q_target"].to_numpy()[flag],
                        "run_length": run_len[flag].astype(int), "hy": hy})
    dev0, dev1 = cfg["development_hy"]
    ho0, ho1 = cfg["holdout_hy"]
    out["period"] = np.select([(hy >= dev0) & (hy <= dev1), (hy >= ho0) & (hy <= ho1)],
                              ["development", "holdout"], "outside")
    fr = pd.DataFrame(fold_rows)
    thr = np.full(len(out), np.nan)
    for _, r in fr.iterrows():
        m = (out["period"] == "development") & (hy >= r.test_hy_first) & (hy <= r.test_hy_last)
        thr[m.to_numpy()] = r["thr_Q95"] if "thr_Q95" in r else np.nan
    out["thr_Q95_train_fold"] = thr
    out["above_thr_Q95"] = np.where(np.isfinite(thr), out["q_target"] > thr, np.nan)
    return out


# ----------------------------------------------------------------------------- modes
def prepare(cfg, bcfg, timer):
    data, samples = {}, {}
    t0 = time.perf_counter()
    for b in bcfg["basins"]:
        df = load_basin_table(cfg["processed_dir"], b["id"])
        data[b["id"]] = df
        samples[b["id"]] = build_samples(df, cfg, bcfg["meteo_vars"], basin_id=b["id"])
        c = samples[b["id"]].counts
        print(f"  basin {b['id']}: {c['n_samples_dev']} development samples, "
              f"{c['n_dropped_nan_dev']} dropped for NaN, {c['n_holdout_excluded']} hold-out excluded, "
              f"{c['n_target_outside_period']} targets outside the period | features "
              + ", ".join(f"{r}: {x.shape[1]}" for r, x in samples[b["id"]].X.items()))
    timer.add("features", time.perf_counter() - t0)
    return data, samples


def run_main(cfg, bcfg, res: Path, quick: bool, timer: Timer, t_start: float) -> None:
    rf_cfg = cfg["baselines"]["random_forest"]
    reps = rf_cfg["representations"][:1] if quick else list(rf_cfg["representations"])
    seeds = rf_cfg["seeds"][:1] if quick else list(rf_cfg["seeds"])
    scheme = cfg["folds"]["scheme"]
    names = {b["id"]: b["name"] for b in bcfg["basins"]}
    print(f"[1/5] features ({', '.join(cfg['representations'])})")
    data, samples = prepare(cfg, bcfg, timer)

    print(f"[2/5] folds ({scheme}) and models: RF {reps} x seeds {seeds}")
    preds, fold_rows = [], []
    for bid, s in samples.items():
        t0 = time.perf_counter()
        folds = make_folds(s.meta, cfg, scheme)
        p, fr = run_basin(s, folds, cfg, reps, seeds, timer)
        preds.append(p)
        fold_rows += fr
        print(f"  basin {bid}: {len(folds)} folds, {len(p)} predictions in {time.perf_counter() - t0:.1f} s")
    pred = pd.concat(preds, ignore_index=True)
    folds_df = pd.DataFrame(fold_rows)
    for (bid, model, rep, seed), g in pred.groupby(["basin_id", "model", "representation", "seed"]):
        assert len(g) == samples[bid].counts["n_samples_dev"], "each development day must be tested once"

    print("[3/5] metrics + HY bootstrap")
    t0 = time.perf_counter()
    pooled, by_fold, draws = evaluate(pred, cfg)
    tab = main_table(pooled, draws, cfg, names)
    timer.add("metrics_bootstrap", time.perf_counter() - t0)

    print("[4/5] writing tables")
    t0 = time.perf_counter()
    folds_df.to_csv(res / "folds.csv", index=False, float_format="%.4f")
    out_pred = pred.copy()
    out_pred["seed"] = out_pred["seed"].astype("Int64").where(out_pred["seed"] != NO_SEED)
    # Full float precision: the tables must be reproducible from this file exactly
    # (rounded thresholds/predictions could flip a day across the threshold).
    out_pred.to_csv(res / "predictions.csv.gz", index=False, date_format="%Y-%m-%d", compression="gzip")
    mp = pooled.copy()
    mp["seed"] = mp["seed"].astype("Int64").where(mp["seed"] != NO_SEED)
    mp.to_csv(res / "metrics_pooled.csv", index=False)   # pooled metrics per seed (no intervals)
    bf = by_fold.copy()
    bf["seed"] = bf["seed"].astype("Int64").where(bf["seed"] != NO_SEED)
    bf.to_csv(res / "metrics_by_fold.csv", index=False, float_format="%.4f")
    tab.to_csv(res / "table_cp2_main.csv", index=False)  # full precision: md and SUMMARY round the same values
    (res / "table_cp2_main.md").write_text("# CP2 main table\n\n" + table_markdown(tab, cfg), encoding="utf-8")
    if QC_CHECK_BASIN in data:
        qc = qceq_check(data[QC_CHECK_BASIN], [r for r in fold_rows if r["basin_id"] == QC_CHECK_BASIN], cfg)
        if len(qc):
            qc.to_csv(res / "isel_qceq_days.csv", index=False, float_format="%.4f")
    timer.add("write_tables", time.perf_counter() - t0)

    print("[5/5] figures")
    t0 = time.perf_counter()
    from src.evaluation.figures_cp2 import plot_protocol, plot_skill
    plot_protocol(cfg, res)
    plot_skill(tab, cfg, res)
    timer.add("figures", time.perf_counter() - t0)
    write_summary(res, cfg)

    info = {"mode": "quick" if quick else "main", "folds_scheme": scheme, **git_info(),
            "config": cfg, "basins": bcfg["basins"], "meteo_vars": bcfg["meteo_vars"],
            "packages": package_versions(), "rf_seeds": seeds, "representations_run": reps,
            "bootstrap_seed": cfg["metrics"]["bootstrap"]["seed"],
            "samples_per_basin": {str(k): v.counts for k, v in samples.items()},
            "runtime_s": {**timer.steps, "total": round(time.perf_counter() - t_start, 2)}}
    (res / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    size = (res / "predictions.csv.gz").stat().st_size / 1e6
    print(f"  predictions.csv.gz: {size:.1f} MB")


def run_loyo(cfg, bcfg, res: Path, timer: Timer, t_start: float) -> None:
    """SHOULD 2: RF-A seed 0 (plus persistence and climatology) on blocked vs pure LOYO folds."""
    rf_cfg = cfg["baselines"]["random_forest"]
    rep, seed = rf_cfg["representations"][0], rf_cfg["seeds"][0]
    print(f"[1/3] features")
    _, samples = prepare(cfg, bcfg, timer)
    ci = cfg["metrics"]["bootstrap"]["ci"]
    thr_names = list(cfg["threshold_quantiles"])
    rows, all_folds = [], []
    for scheme in ("blocked", "loyo"):
        print(f"[2/3] {scheme}: RF {rep} seed {seed} + baselines")
        t0 = time.perf_counter()
        preds = []
        for bid, s in samples.items():
            folds = make_folds(s.meta, cfg, scheme)
            p, fr = run_basin(s, folds, cfg, [rep], [seed], timer)
            preds.append(p)
            all_folds += fr
        pred = pd.concat(preds, ignore_index=True)
        pooled, _, draws = evaluate(pred, cfg)
        for _, r in pooled.iterrows():
            key = (r.basin_id, r.model, r.representation, r.seed)
            row = {"basin_id": r.basin_id, "model": r.model, "representation": r.representation,
                   "label": display_name(r.model, r.representation), "seed": r.seed if r.model == RF else None,
                   "folds": scheme, "n_folds": len(fold_blocks(cfg, scheme)), "n_days": r.n_days,
                   "NSE": r.NSE}
            row["NSE_lo"], row["NSE_hi"] = M.percentile_interval(draws[key]["NSE"], ci)
            for t in thr_names:
                row[f"CSI_{t}"] = r[f"CSI_{t}"]
                row[f"CSI_{t}_lo"], row[f"CSI_{t}_hi"] = M.percentile_interval(draws[key][f"CSI_{t}"], ci)
            rows.append(row)
        dt = time.perf_counter() - t0
        timer.add(f"{scheme}_total", dt)
        print(f"  {scheme}: {dt:.1f} s")
    print("[3/3] writing")
    out = pd.DataFrame(rows).sort_values(["basin_id", "model", "folds"], kind="stable")
    out.to_csv(res / "table_cp2_loyo.csv", index=False, float_format="%.4f")
    pd.DataFrame([r for r in all_folds if r["scheme"] == "loyo"]).to_csv(
        res / "folds_loyo.csv", index=False, float_format="%.4f")
    if (res / "table_cp2_main.csv").exists():
        write_summary(res, cfg)
    info = {"mode": "loyo", **git_info(), "packages": package_versions(),
            "representation": rep, "seed": seed,
            "runtime_s": {**timer.steps, "total": round(time.perf_counter() - t_start, 2)}}
    (res / "run_info_loyo.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="CP2: baselines and evaluation protocol")
    ap.add_argument("--config", default="configs/cp2.yaml")
    ap.add_argument("--folds", choices=["blocked", "loyo"], default=None,
                    help="fold scheme (default: folds.scheme from the config); loyo = RF robustness check")
    ap.add_argument("--quick", action="store_true", help="1 seed, representation A only (debugging)")
    args = ap.parse_args()
    t_start = time.perf_counter()
    cfg, bcfg = load_configs(args.config)
    cfg_snapshot = copy.deepcopy(cfg)
    res = Path(cfg["results_dir"])
    if args.quick:
        res = res / "quick"
    res.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    scheme = args.folds or cfg["folds"]["scheme"]
    if scheme == "loyo":
        run_loyo(cfg_snapshot, bcfg, res, timer, t_start)
    else:
        run_main(cfg_snapshot, bcfg, res, args.quick, timer, t_start)
    total = time.perf_counter() - t_start
    print("  runtime per step [s]: " + ", ".join(f"{k} {v:.1f}" for k, v in timer.steps.items()))
    print(f"Done in {total:.1f} s ({total / 60:.1f} min). Outputs in {res}/")


if __name__ == "__main__":
    main()
