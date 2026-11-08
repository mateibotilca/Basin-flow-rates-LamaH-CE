"""CP2 leakage and protocol tests on synthetic data (run: `pytest -q`)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baselines.models import Climatology, doy_365, fit_predict_rf, fold_thresholds, persistence_predict
from src.baselines.run_cp2 import Timer, run_basin
from src.evaluation import metrics as M
from src.evaluation.folds import HoldoutAccessError, holdout_samples, make_folds
from src.features.build_features import build_features, build_samples, hydro_year, load_yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = load_yaml(ROOT / "configs" / "cp2.yaml")
METEO = load_yaml(ROOT / CFG["basins_config"])["meteo_vars"]
UP = ["q_up_11", "q_up_12"]
LEAD = int(CFG["lead_days"])
PURGE = pd.Timedelta(days=int(CFG["folds"]["purge_days"]))
SM = int(CFG["hydro_year_start_month"])


def synthetic_basin(seed: int = 0, with_nan: bool = False) -> pd.DataFrame:
    """Daily table with the processed-data columns, 1981-01-01 .. 2017-12-31."""
    rng = np.random.default_rng(seed)
    d = pd.date_range("1981-01-01", "2017-12-31", freq="D", name="date")
    n = len(d)
    season = 10 + 6 * np.sin(2 * np.pi * (d.dayofyear.to_numpy() - 100) / 365.25)

    def ar(scale):
        e = rng.gamma(0.4, scale, n)
        x = np.empty(n)
        x[0] = e[0]
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + e[i]
        return x

    df = pd.DataFrame(index=d)
    df["q_target"] = season + ar(4)
    df["target_ckhs"], df["target_qceq"], df["target_qcol"] = 1.0, 0.0, 0.0
    for c in UP:
        df[c] = 0.5 * season + ar(2)
    df["prec"] = rng.gamma(0.6, 5, n)
    t = 8 + 10 * np.sin(2 * np.pi * (d.dayofyear.to_numpy() - 110) / 365.25) + rng.normal(0, 2, n)
    df["2m_temp_mean"], df["2m_temp_min"], df["2m_temp_max"] = t, t - 5, t + 5
    df["swe"] = np.clip(-t, 0, None) * 10
    df["volsw_123"] = 0.3 + 0.05 * rng.random(n)
    if with_nan:
        idx = rng.choice(np.arange(400, n - 400), 40, replace=False)
        df.iloc[idx, df.columns.get_loc("q_target")] = np.nan
        df.iloc[5000:5012, df.columns.get_loc("q_up_11")] = np.nan
    return df[["q_target", "target_ckhs", "target_qceq", "target_qcol", *UP, *METEO]]


INPUTS = ["q_target", *UP, *METEO]


@pytest.fixture(scope="module")
def df():
    return synthetic_basin()


@pytest.fixture(scope="module")
def samples(df):
    return build_samples(df, CFG, METEO, basin_id=1)


# ----------------------------------------------------------------------------- 1 causality
@pytest.mark.parametrize("rep", list(CFG["representations"]))
def test_feature_causality(df, rep):
    rng = np.random.default_rng(42)
    spec = CFG["representations"][rep]
    base = build_features(df, spec, METEO)
    for t in rng.choice(df.index[100:-100], 8, replace=False):
        noisy = df.copy()
        future = noisy.index > t
        noisy.loc[future, INPUTS] = rng.normal(0, 100, (future.sum(), len(INPUTS)))
        f2 = build_features(noisy, spec, METEO)
        pd.testing.assert_series_equal(base.loc[t], f2.loc[t], check_names=False)
        # negative control: changing day t itself must change the features at t
        noisy.loc[t, INPUTS] = noisy.loc[t, INPUTS] + 1000.0
        assert not base.loc[t].equals(build_features(noisy, spec, METEO).loc[t])


def test_feature_names_follow_config(df):
    a = build_features(df, CFG["representations"]["A_engineered"], METEO)
    for name in ["q_target_lag0", "q_target_lag6", "dq_target_1", "q_up_11_lag2", "prec_sum30",
                 "2m_temp_mean_mean7", "dswe_3", "volsw_123_lag0", "doy_sin", "doy_cos"]:
        assert name in a.columns, name
    b = build_features(df, CFG["representations"]["B_raw_lags"], METEO)
    assert b.shape[1] == 7 * (1 + len(UP) + len(METEO)) + 2
    t = df.index[400]
    assert a.at[t, "prec_sum3"] == pytest.approx(df["prec"].loc[t - pd.Timedelta(days=2):t].sum())
    assert a.at[t, "dswe_3"] == pytest.approx(df.at[t, "swe"] - df.at[t - pd.Timedelta(days=3), "swe"])


# ----------------------------------------------------------------------------- 2 target alignment
def test_target_alignment(df, samples):
    m = samples.meta
    assert (m["target_date"] == m.index + pd.Timedelta(days=LEAD)).all()
    np.testing.assert_array_equal(m["y"].to_numpy(), df["q_target"].reindex(m["target_date"]).to_numpy())
    np.testing.assert_array_equal(m["target_hy"].to_numpy(), hydro_year(m["target_date"], SM))
    assert hydro_year(pd.to_datetime(["1981-10-01", "1982-09-30", "1982-10-01"]), SM).tolist() == [1982, 1982, 1983]


# ----------------------------------------------------------------------------- 3 folds
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("scheme", ["blocked", "loyo"])
def test_folds(scheme, with_nan):
    d = synthetic_basin(seed=3, with_nan=with_nan)
    s = build_samples(d, CFG, METEO)
    m = s.meta
    if with_nan:
        assert s.counts["n_dropped_nan_dev"] > 0
    dev0, dev1 = CFG["development_hy"]
    # no hold-out, warm-up or HY2018 target anywhere
    assert m["target_hy"].between(dev0, dev1).all()
    assert (m["target_date"] > pd.Timestamp(CFG["warmup"]["end"])).all()
    folds = make_folds(m, CFG, scheme)
    tested = np.zeros(len(m), int)
    for f in folds:
        tr, te = set(m["target_date"][f.train]), set(m["target_date"][f.test])
        assert tr and te and not (tr & te)
        td = pd.DatetimeIndex(m["target_date"][f.train])
        dist = np.minimum(np.abs((td - f.first_day).days), np.abs((td - f.last_day).days))
        assert dist.min() > PURGE.days, "training target within the purge window"
        assert not (f.train & f.purged).any() and not (f.test & f.purged).any()
        tested += f.test
    assert (tested == 1).all(), "every development target must be tested exactly once"


# ----------------------------------------------------------------------------- 4 train-only fits
def test_thresholds_and_climatology_use_training_years_only(df):
    """Set every target of a test block to 1e6 -> the fold's thresholds and climatology must not
    move. Checked through the pipeline code (run_basin), not only the helper functions."""
    s0 = build_samples(df, CFG, METEO, basin_id=1)
    folds0 = make_folds(s0.meta, CFG)
    w = CFG["baselines"]["climatology"]["smooth_window_days"]
    q = CFG["threshold_quantiles"]
    pred0, rows0 = run_basin(s0, folds0, CFG, [], [], Timer())
    clim0 = pred0[pred0.model == "climatology"].set_index("target_date")["y_pred"]
    for k, f in enumerate(folds0):
        bumped = df.copy()
        bumped.loc[f.first_day:f.last_day, "q_target"] = 1e6
        s1 = build_samples(bumped, CFG, METEO, basin_id=1)
        folds1 = make_folds(s1.meta, CFG)
        f1 = folds1[k]
        assert (s1.meta["y"][f1.test] == 1e6).all()
        # helper level
        y0, y1 = s0.meta["y"][f.train], s1.meta["y"][f1.train]
        assert fold_thresholds(y0, q) == fold_thresholds(y1, q)
        c0 = Climatology(w).fit(s0.meta["target_date"][f.train], y0)
        c1 = Climatology(w).fit(s1.meta["target_date"][f1.train], y1)
        np.testing.assert_array_equal(c0.curve_, c1.curve_)
        # pipeline level: thresholds written to folds.csv and climatology predictions of this fold
        pred1, rows1 = run_basin(s1, folds1, CFG, [], [], Timer())
        for t in q:
            assert rows1[k][f"thr_{t}"] == rows0[k][f"thr_{t}"], (k, t)
        c = pred1[(pred1.model == "climatology") & (pred1.fold == k)].set_index("target_date")["y_pred"]
        pd.testing.assert_series_equal(c, clim0.loc[c.index])


def test_climatology_doy_and_smoothing():
    assert doy_365(pd.to_datetime(["2000-12-31", "2001-12-31", "2000-02-29"])).tolist() == [365, 365, 60]
    d = pd.date_range("1990-01-01", "1999-12-31", freq="D")
    c = Climatology(31).fit(d, np.ones(len(d)) * 5.0)
    np.testing.assert_allclose(c.curve_, 5.0)
    with pytest.raises(ValueError):
        Climatology(30)


# ----------------------------------------------------------------------------- 5 baselines
def test_persistence_equals_lag0(samples):
    np.testing.assert_array_equal(persistence_predict(samples.meta["q_issue"]),
                                  samples.X["A_engineered"]["q_target_lag0"].to_numpy())


def test_rf_deterministic(samples):
    rf_cfg = {**CFG["baselines"]["random_forest"], "n_estimators": 15}
    X, y = samples.X["A_engineered"].to_numpy()[:3000], samples.meta["y"].to_numpy()[:3000]
    p1 = fit_predict_rf(rf_cfg, 0, X, y, X[:500])
    p2 = fit_predict_rf(rf_cfg, 0, X, y, X[:500])
    p3 = fit_predict_rf(rf_cfg, 1, X, y, X[:500])
    np.testing.assert_array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


# ----------------------------------------------------------------------------- 6 hold-out guard
def test_holdout_guard(df, samples):
    assert samples.counts["n_holdout_excluded"] > 0
    with pytest.raises(HoldoutAccessError):
        holdout_samples(samples.meta, CFG)
    s_ho = build_samples(df, CFG, METEO, allow_holdout=True)
    with pytest.raises(HoldoutAccessError):
        make_folds(s_ho.meta, CFG)
    with pytest.raises(HoldoutAccessError):
        holdout_samples(s_ho.meta, CFG)
    ho = holdout_samples(s_ho.meta, CFG, allow_holdout=True)
    assert len(ho) and ho["target_hy"].between(*CFG["holdout_hy"]).all()
    folds = make_folds(s_ho.meta, CFG, allow_holdout=True)      # CP4 path: still never trained/tested
    is_ho = s_ho.meta["target_hy"].between(*CFG["holdout_hy"]).to_numpy()
    for f in folds:
        assert not (f.train & is_ho).any() and not (f.test & is_ho).any()


# ----------------------------------------------------------------------------- metrics
def test_metrics_stats_match_direct_forms():
    rng = np.random.default_rng(0)
    y = rng.gamma(2, 5, 3000)
    p = y + rng.normal(0, 3, 3000)
    groups = np.repeat(np.arange(10), 300)
    thr = {"Q95": np.quantile(y, 0.95)}
    S, names = M.group_stats(y, p, thr, groups, list(range(10)))
    out = M.metrics_from_stats(S.sum(0), names, ["Q95"])
    assert out["NSE"] == pytest.approx(M.nse(y, p))
    assert out["KGE"] == pytest.approx(M.kge(y, p))
    assert out["RMSE"] == pytest.approx(M.rmse(y, p))
    c = M.contingency(y, p, thr["Q95"])
    sc = M.alert_scores(c["hits"], c["misses"], c["false_alarms"])
    for k in ("POD", "FAR", "CSI"):
        assert out[f"{k}_Q95"] == pytest.approx(float(sc[k]))
    # no alerts -> FAR is NaN, not 0
    s0 = M.alert_scores(0, 5, 0)
    assert s0["POD"] == 0 and np.isnan(s0["FAR"]) and s0["CSI"] == 0
    # identity weights reproduce the pooled value
    W = np.ones((1, 10))
    assert M.bootstrap_metrics(S, names, ["Q95"], W)["NSE"][0] == pytest.approx(out["NSE"])
    Wb = M.bootstrap_weights(10, 200, 0)
    assert (Wb.sum(1) == 10).all()
