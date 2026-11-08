"""CP2 baselines: persistence, climatology, Random Forest, and the per-fold thresholds.

All fitted quantities (climatology curve, thresholds, RF) are computed from the training
samples of a fold only; the caller passes training arrays in and never test targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# ----------------------------------------------------------------------------- persistence
def persistence_predict(q_issue) -> np.ndarray:
    """y_hat(t + lead) = q_target(t), i.e. the feature `q_target_lag0`."""
    return np.asarray(q_issue, float).copy()


# ----------------------------------------------------------------------------- climatology
def doy_365(dates) -> np.ndarray:
    """Day of year with 29 Feb (and doy 366) mapped to 365."""
    return np.minimum(pd.DatetimeIndex(dates).dayofyear.to_numpy(), 365)


class Climatology:
    """Mean training target by target day-of-year, smoothed by a circular moving mean."""

    def __init__(self, smooth_window_days: int):
        w = int(smooth_window_days)
        if w < 1 or w % 2 == 0:
            raise ValueError("smooth_window_days must be a positive odd integer")
        self.window = w
        self.curve_: np.ndarray | None = None

    def fit(self, target_dates, y) -> "Climatology":
        doy = doy_365(target_dates)
        y = np.asarray(y, float)
        s = np.bincount(doy - 1, weights=y, minlength=365)
        n = np.bincount(doy - 1, minlength=365).astype(float)
        # Circular moving mean of the day-of-year means (days without data are skipped).
        h = self.window // 2
        mean = np.where(n > 0, s / np.where(n > 0, n, 1), 0.0)
        has = (n > 0).astype(float)
        ext_m = np.concatenate([mean[-h:], mean, mean[:h]]) if h else mean
        ext_h = np.concatenate([has[-h:], has, has[:h]]) if h else has
        k = np.ones(self.window)
        num = np.convolve(ext_m, k, mode="valid")
        den = np.convolve(ext_h, k, mode="valid")
        if (den == 0).any():
            raise ValueError("climatology: a smoothing window has no training data")
        self.curve_ = num / den
        return self

    def predict(self, target_dates) -> np.ndarray:
        return self.curve_[doy_365(target_dates) - 1]


# ----------------------------------------------------------------------------- random forest
def make_rf(rf_cfg: dict, seed: int) -> RandomForestRegressor:
    """Fixed a-priori hyperparameters from the config (no tuning at CP2)."""
    return RandomForestRegressor(
        n_estimators=int(rf_cfg["n_estimators"]),
        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
        max_features=rf_cfg["max_features"],
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=int(seed),
    )


def fit_predict_rf(rf_cfg: dict, seed: int, X_train, y_train, X_test) -> np.ndarray:
    """Fit in parallel, predict single-threaded: the parallel predict sums the trees in
    thread order, which changes the last bits between identical runs."""
    rf = make_rf(rf_cfg, seed).fit(X_train, y_train)
    rf.set_params(n_jobs=1)
    return rf.predict(X_test)


# ----------------------------------------------------------------------------- thresholds
def fold_thresholds(y_train, quantiles: dict[str, float]) -> dict[str, float]:
    """Alert thresholds of a fold: quantiles of the TRAINING targets only."""
    y_train = np.asarray(y_train, float)
    return {name: float(np.quantile(y_train, q)) for name, q in quantiles.items()}
