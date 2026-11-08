"""CP2 metrics: NSE, KGE, RMSE and day-based alert scores (POD, FAR, CSI), plus the
hydrological-year bootstrap.

Every metric is computed from additive sufficient statistics (sums per group), so a
pooled metric, a per-fold metric and a bootstrap resample of whole hydrological years
all use the same code path:
    pooled metric   = f(sum of the per-HY statistics)
    bootstrap draw  = f(sum of the per-HY statistics weighted by how often each HY was drawn)

Definitions
    NSE  = 1 - sum((y - p)^2) / sum((y - mean(y))^2)
    KGE  = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)   (Gupta et al., 2009)
           r = Pearson correlation, alpha = sd(p) / sd(y), beta = mean(p) / mean(y)
    RMSE = sqrt(mean((y - p)^2))   [m3/s]
    alert = p > thr, observed = y > thr (thr = the fold's own training quantile)
    POD = hits / (hits + misses); FAR = false alarms / (hits + false alarms)  (ratio)
    CSI = hits / (hits + misses + false alarms). NaN when a denominator is 0.
"""
from __future__ import annotations

import numpy as np

CONT_STATS = ["n", "sy", "syy", "sp", "spp", "syp"]
CONT_METRICS = ["NSE", "KGE", "RMSE"]
ALERT_METRICS = ["POD", "FAR", "CSI"]
COUNTS = ["hits", "misses", "false_alarms", "correct_negatives"]


# ----------------------------------------------------------------------------- direct forms
def nse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def kge(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    r = np.corrcoef(y, p)[0, 1]
    alpha = p.std() / y.std()
    beta = p.mean() / y.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def contingency(y, p, thr) -> dict:
    """Day-based 2x2 table. `thr` is a scalar or one threshold per sample."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    thr = np.broadcast_to(np.asarray(thr, float), y.shape)
    obs, alert = y > thr, p > thr
    return {"hits": int((obs & alert).sum()), "misses": int((obs & ~alert).sum()),
            "false_alarms": int((~obs & alert).sum()), "correct_negatives": int((~obs & ~alert).sum())}


def _ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def alert_scores(hits, misses, false_alarms) -> dict:
    return {"POD": _ratio(hits, hits + misses),
            "FAR": _ratio(false_alarms, hits + false_alarms),
            "CSI": _ratio(hits, hits + misses + false_alarms)}


# ----------------------------------------------------------------------------- sufficient stats
def group_stats(y, p, thresholds: dict[str, np.ndarray], groups, group_order) -> tuple[np.ndarray, list[str]]:
    """Per-group additive statistics, shape (n_groups, n_stats); groups absent -> zeros."""
    y, p, groups = np.asarray(y, float), np.asarray(p, float), np.asarray(groups)
    pos = {g: i for i, g in enumerate(group_order)}
    gi = np.array([pos[g] for g in groups], dtype=int)
    G = len(group_order)
    cols = [np.ones_like(y), y, y * y, p, p * p, y * p]
    names = list(CONT_STATS)
    for tname, thr in thresholds.items():
        thr = np.broadcast_to(np.asarray(thr, float), y.shape)
        obs, alert = y > thr, p > thr
        cols += [obs & alert, obs & ~alert, ~obs & alert, ~obs & ~alert]
        names += [f"{c}_{tname}" for c in COUNTS]
    out = np.zeros((G, len(cols)))
    for j, c in enumerate(cols):
        out[:, j] = np.bincount(gi, weights=np.asarray(c, float), minlength=G)
    return out, names


def metrics_from_stats(S: np.ndarray, names: list[str], thr_names: list[str]) -> dict[str, np.ndarray]:
    """Metrics from summed statistics. S has shape (..., n_stats); returns arrays of shape (...)."""
    s = {n: S[..., i] for i, n in enumerate(names)}
    n = s["n"]
    with np.errstate(divide="ignore", invalid="ignore"):
        my, mp = s["sy"] / n, s["sp"] / n
        vy = np.maximum(s["syy"] / n - my ** 2, 0)
        vp = np.maximum(s["spp"] / n - mp ** 2, 0)
        cov = s["syp"] / n - my * mp
        sse = s["syy"] - 2 * s["syp"] + s["spp"]
        out = {"NSE": 1 - sse / (n * vy),
               "RMSE": np.sqrt(np.maximum(sse, 0) / n)}
        r = cov / np.sqrt(vy * vp)
        alpha = np.sqrt(vp / vy)
        beta = mp / my
        out["KGE"] = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    for t in thr_names:
        h, m, f = s[f"hits_{t}"], s[f"misses_{t}"], s[f"false_alarms_{t}"]
        for k, v in alert_scores(h, m, f).items():
            out[f"{k}_{t}"] = v
        out[f"n_exceed_days_{t}"] = h + m
        for c in COUNTS:
            out[f"{c}_{t}"] = s[f"{c}_{t}"]
    return out


def metric_names(thr_names: list[str]) -> list[str]:
    return CONT_METRICS + [f"{m}_{t}" for t in thr_names for m in ALERT_METRICS]


# ----------------------------------------------------------------------------- bootstrap
def bootstrap_weights(n_groups: int, n_resamples: int, seed: int) -> np.ndarray:
    """(n_resamples, n_groups) multiplicities: groups (HY) drawn with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    W = np.zeros((n_resamples, n_groups))
    np.add.at(W, (np.repeat(np.arange(n_resamples), n_groups), idx.ravel()), 1)
    return W


def bootstrap_metrics(G_stats: np.ndarray, names: list[str], thr_names: list[str],
                      W: np.ndarray) -> dict[str, np.ndarray]:
    """Bootstrap draws of every metric: W (B x G) @ per-group stats (G x K)."""
    return metrics_from_stats(W @ G_stats, names, thr_names)


def percentile_interval(draws, ci) -> tuple[float, float]:
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(d, ci)
    return float(lo), float(hi)
