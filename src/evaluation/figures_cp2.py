"""CP2 figures (300 dpi PNG + vector PDF, single-column paper width).

fig_cp2_protocol: timeline by hydrological year, one row per blocked fold
                  (train / purge / test), warm-up and hold-out marked.
fig_cp2_skill:    CSI at each alert threshold per basin and model, 90 % HY-bootstrap
                  intervals as error bars; persistence drawn as the reference.

Usage (re-draws both figures from results_dir/table_cp2_main.csv):
    python -m src.evaluation.figures_cp2 --config configs/cp2.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.figures import BASIN_COLORS, GRID, INK, INK_MUTED  # noqa: E402  (also sets rcParams)

TRAIN, TEST, PURGE = BASIN_COLORS[0], BASIN_COLORS[1], INK
HOLDOUT, WARMUP = "#bdbcb7", "#e4e3df"
COL_W = 3.5                                   # single-column width [in]
PURGE_MIN_W = 0.35                            # purge drawn at least this wide [HY] for visibility


def _save(fig, res: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(res / f"{name}.{ext}")
    plt.close(fig)


def _x(date: pd.Timestamp, start_month: int) -> float:
    """Position on an axis where HY h spans [h-1, h)."""
    hy = date.year + 1 if date.month >= start_month else date.year
    first = pd.Timestamp(year=hy - 1, month=start_month, day=1)
    nxt = pd.Timestamp(year=hy, month=start_month, day=1)
    return hy - 1 + (date - first).days / (nxt - first).days


def plot_protocol(cfg: dict, res: Path) -> None:
    sm = int(cfg["hydro_year_start_month"])
    dev0, dev1 = cfg["development_hy"]
    ho0, ho1 = cfg["holdout_hy"]
    blocks = cfg["folds"]["blocks"]
    purge_d = int(cfg["folds"]["purge_days"])
    purge_w = purge_d / 365.25
    draw_w = max(purge_w, PURGE_MIN_W)
    w0 = _x(pd.Timestamp(cfg["warmup"]["start"]), sm)
    w1 = _x(pd.Timestamp(cfg["warmup"]["end"]) + pd.Timedelta(days=1), sm)

    fig, ax = plt.subplots(figsize=(COL_W, 0.28 * len(blocks) + 1.0))
    h = 0.62
    for i, (a, b) in enumerate(blocks):
        yc = len(blocks) - 1 - i
        ax.broken_barh([(w0, w1 - w0)], (yc - h / 2, h), facecolor=WARMUP, edgecolor="none")
        ax.broken_barh([(dev0 - 1, dev1 - dev0 + 1)], (yc - h / 2, h), facecolor=TRAIN, edgecolor="none")
        ax.broken_barh([(a - 1, b - a + 1)], (yc - h / 2, h), facecolor=TEST, edgecolor="none")
        spans = []
        if a > dev0:
            spans.append((a - 1 - draw_w, draw_w))
        if b < dev1:
            spans.append((b, draw_w))
        if spans:
            ax.broken_barh(spans, (yc - h / 2, h), facecolor=PURGE, edgecolor="none")
        ax.broken_barh([(ho0 - 1, ho1 - ho0 + 1)], (yc - h / 2, h), facecolor=HOLDOUT,
                       edgecolor="white", hatch="////", linewidth=0)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"Fold {i + 1}" for i in range(len(blocks))][::-1])
    ax.set_xlim(w0 - 0.3, ho1 + 0.3)
    ax.set_ylim(-0.6, len(blocks) - 0.4)
    centres = [y for y in range(1985, ho1 + 1, 5)]
    ax.set_xticks([c - 0.5 for c in centres])
    ax.set_xticklabels([str(c) for c in centres])
    ax.set_xticks(np.arange(int(w0), ho1 + 1), minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    for s in ("left",):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Hydrological year (Oct-Sep)")
    top = len(blocks) - 0.4
    ax.text((w0 + w1) / 2, top + 0.05, "warm-up", ha="center", va="bottom", fontsize=6, color=INK_MUTED)
    ax.text((dev0 - 1 + dev1) / 2, top + 0.05, f"development HY{dev0}-{dev1}", ha="center", va="bottom",
            fontsize=6, color=INK)
    ax.text((ho0 - 1 + ho1) / 2, top + 0.05, f"hold-out HY{ho0}-{ho1}\n(untouched until CP4)",
            ha="center", va="bottom", fontsize=6, color=INK_MUTED)
    handles = [mpatches.Patch(color=TRAIN, label="train"),
               mpatches.Patch(color=TEST, label="test"),
               mpatches.Patch(color=PURGE, label=f"purge {purge_d} d (not to scale)"),
               mpatches.Patch(facecolor=HOLDOUT, hatch="////", edgecolor="white", label="hold-out"),
               mpatches.Patch(color=WARMUP, label="warm-up (inputs only)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=6,
              handlelength=1.2, columnspacing=1.0)
    _save(fig, res, "fig_cp2_protocol")


def plot_skill(tab: pd.DataFrame, cfg: dict, res: Path) -> None:
    thr_names = list(cfg["threshold_quantiles"])
    order = [("persistence", "none"), ("climatology", "none")] + \
            [("random_forest", r) for r in cfg["baselines"]["random_forest"]["representations"]]
    order = [o for o in order if ((tab.model == o[0]) & (tab.representation == o[1])).any()]
    colors = {("persistence", "none"): INK_MUTED, ("climatology", "none"): BASIN_COLORS[2]}
    rf_cols = [BASIN_COLORS[0], BASIN_COLORS[1]]
    for i, o in enumerate([o for o in order if o[0] == "random_forest"]):
        colors[o] = rf_cols[i % 2]
    markers = {("persistence", "none"): "s", ("climatology", "none"): "D"}
    basins = list(dict.fromkeys(tab["basin_id"]))
    fig, axes = plt.subplots(1, len(basins), figsize=(COL_W, 2.6), sharey=True)
    axes = np.atleast_1d(axes)
    off = np.linspace(-0.27, 0.27, len(order))
    for ax, b in zip(axes, basins):
        g = tab[tab.basin_id == b]
        for j, t in enumerate(thr_names):
            pers = g[(g.model == "persistence")][f"CSI_{t}"]
            if len(pers) and np.isfinite(pers.iloc[0]):
                ax.hlines(pers.iloc[0], j - 0.42, j + 0.42, color=INK_MUTED, linestyle="--", linewidth=0.7,
                          zorder=1)
            for k, o in enumerate(order):
                r = g[(g.model == o[0]) & (g.representation == o[1])]
                if not len(r):
                    continue
                r = r.iloc[0]
                v, lo, hi = r[f"CSI_{t}"], r[f"CSI_{t}_lo"], r[f"CSI_{t}_hi"]
                if not np.isfinite(v):
                    continue
                yerr = [[max(v - lo, 0)], [max(hi - v, 0)]] if np.isfinite(lo) else None
                ax.errorbar(j + off[k], v, yerr=yerr, fmt=markers.get(o, "o"), color=colors[o], ms=3.2,
                            elinewidth=0.8, capsize=1.5, zorder=3,
                            label=r["label"] if (b == basins[0] and j == 0) else None)
        ax.set_xticks(range(len(thr_names)))
        ax.set_xticklabels(thr_names)
        ax.set_xlim(-0.6, len(thr_names) - 0.4)
        ax.set_title(f"{b} {str(g['basin_name'].iloc[0]).split(' at ')[0]}", fontsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("CSI (-)")
    axes[0].set_ylim(bottom=0)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=0.7))
    l.append("persistence (reference)")
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6,
               handlelength=1.5, columnspacing=1.0)
    fig.text(0.5, 0.17, "Alert threshold (training quantile)", ha="center", va="center", fontsize=8)
    fig.subplots_adjust(bottom=0.3, wspace=0.12)
    _save(fig, res, "fig_cp2_skill")


def main() -> None:
    from src.features.build_features import load_yaml
    ap = argparse.ArgumentParser(description="CP2 figures")
    ap.add_argument("--config", default="configs/cp2.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = Path(cfg["results_dir"])
    plot_protocol(cfg, res)
    tab_path = res / "table_cp2_main.csv"
    if tab_path.exists():
        plot_skill(pd.read_csv(tab_path), cfg, res)
    print(f"Figures written to {res}/")


if __name__ == "__main__":
    main()
