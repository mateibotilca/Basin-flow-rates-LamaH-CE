"""CP1 figures (paper resolution: 300 dpi PNG + vector PDF)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed categorical order (validated colour-vision-deficiency-safe for <= 3 series);
# colour follows the basin, never its rank.
BASIN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _save(fig, res_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(res_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"    figure: {res_dir / name}.png/.pdf")


def plot_record_overview(tables: dict, summary: pd.DataFrame, cfg: dict, res_dir: Path) -> None:
    """Full daily record of each target gauge, its Q99 line and the days without data."""
    qcol = [c for c in summary.columns if c.startswith("q") and c.endswith("_m3s") and c[1:3].isdigit()][0]
    n = len(tables)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.9 * n), sharex=True,
                             gridspec_kw={"hspace": 0.45})
    axes = np.atleast_1d(axes)
    for i, (ax, (bid, df)) in enumerate(zip(axes, tables.items())):
        row = summary.set_index("basin_id").loc[bid]
        color = BASIN_COLORS[i % len(BASIN_COLORS)]
        q = df["q_target"]
        ax.plot(q.index, q.values, color=color, lw=0.4)
        ax.axhline(row[qcol], color=INK, lw=0.8, ls="--")
        ax.text(1.0, row[qcol], f" Q99 = {row[qcol]:.0f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", color=INK, fontsize=7)
        gaps = q.index[q.isna()]
        if len(gaps):
            ax.plot(gaps, np.zeros(len(gaps)), "|", color=INK_MUTED, ms=5, mew=0.8, clip_on=False,
                    label=f"no data ({len(gaps)} d)")
            ax.legend(loc="upper left", handlelength=0.8)
        ax.set_title(f"{bid} {row['name']} - {row['regime']}", loc="left", color=INK)
        ax.set_ylabel("Q (m$^3$ s$^{-1}$)")
        ax.set_ylim(bottom=0)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date (UTC)")
    fig.align_ylabels(axes)
    _save(fig, res_dir, "fig_cp1_record_overview")


def plot_regimes(tables: dict, summary: pd.DataFrame, res_dir: Path) -> None:
    """(a) Mean annual cycle of discharge, scaled by the mean; (b) month of Q99 event peaks."""
    events = pd.read_csv(res_dir / "events_Q99.csv", parse_dates=["peak_date"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.3})
    months = np.arange(1, 13)
    width = 0.8 / max(len(tables), 1)
    for i, (bid, df) in enumerate(tables.items()):
        name = summary.set_index("basin_id").loc[bid, "name"]
        color = BASIN_COLORS[i % len(BASIN_COLORS)]
        q = df["q_target"]
        monthly = q.groupby(q.index.month).mean() / q.mean()
        ax1.plot(months, monthly.reindex(months).values, color=color, lw=2, marker="o", ms=3,
                 label=f"{bid} {name}")
        ev = events[events.basin_id == bid]
        counts = ev["peak_date"].dt.month.value_counts().reindex(months, fill_value=0)
        ax2.bar(months + (i - (len(tables) - 1) / 2) * width, counts.values, width=width * 0.9,
                color=color, label=f"{bid}")
    ax1.axhline(1.0, color=INK_MUTED, lw=0.8, ls=":")
    ax1.set_ylabel("Monthly / annual mean Q (-)")
    ax1.set_title("(a) Flow regime", loc="left")
    ax2.set_ylabel("Q99 events (count)")
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.set_title("(b) Seasonality of exceedances", loc="left")
    for ax in (ax1, ax2):
        ax.set_xticks(months)
        ax.set_xticklabels(list("JFMAMJJASOND"))
        ax.set_xlabel("Month")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.2))
    _save(fig, res_dir, "fig_cp1_regimes")
