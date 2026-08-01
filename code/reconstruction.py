"""Redraw Plott and Sunder's Figures 2-6 from the digitized trades.

    python3 code/reconstruction.py     -> fig/reconstruction_m1..m5.png

One panel per market, in the original's own layout: transacted price against the
trade sequence, periods divided by vertical rules, the RE prediction as a solid
horizontal segment and the PI prediction as a dashed one wherever the two differ,
the information-condition arrows along the bottom, and the AVERAGE PRICE and
EFFICIENCY rows beneath the axis.  Hold one of these beside the corresponding page
scan and the digitization can be checked by eye, period by period.

What is recomputed and what is not:

  the price trace     recovered from the plotted dots (out/human_trades.csv)
  AVERAGE PRICE       the recovered trades' own mean, rounded as the original
                      rounds it -- this row is the comparison
  RE and PI lines     computed from the published design parameters
  EFFICIENCY (E/TE)   transcribed from the original, printed in grey.  They are
                      allocation measures and cannot be recomputed from prices.

`montage.py` is the complementary check: it overlays the detected dots back onto
the scan itself, zoomed, for a handful of periods.  This script is the whole
series redrawn.  Neither is part of run_all.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import published_tables as pub          # noqa: E402
from digitize import ANNOTATION_FRANCS, TOP_TICK   # noqa: E402

# The figure each market is plotted in, and the caption the original gives it.
FIGURE_NO = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}

# Information-condition bands, transcribed from the arrows printed along the bottom
# of each figure: (first period, last period, label).  These are the original's own
# wording and its own grouping -- market 1 splits its three `all` periods into "TO
# ALL" and a final "COMMON KNOWLEDGE", which the parameter table does not.
BANDS = {
    1: [(1, 4, "NO INFORMATION"),
        (5, 8, "PRIVATE INFORMATION\nTO THREE INSIDERS"),
        (9, 10, "PRIVATE\nINFORMATION\nTO ALL"),
        (11, 11, "COMMON\nKNOWLEDGE")],
    2: [(1, 4, "NO INFORMATION"),
        (5, 6, "PRIVATE\nINFORMATION\nTO ALL"),
        (7, 11, "PRIVATE INFORMATION TO SIX INSIDERS")],
    3: [(1, 2, "NO\nINFORMATION"),
        (3, 10, "PRIVATE INFORMATION TO SIX INSIDERS"),
        (11, 12, "PRIVATE\nINFORMATION\nTO ALL")],
    4: [(1, 4, "NO INFORMATION"),
        (5, 13, "PRIVATE INFORMATION TO SIX INSIDERS"),
        (14, 14, "NO\nINFORMATION")],
    5: [(1, 3, "NO INFORMATION"),
        (4, 13, "PRIVATE INFORMATION TO SIX INSIDERS")],
}

# The three rows printed under the axis, as a fraction of the axis height.
ROW_Y = {"AVERAGE PRICE": 0.098, "EFFICIENCY  (E)": 0.066, "EFFICIENCY  (TE)": 0.036}
ROW_TOP = 0.125          # the rule that closes the row block off from the plot
PAD = 0.18               # margin between a period's outermost trade and its rule
MIN_WIDTH = 7            # narrowest column that still fits a printed average

# Flags worth marking on the face of the figure.  `count_uncertain` is not one of
# them: it fires on any flat run, is common, and leaves the period's mean exact.
HARD_FLAGS = ("source_inconsistent", "mean_mismatch")


def _layout(g: pd.DataFrame):
    """x centres for every trade, and the period boundaries, in trade units."""
    counts = g.groupby("period")["trade_index"].size().sort_index()
    edges, xs, x = [0.0], {}, 0.0
    for period, n in counts.items():
        width = float(max(n, MIN_WIDTH))
        span = width - 2 * PAD
        xs[period] = (x + PAD + np.linspace(0, span, n) if n > 1
                      else np.array([x + width / 2]))
        x += width
        edges.append(x)
    return xs, edges


def render(market: int, out_path: Path) -> None:
    ht = pd.read_csv(ROOT / "out" / "human_trades.csv")
    g = ht[ht["market"] == market].sort_values(["period", "trade_index"])
    xs, edges = _layout(g)
    periods = sorted(xs)
    top = TOP_TICK[market]      # each figure's axis ends at its topmost printed tick

    mpl.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, top)

    for i, period in enumerate(periods):
        p = g[g["period"] == period]
        lo, hi = edges[i], edges[i + 1]

        # Predictions: solid = RE, dashed = PI where the two differ.
        re, pi = p["re_price"].iloc[0], p["pi_price"].iloc[0]
        ax.plot([lo, hi], [re, re], color="black", lw=1.1, solid_capstyle="butt")
        if abs(pi - re) > 0.5:
            ax.plot([lo, hi], [pi, pi], color="black", lw=1.1, ls=(0, (7, 5)))

        # The trades, in chronological order, one dot each.
        ax.plot(xs[period], p["price"].to_numpy(), color="black", lw=0.55,
                marker="o", ms=1.7, mew=0, zorder=3)

        # The rule closing the period, and the row-block cell walls under it.
        if i:
            ax.axvline(lo, color="black", lw=0.9, ymin=0, ymax=1, zorder=2)

        centre = (lo + hi) / 2
        ax.text(centre, -0.052 * top, f"{period}({p['state'].iloc[0]})",
                ha="center", va="top", fontsize=7.2)
        flagged = any(f in str(p["dot_quality_flag"].iloc[0]) for f in HARD_FLAGS)
        ax.text(centre, ROW_Y["AVERAGE PRICE"] * top,
                f"{p['price'].mean():.0f}" + ("*" if flagged else ""),
                ha="center", va="center", fontsize=7.2)
        for row, table in (("EFFICIENCY  (E)", pub.EFF), ("EFFICIENCY  (TE)", pub.TEFF)):
            val = table[market][period - 1]
            ax.text(centre, ROW_Y[row] * top,
                    f"{val:g}", ha="center", va="center", fontsize=7.2, color="0.45")

    # Row labels and the rule that separates the rows from the plot.
    ax.axhline(ROW_TOP * top, color="black", lw=0.9)
    for row, frac in ROW_Y.items():
        # Left of the tick labels, which run down to 0 in this same strip.
        ax.text(-0.055, frac * top, row, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.6,
                color="black" if row == "AVERAGE PRICE" else "0.45")

    # Information-condition arrows, on the line the original puts them on.
    y = ANNOTATION_FRANCS[market]
    for first, last, label in BANDS[market]:
        lo, hi = edges[periods.index(first)], edges[periods.index(last) + 1]
        ax.annotate("", xy=(lo, y), xytext=(hi, y),
                    arrowprops=dict(arrowstyle="<->", lw=0.7, color="black",
                                    shrinkA=0, shrinkB=0))
        ax.text((lo + hi) / 2, y, label, ha="center", va="center", fontsize=6.4,
                linespacing=1.35, bbox=dict(fc="white", ec="none", pad=1.6))

    ax.set_ylabel("TRANSACTED PRICE", fontsize=8, labelpad=10)
    ax.set_xlabel("PERIOD (STATE)", fontsize=8, labelpad=32)   # clears the period row
    ax.set_yticks(range(0, top + 1, 50))
    ax.set_yticks(range(0, top + 1, 10), minor=True)
    ax.set_yticklabels([str(t) for t in range(0, top + 1, 50)], fontsize=7.2)
    ax.set_xticks([])
    ax.tick_params(axis="y", which="major", length=4, width=0.7, direction="out")
    ax.tick_params(axis="y", which="minor", length=2, width=0.5, direction="out")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)

    fig.suptitle(f"FIG. {FIGURE_NO[market]}.—Market {market}. "
                 "Time series of contract prices, redrawn from the digitized trades",
                 fontsize=8.6, y=0.068)
    fig.text(0.5, 0.005,
             "AVERAGE PRICE is the mean of the recovered trades, not the original's "
             "printed row; efficiency rows (grey) are transcribed, being allocation "
             "measures that prices alone cannot recompute.\n"
             "* the panel cannot produce the average the original printed under it; "
             "recorded, never adjusted.",
             ha="center", fontsize=6.2, color="0.45", linespacing=1.5)
    fig.tight_layout(rect=[0.02, 0.105, 0.995, 0.995])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    (ROOT / "fig").mkdir(parents=True, exist_ok=True)
    for market in range(1, 6):
        out = ROOT / "fig" / f"reconstruction_m{market}.png"
        render(market, out)
        print(f"fig/{out.name}   (compare with the original's Figure {FIGURE_NO[market]})")


if __name__ == "__main__":
    main()
