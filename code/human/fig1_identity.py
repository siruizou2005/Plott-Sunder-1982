"""Figure 1.  The two models predict different prices only when the informed sell,
so every period the original's price test discards is a period in which they buy.

Paper Figure 1 (figures/selection_identity.png).  Run by run_all.py.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / "results" / "human"
sys.path.insert(0, str(HERE))

from figstyle import META_GREY, apply_figure_style  # noqa: E402

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st

from ps1982_params import MARKETS



BUY, SELL = "#1f6fb4", "#c2571a"

T = pd.read_csv(ROOT / "identity.csv")
T = T.rename(columns={"re_price": "re", "pi_price": "pi",
                      "predictions_differ": "separating"})

allp = T[T["side"].notna() & (T["side"] != "")].copy()
allp["gap_re"] = allp["re"] - allp["vbar"]
allp["gap_pi"] = allp["pi"] - allp["vbar"]

apply_figure_style(sizes=(9, 8, 7))
fig1, axA = plt.subplots(1, 1, figsize=(3.6, 2.9))
ax = axA
for side, col, xx in [("seller", SELL, 0), ("buyer", BUY, 1)]:
    g = allp[allp["side"] == side]
    j = np.random.default_rng(3 + xx).normal(0, 0.07, len(g))
    for (_, r), jj in zip(g.iterrows(), j):
        if abs(r["gap_re"] - r["gap_pi"]) > 1e-9:
            ax.plot([xx + jj, xx + jj], [r["gap_re"], r["gap_pi"]], color=col, lw=0.8, alpha=.6, zorder=1)
    ax.scatter(np.full(len(g), xx) + j, g["gap_re"], s=24, color=col, alpha=.8, lw=0, zorder=3)
    ax.scatter(np.full(len(g), xx) + j, g["gap_pi"], s=24, facecolors="none", edgecolors=col, lw=1.0, zorder=2)
ax.axhline(0, color=META_GREY, lw=0.9, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels(["informed\nselling", "informed\nbuying"])
ax.set_ylabel("Predicted price minus $\\bar v$ (francs)")
ax.set_title("The two models differ only\nwhen the informed sell")
ax.set_ylim(-88, 232)
ax.set_xlim(-0.5, 1.6)
ax.text(1.02, 1.0, "$\\bar v$", fontsize=7.5, color=META_GREY, va="center", transform=ax.get_yaxis_transform())
ax.scatter([-0.30], [205], s=24, color=META_GREY, lw=0)
ax.text(-0.24, 205, "RE", fontsize=6.5, color=META_GREY, va="center")
ax.scatter([-0.30], [178], s=24, facecolors="none", edgecolors=META_GREY, lw=1.0)
ax.text(-0.24, 178, "PI", fontsize=6.5, color=META_GREY, va="center")
ax.text(0.98, 46, "RE $=$ PI:\nno test", fontsize=6.5, color=BUY, linespacing=1.25, ha="left")

fig1.tight_layout()

for t in list(axA.texts):
    if t.get_text() == "$\\bar v$":
        t.remove()
axA.text(1.48, 6, "$\\bar v$", fontsize=8, color=META_GREY, va="bottom", ha="right")

(REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig1.savefig(REPO / "results" / "figures" / "selection_identity.png", dpi=300)
