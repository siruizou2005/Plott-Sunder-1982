"""The RE and PI price predictions differ only in downward periods, so every period
the original price test cannot use is an upward period.

Paper Section 3.2 (figures/selection_identity.png).  Run by run_all.py.
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



# Same colors as the paper's other figures: downward periods red, upward periods blue.
BUY, SELL = "#4C72B0", "#C44E52"
# Printed at 0.62 of the 430pt text block (Figure 2), so fonts are printed sizes.
WIDTH_IN = 0.62 * 430.0 / 72.27

T = pd.read_csv(ROOT / "identity.csv")
T = T.rename(columns={"re_price": "re", "pi_price": "pi",
                      "predictions_differ": "separating"})

allp = T[T["side"].notna() & (T["side"] != "")].copy()
allp["gap_re"] = allp["re"] - allp["vbar"]
allp["gap_pi"] = allp["pi"] - allp["vbar"]

apply_figure_style(sizes=(9, 8, 8))
mpl.rcParams["savefig.bbox"] = "standard"
fig1, ax = plt.subplots(1, 1, figsize=(WIDTH_IN, 2.85), layout="constrained")
for side, col, xx in [("seller", SELL, 0), ("buyer", BUY, 1)]:
    g = allp[allp["side"] == side]
    j = np.random.default_rng(3 + xx).normal(0, 0.07, len(g))
    for (_, r), jj in zip(g.iterrows(), j):
        if abs(r["gap_re"] - r["gap_pi"]) > 1e-9:
            ax.plot([xx + jj, xx + jj], [r["gap_re"], r["gap_pi"]], color=col, lw=0.8, alpha=.6, zorder=1)
    ax.scatter(np.full(len(g), xx) + j, g["gap_re"], s=20, color=col, alpha=.85, lw=0, zorder=3)
    ax.scatter(np.full(len(g), xx) + j, g["gap_pi"], s=20, facecolors="white", edgecolors=col, lw=1.0, zorder=2)
ax.axhline(0, color=META_GREY, lw=0.9, ls="--")
ax.set_xticks([0, 1])
ax.set_xticklabels(["downward\nperiods", "upward\nperiods"])
ax.set_ylabel("Predicted price minus $\\mu$ (francs)")
ax.set_ylim(-88, 232)
ax.set_xlim(-0.5, 1.6)
ax.scatter([-0.32], [205], s=20, color=META_GREY, lw=0)
ax.text(-0.25, 205, "RE", fontsize=8, color=META_GREY, va="center")
ax.scatter([-0.32], [178], s=20, facecolors="white", edgecolors=META_GREY, lw=1.0)
ax.text(-0.25, 178, "PI", fontsize=8, color=META_GREY, va="center")
ax.text(1.0, 50, "RE $=$ PI:\nno test", fontsize=8, color=BUY, linespacing=1.2, ha="left")
ax.text(1.58, 6, "$\\mu$", fontsize=8, color=META_GREY, va="bottom", ha="right")

(REPO / "results" / "figures").mkdir(parents=True, exist_ok=True)
# Saved as RGB on a white background (no alpha channel).
fig1.savefig(REPO / "results" / "figures" / "selection_identity.png", dpi=300,
             facecolor="white", pil_kwargs={"optimize": False})
from PIL import Image  # noqa: E402
_path = REPO / "results" / "figures" / "selection_identity.png"
Image.open(_path).convert("RGB").save(_path)
