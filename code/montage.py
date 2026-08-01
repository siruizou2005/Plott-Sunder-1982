"""Overlay the detected dots on the original scan crops, so the detection is auditable.

Twelve representative periods: one per figure showing a clean detection, plus every
period the digitization flags, plus the two periods whose panels contradict the
printed averages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(HERE))

import digitize as dg          # noqa: E402
import human_trades as ht      # noqa: E402


PANELS = [
    (3, 7, "converges to RE over 19 trades"),
    (3, 11, "long flat run: count uncertain, mean exact"),
    (5, 10, "run sits 4 francs below the RE line"),
    (1, 1, "smallest dots of the five figures"),
    (2, 9, "trace crosses the dashed PI line"),
    (1, 2, "flagged: digitized mean 3.6 francs low"),
    (4, 7, "printed mean 161 lies outside all plotted ink"),
    (4, 1, "printed mean 784; only 3 dots on the axis range"),
]


def render(out_path: Path):
    _, quality, det = ht.build()
    q = quality.set_index(["market", "period"])

    fig, axes = plt.subplots(4, 2, figsize=(11.0, 13.6))
    for ax, (market, period, note) in zip(axes.ravel(), PANELS):
        cal = det[("cal", market)]
        d = det[(market, period)]
        x0, x1 = dg.period_window(cal, period)
        # Vertical window: the plotted ink plus a margin, never below the annotation.
        col = cal["ink"][:, x0:x1]
        rows = [r for r in np.flatnonzero(col.any(1))
                if cal["ytop"] <= r <= cal["ybot"]]
        acut = int((dg.ANNOTATION_FRANCS[market] + dg.ANNOTATION_HALF_WIDTH
                    - cal["intercept"]) / cal["slope"])
        rows = [r for r in rows if r < acut]
        ylo, yhi = min(rows) - 45, max(rows) + 45
        # A period whose prices barely move would otherwise be stretched until the
        # 7-pixel RE rule fills half the panel; hold a floor of 55 francs of range.
        span_min = int(55.0 / abs(cal["slope"]))
        if yhi - ylo < span_min:
            pad = (span_min - (yhi - ylo)) // 2
            ylo, yhi = ylo - pad, yhi + pad
        xlo = max(x0 - 8, cal["axis"] + 9)      # keep the y-axis furniture out
        crop = cal["ink"][ylo:yhi, xlo:x1 + 8]
        ax.imshow(~crop, cmap="gray", vmin=0, vmax=1,
                  extent=[xlo, x1 + 8, yhi, ylo], aspect="auto",
                  interpolation="antialiased")

        for c in d["cores"]:
            yy = ((c["francs"] / cal["slope"])
                  + cal["shear_a"] * (c["xc"] + d["x0"]) + cal["shear_b"])
            ax.plot([c["xmin"] + d["x0"], c["xmax"] + d["x0"]], [yy, yy],
                    color="#d62728", lw=2.4, solid_capstyle="butt", zorder=3)

        row = q.loc[(market, period)]
        ax.set_title(f"Market {market}, period {period} (state {row.state}) — {note}",
                     loc="left", fontsize=7.6, pad=4)
        printed = row.mean_printed
        digi = row.mean_digitized
        txt = (f"printed {printed:.0f}\ndigitized "
               + ("—" if not np.isfinite(digi) else f"{digi:.1f}")
               + f"\nn = {int(row.n_trades)}")
        ax.text(0.985, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.6, linespacing=1.35,
                bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2, alpha=0.92))

        # Franc gridlines at the RE level and the uninformed level.
        lines = []
        for lvl, style, lbl in ((row.re_price, "-", "RE"), (row.vbar, ":", r"$\bar v$")):
            yy = (lvl / cal["slope"]) + cal["shear_a"] * ((x0 + x1) / 2) + cal["shear_b"]
            if ylo < yy < yhi:
                ax.axhline(yy, color="#1f77b4", lw=0.7, ls=style, zorder=2)
                lines.append((yy, lbl))
        # Label the reference lines only where they will not collide with each other.
        lines.sort()
        for k, (yy, lbl) in enumerate(lines):
            if k and yy - lines[k - 1][0] < 0.06 * (yhi - ylo):
                continue
            ax.text(x1 + 7, yy, lbl, color="#1f77b4", fontsize=6,
                    va="center", ha="left", clip_on=False)
        ax.set_xticks([])
        # y ticks in francs
        lo_f = float(dg.to_francs(cal, yhi, (x0 + x1) / 2))
        hi_f = float(dg.to_francs(cal, ylo, (x0 + x1) / 2))
        ticks = [t for t in range(0, 1001, 50) if lo_f + 4 < t < hi_f - 4]
        ax.set_yticks([(t / cal["slope"]) + cal["shear_a"] * ((x0 + x1) / 2)
                       + cal["shear_b"] for t in ticks])
        ax.set_yticklabels([str(t) for t in ticks], fontsize=6)
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("price (francs)", fontsize=7, labelpad=8)
    fig.suptitle("Every trade in Plott & Sunder's Figures 2-6 is recoverable from the scans",
                 fontsize=10.0, x=0.012, ha="left", y=0.995)
    fig.text(0.012, 0.012,
             "Each red bar is one detected dot core; a bar wider than one dot is a run of "
             "consecutive trades at that price, split on the figure's 12-pixel trade grid.\n"
             "Blue solid = RE prediction, dotted = the uninformed level v-bar. "
             "Printed means are Plott & Sunder's own, from the AVERAGE PRICE row.",
             fontsize=6.8, va="bottom", color="0.25")
    fig.tight_layout(rect=[0, 0.035, 1, 0.977], h_pad=1.8)
    fig.savefig(out_path, dpi=300)
    return fig


if __name__ == "__main__":
    (STUDY / "fig").mkdir(parents=True, exist_ok=True)
    render(STUDY / "fig" / "digitization_audit.png")
    print("wrote", STUDY / "fig" / "D0_digitization_check.png")
