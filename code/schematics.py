"""Design exhibits for Sections 2, 3, and 5 of the paper.

These exhibits use the published market parameters only, not transaction data:

  results/tables/market_summary.tex      Section 2.1, the five markets
  results/tables/market4_dividends.tex   Section 2.2, market 4 dividends and prior values
  results/tables/market4_levels.tex      Section 2.2, market 4 price levels by state
  results/figures/position_scale.pdf     Section 5.1, the price-position scale D

Called by code/analyze.py (stage 2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "human"))
import ps1982_params as P  # noqa: E402
from printsize import RC, width  # noqa: E402

ANALYSIS = ROOT / "results" / "analysis"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
UP_COLOR, DOWN_COLOR = "#4C72B0", "#C44E52"


def _span(periods: list[int]) -> str:
    if not periods:
        return "--"
    runs, start = [], periods[0]
    for a, b in zip(periods, periods[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}--{a}")
            start = b
    return ", ".join(runs)


def _frac(x: float) -> str:
    for num, den in ((1, 3), (2, 3)):
        if abs(x - num / den) < 1e-9:
            return f"{num}/{den}"
    return f"{x:.2f}"


def market_summary() -> pd.DataFrame:
    rows = []
    for m, par in P.MARKETS.items():
        info = list(par.sequence_info)
        rows.append(dict(
            market=m, traders=par.n_investors,
            insiders=f"{par.insiders_per_type} of {par.n_per_type}",
            states=len(par.states),
            prior=", ".join(f"{s}: {_frac(par.prior[s])}" for s in par.states),
            noinfo=_span([i + 1 for i, x in enumerate(info) if x == "none"]),
            insider=_span([i + 1 for i, x in enumerate(info) if x == "insider"]),
            all_informed=_span([i + 1 for i, x in enumerate(info) if x == "all"]),
            periods=par.n_periods))
    out = pd.DataFrame(rows)
    out.to_csv(ANALYSIS / "market_summary.csv", index=False)
    lines = [r"\begin{tabular}{rrcclccc}", r"\toprule",
             r" & & Insiders & & & \multicolumn{3}{c}{Periods} \\",
             r"\cmidrule(lr){6-8}",
             r"Market & Traders & per type & States & Prior $\pi(\theta)$ & No-information & Insider & All-informed \\",
             r"\midrule"]
    for _, r in out.iterrows():
        lines.append(f"{r.market} & {r.traders} & {r.insiders} & {r.states} & {r.prior} & "
                     f"{r.noinfo} & {r.insider} & {r.all_informed}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "market_summary.tex").write_text("\n".join(lines) + "\n")
    return out


def market4_tables() -> None:
    par = P.MARKETS[4]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             rf"Type & State $X$ ($\pi={par.prior['X']:.1f}$) & State $Y$ ($\pi={par.prior['Y']:.1f}$) & Prior expected value \\",
             r"\midrule"]
    for t in par.types:
        d = par.dividends[t]
        lines.append(f"{t} & {d['X']} & {d['Y']} & {par.prior_ev[t]:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "market4_dividends.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrr}", r"\toprule",
             r"State & $P_{\RE}(\theta)$ & $P_{\PI}(\theta)$ \\", r"\midrule"]
    for s in par.states:
        lines.append(f"{s} & {P.re_price(4, s):.0f} & {P.pi_price(4, s):.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "market4_levels.tex").write_text("\n".join(lines) + "\n")


def position_scale_figure() -> None:
    """Figure 3, printed at 0.9 of the text width."""
    mu = P.VBAR[4]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 1, figsize=(width(0.9), 2.7), layout="constrained")
        cases = [(axes[0], "X", "Upward period (market 4, state $X$)", 140, 445, UP_COLOR),
                 (axes[1], "Y", "Downward period (market 4, state $Y$)", 140, 445, DOWN_COLOR)]
        for ax, state, title, lo, hi, color in cases:
            re_ = P.re_price(4, state)
            ax.set_xlim(lo, hi)
            ax.set_ylim(-1.3, 1.4)
            ax.axis("off")
            ax.annotate("", xy=(hi, 0), xytext=(lo, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=.9))
            ax.text(hi, -.42, "price", ha="right", va="top", fontsize=8)
            left, right = sorted((mu, re_))
            ax.fill_between([left, right], -.12, .12, color=color, alpha=.18, lw=0)
            for x, d_label, p_label in ((mu, "$D=0$", r"$\mu=$" + f"{mu:.0f}"),
                                        (re_, "$D=1$", r"$P_{\mathrm{RE}}=$" + f"{re_:.0f}")):
                ax.plot([x, x], [-.18, .18], color="black", lw=1.2)
                ax.text(x, .3, d_label, ha="center", va="bottom", fontsize=9)
                ax.text(x, -.3, p_label, ha="center", va="top", fontsize=8)
            ax.annotate("", xy=(re_, .64), xytext=(mu, .64),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
            ax.text((mu + re_) / 2, .72, "adjustment", ha="center", va="bottom", fontsize=8, color=color)
            if re_ > mu:
                ax.text(hi, -.95, "$D>1$: beyond the informed value", ha="right", fontsize=8, color="#555555")
                ax.text(lo, -.95, r"$D<0$: on the far side of $\mu$", ha="left", fontsize=8, color="#555555")
            else:
                ax.text(lo, -.95, "$D>1$: beyond the informed value", ha="left", fontsize=8, color="#555555")
                ax.text(hi, -.95, r"$D<0$: on the far side of $\mu$", ha="right", fontsize=8, color="#555555")
            ax.text(lo, 1.38, title, ha="left", va="top", fontsize=9)
        fig.savefig(FIGURES / "position_scale.pdf", metadata={"CreationDate": None})
        plt.close(fig)


def run() -> None:
    market_summary()
    market4_tables()
    position_scale_figure()
