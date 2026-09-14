"""Stage 2.  Build the per-period within-period table the paper's results rest on.

Input   results/human/trades.csv          (stage 1, load_trades.py)
Output  results/human/period_table.csv          one row per period of every market

The engine is q1_within.py: it recomputes each period's geometry from the design
parameters (the uninformed level vbar, the fully revealing price RE, the informed
side, the competitive interval) and measures the price path inside the period
(D at the first and last trade, the slope of D on position, francs moved, time to
reach the RE end of the interval).  This file only calls it and writes the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import q1_within as Q

ROOT = HERE.parents[1] / "results" / "human"
TAB = ROOT


def main() -> None:
    hp = Q.human_period_table()
    hp["layer"] = "L3_appendix_transcribed"
    hp["source"] = "results/human/trades.csv"

    # Market 5's buyer-side competitive interval straddles vbar (its second-highest
    # informed valuation, 170, is below vbar = 212.5), so "inside the interval" is
    # already satisfied at the uninformed level and the interval criterion carries no
    # information there.  Flagged rather than silently pooled.
    hp["band_degenerate"] = hp["band_lo_D"] < 0

    TAB.mkdir(parents=True, exist_ok=True)
    hp.to_csv(TAB / "period_table.csv", index=False)
    print(f"results/human/period_table.csv  {len(hp)} periods, {hp.shape[1]} columns")


if __name__ == "__main__":
    main()
