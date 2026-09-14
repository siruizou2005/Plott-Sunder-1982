"""Stage 1: rebuild the human period data from the Appendix B transcription.

    python3 code/human/run_all.py

Steps, each of which can also be run on its own:
  1  load_trades.py    data/human/plott_sunder_1982_prices.csv -> results/human/trades.csv
                       joins the design parameters onto every transaction and
                       validates each period against the printed average price
                       (also writes validation.csv and rounding_check.csv)
  2  period_table.py   trades.csv -> results/human/period_table.csv
                       one row per period: benchmarks plus within-period path measures
  3  results.py        period_table.csv -> results/human/*.csv
                       classification ledger, contrasts, robustness, familiarity and
                       opening-window checks
  4  fig1_identity.py  identity.csv -> results/figures/selection_identity.png
                       paper Figure 1
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["load_trades", "period_table", "results", "fig1_identity"]


def main() -> None:
    sys.path.insert(0, str(HERE))
    for i, name in enumerate(STAGES, 1):
        t0 = time.time()
        print(f"\n=== step {i}: {name}.py " + "=" * (46 - len(name)))
        runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
        print(f"--- {name}.py finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
