"""Run the whole pipeline, from the page scans to the paper's tables and figures.

    python3 code/run_all.py

Stages, each of which can also be run on its own:
  1  human_trades.py   data/scan_p17-21.png -> out/human_trades.csv
                       recovers the individual trades from the plotted dots and
                       validates every period against the printed average price
  2  period_table.py   out/human_trades.csv -> tab/period_table.csv
                       one row per period: geometry plus within-period path measures
  3  results.py        tab/period_table.csv -> tab/*.csv
                       every number the paper reports
  4  figures.py        tab/*.csv -> fig/fig1-4.png

Not part of the pipeline, run it when the digitization needs auditing by eye:
     montage.py       overlays the detected dots on the scan crops
                      -> fig/digitization_audit.png
"""
from __future__ import annotations

import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["human_trades", "period_table", "results", "figures"]


def main() -> None:
    sys.path.insert(0, str(HERE))
    for i, name in enumerate(STAGES, 1):
        t0 = time.time()
        print(f"\n=== stage {i}: {name}.py " + "=" * (46 - len(name)))
        runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
        print(f"--- {name}.py finished in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
