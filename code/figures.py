"""Stage 4.  Render the paper's four figures into fig/.

Input   tab/*.csv  (stage 3, results.py)
Output  fig/fig1_identity.png  fig2_digitization.png
        fig3_within_period.png fig4_baseline.png
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIGURES = ["fig1_identity", "fig2_digitization", "fig3_within_period", "fig4_baseline"]


def main() -> None:
    (HERE.parent / "fig").mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
        print(f"fig/{name}.png")


if __name__ == "__main__":
    main()
