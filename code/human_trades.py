"""Build the trade-level human price record from Plott & Sunder's Figures 2-6.

Writes
    study78/out/human_trades.csv            one row per recovered trade
    study78/tab/D0_digitization_quality.csv one row per period, with residuals
    study78/fig/D0_digitization_check.png   detection overlaid on the scans

The digitization itself lives in `digitize.py`.  This script drives it, joins the
result against `ps1982.markets` for the market parameters, validates every period
against the AVERAGE PRICE row the original prints under each panel, and flags rather
than adjusts any period that disagrees.

Market parameters are read from the repo, never hardcoded, and market 1's RE is
taken from `theory_price(period)`, which conditions on the ten-draw clue sample that
period actually drew rather than on the state letter -- market 1's insiders held an
imperfect signal, so its RE differs period by period.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent                          # the package root
sys.path.insert(0, str(HERE))

import digitize as dg                        # noqa: E402
from ps1982_params import MARKETS            # noqa: E402

# The AVERAGE PRICE row printed under each panel -- the validation ground truth.
# Transcribed from the figures in published_tables.py.
from published_tables import PRICES as PRINTED_MEAN   # noqa: E402

# Periods the original's own figure cannot be reconciled with its own printed mean.
# Recorded here rather than silently dropped; see the notes in the quality table.
KNOWN_ANOMALIES = {
    (4, 1): "printed mean 784 with prices off the top of the axis; the panel plots "
            "only 3 dots inside the 0-400 range, so the period is not recoverable",
    (4, 7): "every dot in the panel lies between 164 and 182 francs, so no set of "
            "trades in this panel can average the printed 161",
    (4, 8): "every dot in the panel lies between 169 and 182 francs, so no set of "
            "trades in this panel can average the printed 164",
}

RESID_TOL = 3.0     # francs; P&S rounded the printed means to integers


def figure_path(market: int) -> Path:
    """Market m is plotted in the original's Figure m+1, printed on scan page m+16."""
    return STUDY / "data" / f"scan_p{16 + market}.png"


def side_of(vbar: float, re_price: float) -> str:
    """Which side the informed profit on.  Empty in a no-information period."""
    if re_price > vbar:
        return "buyer"
    if re_price < vbar:
        return "seller"
    return ""


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    trade_rows, period_rows, detections = [], [], {}
    for market in (1, 2, 3, 4, 5):
        gray = dg.load_upright(figure_path(market), market)
        cal = dg.calibrate(gray, market)
        pitch = dg.grid_pitch(cal)
        mk = MARKETS[market]
        vbar = float(max(mk.prior_ev.values()))

        for period in range(1, dg.N_PERIODS[market] + 1):
            state = mk.sequence_states[period - 1]
            info = mk.sequence_info[period - 1]
            theory = mk.theory_price(period)
            re_price, pi_price = float(theory["RE"]), float(theory["PI"])
            det = dg.detect_trades(cal, period, pitch, re_price, pi_price)
            detections[(market, period)] = det

            prices = det["prices"]
            printed = float(PRINTED_MEAN[market][period - 1])
            digitized = float(prices.mean()) if prices.size else np.nan
            resid = digitized - printed

            flags = []
            if (market, period) in KNOWN_ANOMALIES:
                flags.append("source_inconsistent")
            elif not np.isfinite(resid):
                flags.append("no_dots")
            elif abs(resid) > RESID_TOL:
                flags.append("mean_mismatch")
            if det["run_flags"]:
                flags.append("count_uncertain:" + ",".join(det["run_flags"]))
            if det["unmatched_bands"]:
                flags.append("unmatched_band:"
                             + ",".join(str(b) for b in det["unmatched_bands"]))
            flag = ";".join(flags) if flags else "ok"

            side = side_of(vbar, re_price) if info != "none" else ""
            separating = bool(info == "insider" and abs(re_price - pi_price) > 0.5)
            denom = re_price - vbar

            period_rows.append(dict(
                market=market, period=period, state=state, info=info,
                vbar=round(vbar, 4), re_price=re_price, pi_price=pi_price,
                side=side, separating=separating,
                n_trades=int(prices.size), n_cores=int(det["n_cores"]),
                mean_printed=printed,
                mean_digitized=round(digitized, 3) if np.isfinite(digitized) else np.nan,
                mean_residual=round(resid, 3) if np.isfinite(resid) else np.nan,
                first_price=round(float(prices[0]), 2) if prices.size else np.nan,
                last_price=round(float(prices[-1]), 2) if prices.size else np.nan,
                min_price=round(float(prices.min()), 2) if prices.size else np.nan,
                max_price=round(float(prices.max()), 2) if prices.size else np.nan,
                dot_quality_flag=flag,
                n_runs_split=len(det["run_flags"]),
                note=KNOWN_ANOMALIES.get((market, period), ""),
            ))

            for i, price in enumerate(prices, start=1):
                trade_rows.append(dict(
                    market=market, period=period, trade_index=i,
                    n_trades_in_period=int(prices.size), price=round(float(price), 2),
                    state=state, info=info, vbar=round(vbar, 4),
                    re_price=re_price, pi_price=pi_price, side=side,
                    D=(round((price - vbar) / denom, 5)
                       if abs(denom) > 1e-9 else np.nan),
                    separating=separating,
                    period_mean_digitized=(round(digitized, 3)
                                           if np.isfinite(digitized) else np.nan),
                    period_mean_printed=printed,
                    mean_residual=(round(resid, 3) if np.isfinite(resid) else np.nan),
                    dot_quality_flag=flag,
                ))

        detections[("cal", market)] = cal
        detections[("pitch", market)] = pitch
    return pd.DataFrame(trade_rows), pd.DataFrame(period_rows), detections


if __name__ == "__main__":
    trades, quality, _ = build()
    (STUDY / "out").mkdir(parents=True, exist_ok=True)
    (STUDY / "tab").mkdir(parents=True, exist_ok=True)
    trades.to_csv(STUDY / "out" / "human_trades.csv", index=False)
    quality.to_csv(STUDY / "tab" / "D0_digitization_quality.csv", index=False)
    ok = quality[quality.dot_quality_flag.str.startswith(("ok", "count_uncertain",
                                                          "unmatched_band"))]
    res = ok.mean_residual.abs().dropna()
    print(f"trades={len(trades)} periods={len(quality)} "
          f"median|resid|={res.median():.2f} p90={res.quantile(.9):.2f} "
          f"max={res.max():.2f}")
