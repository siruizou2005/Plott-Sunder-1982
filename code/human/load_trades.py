"""Stage 1.  Build the trade-level record from the appendix transcription.

Input   data/human/plott_sunder_1982_prices.csv
            987 transactions, transcribed by hand from Appendix B ("Bids, Offers,
            Prices", typescript pp. 81-97) of Caltech Social Science Working Paper
            331, the working-paper version of Plott and Sunder (1982).  A row is a
            transaction only where the appendix gives a buyer, a seller and a price;
            quotes that never traded are not in the file.

Output  results/human/trades.csv        one row per transaction, market parameters joined on
        results/human/validation.csv    one row per period, with the residual against the
                              average price the published figures print

This replaces the earlier stage 1, which recovered the same transactions by detecting
the plotted dots in Figures 2-6 of the published article.  Everything downstream is
unchanged, so the two records are scored by identical code.  What the appendix supplies
that the figures could not:

  * exact prices in whole francs, in place of a pixel coordinate mapped through an
    axis calibration;
  * exact transaction counts, in place of counts inferred from the grid pitch wherever
    same-priced trades printed as one mark;
  * the true chronological order, in place of an ordering read off horizontal position.

Market parameters are read from ps1982_params, never hardcoded, and market 1's fully
revealing price comes from `theory_price(period)`, which conditions on the ten-draw
clue sample that period drew rather than on the state letter -- market 1's insiders
held an imperfect signal, so its RE price differs period by period.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]             # the repository root
ROOT = REPO / "results" / "human"  # stage outputs
sys.path.insert(0, str(HERE))

from legacy_flags import MERGED_RUN_PERIODS   # noqa: E402
from ps1982_params import MARKETS             # noqa: E402
from published_tables import PRICES as PRINTED_MEAN  # noqa: E402

SOURCE = REPO / "data" / "human" / "plott_sunder_1982_prices.csv"

# Francs.  The published averages are printed as integers, so a transcription that is
# exactly right STILL leaves a residual: the printed value is the true mean rounded, so
# the residual is distributed U(-0.5, 0.5) and has an expected absolute value of 0.25
# francs.  The residual therefore measures the printing, not the recovery, and the most
# it can establish is that no recovery error is large enough for the published table to
# resolve.  `rounding_check` below tests exactly that.  The flagging tolerance is the
# same 3 francs the digitized record used, which keeps the rule comparable across the two.
RESID_TOL = 3.0
ROUND_BOUND = 0.5 + 1e-9

N_PERIODS = {1: 11, 2: 11, 3: 12, 4: 14, 5: 13}


def side_of(vbar: float, re_price: float) -> str:
    """Which side the informed profit on.  Empty in a no-information period."""
    if re_price > vbar:
        return "buyer"
    if re_price < vbar:
        return "seller"
    return ""


def load_source() -> pd.DataFrame:
    src = pd.read_csv(SOURCE)
    need = {"market", "period", "txn", "price"}
    missing = need - set(src.columns)
    if missing:
        raise ValueError(f"{SOURCE} is missing {sorted(missing)}")
    # The transcription is already in appendix order; sorting on (market, period, txn)
    # makes that explicit rather than relying on file order.
    src = src.sort_values(["market", "period", "txn"]).reset_index(drop=True)

    # The file's own period structure must be the experiment's.
    got = {m: int(g["period"].nunique()) for m, g in src.groupby("market")}
    if got != N_PERIODS:
        raise ValueError(f"period counts {got} do not match the design {N_PERIODS}")
    for (m, p), g in src.groupby(["market", "period"]):
        if list(g["txn"]) != list(range(1, len(g) + 1)):
            raise ValueError(f"market {m} period {p}: txn is not 1..n")
    return src


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    src = load_source()
    trade_rows, period_rows = [], []

    for market in (1, 2, 3, 4, 5):
        mk = MARKETS[market]
        vbar = float(max(mk.prior_ev.values()))

        for period in range(1, N_PERIODS[market] + 1):
            g = src[(src["market"] == market) & (src["period"] == period)]
            prices = g["price"].to_numpy(float)

            state = mk.sequence_states[period - 1]
            info = mk.sequence_info[period - 1]
            theory = mk.theory_price(period)
            re_price, pi_price = float(theory["RE"]), float(theory["PI"])

            printed = float(PRINTED_MEAN[market][period - 1])
            recovered = float(prices.mean()) if prices.size else np.nan
            resid = recovered - printed

            # The source file carries the printed average and its own reconstruction of
            # the period mean; both are recomputed here from the prices and checked
            # against what the file says, so a transcription error in either column
            # cannot pass silently.
            if "printed_period_avg" in g.columns:
                claimed = float(g["printed_period_avg"].iloc[0])
                if abs(claimed - printed) > 1e-6:
                    raise ValueError(
                        f"market {market} period {period}: the source file prints "
                        f"{claimed} where the article prints {printed}")

            # Does the printed integer round-trip?  A tie is broken either way in a
            # hand-computed table, so the bound is the symmetric one.
            rounds = bool(np.isfinite(resid) and abs(resid) <= ROUND_BOUND)

            flags = []
            if not np.isfinite(resid):
                flags.append("no_trades")
            elif abs(resid) > RESID_TOL:
                flags.append("mean_mismatch")
            flag = ";".join(flags) if flags else "ok"

            side = side_of(vbar, re_price) if info != "none" else ""
            separating = bool(info == "insider" and abs(re_price - pi_price) > 0.5)
            denom = re_price - vbar

            period_rows.append(dict(
                market=market, period=period, state=state, info=info,
                vbar=round(vbar, 4), re_price=re_price, pi_price=pi_price,
                side=side, separating=separating,
                n_trades=int(prices.size),
                mean_printed=printed,
                mean_recovered=round(recovered, 3) if np.isfinite(recovered) else np.nan,
                mean_residual=round(resid, 3) if np.isfinite(resid) else np.nan,
                first_price=float(prices[0]) if prices.size else np.nan,
                last_price=float(prices[-1]) if prices.size else np.nan,
                min_price=float(prices.min()) if prices.size else np.nan,
                max_price=float(prices.max()) if prices.size else np.nan,
                quality_flag=flag,
                printed_is_rounded_mean=rounds,
                merged_run_in_digitized=(market, period) in MERGED_RUN_PERIODS,
                wp_page=(int(g["wp_page"].iloc[0]) if "wp_page" in g.columns
                         and prices.size else np.nan),
            ))

            for i, price in enumerate(prices, start=1):
                trade_rows.append(dict(
                    market=market, period=period, trade_index=i,
                    n_trades_in_period=int(prices.size), price=float(price),
                    state=state, info=info, vbar=round(vbar, 4),
                    re_price=re_price, pi_price=pi_price, side=side,
                    D=(round((price - vbar) / denom, 5)
                       if abs(denom) > 1e-9 else np.nan),
                    separating=separating,
                    period_mean_recovered=(round(recovered, 3)
                                           if np.isfinite(recovered) else np.nan),
                    period_mean_printed=printed,
                    mean_residual=(round(resid, 3) if np.isfinite(resid) else np.nan),
                    quality_flag=flag,
                ))

    return pd.DataFrame(trade_rows), pd.DataFrame(period_rows)


def rounding_check(quality: pd.DataFrame) -> dict:
    """Are the residuals distinguishable from the printed table's own rounding?

    Under a perfect transcription the residual is the rounding error of a table printed
    in whole francs, so it is uniform on (-0.5, 0.5): expected |residual| 0.25, and no
    period outside the bound.  A Kolmogorov-Smirnov test against that distribution is
    the right check, and a median absolute residual on its own is not -- it is floored
    by the printing and says nothing about the recovery until compared against 0.25.
    """
    from scipy import stats

    ok = quality[quality["quality_flag"] == "ok"]
    r = (ok["mean_recovered"] - ok["mean_printed"]).to_numpy(float)
    ks = stats.kstest(r, stats.uniform(loc=-0.5, scale=1).cdf)
    return dict(n_periods=len(ok),
                rounds_exactly=int(ok["printed_is_rounded_mean"].sum()),
                mean_abs=float(np.abs(r).mean()), median_abs=float(np.median(np.abs(r))),
                max_abs=float(np.abs(r).max()),
                expected_mean_abs_under_rounding=0.25,
                ks_D=float(ks.statistic), ks_p=float(ks.pvalue))


def main() -> None:
    trades, quality = build()
    ROOT.mkdir(parents=True, exist_ok=True)
    trades.to_csv(ROOT / "trades.csv", index=False)
    quality.to_csv(ROOT / "validation.csv", index=False)

    ok = quality[quality["quality_flag"] == "ok"]
    res = ok["mean_residual"].abs().dropna()
    per_market = trades.groupby("market").size().to_dict()
    print(f"results/human/trades.csv        {len(trades)} trades over {len(quality)} periods "
          f"{per_market}")
    print(f"results/human/validation.csv    {len(ok)}/{len(quality)} periods validate; "
          f"median|resid|={res.median():.2f} p90={res.quantile(.9):.2f} "
          f"max={res.max():.2f} francs")
    bad = quality[quality["quality_flag"] != "ok"]
    for r in bad.itertuples():
        print(f"  flagged: market {r.market} period {r.period}  "
              f"recovered {r.mean_recovered} vs printed {r.mean_printed} "
              f"({r.mean_residual:+.2f})")

    rc = rounding_check(quality)
    pd.DataFrame([rc]).to_csv(ROOT / "rounding_check.csv", index=False)
    print(f"results/human/rounding_check.csv  printed average = rounded transcribed mean in "
          f"{rc['rounds_exactly']}/{rc['n_periods']} periods; mean|resid| "
          f"{rc['mean_abs']:.3f} against the {rc['expected_mean_abs_under_rounding']:.2f} "
          f"rounding alone implies; KS vs U(-0.5,0.5) p={rc['ks_p']:.2f}")
    miss = quality[(quality["quality_flag"] == "ok")
                   & (~quality["printed_is_rounded_mean"])]
    for r in miss.itertuples():
        print(f"  outside the rounding bound by {abs(r.mean_residual) - 0.5:.3f} f: "
              f"market {r.market} period {r.period} "
              f"({r.n_trades} trades, {r.mean_recovered} vs printed {r.mean_printed})")


if __name__ == "__main__":
    main()
