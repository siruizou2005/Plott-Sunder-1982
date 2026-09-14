"""Q1: within-period price convergence by informed side.

Two convergence notions, kept strictly apart:

  WITHIN-period  -- inside ONE trading period, does the price path travel from the
                    uninformed level vbar toward RE as trades accumulate?  Unit of
                    observation = the TRADE.  Measures: D at the first trade, D at
                    the last trade, OLS slope of D on (normalized) trade index,
                    first-half vs second-half slope, first trade inside the
                    competitive band.

  CROSS-period   -- across the session, does the k-th repetition of a side / of a
                    state land closer to RE than the (k-1)-th?  Unit of observation =
                    the PERIOD.  This is what results/human/H3_human_ordinal.csv,
                    results/human/H5_m35_ordinal.csv and results/human/T10_learning.csv measure.

Data: the TRANSCRIBED trade record, re-scored by informed side (results/human/trades.csv --
987 trades, 61 periods, 5 markets, from Appendix B of the working paper).  The
published per-period averages are in code/published_tables.py and are used only to
validate the transcription, never to fit it.

D is recomputed here from the design parameters and is defined ONLY where the informed side
exists (RE != vbar).  The shipped `D` column of trades.csv is NOT used: it is
computed for no-information periods too, where RE ~= vbar, so it explodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "results" / "human"   # stage outputs
sys.path.insert(0, str(HERE))
import ps1982_params as MK  # noqa: E402
from legacy_flags import MERGED_RUN_PERIODS as LEGACY_MERGED_RUNS  # noqa: E402

OUT, TAB = ROOT, ROOT
# The one period whose transcribed mean cannot be reconciled with the mean the
# article prints.  Under the figure-digitized record this set also held market 4's
# periods 7 and 8, which the appendix reproduces to 0.41 and 0.17 francs: they were
# failures of the digitizer, not of the source.  See legacy_flags.py.
FLAGGED = {(1, 2)}   # mean_mismatch: 263.93 transcribed against a printed 269
MINTR = 4        # a slope needs a path; 4 trades is the floor we use throughout


# ------------------------------------------------------------------ geometry
def period_geometry(mnum: int, period: int, card: str | None = None,
                    state: str | None = None, info: str | None = None) -> dict:
    """vbar, RE, the competitive band, and the band's width in D, for one period.

    The competitive allocation is supported by any price between the SECOND-highest
    and the HIGHEST informed valuation in the realized state: at such a price the
    top type wants to hold and no one else does.  That interval has a real width in
    D which differs by market and by side, so it is computed, never assumed.
    """
    m = MK.MARKETS[mnum]
    # `state`/`info` are taken from the LOG when given.  They must be, for the agent
    # runs: the `*_random_*` sessions and the m7/m8 controls redraw the state sequence
    # from the prior, so 122 of 424 agent period-states differ from the market's own
    # Table-1 row.  Deriving RE from the period index there scores a period against the
    # wrong state.  For the HUMAN data the canonical sequence IS the realization.
    info = info if info is not None else m.sequence_info[period - 1]
    state = state if state is not None else m.sequence_states[period - 1]
    if card is None:
        # markets 2-5: the card IS the state letter.  Market 1: a 10-draw sample,
        # so RE differs period by period and must come from the posterior, never the
        # state -- see markets.py theory_price.
        card = m.paper_clue_cards.get(period) if m.imperfect else state
    post = m.posterior_from_card(card if info != "none" else None)
    ev = {t: sum(post[s] * m.dividends[t][s] for s in m.states) for t in m.types}
    vals = sorted(ev.values(), reverse=True)
    re_p, second = vals[0], vals[1]
    vbar = max(m.prior_ev.values())
    span = re_p - vbar
    side = None if abs(span) < 1e-6 else ("buyer" if span > 0 else "seller")
    lo, hi = min(second, re_p), max(second, re_p)          # band in francs
    # band in D: D(second) .. D(RE)=1 ; ordered so band_lo_D <= band_hi_D
    if side is None:
        dlo = dhi = np.nan
    else:
        d_second = (second - vbar) / span
        dlo, dhi = min(d_second, 1.0), max(d_second, 1.0)
    return dict(market=mnum, period=period, state=state, info=info, side=side,
                vbar=vbar, re_price=re_p, second_val=second, span_francs=span,
                band_lo_price=lo, band_hi_price=hi,
                band_lo_D=dlo, band_hi_D=dhi, band_width_D=dhi - dlo,
                band_width_francs=hi - lo)


# ------------------------------------------------------------------ path measures
def _slope(x, y):
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan, np.nan
    r = stats.linregress(x, y)
    return r.slope, r.pvalue


def path_measures(prices: np.ndarray, geo: dict) -> dict:
    """All within-period path measures for one period's ordered trade prices."""
    n = len(prices)
    vbar, re_p, span = geo["vbar"], geo["re_price"], geo["span_francs"]
    D = (prices - vbar) / span
    idx = np.arange(1, n + 1, dtype=float)
    u = (idx - 1) / (n - 1) if n > 1 else np.array([0.5])

    out = dict(n_trades=n,
               D_first=D[0], D_last=D[-1], D_mean=D.mean(),
               D_min=D.min(), D_max=D.max(),
               D_change=D[-1] - D[0],
               p_first=prices[0], p_last=prices[-1], p_mean=prices.mean(),
               francs_first=prices[0] - vbar, francs_last=prices[-1] - vbar,
               francs_change=prices[-1] - prices[0],
               gap_francs_first=abs(re_p - prices[0]),
               gap_francs_last=abs(re_p - prices[-1]))
    out["slope_D_per_trade"], out["p_slope_trade"] = _slope(idx, D)
    out["slope_D_per_norm"], out["p_slope_norm"] = _slope(u, D)
    out["slope_francs_per_trade"], _ = _slope(idx, prices - vbar)

    h1, h2 = u <= 0.5, u > 0.5
    out["slope_D_norm_h1"], out["p_slope_h1"] = _slope(u[h1], D[h1])
    out["slope_D_norm_h2"], out["p_slope_h2"] = _slope(u[h2], D[h2])
    # per-TRADE half slopes as well, so both scalings are available without rescaling.
    out["slope_D_trade_h1"], _ = _slope(idx[h1], D[h1])
    out["slope_D_trade_h2"], _ = _slope(idx[h2], D[h2])
    out["D_h1_mean"] = D[h1].mean() if h1.any() else np.nan
    out["D_h2_mean"] = D[h2].mean() if h2.any() else np.nan

    # deciles / quintiles of normalized index (mean D of trades falling in each bin)
    for nb, tag in ((10, "dec"), (5, "qui")):
        b = np.minimum((u * nb).astype(int), nb - 1)
        for j in range(nb):
            sel = b == j
            out[f"D_{tag}{j+1}"] = D[sel].mean() if sel.any() else np.nan

    # ---- time to target -------------------------------------------------
    inband = (prices >= geo["band_lo_price"] - 1e-9) & (prices <= geo["band_hi_price"] + 1e-9)
    out["n_in_band"] = int(inband.sum())
    out["frac_in_band"] = inband.mean()
    out["ttb_index"] = float(idx[inband][0]) if inband.any() else np.nan
    out["ttb_frac"] = out["ttb_index"] / n if inband.any() else np.nan
    out["reached_band"] = bool(inband.any())
    out["ended_in_band"] = bool(inband[-1])

    hit = D >= 0.8
    out["tt80_index"] = float(idx[hit][0]) if hit.any() else np.nan
    out["tt80_frac"] = out["tt80_index"] / n if hit.any() else np.nan
    out["reached_D80"] = bool(hit.any())
    out["ended_D80"] = bool(hit[-1])

    # The band in PRICE space is not equally hard in D on the two sides: for the buyer
    # side the second-highest informed valuation lies BELOW RE, so the band is
    # [1 - w, 1]; for the seller side it lies ABOVE RE in D, so the band is [1, 1 + w]
    # and entering it means having reached or overshot RE.  A criterion that IS
    # comparable across sides: D within w of 1 on the near side, i.e. D >= 1 - w.
    w = geo["band_width_D"]
    near = D >= 1 - w
    out["band_width_D"] = w
    out["ttn_index"] = float(idx[near][0]) if near.any() else np.nan
    out["ttn_frac"] = out["ttn_index"] / n if near.any() else np.nan
    out["reached_near"] = bool(near.any())
    out["ended_near"] = bool(near[-1])

    # support asymmetry: trades strictly beyond RE (overshoot)
    beyond = (prices > re_p + 1e-9) if geo["side"] == "buyer" else (prices < re_p - 1e-9)
    out["n_beyond_RE"] = int(beyond.sum())
    out["frac_beyond_RE"] = beyond.mean()
    out["any_beyond_RE"] = bool(beyond.any())
    return out


# ------------------------------------------------------------------ builders
def human_period_table() -> pd.DataFrame:
    ht = pd.read_csv(OUT / "trades.csv")
    rows = []
    for (mnum, period), g in ht.groupby(["market", "period"], sort=True):
        g = g.sort_values("trade_index")
        m = MK.MARKETS[mnum]
        card = m.paper_clue_cards.get(period) if m.imperfect else None
        geo = period_geometry(mnum, period, card)
        # sanity: the record's own side/RE must agree with ours
        assert geo["state"] == g["state"].iloc[0], (mnum, period)
        r = dict(geo)
        r["quality_flag"] = g["quality_flag"].iloc[0]
        # The appendix gives exact transaction counts, so no period's count is
        # uncertain any more.  The column keeps the name the published robustness
        # table used and now marks the periods the DIGITIZER could not resolve, so
        # that its sensitivity row can be recomputed on the same 23 periods.
        r["count_uncertain"] = (mnum, period) in LEGACY_MERGED_RUNS
        r["flagged"] = (mnum, period) in FLAGGED
        r["mean_printed"] = g["period_mean_printed"].iloc[0]
        r["mean_recovered"] = g["period_mean_recovered"].iloc[0]
        r["mean_residual"] = g["mean_residual"].iloc[0]
        if geo["side"] is not None:
            r.update(path_measures(g["price"].to_numpy(float), geo))
        else:
            r["n_trades"] = len(g)
        rows.append(r)
    df = pd.DataFrame(rows)
    return _add_k(df)


def _add_k(df: pd.DataFrame) -> pd.DataFrame:
    """k_side = k-th insider period on that side; k_state = k-th of that state."""
    df = df.sort_values(["market", "period"]).copy()
    df["k_side"] = np.nan
    df["k_state"] = np.nan
    ins = df["info"] == "insider"
    for mnum, g in df[ins].groupby("market"):
        for side, gg in g.groupby("side"):
            df.loc[gg.index, "k_side"] = np.arange(1, len(gg) + 1)
        for st, gg in g.groupby("state"):
            df.loc[gg.index, "k_state"] = np.arange(1, len(gg) + 1)
    # market 1's clue is a 10-draw sample, so RE differs every period: "the same
    # state repeated" is not a repetition of the same problem.  k is not defined there.
    df.loc[df["market"] == 1, "k_state"] = np.nan
    return df


# ------------------------------------------------------------------ tests
def welch(a, b, label, measure, extra=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return dict(sample=label, measure=measure, n_seller=len(a), n_buyer=len(b),
                    mean_seller=a.mean() if len(a) else np.nan,
                    mean_buyer=b.mean() if len(b) else np.nan,
                    diff=np.nan, t=np.nan, p=np.nan, df=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan, **(extra or {}))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dfree = se**4 / ((a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1))
    crit = stats.t.ppf(0.975, dfree)
    d = a.mean() - b.mean()
    return dict(sample=label, measure=measure, n_seller=len(a), n_buyer=len(b),
                mean_seller=a.mean(), mean_buyer=b.mean(), sd_seller=a.std(ddof=1),
                sd_buyer=b.std(ddof=1), diff=d, t=t, p=p, df=dfree,
                ci_lo=d - crit*se, ci_hi=d + crit*se, ci_width=2*crit*se,
                **(extra or {}))


def one_sample(x, label, measure, mu=0.0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return dict(sample=label, measure=measure, n=len(x),
                    mean=x.mean() if len(x) else np.nan, p=np.nan)
    t, p = stats.ttest_1samp(x, mu)
    se = x.std(ddof=1)/np.sqrt(len(x))
    crit = stats.t.ppf(0.975, len(x)-1)
    try:
        w, pw = stats.wilcoxon(x - mu)
    except Exception:
        w, pw = np.nan, np.nan
    return dict(sample=label, measure=measure, n=len(x), mean=x.mean(),
                sd=x.std(ddof=1), mu=mu, t=t, p=p,
                ci_lo=x.mean()-crit*se, ci_hi=x.mean()+crit*se,
                ci_width=2*crit*se, wilcoxon_p=pw)


def cluster_ols(df, measure, cluster="market", label=""):
    """measure ~ 1 + buyer, SE clustered on `cluster`.  Small-G: reported, not trusted."""
    import statsmodels.api as sm
    d = df.dropna(subset=[measure]).copy()
    d["buyer"] = (d["side"] == "buyer").astype(float)
    if d["buyer"].nunique() < 2:
        return None
    X = sm.add_constant(d[["buyer"]])
    G = d[cluster].nunique()
    fit = sm.OLS(d[measure], X).fit(cov_type="cluster",
                                    cov_kwds={"groups": d[cluster]})
    return dict(sample=label, measure=measure, model=f"OLS, SE clustered on {cluster}",
                n=int(fit.nobs), n_clusters=G,
                intercept_seller=fit.params["const"],
                coef_buyer_minus_seller=fit.params["buyer"],
                se=fit.bse["buyer"], t=fit.tvalues["buyer"], p=fit.pvalues["buyer"],
                ci_lo=fit.conf_int().loc["buyer", 0], ci_hi=fit.conf_int().loc["buyer", 1],
                small_cluster_caveat=f"G={G}; cluster-robust SE is anti-conservative "
                                     f"below ~30 clusters")
