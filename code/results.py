"""Stage 3.  Every number the paper reports, from the period table.

Input   tab/period_table.csv          (stage 2, period_table.py)
        out/human_trades.csv          (stage 1, for the francs-denominated paths)

Output, one file per table or claim in the paper:
  tab/identity.csv              the RE/PI crosstabulation behind Proposition 1
  tab/within_period.csv         the main contrast, seller-side vs buyer-side
  tab/robustness.csv            same contrast under four inference schemes and
                                four samples (the paper's Table 7)
  tab/established.csv           the k>=2 and k>=3 subsamples (the authors' own
                                carve-out for a new and unknown state)
  tab/by_market.csv             the contrast disaggregated by market
  tab/interval_scoring.csv      scoring against the competitive interval instead
                                of the point prediction
  tab/repetition.csv            first- and last-trade D against repetition of the
                                same state
  tab/baseline.csv              the information-free account and why it fails
  tab/baseline_sensitivity.csv  that account under four benchmark constructions
  tab/endowment_capacity.csv    the capacity refutation, from endowments alone

The paper restricts its within-period results to markets 3-5.  Market 1 gave the
informed a ten-draw sample rather than the state, so its RE price differs period
by period and no state recurs under identical information; market 2 carried
full-information training periods.  Both departures are ones the original notes.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import ps1982_params as P

ROOT = Path(__file__).resolve().parent.parent
OUT, TAB = ROOT / "out", ROOT / "tab"

MEASURES = [
    ("D_first", "$D$ at first trade"),
    ("D_last", "$D$ at last trade"),
    ("D_mean", "$D$, period average"),
    ("D_change", "Change across period"),
    ("slope_D_per_norm", "Slope on position"),
    ("francs_change", "Francs moved in period"),
    ("gap_francs_last", "Francs from RE, last trade"),
]

NPERM = 20_000
SEED = 20260731


# --------------------------------------------------------------- helpers
def insider(A: pd.DataFrame, markets=(3, 4, 5)) -> pd.DataFrame:
    """Insider periods of the given markets, both sides present."""
    return A[(A["market"].isin(markets)) & (A["info"] == "insider")
             & (A["side"].isin(["buyer", "seller"]))].copy()


def contrast(d: pd.DataFrame, col: str) -> dict:
    s = d[d["side"] == "seller"][col].dropna().values
    b = d[d["side"] == "buyer"][col].dropna().values
    return dict(n_sell=len(s), n_buy=len(b), mean_sell=s.mean(), mean_buy=b.mean(),
                diff=s.mean() - b.mean(),
                p_welch=stats.ttest_ind(s, b, equal_var=False).pvalue,
                p_mwu=stats.mannwhitneyu(s, b).pvalue)


def permutation_within_market(d: pd.DataFrame, col: str, nperm: int = NPERM) -> float:
    """Reassign the side label WITHIN each market.

    The 27 insider periods come from only three markets, and every period of one
    (market, side) cell shares a vbar, an RE price and a group of subjects.  A test
    that shuffles the side label across markets would credit the contrast with
    between-market variation it has not earned; this one does not.
    """
    rng = np.random.default_rng(SEED)
    obs = d[d.side == "seller"][col].mean() - d[d.side == "buyer"][col].mean()
    cells = [g for _, g in d.groupby("market")]
    hits = 0
    for _ in range(nperm):
        s_parts, b_parts = [], []
        for g in cells:
            lab = rng.permutation(g["side"].values)
            s_parts.append(g[col].values[lab == "seller"])
            b_parts.append(g[col].values[lab == "buyer"])
        stat = np.concatenate(s_parts).mean() - np.concatenate(b_parts).mean()
        hits += abs(stat) >= abs(obs) - 1e-12
    return (hits + 1) / (nperm + 1)


def period_paths(HT: pd.DataFrame) -> pd.DataFrame:
    """Francs-denominated path of every period, for the benchmark comparisons.

    D normalizes by (RE - vbar), which differs by side, so the benchmark section
    works in francs instead: a franc moved is a franc moved on either side.
    """
    rows = []
    for (m, p), g in HT.groupby(["market", "period"]):
        g = g.sort_values("trade_index")
        if len(g) < 2:
            continue
        u = (g["trade_index"] - 1) / (g["n_trades_in_period"] - 1)
        pr = g["price"].values
        rows.append(dict(
            market=m, period=p, info=g["info"].iloc[0], side=g["side"].iloc[0],
            n=len(g), vbar=g["vbar"].iloc[0], re=g["re_price"].iloc[0],
            first=pr[0], last=pr[-1], mean=pr.mean(),
            francs_change=pr[-1] - pr[0], slope_f=np.polyfit(u.values, pr, 1)[0],
            first_minus_vbar=pr[0] - g["vbar"].iloc[0],
            mean_minus_vbar=pr.mean() - g["vbar"].iloc[0],
            flag=g["dot_quality_flag"].iloc[0]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 1. identity
def identity_table() -> pd.DataFrame:
    """Proposition 1: the two models predict different prices iff the informed sell.

    Recomputed from the design parameters alone for every insider period of all five
    markets.  RE is the highest expected dividend any type holds once the clue is
    known; PI is the highest valuation anyone who has not seen the clue holds, or RE
    where that is higher.  In markets 2-5 the clue names the state, so the posterior
    is degenerate; in market 1 it is a ten-draw urn sample, so the posterior is
    computed from the card the original printed and RE is not a function of the state
    alone.

    The crosstabulation of side against 'the predictions differ' is exactly diagonal,
    which is the paper's point: the original's price test is scored on
    informed-selling periods and discards informed-buying periods, with no overlap.
    """
    rows = []
    for m in (1, 2, 3, 4, 5):
        par = P.MARKETS[m]
        vbar = P.VBAR[m]
        for i, (state, info) in enumerate(zip(par.sequence_states, par.sequence_info), 1):
            if info != "insider":
                continue
            card = par.paper_clue_cards.get(i) if par.imperfect else state
            post = par.posterior_from_card(card)
            ev = {t: sum(post[s] * par.dividends[t][s] for s in par.states)
                  for t in par.types}
            re_p = max(ev.values())
            pi_p = max(vbar, re_p)
            # market 1's card is a binary string; a CSV round-trip would read
            # "0100101010" back as a number and lose the leading zeros, so the
            # posterior-relevant summary (how many of the ten draws were marked) is
            # written alongside it.
            rows.append(dict(
                market=m, period=i, state=state,
                clue_card=(f"'{card}" if (par.imperfect and card) else ""),
                clue_ones=(card.count("1") if (par.imperfect and card) else np.nan),
                clue_draws=(len(card) if (par.imperfect and card) else np.nan),
                vbar=vbar, re_price=re_p, pi_price=pi_p,
                side=("buyer" if re_p > vbar else "seller" if re_p < vbar else ""),
                predictions_differ=(abs(re_p - pi_p) > 1e-9)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 2. main contrast
def within_period_table(ins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, lab in MEASURES:
        r = dict(measure=lab, column=col)
        r.update(contrast(ins, col))
        rows.append(r)
    return pd.DataFrame(rows)


def robustness_table(ins: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Panel A varies the inference scheme; Panel B varies the sample."""
    d = ins.copy()
    d["buy"] = (d["side"] == "buyer").astype(int)

    rows = []
    for col, lab in MEASURES[:-1]:
        c = contrast(d, col)
        ols = smf.ols(f"{col} ~ buy + C(market)", data=d).fit()
        clu = smf.ols(f"{col} ~ buy + C(market)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["market"]})
        rows.append(dict(panel="A: inference scheme", sample="markets 3-5, 27 periods",
                         measure=lab, diff=c["diff"], p_welch=c["p_welch"],
                         p_mwu=c["p_mwu"],
                         p_perm_within_market=permutation_within_market(d, col),
                         fe_diff=-ols.params["buy"], p_fe_ols=ols.pvalues["buy"],
                         p_fe_cluster=clu.pvalues["buy"]))

    samples = {
        "Markets 3-5, all 27 periods": ins,
        "Four flagged periods dropped": ins[~ins["flagged"].astype(bool)],
        "23 merged-run periods dropped": ins[~ins["count_uncertain"].astype(bool)],
        "All five markets": insider(A, markets=(1, 2, 3, 4, 5)),
    }
    for lab, sub in samples.items():
        for col, mlab in [("D_first", "$D$ at first trade"),
                          ("francs_change", "Francs moved in period")]:
            c = contrast(sub, col)
            rows.append(dict(panel="B: sample", sample=lab, measure=mlab, **c))
    return pd.DataFrame(rows)


def by_market_table(ins: pd.DataFrame) -> pd.DataFrame:
    g = ins.groupby(["market", "side"]).agg(
        n=("period", "size"), re_price=("re_price", "mean"), vbar=("vbar", "mean"),
        D_first=("D_first", "mean"), D_last=("D_last", "mean"),
        slope=("slope_D_per_norm", "mean"), francs_moved=("francs_change", "mean"),
    ).reset_index()
    g["required_francs"] = g["re_price"] - g["vbar"]
    return g


def established_table(ins: pd.DataFrame) -> pd.DataFrame:
    """The original exempts 'a new and unknown state of nature' from its own
    instantaneity claim.  A state's first occurrence IS that exempted case, so the
    claim is retested with those periods conceded."""
    rows = []
    for lab, q in [("all", ins["k_state"] >= 1),
                   ("k>=2", ins["k_state"] >= 2),
                   ("k>=3", ins["k_state"] >= 3)]:
        for side in ["seller", "buyer"]:
            v = ins[q & (ins["side"] == side)]
            f = v["D_first"].dropna().values
            l = v["D_last"].dropna().values
            s = v["slope_D_per_norm"].dropna().values
            if len(f) < 2:
                continue
            rows.append(dict(
                sample=lab, side=side, n=len(f),
                D_first=f.mean(), p_first_vs_1=stats.ttest_1samp(f, 1).pvalue,
                w_first_vs_1=stats.wilcoxon(f - 1).pvalue,
                D_last=l.mean(), slope=s.mean(),
                p_slope_vs_0=stats.ttest_1samp(s, 0).pvalue,
                francs_moved=v["francs_change"].mean()))
        sub = ins[q]
        c = contrast(sub, "D_first")
        rows.append(dict(sample=lab, side="between-side contrast", n=len(sub),
                         D_first=c["diff"], p_first_vs_1=c["p_welch"]))
    return pd.DataFrame(rows)


def interval_table(ins: pd.DataFrame) -> pd.DataFrame:
    """Scoring against the competitive interval [second-highest informed valuation,
    RE] rather than the point prediction.  This is what pins the paper's claim to
    SPEED: under interval scoring the level difference is not distinguishable, while
    the timing difference survives."""
    rows = []
    ins = ins.copy()
    ins["dist_below_band"] = np.maximum(0, ins["band_lo_D"] - ins["D_mean"])
    for lab, sub in [("markets 3-5", ins),
                     ("excluding market 5 buy side (degenerate interval)",
                      ins[~((ins["market"] == 5) & (ins["side"] == "buyer"))]),
                     ("markets 3 and 4 only", ins[ins["market"].isin([3, 4])])]:
        c = contrast(sub, "dist_below_band")
        rows.append(dict(sample=lab, measure="distance below interval ($D$ units)", **c))

    nd = ins[ins["band_width_D"].notna() & (ins["band_width_D"] > 0)]
    for col, lab in [("tt80_frac", "position of first trade at $D\\ge0.8$"),
                     ("frac_in_band", "fraction of trades inside the interval")]:
        c = contrast(nd, col)
        rows.append(dict(sample="markets 3-5, non-degenerate intervals", measure=lab, **c))
    return pd.DataFrame(rows)


def repetition_table(ins: pd.DataFrame) -> pd.DataFrame:
    """What repetition changes.  The unit is the STATE, following the original's own
    'repeated occurrences of the state'."""
    rows = []
    for side in ["buyer", "seller"]:
        for col in ["D_first", "D_last", "D_mean", "slope_D_per_norm"]:
            g = ins[(ins["side"] == side) & ins["k_state"].notna()][["k_state", col]].dropna()
            if len(g) < 4:
                continue
            fit = smf.ols(f"{col} ~ k_state", data=g).fit()
            sp = stats.spearmanr(g["k_state"], g[col])
            rows.append(dict(side=side, measure=col, n=len(g),
                             slope=fit.params["k_state"], se=fit.bse["k_state"],
                             p_ols=fit.pvalues["k_state"],
                             rho=sp.statistic, p_spearman=sp.pvalue))
    return pd.DataFrame(rows)


# --------------------------------------------------------------- 3. the baseline
def benchmarks(PA: pd.DataFrame) -> dict:
    """The four constructions of the no-information benchmark.

    The PRIMARY benchmark excludes each market's opening period.  Period 1 of a
    market carries a large downward transient (market 4's alone is -210 francs and is
    also flagged source_inconsistent) which flips the sign of the mean drift: all 18
    periods give -15.8 francs, while excluding opening periods gives +7.7 with 12 of
    13 positive.  The exclusion is a concession to the account under test, not a
    convenience -- it gives that account the largest drift it can claim.
    """
    NI = PA[PA["info"] == "none"].copy()
    NI["flagged"] = NI["flag"].astype(str).str.contains("mean_mismatch|source_inconsistent")
    return {
        "All 18 no-information periods": NI,
        "Excluding each market's opening period (primary)": NI[NI["period"] != 1],
        "Excluding opening and flagged periods": NI[(NI["period"] != 1) & (~NI["flagged"])],
        "Markets 3-5 only, excluding opening periods":
            NI[(NI["market"].isin([3, 4, 5])) & (NI["period"] != 1)],
    }


def baseline_tables(PA: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bm = benchmarks(PA)
    primary = bm["Excluding each market's opening period (primary)"]
    ins35 = PA[(PA["info"] == "insider") & (PA["market"].isin([3, 4, 5]))]
    SELL = ins35[ins35["side"] == "seller"]
    BUY = ins35[ins35["side"] == "buyer"]

    drift = primary["francs_change"].mean()
    disc = primary["first_minus_vbar"].mean()
    npos = int((primary["francs_change"] > 0).sum())

    # The benchmark-free comparison: whatever value the benchmark takes, a
    # state-independent account predicts ONE opening level, and the two sides open on
    # opposite sides of vbar.
    apart = BUY["first_minus_vbar"].mean() - SELL["first_minus_vbar"].mean()

    rows = [
        ("A within-period upward drift exists absent information",
         f"{drift:+.1f} f, {npos} of {len(primary)} periods positive, "
         f"sign p={stats.binomtest(npos, len(primary), 0.5).pvalue:.3f}",
         "confirmed"),
        ("Opening trades fall at one common level, whatever that level is",
         f"seller {SELL['first_minus_vbar'].mean():+.1f} f and buyer "
         f"{BUY['first_minus_vbar'].mean():+.1f} f relative to vbar, {apart:.1f} f apart "
         f"on opposite sides, p="
         f"{stats.ttest_ind(BUY['first_minus_vbar'], SELL['first_minus_vbar'], equal_var=False).pvalue:.4f}",
         "rejected"),
        ("The opening level is independent of the state",
         f"the seller side's first trade is indistinguishable from its own RE price "
         f"({(SELL['first'] - SELL['re']).mean():+.1f} f, "
         f"p={stats.ttest_1samp(SELL['first'] - SELL['re'], 0).pvalue:.2f}), while the "
         f"buyer side's is {(BUY['first'] - BUY['re']).mean():+.1f} f below its own "
         f"(p={stats.ttest_1samp(BUY['first'] - BUY['re'], 0).pvalue:.4f})",
         "rejected"),
        ("The same francs are moved within a period on both sides",
         f"buyer {BUY['francs_change'].mean():+.1f} f vs seller "
         f"{SELL['francs_change'].mean():+.1f} f, p="
         f"{stats.ttest_ind(BUY['francs_change'], SELL['francs_change'], equal_var=False).pvalue:.4f}",
         "rejected"),
        ("The drift accounts for the buying side's climb",
         f"{drift:+.1f} f of {BUY['francs_change'].mean():+.1f} f = "
         f"{100 * drift / BUY['francs_change'].mean():.0f} per cent; the climb is "
         f"{BUY['francs_change'].mean() / drift:.1f} times the drift",
         "rejected"),
    ]
    B = pd.DataFrame(rows, columns=["prediction", "evidence", "verdict"])

    # sensitivity of every affected figure to the benchmark construction
    srows = []
    for lab, sub in benchmarks(PA).items():
        d, dc = sub["francs_change"].mean(), sub["first_minus_vbar"].mean()
        npos = int((sub["francs_change"] > 0).sum())
        r = dict(benchmark=lab, n=len(sub), drift=d, opening_vs_vbar=dc,
                 positive=f"{npos}/{len(sub)}",
                 sign_p=stats.binomtest(npos, len(sub), 0.5).pvalue,
                 buyer_climb_over_drift=BUY["francs_change"].mean() / d,
                 drift_share_pct=100 * d / BUY["francs_change"].mean())
        for side, arr in [("seller", SELL), ("buyer", BUY)]:
            miss = arr["first"] - (arr["vbar"] + dc)
            r[f"{side}_opening_miss"] = miss.mean()
            r[f"{side}_opening_p"] = stats.ttest_1samp(miss, 0).pvalue
            r[f"{side}_move_vs_benchmark"] = arr["francs_change"].mean() - d
            r[f"{side}_move_p"] = stats.ttest_ind(
                arr["francs_change"], sub["francs_change"], equal_var=False).pvalue
        srows.append(r)
    return B, pd.DataFrame(srows)


def endowment_capacity() -> pd.DataFrame:
    """Refute the capacity account from the endowments alone.

    Every investor holds 10,000 francs and two certificates against a total supply of
    twenty-four units.  On the buying side one informed trader's cash alone would
    purchase the whole of the outstanding supply.  The one endowment constraint the
    design imposes falls on the SELLING side: six insiders hold twelve of twenty-four
    units between them and Instruction Set 2 forbids short sales ('Your holdings of
    certificates may never go below zero').  The binding constraint is on the side
    that adjusts instantaneously, which is the opposite of what capacity requires.
    """
    rows = []
    for m in (3, 4, 5):
        par = P.MARKETS[m]
        supply = P.INITIAL_CERTS * len(par.dividends) * par.n_per_type
        n_insiders = par.insiders_per_type * len(par.dividends)
        for state in par.states:
            side = P.informed_side(m, state)
            if not side:
                continue
            re_p, vbar = P.re_price(m, state), P.VBAR[m]
            rows.append(dict(
                market=m, state=state, informed_side=side, vbar=vbar, re_price=re_p,
                total_supply=supply,
                units_one_insider_buys_at_re=P.INITIAL_CASH / re_p,
                units_one_insider_buys_at_vbar=P.INITIAL_CASH / vbar,
                units_all_insiders_can_sell=n_insiders * P.INITIAL_CERTS,
                cash_binds_before_supply=(P.INITIAL_CASH / re_p) < supply,
                short_sales_permitted=False))
    return pd.DataFrame(rows).drop_duplicates()


# --------------------------------------------------------------- 4. figure inputs
def trajectory_table(ins: pd.DataFrame) -> pd.DataFrame:
    """Mean D in each decile of a period's trade sequence, by side.

    Each period is split into ten equal slices of its own trade count, so periods of
    different length are comparable; the columns D_dec1..D_dec10 are built by
    q1_within.path_measures.
    """
    rows = []
    for side in ("seller", "buyer"):
        g = ins[ins["side"] == side]
        for j in range(1, 11):
            v = g[f"D_dec{j}"].dropna()
            rows.append(dict(sample="human m3-5 insider (primary)", side=side,
                             bin=j, nbins=10, bin_lo=(j - 1) / 10, bin_hi=j / 10,
                             n_periods=len(v), D_mean=v.mean(),
                             D_sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                             D_median=v.median(),
                             se=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan))
    T = pd.DataFrame(rows)
    T["layer"] = "L2_human_digitized"
    return T


def trade_level_table(HT: pd.DataFrame, ins: pd.DataFrame) -> pd.DataFrame:
    """Every trade of the markets 3-5 insider periods, carrying its period's geometry.

    D is recomputed here rather than taken from human_trades.csv, whose shipped D
    column is defined for no-information periods too, where RE is close to vbar and
    the ratio explodes.
    """
    gmap = {(r.market, r.period): r for r in ins.itertuples()}
    ht = HT.copy()
    ht["_k"] = list(zip(ht.market, ht.period))
    ht = ht[ht["_k"].isin(gmap)].copy()
    ht["vb"] = [gmap[k].vbar for k in ht["_k"]]
    ht["sp"] = [gmap[k].span_francs for k in ht["_k"]]
    ht["sd"] = [gmap[k].side for k in ht["_k"]]
    ht["flg"] = [gmap[k].flagged for k in ht["_k"]]
    ht["cu"] = [gmap[k].count_uncertain for k in ht["_k"]]
    ht["Dc"] = (ht.price - ht.vb) / ht.sp
    ht["u"] = (ht.trade_index - 1) / (ht.n_trades_in_period - 1)
    ht["buy"] = (ht.sd == "buyer").astype(float)
    ht["pid"] = ht.market * 100 + ht.period
    return ht


# --------------------------------------------------------------- driver
def main() -> None:
    A = pd.read_csv(TAB / "period_table.csv")
    HT = pd.read_csv(OUT / "human_trades.csv")
    ins = insider(A)
    PA = period_paths(HT)

    B, S = baseline_tables(PA)
    products = {
        "identity.csv": identity_table(),
        "within_period.csv": within_period_table(ins),
        "robustness.csv": robustness_table(ins, A),
        "by_market.csv": by_market_table(ins),
        "established.csv": established_table(ins),
        "interval_scoring.csv": interval_table(ins),
        "repetition.csv": repetition_table(ins),
        "baseline.csv": B,
        "baseline_sensitivity.csv": S,
        "endowment_capacity.csv": endowment_capacity(),
        "period_paths_francs.csv": PA,
        "trajectory.csv": trajectory_table(ins),
        "trade_level_scored.csv": trade_level_table(HT, ins),
    }
    TAB.mkdir(parents=True, exist_ok=True)
    for name, df in products.items():
        df.to_csv(TAB / name, index=False)
        print(f"tab/{name:28s} {len(df):3d} rows")


if __name__ == "__main__":
    main()
