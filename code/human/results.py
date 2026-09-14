"""Stage 3.  The human-data results the paper cites, from the period table.

The arithmetic is that of the author's earlier figure-digitized study, rerun on the
appendix transcription; only the flagged set in q1_within.py and the sample rows of
robustness_table below change, because the transcription makes those data-quality
caveats obsolete.

Input   results/human/period_table.csv   (stage 2, period_table.py)

Output
  results/human/identity.csv             RE/PI classification of the 36 insider periods
                                         (Proposition 1, Table 1, Figure 1, Table B.4)
  results/human/robustness.csv           the sell-buy contrasts under four inference
                                         schemes and alternative samples (Table E.2)
  results/human/established.csv          familiar-state subsamples, k>=2 and k>=3
                                         (Section 3.3)
  results/human/trajectory.csv           ten-bin transaction paths; bin 1 is the
                                         opening-window check (Section 3.3)
  results/human/endowment_capacity.csv   purchasing capacity at the benchmarks
                                         (assumption A4, footnote in Section 2.2)

The within-period results use markets 3-5.  Market 1 gave the informed a ten-draw
sample rather than the state, and market 2 carried full-information training periods.
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

ROOT = Path(__file__).resolve().parents[2] / "results" / "human"
OUT, TAB = ROOT, ROOT

MEASURES = [
    ("D_first", "$D$ at first trade"),
    ("D_last", "$D$ at last trade"),
    ("D_mean", "$D$, period average"),
    ("D_change", "Change across period"),
    ("slope_D_per_norm", "Slope on position"),
    ("francs_change", "Francs moved in period"),
    # The two distance-to-RE measures.  The first-trade one is the paper's headline
    # number, and it is an absolute distance on purpose: the claim being tested is
    # that a period OPENS at RE, which is a question about how far the first trade
    # sits from RE, not about which side of RE it sits on.  Averaging the signed
    # deviation instead lets a period that opens 25 francs high cancel one that opens
    # 25 francs low, so a side whose openings merely straddle RE would score as if
    # every period had opened on it.
    ("gap_francs_first", "Francs from RE, first trade"),
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


def robustness_table(ins: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """Panel A varies the inference scheme; Panel B varies the sample."""
    d = ins.copy()
    d["buy"] = (d["side"] == "buyer").astype(int)

    rows = []
    # Panel A varies the inference scheme for the path measures.  The two
    # distance-to-RE measures stay out of it and are reported in the main table
    # only; naming them here keeps that choice from riding on their position in
    # MEASURES, as an end-of-list slice did before.
    for col, lab in [m for m in MEASURES if not m[0].startswith("gap_francs")]:
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

    # The sample rows.  Two of them exist to be read against the figure-digitized
    # analysis rather than on their own terms: the flagged set is now a single period
    # (market 1's period 2) instead of four, and no period's transaction count is
    # uncertain any more, so dropping the 23 periods the DIGITIZER could not resolve
    # measures what that sensitivity row was actually measuring.
    samples = {
        "Markets 3-5, all 27 periods": ins,
        "Flagged period dropped": ins[~ins["flagged"].astype(bool)],
        "23 periods merged-run in the digitized record dropped":
            ins[~ins["count_uncertain"].astype(bool)],
        "All five markets": insider(A, markets=(1, 2, 3, 4, 5)),
    }
    for lab, sub in samples.items():
        for col, mlab in [("D_first", "$D$ at first trade"),
                          ("francs_change", "Francs moved in period")]:
            c = contrast(sub, col)
            rows.append(dict(panel="B: sample", sample=lab, measure=mlab, **c))
    return pd.DataFrame(rows)




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






# --------------------------------------------------------------- 3. the baseline




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

    The F_ columns carry the same deciles in francs from RE rather than in D.  They
    are exact, not a second measurement: D = (price - vbar) / (RE - vbar), so
    price - RE = span_francs * (D - 1) identically.  Francs are what the figure plots,
    because D normalizes by a span that differs by side -- the buyer side has to cross
    64 to 180 francs and the seller side 32 to 45 -- so D makes the two sides look
    equidistant from RE when in francs they are not.
    """
    rows = []
    for side in ("seller", "buyer"):
        g = ins[ins["side"] == side]
        for j in range(1, 11):
            v = g[f"D_dec{j}"].dropna()
            f = (g["span_francs"] * (g[f"D_dec{j}"] - 1)).dropna()
            rows.append(dict(sample="human m3-5 insider (primary)", side=side,
                             bin=j, nbins=10, bin_lo=(j - 1) / 10, bin_hi=j / 10,
                             n_periods=len(v), D_mean=v.mean(),
                             D_sd=v.std(ddof=1) if len(v) > 1 else np.nan,
                             D_median=v.median(),
                             se=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                             F_mean=f.mean(), F_median=f.median(),
                             F_sd=f.std(ddof=1) if len(f) > 1 else np.nan,
                             F_se=(f.std(ddof=1) / np.sqrt(len(f))
                                   if len(f) > 1 else np.nan)))
    T = pd.DataFrame(rows)
    T["layer"] = "L3_appendix_transcribed"
    return T




# --------------------------------------------------------------- driver


def main() -> None:
    A = pd.read_csv(TAB / "period_table.csv")
    ins = insider(A)
    products = {
        "identity.csv": identity_table(),
        "robustness.csv": robustness_table(ins, A),
        "established.csv": established_table(ins),
        "endowment_capacity.csv": endowment_capacity(),
        "trajectory.csv": trajectory_table(ins),
    }
    TAB.mkdir(parents=True, exist_ok=True)
    for name, df in products.items():
        df.to_csv(TAB / name, index=False)
        print(f"results/human/{name:24s} {len(df):3d} rows")


if __name__ == "__main__":
    main()
