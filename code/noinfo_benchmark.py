"""The no-information benchmark: Section 4 and Appendix C of the paper.

In a no-information period nobody is informed, so RE and PI both predict the prior
value mu.  The observed no-information price p0 (the mean, over a market's
no-information periods, of each period's mean transaction price) shows where prices
sit when no one can learn anything.  Section 4 sets each market's downward-period
prices beside p0; Appendix C checks the comparison against the treatment of market 4's
period 1, the timing of the no-information periods, trade-level inference, and the use
of opening or closing rather than mean prices.

Called by code/analyze.py (stage 2).  Inputs are the stage-1 files
results/human/trades.csv and results/human/period_table.csv.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
HUMAN = ROOT / "results" / "human"
ANALYSIS = ROOT / "results" / "analysis"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Market 4's period 1 is left out of the benchmark: six of its eight trades are at
# 500-1,500 francs, above every dividend in the design (Appendix C reports both).
EXCLUDED = {(4, 1)}
# Market 4's period 14 is the only no-information period that follows the insider block.
AFTER_INSIDER_BLOCK = {(4, 14)}
DOWN_COLOR, NOINFO_COLOR = "#C44E52", "#444444"
WILD_DRAWS = 9_999
WILD_SEED = 20261007
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def period_prices() -> pd.DataFrame:
    """One row per period: first, mean, and last price, joined to the design."""
    trades = pd.read_csv(HUMAN / "trades.csv")
    period = pd.read_csv(HUMAN / "period_table.csv")
    prices = (trades.groupby(["market", "period"])["price"]
              .agg(p_first="first", p_mean="mean", p_last="last", n_trades="size")
              .reset_index())
    out = prices.merge(period[["market", "period", "state", "info", "side", "vbar", "re_price"]],
                       on=["market", "period"])
    key = list(zip(out["market"], out["period"]))
    out["excluded"] = [k in EXCLUDED for k in key]
    out["after_insider_block"] = [k in AFTER_INSIDER_BLOCK for k in key]
    out["group"] = np.where(out["info"].eq("none"), "no-information",
                            np.where(out["info"].eq("insider") & out["side"].eq("seller"),
                                     "downward", ""))
    return out


def _compare(noinfo: pd.Series, down: pd.Series) -> tuple[float, float, float]:
    """Downward minus no-information mean, Welch p, and rank-sum p (periods as units)."""
    diff = down.mean() - noinfo.mean()
    if len(noinfo) < 2 or len(down) < 2:
        return diff, np.nan, np.nan
    welch = stats.ttest_ind(down, noinfo, equal_var=False).pvalue
    ranksum = stats.mannwhitneyu(down, noinfo, alternative="two-sided").pvalue
    return diff, welch, ranksum


def _span(periods: pd.Series) -> str:
    """Compact period list: 1,2,3,4,14 -> '1--4, 14'."""
    p = sorted(int(x) for x in periods)
    runs, start = [], p[0]
    for a, b in zip(p, p[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}--{a}")
            start = b
    return ", ".join(runs)


def benchmark_table(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in pp.groupby("market"):
        noinfo = g[g["group"].eq("no-information") & ~g["excluded"]]
        down = g[g["group"].eq("downward")]
        diff, welch, ranksum = _compare(noinfo["p_mean"], down["p_mean"])
        rows.append(dict(
            market=int(market), mu=g["vbar"].iloc[0],
            re_downward=", ".join(f"{x:.1f}" for x in sorted(down["re_price"].unique(), reverse=True)),
            noinfo_periods=_span(noinfo["period"]), n_noinfo=len(noinfo),
            p0=noinfo["p_mean"].mean(),
            downward_periods=_span(down["period"]), n_downward=len(down),
            p_downward=down["p_mean"].mean(), difference=diff,
            mu_minus_p0=g["vbar"].iloc[0] - noinfo["p_mean"].mean(),
            p_welch=welch, p_ranksum=ranksum))
    return pd.DataFrame(rows)


def variants_table(pp: pd.DataFrame) -> pd.DataFrame:
    """Appendix C.1, C.2, and C.4: alternative no-information samples and prices."""
    rows = []
    for market, g in pp.groupby("market"):
        down = g[g["group"].eq("downward")]
        noinfo_all = g[g["group"].eq("no-information")]
        noinfo = noinfo_all[~noinfo_all["excluded"]]
        versions = [("Baseline: period means", noinfo, "p_mean")]
        if noinfo_all["excluded"].any():
            versions.append(("Period 1 included", noinfo_all, "p_mean"))
        if noinfo["after_insider_block"].any():
            versions.append(("Before the insider block only", noinfo[~noinfo["after_insider_block"]], "p_mean"))
            versions.append(("After the insider block only", noinfo[noinfo["after_insider_block"]], "p_mean"))
        versions += [("Opening prices", noinfo, "p_first"), ("Closing prices", noinfo, "p_last")]
        for label, sub, col in versions:
            diff, welch, ranksum = _compare(sub[col], down[col])
            rows.append(dict(market=int(market), version=label,
                             noinfo_periods=_span(sub["period"]), n_noinfo=len(sub),
                             noinfo_price=sub[col].mean(), n_downward=len(down),
                             downward_price=down[col].mean(), difference=diff,
                             p_welch=welch, p_ranksum=ranksum))
    return pd.DataFrame(rows)


def _cluster_t(X: np.ndarray, y: np.ndarray, idx: list[np.ndarray]) -> tuple[float, float]:
    """OLS coefficient on the second regressor and its CR1 cluster-robust t statistic."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for i in idx:
        s = X[i].T @ u[i]
        meat += np.outer(s, s)
    g, n, k = len(idx), len(y), X.shape[1]
    vcov = xtx_inv @ meat @ xtx_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta[1], beta[1] / np.sqrt(vcov[1, 1])


def wild_cluster_p(df: pd.DataFrame, rng: np.random.Generator) -> float:
    """Restricted wild cluster bootstrap-t p-value (Webb six-point weights), H0: no
    difference.  Clusters are periods; with six to nine of them the usual
    cluster-robust p-value can be far too small."""
    X = np.column_stack([np.ones(len(df)), df["downward"].to_numpy(float)])
    y = df["price"].to_numpy(float)
    labels = df["period"].to_numpy()
    idx = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    _, t_obs = _cluster_t(X, y, idx)
    fitted0 = np.full_like(y, y.mean())
    resid0 = y - fitted0
    hits = 0
    for _ in range(WILD_DRAWS):
        weights = rng.choice(WEBB, len(idx))
        y_star = fitted0.copy()
        for w, i in zip(weights, idx):
            y_star[i] += w * resid0[i]
        hits += abs(_cluster_t(X, y_star, idx)[1]) >= abs(t_obs)
    return (hits + 1) / (WILD_DRAWS + 1)


def trade_level_table(pp: pd.DataFrame) -> pd.DataFrame:
    """Appendix C.3: trades as observations, periods as clusters or random effects."""
    trades = pd.read_csv(HUMAN / "trades.csv")
    keep = pp[pp["group"].ne("") & ~pp["excluded"]][["market", "period", "group"]]
    d = trades.merge(keep, on=["market", "period"])
    d["downward"] = d["group"].eq("downward").astype(int)
    rng = np.random.default_rng(WILD_SEED)
    rows = []
    for market, g in d.groupby("market"):
        g = g.reset_index(drop=True)
        ols = smf.ols("price ~ downward", g).fit(cov_type="cluster",
                                                 cov_kwds={"groups": g["period"]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mixed = smf.mixedlm("price ~ downward", g, groups=g["period"]).fit(reml=True)
        rows.append(dict(market=int(market), trades=len(g), periods=g["period"].nunique(),
                         difference_ols=ols.params["downward"],
                         p_cluster=ols.pvalues["downward"],
                         p_wild_cluster=wild_cluster_p(g, rng),
                         difference_random_effects=mixed.params["downward"],
                         p_random_effects=mixed.pvalues["downward"]))
    return pd.DataFrame(rows)


def _p(p: float) -> str:
    if pd.isna(p):
        return "--"
    return r"$<$0.001" if p < 0.001 else f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def _p3(p: float) -> str:
    if pd.isna(p):
        return "--"
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def _signed(x: float, digits: int = 1) -> str:
    text = f"{x:+.{digits}f}"
    return text.replace("-", "$-$") if text.startswith("-") else text


def write_tables(bench: pd.DataFrame, variants: pd.DataFrame, trade_level: pd.DataFrame,
                 pp: pd.DataFrame) -> None:
    lines = [r"\begin{tabular}{lrlrrrrrrr}", r"\toprule",
             r" & & & \multicolumn{2}{c}{No-information} & \multicolumn{2}{c}{Downward} & & \multicolumn{2}{c}{$p$-value} \\",
             r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){9-10}",
             r"Market & $\mu$ & $P_{\RE}$ & Periods & $\bar p_0$ & Periods & Mean & Difference & Welch & Rank-sum \\",
             r"\midrule"]
    for _, r in bench.iterrows():
        lines.append(f"{r.market} & {r.mu:.1f} & {r.re_downward} & {r.n_noinfo} & {r.p0:.1f} & "
                     f"{r.n_downward} & {r.p_downward:.1f} & {_signed(r.difference)} & "
                     f"{_p(r.p_welch)} & {_p(r.p_ranksum)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "noinfo_downward.tex").write_text("\n".join(lines) + "\n")

    # Appendix C: the no-information periods one by one.
    rows = []
    for _, r in pp[pp["group"].eq("no-information")].sort_values(["market", "period"]).iterrows():
        timing = "after" if r.after_insider_block else "before"
        note = "excluded" if r.excluded else ""
        rows.append(f"{int(r.market)} & {int(r.period)} & {r.state} & {timing} & {int(r.n_trades)} & "
                    f"{r.p_first:.0f} & {r.p_mean:.1f} & {r.p_last:.0f} & {note}" + r" \\")
    _longtable("appendix_noinfo_periods.tex", "rrlrrrrrl",
               r"The 18 no-information periods", "tab:app-noinfo-periods",
               r"Market & Period & State & Insider block & Trades & First & Mean & Last & Note \\",
               rows, r"\footnotesize")

    rows = []
    for _, r in variants.iterrows():
        rows.append(f"{r.market} & {r.version} & {r.noinfo_periods} & {r.noinfo_price:.1f} & "
                    f"{r.downward_price:.1f} & {_signed(r.difference)} & {_p3(r.p_welch)} & "
                    f"{_p3(r.p_ranksum)}" + r" \\")
    _longtable("appendix_noinfo_variants.tex", r"rp{3.6cm}lrrrrr",
               r"No-information and downward prices under alternative definitions",
               "tab:app-noinfo-variants",
               r"Market & Version & Periods & $\bar p_0$ & Downward & Difference & Welch $p$ & Rank-sum $p$ \\",
               rows, r"\scriptsize")

    rows = []
    for _, r in trade_level.iterrows():
        rows.append(f"{r.market} & {r.trades} & {r.periods} & {_signed(r.difference_ols)} & "
                    f"{_p3(r.p_cluster)} & {_p3(r.p_wild_cluster)} & "
                    f"{_signed(r.difference_random_effects)} & {_p3(r.p_random_effects)}" + r" \\")
    _longtable("appendix_noinfo_trade_level.tex", "rrrrrrrr",
               r"Trade-level comparison of downward and no-information prices",
               "tab:app-noinfo-trades",
               r" & & & \multicolumn{3}{c}{OLS} & \multicolumn{2}{c}{Random period effect} \\ \cmidrule(lr){4-6}\cmidrule(lr){7-8}"
               "\n" r"Market & Trades & Periods & Difference & Cluster $p$ & Wild $p$ & Difference & $p$ \\",
               rows, r"\footnotesize")


def _longtable(filename: str, spec: str, caption: str, label: str, header: str,
               rows: list[str], font: str) -> None:
    ncol = header.splitlines()[-1].count("&") + 1
    lines = [font, rf"\begin{{longtable}}{{{spec}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", header, r"\midrule",
             r"\endfirsthead", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Continued}}}}\\",
             r"\toprule", header, r"\midrule", r"\endhead", r"\midrule",
             rf"\multicolumn{{{ncol}}}{{r}}{{\textit{{Continued on next page}}}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    (TABLES / filename).write_text("\n".join(lines + rows + [r"\end{longtable}"]) + "\n")


def figure(pp: pd.DataFrame) -> None:
    """Five panels: no-information and downward period means against mu and P_RE."""
    span = 100.0
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6))
    for ax, market in zip(axes.flat, range(1, 6)):
        g = pp[pp["market"].eq(market)]
        noinfo = g[g["group"].eq("no-information") & ~g["excluded"]]
        down = g[g["group"].eq("downward")]
        mu = g["vbar"].iloc[0]
        res = sorted(down["re_price"].unique())
        values = list(noinfo["p_mean"]) + list(down["p_mean"]) + [mu] + res
        center = (min(values) + max(values)) / 2
        ax.set_ylim(center - span / 2, center + span / 2)
        ax.axhline(mu, color="black", lw=1.0, ls=":")
        ax.text(1.0, mu, r" $\mu$", transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        for re_ in res:
            ax.axhline(re_, color=DOWN_COLOR, lw=1.0, ls="--")
            label = r" $P_{\mathrm{RE}}$"
            if len(res) > 1:  # market 1: the clue-conditional value differs by period
                label += " (period " + ", ".join(str(int(x)) for x in down.loc[down["re_price"].eq(re_), "period"]) + ")"
            ax.text(1.0, re_, label, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8 if len(res) > 1 else 9, color=DOWN_COLOR)
        ax.scatter(noinfo["period"], noinfo["p_mean"], s=38, facecolors="white",
                   edgecolors=NOINFO_COLOR, lw=1.3, zorder=3, label="No-information period")
        ax.scatter(down["period"], down["p_mean"], s=38, marker="s", color=DOWN_COLOR,
                   zorder=3, label="Downward period")
        if market == 4:
            p1 = g[g["excluded"]].iloc[0]
            top = center + span / 2
            ax.annotate(f"period 1: {p1.p_mean:.0f} (excluded)", xy=(1, top - 1),
                        xytext=(2.0, top - 9), fontsize=7.5, color=NOINFO_COLOR,
                        arrowprops=dict(arrowstyle="->", color=NOINFO_COLOR, lw=.8))
        n = int(g["period"].max())
        ax.set_xlim(0.3, n + 0.7)
        ax.set_xticks(range(1, n + 1, 2 if n > 12 else 1))
        ax.set_title(f"Market {market}", loc="left", fontsize=10)
        ax.set_xlabel("Period", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean price (francs)", fontsize=9)
    legend_ax = axes.flat[5]
    legend_ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="black", ls=":", lw=1.0),
                plt.Line2D([], [], color=DOWN_COLOR, ls="--", lw=1.0)]
    labels += [r"Prior value $\mu$", r"Informed value $P_{\mathrm{RE}}$ (downward state)"]
    legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "noinfo_downward.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def run() -> dict:
    """Write every Section 4 and Appendix C output; return the benchmark table."""
    pp = period_prices()
    bench = benchmark_table(pp)
    variants = variants_table(pp)
    trade_level = trade_level_table(pp)
    for df, name in ((pp[pp["group"].ne("")], "noinfo_downward_periods.csv"),
                     (bench, "noinfo_downward.csv"), (variants, "noinfo_variants.csv"),
                     (trade_level, "noinfo_trade_level.csv")):
        df.to_csv(ANALYSIS / name, index=False)
    write_tables(bench, variants, trade_level, pp)
    figure(pp)
    m1_p2 = pp[(pp["market"] == 1) & (pp["period"] == 2)].iloc[0]
    m1 = pp[pp["market"].eq(1) & pp["group"].eq("no-information")]["p_mean"]
    return {
        "p0": bench.set_index("market")["p0"].to_dict(),
        "market1_p0_with_printed_period2": float((m1.sum() - m1_p2.p_mean + 269) / len(m1)),
        "market4_period1_prices": pd.read_csv(HUMAN / "trades.csv")
            .query("market == 4 and period == 1")["price"].tolist(),
    }
