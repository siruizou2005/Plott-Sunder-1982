#!/usr/bin/env python3
"""Reproduce the human, symmetric-market, and disclosure-ladder results.

Stage 2 of the replication package.  The script reads the shipped inputs in
``data/`` and the stage-1 human outputs in ``results/human/`` (written by
``code/human/run_all.py``), and writes every derived CSV, LaTeX table, figure, and the
machine-readable summary used by the paper.

The simulation engine records the constructed markets A and B under the internal
codes 7 and 8.  Those codes survive only in run IDs (``m7_*``, ``m8_*``) and in the
shipped engine records; every output of this script calls the markets A and B.
"""

from __future__ import annotations

import itertools
import json
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import noinfo_benchmark
import schematics
from printsize import RC, width


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HUMAN = ROOT / "results" / "human"          # stage-1 outputs of code/human/run_all.py
LLM_RUNS = DATA / "llm" / "metrics"
LLM_TRANSACTIONS = DATA / "llm" / "llm_transactions.csv"
ANALYSIS = ROOT / "results" / "analysis"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Internal engine codes of markets A and B.
MARKET_LABEL = {7: "A", 8: "B"}


def market_label(market: object) -> str:
    """Paper name of a market: 1-5 unchanged, engine codes 7 and 8 become A and B."""
    try:
        code = int(market)
    except (TypeError, ValueError):
        return str(market)
    return MARKET_LABEL.get(code, str(code))


def to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write an output CSV with markets named as in the paper."""
    out = df.copy()
    if "market" in out.columns:
        out["market"] = out["market"].map(market_label)
    out.to_csv(path, index=False)


RNG = np.random.default_rng(20260907)

# The paper calls a period by its direction of adjustment: downward when the informed
# value P_RE lies below the prior value mu (every insider would sell at mu), upward
# when it lies above.  Engine records and CSVs keep the side names seller and buyer.
PERIOD_TYPE = {"seller": "Downward", "buyer": "Upward"}
UP_COLOR, DOWN_COLOR = "#4C72B0", "#C44E52"


def period_type(side: str) -> str:
    return PERIOD_TYPE[side]


def latex_escape(value: object) -> str:
    """Escape compact strings written into generated LaTeX tables."""
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"),
                     ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def ensure_dirs() -> None:
    for path in (ANALYSIS, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def load_metric_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return period-price rows and period-level mechanism rows."""
    discovery_rows: list[dict] = []
    mechanism_rows: list[dict] = []
    for path in sorted((LLM_RUNS).glob("**/*.metrics.json")):
        rel = path.relative_to(LLM_RUNS)
        group = rel.parts[0]
        run_id = path.parent.name
        obj = json.loads(path.read_text())
        for session_id, session in obj["sessions"].items():
            meta = session["meta"]
            market = int(meta["market"])
            seed = int(meta.get("seed", -1))
            paper = session["paper"]
            llm = session.get("llm", {})

            discovery = paper["discovery_by_informed_side"]["periods"]
            prices = {int(r["period"]): r for r in paper.get("prices", [])}
            for row in discovery:
                pr = prices.get(int(row["period"]), {})
                denominator = row["re"] - row["uninformed_level"]
                discovery_rows.append(
                    dict(group=group, run_id=run_id, session=session_id,
                         market=market, seed=seed, **row,
                         n_trades=pr.get("n_trades"),
                         first_price=pr.get("first_price"),
                         last_price=pr.get("last_price"),
                         first_discovery=((pr.get("first_price") - row["uninformed_level"]) /
                                          denominator if pr.get("first_price") is not None else np.nan),
                         last_discovery=((pr.get("last_price") - row["uninformed_level"]) /
                                         denominator if pr.get("last_price") is not None else np.nan),
                         change_discovery=((pr.get("last_price") - pr.get("first_price")) /
                                           denominator if pr.get("first_price") is not None and
                                           pr.get("last_price") is not None else np.nan))
                )

            post = {int(r["period"]): r for r in llm.get("posterior_convergence", [])}
            basis = {int(r["period"]): r for r in llm.get("basis_drift", [])}
            eff = {int(k): v for k, v in paper.get("efficiency", {}).items() if v}
            profits = {int(k): v for k, v in paper.get("insider_profit_ratio", {}).items() if v}
            disc_by_period = {int(r["period"]): r for r in discovery}
            for period in sorted(set(prices) | set(post) | set(eff) | set(profits)):
                pr = prices.get(period, {})
                po = post.get(period, {})
                ba = basis.get(period, {})
                counts = ba.get("counts", {})
                uninformed_basis_n = sum(counts.get(k, 0) for k in
                                         ("prior", "price", "others_behavior"))
                drow = disc_by_period.get(period, {})
                prof = profits.get(period, {})
                erow = eff.get(period, {})
                mechanism_rows.append(dict(
                    group=group, run_id=run_id, session=session_id,
                    market=market, seed=seed, period=period,
                    state=pr.get("state"), info=pr.get("info"),
                    side=drow.get("side"), discovery=drow.get("discovery"),
                    n_trades=pr.get("n_trades"),
                    true_state_belief=po.get("uninformed_mean"),
                    price_basis_share=(counts.get("price", 0) / uninformed_basis_n
                                       if uninformed_basis_n else np.nan),
                    E_pct=erow.get("E_pct"), TE_pct=erow.get("TE_pct"),
                    insider_profit=prof.get("insider_mean"),
                    uninformed_profit=prof.get("uninformed_mean"),
                    insider_advantage_pct=prof.get("ratio_pct"),
                ))
    return pd.DataFrame(discovery_rows), pd.DataFrame(mechanism_rows)


def exact_sign_flip_p(delta: np.ndarray) -> float:
    """Two-sided randomization p-value under sign exchangeability."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    observed = abs(delta.mean())
    n = len(delta)
    if n == 0:
        return np.nan
    if n <= 20:
        vals = []
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            vals.append(abs(np.mean(delta * np.asarray(signs))))
        return float(np.mean(np.asarray(vals) >= observed - 1e-12))
    signs = RNG.choice((-1.0, 1.0), size=(200_000, n))
    return float(np.mean(np.abs((signs * delta).mean(axis=1)) >= observed))


def cluster_bootstrap_mean_ci(df: pd.DataFrame, value: str,
                              cluster: str = "run_id",
                              draws: int = 20_000) -> tuple[float, float]:
    groups = [g[value].dropna().to_numpy() for _, g in df.groupby(cluster)]
    if not groups:
        return np.nan, np.nan
    estimates = np.empty(draws)
    for b in range(draws):
        chosen = RNG.integers(0, len(groups), len(groups))
        estimates[b] = np.concatenate([groups[i] for i in chosen]).mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def session_equal_path(g: pd.DataFrame, draws: int = 20_000) -> pd.DataFrame:
    """Return session-equal path means and session-bootstrap intervals.

    Transactions are first averaged within period-bin and then within session-bin.
    Resampling whole sessions preserves the dependence among periods from the same
    model history.  The displayed estimand gives each session equal weight.
    """
    period_bin = (g.groupby(["run_id", "session", "period", "bin"], observed=True)
                    ["discovery"].mean().reset_index())
    session_bin = (period_bin.groupby(["run_id", "session", "bin"], observed=True)
                             ["discovery"].mean().reset_index())
    panel = session_bin.pivot_table(index=["run_id", "session"], columns="bin",
                                    values="discovery", aggfunc="first")
    means = panel.mean(axis=0)
    boot = np.empty((draws, panel.shape[1]))
    values = panel.to_numpy(dtype=float)
    for b in range(draws):
        selected = RNG.integers(0, len(panel), len(panel))
        boot[b, :] = np.nanmean(values[selected, :], axis=0)
    return pd.DataFrame({
        "bin": means.index.astype(int),
        "mean": means.to_numpy(),
        "low": np.nanquantile(boot, .025, axis=0),
        "high": np.nanquantile(boot, .975, axis=0),
        "sessions": panel.notna().sum(axis=0).to_numpy(),
    })


def human_analysis() -> dict:
    period = pd.read_csv(HUMAN / "period_table.csv")
    trades = pd.read_csv(HUMAN / "trades.csv")
    transcription = pd.read_csv(DATA / "human" / "plott_sunder_1982_prices.csv")
    identity = pd.read_csv(HUMAN / "identity.csv")
    main = period.query("market in [3, 4, 5] and info == 'insider'").copy()
    # Signed deviations from the RE point benchmark (negative = below it) are the main
    # human franc outcomes; the gap_francs_* columns are absolute deviations.
    main["dev_francs_first"] = main["p_first"] - main["re_price"]
    main["dev_francs_last"] = main["p_last"] - main["re_price"]

    rows = []
    for side, g in main.groupby("side"):
        rows.append(dict(
            side=side, periods=len(g), trades=int(g["n_trades"].sum()),
            first_D_mean=g["D_first"].mean(), last_D_mean=g["D_last"].mean(),
            mean_D=g["D_mean"].mean(), change_D_mean=g["D_change"].mean(),
            first_gap_mean=g["gap_francs_first"].mean(),
            first_gap_median=g["gap_francs_first"].median(),
            last_gap_mean=g["gap_francs_last"].mean(),
            last_gap_median=g["gap_francs_last"].median(),
            francs_change_mean=g["francs_change"].mean(),
            francs_change_median=g["francs_change"].median(),
            first_dev_mean=g["dev_francs_first"].mean(),
            first_dev_median=g["dev_francs_first"].median(),
            last_dev_mean=g["dev_francs_last"].mean(),
            last_dev_median=g["dev_francs_last"].median(),
        ))
    summary = pd.DataFrame(rows).sort_values("side")
    to_csv(summary, ANALYSIS / "human_summary.csv")

    sell = main.query("side == 'seller'")
    buy = main.query("side == 'buyer'")
    tests = []
    for col in ("D_first", "D_last", "D_mean", "D_change",
                "dev_francs_first", "dev_francs_last",
                "gap_francs_first", "gap_francs_last", "francs_change"):
        result = stats.ttest_ind(sell[col], buy[col], equal_var=False, nan_policy="omit")
        tests.append(dict(measure=col, sell_mean=sell[col].mean(),
                          buy_mean=buy[col].mean(), difference=sell[col].mean()-buy[col].mean(),
                          p_welch=result.pvalue))
    to_csv(pd.DataFrame(tests), ANALYSIS / "human_tests.csv")

    # Distributional summaries quoted in the text and the supplementary appendix.
    dist = []
    for side, g in main.groupby("side"):
        first, last, change = g["dev_francs_first"], g["dev_francs_last"], g["francs_change"]
        ci = stats.t.interval(.95, len(g) - 1, loc=g["D_first"].mean(),
                              scale=stats.sem(g["D_first"]))
        dist.append(dict(
            side=side, periods=len(g),
            open_below=int((first < 0).sum()), open_at=int((first == 0).sum()),
            open_above=int((first > 0).sum()),
            close_below=int((last < 0).sum()), close_at=int((last == 0).sum()),
            close_above=int((last > 0).sum()),
            price_rises=int((change > 0).sum()), price_falls=int((change < 0).sum()),
            price_unchanged=int((change == 0).sum()),
            open_abs_dev_mean=first.abs().mean(), open_abs_dev_median=first.abs().median(),
            close_abs_dev_mean=last.abs().mean(), close_abs_dev_median=last.abs().median(),
            D_first_ci95_low=ci[0], D_first_ci95_high=ci[1],
        ))
    to_csv(pd.DataFrame(dist).sort_values("side"), ANALYSIS / "human_distribution.csv")

    colors = {"seller": DOWN_COLOR, "buyer": UP_COLOR}

    # Within-period human paths in francs relative to RE.
    h = trades.query("market in [3, 4, 5] and info == 'insider'").copy()
    h["u"] = np.where(h["n_trades_in_period"] > 1,
                      (h["trade_index"] - 1) / (h["n_trades_in_period"] - 1), 0)
    h["gap"] = h["price"] - h["re_price"]
    # Printed at 0.94 of the text width (Figure 5).
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(width(0.94), 2.75), sharey=True, layout="constrained")
        for ax, side in zip(axes, ("seller", "buyer")):
            g = h.query("side == @side")
            for _, p in g.groupby(["market", "period"]):
                ax.plot(p["u"], p["gap"], color=colors[side], alpha=.18, lw=.7)
            bins = pd.cut(g["u"], np.linspace(0, 1, 11), include_lowest=True, labels=False)
            g = g.assign(bin=bins)
            by_period = g.groupby(["market", "period", "bin"], observed=True)["gap"].mean().reset_index()
            agg = by_period.groupby("bin")["gap"].agg(["mean", "std", "count"]).reset_index()
            x = (agg["bin"] + .5) / 10
            se = agg["std"] / np.sqrt(agg["count"])
            ax.plot(x, agg["mean"], color=colors[side], lw=2.2)
            ax.fill_between(x, agg["mean"] - 1.96*se, agg["mean"] + 1.96*se,
                            color=colors[side], alpha=.15)
            ax.axhline(0, color="black", lw=.9)
            ax.set_title(f"{period_type(side)} periods", loc="left")
            ax.set_xlabel("Trade order in period, $u$")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(r"Price minus $P_{\mathrm{RE}}$ (francs)")
        fig.savefig(FIGURES / "human_paths.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    ctab = pd.crosstab(identity["side"], identity["predictions_differ"])
    return {
        "all_human_trades_transcribed": int(len(transcription)),
        "main_human_trades": int(main["n_trades"].sum()),
        "main_human_periods": int(len(main)),
        "selection_crosstab": ctab.to_dict(),
        "summary": summary.set_index("side").to_dict(orient="index"),
    }


def design_table() -> pd.DataFrame:
    """The three conditions that vary with the direction of adjustment (Section 3.3).

    Computed from the dividend schedules and priors in market_specs():
      distance       P_RE - mu, the distance prices must travel;
      net insiders   insiders who would buy at mu minus insiders who would sell at mu
                     (two insiders per type);
      uninformed     uninformed traders whose prior value already lets them trade at
                     P_RE in the direction of adjustment (prior value below P_RE in a
                     downward state); none can in an upward state, since P_RE > mu.
    """
    specs = market_specs()
    rows = []
    for market in (3, 4, 5, 7, 8):
        spec = specs[market]
        prior_ev = {t: sum(p * d for p, d in zip(spec["prior"], vals))
                    for t, vals in spec["div"].items()}
        mu = max(prior_ev.values())
        per_type = spec["insiders"] // len(spec["div"])
        uninformed_per_type = spec["n"] // len(spec["div"]) - per_type
        cells = {"buyer": [], "seller": []}
        for i, state in enumerate(spec["states"]):
            values = {t: vals[i] for t, vals in spec["div"].items()}
            re = max(values.values())
            side = "buyer" if re > mu else "seller"
            buy = sum(per_type for v in values.values() if v > mu)
            sell = sum(per_type for v in values.values() if v < mu)
            helpers = (sum(uninformed_per_type for v in prior_ev.values() if v < re)
                       if side == "seller" else 0)
            cells[side].append((re - mu, buy - sell, helpers))
        row = dict(market=market)
        for side, label in (("buyer", "up"), ("seller", "down")):
            vals = sorted(set(cells[side]), key=lambda c: c[0])
            row[f"{label}_distance"] = " or ".join(f"{c[0]:+g}" for c in vals)
            row[f"{label}_net_insiders"] = " or ".join(sorted({f"{c[1]:+d}" for c in vals}))
            row[f"{label}_uninformed_at_target"] = " or ".join(sorted({str(c[2]) for c in vals}))
        rows.append(row)
    out = pd.DataFrame(rows)
    to_csv(out, ANALYSIS / "design_asymmetries.csv")
    return out


def select_samples(discovery: pd.DataFrame) -> pd.DataFrame:
    original = discovery.query("group in ['m1','m2','m3','m4','m5']").copy()
    market4 = discovery.query("group == 'm4'").copy()
    symmetric = discovery[
        discovery["run_id"].str.fullmatch(r"m[78]_ctrl_(42|43|44)")
    ].copy()
    original["sample"] = "Original markets 1-5 (background)"
    market4["sample"] = "Original market 4"
    symmetric["sample"] = "Symmetric markets A and B"
    return pd.concat([market4, symmetric, original], ignore_index=True)


def symmetric_analysis(discovery: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sample = select_samples(discovery)
    to_csv(sample, ANALYSIS / "llm_price_discovery_periods.csv")
    rows = []
    for (name, side), g in sample.groupby(["sample", "side"]):
        lo, hi = cluster_bootstrap_mean_ci(g, "discovery")
        rows.append(dict(sample=name, side=side, sessions=g["run_id"].nunique(),
                         periods=len(g), first_discovery=g["first_discovery"].mean(),
                         mean_discovery=g["discovery"].mean(),
                         last_discovery=g["last_discovery"].mean(),
                         change_discovery=g["change_discovery"].mean(),
                         ci_low=lo, ci_high=hi,
                         mean_francs_moved=(g["mean_price"]-g["uninformed_level"]).mean()))
    out = pd.DataFrame(rows)
    to_csv(out, ANALYSIS / "symmetric_comparison.csv")

    # Session-level buy/sell gaps provide the correct small-sample unit.
    gaps = sample.groupby(["sample", "run_id", "side"])["discovery"].mean().unstack()
    gaps["sell_minus_buy"] = gaps["seller"] - gaps["buyer"]
    to_csv(gaps.reset_index(), ANALYSIS / "symmetric_session_gaps.csv")

    # Direct within-period comparison: market 4 is the parent design for markets A and B.
    tx = pd.read_csv(LLM_TRANSACTIONS)
    tx["sample"] = np.where(
        tx["group"].eq("m4"), "Original market 4",
        np.where(tx["run_id"].str.fullmatch(r"m[78]_ctrl_(42|43|44)"),
                 "Symmetric markets A and B", "")
    )
    tx = tx[tx["sample"].ne("")].copy()
    # Printed at 0.78 of the text width (Figure 6).
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(width(0.78), 2.65), sharex=True, sharey=True,
                                 layout="constrained")
        colors = {"buyer": UP_COLOR, "seller": DOWN_COLOR}
        for ax, name in zip(axes, ("Original market 4", "Symmetric markets A and B")):
            panel = tx.query("sample == @name")
            for side in ("buyer", "seller"):
                g = panel.query("side == @side").copy()
                bins = pd.cut(g["u"], np.linspace(0, 1, 11), include_lowest=True, labels=False)
                g["bin"] = bins
                agg = session_equal_path(g)
                x = (agg["bin"].to_numpy() + .5) / 10
                ax.plot(x, agg["mean"], color=colors[side], lw=2.0,
                        label=f"{period_type(side)} periods")
                ax.fill_between(x, agg["low"], agg["high"],
                                color=colors[side], alpha=.14)
            ax.axhline(1, color="black", lw=.9, ls="--")
            ax.set_title({"Original market 4": "Market 4",
                          "Symmetric markets A and B": "Symmetric markets A and B"}[name], loc="left")
            ax.set_xlabel("Trade order in period, $u$")
            ax.set_ylim(-.45, 1.65)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Price position, $D$")
        axes[1].legend(frameon=False, loc="lower right")
        fig.savefig(FIGURES / "symmetric_markets.pdf", metadata={"CreationDate": None})
        plt.close(fig)
    return out, {k: v for k, v in gaps.groupby(level=0)["sell_minus_buy"].mean().items()}


def ladder_map() -> dict[str, str]:
    return {"control": "0", "disclosed": "1", "ladder1b": "1b",
            "ladder2": "2", "ladder3": "3"}


def ladder_selected(discovery: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for _, row in discovery.iterrows():
        run = row["run_id"]
        group = row["group"]
        if group == "control" and run in {"m7_ctrl_42", "m7_ctrl_45", "m8_ctrl_42", "m8_ctrl_44"}:
            keep.append(True)
        elif group == "disclosed" and run in {"m7_disc_42", "m8_disc_42"}:
            keep.append(True)
        elif group == "ladder1b" and run in {"m7_lad1b_42", "m8_lad1b_42"}:
            keep.append(True)
        elif group == "ladder2" and run in {"m7_lad2_42", "m7_lad2_45", "m8_lad2_42", "m8_lad2_44"}:
            keep.append(True)
        elif group == "ladder3" and run in {"m7_lad3_42", "m7_lad3_45", "m8_lad3_42", "m8_lad3_44"}:
            keep.append(True)
        else:
            keep.append(False)
    x = discovery.loc[keep].copy()
    x["rung"] = x["group"].map(ladder_map())
    return x


def ladder_analysis(discovery: pd.DataFrame, mechanisms: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    lad = ladder_selected(discovery)
    to_csv(lad, ANALYSIS / "ladder_price_discovery_periods.csv")
    order = ["0", "1", "1b", "2", "3"]

    session_details: list[dict] = []

    def paired(before: str, after: str, label: str,
               seed42_only: bool = False) -> list[dict]:
        x = lad[lad["rung"].isin([before, after])].copy()
        if seed42_only:
            x = x.query("seed == 42")
        w = x.pivot_table(index=["market", "seed", "period", "side"],
                          columns="rung", values="discovery", aggfunc="first").dropna()
        rows = []
        for side, g in w.reset_index().groupby("side"):
            g = g.copy()
            g["delta"] = g[after] - g[before]
            sessions = (g.groupby(["market", "seed"])
                          .agg(periods=("delta", "size"),
                               before_mean=(before, "mean"),
                               after_mean=(after, "mean"),
                               mean_change=("delta", "mean"))
                          .reset_index())
            for _, session_row in sessions.iterrows():
                session_details.append(dict(
                    step=label, side=side, before_rung=before, after_rung=after,
                    market=int(session_row.market), seed=int(session_row.seed),
                    paired_periods=int(session_row.periods),
                    before_mean=session_row.before_mean,
                    after_mean=session_row.after_mean,
                    mean_change=session_row.mean_change,
                ))
            delta = sessions["mean_change"].to_numpy()
            rows.append(dict(
                step=label, side=side, sessions=len(sessions), periods=len(g),
                before_mean=sessions["before_mean"].mean(),
                after_mean=sessions["after_mean"].mean(),
                mean_change=delta.mean(), improved_sessions=int((delta > 0).sum()),
                p_session_sign_flip=exact_sign_flip_p(delta),
            ))
        return rows

    effects = pd.DataFrame(
        paired("0", "1", "0 to 1: dividends and clue count", True)
        + paired("1", "1b", "1 to 1b: implementation changes", True)
        + paired("1b", "2", "1b to 2: period announcement", True)
        + paired("2", "3", "2 to 3: same insiders")
        + paired("0", "3", "0 to 3: all steps")
    )
    to_csv(effects, ANALYSIS / "ladder_paired_effects.csv")
    session_effects = pd.DataFrame(session_details)
    to_csv(session_effects, ANALYSIS / "ladder_session_effects.csv")

    # Mechanism outcomes for the full seed-42 ladder.
    m = mechanisms.query("market in [7, 8] and seed == 42 and info == 'insider'").copy()
    allowed = {
        "m7_ctrl_42":"0", "m8_ctrl_42":"0", "m7_disc_42":"1", "m8_disc_42":"1",
        "m7_lad1b_42":"1b", "m8_lad1b_42":"1b", "m7_lad2_42":"2", "m8_lad2_42":"2",
        "m7_lad3_42":"3", "m8_lad3_42":"3",
    }
    m = m[m["run_id"].isin(allowed)].copy(); m["rung"] = m["run_id"].map(allowed)
    grouped = []
    for (rung, side), g in m.groupby(["rung", "side"], dropna=False):
        if pd.isna(side):
            continue
        grouped.append(dict(
            rung=rung, side=side, periods=len(g), price_periods=g["discovery"].notna().sum(),
            discovery=g["discovery"].mean(), true_state_belief=g["true_state_belief"].mean(),
            price_basis_share=g["price_basis_share"].mean(), trades=g["n_trades"].mean(),
            TE_pct=g["TE_pct"].mean(),
            insider_advantage_pct=(100*g["insider_profit"].mean()/g["uninformed_profit"].mean()),
        ))
    mech = pd.DataFrame(grouped)
    mech["rung_order"] = mech["rung"].map({v:i for i,v in enumerate(order)})
    mech = mech.sort_values(["rung_order", "side"]).drop(columns="rung_order")
    to_csv(mech, ANALYSIS / "ladder_mechanisms_seed42.csv")

    # Printed at 0.88 of the text width (Figure 7).
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(width(0.88), 2.75), sharey=True, layout="constrained")
        for ax, side in zip(axes, ("buyer", "seller")):
            g = (lad.query("seed == 42 and side == @side")
                    .groupby(["market", "rung"])["discovery"].mean().unstack())
            g = g.reindex(columns=order)
            for market, row in g.iterrows():
                ax.plot(range(len(order)), row.to_numpy(), color="#999999", alpha=.8,
                        marker="o", ms=3, lw=.9,
                        label=f"Market {market_label(market)}" if side == "buyer" else None)
            ax.plot(range(len(order)), g.mean(axis=0).to_numpy(), marker="o", ms=4.5,
                    color={"buyer": UP_COLOR, "seller": DOWN_COLOR}[side], lw=2.0,
                    label="Two-session mean" if side == "buyer" else None)
            ax.axhline(1, color="black", lw=.9, ls="--")
            ax.set_xticks(range(len(order)), order)
            ax.set_xlabel("Disclosure step")
            ax.set_title(f"{period_type(side)} periods", loc="left")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel("Mean price position, $D$")
        fig.savefig(FIGURES / "disclosure_ladder.pdf", metadata={"CreationDate": None})
        plt.close(fig)

    # Put the market-level heterogeneity in the main text rather than hiding it in an
    # average. Each marker is one market-seed session-pair effect.
    focus_steps = ["1b to 2: period announcement", "2 to 3: same insiders"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.15), sharey=True)
    colors = {"buyer": UP_COLOR, "seller": DOWN_COLOR}
    markers = {"buyer": "o", "seller": "s"}
    for ax, step in zip(axes, focus_steps):
        panel = session_effects.query("step == @step")
        for side, offset in (("buyer", -.07), ("seller", .07)):
            g = panel.query("side == @side")
            ax.scatter(g["market"] + offset, g["mean_change"], s=50,
                       marker=markers[side], color=colors[side],
                       label=f"{period_type(side)} periods", zorder=3)
            for _, row in g.iterrows():
                if len(g) > 2:
                    ax.annotate(str(int(row.seed)),
                                (row.market + offset, row.mean_change),
                                xytext=(3, 3), textcoords="offset points",
                                fontsize=7, color=colors[side])
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks([7, 8], [f"Market {market_label(m)}" for m in (7, 8)])
        ax.set_xlim(6.6, 8.4)
        ax.set_title("Step 1b to 2: period announcement" if "announcement" in step
                     else "Step 2 to 3: same insiders")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Paired change in mean price position")
    axes[1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "disclosure_market_heterogeneity.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    return effects, mech


def market_specs() -> dict[int, dict]:
    """Parameters used by the original and symmetric markets."""
    return {
        1: dict(n=9, insiders=3, states=("X", "Y"), prior=(1/3, 2/3), periods=11,
                info="none 1--4; insider 5--8; all 9--11", usd=.002,
                div={"I": (150, 350), "II": (250, 300), "III": (300, 100)}),
        2: dict(n=12, insiders=6, states=("X", "Y"), prior=(1/3, 2/3), periods=11,
                info="none 1--4; all 5--6; insider 7--11", usd=.002,
                div={"I": (100, 350), "II": (200, 300), "III": (240, 175)}),
        3: dict(n=12, insiders=6, states=("X", "Y"), prior=(.4, .6), periods=12,
                info="none 1--2; insider 3--10; all 11--12", usd=.003,
                div={"I": (400, 100), "II": (300, 150), "III": (125, 175)}),
        4: dict(n=12, insiders=6, states=("X", "Y"), prior=(.4, .6), periods=14,
                info="none 1--4; insider 5--13; none 14", usd=.003,
                div={"I": (375, 100), "II": (275, 150), "III": (100, 175)}),
        5: dict(n=12, insiders=6, states=("X", "Y", "Z"), prior=(.35, .25, .40),
                periods=13, info="none 1--3; insider 4--13", usd=.003,
                div={"I": (120, 170, 320), "II": (155, 245, 135),
                     "III": (180, 100, 160)}),
        7: dict(n=12, insiders=6, states=("X", "Y"), prior=(.6, .4), periods=14,
                info="none 1--4; insider 5--13; none 14", usd=.003,
                div={"I": (360, 110), "II": (330, 130), "III": (290, 160)}),
        8: dict(n=12, insiders=6, states=("X", "Y"), prior=(.6, .4), periods=14,
                info="none 1--4; insider 5--13; none 14", usd=.003,
                div={"I": (380, 180), "II": (350, 200), "III": (400, 100)}),
    }


def run_is_used(group: str, run_id: str) -> bool:
    if group in {"m1", "m2", "m3", "m4", "m5", "rounds"}:
        return True
    allowed = {
        "control": {"m7_ctrl_42", "m7_ctrl_43", "m7_ctrl_44", "m7_ctrl_45",
                    "m8_ctrl_42", "m8_ctrl_43", "m8_ctrl_44"},
        "disclosed": {"m7_disc_42", "m8_disc_42"},
        "ladder1b": {"m7_lad1b_42", "m8_lad1b_42"},
        "ladder2": {"m7_lad2_42", "m7_lad2_45", "m8_lad2_42", "m8_lad2_44"},
        "ladder3": {"m7_lad3_42", "m7_lad3_45", "m8_lad3_42", "m8_lad3_44"},
    }
    return run_id in allowed.get(group, set())


def build_run_inventory() -> pd.DataFrame:
    labels = {"control": "0", "disclosed": "1", "ladder1b": "1b",
              "ladder2": "2", "ladder3": "3", "rounds": "round check"}
    rows: list[dict] = []
    for path in sorted((LLM_RUNS).glob("**/*.metrics.json")):
        rel = path.relative_to(LLM_RUNS)
        group, run_id = rel.parts[0], path.parent.name
        if not run_is_used(group, run_id):
            continue
        obj = json.loads(path.read_text())
        meta_paths = list(path.parent.glob("*.meta.json"))
        outer = json.loads(meta_paths[0].read_text()) if meta_paths else {}
        config = outer.get("config", {})
        states = outer.get("sequence", {}).get("states", [])
        info = outer.get("sequence", {}).get("info", [])
        model = ((config.get("agents") or [{}])[0].get("model") or "unknown")
        rounds = config.get("max_rounds_per_period", 3)
        for session_id, session in obj["sessions"].items():
            smeta = session["meta"]
            prices = {int(p["period"]): p for p in session["paper"].get("prices", [])}
            insider_periods = [i + 1 for i, x in enumerate(info) if x == "insider"]
            traded = [p for p in insider_periods if prices.get(p, {}).get("n_trades", 0) > 0]
            no_trade = [p for p in insider_periods if p not in traded]
            treatment = (labels.get(group, "original") if group not in {"m1", "m2", "m3", "m4", "m5"}
                         else "original")
            if group == "control" and run_id in {"m7_ctrl_42", "m7_ctrl_43", "m7_ctrl_44",
                                                 "m8_ctrl_42", "m8_ctrl_43", "m8_ctrl_44"}:
                treatment = "0 / symmetric"
            rows.append(dict(
                group=group, run_id=run_id, session=session_id,
                market=int(smeta["market"]), treatment=treatment,
                seed=int(smeta.get("seed", config.get("seed", -1))),
                model=model, rounds=int(rounds),
                preset=smeta.get("sequence_preset", config.get("sequence_preset", "")),
                state_sequence="".join(states),
                insider_periods=len(insider_periods), valid_periods=len(traded),
                no_trade_periods=",".join(map(str, no_trade)) if no_trade else "--",
                status=outer.get("status", "done"),
            ))
    out = pd.DataFrame(rows).sort_values(["group", "market", "seed", "run_id"])
    to_csv(out, ANALYSIS / "llm_run_inventory.csv")
    return out


def appendix_analysis(discovery: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the audit, parameter, run-list, and robustness appendix data."""
    period = pd.read_csv(HUMAN / "period_table.csv")
    validation = pd.read_csv(HUMAN / "validation.csv")
    robustness = pd.read_csv(HUMAN / "robustness.csv")
    main_h = period.query("market in [3, 4, 5] and info == 'insider'").copy()
    to_csv(validation, ANALYSIS / "human_validation_61_periods.csv")
    to_csv(main_h, ANALYSIS / "human_main_27_periods.csv")

    by_market = (main_h.groupby(["market", "side"])
                 .agg(periods=("period", "size"), trades=("n_trades", "sum"),
                      D_first=("D_first", "mean"), D_mean=("D_mean", "mean"),
                      D_last=("D_last", "mean"), change_francs=("francs_change", "mean"))
                 .reset_index())
    to_csv(by_market, ANALYSIS / "human_results_by_market.csv")

    specs = market_specs()
    structures, dividends, benchmarks = [], [], []
    for market, spec in specs.items():
        prior_text = ", ".join(f"{s}:{p:.2f}" for s, p in zip(spec["states"], spec["prior"]))
        structures.append(dict(market=market, traders=spec["n"], informed=spec["insiders"],
                               states="/".join(spec["states"]), prior=prior_text,
                               periods=spec["periods"], information_schedule=spec["info"],
                               francs_per_dollar=int(round(1/spec["usd"]))))
        prior_ev = {t: sum(p*d for p, d in zip(spec["prior"], vals))
                    for t, vals in spec["div"].items()}
        vbar = max(prior_ev.values())
        for t, vals in spec["div"].items():
            dividends.append(dict(market=market, type=t,
                                  **{s: vals[i] for i, s in enumerate(spec["states"])}))
        if market != 1:
            for i, state in enumerate(spec["states"]):
                re = max(vals[i] for vals in spec["div"].values())
                benchmarks.append(dict(market=market, state=state, vbar=vbar,
                                       RE=re, PI=max(re, vbar), distance=re-vbar,
                                       side="buyer" if re > vbar else "seller"))
    structures = pd.DataFrame(structures)
    dividends = pd.DataFrame(dividends).fillna("--")
    benchmarks = pd.DataFrame(benchmarks)
    to_csv(structures, ANALYSIS / "market_structures.csv")
    to_csv(dividends, ANALYSIS / "market_dividends.csv")
    to_csv(benchmarks, ANALYSIS / "theory_benchmarks_exact_signal.csv")

    identity = pd.read_csv(HUMAN / "identity.csv")
    m1 = identity.query("market == 1").copy()
    m1["posterior_X"] = m1["clue_ones"].apply(
        lambda k: ((1/3)*(.2**k)*(.8**(10-k))) /
                  ((1/3)*(.2**k)*(.8**(10-k)) + (2/3)*(.4**k)*(.6**(10-k))))
    to_csv(m1, ANALYSIS / "market1_posterior_benchmarks.csv")

    lad = ladder_selected(discovery)
    full42 = lad.query("seed == 42")
    wide = full42.pivot_table(index=["market", "period", "side"], columns="rung",
                              values="discovery", aggfunc="first").dropna()
    common_rows = []
    for side, g in wide.reset_index().groupby("side"):
        for rung in ["0", "1", "1b", "2", "3"]:
            common_rows.append(dict(side=side, rung=rung, period_pairs=len(g),
                                    discovery=g[rung].mean(),
                                    change_from_0=(g[rung] - g["0"]).mean()))
    common = pd.DataFrame(common_rows)
    to_csv(common, ANALYSIS / "ladder_common_support_seed42.csv")

    # The already-completed round-length check uses the two market-4 sequences for which
    # 3-, 4-, 5-, and 6-round runs are all available.
    round_rows = []
    base_runs = {20250755: "m4_paper_0", 20250757: "m4_random_0"}
    rdata = discovery[(discovery["group"].eq("rounds")) |
                      (discovery["run_id"].isin(base_runs.values()))].copy()
    for run_id, g in rdata.groupby("run_id"):
        seed = int(g["seed"].iloc[0])
        if seed not in base_runs:
            continue
        rounds = 3 if run_id == base_runs[seed] else int(run_id.split("_r")[1].split("_")[0])
        sequence = "published" if seed == 20250755 else "prior redraw"
        for side, s in g.groupby("side"):
            round_rows.append(dict(sequence=sequence, seed=seed, rounds=rounds,
                                   side=side, periods=len(s), D_first=s["first_discovery"].mean(),
                                   D_mean=s["discovery"].mean(), D_last=s["last_discovery"].mean(),
                                   trades=int(s["n_trades"].sum())))
    rounds = pd.DataFrame(round_rows).sort_values(["sequence", "side", "rounds"])
    to_csv(rounds, ANALYSIS / "extended_rounds_market4.csv")

    inventory = build_run_inventory()

    # A fixed-version hash list covers all shipped inputs and the analysis programs.
    manifest_rows = []
    paths = sorted(DATA.glob("**/*")) + sorted((ROOT / "code").glob("**/*.py"))
    for path in paths:
        if not path.is_file() or path.name.startswith("."):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_rows.append(dict(file=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                                  sha256=digest))
    manifest = pd.DataFrame(manifest_rows)
    to_csv(manifest, ANALYSIS / "file_manifest_sha256.csv")
    return {"validation": validation, "main_h": main_h, "by_market": by_market,
            "robustness": robustness, "structures": structures, "dividends": dividends,
            "benchmarks": benchmarks, "market1": m1, "common": common,
            "rounds": rounds, "inventory": inventory, "manifest": manifest}


def latex_tables(human: dict, design: pd.DataFrame, symmetric: pd.DataFrame,
                 effects: pd.DataFrame, mechanisms: pd.DataFrame) -> None:
    hs = pd.DataFrame(human["summary"]).T.loc[["seller", "buyer"]]
    dist = pd.read_csv(ANALYSIS / "human_distribution.csv").set_index("side")

    def one(x: float) -> str:
        text = f"{x:.1f}"
        text = "0.0" if text == "-0.0" else text
        return text.replace("-", "$-$")

    def francs(mean: float, median: float) -> str:
        """Signed mean with the signed median in parentheses."""
        return f"{one(mean)} ({one(median)})"

    # Section 5: human price positions (main text, two decimals for D).
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & & & & & & \multicolumn{2}{c}{Opening $p_1-P_{\RE}$} & \\",
             r"\cmidrule(lr){7-8}",
             r"Period type & Periods & Trades & $D_1$ & $\bar D$ & $D_T$ & Signed & Absolute & $p_T-p_1$ \\",
             r"\midrule"]
    for side, r in hs.iterrows():
        lines.append(f"{period_type(side)} & {int(r.periods)} & {int(r.trades)} & {r.first_D_mean:.2f} & "
                     f"{r.mean_D:.2f} & {r.last_D_mean:.2f} & {francs(r.first_dev_mean, r.first_dev_median)} & "
                     f"{francs(dist.loc[side, 'open_abs_dev_mean'], dist.loc[side, 'open_abs_dev_median'])} & "
                     f"{francs(r.francs_change_mean, r.francs_change_median)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "human_adjustment.tex").write_text("\n".join(lines) + "\n")

    # Section 3.3 (markets 3-5) and Section 6.3 (with A and B).
    def design_lines(rows: pd.DataFrame) -> list[str]:
        out = [r"\begin{tabular}{lcccccc}", r"\toprule",
               r" & \multicolumn{2}{c}{Distance $P_{\RE}-\mu$} & \multicolumn{2}{c}{Insiders buying minus selling at $\mu$} & \multicolumn{2}{c}{Uninformed trading at $P_{\RE}$ at prior value} \\",
               r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
               r"Market & Upward & Downward & Upward & Downward & Upward & Downward \\",
               r"\midrule"]
        for _, r in rows.iterrows():
            cells = [r.up_distance, r.down_distance, r.up_net_insiders, r.down_net_insiders,
                     r.up_uninformed_at_target, r.down_uninformed_at_target]
            cells = [str(c).replace("-", "$-$") for c in cells]
            out.append(f"{market_label(r.market)} & " + " & ".join(cells) + r" \\")
        return out + [r"\bottomrule", r"\end{tabular}"]

    (TABLES / "design_asymmetries.tex").write_text(
        "\n".join(design_lines(design[design["market"].isin([3, 4, 5])])) + "\n")
    (TABLES / "design_symmetric.tex").write_text("\n".join(design_lines(design)) + "\n")

    # Section 6: LLM-agent markets with original and symmetric parameters.
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Sample & Period type & Sessions & Periods & $D_1$ & $\bar D$ & $D_T$ & $D_T-D_1$ \\", r"\midrule"]
    order = {"Original market 4": 0, "Symmetric markets A and B": 1,
             "Original markets 1-5 (background)": 2}
    shown = symmetric.assign(sample_order=symmetric["sample"].map(order)).sort_values(["sample_order", "side"])
    previous = None
    for _, r in shown.iterrows():
        label = {"Original market 4": "Market 4",
                 "Symmetric markets A and B": "Markets A and B",
                 "Original markets 1-5 (background)": "Markets 1--5"}[r["sample"]]
        if previous is not None and r["sample"] != previous:
            lines.append(r"\addlinespace")
        previous = r["sample"]
        cells = [f"{x:.2f}".replace("-", "$-$") for x in
                 (r.first_discovery, r.mean_discovery, r.last_discovery)]
        change = f"{r.change_discovery:+.2f}".replace("-", "$-$")
        lines.append(f"{label} & {period_type(r.side)} & {int(r.sessions)} & {int(r.periods)} & "
                     + " & ".join(cells) + f" & {change}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "symmetric_comparison.tex").write_text("\n".join(lines) + "\n")

    # Appendix E.5: paired changes along the disclosure steps.
    lines = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Step & Period type & Sessions & Period pairs & Before & After & Change & Positive & Session $p$ \\", r"\midrule"]
    for _, r in effects.iterrows():
        p_text = f"{r.p_session_sign_flip:.3f}"
        lines.append(f"{r.step} & {period_type(r.side)} & {int(r.sessions)} & {int(r.periods)} & {r.before_mean:.3f} & {r.after_mean:.3f} & {r.mean_change:+.3f} & {int(r.improved_sessions)}/{int(r.sessions)} & {p_text}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "ladder_effects.tex").write_text("\n".join(lines)+"\n")

    lines = [r"\begin{tabular}{clrrrrr}", r"\toprule",
             r"Step & Period type & $\bar D$ & True-state belief & Price basis & TE (\%) & Insider profit ratio (\%) \\", r"\midrule"]
    for _, r in mechanisms.iterrows():
        lines.append(f"{r.rung} & {period_type(r.side)} & {r.discovery:.3f} & {r.true_state_belief:.3f} & {r.price_basis_share:.3f} & {r.TE_pct:.1f} & {r.insider_advantage_pct:.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "ladder_mechanisms.tex").write_text("\n".join(lines)+"\n")


def write_longtable(filename: str, spec: str, caption: str, label: str,
                    header: str, rows: list[str], font: str = r"\scriptsize") -> None:
    lines = [font, rf"\begin{{longtable}}{{{spec}}}",
             rf"\caption{{{caption}}}\label{{{label}}}\\", r"\toprule", header,
             r"\midrule", r"\endfirsthead", rf"\multicolumn{{{header.count('&') + 1}}}{{l}}{{\textit{{Continued}}}}\\",
             r"\toprule", header, r"\midrule", r"\endhead", r"\midrule",
             rf"\multicolumn{{{header.count('&') + 1}}}{{r}}{{\textit{{Continued on next page}}}}\\",
             r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    lines += rows + [r"\end{longtable}"]
    (TABLES / filename).write_text("\n".join(lines) + "\n")


def appendix_latex_tables(a: dict[str, pd.DataFrame]) -> None:
    validation_rows = []
    for _, r in a["validation"].sort_values(["market", "period"]).iterrows():
        validation_rows.append(
            f"{int(r.market)} & {int(r.period)} & {int(r.n_trades)} & {r.mean_printed:.1f} & "
            f"{r.mean_recovered:.3f} & {r.mean_residual:+.3f} & "
            f"{latex_escape(r.quality_flag)}" + " \\\\")
    write_longtable("appendix_human_validation.tex", "rrrrrrl",
                    r"Period-by-period audit of the 987 human transactions",
                    "tab:app-human-validation",
                    r"Market & Period & Trades & Printed mean & Transcribed mean & Error & Check \\",
                    validation_rows)

    main_rows = []
    for _, r in a["main_h"].sort_values(["market", "period"]).iterrows():
        main_rows.append(
            f"{int(r.market)} & {int(r.period)} & {r.state} & {period_type(r.side)} & {int(r.n_trades)} & "
            f"{r.vbar:.1f} & {r.re_price:.1f} & {r.p_first:.1f} & {r.p_mean:.1f} & {r.p_last:.1f} & "
            f"{r.D_first:.3f} & {r.D_mean:.3f} & {r.D_last:.3f}" + " \\\\")
    write_longtable("appendix_human_main_periods.tex", "rrllrrrrrrrrr",
                    r"The 27 human insider periods in the main sample",
                    "tab:app-human-main",
                    r"Mkt. & Per. & State & Type & $N$ & $\mu$ & $P_{\RE}$ & $p_1$ & $\bar p$ & $p_T$ & $D_1$ & $\bar D$ & $D_T$ \\",
                    main_rows, r"\tiny")

    by_market_rows = []
    for _, r in a["by_market"].iterrows():
        by_market_rows.append(
            f"{int(r.market)} & {period_type(r.side)} & {int(r.periods)} & {int(r.trades)} & "
            f"{r.D_first:.3f} & {r.D_mean:.3f} & {r.D_last:.3f} & {r.change_francs:+.1f}" + " \\\\")
    write_longtable("appendix_human_by_market.tex", "rlrrrrrr",
                    r"Human price positions by market and period type",
                    "tab:app-human-market",
                    r"Market & Period type & Periods & Trades & $D_1$ & $\bar D$ & $D_T$ & Mean $p_T-p_1$ \\",
                    by_market_rows)

    rob_rows = []
    shown = a["robustness"].copy()
    fmt_p = lambda p: "--" if pd.isna(p) else (r"$<$0.001" if p < 0.001 else f"{p:.3f}")
    for _, r in shown.iterrows():
        rob_rows.append(
            f"{latex_escape(r.panel)} & {latex_escape(r['sample'])} & {r.measure} & "
            f"{r['diff']:+.3f} & {fmt_p(r.p_welch)} & {fmt_p(r.p_mwu)} & "
            f"{fmt_p(r.p_perm_within_market)} & {fmt_p(r.p_fe_cluster)}" + " \\\\")
    write_longtable("appendix_human_robustness.tex",
                    r"p{2.1cm}p{3.5cm}p{3.0cm}rrrrr",
                    r"Human-data inference and sample robustness",
                    "tab:app-human-robustness",
                    r"Panel & Sample & Measure & Difference & Welch $p$ & MWU $p$ & Within-market $p$ & Clustered FE $p$ \\",
                    rob_rows, r"\tiny")

    structure_rows = []
    for _, r in a["structures"].iterrows():
        structure_rows.append(
            f"{market_label(r.market)} & {int(r.traders)} & {int(r.informed)} & {r.states} & "
            f"{latex_escape(r.prior)} & {int(r.periods)} & {latex_escape(r.information_schedule)} & "
            f"{int(r.francs_per_dollar)}" + " \\\\")
    write_longtable("appendix_market_structure.tex", r"rrrrlrp{4.9cm}r",
                    r"Market structure, priors, and information schedules",
                    "tab:app-market-structure",
                    r"Market & Traders & Informed & States & Prior & Periods & Information schedule & Francs/\$ \\",
                    structure_rows, r"\footnotesize")

    dividend_rows = []
    for _, r in a["dividends"].iterrows():
        z = "--" if str(r.Z) == "--" else f"{float(r.Z):.0f}"
        dividend_rows.append(f"{market_label(r.market)} & {r['type']} & {float(r.X):.0f} & {float(r.Y):.0f} & {z}" + " \\\\")
    write_longtable("appendix_market_dividends.tex", "rlrrr",
                    r"Dividend schedules (francs per certificate)", "tab:app-dividends",
                    r"Market & Type & State X & State Y & State Z \\", dividend_rows)

    benchmark_rows = []
    for _, r in a["benchmarks"].iterrows():
        benchmark_rows.append(
            f"{market_label(r.market)} & {r.state} & {r.vbar:.1f} & {r.RE:.1f} & {r.PI:.1f} & "
            f"{r.distance:+.1f} & {period_type(r.side)}" + " \\\\")
    write_longtable("appendix_theory_benchmarks.tex", "rlrrrrl",
                    r"Prior value, informed value, and PI prediction by market and state", "tab:app-benchmarks",
                    r"Market & State & $\mu$ & $P_{\RE}$ & $P_{\PI}$ & $P_{\RE}-\mu$ & Period type \\",
                    benchmark_rows)

    m1_rows = []
    for _, r in a["market1"].iterrows():
        clue = str(r.clue_card).lstrip("'")
        m1_rows.append(
            f"{int(r.period)} & {r.state} & {latex_escape(clue)} & {int(r.clue_ones)} & "
            f"{r.posterior_X:.4f} & {r.re_price:.3f} & {r.pi_price:.3f} & "
            f"{period_type(r.side)} & {'Yes' if r.predictions_differ else 'No'}" + " \\\\")
    write_longtable("appendix_market1_posterior.tex", "rrlrrrrll",
                    r"Market 1: clue-conditional benchmarks in the insider periods",
                    "tab:app-market1",
                    r"Period & State & Ten-draw clue & Ones & $q(X\mid c)$ & $P_{\RE}(c)$ & $P_{\PI}(c)$ & Period type & Differ? \\",
                    m1_rows, r"\footnotesize")

    inventory_rows = []
    for _, r in a["inventory"].iterrows():
        model = str(r.model).replace("deepseek-v4-flash", "DS-V4-Flash")
        preset = "published" if str(r.preset).startswith("paper") else "prior redraw"
        inventory_rows.append(
            f"{latex_escape(r.run_id)} & {market_label(r.market)} & {latex_escape(r.treatment)} & {int(r.seed)} & "
            f"{int(r.rounds)} & {preset} & {r.state_sequence} & "
            f"{int(r.valid_periods)}/{int(r.insider_periods)} & {r.no_trade_periods} & {latex_escape(model)}" + " \\\\")
    write_longtable("appendix_run_inventory.tex",
                    r"p{2.6cm}rrrp{.8cm}p{1.6cm}p{3.0cm}p{1.0cm}p{1.0cm}p{2.0cm}",
                    r"LLM sessions and sample selection", "tab:app-run-list",
                    r"Run & Mkt. & Step & Seed & Turns & Preset & Realized states & Traded & No-trade & Model \\",
                    inventory_rows, r"\tiny")

    session_effects = pd.read_csv(ANALYSIS / "ladder_session_effects.csv")
    effect_rows = []
    for _, r in session_effects.iterrows():
        effect_rows.append(
            f"{latex_escape(r.step)} & {period_type(r.side)} & {market_label(r.market)} & {int(r.seed)} & "
            f"{int(r.paired_periods)} & {r.before_mean:.3f} & {r.after_mean:.3f} & {r.mean_change:+.3f}" + " \\\\")
    write_longtable("appendix_ladder_sessions.tex", r"p{3.3cm}lrrrrrr",
                    r"Disclosure effects by market--seed session pair",
                    "tab:app-ladder-sessions",
                    r"Step & Type & Market & Seed & Pairs & Before & After & Change \\",
                    effect_rows, r"\footnotesize")

    gaps = pd.read_csv(ANALYSIS / "symmetric_session_gaps.csv")
    gap_rows = []
    for _, r in gaps.iterrows():
        sample = {"Original market 4": "Market 4",
                  "Symmetric markets A and B": "Markets A and B",
                  "Original markets 1-5 (background)": "Markets 1--5"}[r["sample"]]
        gap_rows.append(f"{sample} & {latex_escape(r.run_id)} & {r.buyer:.3f} & {r.seller:.3f} & {r.sell_minus_buy:+.3f}" + " \\\\")
    write_longtable("appendix_symmetric_sessions.tex", r"p{3.0cm}p{2.7cm}rrr",
                    r"Mean price position by session in the design comparison",
                    "tab:app-symmetric-sessions",
                    r"Sample & Run & Upward & Downward & Downward minus upward \\", gap_rows)

    common_rows = []
    for _, r in a["common"].sort_values(["side", "rung"]).iterrows():
        common_rows.append(f"{period_type(r.side)} & {r.rung} & {int(r.period_pairs)} & {r.discovery:.3f} & {r.change_from_0:+.3f}" + " \\\\")
    write_longtable("appendix_common_support.tex", "lrrrr",
                    r"The five disclosure steps on a common seed-42 period support",
                    "tab:app-common-support",
                    r"Period type & Step & Market--period cells & $\bar D$ & Change from step 0 \\",
                    common_rows)

    round_rows = []
    for _, r in a["rounds"].iterrows():
        round_rows.append(
            f"{latex_escape(r.sequence)} & {int(r.rounds)} & {period_type(r.side)} & {int(r.periods)} & "
            f"{int(r.trades)} & {r.D_first:.3f} & {r.D_mean:.3f} & {r.D_last:.3f}" + " \\\\")
    write_longtable("appendix_rounds.tex", "lrlrrrrr",
                    r"Market 4 with more turns per trader",
                    "tab:app-rounds",
                    r"State sequence & Turns & Period type & Periods & Trades & $D_1$ & $\bar D$ & $D_T$ \\",
                    round_rows)


def human_appendix_d(p0: dict) -> None:
    """Appendix D.3-D.6: five-market pooling, the closeness index A, distances from the
    no-information price, and the familiar-state and opening-window checks."""
    period = pd.read_csv(HUMAN / "period_table.csv")
    ins = period.query("info == 'insider'").copy()
    ins["dev_first"] = ins["p_first"] - ins["re_price"]
    ins["dev_last"] = ins["p_last"] - ins["re_price"]

    def fmt(x: float, digits: int = 2) -> str:
        text = f"{x:.{digits}f}"
        text = "0.0" if text in ("-0.0", "-0.00") else text
        return text.replace("-", "$-$")

    # D.3: markets 3-5, markets 1-2, and all five markets.
    rows = []
    for label, sub in (("Markets 3--5", ins[ins["market"].isin([3, 4, 5])]),
                       ("Markets 1--2", ins[ins["market"].isin([1, 2])]),
                       ("Markets 1--5", ins)):
        for side in ("seller", "buyer"):
            g = sub[sub["side"].eq(side)]
            rows.append(dict(sample=label, side=side, periods=len(g), trades=int(g["n_trades"].sum()),
                             D_first=g["D_first"].mean(), D_mean=g["D_mean"].mean(),
                             D_last=g["D_last"].mean(), dev_first=g["dev_first"].mean(),
                             dev_last=g["dev_last"].mean(), change=g["francs_change"].mean()))
    five = pd.DataFrame(rows)
    to_csv(five, ANALYSIS / "human_five_markets.csv")
    body = []
    for _, r in five.iterrows():
        body.append(f"{r['sample'].replace('Markets ', '')} & {period_type(r.side)} & {r.periods} & {r.trades} & {fmt(r.D_first)} & "
                    f"{fmt(r.D_mean)} & {fmt(r.D_last)} & {fmt(r.dev_first, 1)} & {fmt(r.dev_last, 1)} & "
                    f"{fmt(r.change, 1)}" + r" \\")
    write_longtable("appendix_human_five_markets.tex", "llrrrrrrrr",
                    r"Price positions in markets 3--5, markets 1--2, and all five markets",
                    "tab:app-human-five",
                    r"Markets & Type & Periods & Trades & $D_1$ & $\bar D$ & $D_T$ & $p_1-P_{\RE}$ & $p_T-P_{\RE}$ & $p_T-p_1$ \\",
                    body, r"\scriptsize")

    # D.4: closeness A = 1 - |p - P_RE| / |P_RE - mu| = 1 - |D - 1|.
    main = ins[ins["market"].isin([3, 4, 5])].copy()
    rows = []
    for d_col, label in (("D_first", "opening"), ("D_mean", "period mean"), ("D_last", "closing")):
        a = 1 - (main[d_col] - 1).abs()
        for measure, values in ((f"$D$, {label} price", main[d_col]), (f"$A$, {label} price", a)):
            down, up = values[main["side"].eq("seller")], values[main["side"].eq("buyer")]
            rows.append(dict(measure=measure, downward=down.mean(), upward=up.mean(),
                             difference=down.mean() - up.mean(),
                             p_welch=stats.ttest_ind(down, up, equal_var=False).pvalue))
    closeness = pd.DataFrame(rows)
    to_csv(closeness, ANALYSIS / "human_closeness.csv")
    fmt_p = lambda p: r"$<$0.001" if p < 0.001 else f"{p:.3f}"
    body = [f"{r.measure} & {fmt(r.downward, 3)} & {fmt(r.upward, 3)} & {fmt(r.difference, 3)} & {fmt_p(r.p_welch)}" + r" \\"
            for _, r in closeness.iterrows()]
    write_longtable("appendix_human_closeness.tex", "lrrrr",
                    r"Price position $D$ and closeness $A$ in markets 3--5",
                    "tab:app-human-closeness",
                    r"Measure & Downward & Upward & Downward minus upward & Welch $p$ \\",
                    body, r"\footnotesize")

    # D.5: distance to P_RE measured from the no-information price p0.
    rows = []
    for (market, side, re), g in main.groupby(["market", "side", "re_price"]):
        base = p0[market]
        dist0 = re - base
        same_direction = np.sign(dist0) == np.sign(re - g["vbar"].iloc[0])
        rows.append(dict(market=int(market), side=side, re=re, mu=g["vbar"].iloc[0], p0=base,
                         periods=len(g), distance_mu=re - g["vbar"].iloc[0], distance_p0=dist0,
                         D0_first=((g["p_first"] - base) / dist0).mean() if same_direction else np.nan,
                         D0_last=((g["p_last"] - base) / dist0).mean() if same_direction else np.nan))
    dist = pd.DataFrame(rows).sort_values(["market", "side", "re"], ascending=[True, False, True])
    to_csv(dist, ANALYSIS / "human_noinfo_distance.csv")
    body = []
    for _, r in dist.iterrows():
        d0 = "--" if pd.isna(r.D0_first) else fmt(r.D0_first)
        dT = "--" if pd.isna(r.D0_last) else fmt(r.D0_last)
        body.append(f"{r.market} & {period_type(r.side)} & {r.re:.0f} & {r.mu:.1f} & {r.p0:.1f} & {r.periods} & "
                    f"{fmt(r.distance_mu, 1)} & {fmt(r.distance_p0, 1)} & {d0} & {dT}" + r" \\")
    write_longtable("appendix_human_noinfo_distance.tex", "rlrrrrrrrr",
                    r"Distances to the informed value from the prior value and from the no-information price",
                    "tab:app-human-noinfo-distance",
                    r"Market & Period type & $P_{\RE}$ & $\mu$ & $\bar p_0$ & Periods & $P_{\RE}-\mu$ & $P_{\RE}-\bar p_0$ & $D^0_1$ & $D^0_T$ \\",
                    body, r"\footnotesize")

    # D.6: familiar states and the first tenth of each period's trades.
    est = pd.read_csv(HUMAN / "established.csv")
    traj = pd.read_csv(HUMAN / "trajectory.csv").query("bin == 1")
    labels = {"all": "All insider periods", "k>=2": "State seen in at least one earlier insider period",
              "k>=3": "State seen in at least two earlier insider periods"}
    body = []
    for sample in ("all", "k>=2", "k>=3"):
        g = est[est["sample"].eq(sample)].set_index("side")
        for side in ("seller", "buyer"):
            r = g.loc[side]
            body.append(f"{labels[sample]} & {period_type(side)} & {int(r.n)} & {fmt(r.D_first, 3)} & "
                        f"{fmt_p(r.p_first_vs_1)} & {fmt(r.D_last, 3)}" + r" \\")
        c = g.loc["between-side contrast"]
        body.append(f" & Difference & {int(c.n)} & {fmt(c.D_first, 3)} & {fmt_p(c.p_first_vs_1)} & " + r"\\")
        body.append(r"\addlinespace")
    write_longtable("appendix_human_familiar.tex", r"p{5.2cm}lrrrr",
                    r"Opening positions in periods with familiar states",
                    "tab:app-human-familiar",
                    r"Sample & Period type & Periods & $D_1$ & $p$ & $D_T$ \\",
                    body[:-1], r"\footnotesize")
    window = traj.set_index("side")[["n_periods", "D_mean", "F_mean", "F_median"]]
    to_csv(window.reset_index(), ANALYSIS / "human_opening_window.csv")


def selection_tables() -> None:
    """Paper Table 1 (the 61-period ledger) and the A/B competitive-interval table."""
    period = pd.read_csv(HUMAN / "period_table.csv")
    rows = []
    for market, g in period.groupby("market"):
        insider = g[g["info"].eq("insider")]
        rows.append(dict(market=int(market), periods=len(g),
                         no_information=int(g["info"].eq("none").sum()),
                         all_informed=int(g["info"].eq("all").sum()),
                         informed_selling=int(insider["side"].eq("seller").sum()),
                         informed_buying=int(insider["side"].eq("buyer").sum())))
    ledger = pd.DataFrame(rows)
    to_csv(ledger, ANALYSIS / "selection_ledger.csv")
    cols = ["periods", "no_information", "all_informed", "informed_selling", "informed_buying"]
    lines = [r"% Counts computed from results/human/period_table.csv by code/analyze.py.",
             r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & & & & \multicolumn{2}{c}{Insider periods} \\",
             r"\cmidrule(lr){5-6}",
             r"Market & Periods & No-information & All-informed & Downward & Upward \\",
             r"\midrule"]
    for _, r in ledger.iterrows():
        lines.append(" & ".join([str(r["market"])] + [str(r[c]) for c in cols]) + r" \\")
    lines += [r"\midrule",
              " & ".join(["Total"] + [str(int(ledger[c].sum())) for c in cols]) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (TABLES / "selection_ledger.tex").write_text("\n".join(lines) + "\n")

    # Competitive-price intervals in A/B: [second-highest, highest] type valuation in
    # the realized state; the RE point benchmark is the upper endpoint.
    specs = market_specs()
    interval_rows = []
    for market in (7, 8):
        spec = specs[market]
        vbar = max(sum(p * d for p, d in zip(spec["prior"], vals))
                   for vals in spec["div"].values())
        for i, state in enumerate(spec["states"]):
            v1, v2 = sorted((vals[i] for vals in spec["div"].values()), reverse=True)[:2]
            width = (v1 - v2) / abs(v1 - vbar)
            buying = v1 > vbar
            interval_rows.append(dict(
                market=market, state=state, side="upward" if buying else "downward",
                vbar=vbar, v2=v2, re=v1, price_low=v2, price_high=v1,
                D_low=1 - width if buying else 1.0, D_high=1.0 if buying else 1 + width))
    intervals = pd.DataFrame(interval_rows)
    to_csv(intervals, ANALYSIS / "competitive_intervals_AB.csv")
    lines = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Informed values and competitive-price intervals in markets A and B}",
             r"\label{tab:app-competitive-intervals}", r"\footnotesize",
             r"\fittable{\begin{tabular}{lllrrrrl}", r"\toprule",
             r"Market & State & Period type & $\mu$ & $v_{(2)}$ & $P_{\RE}=v_{(1)}$ & $\mathcal P^{CE}$ & $\mathcal D^{CE}$ \\",
             r"\midrule"]
    for _, r in intervals.iterrows():
        lines.append(
            f"{market_label(r.market)} & {r.state} & {r.side.title()} & {r.vbar:.0f} & {r.v2:.0f} & "
            f"{r.re:.0f} & $[{r.price_low:.0f},{r.price_high:.0f}]$ & "
            f"$[{r.D_low:.1f},{r.D_high:.1f}]$" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption*{\footnotesize Notes: Valuations and price intervals are in francs per "
              r"certificate. $v_{(2)}$ is the second-highest type value in the realized state; $P_{\RE}$ is "
              r"the highest and the upper price endpoint. $\mathcal D^{CE}$ maps the price interval "
              r"through $D=(p-\mu)/(P_{\RE}-\mu)$. Source: dividend schedules in Table "
              r"\ref{tab:app-dividends}; author's calculations.}",
              r"\end{table}"]
    (TABLES / "appendix_competitive_intervals.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    discovery, mechanisms = load_metric_rows()
    to_csv(discovery, ANALYSIS / "all_llm_discovery_periods.csv")
    to_csv(mechanisms, ANALYSIS / "all_llm_mechanisms.csv")
    human = human_analysis()
    design = design_table()
    symmetric, gaps = symmetric_analysis(discovery)
    effects, mech = ladder_analysis(discovery, mechanisms)
    latex_tables(human, design, symmetric, effects, mech)
    appendix = appendix_analysis(discovery)
    appendix_latex_tables(appendix)
    selection_tables()
    schematics.run()
    benchmark = noinfo_benchmark.run()
    human_appendix_d(benchmark["p0"])
    summary = {
        "human": human,
        "symmetric_sell_minus_buy": gaps,
        "noinfo_benchmark": benchmark,
        "llm_metric_files": len(list((LLM_RUNS).glob("**/*.metrics.json"))),
        "notes": [
            "Human transactions are transcribed from Appendix B of Caltech Social Science Working Paper 331.",
            "The transcription contains 987 trades; the markets 3-5 insider sample contains 474 trades.",
            "The original-market pool includes all supplied m1-m5 sessions, including one Gemini market-3 session.",
            "The symmetric control uses prereported seeds 42, 43, and 44 for each of markets A and B (run IDs m7_* and m8_*).",
            "Disclosure effects first pair periods, then average within each market-seed session pair.",
            "Sign flips use session-pair means as the exchangeable units.",
            "Figure 3 first averages period bins within sessions and bootstraps whole sessions.",
            "Appendix E reports the completed market-4 four-, five-, and six-round checks.",
        ],
    }
    (ANALYSIS / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
