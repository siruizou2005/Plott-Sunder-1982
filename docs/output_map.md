# Output map

This file maps each exhibit and cited number in the paper to the program that produces it and the file that holds it. Exhibit numbers follow the October 2026 manuscript ("Do Prices Reveal What Insiders Know? A Reexamination of Plott and Sunder (1982)"); LaTeX labels are given so the map survives renumbering. All paths are relative to the repository root. Run IDs use the prefix `m7` for market A and `m8` for market B, as in the paper.

The paper calls an insider period *downward* when the informed value P_RE lies below the prior value μ (every insider would sell at μ) and *upward* when it lies above. The CSV files keep the engine's side names: `seller` is a downward period and `buyer` an upward one; `vbar` is μ.

Programs:

- **Stage 1**: `code/human/run_all.py`, which runs `load_trades.py`, `period_table.py`, `results.py`, and `fig1_identity.py`.
- **Stage 2**: `code/analyze.py`, which also calls `code/schematics.py` (design exhibits) and `code/noinfo_benchmark.py` (Section 4 and Appendix C).

## Main text

| Exhibit | Label | Output file | Program | Numbers behind it |
|---|---|---|---|---|
| Table 1: the five original markets | `tab:markets` | `results/tables/market_summary.tex` | Stage 2 (`schematics.market_summary`) | `results/analysis/market_summary.csv` |
| Table 2: market 4 dividends, prior values, and price levels | `tab:market4` | `results/tables/market4_dividends.tex`, `market4_levels.tex` | Stage 2 (`schematics.market4_tables`) | parameters in `code/human/ps1982_params.py` |
| Figure 1: supply and demand in market 4 | `fig:demand` | `results/figures/market4_demand.pdf` | Stage 2 (`schematics.market4_demand_figure`) | parameters in `code/human/ps1982_params.py` |
| Table 3: the 61 periods by information condition and period type | `tab:ledger` | `results/tables/selection_ledger.tex` | Stage 2 (`selection_tables`) | `results/analysis/selection_ledger.csv`; classification in `results/human/identity.csv` |
| Figure 2: RE and PI differ only in downward periods | `fig:selection` | `results/figures/selection_identity.png` | Stage 1 (`fig1_identity.py`) | `results/human/identity.csv` |
| Table 4: differences between upward and downward periods, markets 3–5 | `tab:design` | `results/tables/design_asymmetries.tex` | Stage 2 (`design_table`) | `results/analysis/design_asymmetries.csv` |
| Table 5: no-information and downward-period prices | `tab:noinfo` | `results/tables/noinfo_downward.tex` | Stage 2 (`noinfo_benchmark`) | `results/analysis/noinfo_downward.csv` |
| Figure 3: no-information and downward-period prices by market | `fig:noinfo` | `results/figures/noinfo_downward.pdf` | Stage 2 (`noinfo_benchmark`) | `results/analysis/noinfo_downward_periods.csv` |
| Figure 4: the price position D in market 4 | `fig:scale` | `results/figures/position_scale.pdf` | Stage 2 (`schematics.position_scale_figure`) | parameters only |
| Table 6: price positions at the opening, over the period, and at the close | `tab:human` | `results/tables/human_adjustment.tex` | Stage 2 (`human_analysis`) | `results/analysis/human_summary.csv`, `human_tests.csv`, `human_distribution.csv` |
| Figure 5: transaction prices within downward and upward periods | `fig:humanpaths` | `results/figures/human_paths.pdf` | Stage 2 (`human_analysis`) | `results/human/trades.csv` |
| Table 7: price positions in LLM markets | `tab:symmetric` | `results/tables/symmetric_comparison.tex` | Stage 2 (`symmetric_analysis`) | `results/analysis/symmetric_comparison.csv`, `llm_price_discovery_periods.csv` |
| Table 8: differences between upward and downward periods, markets 3–5, A, and B | `tab:design-ab` | `results/tables/design_symmetric.tex` | Stage 2 (`design_table`) | `results/analysis/design_asymmetries.csv` |
| Figure 6: price positions within periods in LLM markets | `fig:symmetric` | `results/figures/symmetric_markets.pdf` | Stage 2 (`symmetric_analysis`) | `data/llm/llm_transactions.csv` |
| Table 9: steps of disclosure | `tab:steps` | typed in the manuscript | none | treatment text in `data/llm/prompts/` |
| Figure 7: price positions across the disclosure steps at seed 42 | `fig:ladder` | `results/figures/disclosure_ladder.pdf` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_price_discovery_periods.csv` |

## Appendices

| Exhibit | Label | Output file | Program | Numbers behind it |
|---|---|---|---|---|
| Table A1: market structure | `tab:app-market-structure` | `results/tables/appendix_market_structure.tex` | Stage 2 | `results/analysis/market_structures.csv` |
| Table A2: dividend schedules | `tab:app-dividends` | `results/tables/appendix_market_dividends.tex` | Stage 2 | `results/analysis/market_dividends.csv` |
| Table A3: prior values, informed values, PI predictions | `tab:app-benchmarks` | `results/tables/appendix_theory_benchmarks.tex` | Stage 2 | `results/analysis/theory_benchmarks_exact_signal.csv` |
| Table A4: market 1 clue-conditional benchmarks | `tab:app-market1` | `results/tables/appendix_market1_posterior.tex` | Stage 2 | `results/analysis/market1_posterior_benchmarks.csv` |
| Table B1: audit of the 987 transactions | `tab:app-human-validation` | `results/tables/appendix_human_validation.tex` | Stage 2 | `results/human/validation.csv`, `rounding_check.csv` |
| Table B2: the 27 main-sample periods | `tab:app-human-main` | `results/tables/appendix_human_main_periods.tex` | Stage 2 | `results/analysis/human_main_27_periods.csv` |
| Table B3: inconsistencies in the original | `tab:app-inconsistencies` | typed in the manuscript | none | data sources cited in the table |
| Figure B1: market 4, periods 7–8 | `fig:app-market4-offset` | `results/figures/market4_periods78.pdf` | static file | left panel: `data/human/plott_sunder_1982_prices.csv`; right panel: working-paper Figure 10 |
| Table C1: the 18 no-information periods | `tab:app-noinfo-periods` | `results/tables/appendix_noinfo_periods.tex` | Stage 2 (`noinfo_benchmark`) | `results/analysis/noinfo_downward_periods.csv` |
| Table C2: alternative definitions of the benchmark | `tab:app-noinfo-variants` | `results/tables/appendix_noinfo_variants.tex` | Stage 2 (`noinfo_benchmark`) | `results/analysis/noinfo_variants.csv` |
| Table C3: trade-level inference | `tab:app-noinfo-trades` | `results/tables/appendix_noinfo_trade_level.tex` | Stage 2 (`noinfo_benchmark`) | `results/analysis/noinfo_trade_level.csv` |
| Table D1: human results by market | `tab:app-human-market` | `results/tables/appendix_human_by_market.tex` | Stage 2 | `results/analysis/human_results_by_market.csv` |
| Table D2: human inference and sample robustness | `tab:app-human-robustness` | `results/tables/appendix_human_robustness.tex` | Stage 1 (`results.py`), typeset by Stage 2 | `results/human/robustness.csv` |
| Table D3: markets 3–5, 1–2, and all five | `tab:app-human-five` | `results/tables/appendix_human_five_markets.tex` | Stage 2 (`human_appendix_d`) | `results/analysis/human_five_markets.csv` |
| Table D4: price position D and closeness A | `tab:app-human-closeness` | `results/tables/appendix_human_closeness.tex` | Stage 2 (`human_appendix_d`) | `results/analysis/human_closeness.csv` |
| Table D5: distances from the no-information price | `tab:app-human-noinfo-distance` | `results/tables/appendix_human_noinfo_distance.tex` | Stage 2 (`human_appendix_d`) | `results/analysis/human_noinfo_distance.csv` |
| Table D6: familiar states | `tab:app-human-familiar` | `results/tables/appendix_human_familiar.tex` | Stage 2 (`human_appendix_d`) | `results/human/established.csv`; opening window in `results/analysis/human_opening_window.csv` |
| Table E1: institutional comparison | `tab:app-institution` | typed in the manuscript | none | simulation rules in each session's `meta.json` |
| Table E2: LLM sessions | `tab:app-run-list` | `results/tables/appendix_run_inventory.tex` | Stage 2 (`build_run_inventory`) | `results/analysis/llm_run_inventory.csv` |
| Table E3: competitive-price intervals in A and B | `tab:app-competitive-intervals` | `results/tables/appendix_competitive_intervals.tex` | Stage 2 (`selection_tables`) | `results/analysis/competitive_intervals_AB.csv` |
| Table E4: design comparison by session | `tab:app-symmetric-sessions` | `results/tables/appendix_symmetric_sessions.tex` | Stage 2 | `results/analysis/symmetric_session_gaps.csv` |
| Table E5: paired changes across the disclosure steps | `tab:ladder` | `results/tables/ladder_effects.tex` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_paired_effects.csv` |
| Figure E1: disclosure responses by market and session | `fig:marketheterogeneity` | `results/figures/disclosure_market_heterogeneity.pdf` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_session_effects.csv` |
| Table E6: beliefs, allocations, and profits at seed 42 | `tab:mechanisms` | `results/tables/ladder_mechanisms.tex` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_mechanisms_seed42.csv` |
| Table E7: disclosure effects by session pair | `tab:app-ladder-sessions` | `results/tables/appendix_ladder_sessions.tex` | Stage 2 | `results/analysis/ladder_session_effects.csv` |
| Table E8: seed-42 common support | `tab:app-common-support` | `results/tables/appendix_common_support.tex` | Stage 2 | `results/analysis/ladder_common_support_seed42.csv` |
| Table E9: market 4 with more turns | `tab:app-rounds` | `results/tables/appendix_rounds.tex` | Stage 2 | `results/analysis/extended_rounds_market4.csv` |
| Table F1: map from evidence to programs | `tab:app-output-map` | typed in the manuscript | none | this file |

The rendered treatment prompts quoted in Appendix E.1 are in `data/llm/prompts/`.

## Numbers cited in the text

| Where | Statement | File and field |
|---|---|---|
| Section 2.1 | 61 periods: 18 no-information, 7 all-informed, 36 insider | `results/analysis/selection_ledger.csv` |
| Section 3.2, (A4) | One trader's cash buys 25 certificates at 400 francs | `results/human/endowment_capacity.csv` |
| Section 3.2 | 17 downward and 19 upward insider periods; RE and PI differ exactly in the downward periods | `results/human/identity.csv` (`side`, `predictions_differ`); `analysis_summary.json` (`human.selection_crosstab`) |
| Section 3.3 | Distances, insiders buying minus selling at μ, uninformed traders able to trade at P_RE | `results/analysis/design_asymmetries.csv` |
| Section 4.2 | No-information prices 263.7, 257.2, 225.8, 162.6, 170.7; downward means 289.5, 287.2, 171.4, 164.0, 177.0; Welch and rank-sum p-values | `results/analysis/noinfo_downward.csv` |
| Section 4.2 | Market 1 mean prices from 239 (period 1) to 347 (period 11); market 2 period 9 at 230–240 in its second half | `results/human/period_table.csv` (`p_mean`); `data/human/plott_sunder_1982_prices.csv` |
| Section 4.1, Appendix C.1 | Market 4 period 1: six of eight trades at 500–1,500 francs; mean 784 | `analysis_summary.json` (`noinfo_benchmark.market4_period1_prices`); `results/analysis/noinfo_downward_periods.csv` |
| Section 5.1 footnote, Appendix B.1 | Market 1 no-information price 263.7, or 264.9 with the printed 269 for period 2 | `analysis_summary.json` (`noinfo_benchmark.market1_p0_with_printed_period2`) |
| Section 5.2 | Upward opening −82.5 francs, D₁ 0.30, 14/14 below P_RE, 4 with D₁ < 0; downward opening −3.1 signed and 15.4 absolute; 8 below, 2 at, 3 above | `results/analysis/human_summary.csv`, `human_distribution.csv`; D₁ < 0 from `human_main_27_periods.csv` |
| Section 5.2 | Change 70.8 (median 59) upward and −2.3 (median 0) downward; 11 rise and 3 unchanged | `results/analysis/human_summary.csv`, `human_distribution.csv` |
| Section 5.2 | Closing −11.7 and −5.4 francs (p = 0.186), medians −5; D_T 0.89 and 1.15 (p < 0.001); 13/14 upward close below | `results/analysis/human_tests.csv` (`dev_francs_last`, `D_last`), `human_distribution.csv` |
| Section 5.2 | Familiar states: D₁ 0.99 (7 downward) and 0.72 (6 upward); upward below 1 (p = 0.017); difference p = 0.28 | `results/human/established.csv`, rows `k>=3` |
| Section 5.2, Appendix D.3 | All five markets: upward opening −68.7 francs; markets 1–2 upward change 6.0 | `results/analysis/human_five_markets.csv` |
| Section 6.2 | Markets 1–5: 26 sessions, 188 periods, downward D̄ 1.10 and D_T 1.27, upward −0.09 and −0.14; market 4: 1.29, 0.11, 0.22 | `results/analysis/symmetric_comparison.csv` |
| Section 6.2 | Human market 4 upward D from 0.29 to 0.89 | `results/analysis/human_results_by_market.csv` |
| Section 6.3 | Gap 1.18 (market 4) and −0.01 (A and B); A and B 0.41 and 0.40, closing 0.49 and 0.76; session-weighted gap −0.02 | `results/analysis/symmetric_comparison.csv`; `analysis_summary.json` (`symmetric_sell_minus_buy`) |
| Section 6.4 | Step 0 to 3: upward 0.31 to 0.82 (4/4), downward 0.47 to 0.73 (3/4); 0 to 1 downward −0.31 (A) and +0.32 (B) | `results/analysis/ladder_paired_effects.csv`, `ladder_session_effects.csv` |
| Section 6.4, Appendix E.5 | True-state belief 0.58 to 0.82 (downward, 1b to 2) and 0.86 to 0.91 (upward, 2 to 3); efficiency and profit ratios | `results/analysis/ladder_mechanisms_seed42.csv` |
| Appendix C | Variants, trade-level p-values, opening and closing versions | `results/analysis/noinfo_variants.csv`, `noinfo_trade_level.csv` |
| Appendix D.4 | Opening gap 0.79 in D and 0.29 in A | `results/analysis/human_closeness.csv` |
| Appendix D.5 | From the no-information price, upward D⁰₁ 0.33–0.71 and D⁰_T 0.83–0.99 | `results/analysis/human_noinfo_distance.csv` |
| Appendix D.6 | First-tenth medians −5 (downward) and −65 (upward) francs | `results/analysis/human_opening_window.csv` (`F_median`) |

## Other generated files

- `results/human/trades.csv`: every transcribed transaction, joined to its market, period, realized state, information condition, μ (`vbar`), RE and PI benchmarks, side, and the index D.
- `results/human/period_table.csv`: one row per period (61 rows), with benchmarks, first, mean, and last prices, D₁, D̄, D_T, and within-period path measures.
- `results/analysis/all_llm_discovery_periods.csv` and `all_llm_mechanisms.csv`: period-level rows from every shipped `metrics.json`, the common input to the LLM tables. The engine calls the price position D `discovery`.
- `results/analysis/file_manifest_sha256.csv`: size and SHA-256 of every file under `data/` and `code/`, written on each run.
- `results/analysis/analysis_summary.json`: headline counts and notes printed at the end of Stage 2.
