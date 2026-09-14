# Output map

This file maps each exhibit and cited number in the paper to the program that produces it and the file that holds it. Table and figure numbers follow the September 2026 manuscript; LaTeX labels are given so the map survives renumbering. All paths are relative to the repository root. Run IDs use the prefix `m7` for market A and `m8` for market B, as in the paper.

Programs:

- **Stage 1**: `code/human/run_all.py`, which runs `load_trades.py`, `period_table.py`, `results.py`, and `fig1_identity.py`.
- **Stage 2**: `code/analyze.py`.

## Main text

| Exhibit | Label | Output file | Program | Numbers behind it |
|---|---|---|---|---|
| Table 1: the 61 periods by information condition and direction | `tab:ledger` | `results/tables/selection_ledger.tex` | Stage 2 (`selection_tables`) | `results/analysis/selection_ledger.csv`; classification in `results/human/identity.csv` |
| Figure 1: RE and PI predictions differ only when insiders sell | `fig:selection` | `results/figures/selection_identity.png` | Stage 1 (`fig1_identity.py`) | `results/human/identity.csv` |
| Table 2: human price positions | `tab:human` | `results/tables/human_adjustment.tex` | Stage 2 (`human_analysis`) | `results/analysis/human_summary.csv`, `human_tests.csv` |
| Figure 2: human transaction paths | `fig:humanpaths` | `results/figures/human_paths.pdf` | Stage 2 (`human_analysis`) | `results/human/trades.csv` |
| Table 3: direction and bundled market conditions | `tab:design` | `results/tables/design_asymmetries.tex` | Stage 2 (`design_table`) | `results/analysis/design_asymmetries.csv`; benchmarks in `theory_benchmarks_exact_signal.csv` |
| Table 4: adjustment in original and constructed LLM markets | `tab:symmetric` | `results/tables/symmetric_comparison.tex` | Stage 2 (`symmetric_analysis`) | `results/analysis/symmetric_comparison.csv`, `llm_price_discovery_periods.csv` |
| Figure 3: pooled directional gap under the redesign | `fig:symmetric` | `results/figures/symmetric_markets.pdf` | Stage 2 (`symmetric_analysis`) | `data/llm/llm_transactions.csv` |
| Figure 4: disclosure responses by market and session | `fig:marketheterogeneity` | `results/figures/disclosure_market_heterogeneity.pdf` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_session_effects.csv` |
| Figure 5: five-rung paths at seed 42 | `fig:ladder` | `results/figures/disclosure_ladder.pdf` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_price_discovery_periods.csv` |
| Table 5: paired changes along the ladder | `tab:ladder` | `results/tables/ladder_effects.tex` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_paired_effects.csv` |
| Table 6: beliefs, allocations, and profits at seed 42 | `tab:mechanisms` | `results/tables/ladder_mechanisms.tex` | Stage 2 (`ladder_analysis`) | `results/analysis/ladder_mechanisms_seed42.csv` |

## Appendices

| Exhibit | Label | Output file | Program | Numbers behind it |
|---|---|---|---|---|
| Table A.1: audit of the 987 transactions | `tab:app-human-validation` | `results/tables/appendix_human_validation.tex` | Stage 2 | `results/human/validation.csv`, `rounding_check.csv` |
| Table A.2: the 27 main-sample periods | `tab:app-human-main` | `results/tables/appendix_human_main_periods.tex` | Stage 2 | `results/analysis/human_main_27_periods.csv` |
| Table A.3: internal inconsistencies | `tab:app-inconsistencies` | typed in the manuscript | none | data sources cited in the table |
| Figure A.1: market 4, periods 7–8 | `fig:app-market4-offset` | `results/figures/market4_periods78.pdf` | static file | left panel: `data/human/plott_sunder_1982_prices.csv`; right panel: working-paper Figure 10 |
| Table B.1: market structure | `tab:app-market-structure` | `results/tables/appendix_market_structure.tex` | Stage 2 | `results/analysis/market_structures.csv` |
| Table B.2: dividend schedules | `tab:app-dividends` | `results/tables/appendix_market_dividends.tex` | Stage 2 | `results/analysis/market_dividends.csv` |
| Table B.3: exact-signal benchmarks | `tab:app-benchmarks` | `results/tables/appendix_theory_benchmarks.tex` | Stage 2 | `results/analysis/theory_benchmarks_exact_signal.csv` |
| Table B.4: market 1 posterior benchmarks | `tab:app-market1` | `results/tables/appendix_market1_posterior.tex` | Stage 2 | `results/analysis/market1_posterior_benchmarks.csv` |
| Table B.5: competitive-price intervals in A/B | `tab:app-competitive-intervals` | `results/tables/appendix_competitive_intervals.tex` | Stage 2 (`selection_tables`) | `results/analysis/competitive_intervals_AB.csv` |
| Table C.1: institutional comparison | `tab:app-institution` | typed in the manuscript | none | simulation rules in each session's `meta.json` |
| Table D.1: LLM run inventory | `tab:app-run-list` | `results/tables/appendix_run_inventory.tex` | Stage 2 (`build_run_inventory`) | `results/analysis/llm_run_inventory.csv` |
| Table E.1: human results by market | `tab:app-human-market` | `results/tables/appendix_human_by_market.tex` | Stage 2 | `results/analysis/human_results_by_market.csv` |
| Table E.2: human inference and sample robustness | `tab:app-human-robustness` | `results/tables/appendix_human_robustness.tex` | Stage 1 (`results.py`), typeset by Stage 2 | `results/human/robustness.csv` |
| Table E.3: design comparison by session | `tab:app-symmetric-sessions` | `results/tables/appendix_symmetric_sessions.tex` | Stage 2 | `results/analysis/symmetric_session_gaps.csv` |
| Table E.4: disclosure effects by session pair | `tab:app-ladder-sessions` | `results/tables/appendix_ladder_sessions.tex` | Stage 2 | `results/analysis/ladder_session_effects.csv` |
| Table E.5: seed-42 common support | `tab:app-common-support` | `results/tables/appendix_common_support.tex` | Stage 2 | `results/analysis/ladder_common_support_seed42.csv` |
| Table E.6: longer trading horizons | `tab:app-rounds` | `results/tables/appendix_rounds.tex` | Stage 2 | `results/analysis/extended_rounds_market4.csv` |

The rendered treatment prompts quoted in Appendix C are in `data/llm/prompts/`.

## Numbers cited in the text

| Where | Statement | File and field |
|---|---|---|
| Section 2.2, footnote | One trader's cash buys at least 25 certificates at 400 francs | `results/human/endowment_capacity.csv` |
| Section 2.3 | 17 selling and 19 buying insider periods; RE and PI differ exactly in the selling periods | `results/human/identity.csv` (`side`, `predictions_differ`); `results/analysis/analysis_summary.json` (`selection_crosstab`) |
| Section 3.1, Appendix A.1 | 60 of 61 periods validate; two periods 0.022 and 0.024 francs beyond the rounding bound | `results/human/validation.csv`, `results/human/rounding_check.csv` (also printed by `load_trades.py`) |
| Section 3.2 | Opening gap 79.4 francs (p = 0.001); change gap 73.1 (p = 0.001); closing gap 6.3 (p = 0.186); D₁ gap 0.789 (p = 0.001); D_T different (p < 0.001) | `results/analysis/human_tests.csv` (`dev_francs_first`, `francs_change`, `dev_francs_last`, `D_first`, `D_last`) |
| Section 3.2 | All 14 buying periods open below the benchmark; 11 rise and 3 are unchanged; selling paths show 6 rises, 5 falls, 2 unchanged; 13 of 14 buying periods close below | `results/analysis/human_distribution.csv` |
| Section 3.2, Appendix E.1 | Mean absolute opening deviation 15.4 (selling); median absolute opening deviations 15 and 70, closing 5 and 5; selling D₁ 95% interval [0.79, 1.40]; 8 below, 2 at, 3 above at the selling opening | `results/analysis/human_distribution.csv` |
| Section 3.3 | First-tenth medians of −5 (selling) and −65 (buying) francs | `results/human/trajectory.csv`, bin 1, `F_median` |
| Section 3.3 | Familiar states: opening D of 0.99 (7 selling) and 0.72 (6 buying); buying below 1 (p = 0.017); between-side p = 0.28 | `results/human/established.csv`, rows `k>=3` |
| Section 4 | Target distances, informed incentives, and free-rider counts; the four market-5 buying periods | `results/analysis/theory_benchmarks_exact_signal.csv`, `design_asymmetries.csv`, `human_main_27_periods.csv` |
| Section 5.1 | 45 traded market-4 periods, 53 in A/B; 26 background sessions with 188 traded periods | `results/analysis/symmetric_comparison.csv` (`periods`, `sessions`) |
| Section 5.2 | Sell–buy gap in D̄: 1.175 (market 4), −0.013 (A/B); session-weighted A/B gap −0.022 | `results/analysis/symmetric_comparison.csv`; `analysis_summary.json` (`symmetric_sell_minus_buy`) |
| Section 5.2 | Human market-4 D from 0.288 to 0.891; LLM baseline closes at 0.221 | `results/analysis/human_results_by_market.csv`; `symmetric_comparison.csv` |
| Section 6 | Paired changes, session counts, and sign-flip p-values | `results/analysis/ladder_paired_effects.csv`, `ladder_session_effects.csv` |
| Section 6.3 | True-state belief, transaction efficiency, and profit ratios by rung | `results/analysis/ladder_mechanisms_seed42.csv` |

## Other generated files

- `results/human/trades.csv`: every transcribed transaction, joined to its market, period, realized state, information condition, v̄, RE and PI benchmarks, informed side, and the index D.
- `results/human/period_table.csv`: one row per period (61 rows), with benchmarks, first, mean, and last prices, D₁, D̄, D_T, and within-period path measures.
- `results/analysis/all_llm_discovery_periods.csv` and `all_llm_mechanisms.csv`: period-level rows from every shipped `metrics.json`, the common input to the LLM tables.
- `results/analysis/file_manifest_sha256.csv`: size and SHA-256 of every file under `data/` and `code/`, written on each run.
- `results/analysis/analysis_summary.json`: headline counts and notes printed at the end of Stage 2.
