# Data codebook

All prices are in experimental francs. Markets 1–5 are the original Plott and Sunder (1982) markets, and markets A and B are the constructed markets. As in the paper's run inventory, run IDs use the prefix `m7` for market A and `m8` for market B. The engine's own records (`meta.json`, `metrics.json`) store A and B under the internal codes 7 and 8. Every CSV file uses A and B.

## `human/plott_sunder_1982_prices.csv`

987 transactions in 61 periods across markets 1–5, transcribed from Appendix B, "Bids, Offers, Prices" (typescript pp. 81–97), of Plott and Sunder (1980), Caltech Social Science Working Paper 331. The working paper is the pre-publication version of Plott and Sunder (1982, *JPE* 90(4): 663–698).

| Column | Description |
|---|---|
| `market` | Market number, 1–5. |
| `period` | Trading period ("YEAR" in the appendix): 1–11, 1–11, 1–12, 1–14, and 1–13 in markets 1–5. |
| `txn` | Transaction number within the period, in the order the appendix lists it. |
| `price` | Transaction price. |
| `wp_page` | Working-paper typescript page on which the period's listing begins. |
| `printed_period_avg` | Average price printed under the period's panel in working-paper Figures 7–11 (published Figures 2–6). |
| `recon_period_avg` | Mean of the transcribed prices in the period. |
| `period_validated` | `yes` when `recon_period_avg` reproduces `printed_period_avg` to within one franc; `NO` otherwise. |

**Transcription.** The 17 appendix pages were rendered at 400 dpi, cut into their 65 printed columns, and read by hand. No character in the file comes from automated recognition. A row is a transaction only when the appendix gives a buyer, a seller, and a numeric price. Standing bids and offers that never traded are excluded. Buyer and seller identifiers were transcribed but are not released, because the published paper provides no record against which to check them.

**Validation.** 60 of 61 periods reproduce the printed average to within one franc. In 58 of them the printed integer is exactly the rounded transcribed mean (`results/human/rounding_check.csv`). The exception is market 1, period 2 (typescript pp. 81–82, 14 rows flagged `NO`). The appendix lists 14 trades: 250, 255, 265, 260, 255, 260, 260, 275, 275, 275, 260, 275, 270, 260. They sum to 3,695 and average 263.93, against a printed 269. Reaching 269 would require a sum of 3,766, so a single missing trade would have to be at 340 francs, far outside the panel's plotted range. The neighbouring periods match (239.41 against 239, and 271.79 against 272). Either the appendix omits a trade or the printed average is wrong. The period has no insiders and enters no comparison in the paper.

## `llm/metrics/<group>/<run_id>/`

One directory per LLM session (51 in total), each holding two files named by the session's start stamp:

- **`<stamp>.meta.json`**: the run record written by the engine. It holds the scenario file name (`scenario`), the full configuration (`config`: market, seed, state-sequence preset, turns per period, trading and memory rules, and agent settings including model, temperature, and reasoning), the realized state and information sequence (`sequence.states`, `sequence.info`), per-period summaries, and token totals. Endpoints and credentials appear only as environment-variable names.
- **`<stamp>.metrics.json`**: the engine's summary of the session's event log (`python -m ps1982 metrics`), under `sessions.<id>`:
  - `meta`: market, seed, and sequence preset.
  - `paper.prices`: per period, the state, information condition, number of trades, and first, last, and mean transaction price.
  - `paper.discovery_by_informed_side.periods`: per insider period, the informed side, the RE point benchmark (`re`), the highest uninformed prior valuation (`uninformed_level`), the mean price, and the normalized index D (`discovery`).
  - `paper.efficiency`: allocative efficiency (`E_pct`) and transaction efficiency (`TE_pct`) by period.
  - `paper.insider_profit_ratio`: mean insider and uninformed profits by period.
  - `llm.posterior_convergence`: uninformed traders' mean reported probability of the realized state.
  - `llm.basis_drift`: counts of the basis traders cite for their beliefs (`prior`, `price`, `others_behavior`).
  - `book`, and the remaining `paper` and `llm` entries: further engine diagnostics not used in the paper.

The stamp links each session to its raw log (`<stamp>.jsonl`; see `raw_log_manifest.csv`). The README describes the groups, and `results/analysis/llm_run_inventory.csv` lists every session.

## `llm/llm_transactions.csv`

6,808 public trades from the insider periods of the 45 sessions outside the `rounds` group, extracted from the raw logs by `code/extract_llm_transactions.py`. The script checks that each period's trade count equals the count in `metrics.json`.

| Column | Description |
|---|---|
| `group`, `run_id`, `session` | Session identifiers, as in `metrics/`. |
| `market` | Market: 1–5, A, or B. |
| `seed` | Session seed. |
| `period` | Trading period. |
| `state` | Realized state. |
| `info` | Information condition (always `insider`). |
| `side` | Informed direction at the prior benchmark: `buyer` if P_RE > v̄, `seller` if P_RE < v̄. |
| `uninformed_level` | v̄, the highest uninformed prior valuation. |
| `re_price` | P_RE, the fully revealing point benchmark. |
| `trade_index` | Position k of the trade within the period, 1…N. |
| `n_trades` | N, number of trades in the period. |
| `u` | Relative transaction position (k−1)/(N−1); 0 when N = 1. |
| `price` | Transaction price. |
| `discovery` | Normalized index D = (price − v̄)/(P_RE − v̄): 0 at v̄, 1 at P_RE. |

## `llm/prompts/`

Rendered system prompts received by a Type I trader in market A at each rung of the disclosure ladder. Other trader types and market B differ only in the type name and dividend lines.

| File | Rung |
|---|---|
| `marketA_typeI_rung0_baseline.txt` | 0: baseline instructions. |
| `marketA_typeI_rung1_structure.txt` | 1: all dividend schedules and the number of informed traders disclosed. |
| `marketA_typeI_rung1b_implementation.txt` | 1b: adds the profit objective, clue-certainty emphasis, and persistent memo. |
| `marketA_typeI_rung2_info_status.txt` | 2: adds the current-period information-status announcement. |
| `marketA_typeI_rung3_fixed.txt` | 3: discloses that informed identities are fixed across periods. |
| `rung2_period_announcements.txt` | The two period lines shown at rungs 2 and 3. |

## `llm/raw_log_manifest.csv`

The 51 raw JSONL event logs (3.3 GB), which are not included in this repository. They contain every prompt, model response, stated reasoning, belief report, quote, and trade. Part of this material can be browsed at <https://plott-demo.siruizou.com/>. The complete logs are available from the author on request (<siruizou2005@gmail.com>).

`file` is the log's path relative to the simulation repository root, followed by its size in `bytes` and its `sha256` hash.
