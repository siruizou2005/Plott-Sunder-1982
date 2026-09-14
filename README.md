# Replication Package: Trading Direction and Price Discovery

Data and code for

> Sirui Zou. "Trading Direction and Price Discovery: Human Evidence and LLM Experiments." Working paper, September 2026.

The package reproduces every table and figure in the paper, from two inputs:

1. **Human data.** A row-by-row transcription of every transaction in the Plott and Sunder (1982) experimental asset markets: 987 trades in 61 periods across five markets.
2. **LLM data.** Per-session outputs of the 51 large-language-model trading sessions used in the paper, recorded by the simulation engine [`Plott-Sunder-1982-LLM`](https://github.com/siruizou2005/Plott-Sunder-1982-LLM) at commit [`4e8466a`](https://github.com/siruizou2005/Plott-Sunder-1982-LLM/commit/4e8466a463119f8d327b135ea9f44e12fc5cb2c4).

No API access, model calls, or raw conversation logs are needed to reproduce the results. The full run takes under a minute on a laptop.

## Quick start

```bash
pip install -r requirements.txt
make all
```

`make all` runs two stages:

| Stage | Command | Reads | Writes |
|---|---|---|---|
| 1. Human data | `python3 code/human/run_all.py` | `data/human/plott_sunder_1982_prices.csv` | `results/human/*.csv`, `results/figures/selection_identity.png` |
| 2. Analysis | `python3 code/analyze.py` | `data/`, `results/human/` | `results/analysis/*.csv`, `results/tables/*.tex`, `results/figures/*` |

An optional stage rebuilds the compact LLM transaction file from the raw simulation logs (see [Raw simulation logs](#raw-simulation-logs)):

```bash
make transactions RUNS=/path/to/Plott-Sunder-1982-LLM/runs
```

The code was run with Python 3.13 and the package versions pinned in `requirements.txt`. PDF figures and all LaTeX table fragments regenerate identically to those in the manuscript. Because of font and version differences in matplotlib, `selection_identity.png` may differ by a few pixels of padding.

## Repository layout

```
.
├── data/                                   inputs (see data/README.md for the codebook)
│   ├── human/
│   │   └── plott_sunder_1982_prices.csv    987 transcribed transactions, 5 markets, 61 periods
│   └── llm/
│       ├── metrics/<group>/<run_id>/       per-session metrics.json and meta.json (51 sessions)
│       ├── llm_transactions.csv            6,808 public trades in the insider periods used
│       ├── prompts/                        rendered system prompts for the disclosure ladder
│       └── raw_log_manifest.csv            size and SHA-256 of the 51 raw JSONL logs (not shipped)
├── code/
│   ├── human/                              stage 1: human period data, Table 1 ledger, Figure 1
│   ├── analyze.py                          stage 2: every other table and figure
│   └── extract_llm_transactions.py         optional: raw JSONL logs -> llm_transactions.csv
├── results/                                generated outputs, committed for inspection
│   ├── human/                              period-level human data and supplementary checks
│   ├── analysis/                           CSV summaries behind every number in the text
│   ├── tables/                             LaTeX fragments \input by the manuscript
│   └── figures/                            PDF/PNG figures included by the manuscript
├── docs/
│   └── output_map.md                       paper exhibit -> program -> output file
├── Makefile
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

`docs/output_map.md` lists, for each table, figure, and number cited in the paper, the program that produces it and the file that holds it.

## Data sources

### Human transactions

Plott and Sunder (1982) publish period-average prices and price plots, but not individual transactions. The working-paper version, Caltech Social Science Working Paper 331 (Plott and Sunder, 1980; revised August 1980), prints the full record of bids, offers, and trades in its Appendix B, "Bids, Offers, Prices" (typescript pp. 81–97). `data/human/plott_sunder_1982_prices.csv` transcribes that appendix by hand, row by row; no text in it comes from automated character recognition. A row is a transaction only when the appendix lists a buyer, a seller, and a numeric price. Unaccepted bids and offers are not included.

Every period is checked against the published paper's own arithmetic: the mean of the transcribed prices must reproduce the average price printed under the corresponding panel. The check passes, within one franc, in 60 of the 61 periods. The exception is market 1, period 2, where the 14 listed trades average 263.93 francs against a printed 269. That period is a no-information period, outside every comparison in the paper. `results/human/validation.csv` and `results/human/rounding_check.csv` report the check period by period.

The appendix also records buyer and seller identifiers. They were transcribed but are not released, because the published paper offers nothing to check them against.

Market parameters (dividend schedules, priors, realized states, and information schedules) are the published values of Plott and Sunder (1982, Tables 1–2 and Figures 2–6). They are coded in `code/human/ps1982_params.py` and `code/analyze.py`.

### LLM sessions

The simulation engine, scenario files, and prompt templates are in the public repository [`siruizou2005/Plott-Sunder-1982-LLM`](https://github.com/siruizou2005/Plott-Sunder-1982-LLM). All sessions in this package were run with commit `4e8466a`. Each session's `meta.json` records its scenario, configuration, seed, and realized state and information sequence. Each `metrics.json` is the engine's summary of the session's event log (`python -m ps1982 metrics`). It contains per-period prices, the normalized price index, allocative efficiency, insider profits, and belief reports.

The 51 sessions comprise:

| Group | Sessions | Role in the paper |
|---|---:|---|
| `m1`–`m5` | 26 | Original markets 1–5: two sessions on the published state sequence and three prior redraws per market, plus one market-3 session with a second model vendor (Gemini 3.5 Flash, `m3_gem_paper`). Market 4 is the reference design. |
| `control` | 7 | Rung 0 (baseline) of constructed markets A (`m7_*`, seeds 42–45) and B (`m8_*`, seeds 42–44). Seeds 42–44 form the symmetric-market sample. |
| `disclosed` | 2 | Rung 1: dividend schedules and informed counts disclosed (seed 42). |
| `ladder1b` | 2 | Rung 1b: profit objective, clue certainty, and persistent memo (seed 42). |
| `ladder2` | 4 | Rung 2: current-period information-status announcement (A seeds 42, 45; B seeds 42, 44). |
| `ladder3` | 4 | Rung 3: identity-fixedness disclosure (same seeds as rung 2). |
| `rounds` | 6 | Market 4 with four, five, and six turns per trader, on the published and prior-redraw sequences. |

As in the paper's run inventory, run IDs use the prefix `m7` for market A and `m8` for market B. All outputs refer to the markets as A and B. Unless a table says otherwise, sessions use DeepSeek-V4-Flash at temperature 0.7 with reasoning enabled. `results/analysis/llm_run_inventory.csv` lists every session with its model, seed, realized states, and no-trade periods.

Model outputs are not deterministic. Re-running a scenario produces a new session, not a copy of the recorded one. Reproduction therefore starts from the recorded session outputs shipped here.

### Raw simulation logs

Each session also wrote a JSONL event log with the complete agent record: every prompt, model response, stated reasoning, belief report, quote, and trade. The 51 logs total 3.3 GB, and several exceed GitHub's 100 MB file limit, so they are not included here.

- **Online viewer.** Part of this material can be browsed at <https://plott-demo.siruizou.com/>.
- **Complete logs.** The full raw logs, including all agent reasoning, are available from the author on request: <siruizou2005@gmail.com>.

`data/llm/raw_log_manifest.csv` lists each log's path, size, and SHA-256 hash, so any copy can be verified. The logs are needed only to rebuild `data/llm/llm_transactions.csv` or to recompute `metrics.json` with the engine. None of the reported results reads them directly.

## What is not in this package

- The raw JSONL agent logs (see [Raw simulation logs](#raw-simulation-logs)).
- The simulation engine itself, which is versioned in its own public repository at the commit cited above.
- The Plott and Sunder (1980, 1982) texts and page images, which remain the rights holders' property. The right panel of `results/figures/market4_periods78.pdf` reproduces part of working-paper Figure 10 for comparison. That figure is a static file with no generating script.
- An earlier version of the human data, recovered by digitizing the plotted points in the published figures, which the transcription supersedes. That version remains in this repository's history under the tag `v0-digitized`. `code/human/legacy_flags.py` keeps only the list of 23 periods that version could not resolve, for one robustness row.

## References

Plott, Charles R., and Shyam Sunder. 1980. "Efficiency of Experimental Security Markets with Insider Information: An Application of Rational Expectations Models." Social Science Working Paper 331, California Institute of Technology.

Plott, Charles R., and Shyam Sunder. 1982. "Efficiency of Experimental Security Markets with Insider Information: An Application of Rational-Expectations Models." *Journal of Political Economy* 90 (4): 663–698.

## License

Code is released under the BSD 3-Clause License; data, results, and documentation under CC BY 4.0. See `LICENSE` for details and exclusions.
