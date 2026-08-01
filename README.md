# Fast When Insiders Sell, Slow When They Buy

Replication package for the paper on within-period price adjustment in
Plott and Sunder (1982), *Journal of Political Economy* 90(4), 663–698.

Every number in the paper is recomputed here from two inputs: the five page scans of
the original's Figures 2–6, and the design parameters the original published. There
are no other data, and the package is self-contained — nothing outside this directory
is read. The manuscript itself is not distributed here; this repository is the code,
the data and the tables.

A companion repository, [Plott-Sunder-1982-LLM](https://github.com/siruizou2005/Plott-Sunder-1982-LLM),
replays the same five markets with LLM agents in place of human subjects. It is a
separate program with a separate question; the only thing the two share is the design
parameters, and the check that they agree on them is `code/test_against_engine.py`
below. This package does not need it to run.

## Run it

```
python3 code/run_all.py          # scans -> tables -> figures, about 10 seconds
```

Requires `numpy`, `pandas`, `scipy`, `statsmodels`, `matplotlib` and `Pillow`
(`pip install -r requirements.txt`).

## Pipeline

| Stage | Script | Input | Output |
|---|---|---|---|
| 1 | `code/human_trades.py` | `data/scan_p17-21.png` | `out/human_trades.csv`, `tab/D0_digitization_quality.csv` |
| 2 | `code/period_table.py` | `out/human_trades.csv` | `tab/period_table.csv` |
| 3 | `code/results.py` | `tab/period_table.csv` | the result tables below |
| 4 | `code/figures.py` | `tab/*.csv` | `fig/fig1-4.png` |

**Stage 1** recovers the individual transactions from the dots plotted in the
original's figures. The panels are 3300×5300 px; trade dots are 11–15 px thick
against reference lines of 6–8 px, so a distance transform separates them, and trades
sit on a regular x-grid whose pitch resolves flat runs into counts. Each panel is
rotated 0.2–0.3° and carries a full-width information-condition arrow, both corrected
for. 937 trades across all 61 periods.

Validation: each period's recovered mean is compared against the AVERAGE PRICE the
original printed under the panel. Median absolute residual 0.52 francs, 90th
percentile 1.26, maximum 2.80 — at the rounding floor of a table printed in whole
francs. **No step of the digitization fits to the printed mean**; it is held back
purely as a target. Three periods of market 4 are flagged `source_inconsistent`
because the printed mean cannot be produced by any trades plotted in the panel, and
market 1 period 2 is flagged `mean_mismatch`; they are recorded, not adjusted.

**Stage 2** recomputes each period's geometry from the design parameters — the
uninformed level v̄, the fully revealing price RE, which side the informed profit on,
the competitive interval — and measures the path inside the period.

**Stage 3** writes one file per table or claim:

| File | What it holds |
|---|---|
| `identity.csv` | Proposition 1's crosstabulation: 17 informed-selling periods scored, 19 informed-buying discarded, exactly diagonal |
| `within_period.csv` | the main contrast (paper Table 3) |
| `robustness.csv` | Panel A four inference schemes, Panel B four samples (paper Table 7) |
| `by_market.csv` | the contrast disaggregated by market |
| `established.csv` | the k≥2 and k≥3 subsamples, the authors' own carve-out |
| `interval_scoring.csv` | scoring against the competitive interval, not the point |
| `repetition.csv` | what repetition of the same state changes |
| `baseline.csv` | the information-free account and each prediction it fails |
| `baseline_sensitivity.csv` | that account under four benchmark constructions |
| `endowment_capacity.csv` | the capacity refutation, from endowments alone |

## Two things a reader should know

**The no-information benchmark excludes each market's opening period.** Period 1 of a
market carries a large downward transient — market 4's alone is −210 francs, and it is
also flagged `source_inconsistent` — which flips the sign of the mean drift: all 18
periods give −15.8 francs, while excluding opening periods gives +7.7 with 12 of 13
positive. The exclusion is a concession to the account being tested, since it gives
that account the largest drift it can claim. `baseline_sensitivity.csv` reports all
four constructions, and the paper's conclusion holds at every one. Under the primary
construction the selling side does not deviate from the predicted opening level
(p = 0.18), which is why the paper leads with the benchmark-free comparison instead:
the two sides' opening trades fall on opposite sides of v̄, 75.7 francs apart.

**The 27 insider periods come from only three markets.** Welch tests on 27 periods
overstate independence, since every period of one (market, side) cell shares a v̄, an
RE price and a group of subjects. `robustness.csv` Panel A therefore also reports
within-market permutation tests and market fixed effects with standard errors
clustered on market. The level and francs contrasts survive all four; the
within-period slope contrast weakens to p = 0.065 under clustering, which the paper
states rather than omits.

## Notes on provenance

`code/ps1982_params.py` replaces `ps1982/markets.py` of the
[LLM-agent repository](https://github.com/siruizou2005/Plott-Sunder-1982-LLM)'s
simulation engine, so this package runs standalone. It has two parts, with different
provenance, and an auditor should treat them differently:

- The **data block** (`_MARKETS`, `INITIAL_CERTS`, `INITIAL_CASH`, `FIXED_COST`,
  `URN`, `CLUE_DRAWS`) was machine-emitted by printing the engine's own objects, so it
  is value-for-value identical to the engine and can be diffed against it. Everything
  in it is a published parameter of the original experiment: the dividend table and
  priors (Table 1), each period's realized state and information condition (Figures
  2–6), and the endowments (Instruction Set 2).
- The **methods** on the `_Market` shim (`posterior_from_card`, `card_for`,
  `theory_price`, `re_price`, `pi_price`, `informed_side`, and the derived properties)
  were **re-implemented by hand**, not copied. They reproduce the engine's arithmetic
  but are not textually equivalent to it: the docstrings are shorter, and `card_for`
  drops the engine's `rnd` parameter and its state-redraw fallback, which exist to
  serve resampled machine-agent runs and are unreachable here — this package only ever
  reads the state sequence the original published. Do not diff these against the
  engine and expect a match; check them against the arithmetic instead.

`python3 code/test_against_engine.py [path/to/Plott-Sunder-1982-LLM]` is that check.
With no argument it looks for a checkout beside this one. It imports both
modules and compares them value-for-value — the constants, every parameter field, and
`card_for`, `posterior_from_card`, `theory_price`, `re_price`, `pi_price` and
`informed_side` — on all 101 periods of all 8 markets in the engine's table, and
confirms that both raise where market 1 has no printed clue card. It currently passes.
The engine is not needed to run the package, only to run this check, which skips with
a message when the engine is not on disk.

One deliberate difference in behaviour: `re_price` keeps RE exact where the engine's
`theory_price` rounds to whole francs. This affects only market 1's four sampled-clue
periods, by at most 0.45 francs, and never changes a side or the diagonal.
`theory_price`, which stage 1 uses, keeps the engine's rounding.

`code/q1_within.py` and `code/digitize.py` are the original analysis modules with
their absolute paths repointed at this directory; `q1_within.py` additionally has its
machine-agent branch removed, this paper being human-data only. `code/figstyle.py`
holds the figure-style helpers, copied so the figures render identically elsewhere.

Known issue in the source, worth recording: the original's footnote 6 lists market
3's separating periods as 3, 5, 7, 8, 10, but the correct set is 3, 5, 6, 8, 10.
Period 6 is state Y with RE 175 against PI 220 (the predictions differ, average price
173) while period 7 is state X with RE = PI = 400 (they coincide, average price 364).
The count of 17 is unaffected, so no published claim changes — but a fresh
recomputation will disagree with that list, and the disagreement is the original's
typo, not a bug here.

`code/published_tables.py` holds the numbers transcribed from the printed rows of the
figures. `python3 code/montage.py` overlays the detected dots on the scan crops and
writes `fig/digitization_audit.png`, so the detection can be checked by eye rather
than taken on trust; it is not part of `run_all.py`.

## License

CC BY 4.0 — see `LICENSE`. The grant covers this package's own code, tables and
figures. It does not extend to `data/scan_p17-21.png`, which are page scans of the
original article and remain the publisher's copyright; they are included only as the
input the digitization is run on.
