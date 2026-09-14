.PHONY: all human analysis transactions clean

PYTHON ?= python3
# Directory holding the raw JSONL logs of the simulation runs (only for `make transactions`).
RUNS ?= ../Plott-Sunder-1982-LLM/runs

export MPLBACKEND := Agg
export OPENBLAS_NUM_THREADS := 1
export VECLIB_MAXIMUM_THREADS := 1

all: human analysis

# Stage 1: Appendix B transcription -> human period data, Table 1 ledger, Figure 1.
human:
	$(PYTHON) code/human/run_all.py

# Stage 2: every table and figure in the paper, plus CSV summaries.
analysis: human
	$(PYTHON) code/analyze.py

# Optional stage 0: rebuild data/llm/llm_transactions.csv from the raw JSONL logs.
transactions:
	$(PYTHON) code/extract_llm_transactions.py --runs $(RUNS)

# Remove generated outputs. market4_periods78.pdf is a static figure and is kept.
clean:
	rm -rf results/human results/analysis results/tables
	find results/figures -type f ! -name market4_periods78.pdf -delete
