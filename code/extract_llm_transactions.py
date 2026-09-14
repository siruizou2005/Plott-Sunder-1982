#!/usr/bin/env python3
"""Extract compact transaction paths from the source simulation JSONL logs.

The package ships this compact CSV (data/llm/llm_transactions.csv) so that figures
and tables reproduce without several gigabytes of model prompts and responses.  Each
retained row is a public trade from an insider period used by the paper.

Rebuilding it requires the raw JSONL event logs of the simulation runs, which are
available from the author on request (see data/llm/raw_log_manifest.csv for the files
and their SHA-256 hashes):

    python3 code/extract_llm_transactions.py --runs /path/to/Plott-Sunder-1982-LLM/runs

The script matches each shipped metrics file to its log by run ID and timestamp, and
stops if a period's trade count disagrees with the count recorded in the metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = ROOT / "data" / "llm" / "metrics"
DEFAULT_RUNS = Path(os.environ.get("PS1982_RUNS", ROOT.parent / "Plott-Sunder-1982-LLM" / "runs"))
OUTPUT = ROOT / "data" / "llm" / "llm_transactions.csv"
# Internal engine codes of markets A and B; the output uses the paper's names.
MARKET_LABEL = {7: "A", 8: "B"}


def used_run(group: str, run_id: str) -> bool:
    if group in {"m1", "m2", "m3", "m4", "m5"}:
        return True
    selected = {
        "control": {
            "m7_ctrl_42", "m7_ctrl_43", "m7_ctrl_44", "m7_ctrl_45",
            "m8_ctrl_42", "m8_ctrl_43", "m8_ctrl_44",
        },
        "disclosed": {"m7_disc_42", "m8_disc_42"},
        "ladder1b": {"m7_lad1b_42", "m8_lad1b_42"},
        "ladder2": {"m7_lad2_42", "m7_lad2_45", "m8_lad2_42", "m8_lad2_44"},
        "ladder3": {"m7_lad3_42", "m7_lad3_45", "m8_lad3_42", "m8_lad3_44"},
    }
    return run_id in selected.get(group, set())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                        help="directory holding the raw JSONL logs (searched recursively)")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="output CSV")
    args = parser.parse_args()
    if not args.runs.is_dir():
        raise SystemExit(f"raw log directory not found: {args.runs}")

    source_logs: dict[tuple[str, str], Path] = {}
    for path in args.runs.rglob("*.jsonl"):
        source_logs[(path.parent.name, path.stem)] = path

    rows: list[dict] = []
    missing: list[str] = []
    for metrics_path in sorted(METRICS_ROOT.glob("**/*.metrics.json")):
        rel = metrics_path.relative_to(METRICS_ROOT)
        group = rel.parts[0]
        run_id = metrics_path.parent.name
        if not used_run(group, run_id):
            continue
        stem = metrics_path.name.removesuffix(".metrics.json")
        log_path = source_logs.get((run_id, stem))
        if log_path is None:
            missing.append(f"{run_id}/{stem}")
            continue

        metrics = json.loads(metrics_path.read_text())
        period_meta: dict[tuple[int, int], dict] = {}
        run_meta: dict[int, dict] = {}
        for session_id, session in metrics["sessions"].items():
            sid = int(session_id)
            meta = session["meta"]
            run_meta[sid] = {
                "market": MARKET_LABEL.get(int(meta["market"]), int(meta["market"])),
                "seed": int(meta.get("seed", -1)),
            }
            prices = {int(x["period"]): x for x in session["paper"].get("prices", [])}
            for d in session["paper"]["discovery_by_informed_side"]["periods"]:
                period = int(d["period"])
                pr = prices[period]
                period_meta[(sid, period)] = {
                    "state": d["state"],
                    "info": "insider",
                    "side": d["side"],
                    "uninformed_level": float(d["uninformed_level"]),
                    "re_price": float(d["re"]),
                    "reported_n_trades": int(pr["n_trades"]),
                }

        trades: dict[tuple[int, int], list[float]] = {}
        with log_path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("type") != "trade":
                    continue
                key = (int(event.get("session", 0)), int(event["period"]))
                if key in period_meta:
                    trades.setdefault(key, []).append(float(event["payload"]["price"]))

        for key, prices in trades.items():
            sid, period = key
            pm = period_meta[key]
            if len(prices) != pm["reported_n_trades"]:
                raise ValueError(
                    f"trade count mismatch for {run_id}, session {sid}, period {period}: "
                    f"{len(prices)} != {pm['reported_n_trades']}"
                )
            denominator = pm["re_price"] - pm["uninformed_level"]
            n = len(prices)
            for index, price in enumerate(prices, start=1):
                rows.append({
                    "group": group,
                    "run_id": run_id,
                    "session": sid,
                    **run_meta[sid],
                    "period": period,
                    **{k: v for k, v in pm.items() if k != "reported_n_trades"},
                    "trade_index": index,
                    "n_trades": n,
                    "u": (index - 1) / (n - 1) if n > 1 else 0.0,
                    "price": price,
                    "discovery": (price - pm["uninformed_level"]) / denominator,
                })

    if missing:
        raise FileNotFoundError("Missing source logs: " + ", ".join(missing))
    out = pd.DataFrame(rows).sort_values(
        ["group", "run_id", "session", "period", "trade_index"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out):,} trades from {out['run_id'].nunique()} sessions to {args.out}")


if __name__ == "__main__":
    main()
