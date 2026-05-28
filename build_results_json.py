#!/usr/bin/env python3
"""
Reads all per-parser CSV files produced by run_all.py and writes a single
consolidated results.json used by the paper figure scripts.

Run this after run_all.py has finished:
    python build_results_json.py
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "estimators" / "results"

BASELINE_PARSERS = [
    "SPRING",
    "IBM_parser",
    "HMG-AMR-T5",
    "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44",
    "ibm_amr3.0_neur_al",
]

BEAMS = [3, 5, 8, 10]


def read_csv(path):
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ue = row["UE_Method"]
            data[ue] = {
                "S++": float(row["S++"]),
                "S2":  float(row["S2"]),
                "SB":  float(row["SB"]),
            }
    return data


def main():
    combined = {}
    for parser in BASELINE_PARSERS:
        beam_data = {}
        for beam in BEAMS:
            csv_path = RESULTS_DIR / f"results_{parser.replace('/', '_')}_beam{beam}.csv"
            if not csv_path.exists():
                print(f"  MISSING: {csv_path.name}")
                continue
            beam_data[str(beam)] = read_csv(csv_path)
        if beam_data:
            combined[parser] = beam_data
            print(f"  Loaded {parser}: beams={list(beam_data.keys())}")

    out = RESULTS_DIR / "results.json"
    out.write_text(json.dumps(combined, indent=2))
    print(f"\nWrote {out}  ({len(combined)} parsers)")


if __name__ == "__main__":
    main()
