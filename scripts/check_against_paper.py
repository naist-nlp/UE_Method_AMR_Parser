#!/usr/bin/env python3
"""
check_against_paper.py — compare the reproduced numbers with the paper.

  * every value of the 24 PRR tables (6 parsers x 4 beams x 17 UE methods x
    3 quality metrics) against expected/, the tables behind the paper's
    Table 4 and Figures 1-11
  * Tables 1 and 3 (top-1 Smatch and Smatch++) against the published values

    python3 run_all.py --n_samples 0
    python3 scripts/check_against_paper.py

Exit code 0 only if everything matches.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import DISPLAY, RESULTS_DIR, table1, table3   # noqa: E402

EXPECTED_DIR = Path(__file__).resolve().parent.parent / "expected"

# Published values: Table 1 (Smatch, Smatch++) and Table 3 (LLM rows)
PAPER_TABLE1 = {
    "SPRING": (0.8449, 0.8438),
    "amr3joint_ontowiki2_seed44": (0.8374, 0.8378),
    "HMG-AMR-BART": (0.8358, 0.8380),
    "IBM_parser": (0.8128, 0.8181),
    "ibm_amr3.0_neur_al": (0.8115, 0.8174),
    "HMG-AMR-T5": (0.7693, 0.7797),
}
PAPER_TABLE3_LLM = {"Qwen3-32B_p1": 0.1214, "Ministral-14B_p1": 0.0897}


def read(path: Path) -> dict:
    with open(path) as f:
        return {r["UE_Method"]: r for r in csv.DictReader(f)}


def main() -> int:
    n_val = n_bad = 0
    missing = []
    for exp_file in sorted(EXPECTED_DIR.glob("results_*.csv")):
        ours_file = RESULTS_DIR / exp_file.name
        if not ours_file.exists():
            missing.append(exp_file.name)
            continue
        exp, ours = read(exp_file), read(ours_file)
        for m, row in exp.items():
            for q in ("S++", "S2", "SB"):
                n_val += 1
                got = ours.get(m, {}).get(q)
                if got != row[q]:
                    n_bad += 1
                    if n_bad <= 10:
                        print(f"  DIFF {exp_file.stem} {m} {q}: reproduced {got}, paper {row[q]}")
    print(f"PRR tables: {n_val - n_bad}/{n_val} values identical to the paper"
          + (f"; missing {len(missing)} files (run run_all.py --n_samples 0)" if missing else ""))

    t_val = t_bad = 0
    t1, t3 = table1(), table3()
    checks = [(f"Table 1 Smatch   {DISPLAY[p]}", t1[p][0], v[0]) for p, v in PAPER_TABLE1.items()]
    checks += [(f"Table 1 Smatch++ {DISPLAY[p]}", t1[p][1], v[1]) for p, v in PAPER_TABLE1.items()]
    checks += [(f"Table 3 Smatch   {DISPLAY[p]}", t3[p][0], v) for p, v in PAPER_TABLE3_LLM.items()]
    for name, got, paper in checks:
        t_val += 1
        if got is None or f"{got:.4f}" != f"{paper:.4f}":
            t_bad += 1
            print(f"  DIFF {name}: reproduced {'n/a' if got is None else f'{got:.4f}'}, paper {paper:.4f}")
    print(f"Tables 1 and 3: {t_val - t_bad}/{t_val} values identical to the paper "
          "(Smatch++ recomputed from the score files, Smatch from the paper's run log)")

    # Smatch re-run: random restarts seeded from the clock, so only close to the paper
    paper_smatch = {**{p: v[0] for p, v in PAPER_TABLE1.items()}, **PAPER_TABLE3_LLM}
    rerun = {p: t3[p][1] for p in paper_smatch}
    if all(v is not None for v in rerun.values()):
        diff = max(abs(rerun[p] - v) for p, v in paper_smatch.items())
        print(f"Smatch re-run: largest difference to the paper {diff:.4f} "
              "(random restarts of the Smatch hill climbing)")

    return 0 if (n_bad == 0 and t_bad == 0 and not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
