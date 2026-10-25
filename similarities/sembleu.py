"""
Official SemBLEU wrapper.

This module:
1. Creates temporary AMR files
2. Calls official SemBLEU evaluator
3. Parses score
4. Returns normalized similarity in [0,1]

ACL 2019:
SemBleu: A Robust Metric for AMR Parsing Evaluation
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

# Resolved from this file so the wrapper works from any working directory
SEMBLEU_DIR = Path(__file__).resolve().parent / "sembleu"

EVAL_PY = SEMBLEU_DIR / "src" / "eval.py"


# ============================================================
# TEMP FILE WRITER
# ============================================================

def write_temp_amr(amr_text: str, filepath: Path):
    """
    Write single AMR to file.

    SemBLEU expects:
    - AMR
    - blank newline
    """

    with open(filepath, "w") as f:

        f.write(amr_text.strip())

        f.write("\n\n")


# ============================================================
# OUTPUT PARSER
# ============================================================

def parse_score(output: str) -> float:
    """
    Parse SemBLEU score from stdout.

    We extract the FIRST standalone numeric line
    before the timing output.

    Example stdout:

        loading ...
        evaluating ...
        1.0
        time: ...

    Returns
    -------
    float
        SemBLEU score in [0,1]
    """

    lines = output.strip().splitlines()

    for line in lines:

        line = line.strip()

        # Skip logging lines
        if (
            line.startswith("loading")
            or line.startswith("evaluating")
            or line.startswith("time:")
            or line.startswith("Auto_reweigh")
        ):
            continue

        # Try parsing standalone score line
        try:

            val = float(line)

            # normalize if needed
            if val > 1:
                val /= 100.0

            return val

        except:
            pass

    raise ValueError(
        f"Could not parse SemBLEU output:\n{output}"
    )

# ============================================================
# MAIN API
# ============================================================

def compute(amr1: str, amr2: str) -> float:
    """
    Compute SemBLEU similarity.

    Parameters
    ----------
    amr1 : str
        First AMR graph

    amr2 : str
        Second AMR graph

    Returns
    -------
    float
        SemBLEU similarity in [0,1]
    """

    with tempfile.TemporaryDirectory() as tmpdir:

        tmpdir = Path(tmpdir)

        pred_file = tmpdir / "pred.amr"
        gold_file = tmpdir / "gold.amr"

        # Strip alignment markers (e.g. ~0, ~1 from IBM parser; ~e.0 handled by SemBLEU internally)
        amr1 = re.sub(r'~\d+', '', amr1)
        amr2 = re.sub(r'~\d+', '', amr2)

        # Write temporary AMR files
        write_temp_amr(amr1, pred_file)
        write_temp_amr(amr2, gold_file)

        # sys.executable: the environment's own python (the runs used `python3`
        # inside the activated env, which is the same interpreter)
        cmd = [
            sys.executable,
            str(EVAL_PY),
            str(pred_file),
            str(gold_file)
        ]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            stdout = result.stdout
            score = parse_score(stdout)

            return float(score)

        except subprocess.CalledProcessError as e:

            # print("\nSemBLEU execution failed")
            # print(e.stderr)

            return 0.0

        except Exception:
            return 0.0


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    amr1 = """
    (b / boy
        :ARG0-of (s / smile-01))
    """

    amr2 = """
    (b / boy
        :ARG0-of (l / laugh-01))
    """

    score = compute(amr1, amr2)

    print("\nSemBLEU score:")
    print(score)