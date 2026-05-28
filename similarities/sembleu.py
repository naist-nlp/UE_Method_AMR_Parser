"""SemBLEU quality metric wrapper (used to score top-1 candidate against gold)."""

import re
import subprocess
import tempfile
from pathlib import Path

SEMBLEU_DIR = Path(__file__).resolve().parent / "sembleu"
EVAL_PY = SEMBLEU_DIR / "src" / "eval.py"


def _write_amr_file(amr_text, filepath):
    with open(filepath, "w") as f:
        f.write(amr_text.strip())
        f.write("\n\n")


def _parse_score(output):
    for line in output.strip().splitlines():
        line = line.strip()
        if line.startswith(("loading", "evaluating", "time:", "Auto_reweigh")):
            continue
        try:
            val = float(line)
            return val / 100.0 if val > 1 else val
        except Exception:
            pass
    raise ValueError(f"Could not parse SemBLEU output:\n{output}")


def compute(amr1, amr2):
    """Compute SemBLEU similarity between two AMR graphs. Returns float in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pred_file = tmpdir / "pred.amr"
        gold_file = tmpdir / "gold.amr"

        amr1 = re.sub(r'~\d+', '', amr1)
        amr2 = re.sub(r'~\d+', '', amr2)

        _write_amr_file(amr1, pred_file)
        _write_amr_file(amr2, gold_file)

        try:
            result = subprocess.run(
                ["python3", str(EVAL_PY), str(pred_file), str(gold_file)],
                capture_output=True, text=True, check=True,
            )
            return float(_parse_score(result.stdout))
        except Exception:
            return 0.0
