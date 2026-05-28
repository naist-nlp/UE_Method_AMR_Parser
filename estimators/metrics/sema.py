"""SEMA pairwise similarity using the in-process Python API."""

import sys
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_SEMA_DIR = _HERE.parent.parent / "similarities" / "sema"

if str(_SEMA_DIR) not in sys.path:
    sys.path.insert(0, str(_SEMA_DIR))

_Sema = None
_SemaAMR = None


def _load_sema():
    global _Sema, _SemaAMR
    if _Sema is None:
        from sema import Sema
        from amr import AMR as SemaAMR
        _Sema = Sema
        _SemaAMR = SemaAMR


def _amr_to_oneline(text):
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return " ".join(lines)


def _score_pair(amr1_text, amr2_text):
    _load_sema()
    try:
        a1 = _SemaAMR.parse_AMR_line(_amr_to_oneline(amr1_text))
        a2 = _SemaAMR.parse_AMR_line(_amr_to_oneline(amr2_text))
        if a1 is None or a2 is None:
            return 0.0
        scorer = _Sema()
        scorer.compute_sema(a1, a2)
        p = scorer.m / scorer.c if scorer.c > 0 else 0.0
        r = scorer.m / scorer.t if scorer.t > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    except Exception:
        return 0.0


def compute_single(amr1, amr2):
    return _score_pair(amr1, amr2)


def compute_matrix(graphs):
    """Compute k x k pairwise SEMA similarity matrix."""
    k = len(graphs)
    mat = np.eye(k, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            s = _score_pair(graphs[i], graphs[j])
            mat[i, j] = s
            mat[j, i] = s
    return mat
