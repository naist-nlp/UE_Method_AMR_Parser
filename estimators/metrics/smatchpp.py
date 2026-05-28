"""Smatch++ pairwise similarity using the in-process Python API."""

import sys
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_SMATCHPP_DIR = _HERE.parent.parent / "similarities" / "smatchpp"

if str(_SMATCHPP_DIR) not in sys.path:
    sys.path.insert(0, str(_SMATCHPP_DIR))

from smatchpp import Smatchpp as _Smatchpp

_SCORER = None


def _get_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = _Smatchpp()
    return _SCORER


def _clean(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines)


def compute_single(amr1, amr2):
    """Compute Smatch++ F1 between two AMR graphs."""
    try:
        result = _get_scorer().score_pair(_clean(amr1), _clean(amr2))
        return float(result["main"]["F1"]) / 100.0
    except Exception:
        return 0.0


def compute_matrix(graphs):
    """Compute k x k pairwise Smatch++ similarity matrix."""
    k = len(graphs)
    mat = np.eye(k, dtype=float)
    scorer = _get_scorer()
    for i in range(k):
        for j in range(i + 1, k):
            try:
                result = scorer.score_pair(_clean(graphs[i]), _clean(graphs[j]))
                s = float(result["main"]["F1"]) / 100.0
            except Exception:
                s = 0.0
            mat[i, j] = s
            mat[j, i] = s
    return mat
