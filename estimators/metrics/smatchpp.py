"""
Smatch++ similarity for AMR graphs via Python API (in-process).

Uses the local smatchpp library.
"""

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


def _get_scorer() -> _Smatchpp:
    global _SCORER
    if _SCORER is None:
        _SCORER = _Smatchpp()
    return _SCORER


def compute_single(amr1: str, amr2: str) -> float:
    """Compute Smatch++ F1 between two AMR graphs."""
    try:
        a1 = _clean(amr1)
        a2 = _clean(amr2)
        result = _get_scorer().score_pair(a1, a2)
        return float(result["main"]["F1"]) / 100.0
    except Exception:
        return 0.0


def _clean(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines)


def compute_matrix(graphs: List[str]) -> np.ndarray:
    """
    Compute k×k pairwise Smatch++ similarity matrix.

    Diagonal is 1.0; upper triangle computed, then symmetrised.
    """
    k = len(graphs)
    mat = np.eye(k, dtype=float)
    scorer = _get_scorer()

    for i in range(k):
        for j in range(i + 1, k):
            try:
                a1 = _clean(graphs[i])
                a2 = _clean(graphs[j])
                result = scorer.score_pair(a1, a2)
                s = float(result["main"]["F1"]) / 100.0
            except Exception:
                s = 0.0
            mat[i, j] = s
            mat[j, i] = s

    return mat
