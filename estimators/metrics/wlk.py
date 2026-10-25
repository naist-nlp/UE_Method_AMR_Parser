"""
Weisfeiler-Leman Kernel (WLK) similarity for AMR graphs via subprocess (batch mode).
"""

import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_WLK_DIR = _HERE.parent.parent / "similarities" / "weisfeiler-leman-amr-metrics" / "src"
_WLK_SCRIPT = _WLK_DIR / "main_wlk.py"


def _amr_to_file_entry(text: str) -> str:
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines) + "\n\n"


def _parse_wlk_output(stdout: str, n_pairs: int) -> List[float]:
    """Parse one float per line from WLK output."""
    scores = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            scores.append(float(line))
        except ValueError:
            pass
    return scores[:n_pairs]


def _batch_score(graphs_a: List[str], graphs_b: List[str]) -> List[float]:
    n_pairs = len(graphs_a)
    if n_pairs == 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        f1 = tmpdir / "a.amr"
        f2 = tmpdir / "b.amr"
        f1.write_text("".join(_amr_to_file_entry(g) for g in graphs_a))
        f2.write_text("".join(_amr_to_file_entry(g) for g in graphs_b))

        result = subprocess.run(
            [sys.executable, str(_WLK_SCRIPT), "-a", str(f1), "-b", str(f2)],
            capture_output=True,
            text=True,
            cwd=str(_WLK_DIR),
        )

    # A failed run scores every pair 0.0, as in the paper's runs. This happened
    # for IBM-Neural at k = 5, 8, 10: one candidate graph with an empty concept
    # makes the Smatch++ standardiser used by WLK raise IndexError.
    if result.returncode != 0:
        _warn_failed(n_pairs, result.stderr)
        return [0.0] * n_pairs

    scores = _parse_wlk_output(result.stdout, n_pairs)
    if len(scores) != n_pairs:
        _warn_failed(n_pairs, f"{len(scores)} scores parsed")
        return [0.0] * n_pairs
    return scores


def _warn_failed(n_pairs: int, reason: str) -> None:
    last = reason.strip().splitlines()[-1] if reason.strip() else "no error output"
    print(f"  [wlk] WARNING: subprocess failed, all {n_pairs} pairs scored 0.0 ({last})",
          file=sys.stderr)


def compute_matrix_batch(all_graphs: List[List[str]]) -> List[np.ndarray]:
    """Compute pairwise WLK matrices for all sentences in one subprocess call."""
    meta: List[tuple] = []
    pairs_a: List[str] = []
    pairs_b: List[str] = []

    for sent_idx, graphs in enumerate(all_graphs):
        k = len(graphs)
        for i in range(k):
            for j in range(i + 1, k):
                meta.append((sent_idx, i, j))
                pairs_a.append(graphs[i])
                pairs_b.append(graphs[j])

    scores = _batch_score(pairs_a, pairs_b) if pairs_a else []

    mats = [np.eye(len(g), dtype=float) for g in all_graphs]
    for (sent_idx, i, j), s in zip(meta, scores):
        mats[sent_idx][i, j] = s
        mats[sent_idx][j, i] = s

    return mats


def compute_matrix(graphs: List[str]) -> np.ndarray:
    """Compute k×k pairwise WLK similarity matrix."""
    k = len(graphs)
    mat = np.eye(k, dtype=float)

    pairs_a, pairs_b, pair_idx = [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            pairs_a.append(graphs[i])
            pairs_b.append(graphs[j])
            pair_idx.append((i, j))

    if not pairs_a:
        return mat

    scores = _batch_score(pairs_a, pairs_b)
    for (i, j), s in zip(pair_idx, scores):
        mat[i, j] = s
        mat[j, i] = s

    return mat
