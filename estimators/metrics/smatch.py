"""
Smatch similarity for AMR graphs via subprocess (batch mode).

Uses the amr-metric-suite py3 implementation.
"""

import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_SMATCH_DIR = _HERE.parent.parent / "similarities" / "amr-metric-suite" / "py3-Smatch-and-S2match" / "smatch"
_SMATCH_SCRIPT = _SMATCH_DIR / "smatch.py"


def _amr_to_file_entry(text: str) -> str:
    """Strip # metadata and return clean AMR string with trailing blank line."""
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines) + "\n\n"


def _parse_batch_f1(stdout: str, n_pairs: int) -> List[float]:
    """Parse F1 scores from smatch --ms output. Returns one float per pair."""
    scores = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or ":" in line:
            continue
        try:
            scores.append(float(line))
        except ValueError:
            pass
    # The last token from Document F-score line is also parsed; drop it
    # We should have exactly n_pairs scores (the per-pair F1s before the document average)
    if len(scores) > n_pairs:
        scores = scores[:n_pairs]
    return scores


def _batch_score(graphs_a: List[str], graphs_b: List[str]) -> List[float]:
    """Score n_pairs pairs in one subprocess call."""
    n_pairs = len(graphs_a)
    if n_pairs == 0:
        return []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        f1 = tmpdir / "pred.amr"
        f2 = tmpdir / "gold.amr"
        f1.write_text("".join(_amr_to_file_entry(g) for g in graphs_a))
        f2.write_text("".join(_amr_to_file_entry(g) for g in graphs_b))

        result = subprocess.run(
            [sys.executable, str(_SMATCH_SCRIPT), "-f", str(f1), str(f2), "--ms", "--pr"],
            capture_output=True,
            text=True,
            cwd=str(_SMATCH_DIR),
        )

    # A failed run scores every pair in the call 0.0, as in the paper's runs.
    if result.returncode != 0:
        _warn_failed(n_pairs, result.stderr)
        return [0.0] * n_pairs

    scores = _parse_batch_f1(result.stdout, n_pairs)
    if len(scores) != n_pairs:
        _warn_failed(n_pairs, f"{len(scores)} scores parsed")
        return [0.0] * n_pairs
    return scores


def _warn_failed(n_pairs: int, reason: str) -> None:
    last = reason.strip().splitlines()[-1] if reason.strip() else "no error output"
    print(f"  [smatch] WARNING: subprocess failed, all {n_pairs} pairs scored 0.0 ({last})",
          file=sys.stderr)


def compute_single(amr1: str, amr2: str) -> float:
    """Compute Smatch F1 between two AMR graphs."""
    scores = _batch_score([amr1], [amr2])
    return scores[0] if scores else 0.0


_CHUNK_SIZE = 100  # max sentences per subprocess call (~k*(k-1)/2 * 100 pairs)


def compute_matrix_batch(all_graphs: List[List[str]]) -> List[np.ndarray]:
    """
    Compute pairwise similarity matrices for multiple sentences.

    Processes sentences in chunks of _CHUNK_SIZE to avoid subprocess failures
    on very large batches (thousands of pairs in a single smatch call).
    """
    mats = [np.eye(len(g), dtype=float) for g in all_graphs]

    for chunk_start in range(0, len(all_graphs), _CHUNK_SIZE):
        chunk = all_graphs[chunk_start:chunk_start + _CHUNK_SIZE]

        meta: List[tuple] = []
        pairs_a: List[str] = []
        pairs_b: List[str] = []

        for local_idx, graphs in enumerate(chunk):
            k = len(graphs)
            for i in range(k):
                for j in range(i + 1, k):
                    meta.append((chunk_start + local_idx, i, j))
                    pairs_a.append(graphs[i])
                    pairs_b.append(graphs[j])

        if not pairs_a:
            continue

        scores = _batch_score(pairs_a, pairs_b)

        for (sent_idx, i, j), s in zip(meta, scores):
            mats[sent_idx][i, j] = s
            mats[sent_idx][j, i] = s

    return mats


def compute_matrix(graphs: List[str]) -> np.ndarray:
    """
    Compute k×k pairwise Smatch similarity matrix.

    Only upper triangle is computed; matrix is symmetrised.
    Diagonal is 1.0 by definition.
    """
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
