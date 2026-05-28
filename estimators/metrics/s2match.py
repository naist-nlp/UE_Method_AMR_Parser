"""S2match pairwise similarity via subprocess (batch mode, uses GloVe vectors)."""

import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_S2MATCH_DIR = (
    _HERE.parent.parent / "similarities" / "amr-metric-suite" / "py3-Smatch-and-S2match" / "smatch"
)
_S2MATCH_SCRIPT = _S2MATCH_DIR / "s2match.py"
_VECTORS = (
    _HERE.parent.parent / "similarities" / "amr-metric-suite" / "vectors" / "glove.6B.100d.txt"
)


def _amr_to_file_entry(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines) + "\n\n"


def _parse_batch_f1(stdout, n_pairs):
    scores = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Smatch score F1 "):
            try:
                scores.append(float(line.split()[-1]))
            except ValueError:
                pass
    if len(scores) > n_pairs:
        scores = scores[:n_pairs]
    return scores


def _batch_score(graphs_a, graphs_b):
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
            [sys.executable, str(_S2MATCH_SCRIPT), "-f", str(f1), str(f2),
             "-vectors", str(_VECTORS), "--ms", "--pr"],
            capture_output=True, text=True, cwd=str(_S2MATCH_DIR),
        )
    if result.returncode != 0:
        return [0.0] * n_pairs
    scores = _parse_batch_f1(result.stdout, n_pairs)
    if len(scores) != n_pairs:
        return [0.0] * n_pairs
    return scores


_CHUNK_SIZE = 100


def compute_matrix_batch(all_graphs):
    """Compute pairwise S2match matrices for all sentences, chunked to keep subprocesses manageable."""
    mats = [np.eye(len(g), dtype=float) for g in all_graphs]
    for chunk_start in range(0, len(all_graphs), _CHUNK_SIZE):
        chunk = all_graphs[chunk_start:chunk_start + _CHUNK_SIZE]
        meta, pairs_a, pairs_b = [], [], []
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


def compute_single(amr1, amr2):
    scores = _batch_score([amr1], [amr2])
    return scores[0] if scores else 0.0


def compute_matrix(graphs):
    """Compute k x k pairwise S2match similarity matrix."""
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
