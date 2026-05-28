"""Weisfeiler-Leman Kernel (WLK) pairwise similarity via subprocess (batch mode)."""

import sys
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_WLK_DIR = _HERE.parent.parent / "similarities" / "weisfeiler-leman-amr-metrics" / "src"
_WLK_SCRIPT = _WLK_DIR / "main_wlk.py"


def _amr_to_file_entry(text):
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(lines) + "\n\n"


def _parse_wlk_output(stdout, n_pairs):
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


def _batch_score(graphs_a, graphs_b):
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
            capture_output=True, text=True, cwd=str(_WLK_DIR),
        )
    if result.returncode != 0:
        return [0.0] * n_pairs
    scores = _parse_wlk_output(result.stdout, n_pairs)
    if len(scores) != n_pairs:
        return [0.0] * n_pairs
    return scores


def compute_matrix_batch(all_graphs):
    """Compute pairwise WLK matrices for all sentences in one subprocess call."""
    meta, pairs_a, pairs_b = [], [], []
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


def compute_matrix(graphs):
    """Compute k x k pairwise WLK similarity matrix."""
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
