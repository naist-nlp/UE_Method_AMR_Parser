"""
Registry of AMR pairwise similarity metrics.

Each metric exposes:
    compute_matrix(graphs: List[str]) -> np.ndarray  shape (k, k)
    compute_single(amr1: str, amr2: str) -> float    (optional, for quality scoring)
"""

from .smatch   import compute_matrix as smatch_matrix,   compute_matrix_batch as smatch_batch,   compute_single as smatch_single
from .smatchpp import compute_matrix as smatchpp_matrix, compute_single as smatchpp_single
from .s2match  import compute_matrix as s2match_matrix,  compute_matrix_batch as s2match_batch,  compute_single as s2match_single
from .sema     import compute_matrix as sema_matrix,     compute_single as sema_single
from .wlk      import compute_matrix as wlk_matrix,      compute_matrix_batch as wlk_batch

# Ordered as in the results table
AMR_SIMILARITY_METRICS = {
    "smatch":   smatch_matrix,
    "smatchpp": smatchpp_matrix,
    "s2match":  s2match_matrix,
    "sema":     sema_matrix,
    "wlk":      wlk_matrix,
}

# Batch versions that compute all sentences in ONE subprocess call.
# Subprocess-based metrics (smatch, s2match, wlk) benefit most.
# smatchpp and sema use in-process Python APIs and fall back to per-sentence compute.
AMR_SIMILARITY_METRICS_BATCH = {
    "smatch":   smatch_batch,
    "smatchpp": None,   # in-process, use per-sentence path
    "s2match":  s2match_batch,
    "sema":     None,   # in-process, use per-sentence path
    "wlk":      wlk_batch,
}

# Metrics available for quality scoring (top-1 vs gold)
QUALITY_METRICS = {
    "smatchpp": smatchpp_single,
    "s2match":  s2match_single,
    # sembleu is wired in pipeline directly via similarities/sembleu.py
}
