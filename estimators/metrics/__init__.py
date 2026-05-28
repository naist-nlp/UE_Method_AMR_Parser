"""Registry of AMR pairwise similarity metrics used for UE estimation."""

from .smatch   import compute_matrix as smatch_matrix,   compute_matrix_batch as smatch_batch
from .smatchpp import compute_matrix as smatchpp_matrix
from .s2match  import compute_matrix as s2match_matrix,  compute_matrix_batch as s2match_batch
from .sema     import compute_matrix as sema_matrix
from .wlk      import compute_matrix as wlk_matrix,      compute_matrix_batch as wlk_batch

AMR_SIMILARITY_METRICS = {
    "smatch":   smatch_matrix,
    "smatchpp": smatchpp_matrix,
    "s2match":  s2match_matrix,
    "sema":     sema_matrix,
    "wlk":      wlk_matrix,
}

# Batch variants process all sentences in a single subprocess call.
# Metrics without a batch version fall back to per-sentence computation.
AMR_SIMILARITY_METRICS_BATCH = {
    "smatch":   smatch_batch,
    "smatchpp": None,
    "s2match":  s2match_batch,
    "sema":     None,
    "wlk":      wlk_batch,
}
