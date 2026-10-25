"""
Full UE pipeline for k-best AMR parsing.

Steps:
  1. Load k-best candidates per sentence
  2. Build pairwise similarity matrices (N, k, k) for each AMR metric
  3. Run lm_polygraph UE estimators: ue_vec (N,) per method
  4. Score top-1 candidate vs gold: qual_vec (N,) per quality metric
  5. Compute PRR(ue_vec, qual_vec) for every (UE method, quality metric) pair

Steps 2-4 are cached as .npz files of numbers only (no sentences or graphs).
When every cache file exists, the pipeline runs without the k-best data.
"""

import re
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# ISI neural-aligner tags, e.g. "go-02~1" → "go-02".  IBM parser output only.
_ISI_TAG_RE = re.compile(r'~\d+')


def _clean_graph(graph: str) -> str:
    """Strip ISI alignment tags and leading comment lines from a graph string."""
    lines = [l for l in graph.splitlines() if not l.strip().startswith('#')]
    return _ISI_TAG_RE.sub('', '\n'.join(lines)).strip()


# _clean_graph was added to load_data between the paper's runs for these
# parsers (started 14 May 2026) and those for the IBM parsers (16 May 2026),
# so their candidates were used as stored, comment lines included. Of the
# scores, only LexicalSimilarity reads the graph strings directly (the metric
# wrappers drop comment lines themselves; these outputs have no ~N tags).
UNCLEANED_IN_PAPER_RUNS = {"SPRING", "HMG-AMR-T5", "HMG-AMR-BART"}

# lm_polygraph estimators (requires polygraph conda env)
from lm_polygraph.estimators import (
    LexicalSimilarity,
    NumSemSets,
    EigValLaplacian,
    DegMat,
    Eccentricity,
)
from lm_polygraph.ue_metrics.ue_metric import normalize

# Local AMR similarity metrics
from estimators.metrics import AMR_SIMILARITY_METRICS, AMR_SIMILARITY_METRICS_BATCH
from estimators.metrics.smatchpp import compute_single as smatchpp_single
from estimators.metrics.s2match import compute_single as s2match_single

# SemBLEU lives under similarities/
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from similarities.sembleu import compute as sembleu_single

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARSERS = [
    "SPRING",
    "IBM_parser",
    "HMG-AMR-T5",
    "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44",   # MBSE in the paper
    "ibm_amr3.0_neur_al",           # IBM-Neural in the paper
]
BEAM_SIZES = [3, 5, 8, 10]

QUALITY_METRIC_FNS = {
    "S++": smatchpp_single,
    "S2":  s2match_single,
    "SB":  sembleu_single,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_sent_dir(data_dir: Path, parser: str, beam_size: int) -> Path:
    base = data_dir / parser / f"kbest_beam{beam_size}"
    nested = base / f"kbest_beam{beam_size}" / "sentences"
    if nested.exists():
        return nested
    flat = base / "sentences"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"sentences dir not found under {base}")


def load_data(
    parser: str,
    beam_size: int,
    n_samples: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> List[dict]:
    """Load sentence dicts for one parser/beam_size, skipping sentences with empty gold graphs."""
    if data_dir is None:
        data_dir = _HERE / "data" / "k_top_AMR_parsers"
    sent_dir = _get_sent_dir(data_dir, parser, beam_size)
    files = sorted(sent_dir.glob("sent_*.json"))
    if not files:
        raise FileNotFoundError(f"No sentence files in {sent_dir}")
    sentences = [json.loads(f.read_text()) for f in files]
    # Strip ISI alignment tags (e.g. ~5) present in IBM parser output
    if parser not in UNCLEANED_IN_PAPER_RUNS:
        for s in sentences:
            for c in s.get("candidates", []):
                c["graph"] = _clean_graph(c["graph"])
    valid = [s for s in sentences if s.get("gold_graph", "").strip()]
    if len(valid) < len(sentences):
        print(f"  [load_data] skipped {len(sentences) - len(valid)} sentences with empty gold_graph")
    if n_samples and n_samples > 0:
        valid = valid[:n_samples]
    return valid


def cached_n(cache_dir: Path, parser: str, beam_size: int) -> int:
    """Number of sentences N of the largest cached run for one parser/beam_size."""
    ns = [
        int(re.search(r"_n(\d+)\.npz$", p.name).group(1))
        for p in cache_dir.glob(f"{parser}_beam{beam_size}_qual_*_n*.npz")
    ]
    if not ns:
        raise FileNotFoundError(f"no cached scores for {parser} beam{beam_size} in {cache_dir}")
    return max(ns)


# ---------------------------------------------------------------------------
# Similarity matrix computation
# ---------------------------------------------------------------------------


def build_sim_matrices(
    sentences: Optional[List[dict]],
    similarity_metrics: List[str],
    cache_dir: Optional[Path] = None,
    parser: str = "",
    beam_size: int = 0,
    n: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute pairwise similarity matrices for each metric.

    Uses batch subprocess calls where available (smatch, s2match, wlk) to
    amortize startup / model-loading overhead across all sentences.

    Results are cached in cache_dir as .npz files so interrupted runs can resume.
    sentences=None (no k-best data) reads every matrix from the cache; n is then
    the number of sentences in the cached run.

    Returns: {metric_name: np.ndarray shape (N, k, k)}
    """
    if sentences is not None:
        n = len(sentences)
    matrices: Dict[str, np.ndarray] = {}

    for metric_name in tqdm(similarity_metrics, desc="sim metrics"):
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{parser}_beam{beam_size}_{metric_name}_n{n}.npz"

        if cache_path is not None and cache_path.exists():
            matrices[metric_name] = np.load(cache_path)["matrix"]
            tqdm.write(f"  [{metric_name}] loaded from cache")
            continue

        if sentences is None:
            raise FileNotFoundError(f"no k-best data and no cached matrix {cache_path}")
        all_graphs = [[c["graph"] for c in s["candidates"]] for s in sentences]
        tqdm.write(f"  [{metric_name}] computing...")
        batch_fn = AMR_SIMILARITY_METRICS_BATCH.get(metric_name)
        if batch_fn is not None:
            mats = batch_fn(all_graphs)
        else:
            compute_fn = AMR_SIMILARITY_METRICS[metric_name]
            mats = [compute_fn(graphs) for graphs in tqdm(
                all_graphs, desc=f"  {metric_name}", leave=False
            )]

        stacked = np.stack(mats, axis=0)
        matrices[metric_name] = stacked

        if cache_path is not None:
            np.savez_compressed(cache_path, matrix=stacked)

    return matrices


# ---------------------------------------------------------------------------
# Stats dict builder
# ---------------------------------------------------------------------------


def _build_stats(sentences: Optional[List[dict]], sim_matrix: np.ndarray) -> dict:
    """
    Build the stats dict expected by lm_polygraph estimators.

    The graph-based estimators read only the matrices and the number of samples
    in sample_texts, so without k-best data empty strings stand in for the graphs.
    """
    if sentences is not None:
        sample_texts = [[c["graph"] for c in s["candidates"]] for s in sentences]
    else:
        n, k, _ = sim_matrix.shape
        sample_texts = [[""] * k for _ in range(n)]
    return {
        "sample_texts": sample_texts,
        "semantic_matrix_entail": sim_matrix,         # (N, k, k)
        "semantic_matrix_contra": 1.0 - sim_matrix,   # (N, k, k)
    }


# ---------------------------------------------------------------------------
# UE estimator definitions
# ---------------------------------------------------------------------------


def _make_ue_estimators(metric_names: List[str]) -> Dict[str, object]:
    """
    Build the full set of UE estimators as a dict: name -> estimator.

    LexicalSimilarity and NumSemSets appear once (metric-independent or tied
    to the first similarity metric in metric_names).
    """
    estimators: Dict[str, object] = {
        "LexicalSimilarity": LexicalSimilarity(metric="rougeL"),
        "NumSemSets":        NumSemSets(),
    }
    for mn in metric_names:
        label = mn  # e.g. "smatch", "smatchpp", ...
        estimators[f"EigValLaplacian_{label}"] = EigValLaplacian(
            similarity_score="NLI_score", affinity="entail"
        )
        estimators[f"DegMat_{label}"] = DegMat(
            similarity_score="NLI_score", affinity="entail"
        )
        estimators[f"Eccentricity_{label}"] = Eccentricity(
            similarity_score="NLI_score", affinity="entail"
        )
    return estimators


# ---------------------------------------------------------------------------
# UE vector computation
# ---------------------------------------------------------------------------


def run_ue_estimators(
    sentences: Optional[List[dict]],
    sim_matrices: Dict[str, np.ndarray],
    similarity_metrics: List[str],
    cache_dir: Optional[Path] = None,
    parser: str = "",
    beam_size: int = 0,
    use_stored: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run all UE estimators.

    LexicalSimilarity is the only estimator that reads the graph strings, so its
    scores are cached in cache_dir like the similarity matrices.

    use_stored: return the scores stored in <parser>_beam<k>_ue_n<N>.npz when
    it holds every estimator. They were computed on the machine that reproduces
    the paper's runs. Recomputing gives the same scores up to the last bits of
    the eigen-decompositions, which differ between CPUs; for Eccentricity this
    can move an eigenvalue across its 0.9 cut-off, and for every estimator it
    can change the order of tied scores.

    Returns: {estimator_name: np.ndarray (N,)}
    """
    estimators = _make_ue_estimators(similarity_metrics)
    ue_vectors: Dict[str, np.ndarray] = {}

    # LexicalSimilarity only needs sample_texts; use first metric's matrix for stats
    first_metric = similarity_metrics[0]
    n = len(sim_matrices[first_metric])

    if use_stored and cache_dir is not None and ue_path(cache_dir, parser, beam_size, n).exists():
        with np.load(ue_path(cache_dir, parser, beam_size, n)) as z:
            if set(estimators) <= set(z.files):
                tqdm.write("  [UE] stored scores loaded from cache")
                return {name: z[name] for name in estimators}

    base_stats = _build_stats(sentences, sim_matrices[first_metric])

    lex_cache = None
    if cache_dir is not None:
        lex_cache = cache_dir / f"{parser}_beam{beam_size}_ue_LexicalSimilarity_n{n}.npz"
    if lex_cache is not None and lex_cache.exists():
        ue_vectors["LexicalSimilarity"] = np.load(lex_cache)["scores"]
        tqdm.write("  [LexicalSimilarity] loaded from cache")
    else:
        if sentences is None:
            raise FileNotFoundError(f"no k-best data and no cached LexicalSimilarity scores {lex_cache}")
        lex_est = estimators["LexicalSimilarity"]
        ue_vectors["LexicalSimilarity"] = lex_est(base_stats)
        if lex_cache is not None:
            np.savez_compressed(lex_cache, scores=ue_vectors["LexicalSimilarity"])

    # NumSemSets uses the first AMR similarity metric
    num_est = estimators["NumSemSets"]
    ue_vectors["NumSemSets"] = num_est(base_stats)

    # Graph-based estimators: one set per similarity metric
    for mn in similarity_metrics:
        stats = _build_stats(sentences, sim_matrices[mn])
        ue_vectors[f"EigValLaplacian_{mn}"] = estimators[f"EigValLaplacian_{mn}"](stats)
        ue_vectors[f"DegMat_{mn}"] = estimators[f"DegMat_{mn}"](stats)
        ue_vectors[f"Eccentricity_{mn}"] = estimators[f"Eccentricity_{mn}"](stats)

    return ue_vectors


# ---------------------------------------------------------------------------
# Quality vector computation
# ---------------------------------------------------------------------------


def compute_quality_vectors(
    sentences: Optional[List[dict]],
    quality_metrics: Optional[List[str]] = None,
    cache_dir: Optional[Path] = None,
    parser: str = "",
    beam_size: int = 0,
    n: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Score top-1 candidate vs gold for each sentence.

    s2match is batched (one subprocess call for all N sentences).
    smatchpp uses in-process API. sembleu uses per-sentence subprocess.
    Results are cached in cache_dir as .npz files.
    sentences=None (no k-best data) reads every vector from the cache.

    Returns: {quality_metric_name: np.ndarray (N,)}
    """
    if quality_metrics is None:
        quality_metrics = list(QUALITY_METRIC_FNS.keys())
    if sentences is not None:
        n = len(sentences)

    qual_vectors: Dict[str, np.ndarray] = {}

    for qm in tqdm(quality_metrics, desc="quality metrics", leave=False):
        # Check cache
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{parser}_beam{beam_size}_qual_{qm}_n{n}.npz"
        if cache_path is not None and cache_path.exists():
            qual_vectors[qm] = np.load(cache_path)["scores"]
            tqdm.write(f"  [qual {qm}] loaded from cache")
            continue
        if sentences is None:
            raise FileNotFoundError(f"no k-best data and no cached quality scores {cache_path}")
        top1_graphs = [s["candidates"][0]["graph"] for s in sentences]
        gold_graphs  = [s["gold_graph"] for s in sentences]
        score_fn = QUALITY_METRIC_FNS[qm]
        scores_list = []
        for top1, gold in tqdm(
            zip(top1_graphs, gold_graphs), total=len(top1_graphs),
            desc=f"  {qm}", leave=False
        ):
            try:
                scores_list.append(float(score_fn(top1, gold)))
            except Exception:
                scores_list.append(0.0)
        scores = np.array(scores_list, dtype=float)

        qual_vectors[qm] = scores
        if cache_path is not None:
            np.savez_compressed(cache_path, scores=scores)

    return qual_vectors


# ---------------------------------------------------------------------------
# PRR computation
# ---------------------------------------------------------------------------


_ORDER_TOL = 1e-9   # float noise allowed when checking that a stored order sorts a vector


def _rejection_area(order: np.ndarray, target: np.ndarray) -> float:
    """
    PredictionRejectionArea.__call__ of lm_polygraph 0.7.0 (max_rejection=1.0),
    with the sort order of the uncertainty scores passed in.
    """
    target = normalize(target)
    num_obs = len(order)
    num_rej = int(1.0 * num_obs)
    sorted_metrics = np.array(target)[order]
    cumsum = np.cumsum(sorted_metrics)[-num_rej:]
    scores = (cumsum / np.arange((num_obs - num_rej) + 1, num_obs + 1))[::-1]
    return np.sum(scores) / num_rej


def _sort_order(name: str, vec: np.ndarray, orders: Optional[dict], used: list) -> np.ndarray:
    """
    np.argsort(vec), which is what PredictionRejectionArea uses, unless `orders`
    holds an order for `name` that sorts vec (up to float noise).

    np.argsort is not stable: the order it gives tied scores depends on the
    CPU's SIMD sort routine, and ties are common here (e.g. all k candidates
    identical). Orders recorded on a machine that reproduces the paper's runs
    therefore give the paper's PRR values on any machine.
    """
    vec = np.asarray(vec, dtype=float)
    if orders is not None and name in orders:
        o = np.asarray(orders[name], dtype=np.int64)
        if (len(o) == len(vec)
                and np.array_equal(np.sort(o), np.arange(len(vec)))
                and np.all(np.diff(vec[o]) >= -_ORDER_TOL)):
            used.append(name)
            return o
    return np.argsort(vec)


def sort_orders(
    ue_vectors: Dict[str, np.ndarray],
    qual_vectors: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """The np.argsort order of every vector compute_prr sorts, on this machine."""
    n = len(next(iter(qual_vectors.values())))
    orders = {name: np.argsort(np.asarray(v, dtype=float)) for name, v in ue_vectors.items()}
    for qm_name, qual_vec in qual_vectors.items():
        orders[f"oracle_{qm_name}"] = np.argsort(-np.asarray(qual_vec, dtype=float))
    orders["random"] = np.argsort(np.zeros(n))
    return orders


def compute_prr(
    ue_vectors: Dict[str, np.ndarray],
    qual_vectors: Dict[str, np.ndarray],
    orders: Optional[dict] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized PRR per paper formula (12):
        PRR = (AUCPRunc - AUCPRrandom) / (AUCPRoracle - AUCPRrandom)

    - Oracle UE = -qual_vec : lowest quality → highest UE → rejected first.
    - Random  UE = 0.0 for every sentence. All scores tie, so this area is the
      one of the order in which the sort returns tied entries (stored as
      "random" in the order files).
    - orders: optional {vector name: sort order}, see _sort_order().

    Returns: {ue_method: {quality_metric: prr_normalized}}
    """
    results: Dict[str, Dict[str, float]] = {}
    used: list = []

    # Pre-compute oracle and random areas for each quality metric
    oracle_areas: Dict[str, float] = {}
    random_areas: Dict[str, float] = {}
    for qm_name, qual_vec in qual_vectors.items():
        N = len(qual_vec)
        try:
            oracle_areas[qm_name] = float(_rejection_area(
                _sort_order(f"oracle_{qm_name}", -qual_vec, orders, used),
                qual_vec,
            ))
            random_areas[qm_name] = float(_rejection_area(
                _sort_order("random", np.zeros(N), orders, used),
                qual_vec,
            ))
        except Exception:
            oracle_areas[qm_name] = float("nan")
            random_areas[qm_name] = float("nan")

    for ue_name, ue_vec in ue_vectors.items():
        results[ue_name] = {}
        ue_order = _sort_order(ue_name, ue_vec, orders, used)
        for qm_name, qual_vec in qual_vectors.items():
            try:
                raw_area = float(_rejection_area(ue_order, qual_vec))
                oracle = oracle_areas[qm_name]
                random = random_areas[qm_name]
                denom = oracle - random
                if abs(denom) > 1e-9 and not np.isnan(denom):
                    prr_normalized = (raw_area - random) / denom
                else:
                    prr_normalized = float("nan")
                results[ue_name][qm_name] = prr_normalized
            except Exception:
                results[ue_name][qm_name] = float("nan")

    if orders is not None:
        n_vectors = len(ue_vectors) + len(qual_vectors) + 1   # UE, oracle per metric, random
        n_used = len(set(used))
        print(f"  tie order: stored order used for {n_used}/{n_vectors} sorted vectors"
              + ("" if n_used == n_vectors else " (np.argsort for the rest)"))
    return results


def order_path(cache_dir: Path, parser: str, beam_size: int, n: int) -> Path:
    return cache_dir / f"{parser}_beam{beam_size}_order_n{n}.npz"


def ue_path(cache_dir: Path, parser: str, beam_size: int, n: int) -> Path:
    return cache_dir / f"{parser}_beam{beam_size}_ue_n{n}.npz"


def load_orders(cache_dir: Optional[Path], parser: str, beam_size: int, n: int) -> Optional[dict]:
    """Stored sort orders for one parser/beam_size, or None if there are none."""
    if cache_dir is None or not order_path(cache_dir, parser, beam_size, n).exists():
        return None
    with np.load(order_path(cache_dir, parser, beam_size, n)) as z:
        return {name: z[name] for name in z.files}


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    parser: str,
    beam_size: int,
    n_samples: Optional[int] = None,
    similarity_metrics: Optional[List[str]] = None,
    quality_metrics: Optional[List[str]] = None,
    data_dir: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
    tie_order: str = "original",
    recompute_ue: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run the full UE pipeline for one (parser, beam_size) combination.

    Without k-best data for this parser/beam_size, every score is read from
    cache_dir. tie_order="original" uses the stored sort orders when present
    (see _sort_order); tie_order="numpy" always uses np.argsort.
    recompute_ue=True recomputes the UE scores instead of using the stored ones
    (see run_ue_estimators).

    Returns PRR results dict: {ue_method: {quality_metric: prr_score}}.
    """
    if similarity_metrics is None:
        similarity_metrics = list(AMR_SIMILARITY_METRICS.keys())
    if quality_metrics is None:
        quality_metrics = list(QUALITY_METRIC_FNS.keys())

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Parser: {parser}  |  Beam: {beam_size}  |  n_samples: {n_samples or 'all'}")
        print(f"{'='*60}")

    # 1. Load data (or fall back to the cached scores)
    try:
        sentences = load_data(parser, beam_size, n_samples, data_dir)
        N = len(sentences)
        k = sentences[0]["num_candidates"]
        if verbose:
            print(f"Loaded {N} sentences, k={k} candidates each")
    except FileNotFoundError:
        if cache_dir is None:
            raise
        sentences = None
        N = n_samples or cached_n(cache_dir, parser, beam_size)
        if verbose:
            print(f"No k-best data for {parser} beam{beam_size}: using cached scores (N={N})")

    # 2. Build similarity matrices
    if verbose:
        print(f"Building similarity matrices for: {similarity_metrics}")
    sim_matrices = build_sim_matrices(
        sentences, similarity_metrics,
        cache_dir=cache_dir, parser=parser, beam_size=beam_size, n=N,
    )

    # 3. Run UE estimators
    if verbose:
        print("Running UE estimators...")
    ue_vectors = run_ue_estimators(
        sentences, sim_matrices, similarity_metrics,
        cache_dir=cache_dir, parser=parser, beam_size=beam_size,
        use_stored=not recompute_ue,
    )

    # 4. Compute quality vectors
    if verbose:
        print(f"Computing quality scores ({quality_metrics})...")
    qual_vectors = compute_quality_vectors(
        sentences, quality_metrics,
        cache_dir=cache_dir, parser=parser, beam_size=beam_size, n=N,
    )

    # 5. PRR = (AUCPRunc - AUCPRrandom) / (AUCPRoracle - AUCPRrandom)
    if verbose:
        print("Computing PRR = (AUCPRunc - random) / (AUCPRoracle - random)...")
    orders = load_orders(cache_dir, parser, beam_size, N) if tie_order == "original" else None
    prr_results = compute_prr(ue_vectors, qual_vectors, orders=orders)

    return prr_results
