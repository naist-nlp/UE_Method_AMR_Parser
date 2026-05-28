"""
UE pipeline for k-best AMR parsing.

Steps:
  1. Load k-best candidates per sentence
  2. Build pairwise similarity matrices (N, k, k) for each AMR similarity metric
  3. Run UE estimators to produce uncertainty scores (N,) per method
  4. Score the top-1 candidate against gold using quality metrics (N,)
  5. Compute normalized PRR for every (UE method, quality metric) pair
"""

import re
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

# Strip ISI neural-aligner tags like "go-02~1" from IBM parser output
_ISI_TAG_RE = re.compile(r'~\d+')


def _clean_graph(graph):
    lines = [l for l in graph.splitlines() if not l.strip().startswith('#')]
    return _ISI_TAG_RE.sub('', '\n'.join(lines)).strip()


from lm_polygraph.estimators import (
    LexicalSimilarity,
    EigValLaplacian,
    DegMat,
    Eccentricity,
)
from lm_polygraph.ue_metrics import PredictionRejectionArea

from estimators.metrics import AMR_SIMILARITY_METRICS, AMR_SIMILARITY_METRICS_BATCH
from estimators.metrics.smatchpp import compute_single as smatchpp_single
from estimators.metrics.s2match import compute_single as s2match_single

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))
from similarities.sembleu import compute as sembleu_single

PARSERS = [
    "SPRING",
    "IBM_parser",
    "HMG-AMR-T5",
    "HMG-AMR-BART",
    "amr3joint_ontowiki2_seed44",
    "ibm_amr3.0_neur_al",
]

BEAM_SIZES = [3, 5, 8, 10]

QUALITY_METRIC_FNS = {
    "S++": smatchpp_single,
    "S2":  s2match_single,
    "SB":  sembleu_single,
}


def _get_sent_dir(data_dir, parser, beam_size):
    base = data_dir / parser / f"kbest_beam{beam_size}"
    nested = base / f"kbest_beam{beam_size}" / "sentences"
    if nested.exists():
        return nested
    flat = base / "sentences"
    if flat.exists():
        return flat
    raise FileNotFoundError(f"sentences dir not found under {base}")


def load_data(parser, beam_size, n_samples=None, data_dir=None):
    """Load sentence JSON files for one (parser, beam_size), skipping empty gold graphs."""
    if data_dir is None:
        data_dir = _HERE / "data" / "k_top_AMR_parsers"
    sent_dir = _get_sent_dir(data_dir, parser, beam_size)
    files = sorted(sent_dir.glob("sent_*.json"))
    if not files:
        raise FileNotFoundError(f"No sentence files in {sent_dir}")
    sentences = [json.loads(f.read_text()) for f in files]
    for s in sentences:
        for c in s.get("candidates", []):
            c["graph"] = _clean_graph(c["graph"])
    valid = [s for s in sentences if s.get("gold_graph", "").strip()]
    if len(valid) < len(sentences):
        print(f"  [load_data] skipped {len(sentences) - len(valid)} sentences with empty gold")
    if n_samples and n_samples > 0:
        valid = valid[:n_samples]
    return valid


def build_sim_matrices(sentences, similarity_metrics, cache_dir=None, parser="", beam_size=0):
    """
    Compute pairwise similarity matrices for each metric.

    Uses batch subprocess calls for smatch, s2match, and wlk to amortize
    per-process startup overhead. Results are cached to disk so interrupted
    runs can resume from where they left off.

    Returns: {metric_name: np.ndarray shape (N, k, k)}
    """
    all_graphs = [[c["graph"] for c in s["candidates"]] for s in sentences]
    matrices = {}

    for metric_name in tqdm(similarity_metrics, desc="sim metrics"):
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{parser}_beam{beam_size}_{metric_name}_n{len(sentences)}.npz"

        if cache_path is not None and cache_path.exists():
            matrices[metric_name] = np.load(cache_path)["matrix"]
            tqdm.write(f"  [{metric_name}] loaded from cache")
            continue

        tqdm.write(f"  [{metric_name}] computing...")
        batch_fn = AMR_SIMILARITY_METRICS_BATCH.get(metric_name)
        if batch_fn is not None:
            mats = batch_fn(all_graphs)
        else:
            compute_fn = AMR_SIMILARITY_METRICS[metric_name]
            mats = [compute_fn(graphs) for graphs in tqdm(all_graphs, desc=f"  {metric_name}", leave=False)]

        stacked = np.stack(mats, axis=0)
        matrices[metric_name] = stacked
        if cache_path is not None:
            np.savez_compressed(cache_path, matrix=stacked)

    return matrices


def _build_stats(sentences, sim_matrix):
    """Build the stats dict expected by lm_polygraph estimators."""
    sample_texts = [[c["graph"] for c in s["candidates"]] for s in sentences]
    return {
        "sample_texts": sample_texts,
        "semantic_matrix_entail": sim_matrix,
        "semantic_matrix_contra": 1.0 - sim_matrix,
    }


def _make_ue_estimators(metric_names):
    """Build UE estimators: LexSim once, plus EigVal/DegMat/Ecc per similarity metric."""
    estimators = {
        "LexicalSimilarity": LexicalSimilarity(metric="rougeL"),
    }
    for mn in metric_names:
        estimators[f"EigValLaplacian_{mn}"] = EigValLaplacian(
            similarity_score="NLI_score", affinity="entail"
        )
        estimators[f"DegMat_{mn}"] = DegMat(
            similarity_score="NLI_score", affinity="entail"
        )
        estimators[f"Eccentricity_{mn}"] = Eccentricity(
            similarity_score="NLI_score", affinity="entail"
        )
    return estimators


def run_ue_estimators(sentences, sim_matrices, similarity_metrics):
    """Run all UE estimators. Returns {estimator_name: np.ndarray (N,)}."""
    estimators = _make_ue_estimators(similarity_metrics)
    ue_vectors = {}

    first_metric = similarity_metrics[0]
    base_stats = _build_stats(sentences, sim_matrices[first_metric])
    ue_vectors["LexicalSimilarity"] = estimators["LexicalSimilarity"](base_stats)

    for mn in similarity_metrics:
        stats = _build_stats(sentences, sim_matrices[mn])
        ue_vectors[f"EigValLaplacian_{mn}"] = estimators[f"EigValLaplacian_{mn}"](stats)
        ue_vectors[f"DegMat_{mn}"]          = estimators[f"DegMat_{mn}"](stats)
        ue_vectors[f"Eccentricity_{mn}"]    = estimators[f"Eccentricity_{mn}"](stats)

    return ue_vectors


def compute_quality_vectors(sentences, quality_metrics=None, cache_dir=None, parser="", beam_size=0):
    """
    Score the top-1 candidate against the gold graph for each sentence.

    Returns: {quality_metric_name: np.ndarray (N,)}
    """
    if quality_metrics is None:
        quality_metrics = list(QUALITY_METRIC_FNS.keys())

    top1_graphs = [s["candidates"][0]["graph"] for s in sentences]
    gold_graphs  = [s["gold_graph"] for s in sentences]
    qual_vectors = {}

    for qm in tqdm(quality_metrics, desc="quality metrics", leave=False):
        cache_path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{parser}_beam{beam_size}_qual_{qm}_n{len(sentences)}.npz"
        if cache_path is not None and cache_path.exists():
            qual_vectors[qm] = np.load(cache_path)["scores"]
            tqdm.write(f"  [qual {qm}] loaded from cache")
            continue

        score_fn = QUALITY_METRIC_FNS[qm]
        scores_list = []
        for top1, gold in tqdm(zip(top1_graphs, gold_graphs), total=len(top1_graphs), desc=f"  {qm}", leave=False):
            try:
                scores_list.append(float(score_fn(top1, gold)))
            except Exception:
                scores_list.append(0.0)

        scores = np.array(scores_list, dtype=float)
        qual_vectors[qm] = scores
        if cache_path is not None:
            np.savez_compressed(cache_path, scores=scores)

    return qual_vectors


def compute_prr(ue_vectors, qual_vectors):
    """
    Compute normalized PRR per the paper formula:
        PRR = (AUCPR_unc - AUCPR_random) / (AUCPR_oracle - AUCPR_random)

    PRR = 0 means random performance, PRR = 1 means oracle performance,
    and negative values mean worse than random.

    Returns: {ue_method: {quality_metric: prr_score}}
    """
    prr_fn = PredictionRejectionArea()
    results = {}

    oracle_areas = {}
    random_areas = {}
    for qm_name, qual_vec in qual_vectors.items():
        N = len(qual_vec)
        try:
            oracle_areas[qm_name] = float(prr_fn(
                estimator=(-qual_vec).tolist(),
                target=qual_vec.tolist(),
            ))
            random_areas[qm_name] = float(prr_fn(
                estimator=[0.0] * N,
                target=qual_vec.tolist(),
            ))
        except Exception:
            oracle_areas[qm_name] = float("nan")
            random_areas[qm_name] = float("nan")

    for ue_name, ue_vec in ue_vectors.items():
        results[ue_name] = {}
        for qm_name, qual_vec in qual_vectors.items():
            try:
                raw_area = float(prr_fn(
                    estimator=ue_vec.tolist(),
                    target=qual_vec.tolist(),
                ))
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

    return results


def run_pipeline(
    parser,
    beam_size,
    n_samples=None,
    similarity_metrics=None,
    quality_metrics=None,
    data_dir=None,
    cache_dir=None,
    verbose=True,
):
    """
    Run the full UE pipeline for one (parser, beam_size) pair.

    Returns: {ue_method: {quality_metric: prr_score}}
    """
    if similarity_metrics is None:
        similarity_metrics = list(AMR_SIMILARITY_METRICS.keys())
    if quality_metrics is None:
        quality_metrics = list(QUALITY_METRIC_FNS.keys())

    if verbose:
        print(f"\nParser: {parser}  |  Beam: {beam_size}  |  n_samples: {n_samples or 'all'}")

    sentences = load_data(parser, beam_size, n_samples, data_dir)
    N = len(sentences)
    k = sentences[0]["num_candidates"]
    if verbose:
        print(f"Loaded {N} sentences, k={k} candidates each")

    if verbose:
        print(f"Building similarity matrices: {similarity_metrics}")
    sim_matrices = build_sim_matrices(
        sentences, similarity_metrics,
        cache_dir=cache_dir, parser=parser, beam_size=beam_size,
    )

    if verbose:
        print("Running UE estimators...")
    ue_vectors = run_ue_estimators(sentences, sim_matrices, similarity_metrics)

    if verbose:
        print(f"Computing quality scores: {quality_metrics}")
    qual_vectors = compute_quality_vectors(
        sentences, quality_metrics,
        cache_dir=cache_dir, parser=parser, beam_size=beam_size,
    )

    if verbose:
        print("Computing PRR = (AUCPR_unc - random) / (AUCPR_oracle - random)...")
    prr_results = compute_prr(ue_vectors, qual_vectors)

    return prr_results
