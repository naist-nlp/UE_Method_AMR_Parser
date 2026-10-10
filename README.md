# How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing

Code for the paper **"How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing"**.

This repository provides the full pipeline to reproduce the PRR (Prediction Rejection Ratio) scores reported in the paper for six AMR parsers across four beam sizes (k = 3, 5, 8, 10).


## What This Code Does

Given the k-best candidate outputs from an AMR parser, we evaluate how well different uncertainty estimation (UE) methods can identify which predictions are likely wrong. We measure this using the Prediction Rejection Ratio (PRR):

    PRR = (AUCPR_unc - AUCPR_random) / (AUCPR_oracle - AUCPR_random)

A PRR of 1 means the UE method perfectly ranks sentences by quality. A PRR of 0 means it does no better than random. Negative values mean it is worse than random.

We test 13 UE methods across three AMR quality metrics (Smatch++, S2match, SemBLEU) and four beam sizes.


## Repository Structure

    run_all.py               - Main entry point: runs the full UE pipeline
    build_results_json.py    - Merges per-parser CSVs into a single results.json

    estimators/
        pipeline.py          - Core pipeline (load data, compute similarity, run UE, compute PRR)
        metrics/
            smatch.py        - Smatch similarity (subprocess batch)
            smatchpp.py      - Smatch++ similarity (in-process)
            s2match.py       - S2match similarity (subprocess batch, uses GloVe)
            sema.py          - SEMA similarity (in-process)
            wlk.py           - Weisfeiler-Leman kernel similarity (subprocess batch)

    similarities/
        sembleu.py           - SemBLEU quality metric wrapper
        smatchpp/            - Smatch++ library (third-party)
        sema/                - SEMA library (third-party)
        amr-metric-suite/    - Smatch and S2match scripts (third-party)
        weisfeiler-leman-amr-metrics/  - WLK metric (third-party)
        sembleu/             - SemBLEU evaluator (third-party)

    paper_plots/
        shared.py            - Constants and helpers shared by all figure scripts
        p01_summary_allbeams.py  - Figure 1: mean PRR per parser
        p02_prr_vs_beam_avg.py   - Figure 2: PRR vs beam (all methods averaged)
        p03_radar.py             - Figure 3: radar chart beam 10 (EigVal, DegMat, Ecc)
        p04_radar_lexsim.py      - Figure 4: radar chart beam 10 (all four families)
        p05_prr_vs_smatch.py     - Figure 5: scatter PRR vs Smatch accuracy
        p06_ue_stability.py      - Figure 6: UE stability heatmap
        p07_ue_corr_heatmap.py   - Figure 7: pairwise UE correlation heatmap
        p08_prr_vs_beam_grid.py  - Figure 8: 2x3 grid PRR vs beam per parser
        p09_prr_vs_beam_quality.py - Figure 9: 1x6 PRR vs beam per parser
        run_all.py               - Runs all figure scripts

    data/
        README.md            - Instructions for downloading the parser output data


## Reproducing Results Without the Raw Data

The pre-computed similarity matrices and quality scores are included in this repository under `estimators/results/cache/`. These are compressed numpy arrays containing only floating-point numbers — no text, no sentences, nothing from the original dataset.

This means you can reproduce all PRR scores directly without downloading any parser output data:

    conda activate polygraph
    python run_all.py --n_samples 0

The pipeline will detect the cached files and skip recomputation, running in seconds.

To generate the paper figures after that:

    python build_results_json.py
    python paper_plots/run_all.py


## Setup (only needed to run on new data)

### Step 1: Download the data

The k-best parser outputs are hosted on Google Drive due to their size. Download and unpack them into the data/ directory following the instructions in data/README.md.

### Step 2: Install dependencies

The pipeline requires two conda environments.

The main polygraph environment (used for run_all.py):

    conda create -n polygraph python=3.10
    conda activate polygraph
    pip install lm-polygraph tqdm numpy scipy

The GloVe vectors for S2match must be placed at:

    similarities/amr-metric-suite/vectors/glove.6B.100d.txt

Download from: https://nlp.stanford.edu/projects/glove/ (glove.6B.zip, use the 100d file)

### Step 3: Verify third-party tools

The similarity metrics call external scripts via subprocess. Make sure these directories are intact:

    similarities/amr-metric-suite/py3-Smatch-and-S2match/smatch/smatch.py
    similarities/amr-metric-suite/py3-Smatch-and-S2match/smatch/s2match.py
    similarities/weisfeiler-leman-amr-metrics/src/main_wlk.py
    similarities/sembleu/src/eval.py
    similarities/smatchpp/   (importable Python package)
    similarities/sema/       (importable Python package)


## Running the Pipeline

All commands should be run from the project root directory.

Quick test with 10 sentences to verify the setup works:

    conda activate polygraph
    python run_all.py --n_samples 10

Full run over all 1898 sentences for all parsers and beams:

    python run_all.py --n_samples 0

Run a specific parser and beam:

    python run_all.py --parsers SPRING --beams 10 --n_samples 0

Results are saved as:

    estimators/results/results_PARSER_beamN.csv   - per-parser, per-beam PRR table
    estimators/results/results.json               - all results combined

The pipeline resumes automatically if interrupted. Similarity matrices are cached in estimators/results/cache/ so already-computed work is not repeated.

To disable caching:

    python run_all.py --no_cache


## Generating Paper Figures

After the pipeline finishes, merge the CSVs and run the figure scripts:

    python build_results_json.py
    python paper_plots/run_all.py

Figures are saved as PDF and PNG in paper_figures/.


## Parsers

The six parsers evaluated in the paper are:

    SPRING                     - Bevilacqua et al. 2021
    IBM_parser                 - IBM Transition-based parser
    HMG-AMR-T5                 - Huang et al. T5-based
    HMG-AMR-BART               - Huang et al. BART-based
    amr3joint_ontowiki2_seed44 - referred to as MBSE in the paper
    ibm_amr3.0_neur_al         - referred to as IBM-Neural in the paper


## UE Methods

The 13 UE methods evaluated:

    LexicalSimilarity  - lexical similarity (ROUGE-L) across k candidates
    EigValLaplacian_*  - eigenvalue of the graph Laplacian (one per AMR metric)
    DegMat_*           - degree matrix estimator (one per AMR metric)
    Eccentricity_*     - eccentricity estimator (one per AMR metric)

where * is one of: smatch, smatchpp, s2match, sema, wlk


## Quality Metrics

    S++  - Smatch++ (F1 score)
    S2   - S2match (semantic similarity with GloVe vectors)
    SB   - SemBLEU (BLEU-inspired metric for AMR graphs)


## Citation

If you use this code, please cite:

    @inproceedings{kumar2026reliable,
    title= {How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing}, 
    author= {Kumar, Anuj and Ozaki, Shintaro and Prasad, 
    Yamuna and Hayashi, Katsuhiko and Kamigaito, Hidetaka and Watanabe, Taro}, 
    booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing}, 
    year= {2026} }
