# Data

The AMR data is **not needed to reproduce the paper**: the score files in
`estimators/results/cache/` contain every number the results are computed
from. This folder is for recomputing those scores from scratch, which
requires your own licensed copy of AMR 3.0 (LDC2020T02). Neither the corpus
nor the parser outputs on it (which contain its sentences) are distributed
with this repository.

## Layout

    data/
        AMR3.0/                      your copy of the LDC release (see AMR3.0/README.md)
        sentence_order.json          sentence order of the paper's runs (hashed AMR ids)
        k_top_AMR_parsers/           created by the steps below
            gold.amr.txt
            test_sentences.txt
            SPRING/
                kbest_beam3/sentences/sent_0000.json ... sent_1897.json
                kbest_beam5/sentences/
                kbest_beam8/sentences/
                kbest_beam10/sentences/
            IBM_parser/                  IBM-Parser
            HMG-AMR-T5/
            HMG-AMR-BART/
            amr3joint_ontowiki2_seed44/  MBSE
            ibm_amr3.0_neur_al/          IBM-Neural

## Steps

1. **Gold graphs and sentences.** Place AMR 3.0 under `data/AMR3.0/`, then

       python scripts/prepare_data.py gold

   This writes `k_top_AMR_parsers/gold.amr.txt` (the six test files joined,
   1,898 AMRs, identical to the file used for the paper) and
   `k_top_AMR_parsers/test_sentences.txt`.

2. **k-best candidates.** Run each parser's export script with its official
   code and checkpoint, see `kbest_generation/README.md`. The scripts write
   one JSON file per test sentence:

       {
         "sent_idx": 0,
         "sent_id": "<AMR 3.0 id>",
         "sentence": "<sentence>",
         "gold_graph": "<gold AMR, PENMAN>",
         "beam_size": 3,
         "num_candidates": 3,
         "candidates": [{"rank": 1, "graph": "<PENMAN>"}, ...]
       }

   `candidates[0]` is the top-1 parse. Then attach the gold graphs by AMR id:

       python kbest_generation/fix_gold_alignment.py

3. **Check.**

       python scripts/prepare_data.py check

   This checks, for every parser and beam size, the number of sentences and
   candidates, the gold graphs against `gold.amr.txt`, and the sentence order
   against the paper's runs.

4. **Sentence order.** The files must be numbered in the sentence order of
   the paper's runs, which differs between parsers (each followed the order in
   which its own k-best script read the test set). If `check` reports a
   different order, write a reordered copy and use that one:

       python scripts/prepare_data.py reorder --parser SPRING --out_dir data/k_top_AMR_parsers_reordered

5. **Recompute the scores.**

       python run_all.py --n_samples 0 --data_dir data/k_top_AMR_parsers --cache_dir my_cache --tie_order numpy --recompute_ue

`sentence_order.json` stores the order as the first 16 hex digits of the
SHA-256 of each AMR id, so it reveals neither ids nor text.

In the paper's runs the last SPRING sentence (`nw.chtb_0325.10`) carries the
gold graph of `nw.chtb_0325.2` (its k-best file has no AMR id, so the gold
alignment could not correct it). The released score files keep the values of
the runs; `check` reports this case as known. With the correct gold graph,
SPRING's PRR values would change by at most 0.002 (Figure 1 unchanged), and
its Smatch and Smatch++ in Table 1 by +0.0005.
