# Data

The k-best AMR parser outputs are too large to include in this repository.

Download the data from Google Drive (link provided in the paper) and place it here so the directory looks like:

    data/
        k_top_AMR_parsers/
            SPRING/
                kbest_beam3/sentences/sent_0000.json ...
                kbest_beam5/sentences/
                kbest_beam8/sentences/
                kbest_beam10/sentences/
            IBM_parser/
                kbest_beam3/sentences/
                ...
            HMG-AMR-T5/
            HMG-AMR-BART/
            amr3joint_ontowiki2_seed44/
            ibm_amr3.0_neur_al/
            gold.amr.txt

Each sentence file (sent_XXXX.json) contains:
- sentence: the input sentence text
- gold_graph: the reference AMR graph string
- candidates: list of k AMR candidates, each with a "graph" field
- num_candidates: beam size k
