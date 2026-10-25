# k-best generation

Only needed to rebuild the score files from scratch (requires AMR 3.0,
LDC2020T02). These are the scripts we added to each parser's official
repository to write k-best candidates in the format of `data/README.md`. Copy
each into the listed place of the parser's repository and run it with that
repository's environment and the checkpoint listed below.

| Parser (paper) | Repository | Checkpoint | Script → place in repo |
|---|---|---|---|
| SPRING | [SapienzaNLP/spring](https://github.com/SapienzaNLP/spring) | `AMR3.parsing.pt` (bart-large, PENMAN linearisation, pointer tokens, no recategorisation) | `spring/kbest_spring.py` → `scripts/` |
| IBM-Parser | [IBM/transition-amr-parser](https://github.com/IBM/transition-amr-parser) | `amr3.0-structured-bart-large`, seed 42, `checkpoint_wiki.smatch_top5-avg.pt` | k-best files from the repository's decoding script (`run/parse.sh`), converted with `transition-amr-parser/ibm_to_json.py` → `scripts/` |
| MBSE | IBM/transition-amr-parser | `amr3joint_ontowiki2_g2g-structured-bart-large-seed44` (API name `AMR3-joint-ontowiki-seed44`) | `transition-amr-parser/kbest_via_api.py` → `scripts/` |
| IBM-Neural | IBM/transition-amr-parser | `amr3.0-structured-bart-large-neur-al-seed42` (API name `AMR3-structbart-L`) | `transition-amr-parser/kbest_via_api.py` → `scripts/` |
| HMG-AMR-T5 | [YIKMAT/HMG-AMR](https://github.com/YIKMAT/HMG-AMR/tree/main/HMG-AMR(T5)) | `output/t5_amr3/multitask.model` (t5-large) | `HMG-AMR-T5/parser_kbest.py` → `amr_parsing/t5/cli/` |
| HMG-AMR-BART | [YIKMAT/HMG-AMR](https://github.com/YIKMAT/HMG-AMR/tree/main/HMG-AMR(Bart)) | `best-smatch_checkpoint_23_0.8404.pt` (bart-large) | `HMG-AMR-BART/predict_AMR_PRP.py` → `bin/` |

Each script takes `--help`. `kbest_spring.py` and `kbest_via_api.py` decode
once with the largest beam and write the first k candidates for each k.

`fix_gold_alignment.py` (run from this repository) re-attaches the gold graph
of every `sent_*.json` from `data/k_top_AMR_parsers/gold.amr.txt` by AMR id,
for SPRING, IBM-Parser and the two HMG-AMR parsers.

Beam search on GPU is not bit-reproducible, so regenerated candidates can
differ slightly from the ones behind the released score files.
