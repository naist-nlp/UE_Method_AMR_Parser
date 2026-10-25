# Expected results

PRR tables of the runs reported in the paper: one CSV per parser and beam
size, 17 rows (13 UE methods of the paper, plus NumSemSets and the three
estimators on Smatch++ similarity, which were computed but not reported) x 3
quality metrics (S++ = Smatch++, S2 = S²match, SB = SemBLEU).

Table 4 and Figures 1–11 are computed from these values.
`scripts/check_against_paper.py` compares the output of `run_all.py` with them.

`top1_smatch_paper_run.log`: log of the run that gave the Smatch column of
Tables 1 and 3 (mean per-sentence Smatch of the top-1 candidate, beam 3).
