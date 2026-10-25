# AMR 3.0 (LDC2020T02)

Put your own copy of the AMR 3.0 release here. It is licensed by the
Linguistic Data Consortium (https://catalog.ldc.upenn.edu/LDC2020T02) and is
not part of this repository; git ignores everything in this folder except
this file.

After extracting `amr_annotation_3.0_LDC2020T02.tgz`, the test split must be
at

    data/AMR3.0/amr_annotation_3.0/data/amrs/split/test/
        amr-release-3.0-amrs-test-bolt.txt
        amr-release-3.0-amrs-test-consensus.txt
        amr-release-3.0-amrs-test-dfa.txt
        amr-release-3.0-amrs-test-lorelei.txt
        amr-release-3.0-amrs-test-proxy.txt
        amr-release-3.0-amrs-test-xinhua.txt

Then run `python scripts/prepare_data.py gold` (see `data/README.md`).
