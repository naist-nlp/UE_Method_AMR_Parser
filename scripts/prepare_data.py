#!/usr/bin/env python3
"""
prepare_data.py — prepare your own copy of AMR 3.0 (LDC2020T02) for the
k-best export scripts and check the result.

AMR 3.0 is licensed by the LDC and is not distributed with this repository.
Put your copy under data/AMR3.0/ (see data/README.md), then:

    # 1. gold test graphs and test sentences, the inputs of kbest_generation/
    python scripts/prepare_data.py gold

    # 2. after generating k-best outputs (kbest_generation/README.md):
    #    check every parser/beam directory
    python scripts/prepare_data.py check

    # 3. if a parser's files are in a different sentence order than the
    #    paper's runs, write a copy in that order
    python scripts/prepare_data.py reorder --parser SPRING --out_dir data/k_top_AMR_parsers_reordered

Sentence order matters for exact PRR values: see README, "How PRR is computed".
data/sentence_order.json holds that order as hashes of the AMR ids (no ids or
text from the corpus).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from estimators.pipeline import BEAM_SIZES, PARSERS, _get_sent_dir

AMR3_DIR = _ROOT / "data" / "AMR3.0"
KBEST_DIR = _ROOT / "data" / "k_top_AMR_parsers"
ORDER_FILE = _ROOT / "data" / "sentence_order.json"
N_TEST = 1898


def amr_id(sentence: dict) -> str:
    """AMR id of a k-best sentence file (kbest_spring.py stores the whole '# ::id' line)."""
    return (sentence.get("sent_id") or "").split(" ")[0]


def id_hash(sent_id: str) -> str:
    return hashlib.sha256(sent_id.encode("utf-8")).hexdigest()[:16]


def sentence_ids(sents: list, test_ids=None) -> list:
    """
    AMR ids of a parser's sentence files. kbest_spring.py leaves the id of the
    last test sentence empty; with the test ids known, a single empty id is the
    single test id that no other file has.
    """
    ids = [amr_id(s) for s in sents]
    if test_ids is not None and ids.count("") == 1:
        absent = set(test_ids) - set(ids)
        if len(absent) == 1:
            ids[ids.index("")] = absent.pop()
    return ids


# In the paper's runs the last SPRING file (nw.chtb_0325.10) carries the gold
# graph of nw.chtb_0325.2; its scores in the released score files are those of
# the runs. check reports it as known instead of as an error.
KNOWN_GOLD_MISMATCH = {"SPRING": {"nw.chtb_0325.10"}}


def find_test_files(amr3_dir: Path) -> list:
    """The six AMR 3.0 test files (bolt, consensus, dfa, lorelei, proxy, xinhua), sorted."""
    files = sorted(amr3_dir.glob("**/data/amrs/split/test/amr-release-3.0-amrs-test-*.txt"))
    if len(files) != 6:
        raise SystemExit(f"expected 6 AMR 3.0 test files under {amr3_dir}/**/data/amrs/split/test/, "
                         f"found {len(files)}")
    return files


def read_amr_blocks(path: Path) -> list:
    """[(id, sentence, graph)] for every AMR in a file with # ::id / # ::snt headers."""
    out = []
    for block in path.read_text().strip().split("\n\n"):
        sent_id, snt, graph = "", "", []
        for line in block.splitlines():
            if line.startswith("# ::id"):
                sent_id = line.split("# ::id", 1)[1].strip().split()[0]
            elif line.startswith("# ::snt"):
                snt = line.split("# ::snt", 1)[1].strip()
            elif not line.startswith("#"):
                graph.append(line)
        if sent_id:
            out.append((sent_id, snt, "\n".join(graph).strip()))
    return out


def cmd_gold(args):
    """Write gold.amr.txt (the six test files joined, as used for the paper) and test_sentences.txt."""
    files = find_test_files(args.amr3_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold = args.out_dir / "gold.amr.txt"
    gold.write_text("\n\n".join(f.read_text() for f in files))
    blocks = read_amr_blocks(gold)
    (args.out_dir / "test_sentences.txt").write_text("".join(snt + "\n" for _, snt, _ in blocks))
    print(f"{len(blocks)} test AMRs (expected {N_TEST}) from {len(files)} files")
    print(f"written: {gold}\n         {args.out_dir / 'test_sentences.txt'}")


def _graph(text: str) -> str:
    """A PENMAN graph without comment lines and with whitespace collapsed, for comparison."""
    return " ".join(" ".join(l for l in text.splitlines() if not l.strip().startswith("#")).split())


def load_sentences(data_dir: Path, parser: str, beam: int) -> list:
    return [json.loads(f.read_text()) for f in sorted(_get_sent_dir(data_dir, parser, beam).glob("sent_*.json"))]


def cmd_check(args):
    """Check counts, fields, gold graphs and sentence order of every parser/beam directory."""
    order = json.loads(ORDER_FILE.read_text())
    gold_file = args.data_dir / "gold.amr.txt"
    gold = {i: g for i, _, g in read_amr_blocks(gold_file)} if gold_file.exists() else None
    if gold is None:
        print(f"note: {gold_file} not found, gold graphs not compared (run: prepare_data.py gold)")
    all_ok = True
    for parser in args.parsers:
        for beam in args.beams:
            try:
                sents = load_sentences(args.data_dir, parser, beam)
            except FileNotFoundError:
                print(f"{parser:28s} k={beam:<2d} MISSING")
                all_ok = False
                continue
            problems = []
            if len(sents) != N_TEST:
                problems.append(f"{len(sents)} sentences, expected {N_TEST}")
            bad_k = sum(len(s.get("candidates", [])) != beam for s in sents)
            if bad_k:
                problems.append(f"{bad_k} sentences without {beam} candidates")
            no_gold = sum(not s.get("gold_graph", "").strip() for s in sents)
            if no_gold:
                problems.append(f"{no_gold} empty gold graphs")
            ids = sentence_ids(sents, gold)
            notes = []
            if gold is not None:
                wrong = {i for i, s in zip(ids, sents) if _graph(gold.get(i, "")) != _graph(s.get("gold_graph", ""))}
                known = wrong & KNOWN_GOLD_MISMATCH.get(parser, set())
                if known:
                    notes.append(f"gold graph of {', '.join(sorted(known))} as in the paper's runs (README)")
                if wrong - known:
                    problems.append(f"{len(wrong - known)} gold graphs differ from gold.amr.txt "
                                    "(kbest_generation/fix_gold_alignment.py)")
            if [id_hash(i) for i in ids] != order.get(parser):
                problems.append("sentence order differs from the paper's runs (prepare_data.py reorder)")
            all_ok &= not problems
            print(f"{parser:28s} k={beam:<2d} " + ("OK" if not problems else "; ".join(problems))
                  + (f"  [{'; '.join(notes)}]" if notes else ""))
    print("\nall checks passed" if all_ok else "\nsome checks failed")


def cmd_reorder(args):
    """Copy one parser's sentence files, renumbered into the sentence order of the paper's runs."""
    rank = {h: i for i, h in enumerate(json.loads(ORDER_FILE.read_text())[args.parser])}
    gold_file = args.data_dir / "gold.amr.txt"
    test_ids = [i for i, _, _ in read_amr_blocks(gold_file)] if gold_file.exists() else None
    for beam in args.beams:
        sents = load_sentences(args.data_dir, args.parser, beam)
        ids = sentence_ids(sents, test_ids)
        missing = [i for i in ids if id_hash(i) not in rank]
        if missing or len(sents) != len(rank):
            raise SystemExit(f"{args.parser} k={beam}: {len(sents)} sentences, {len(missing)} ids "
                             "not in the paper's test set (is gold.amr.txt in --data_dir?)")
        out = args.out_dir / args.parser / f"kbest_beam{beam}" / "sentences"
        out.mkdir(parents=True, exist_ok=True)
        for i, s in sorted(zip(ids, sents), key=lambda t: rank[id_hash(t[0])]):
            s["sent_idx"] = rank[id_hash(i)]
            (out / f"sent_{s['sent_idx']:04d}.json").write_text(json.dumps(s, indent=2))
        print(f"{args.parser} k={beam}: {len(sents)} files -> {out}")


def cmd_export_order(args):
    """Authors only: write data/sentence_order.json from the k-best data of the paper's runs."""
    test_ids = [i for i, _, _ in read_amr_blocks(args.data_dir / "gold.amr.txt")]
    order = {}
    for parser in PARSERS:
        per_beam = [[id_hash(i) for i in sentence_ids(load_sentences(args.data_dir, parser, b), test_ids)]
                    for b in BEAM_SIZES]
        if any(o != per_beam[0] for o in per_beam):
            raise SystemExit(f"{parser}: sentence order differs between beam sizes")
        order[parser] = per_beam[0]
    ORDER_FILE.write_text(json.dumps(order, indent=1) + "\n")
    print(f"written {ORDER_FILE} ({len(order)} parsers x {len(order[PARSERS[0]])} sentences)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gold", help="gold test graphs and sentences from AMR 3.0")
    g.add_argument("--amr3_dir", type=Path, default=AMR3_DIR)
    g.add_argument("--out_dir", type=Path, default=KBEST_DIR)

    for name, helptext in [("check", "check the k-best data"), ("reorder", "copy one parser in the paper's order"),
                           ("export_order", "authors only: write data/sentence_order.json")]:
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--data_dir", type=Path, default=KBEST_DIR)
        if name != "export_order":
            p.add_argument("--beams", nargs="+", type=int, default=BEAM_SIZES)
        if name == "check":
            p.add_argument("--parsers", nargs="+", default=PARSERS)
        if name == "reorder":
            p.add_argument("--parser", required=True, choices=PARSERS)
            p.add_argument("--out_dir", type=Path, required=True)

    args = ap.parse_args()
    {"gold": cmd_gold, "check": cmd_check, "reorder": cmd_reorder, "export_order": cmd_export_order}[args.cmd](args)


if __name__ == "__main__":
    main()
