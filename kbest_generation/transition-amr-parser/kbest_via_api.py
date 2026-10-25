#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

"""
Generate k-best AMR hypotheses using the Python API.
Used for models (e.g. AMR3-joint-ontowiki-seed44) whose fairseq feature
pipeline (oracle vocab, binarized data) is not set up locally.

Loads the model once at beam=10, nbest=10, then saves 4 datasets:
  kbest_beam3, kbest_beam5, kbest_beam8, kbest_beam10
each containing the top-K candidates per sentence.
"""
import argparse
import json
from pathlib import Path


def read_gold_amr(path):
    text = Path(path).read_text()
    blocks = text.strip().split('\n\n')
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        sent_id, snt = '', ''
        graph_lines = []
        for line in block.splitlines():
            if line.startswith('# ::id'):
                sent_id = line.replace('# ::id', '').strip().split()[0]
            elif line.startswith('# ::snt'):
                snt = line.replace('# ::snt', '').strip()
            elif not line.startswith('#'):
                graph_lines.append(line)
        results.append((sent_id, snt, '\n'.join(graph_lines).strip()))
    return results


def save_dataset(all_results, gold_data, beam_size, model_name, out_base):
    out_dir = Path(out_base) / f'kbest_beam{beam_size}'
    sent_dir = out_dir / 'sentences'
    sent_dir.mkdir(parents=True, exist_ok=True)

    for i, (candidates_10, (sent_id, snt, gold_graph)) in enumerate(
        zip(all_results, gold_data)
    ):
        top_k = candidates_10[:beam_size]
        sent_data = {
            'sent_idx': i,
            'sent_id': sent_id,
            'sentence': snt,
            'gold_graph': gold_graph,
            'beam_size': beam_size,
            'num_candidates': len(top_k),
            'candidates': [{'rank': r + 1, 'graph': g} for r, g in enumerate(top_k)],
        }
        (sent_dir / f'sent_{i:04d}.json').write_text(json.dumps(sent_data, indent=2))

    run_meta = {
        'parser': 'ibm_transition_api',
        'model_name': model_name,
        'beam_size': beam_size,
        'num_sentences': len(gold_data),
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))
    print(f'Saved kbest_beam{beam_size} -> {out_dir}  ({len(gold_data)} sentences)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', required=True,
                        help='Pretrained model name, e.g. AMR3-joint-ontowiki-seed44')
    parser.add_argument('--gold', required=True,
                        help='Gold AMR file (AMR3.0 test set)')
    parser.add_argument('--out-dir', required=True,
                        help='Base output directory (kbest_beam* subdirs created here)')
    parser.add_argument('--beams', nargs='+', type=int, default=[3, 5, 8, 10],
                        help='Beam sizes to save (default: 3 5 8 10)')
    args = parser.parse_args()

    max_beam = max(args.beams)

    from transition_amr_parser.parse import AMRParser
    print(f'Loading {args.model_name} with beam={max_beam}, nbest={max_beam}...')
    amr_parser = AMRParser.from_pretrained(
        args.model_name, beam=max_beam, nbest=max_beam
    )

    gold_data = read_gold_amr(args.gold)
    print(f'Loaded {len(gold_data)} sentences from {args.gold}')

    # Parse all sentences once, collecting up to max_beam candidates each
    all_results = []  # list of lists: all_results[i] = [graph_str_rank1, ...]
    batch = []
    index_map = []

    for i, (sent_id, snt, _) in enumerate(gold_data):
        tokens, _ = amr_parser.tokenize(snt)
        batch.append(tokens)
        index_map.append(i)

    print(f'Parsing {len(batch)} sentences...')
    annotations_batch, _ = amr_parser.parse_sentences(batch)

    for ann in annotations_batch:
        if max_beam == 1:
            all_results.append([ann])
        else:
            # ann is a generator of candidate strings when nbest > 1
            all_results.append(list(ann))

    # Save one dataset per requested beam size
    for beam_size in args.beams:
        save_dataset(all_results, gold_data, beam_size, args.model_name, args.out_dir)

    print('All done.')


if __name__ == '__main__':
    main()
