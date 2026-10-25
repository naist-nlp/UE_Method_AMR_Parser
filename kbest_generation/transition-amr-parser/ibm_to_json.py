import json
import argparse
from pathlib import Path


def read_amr_file(path):
    text = Path(path).read_text()
    blocks = text.strip().split('\n\n')
    results = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        tok_line = ''
        graph_lines = []
        for line in lines:
            if line.startswith('# ::tok'):
                tok_line = line.replace('# ::tok', '').strip()
            else:
                graph_lines.append(line)
        results.append((tok_line, '\n'.join(graph_lines).strip()))
    return results


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-prefix', required=True)
    parser.add_argument('--beam-size', type=int, required=True)
    parser.add_argument('--sentences', required=True)
    parser.add_argument('--gold', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    prefix = args.results_prefix  # keep as string, don't use Path ops
    out_dir = Path(args.out_dir)
    sent_dir = out_dir / 'sentences'
    sent_dir.mkdir(parents=True, exist_ok=True)

    # rank 1: prefix.amr, rank 2: prefix.1.amr, rank 3: prefix.2.amr ...
    rank_files = [f"{prefix}.amr"]
    for n in range(1, args.beam_size):
        rank_files.append(f"{prefix}.{n}.amr")

    for f in rank_files:
        if not Path(f).exists():
            print(f"ERROR: missing {f}")
            exit(1)

    all_ranks = [read_amr_file(f) for f in rank_files]
    n_sentences = len(all_ranks[0])
    print(f"Found {n_sentences} sentences, {args.beam_size} ranks")

    gold_data = read_gold_amr(args.gold)

    for i in range(n_sentences):
        sent_id    = gold_data[i][0] if i < len(gold_data) else ''
        snt        = gold_data[i][1] if i < len(gold_data) else ''
        gold_graph = gold_data[i][2] if i < len(gold_data) else ''

        candidates = []
        for rank_idx, rank_data in enumerate(all_ranks):
            _, graph_str = rank_data[i]
            candidates.append({'rank': rank_idx + 1, 'graph': graph_str})

        sent_data = {
            'sent_idx': i,
            'sent_id': sent_id,
            'sentence': snt,
            'gold_graph': gold_graph,
            'beam_size': args.beam_size,
            'num_candidates': len(candidates),
            'candidates': candidates
        }
        (sent_dir / f'sent_{i:04d}.json').write_text(json.dumps(sent_data, indent=2))

    run_meta = {
        'parser': 'ibm_transition',
        'results_prefix': prefix,
        'beam_size': args.beam_size,
        'num_sentences': n_sentences,
        'rank_files': rank_files
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))
    print(f"Done. Saved to {sent_dir}/sent_XXXX.json")


if __name__ == '__main__':
    main()
