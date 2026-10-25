#!/usr/bin/env python3
"""
Generate k-best AMR hypotheses using the SPRING model.
Loads the model once at beam=max_beam, then saves subsets for each requested beam size.
Output format matches the IBM transition-parser kbest JSON format.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from spring_amr.evaluation import predict_amrs
from spring_amr.penman import encode
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer


def parse_gold_metadata(gold_path):
    """Extract ::id and ::snt from each AMR block header."""
    metadata = []
    current = {}
    for line in Path(gold_path).read_text().splitlines():
        if line.startswith('# ::id'):
            current['id'] = line.split('# ::id')[-1].strip()
        elif line.startswith('# ::snt'):
            current['snt'] = line.split('# ::snt')[-1].strip()
        elif line.startswith('(') and current:
            metadata.append(current)
            current = {}
    if current:
        metadata.append(current)
    return metadata


def parse_gold_penman(gold_path):
    """Return list of penman strings (no comment lines) one per sentence."""
    blocks = Path(gold_path).read_text().strip().split('\n\n')
    result = []
    for block in blocks:
        lines = [l for l in block.splitlines() if not l.startswith('#')]
        result.append('\n'.join(lines).strip())
    return result


def save_beam_dataset(all_graphs, gold_metadata, gold_penman, beam_size, out_base, args_info):
    """Save top-K candidates (slice of all_graphs) to out_base/kbest_beam{K}/."""
    out_dir = Path(out_base) / f'kbest_beam{beam_size}'
    out_dir.mkdir(parents=True, exist_ok=True)
    sent_dir = out_dir / 'sentences'
    sent_dir.mkdir(exist_ok=True)

    run_meta = {
        **args_info,
        'beam_size': beam_size,
        'num_sentences': len(all_graphs),
        'timestamp': datetime.now().isoformat(),
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))

    for i, candidates in enumerate(all_graphs):
        top_k = candidates[:beam_size]
        meta = gold_metadata[i] if i < len(gold_metadata) else {}
        sent_data = {
            'sent_idx': i,
            'sent_id': meta.get('id', ''),
            'sentence': meta.get('snt', ''),
            'gold_graph': gold_penman[i] if i < len(gold_penman) else '',
            'beam_size': beam_size,
            'num_candidates': len(top_k),
            'candidates': [
                {'rank': rank, 'graph': encode(g)}
                for rank, g in enumerate(top_k, start=1)
            ],
        }
        (sent_dir / f'sent_{i:04d}.json').write_text(json.dumps(sent_data, indent=2))

    # top-1 flat AMR file for quick smatch
    top1_pieces = [encode(candidates[0]) for candidates in all_graphs]
    top1_path = out_dir / 'pred_top1.txt'
    top1_path.write_text('\n\n'.join(top1_pieces))

    print(f'  Saved beam={beam_size}: {len(all_graphs)} sentences -> {out_dir}')
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description='Generate k-best AMR hypotheses with SPRING, save one dataset per beam size.')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SPRING .pt checkpoint')
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
                        help='Glob pattern(s) for AMR test files')
    parser.add_argument('--gold-path', type=Path, required=True,
                        help='Where to write the concatenated gold AMR file')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='Output base directory (kbest_beam{K}/ subdirs created inside)')
    parser.add_argument('--beams', type=int, nargs='+', default=[3, 5, 8, 10],
                        help='Beam sizes to generate (model runs at max, subsets saved)')
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    max_beam = max(args.beams)
    device = torch.device(args.device)

    print(f'Loading model from {args.checkpoint} ...')
    model, tokenizer = instantiate_model_and_tokenizer(
        'facebook/bart-large',
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=True,
        use_pointer_tokens=True,
        raw_graph=False,
    )
    model.amr_mode = True
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)
    print(f'Model loaded. Running beam={max_beam} ...')

    gold_path = args.gold_path
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True,
        out=gold_path,
        use_recategorization=False,
    )
    loader.device = device

    # Run once at max beam — returns List[List[Graph]]
    all_graphs = predict_amrs(
        loader, model, tokenizer,
        beam_size=max_beam,
        restore_name_ops=False,
        return_all=True,
    )
    print(f'Generated {len(all_graphs)} sentences x {max_beam} candidates.')

    gold_metadata = parse_gold_metadata(gold_path)
    gold_penman = parse_gold_penman(gold_path)

    args_info = {
        'checkpoint': str(args.checkpoint),
        'datasets': args.datasets,
        'max_beam': max_beam,
        'batch_size': args.batch_size,
        'model': 'facebook/bart-large',
        'penman_linearization': True,
        'use_pointer_tokens': True,
    }

    print(f'Saving datasets to {args.out_dir} ...')
    for beam_size in sorted(args.beams):
        save_beam_dataset(all_graphs, gold_metadata, gold_penman, beam_size, args.out_dir, args_info)

    print('Done.')


if __name__ == '__main__':
    main()
