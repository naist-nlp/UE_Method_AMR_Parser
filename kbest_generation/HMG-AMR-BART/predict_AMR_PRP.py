from pathlib import Path
from datetime import datetime
import json

import penman
import torch

from spring_amr import ROOT
from spring_amr.evaluation import predict_amrs, compute_smatch
from spring_amr.penman import encode
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer
from spring_amr.parse_tree import ParseTreeNode
import yaml
import time

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, nargs='+', default=[1, 3, 5, 8, 10],
        help="One or more beam sizes. Each runs separately and saves to its own folder.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold.txt',
        help="Where to write the gold file.")
    parser.add_argument('--output-dir', type=Path, default=None,
        help="Root output dir. Each beam saved under output-dir/kbest_beam{N}/")
    parser.add_argument('--use-recategorization', action='store_true',
        help="Predict using Zhang recategorization on top of our linearization.")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sweeped.yaml',
        help='Use the following config for hparams.')
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--return-all', action='store_true')

    args = parser.parse_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    # Gold path — written once, reused for all beams
    gold_path = args.gold_path
    gold_path.parent.mkdir(parents=True, exist_ok=True)

    # Root output dir for all beams
    root_out_dir = args.output_dir if args.output_dir is not None else args.pred_path.parent
    root_out_dir.mkdir(parents=True, exist_ok=True)

    # Summary log across all beams
    summary_path = root_out_dir / 'smatch_summary.txt'

    # ------------------------------------------------------------------ #
    # Helpers — identical to Spring
    # ------------------------------------------------------------------ #
    def parse_gold_metadata(gold_path):
        metadata = []
        current = {}
        for line in Path(gold_path).read_text().splitlines():
            if line.startswith('# ::id'):
                current['id'] = line.split('# ::id')[-1].strip()
            elif line.startswith('# ::snt'):
                current['snt'] = line.split('# ::snt')[-1].strip()
            elif line.startswith('('):
                graph_lines = [line]
            elif line == '' and current:
                metadata.append(current)
                current = {}
        if current:
            metadata.append(current)
        return metadata

    def parse_gold_penman(gold_path):
        gold_blocks = gold_path.read_text().strip().split('\n\n')
        gold_penman = []
        for block in gold_blocks:
            lines = [l for l in block.splitlines() if not l.startswith('#')]
            gold_penman.append('\n'.join(lines).strip())
        return gold_penman

    # ------------------------------------------------------------------ #
    # Load model ONCE — reused for all beam sizes
    # ------------------------------------------------------------------ #
    device = torch.device(args.device)

    print("Loading model and tokenizer...")
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
        head_list=config['head_list'],
        mg_layer_begin_index=config['mg_layer_begin_index'],
        mg_layer_end_index=config['mg_layer_end_index'],
        mg_list=config['mg_list'],
        hie_layer_begin_index=config['hie_layer_begin_index'],
        hie_layer_end_index=config['hie_layer_end_index'],
        hie_mode=config['hie_mode'],
        lstm_state=config['lstm_state']
    )
    model.amr_mode = True
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'], strict=False)
    model.to(device)
    print("Model loaded.\n")

    # ------------------------------------------------------------------ #
    # Load data ONCE — gold file written here, reused for all beams
    # ------------------------------------------------------------------ #
    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out=gold_path,
        use_recategorization=args.use_recategorization,
        p_mg=config['p_mg'],
        constituent_file=config['test_constituent'],
        sentence_file=config['test_sentence'],
        clause_file=config['test_clause']
    )
    loader.device = device

    # Parse gold once (after loader writes it)
    # We'll re-parse after first beam if needed — gold doesn't change
    gold_metadata = None
    gold_penman   = None

    # ------------------------------------------------------------------ #
    # Run each beam size — same structure as Spring, one folder per beam
    # ------------------------------------------------------------------ #
    all_smatch = {}

    for beam_size in args.beam_size:

        print(f"\n{'='*60}")
        print(f"Running beam_size={beam_size}")
        print(f"{'='*60}")

        pred_path = args.pred_path  # used only if not return_all

        now_time = time.process_time()
        print("Begin:" + str(now_time))

        graphs = predict_amrs(
            loader,
            model,
            tokenizer,
            beam_size=beam_size,
            restore_name_ops=args.restore_name_ops,
            return_all=args.return_all,
        )

        end_time = time.process_time()
        print("End:" + str(end_time))
        print("Total_Time:" + str(end_time - now_time))

        # Parse gold metadata after first beam (gold file now exists)
        if gold_metadata is None:
            gold_metadata = parse_gold_metadata(gold_path)
            gold_penman   = parse_gold_penman(gold_path)

        if args.return_all:
            # ---- k-best mode: mirrors Spring exactly ----
            out_dir = root_out_dir / f"kbest_beam{beam_size}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # 1. run_meta.json
            run_meta = {
                'checkpoint'          : str(args.checkpoint),
                'beam_size'           : beam_size,
                'batch_size'          : args.batch_size,
                'model'               : args.model,
                'penman_linearization': args.penman_linearization,
                'use_pointer_tokens'  : args.use_pointer_tokens,
                'use_recategorization': args.use_recategorization,
                'restore_name_ops'    : args.restore_name_ops,
                'datasets'            : args.datasets,
                'num_sentences'       : len(graphs),
                'timestamp'           : datetime.now().isoformat(),
            }
            (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))

            # 2. per-sentence JSON (sentences/sent_XXXX.json)
            sent_dir = out_dir / 'sentences'
            sent_dir.mkdir(exist_ok=True)

            for i, candidates in enumerate(graphs):
                meta = gold_metadata[i] if i < len(gold_metadata) else {}
                sent_data = {
                    'sent_idx'      : i,
                    'sent_id'       : meta.get('id', ''),
                    'sentence'      : meta.get('snt', ''),
                    'gold_graph'    : gold_penman[i] if i < len(gold_penman) else '',
                    'beam_size'     : beam_size,
                    'num_candidates': len(candidates),
                    'candidates'    : [
                        {'rank': rank, 'graph': encode(g)}
                        for rank, g in enumerate(candidates, start=1)
                    ]
                }
                (sent_dir / f'sent_{i:04d}.json').write_text(json.dumps(sent_data, indent=2))

            # 3. flat txt (human readable — Spring format)
            flat_lines = []
            for i, candidates in enumerate(graphs):
                meta = gold_metadata[i] if i < len(gold_metadata) else {}
                flat_lines.append(f"# ::sent_idx {i}")
                flat_lines.append(f"# ::sent_id {meta.get('id', '')}")
                flat_lines.append(f"# ::snt {meta.get('snt', '')}")
                flat_lines.append(f"# ::beam_size {beam_size}")
                flat_lines.append(f"# ::gold\n{gold_penman[i] if i < len(gold_penman) else ''}")
                flat_lines.append(f"# ::num_candidates {len(candidates)}")
                for rank, g in enumerate(candidates, start=1):
                    flat_lines.append(f"# ::rank {rank}\n{encode(g)}")
                flat_lines.append("")
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text('\n'.join(flat_lines))

            # 4. top-1 AMR file for smatch
            top1_graphs = [candidates[0] for candidates in graphs]
            top1_pieces = [encode(g) for g in top1_graphs]
            top1_path   = out_dir / 'pred_top1.txt'
            top1_path.write_text('\n\n'.join(top1_pieces))

            # compute smatch on top-1
            score = compute_smatch(gold_path, top1_path)
            print(f'Smatch (top-1, beam={beam_size}): {score:.3f}')
            all_smatch[beam_size] = score

            run_meta['smatch_top1'] = round(score, 4)
            (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))

            print(f"Saved {len(graphs)} sentences x {beam_size} candidates")
            print(f"  Run metadata : {out_dir}/run_meta.json")
            print(f"  Per-sentence : {sent_dir}/sent_XXXX.json")
            print(f"  Flat txt     : {pred_path}")
            print(f"  Top-1 AMR    : {top1_path}")

        else:
            # ---- top-1 only mode: mirrors Spring exactly ----
            out_dir = root_out_dir / f"beam_{beam_size}"
            out_dir.mkdir(parents=True, exist_ok=True)
            pred_path = out_dir / 'pred.amr.txt'

            pieces = [encode(g) for g in graphs]
            pred_path.write_text('\n\n'.join(pieces))

            score = compute_smatch(gold_path, pred_path)
            print(f'Smatch (beam={beam_size}): {score:.3f}')
            all_smatch[beam_size] = score

            # save smatch and run_meta per beam
            run_meta = {
                'checkpoint'          : str(args.checkpoint),
                'beam_size'           : beam_size,
                'batch_size'          : args.batch_size,
                'model'               : args.model,
                'penman_linearization': args.penman_linearization,
                'use_pointer_tokens'  : args.use_pointer_tokens,
                'use_recategorization': args.use_recategorization,
                'restore_name_ops'    : args.restore_name_ops,
                'datasets'            : args.datasets,
                'num_sentences'       : len(graphs),
                'smatch'              : round(score, 4),
                'timestamp'           : datetime.now().isoformat(),
            }
            (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))
            (out_dir / 'smatch.txt').write_text(
                f"beam_size : {beam_size}\n"
                f"smatch    : {score:.4f}\n"
                f"pred_file : {pred_path}\n"
                f"gold_file : {gold_path}\n"
                f"checkpoint: {args.checkpoint}\n"
            )
            print(f"  Predictions  : {pred_path}")
            print(f"  Run metadata : {out_dir}/run_meta.json")
            print(f"  Smatch       : {out_dir}/smatch.txt")

    # ------------------------------------------------------------------ #
    # Save overall smatch summary across all beams
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print("SMATCH SUMMARY")
    print(f"{'='*60}")
    with open(summary_path, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset   : {args.datasets}\n")
        f.write(f"Timestamp : {datetime.now().isoformat()}\n\n")
        f.write(f"{'Beam':<10} | Smatch\n")
        f.write("-" * 25 + "\n")
        for beam_size, score in all_smatch.items():
            line = f"beam={beam_size:<6} | {score:.4f}"
            print(line)
            f.write(line + "\n")
    print(f"\nSummary saved to: {summary_path}")
