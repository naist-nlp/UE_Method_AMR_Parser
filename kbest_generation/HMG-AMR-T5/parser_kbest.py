import json
import warnings
import argparse
from pathlib import Path

import penman
import torch
import yaml
from amrlib.utils.logging import silence_penman
from amrlib.models.parse_t5.penman_serializer import PenmanDeSerializer
from penman.models.noop import NoOpModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from amr_utils.datasets.dataset import AMRPenman, add_prefix, AMR_GENERATION
from amr_parsing.t5.models.lg import LG
from amr_parsing.t5.cli import ROOT
from utils.multi_clause import MultiClause
from utils.multi_phrase import MultiPhrase
from utils.sentence_split import SentenceSplit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.simplefilter('ignore')


def init_test_features(config, net):
    mg_list = [int(num) for num in config["mg_list"].split("-")]
    p_mg = [int(num) for num in config["p_mg"].split("-")]
    mg_dict = {
        "mg_phrase_1_len": MultiPhrase(config["test_constituent"], net.tokenizer, p_mg[0]).phrases_len_list,
        "mg_phrase_2_len": MultiPhrase(config["test_constituent"], net.tokenizer, p_mg[1]).phrases_len_list,
        "mg_phrase_3_len": MultiPhrase(config["test_constituent"], net.tokenizer, p_mg[2]).phrases_len_list,
        "mg_subsentence_len": MultiClause(config["test_clause"], net.tokenizer).phrases_len_list,
        "mg_sentence_len": SentenceSplit(config["test_sentence"], net.tokenizer).sentence_len_list,
    }
    return mg_dict, mg_list


def parse_gold(amr_str):
    lines = [l for l in amr_str.splitlines() if not l.startswith('#')]
    return '\n'.join(lines).strip()


def parse_kbest(batch_size, root, num_beams, num_ret_seq, out_dir, net, mg_dict, mg_list):
    silence_penman()
    data_set = AMRPenman(root)

    # add index to each sample as train.py does
    for idx, sample in enumerate(data_set.data):
        data_set.data[idx] += (idx,)

    valid_set = DataLoader(data_set, batch_size=2*batch_size, num_workers=4, shuffle=False)

    net.eval()
    all_sentences  = []
    all_amrs_ref   = []
    all_gids       = []
    all_kbest      = []

    with torch.no_grad():
        for amr_seq, sentence, amr, index in tqdm(valid_set):
            # collect gold metadata
            gids = []
            for a in amr:
                g = penman.decode(a, model=NoOpModel())
                gids.append(g.metadata.get('id', ''))

            # generate flat list: num_ret_seq * batch_size
            graphs_generated = net.generate(
                add_prefix(sentence, AMR_GENERATION),
                index,
                mg_dict,
                mg_list,
                num_beams=num_beams,
                num_ret_seq=num_ret_seq
            )

            batch_size_actual = len(sentence)
            for snum in range(batch_size_actual):
                raw = graphs_generated[snum * num_ret_seq:(snum + 1) * num_ret_seq]
                candidates = []
                for g in raw:
                    gstring = PenmanDeSerializer(g).get_graph_string()
                    candidates.append(gstring if gstring is not None else '(x / amr-empty)')
                all_kbest.append(candidates)

            all_sentences += list(sentence)
            all_amrs_ref  += list(amr)
            all_gids      += gids

    # save JSON per sentence
    out_dir  = Path(out_dir)
    sent_dir = out_dir / 'sentences'
    sent_dir.mkdir(parents=True, exist_ok=True)

    for i, candidates in enumerate(all_kbest):
        gold_str = all_amrs_ref[i] if i < len(all_amrs_ref) else ''
        sent_data = {
            'sent_idx': i,
            'sent_id':  all_gids[i] if i < len(all_gids) else '',
            'sentence': all_sentences[i] if i < len(all_sentences) else '',
            'gold_graph': parse_gold(gold_str),
            'beam_size': num_ret_seq,
            'num_candidates': len(candidates),
            'candidates': [
                {'rank': rank + 1, 'graph': g}
                for rank, g in enumerate(candidates)
            ]
        }
        (sent_dir / f'sent_{i:04d}.json').write_text(json.dumps(sent_data, indent=2))

    run_meta = {
        'parser': 'hmg_amr_t5',
        'num_beams': num_beams,
        'num_ret_seq': num_ret_seq,
        'num_sentences': len(all_kbest),
    }
    (out_dir / 'run_meta.json').write_text(json.dumps(run_meta, indent=2))
    print(f"Saved {len(all_kbest)} sentences x {num_ret_seq} candidates to {sent_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',             required=True)
    parser.add_argument('-b', '--batch',            type=int, default=4)
    parser.add_argument('-m', '--model',            type=str, default='t5-large')
    parser.add_argument('--max_source_length',      type=int, default=512)
    parser.add_argument('--max_target_length',      type=int, default=512)
    parser.add_argument('--num_beams',              type=int, default=5)
    parser.add_argument('--num_ret_seq',            type=int, default=5)
    parser.add_argument('--model_type',             type=str, default='t5')
    parser.add_argument('-c', '--checkpoint',       type=str, required=True)
    parser.add_argument('--data_type',              default='amrdata')
    parser.add_argument('--task_type',              default='text2amr')
    parser.add_argument('-o', '--out-dir',          required=True)
    parser.add_argument('--config', type=Path,      default=ROOT/'configs/sweeped.yaml')
    args = parser.parse_args()

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    net = LG(args.model,
             max_source_length=args.max_source_length,
             max_target_length=args.max_target_length,
             model_type=args.model_type,
             head_list=config['head_list'],
             mg_layer_begin_index=config['mg_layer_begin_index'],
             mg_layer_end_index=config['mg_layer_end_index'],
             mg_list=config['mg_list'],
             hie_layer_begin_index=config['hie_layer_begin_index'],
             hie_layer_end_index=config['hie_layer_end_index'],
             hie_mode=config['hie_mode'],
             lstm_state=config['lstm_state'])

    print("Loading checkpoint from", args.checkpoint)
    checkpoint = torch.load(args.checkpoint)

    # fix weight-norm keys: rename _weight -> weight
    state_dict = checkpoint['model_state_dict']
    fixed = {}
    for k, v in state_dict.items():
        new_k = k.replace('.hh._weight', '.hh.weight').replace('.ih._weight', '.ih.weight')
        fixed[new_k] = v

    if torch.cuda.device_count() <= 1:
        net.model.load_state_dict(fixed, strict=False)
    else:
        net.model.module.load_state_dict(fixed, strict=False)

    mg_dict, mg_list = init_test_features(config, net)

    parse_kbest(
        batch_size=args.batch,
        root=args.test,
        num_beams=args.num_beams,
        num_ret_seq=args.num_ret_seq,
        out_dir=args.out_dir,
        net=net,
        mg_dict=mg_dict,
        mg_list=mg_list,
    )
