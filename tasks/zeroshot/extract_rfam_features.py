"""GPU feature-extraction benchmark for the original zero-shot Rfam task.

No cosine sampling/evaluation is executed by this entry point. Original representation
postprocessing can optionally be exported, outside the model-forward timer.
"""
import argparse
import json
import pickle
from pathlib import Path
import sys

import torch  # Before pandas/scipy-dependent legacy imports on Windows.
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.structRFM.infer import structRFM_infer
from src.structRFM.data.RNAdata import process_mlm_input_seq
from tasks.inference_efficiency import InferenceMeter, output_directory

PRESETS = {
    'teacher': 'model/structRFM_86M',
    'student': 'model/structRFM_11M',
    'dis': 'model/DistilncRFM-EW',
    'dis_w': 'model/DistilncRFM',
}
ORIGINAL_DATA = ROOT / 'tasks/zeroshot/task2_rfam/data/random_sample_max10_n24607_rf4170.pkl'


def root_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


class RfamSequences(Dataset):
    def __init__(self, records, max_nt=512):
        self.records = records
        self.max_nt = max_nt

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        label, key, seq = self.records[i]
        if not isinstance(seq, str) or not seq:
            raise ValueError(f'Empty/invalid sequence at record {i}')
        # Exact legacy truncation: clip raw sequence before tokenization, then T -> U.
        return dict(label=label, key=key, seq=seq[:self.max_nt])


class RfamCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, records):
        tokens = self.tokenizer([process_mlm_input_seq(r['seq']) for r in records],
                                padding=True, return_tensors='pt')
        return dict(tokens, labels=[r['label'] for r in records],
                    keys=[r['key'] for r in records], seqs=[r['seq'] for r in records])


class LastHiddenState(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        # Same MLM forward as extract_raw_feature(), but no attention matrices.
        outputs = self.model(**inputs, output_attentions=False, output_hidden_states=True)
        return outputs.hidden_states[-1]


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=PRESETS, default='teacher')
    p.add_argument('--LM_path', default=None, help='Optional matching checkpoint; no weights are copied.')
    p.add_argument('--data_path', default=None, help='Original Rfam tuple pickle (label, key, sequence).')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--warmup_batches', type=int, default=3)
    p.add_argument('--max_nt', type=int, default=512)
    p.add_argument('--max_samples', type=int, default=None, help='Optional smoke-test subset; default full dataset.')
    p.add_argument('--save_features', action='store_true', help='Export legacy FFT128/CLS/mean features, outside forward timing.')
    p.add_argument('--output_dir', default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    if args.batch_size < 1 or args.warmup_batches < 0 or not 1 <= args.max_nt <= 512:
        raise ValueError('Positive batch_size, nonnegative warmup_batches, and 1 <= max_nt <= 512 required')
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError('max_samples must be positive')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; choose --device cpu explicitly only for debugging')
    checkpoint = root_path(args.LM_path or PRESETS[args.model])
    local_data = ROOT / 'tasks/zeroshot/task2_rfam/data/random_sample_max10_n24607_rf4170.pkl'
    data_path = root_path(args.data_path) if args.data_path else local_data if local_data.is_file() else ORIGINAL_DATA
    if not data_path.is_file():
        raise FileNotFoundError(f'Rfam data not copied; pass --data_path explicitly: {data_path}')
    config = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
    if args.max_nt + 2 > config['max_position_embeddings']:
        raise ValueError('Requested sequence length exceeds pretrained positional embeddings')
    with data_path.open('rb') as f:
        data = pickle.load(f)
    if args.max_samples is not None:
        data = data[:args.max_samples]
    if len(data) == 0:
        raise ValueError('Empty Rfam dataset')
    out = output_directory(ROOT, f'rfam_{args.model}', args.output_dir)
    infer = structRFM_infer(from_pretrained=str(checkpoint), max_length=config['max_position_embeddings'],
                            dim=config['hidden_size'], layer=config['num_hidden_layers'],
                            num_attention_heads=config['num_attention_heads'], guide=False,
                            motif_tree=False, device=device)
    model = LastHiddenState(infer.get_model()).float().to(device).eval()
    loader = DataLoader(RfamSequences(data, args.max_nt), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=RfamCollator(infer.get_tokenizer()))
    meter = InferenceMeter(model, device, args.batch_size, args.warmup_batches,
                           task='Rfam_zero_shot_features', model_name=args.model, LM_path=str(checkpoint),
                           data_path=str(data_path), subset=args.max_samples is not None,
                           hidden_size=config['hidden_size'], num_hidden_layers=config['num_hidden_layers'],
                           num_attention_heads=config['num_attention_heads'], max_nt=args.max_nt,
                           padding='longest within batch; mask=True', attention_mask=True,
                           output_attentions=False, output_kind='last_hidden_state',
                           save_features=args.save_features,
                           pipeline_postprocessing='Optional FFT/CLS/mean export preparation included only if save_features=True; never included in forward timing')
    (out / 'experiment_config.json').write_text(
        json.dumps(dict(vars(args), LM_path=str(checkpoint), data_path=str(data_path),
                        checkpoint_architecture=config), indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    def prepare(batch):
        return {key: batch[key].to(device) for key in ('input_ids', 'attention_mask', 'token_type_ids') if key in batch}
    features = {'fft128': [], 'cls_full': [], 'mean_pooling': []}
    labels, keys, seqs = [], [], []
    if args.save_features:
        from tasks.zeroshot.extract_embs.tools import fftCal
    print(f'Rfam feature extraction: {len(data)} sequences; device={device}; batch={args.batch_size}; FP32')
    with tqdm(total=len(data), desc='Rfam features') as bar:
        for batch, raw in meter.predict(loader, prepare):
            if args.save_features:
                for i, count in enumerate(batch['attention_mask'].sum(1).tolist()):
                    # Never include batch padding in FFT flattening or mean pooling.
                    array = raw[i, :count].numpy()
                    features['fft128'].append(fftCal(array).squeeze(0))
                    features['cls_full'].append(array[0].copy())
                    features['mean_pooling'].append(raw[i, 1:count-1].mean(0).numpy())
                labels.extend(batch['labels']); keys.extend(batch['keys']); seqs.extend(batch['seqs'])
            bar.update(len(batch['input_ids']))
    meter.save(out)
    if args.save_features:
        for name, rows in features.items():
            payload = dict(labels=labels, keys=keys, seqs=seqs, seq_repr=np.asarray(rows))
            with (out / f'{args.model}_features_{name}.pkl').open('wb') as f:
                pickle.dump(payload, f)
    print('Only model feature extraction measured; cosine similarity/evaluation was not run.')
    print('Saved:', out)


if __name__ == '__main__':
    main()
