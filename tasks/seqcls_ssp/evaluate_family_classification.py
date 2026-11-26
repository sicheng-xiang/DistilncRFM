
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from peft import PeftModel
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(ROOT))
from tasks.inference_efficiency import InferenceMeter, add_inference_arguments, merge_for_inference, output_directory
from src.structRFM.model import get_structRFM_for_cls, get_model_scale
from transformers import AutoTokenizer
from collators import SeqClsCollator
from RNAdata import SeqClsDataset
from metrics import SeqClsMetrics

LABEL2ID = {
    "nRC": {
        "5S_rRNA": 0,
        "5_8S_rRNA": 1,
        "tRNA": 2,
        "ribozyme": 3,
        "CD-box": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "riboswitch": 7,
        "IRES": 8,
        "HACA-box": 9,
        "scaRNA": 10,
        "leader": 11,
        "miRNA": 12
    },
    "lncRNA_H": {"lnc": 0, "pc": 1},
    "lncRNA_H_uni": {"lnc": 0, "pc": 1},
    "lncRNA_M": {"lnc": 0, "pc": 1},
    "IRES": {"negative": 0, "positive": 1},
}

ID2LABEL = {task: {v: k for k, v in d.items()} for task, d in LABEL2ID.items()}

#LM_PATH = "/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW"
#LORA_PATH = "/root/autodl-tmp/structRFM-master2/tasks/seqcls_ssp/checkpoints/DistilncRFM-EW"

#LORA_PATH = "/root/autodl-tmp/structRFM-master2/tasks/seqcls_ssp/checkpoints/structRFM_86M"

# LM_PATH = '/root/autodl-tmp/structRFM-master2/model/structRFM_11M'
# LORA_PATH = '/root/autodl-tmp/structRFM-master2/tasks/seqcls_ssp/checkpoints/structRFM_11M'

LM_PATH = str(ROOT / 'model/DistilncRFM')
LORA_PATH = str(ROOT / 'tasks/seqcls_ssp/checkpoints/DistilncRFM')
MODEL_SCALE = "small"
DATASET = "nRC"
MAX_SEQ_LEN = 512
DATA_DIR = str(ROOT / 'data/data')
BATCH_SIZE = 32

import argparse, json
parser = argparse.ArgumentParser('Portable nRC evaluation')
parser.add_argument('--LM_path', default=LM_PATH)
parser.add_argument('--lora_path', default=LORA_PATH)
parser.add_argument('--model_scale', default=MODEL_SCALE)
parser.add_argument('--dataset_dir', default=DATA_DIR)
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--output_dir', default=None, help='默认写到 results/nrc_inference_<时间>，不覆盖旧结果')

parser.add_argument('--checkpoint', default=None, help='全量微调的 best_model 目录；给定时忽略 --lora_path')
from utils import str2bool
parser.add_argument('--attention_mask', type=str2bool, default=False,
                    help='LoRA only: match training mask setting; full checkpoints use their saved configuration.')
add_inference_arguments(parser)
args = parser.parse_args()
outdir = output_directory(ROOT, 'nrc', args.output_dir)


def root_path(path):
    """Resolve relative paths against the project root."""
    p = Path(path).expanduser()
    return str((p if p.is_absolute() else ROOT / p).resolve())


for key in ('LM_path', 'lora_path', 'dataset_dir', 'output_dir', 'checkpoint'):
    if getattr(args, key):
        setattr(args, key, root_path(getattr(args, key)))
LM_PATH, LORA_PATH, MODEL_SCALE = args.LM_path, args.lora_path, args.model_scale
DATA_DIR, BATCH_SIZE, DEVICE = args.dataset_dir, args.batch_size, args.device
USE_MASK, REPLACE_T, REPLACE_U = args.attention_mask, True, False
if args.checkpoint:
    CKPT = Path(args.checkpoint)
    run_cfg = json.loads((CKPT / 'run_config.json').read_text(encoding='utf-8'))
    LM_PATH = root_path(run_cfg['LM_path'])
    MODEL_SCALE, DATASET, MAX_SEQ_LEN = run_cfg['model_scale'], run_cfg['dataset'], run_cfg['max_seq_len']
    USE_MASK, REPLACE_T, REPLACE_U = run_cfg['attention_mask'], run_cfg['replace_T'], run_cfg['replace_U']

NUM_CLASS = len(LABEL2ID[DATASET])
LABEL2ID_TASK = LABEL2ID[DATASET]

print("Loading model...")
model_paras = get_model_scale(MODEL_SCALE)
tokenizer = AutoTokenizer.from_pretrained(str(CKPT) if args.checkpoint else LM_PATH)

model = get_structRFM_for_cls(
    num_class=NUM_CLASS,
    from_pretrained=LM_PATH,
    freeze_base=True,
    use_mean_feature=False,
    **model_paras
)
if args.checkpoint:
    model.load_state_dict(torch.load(CKPT / 'model_state.pt', map_location='cpu', weights_only=True))
else:
    model = PeftModel.from_pretrained(model, LORA_PATH)
model = merge_for_inference(model)
model.to(DEVICE)
model.eval()
print(f"Model loaded from {args.checkpoint or LORA_PATH} (attention_mask={USE_MASK})")

print("Loading test dataset...")
dataset = SeqClsDataset(
    fasta_dir=DATA_DIR,
    prefix=DATASET,
    tokenizer=tokenizer,
    train=False
)
print(f"Test samples: {len(dataset)}")

collator = SeqClsCollator(
    max_seq_len=MAX_SEQ_LEN,
    tokenizer=tokenizer,
    label2id=LABEL2ID_TASK,
    replace_T=REPLACE_T,
    replace_U=REPLACE_U,
    is_structRFM=True
)

from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collator,
    num_workers=0
)

print("Predicting...")
outputs_list = []
labels_list = []

def prepare_batch(data):
    kwargs = dict(input_ids=data['input_ids'].to(DEVICE))
    if USE_MASK:
        kwargs['attention_mask'] = data['attention_mask'].to(DEVICE)
    return kwargs

meter = InferenceMeter(model, DEVICE, BATCH_SIZE, args.warmup_batches,
                       task=DATASET, LM_path=LM_PATH, lora_path=None if args.checkpoint else LORA_PATH,
                       checkpoint=args.checkpoint, attention_mask=USE_MASK, max_seq_len=MAX_SEQ_LEN)
for data, logits in meter.predict(dataloader, prepare_batch):
    outputs_list.append(logits)
    labels_list.append(data['labels'].cpu())
meter.save(outdir)

outputsataset = torch.concat(outputs_list, axis=0)
labels_dataset = torch.concat(labels_list, axis=0)

metric_calculator = SeqClsMetrics(metrics=["f1s", "recall", "precision", "mcc", "accuracy"])
metrics_result = metric_calculator(outputsataset, labels_dataset)

print("\n" + "="*50)
print(f"Classification - {DATASET}")
print("="*50)
print(f"F1: {metrics_result['f1s']:.4f}")
print(f"Recall: {metrics_result['recall']:.4f}")
print(f"Precision: {metrics_result['precision']:.4f}")
print(f"MCC: {metrics_result['mcc']:.4f}")
print(f"Accuracy: {metrics_result['accuracy']:.4f}")
print("="*50)



outdir.mkdir(parents=True, exist_ok=True)
(outdir/'metrics.json').write_text(json.dumps(metrics_result, indent=2, default=float)+'\n', encoding='utf-8')
np.savez(outdir/'predictions.npz', logits=outputsataset.detach().cpu().numpy(), labels=labels_dataset.cpu().numpy())
