
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
"""复评 circRNA <512 nt 测试子集。修改时间：2026-10-14。
必须显式指定训练时的底座：LoRA 配置未记录底座，不能从目录名可靠推断。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from tasks.inference_efficiency import InferenceMeter, add_inference_arguments, merge_for_inference, output_directory


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    
    
    
    p.add_argument('--LM_path', default=str(ROOT / 'model/DistilncRFM-EW'), help='必须与适配器训练时使用的底座一致')

    
    #p.add_argument('--lora_path', default=str(HERE/'lora_circrna_lt512_dis_w_output/lora_r128_alpha256/best_lora_f10.8109_auc0.8705'))
    
    #p.add_argument('--lora_path', default='/root/autodl-tmp/structRFM-master2/tasks/RNA subcellular localization prediction/lora_lncrna/lora_circrna_lt512_student2_output/lora_r128_alpha256/best_lora_f10.7808_auc0.8597')

    #p.add_argument('--lora_path', default='/root/autodl-tmp/structRFM-master2/tasks/RNA subcellular localization prediction/lora_lncrna/lora_circrna_lt512_teacher_output/lora_r128_alpha256/best_lora_f10.8186_auc0.8934')
    p.add_argument('--lora_path', default=str(HERE / 'checkpoints/DistilncRFM-EW'))

    p.add_argument('--data_dir', default=str(HERE.parent/'RNALoc-LM-main/data'))
    p.add_argument('--model_scale', default='small', choices=['small', 'base', 'large'])
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default=None, help='默认优先 CUDA，否则 CPU')
    p.add_argument('--use_mean_feature', action='store_true', help='仅训练时启用此选项才使用')
    p.add_argument('--output_dir', default=None, help='默认写到 results/localization_inference_<时间>')
    add_inference_arguments(p)
    return p.parse_args()


def main():
    args = get_args()
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from peft import PeftModel
    from transformers import AutoTokenizer
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 matthews_corrcoef, roc_auc_score, average_precision_score,
                                 confusion_matrix, classification_report)
    from src.structRFM.model import get_structRFM_for_cls, get_model_scale
    from RNAdata import RNADataset
    from collators import SeqClsCollator

    if args.batch_size <= 0:
        raise ValueError('batch_size 必须大于 0')
    out_dir = output_directory(ROOT, 'localization', args.output_dir)
    for key in ('LM_path', 'lora_path', 'data_dir'):
        value = Path(getattr(args, key)).expanduser()
        setattr(args, key, str((value if value.is_absolute() else ROOT / value).resolve()))
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('Backbone:', args.LM_path, '\nAdapter:', args.lora_path, '\nDevice:', device)
    tokenizer = AutoTokenizer.from_pretrained(args.LM_path)
    dataset = RNADataset(args.data_dir, 'circRNA', 'test', tokenizer,
                         seed=0, train=False, max_nt_exclusive=512)
    
    label2id = {'nucleus': 0, 'cytoplasm': 1}
    collator = SeqClsCollator(max_seq_len=512, tokenizer=tokenizer, label2id=label2id,
                             replace_T=True, replace_U=False, is_structRFM=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=collator)
    model = get_structRFM_for_cls(num_class=2, from_pretrained=args.LM_path,
                                 freeze_base=True, use_mean_feature=args.use_mean_feature,
                                 **get_model_scale(args.model_scale))
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    model = merge_for_inference(model)
    model.to(device).eval()
    logits_all, labels_all, names, lengths = [], [], [], []
    meter = InferenceMeter(model, device, args.batch_size, args.warmup_batches,
                           task='circRNA', LM_path=args.LM_path, lora_path=args.lora_path,
                           attention_mask=False, max_seq_len=512)
    for data, logits in meter.predict(loader, lambda batch: dict(input_ids=batch['input_ids'].to(device))):
        if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
            raise ValueError('模型输出必须是有限的 [batch, 2] logits')
        logits_all.append(logits.numpy())
        labels_all.append(data['labels'].numpy())
        names.extend(data['name'])
        lengths.extend(map(len, data['seq']))
    meter.save(out_dir)
    scores = np.concatenate(logits_all)
    truth = np.concatenate(labels_all)
    
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    if set(truth.tolist()) != {0, 1}:
        raise ValueError('测试集必须含两个类别才能报告 ROC AUC')
    results = {
        'N': len(truth), 'nucleus': int((truth == 0).sum()), 'cytoplasm': int((truth == 1).sum()),
        'Accuracy': accuracy_score(truth, pred),
        'Macro_Precision': precision_score(truth, pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(truth, pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(truth, pred, average='macro', zero_division=0),
        'MCC': matthews_corrcoef(truth, pred),
        'ROC_AUC': roc_auc_score(truth, probs[:, 1]),
        'AUPR_AP': average_precision_score(truth, probs[:, 1]),
        'confusion_matrix': confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
        'positive_class': 'cytoplasm (1)',
        'length_filter': 'raw nucleotide length < 512', 'max_input_tokens': 512,
        'LM_path': str(Path(args.LM_path).resolve()), 'lora_path': str(Path(args.lora_path).resolve()),
        'model_scale': args.model_scale, 'use_mean_feature': args.use_mean_feature,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    with (out_dir/'predictions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['name','length','true_label','predicted_class','class_0_logit','class_1_logit','prob_nucleus','prob_cytoplasm'])
        for i in range(len(truth)):
            w.writerow([names[i], lengths[i], truth[i], pred[i], *scores[i], *probs[i]])
    print(classification_report(truth, pred, labels=[0,1], target_names=['nucleus','cytoplasm'], digits=4, zero_division=0))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print('Results saved:', out_dir)


if __name__ == '__main__':
    main()
