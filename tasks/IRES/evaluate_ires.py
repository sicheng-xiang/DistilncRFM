# 修改原因：独立第四版测试集选模诊断，保留原版与规范验证集实验；修改时间：2026-10-15
# 原始来源：eval3.py（原文件保留）。
# 修改原因：统一softmax与固定阈值，按训练配置加载，保存可复算预测，明确标记测试集选模；修改时间：2026-10-15
import argparse
import json
from pathlib import Path
import sys

# 修改原因：迁移到 Windows 时按项目根目录解析路径，保留原实验计算；修改时间：2026-11-09
from portable_paths import project_path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from tasks.inference_efficiency import InferenceMeter, add_inference_arguments, merge_for_inference, output_directory


def get_args():
    p = argparse.ArgumentParser('IRES test-selected diagnostic evaluation')
    p.add_argument('--checkpoint', default='tasks/IRES/checkpoints/DistilncRFM', help='第四版best_model目录')
    p.add_argument('--LM_path', default=None, help='可选校验；默认读取训练记录')
    p.add_argument('--model_scale', choices=['small', 'base', 'large'], default=None)
    p.add_argument('--dataset_dir', default=None, help='默认读取训练记录的数据路径')
    p.add_argument('--output_dir')
    p.add_argument('--device', default=None)
    p.add_argument('--batch_size', type=int, default=32)
    add_inference_arguments(p)
    return p.parse_args()


def main():
    args = get_args()
    # 修改原因：Windows 下 pandas 自带旧版 msvcp140.dll(14.29)，先于 torch 导入会使 c10.dll 初始化失败(WinError 1114)，故先导入 torch；修改时间：2026-11-09
    import torch
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from peft import PeftModel
    from src.structRFM.model import get_model_scale, get_structRFM_for_cls as get_cls_model
    from ires_mean_model import get_ires_mean_model
    from RNAdata import IRESDataset
    from collators import SeqClsCollator
    from metrics import SeqClsMetrics
    from sklearn.metrics import roc_curve, precision_recall_curve

    checkpoint = Path(project_path(args.checkpoint))
    config_path = checkpoint / 'run_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        config['LM_path'] = project_path(config['LM_path'])
        config['dataset_dir'] = project_path(config['dataset_dir'])
        if args.LM_path: args.LM_path = project_path(args.LM_path)
        if args.dataset_dir: args.dataset_dir = project_path(args.dataset_dir)
        if args.LM_path and Path(args.LM_path).resolve() != Path(config['LM_path']).resolve():
            raise ValueError('LM_path与训练记录不符')
        if args.model_scale and args.model_scale != config['model_scale']:
            raise ValueError('model_scale与训练记录不符')
    else:
        raise ValueError('第四版评估需要run_config.json')
    if config.get('evaluation_protocol') != 'test_selected_diagnostic' or not config.get('attention_mask'):
        raise ValueError('需要第四版测试集选模诊断checkpoint')
    pooling = config.get('pooling')
    use_mean_feature = pooling == 'nucleotide_mean_v1'
    if pooling not in ('cls', 'nucleotide_mean_v1') or config.get('use_mean_feature', use_mean_feature) != use_mean_feature:
        raise ValueError('训练记录的池化配置不一致')
    get_structRFM_for_cls = get_ires_mean_model if use_mean_feature else get_cls_model
    print('DIAGNOSTIC: checkpoint selected on test AUROC; not independent test performance')
    outdir = output_directory(HERE.parents[1], 'ires', args.output_dir)
    device = torch.device(args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = get_structRFM_for_cls(num_class=2, from_pretrained=config['LM_path'],
                                 freeze_base=True, use_mean_feature=use_mean_feature, tokenizer=tokenizer,
                                 **get_model_scale(config['model_scale']))
    if config['use_lora']:
        model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=False)
    else:
        model.load_state_dict(torch.load(checkpoint / 'model_state.pt', map_location='cpu', weights_only=True))
    model = merge_for_inference(model)
    model.to(device).eval()
    dataset = IRESDataset(args.dataset_dir or config['dataset_dir'], '', tokenizer, seed=0, train=False)
    collator = SeqClsCollator(config['max_seq_len'], tokenizer,
                              {'negative': 0, 'positive': 1},
                              replace_T=config['replace_T'], replace_U=config['replace_U'], is_structRFM=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    logits_all, labels_all, names, seqs = [], [], [], []
    def prepare_batch(batch):
        kw = {'input_ids': batch['input_ids'].to(device)}
        if config['attention_mask']:
            kw['attention_mask'] = batch['attention_mask'].to(device)
        return kw
    meter = InferenceMeter(model, device, args.batch_size, args.warmup_batches,
                           task='IRES', LM_path=config['LM_path'], checkpoint=str(checkpoint),
                           attention_mask=config['attention_mask'], max_seq_len=config['max_seq_len'])
    for batch, logits_batch in meter.predict(loader, prepare_batch):
        logits_all.append(logits_batch)
        labels_all.append(batch['labels'])
        names.extend(batch['name']); seqs.extend(batch['seq'])
    meter.save(outdir)
    logits, labels = torch.cat(logits_all), torch.cat(labels_all)
    metric = SeqClsMetrics(['auc'], threshold=config['threshold'])
    metric(logits, labels)
    result = metric.compute()
    result.update(metric.compute_auc_aupr())
    scores = np.asarray(metric.y_score); y = labels.numpy()
    result.update(evaluation_protocol=config['evaluation_protocol'], selection_split='test',
                  dataset_dir=args.dataset_dir or config['dataset_dir'], n=len(y), positive=int(y.sum()), threshold=config['threshold'],
                  attention_mask=config['attention_mask'], checkpoint=str(checkpoint),
                  LM_path=config['LM_path'], model_scale=config['model_scale'],
                  pooling=config['pooling'], score_method='softmax', aupr_definition='trapezoidal PR-AUC', ap_definition='average_precision')
    pd.DataFrame(dict(name=names, seq=seqs, true_label=y,
                      class_0_logit=logits[:, 0].numpy(), class_1_logit=logits[:, 1].numpy(),
                      score=scores, threshold=config['threshold'],
                      predicted_class=(scores >= config['threshold']).astype(int))).to_csv(outdir / 'predictions.csv', index=False)
    (outdir / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    fpr, tpr, _ = roc_curve(y, scores)
    precision, recall, _ = precision_recall_curve(y, scores)
    for name, value in [('true_labels', y), ('predict_probs', scores), ('prec_curve', precision), ('rec_curve', recall)]:
        np.save(outdir / (name+'.npy'), value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f'AUROC={result["auc"]:.4f}')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate')
    axes[1].plot(recall, precision, label=f'PR-AUC={result["aupr"]:.4f}; AP={result["ap"]:.4f}')
    axes[1].set(xlabel='Recall', ylabel='Precision')
    for ax in axes: ax.legend()
    fig.tight_layout(); fig.savefig(outdir / 'roc_pr_curves.png', dpi=200); plt.close(fig)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Saved: {outdir}')


if __name__ == '__main__':
    main()
