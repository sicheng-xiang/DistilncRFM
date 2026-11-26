

import os
import argparse
from datetime import datetime
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


import sys

from pathlib import Path

from portable_paths import project_path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.structRFM.model import get_model_scale, get_structRFM_for_cls as get_cls_model
from ires_mean_model import get_ires_mean_model
from src.structRFM.data import get_mlm_tokenizer


from utils import str2bool, str2list
from collators import SeqClsCollator
from losses import SeqClsLoss
from metrics import SeqClsMetrics
from trainers import SeqClsTrainer
from RNAdata import IRESDataset
# ========== Define constants ==========
MODELS = ["structRFM"]  
MAX_SEQ_LEN = {"structRFM": 512}
TASKS = ["IRES"]
# Binary IRES label mapping.
LABEL2ID = {
    "IRES": {
        "negative": 0,
        "positive": 1
    }
}



def get_args():
    parser = argparse.ArgumentParser('IRES Identification with structRFM + LoRA')
    # model args
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large','small'], default='small')
     
    parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW')
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM')

    
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_11M')

    

    
    #parser.add_argument('--LM_path', type=str, default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM')
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_86M')
    # Dataset arguments.
    parser.add_argument('--dataset_dir', type=str, default="data/data/ires_seq", help='IRES数据根目录')
    parser.add_argument('--dataset', type=str, default="IRES", help='数据集名称')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--replace_T', type=str2bool, default=True)
    parser.add_argument('--replace_U', type=str2bool, default=False)
    
    # Training arguments.
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='全量微调底座学习率')
    parser.add_argument('--head_learning_rate', type=float, default=1e-4)
    parser.add_argument('--eval_steps', type=int, default=50)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--disable_tqdm', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Effective batch = batch_size x gradient_accumulation_steps')
    parser.add_argument('--freeze_base', type=str2bool, default=False)
    parser.add_argument('--pooling', choices=['cls', 'nucleotide_mean_v1'], default='cls')
    parser.add_argument('--num_train_epochs', type=int, default=15)
    parser.add_argument('--metrics', type=str2list, default="F1s,Precision,Recall,Accuracy,Mcc")
    parser.add_argument('--logging_steps', type=int, default=1000, help='日志打印间隔')
    
    # LoRA arguments.
    parser.add_argument('--use_lora', type=str2bool, default=False)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    #parser.add_argument('--lora_target_modules', type=str, default="query,value,key,intermediate.dense,output.dense")
    parser.add_argument('--lora_target_modules', type=str, default="query,value,key,dense")
    parser.add_argument('--load_in_4bit', type=str2bool, default=False)
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4")
    
    # Checkpoint arguments.
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_checkpoints', type=str2bool, default=False,
                        help='Efficiency runs skip saving/restoring duplicate weights; true restores checkpoint saving.')
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--evaluation_strategy', type=str, default='steps')
    parser.add_argument('--load_best_model_at_end', type=str2bool, default=True)
    parser.add_argument('--metric_for_best_model', type=str, default='auc', choices=['auc'])
    parser.add_argument('--greater_is_better', type=str2bool, default=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pos_weight', type=float, default=2.5)
    
    args = parser.parse_args()
    return args


def freeze(model):
    """Freeze all model parameters."""
    for name, para in model.named_parameters():
        para.requires_grad = False

def init_lora_model(model, args):
    """Configure LoRA and attach adapters to the model."""
    if not args.use_lora:
        return model
    
    # Prepare for optional 4-bit training.
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # Resolve LoRA target modules.
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.model_name == "structRFM" and not target_modules:
        target_modules = ["query", "value"]
    
    # Configure LoRA.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        inference_mode=False,
        modules_to_save=["cls"]
    )
    
    # Apply LoRA adapters.
    model = get_peft_model(model, lora_config)
    
    # Training arguments.
    model.print_trainable_parameters()
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append({
                "name": name,
                "num_params": param.numel(),
                "shape": list(param.shape),
                "dtype": str(param.dtype)
            })
    pass
    for idx, p in enumerate(trainable_params, 1):
        pass
    total_trainable = sum(p["num_params"] for p in trainable_params)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable fraction: {total_trainable/total_all*100:.4f}%")
    
    return model


if __name__ == "__main__":
    args = get_args()
    
    import random, numpy as np, json
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    args.use_mean_feature = args.pooling == 'nucleotide_mean_v1'
    get_structRFM_for_cls = get_ires_mean_model if args.use_mean_feature else get_cls_model
    args.evaluation_protocol = 'test_selected_diagnostic'
    args.selection_split = 'test'
    if not args.greater_is_better:
        raise ValueError('第四版固定保存测试AUROC最大的模型，greater_is_better必须为True')
    print(f'DIAGNOSTIC: full original train; test selects best AUROC; pooling={args.pooling}')
    if args.load_in_4bit:
        raise ValueError("旧入口未真正加载4bit权重，本修复版不支持该选项。")
    if not args.train or args.evaluation_strategy not in ('steps', 'epoch') or args.save_strategy != args.evaluation_strategy:
        raise ValueError('train=True required; evaluation_strategy/save_strategy must match (steps or epoch)')
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError('batch_size and gradient_accumulation_steps must be positive')
    if not np.isfinite(args.pos_weight) or args.pos_weight <= 0 or args.num_train_epochs < 1:
        raise ValueError("Invalid pos_weight/num_train_epochs")
    
    
    if args.use_lora or args.freeze_base:
        raise ValueError('第五版要求use_lora=False且freeze_base=False')
    if args.eval_steps < 1 or any(not np.isfinite(x) or x <= 0 for x in [args.learning_rate, args.head_learning_rate]):
        raise ValueError('eval_steps和两个学习率必须为正数')

    
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    
    if args.max_seq_len != 512:
        raise ValueError("本次修复保持max_seq_len=512；长度优化留待后续实验。")
    args.LM_path = project_path(args.LM_path)
    args.dataset_dir = project_path(args.dataset_dir)
    args.output_dir = project_path(args.output_dir or f'results/ires_full_efficiency_{datetime.now():%Y%m%d_%H%M%S_%f}')
    target = Path(args.output_dir) / args.pooling
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve old checkpoints: {target}')

    
    from_pretrained = args.LM_path
    model_paras = get_model_scale(args.model_scale)
    print("!"*20)
    print(args.model_scale)
    print( model_paras)
    print("!"*20)
    # Adjust the model for longer inputs.
    if args.max_seq_len + 2 > 514:
        tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len + 2)
        model = get_structRFM_for_cls(
            num_class=len(LABEL2ID["IRES"]),  # Binary classification.
            from_pretrained=from_pretrained,
            tokenizer=tokenizer,
            pretrained_length=514,
            freeze_base=args.freeze_base and not args.use_lora,
            use_mean_feature=args.use_mean_feature,
            **model_paras
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        model = get_structRFM_for_cls(
            num_class=len(LABEL2ID["IRES"]),  # Binary classification.
            from_pretrained=from_pretrained,
            tokenizer=tokenizer,
            freeze_base=args.freeze_base and not args.use_lora,
            use_mean_feature=args.use_mean_feature,
            **model_paras
        )
    
    # Apply LoRA adapters.
    pass
    pass
    if args.use_lora:
        model = init_lora_model(model, args)

    
    args.device = torch.device(args.device)
    model.to(args.device)
    # Single selected device, matching the other efficiency tasks.
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n===== Parameters =====")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable fraction: {trainable_params/total_params*100:.4f}%")

    
    _loss_fn = SeqClsLoss(pos_weight=args.pos_weight).to(args.device)
    
    
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    backbone_params = list(raw_model.structRFM.parameters())
    head_params = list(raw_model.cls.parameters())
    assert all(p.requires_grad for p in model.parameters()), '全量微调不能有被冻结的参数'
    assert {id(p) for p in backbone_params + head_params} == {id(p) for p in model.parameters()}
    optimizer = AdamW([
        {'params': backbone_params, 'lr': args.learning_rate, 'name': 'backbone'},
        {'params': head_params, 'lr': args.head_learning_rate, 'name': 'head'},
    ], weight_decay=0.01)
    print(f'Full fine-tuning: backbone_lr={args.learning_rate}, head_lr={args.head_learning_rate}, eval_steps={args.eval_steps}')

    _metric = SeqClsMetrics(metrics=args.metrics)

    
    args.output_dir = str(Path(args.output_dir) / args.pooling)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_lora:
        args.output_dir = os.path.join(args.output_dir, f"lora_r{args.lora_r}_alpha{args.lora_alpha}")
        os.makedirs(args.output_dir, exist_ok=True)

    
    if any(Path(args.output_dir).iterdir()):
        raise FileExistsError(f"输出目录非空，请指定新的--output_dir: {args.output_dir}")
    
    dataset_train = IRESDataset(
    fasta_dir=args.dataset_dir,  # IRES sequence directory.
    prefix="",                   # No filename prefix is needed for the IRES layout.
    tokenizer=tokenizer,
    seed=0,
    train=True
    )

    
    dataset_eval = IRESDataset(args.dataset_dir, '', tokenizer, seed=0, train=False)
    args.train_count, args.selection_count = len(dataset_train), len(dataset_eval)
    split_rows = [{"name": row[1], "label": row[2], "split": part}
                  for part, ds in [('train', dataset_train), ('test_selection', dataset_eval)]
                  for row in ds.data]
    with open(Path(args.output_dir) / 'split_manifest.json', 'w') as f:
        json.dump(split_rows, f, indent=2)
    with open(Path(args.output_dir) / 'experiment_config.json', 'w') as f:
        json.dump(dict(vars(args), device=str(args.device)), f, indent=2)
    print(f'Full train={len(dataset_train)}, diagnostic test selection={len(dataset_eval)}')

    
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        label2id=LABEL2ID["IRES"],
        replace_T=args.replace_T,
        replace_U=args.replace_U,
        is_structRFM=True  # Use structRFM tokenization.
    )

    
    seq_cls_trainer = SeqClsTrainer(
        args=args,
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=_collate_fn,
        loss_fn=_loss_fn,
        optimizer=optimizer,
        compute_metrics=_metric,
    )

    
    if args.train:
        seq_cls_trainer.fit()
