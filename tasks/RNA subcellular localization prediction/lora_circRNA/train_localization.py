
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
import os
import argparse
import math
from datetime import datetime
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


import sys
sys.path.insert(0, str(ROOT))
from tasks.training_efficiency import fit_custom
from src.structRFM.model import get_structRFM_for_cls, get_model_scale
from src.structRFM.data import get_mlm_tokenizer

from utils import get_config, get_and_set_device, str2bool, str2list
from collators import SeqClsCollator
from losses import SeqClsLoss
from metrics import SeqClsMetrics
from trainers import SeqClsTrainer
from RNAdata import RNADataset


# ========== Define constants ==========
MODELS = ["structRFM"]
MAX_SEQ_LEN = {"structRFM": 512}
TASKS = ["lncRNA", "miRNA", "circRNA"]

# Map label names to class IDs.
LABEL2ID = {
    
    "circRNA": {"nucleus": 0, "cytoplasm": 1},
    "lncRNA": {
        "nucleus": 0,
        "cytoplasm": 1
    },
    "miRNA": {
        "intracellular": 1,  # Nucleus, cytoplasm, mitochondrion, ribosome: intracellular.
        "extracellular": 0   # Exosome, microvesicle, extracellular: extracellular.
    }
}



def get_args():
    parser = argparse.ArgumentParser('circRNA Subcellular Localization with structRFM + LoRA')
    # model args
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large', 'small'], default='small')
    #parser.add_argument('--LM_path', type=str, default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW')
     
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW')

    
    parser.add_argument('--LM_path', type=str,default=str(ROOT / 'model/structRFM_11M'))

    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/autodl-tmp/structRFM-master2/model2/checkpoint-577854')

    
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_86M')
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM')
    # Dataset arguments.
    parser.add_argument('--data_dir', type=str, default=str(ROOT / 'tasks/RNA subcellular localization prediction/RNALoc-LM-main/data'), help='数据目录')
    
    parser.add_argument('--dataset', type=str, default="circRNA", choices=TASKS, help='数据集类型')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)
    
    # Training arguments.
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--disable_tqdm', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Effective batch = batch_size x gradient_accumulation_steps')
    parser.add_argument('--freeze_base', type=str2bool, default=True)
    parser.add_argument('--use_mean_feature', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=float, default=3, help='warmup轮数')
    parser.add_argument('--use_cosine_lr', type=str2bool, default=False, help='使用余弦学习率衰减')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--metrics', type=str2list, default="F1s,Precision,Recall,Accuracy,Mcc")
    parser.add_argument('--logging_steps', type=int, default=1000, help='日志打印间隔')
    
    # LoRA arguments.
    parser.add_argument('--use_lora', type=str2bool, default=True)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--lora_dropout', type=float, default=0.3)
    parser.add_argument('--lora_target_modules', type=str, default="query,value,key,dense")  
    parser.add_argument('--load_in_4bit', type=str2bool, default=False)
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4")
    
    # Checkpoint arguments.
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_checkpoints', type=str2bool, default=False,
                        help='Efficiency tests skip duplicate adapters; true restores original saves.')
    parser.add_argument('--save_strategy', type=str, default='epoch')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--load_best_model_at_end', type=str2bool, default=True)
    parser.add_argument('--metric_for_best_model', type=str, default='f1s')
    parser.add_argument('--greater_is_better', type=str2bool, default=True)
    parser.add_argument('--save_total_limit', type=int, default=5)
    
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
    
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.model_name == "structRFM" and not target_modules:
        target_modules = ["query", "value"]
    
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
    
    model = get_peft_model(model, lora_config)
    
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
    print("\n=== 可训练参数明细 ===")
    for idx, p in enumerate(trainable_params, 1):
        print(f"[{idx}] 名称: {p['name']} | 数量: {p['num_params']:,} | 形状: {p['shape']}")
    total_trainable = sum(p["num_params"] for p in trainable_params)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"\n可训练参数占比: {total_trainable/total_all*100:.4f}%")
    
    return model



if __name__ == "__main__":
    args = get_args()
    for key in ('LM_path', 'data_dir'):
        value = Path(getattr(args, key)).expanduser()
        setattr(args, key, str((value if value.is_absolute() else ROOT / value).resolve()))
    value = Path(args.output_dir) if args.output_dir else Path('results') / f'localization_lora_{datetime.now():%Y%m%d_%H%M%S_%f}'
    args.output_dir = str((value if value.is_absolute() else ROOT / value).resolve())
    target = Path(args.output_dir)
    if args.use_lora:
        target /= f'lora_r{args.lora_r}_alpha{args.lora_alpha}'
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve existing adapters: {target}')
    if min(args.batch_size, args.gradient_accumulation_steps, args.num_train_epochs) < 1:
        raise ValueError('batch_size, gradient_accumulation_steps and num_train_epochs must be positive')
    if args.load_in_4bit:
        raise ValueError("This migration supports existing non-quantized LoRA runs only")
    print(args)
    
    
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    
    from_pretrained = args.LM_path
    model_paras = get_model_scale(args.model_scale)
    print("!"*20)
    print(args.model_scale)
    print(model_paras)
    print("!"*20)
    
    if args.max_seq_len + 2 > 514:
        tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len + 2)
        model = get_structRFM_for_cls(
            num_class=len(LABEL2ID[args.dataset]),
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
            num_class=len(LABEL2ID[args.dataset]),
            from_pretrained=from_pretrained,
            freeze_base=args.freeze_base and not args.use_lora,
            use_mean_feature=args.use_mean_feature,
            **model_paras
        )
    
    # Apply LoRA adapters.
    print('=== 应用LoRA前的模型结构 ===')
    print(model)
    if args.use_lora:
        model = init_lora_model(model, args)
    print('=== 应用LoRA后的模型结构 ===')
    print(model)


    
    args.device = torch.device(args.device)
    # Single selected GPU, matching the other tasks' efficiency measurement scope.
    model.to(args.device)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n===== 参数统计 =====")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"可训练占比: {trainable_params/total_params*100:.4f}%")

    
    # For lncRNA, cytoplasm (label 1) is the minority class.
    # For miRNA, intracellular (label 1) is the minority class.
    # if args.dataset == "lncRNA":
    #     pos_weight = 1.8
    # else:  # miRNA
    
    _loss_fn = SeqClsLoss(
        pos_weight=1.5,
        label_smoothing=0.1).to(args.device)
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_lora:
        args.output_dir = os.path.join(args.output_dir, f"lora_r{args.lora_r}_alpha{args.lora_alpha}")
        os.makedirs(args.output_dir, exist_ok=True)

    
    dataset_train = RNADataset(
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        prefix="train",
        tokenizer=tokenizer,
        seed=0,
        
        train=True,
        max_nt_exclusive=512
    )

    dataset_eval = RNADataset(
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        prefix="test",
        tokenizer=tokenizer,
        seed=0,
        
        train=False,
        max_nt_exclusive=512
    )
    print('\n===== 数据统计 =====')
    print(f'dataset train: {len(dataset_train)}, test: {len(dataset_eval)}')
    
    # Cosine learning-rate schedule with warmup.
    optimizer = AdamW(
        params=[para for para in model.parameters() if para.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.05
    )
    
    # Cosine learning-rate schedule with warmup.
    updates_per_epoch = math.ceil(math.ceil(len(dataset_train) / args.batch_size) / args.gradient_accumulation_steps)
    num_training_steps = updates_per_epoch * args.num_train_epochs
    num_warmup_steps = int(updates_per_epoch * args.warmup_epochs)
    
    if args.use_cosine_lr:
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"学习率调度: cosine, warmup={args.warmup_epochs}epochs, min_lr={args.min_lr}")
    else:
        lr_scheduler = None
    
    _metric = SeqClsMetrics(metrics=args.metrics)

    
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        label2id=LABEL2ID[args.dataset],
        replace_T=args.replace_T,
        replace_U=args.replace_U,
        is_structRFM=True
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
        lr_scheduler=lr_scheduler,
    )

    
    if args.train:
        fit_custom(seq_cls_trainer, vars(args))
