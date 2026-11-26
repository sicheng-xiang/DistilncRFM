
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import os
import argparse
from datetime import datetime

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from RNAdata import SeqClsDataset
from utils import get_config, get_and_set_device
from utils import str2bool, str2list
from collators import SeqClsCollator
from losses import SeqClsLoss
from metrics import SeqClsMetrics

from lora_efficiency import MeasuredLoraTrainer as SeqClsTrainer, fit_with_efficiency
from tokenizer import RNATokenizer
#from seq_cls import RNABertForSeqCls, RNAFmForSeqCls, RNAMsmForSeqCls


from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer


import sys
sys.path.insert(0, str(ROOT))
from src.structRFM.model import get_structRFM_for_cls, get_model_scale
from src.structRFM.data import get_mlm_tokenizer

# ========== Define constants
MODELS = ["RNABERT", "RNAMSM", "RNAFM", 'structRFM']
MAX_SEQ_LEN = {"RNABERT": 440,
               "RNAMSM": 512,
               "RNAFM": 512,
               "structRFM": 512,
              }

TASKS = ["nRC", "lncRNA_H", "lncRNA_M", 'IRES', 'lncRNA_H_uni']
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
    "lncRNA_H": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_H_uni": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_M": {
        "lnc": 0,
        "pc": 1
    },
    "IRES": {
        "negative": 0,
        "positive": 1
    },
}

#autodl-tmp/structRFM-master2/autodl-tmp/structRFM-master2/model2/checkpoint-577854
def get_args():
    # ========== Configuration
    parser = argparse.ArgumentParser(
        'Implementation of RNA sequence classification.')
    # model args
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large','small'], default='small')
    parser.add_argument('--vocab_path', type=str, default="./vocabs/")
    
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW')

    
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_11M')

    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/autodl-tmp/structRFM-master2/model2/checkpoint-577854')

    
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_86M')

    parser.add_argument('--LM_path', type=str,default=str(ROOT / 'model/DistilncRFM'))
    parser.add_argument('--config_path', type=str,default="./configs/")
    parser.add_argument('--dataset_dir', type=str, default=str(ROOT / 'data/data'))
    parser.add_argument('--dataset', type=str, default="nRC", choices=TASKS)
    parser.add_argument('--replace_T', type=bool, default=True)
    parser.add_argument('--replace_U', type=bool, default=False)

    parser.add_argument('--device', type=str, help='cpu, cuda:0',default='cuda:0')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--attention_mask', type=str2bool, default=False,
                        help='Mask PAD in both training and evaluation; false preserves original LoRA experiments.')
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--disable_tqdm', type=str2bool,
                        default=False, help='Disable tqdm display if true.')
    
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Micro-batch size; default base=16, other scales=32.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                        help='Micro-batches per update; defaults to effective batch 32.')
    parser.add_argument('--freeze_base', type=str2bool, default=True, help='freeze base language model')  
    parser.add_argument('--use_mean_feature', action='store_true')
    parser.add_argument('--use_automodelforseqcls', action='store_true')
    parser.add_argument('--num_train_epochs', type=int, default=100,
                        help='The number of epoch for training.')
    parser.add_argument('--metrics', type=str2list,
                        default="F1s,Precision,Recall,Accuracy,Mcc",)

    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='Update visualdl logs every logging_steps.')
    
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_checkpoints', type=str2bool, default=False,
                        help='Efficiency runs skip duplicate weight saves by default; true restores original saving.')
    
    
    parser.add_argument('--use_lora', type=str2bool, default=True, help='whether to use LoRA')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA hidden dimension')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default="query,value,key,intermediate.dense,output.dense", help='LoRA target modules (split by comma)')
    parser.add_argument('--load_in_4bit', type=str2bool, default=False, help='load model in 4bit for memory efficiency')
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4", help='4bit quantization type (nf4/fp4)')
    
    parser.add_argument('--save_strategy', type=str, default='epoch')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--load_best_model_at_end', type=str2bool, default=True)
    parser.add_argument('--metric_for_best_model', type=str, default='F1s')
    parser.add_argument('--greater_is_better', type=str2bool, default=True)
    parser.add_argument('--save_total_limit',type=int,default=1)
    
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 16 if args.model_scale == 'base' else 32
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = max(1, 32 // max(1, args.batch_size))
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
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,  # Reduce activation memory.
        )
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
        bias="all",  # Keep bias parameters frozen.
        task_type="SEQ_CLS",  # Sequence classification task.
        inference_mode=False,
        modules_to_save=["cls"]  # Keep the classification head trainable.
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
    for key in ('LM_path', 'dataset_dir'):
        value = Path(getattr(args, key)).expanduser()
        setattr(args, key, str((value if value.is_absolute() else ROOT / value).resolve()))
    if args.output_dir is None:
        args.output_dir = str(ROOT / 'results' / f'lora_efficiency_{Path(args.LM_path).name}_{datetime.now():%Y%m%d_%H%M%S_%f}')
    else:
        value = Path(args.output_dir).expanduser()
        args.output_dir = str((value if value.is_absolute() else ROOT / value).resolve())
    final_output = Path(args.output_dir)
    if args.use_lora:
        final_output /= f'lora_r{args.lora_r}_alpha{args.lora_alpha}'
    if final_output.exists() and any(final_output.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve existing adapters: {final_output}')
    if min(args.batch_size, args.gradient_accumulation_steps, args.num_train_epochs) < 1:
        raise ValueError('batch_size, gradient_accumulation_steps and num_train_epochs must be positive')
    if args.load_in_4bit:
        raise ValueError("This migration supports existing non-quantized LoRA runs only")
    # ========== post process
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
        if args.dataset == 'lncRNA_H':
            args.max_seq_len = 3000
        elif args.dataset == 'lncRNA_M':
            args.max_seq_len = 6000

    # ========== args check
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # ========== Build tokenizer, model, criterion
    model = None
    if args.model_name == "RNABERT":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNABERT.rnabert import BertModel
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        model = BertModel(model_config)
        if args.freeze_base and not args.use_lora:  # Leave the backbone available for LoRA injection.
            freeze(model)
        model = RNABertForSeqCls(model)
        model._load_pretrained_bert(args.LM_path)
        
        if args.use_lora:
            model = init_lora_model(model, args)
            
    elif args.model_name == "RNAMSM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        from RNAMSM.model import MSATransformer
        model_config = get_config(
            args.config_path + "{}.json".format(args.model_name))
        model = MSATransformer(**model_config)
        if args.freeze_base and not args.use_lora:
            freeze(model)
        model = RNAMsmForSeqCls(model)
        model._load_pretrained_bert(args.LM_path)
        if args.use_lora:
            model = init_lora_model(model, args)
            
    elif args.model_name == "RNAFM":
        tokenizer = RNATokenizer(args.vocab_path + "{}.txt".format(args.model_name))
        import RNAFM.fm as fm
        model, alphabet = fm.pretrained.rna_fm_t12()
        if args.freeze_base and not args.use_lora:
            freeze(model)
        model = RNAFmForSeqCls(model)
        if args.use_lora:
            model = init_lora_model(model, args)
            
    elif args.model_name == "structRFM":
        from_pretrained = args.LM_path
        model_paras = get_model_scale(args.model_scale)
        
        if args.max_seq_len+2>514:
            tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len+2)
            model = get_structRFM_for_cls(
                num_class=len(LABEL2ID[args.dataset]), 
                from_pretrained=from_pretrained, 
                tokenizer=tokenizer, 
                pretrained_length=514, 
                freeze_base=args.freeze_base and not args.use_lora,  # Leave the backbone available for LoRA injection.
                use_mean_feature=args.use_mean_feature, 
                **model_paras
            )
        else:
            if args.use_automodelforseqcls:
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    pretrained_model_name_or_path=from_pretrained, 
                    num_labels=len(LABEL2ID[args.dataset])
                )
                if args.freeze_base and not args.use_lora:
                    freeze(model.bert.encoder)  # Freeze only the encoder.
            else:
                
                model = get_structRFM_for_cls(
                    num_class=len(LABEL2ID[args.dataset]), 
                    from_pretrained=from_pretrained, 
                    freeze_base=args.freeze_base and not args.use_lora,
                    use_mean_feature=args.use_mean_feature, 
                    **model_paras
                )
                
            tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        
        # Apply LoRA adapters.
        print('before lora')
        pass
        if args.use_lora:
            model = init_lora_model(model, args)
    else:
        raise ValueError("Unknown model name: {}".format(args.model_name))

    
    args.device = torch.device(args.device)
    
    model.to(args.device)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"\n===== Parameters =====")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable fraction: {trainable_params/total_params*100:.4f}%")
    print('\nModel structure:')
    pass
    
    
    _loss_fn = SeqClsLoss().to(args.device)
    
    
    if args.output_dir is None:
        if args.model_name=='structRFM' and args.LM_path and os.path.exists(args.LM_path):
            run_name = os.path.basename(args.LM_path)
            run_name = run_name[:run_name.rfind('.')]
        else:
            run_name = args.model_name
        args.output_dir = os.path.join('outputs', run_name)
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_lora:
        args.output_dir = os.path.join(args.output_dir, f"lora_r{args.lora_r}_alpha{args.lora_alpha}")
        os.makedirs(args.output_dir, exist_ok=True)

    
    dataset_train = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer)
    dataset_eval = SeqClsDataset(fasta_dir=args.dataset_dir, prefix=args.dataset, tokenizer=tokenizer, train=False)
    print('\n===== Dataset sizes =====')
    print(f'dataset train: {len(dataset_train)}, test: {len(dataset_eval)}') 

    
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len, tokenizer=tokenizer,
        label2id=LABEL2ID[args.dataset], replace_T=args.replace_T, replace_U=args.replace_U, is_structRFM=args.model_name.startswith('structRFM'))

    # Training arguments.
    optimizer = AdamW(
        params=[para for para in model.parameters() if para.requires_grad], 
        lr=args.learning_rate,
        weight_decay=0.01  
    )

    
    _metric = SeqClsMetrics(metrics=args.metrics)

    # ========== Trainer ==========
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
        fit_with_efficiency(seq_cls_trainer, total_params, trainable_params)
        
        print(f"\nTraining complete. Best metric: {seq_cls_trainer.max_metric:.4f}")
