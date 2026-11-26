# 修改原因：保留第四版，新增学生全量微调和按步评估入口；修改时间：2026-10-15
# 修改原因：独立第四版测试集选模诊断，保留原版与规范验证集实验；修改时间：2026-10-15
import os
import argparse
from datetime import datetime
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

# ========== 新增LoRA相关导入 ==========
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


import sys
# 修改原因：不依赖启动目录，确保导入本项目；修改时间：2026-10-15
from pathlib import Path
# 修改原因：迁移到 Windows 时按项目根目录解析路径，保留原实验计算；修改时间：2026-11-09
from portable_paths import project_path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.structRFM.model import get_model_scale, get_structRFM_for_cls as get_cls_model
from ires_mean_model import get_ires_mean_model
from src.structRFM.data import get_mlm_tokenizer

# 修改原因：新版入口仅引用修复副本，旧入口行为保持原样；修改时间：2026-10-15
from utils import str2bool, str2list
from collators import SeqClsCollator
from losses import SeqClsLoss
from metrics import SeqClsMetrics
from trainers import SeqClsTrainer
from RNAdata import IRESDataset
# ========== Define constants ==========
MODELS = ["structRFM"]  # 仅保留structRFM
MAX_SEQ_LEN = {"structRFM": 512}
TASKS = ["IRES"]
# IRES任务标签映射（2分类）
LABEL2ID = {
    "IRES": {
        "negative": 0,
        "positive": 1
    }
}


# ========== 参数解析 ==========
def get_args():
    parser = argparse.ArgumentParser('IRES Identification with structRFM + LoRA')
    # model args
    parser.add_argument('--model_name', type=str, default="structRFM", choices=MODELS)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large','small'], default='small')
     #修改！
    parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM-EW')
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM')

    #10M学生
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_11M')

    

    
    #parser.add_argument('--LM_path', type=str, default='/root/autodl-tmp/structRFM-master2/model/DistilncRFM')
    #parser.add_argument('--LM_path', type=str,default='/root/autodl-tmp/structRFM-master2/model/structRFM_86M')
    # 数据参数
    parser.add_argument('--dataset_dir', type=str, default="data/data/ires_seq", help='IRES数据根目录')
    parser.add_argument('--dataset', type=str, default="IRES", help='数据集名称')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--replace_T', type=str2bool, default=True)
    parser.add_argument('--replace_U', type=str2bool, default=False)
    
    # 训练参数
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
    
    # LoRA参数
    parser.add_argument('--use_lora', type=str2bool, default=False)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=256)
    parser.add_argument('--lora_dropout', type=float, default=0.2)
    #parser.add_argument('--lora_target_modules', type=str, default="query,value,key,intermediate.dense,output.dense")
    parser.add_argument('--lora_target_modules', type=str, default="query,value,key,dense")
    parser.add_argument('--load_in_4bit', type=str2bool, default=False)
    parser.add_argument('--bnb_4bit_quant_type', type=str, default="nf4")
    
    # 保存参数
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save_checkpoints', type=str2bool, default=False,
                        help='Efficiency runs skip saving/restoring duplicate weights; true restores checkpoint saving.')
    parser.add_argument('--save_strategy', type=str, default='steps')
    parser.add_argument('--evaluation_strategy', type=str, default='steps')
    parser.add_argument('--load_best_model_at_end', type=str2bool, default=True)
    parser.add_argument('--metric_for_best_model', type=str, default='auc', choices=['auc'])
    parser.add_argument('--greater_is_better', type=str2bool, default=True)
    # 修改原因：原参数未实现；新版仅保存覆盖best_model，避免虚假保留策略；修改时间：2026-10-15
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pos_weight', type=float, default=2.5)
    
    args = parser.parse_args()
    return args

# ========== 工具函数 ==========
def freeze(model):
    """冻结模型所有参数"""
    for name, para in model.named_parameters():
        para.requires_grad = False

def init_lora_model(model, args):
    """适配structRFM的LoRA初始化"""
    if not args.use_lora:
        return model
    
    # 4bit训练准备
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 解析LoRA目标层
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.model_name == "structRFM" and not target_modules:
        target_modules = ["query", "value"]
    
    # LoRA配置
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
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数信息
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

# ========== 主函数 ==========
if __name__ == "__main__":
    args = get_args()
    # 修改原因：固定随机性，并拒绝没有正确实现的运行选项；修改时间：2026-10-15
    import random, numpy as np, json
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # 修改原因：记录测试集选模协议与池化方式，禁止混同独立测试；修改时间：2026-10-15
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
    
    # 修改原因：本次只运行全量微调，避免误冻结或误启用LoRA；修改时间：2026-10-15
    if args.use_lora or args.freeze_base:
        raise ValueError('第五版要求use_lora=False且freeze_base=False')
    if args.eval_steps < 1 or any(not np.isfinite(x) or x <= 0 for x in [args.learning_rate, args.head_learning_rate]):
        raise ValueError('eval_steps和两个学习率必须为正数')

    # 后处理参数
    if args.max_seq_len == 0:
        args.max_seq_len = MAX_SEQ_LEN[args.model_name]
    assert args.replace_T ^ args.replace_U, "Only replace T or U."

    # 修改原因：本任务174nt，无需旧长序列分支；固定受检长度并保存绝对路径以支持异目录评估；修改时间：2026-10-15
    if args.max_seq_len != 512:
        raise ValueError("本次修复保持max_seq_len=512；长度优化留待后续实验。")
    args.LM_path = project_path(args.LM_path)
    args.dataset_dir = project_path(args.dataset_dir)
    args.output_dir = project_path(args.output_dir or f'results/ires_full_efficiency_{datetime.now():%Y%m%d_%H%M%S_%f}')
    target = Path(args.output_dir) / args.pooling
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve old checkpoints: {target}')

    # ========== 构建模型和分词器 ==========
    from_pretrained = args.LM_path
    model_paras = get_model_scale(args.model_scale)
    print("!"*20)
    print(args.model_scale)
    print( model_paras)
    print("!"*20)
    # 适配长序列
    if args.max_seq_len + 2 > 514:
        tokenizer = get_mlm_tokenizer(max_length=args.max_seq_len + 2)
        model = get_structRFM_for_cls(
            num_class=len(LABEL2ID["IRES"]),  # 2分类
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
            num_class=len(LABEL2ID["IRES"]),  # 2分类
            from_pretrained=from_pretrained,
            tokenizer=tokenizer,
            freeze_base=args.freeze_base and not args.use_lora,
            use_mean_feature=args.use_mean_feature,
            **model_paras
        )
    
    # 应用LoRA
    print('=== 应用LoRA前的模型结构 ===')
    print(model)
    if args.use_lora:
        model = init_lora_model(model, args)

    # ========== 设备配置 ==========
    args.device = torch.device(args.device)
    model.to(args.device)
    # Single selected device, matching the other efficiency tasks.
    
    # 打印参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n===== 参数统计 =====")
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"可训练占比: {trainable_params/total_params*100:.4f}%")

    # ========== 损失函数/优化器/指标 ==========
    _loss_fn = SeqClsLoss(pos_weight=args.pos_weight).to(args.device)
    
    # 修改原因：预训练底座小步更新，随机分类头使用独立学习率；修改时间：2026-10-15
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

    # ========== 输出目录 ==========
    args.output_dir = str(Path(args.output_dir) / args.pooling)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.use_lora:
        args.output_dir = os.path.join(args.output_dir, f"lora_r{args.lora_r}_alpha{args.lora_alpha}")
        os.makedirs(args.output_dir, exist_ok=True)

    # 修改原因：新实验不能覆盖旧实验或不同配置；修改时间：2026-10-15
    if any(Path(args.output_dir).iterdir()):
        raise FileExistsError(f"输出目录非空，请指定新的--output_dir: {args.output_dir}")
    # ========== 加载数据集（核心修改） ==========
    dataset_train = IRESDataset(
    fasta_dir=args.dataset_dir,  # 指向 ires_seq 文件夹
    prefix="",                   # 留空（IRES目录结构不需要prefix）
    tokenizer=tokenizer,
    seed=0,
    train=True
    )

    # 修改原因：诊断使用完整原训练集，测试集仅评估和选模；修改时间：2026-10-15
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

    # ========== 数据Collator ==========
    _collate_fn = SeqClsCollator(
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer,
        label2id=LABEL2ID["IRES"],
        replace_T=args.replace_T,
        replace_U=args.replace_U,
        is_structRFM=True  # 适配structRFM
    )

    # ========== 训练器 ==========
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

    # ========== 训练/验证 ==========
    if args.train:
        seq_cls_trainer.fit()
