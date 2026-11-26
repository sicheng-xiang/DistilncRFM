import os
import random
import argparse
from datetime import datetime
# Use tanh for tree-based adaptive weights.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboard
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from .model import get_structRFM, get_llama_causal_model, get_model_scale
from .data import get_mlm_tokenizer, get_ar_tokenizer, preprocess_and_load_dataset
import warnings
from transformers import EarlyStoppingCallback

# Suppress redundant warnings.
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="barrier\(\): using the device under current context")
warnings.filterwarnings("ignore", category=UserWarning, message="Guessing device ID based on global rank")

class DistillMLMTrainer(Trainer):
    def __init__(
        self,
        teacher_model_path,
        distill_temperature=2.0,
        
        
        use_tree_adaptive_weight=False,
        **kwargs
    ):
        """Initialize MLM and feature, soft-label, and relation distillation. teacher_model_path selects the pretrained teacher; distill_temperature controls soft-label KL. use_tree_adaptive_weight enables tree-constrained adaptive weights; otherwise use equal weights."""
        super().__init__(**kwargs)

        # Load the 768-dimensional teacher.
        self.teacher_tokenizer = get_mlm_tokenizer(max_length=514)
        self.teacher_model = get_structRFM(
            dim=768,
            layer=12,
            num_attention_heads=12,
            from_pretrained=teacher_model_path,
            tokenizer=self.teacher_tokenizer,
            max_length=514,
            output_hidden_states=True
        )

        # Freeze the teacher, use evaluation mode, and align its device.
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.args.local_rank != -1
        )

        self.teacher_model = self.teacher_model.to(self.args.device)

        if self.is_distributed:
            torch.distributed.barrier()
            self.teacher_model = self._sync_teacher_model_across_processes()

        # Project student features from 384 to 768 dimensions.
        
        
        
        if hasattr(self.model, 'feature_up_sample'):
            self.model.feature_up_sample = self.model.feature_up_sample.to(self.args.device)
            print("\n✅ 检测到已有 feature_up_sample，直接继承 checkpoint 中的权重，不重新初始化")
        else:
            # Create the projection only if it is absent.
            self.model.feature_up_sample = nn.Linear(384, 768).to(self.args.device)
            print("\n⚠️ 模型中不存在 feature_up_sample，已重新创建 nn.Linear(384, 768)")

        self.feature_up_sample = self.model.feature_up_sample

        # Define the loss functions.
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.feature_loss_fct = nn.MSELoss(reduction='mean')
        self.soft_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.global_sim_loss_fct = nn.MSELoss(reduction='mean')

        # Fixed distillation temperature.
        self.distill_temperature = distill_temperature
        self.device = self.args.device

        #！！============================================================
        # Initialize tree-based adaptive distillation weights.
        #！！============================================================

        self.use_tree_adaptive_weight = use_tree_adaptive_weight

        # Use the first 50 training batches to establish a fixed baseline.
        self.adaptive_warmup_batches = 50

        # EMA smoothing coefficient.
        self.adaptive_ema_beta = 0.99

        # Prevent division by zero.
        self.adaptive_eps = 1e-8

        # The feature root has two branches: relation and output (degree 2).
        self.adaptive_root_degree = 2

        
        # Count training batches only, excluding evaluation.
        self.adaptive_train_batch_count = 0

        # Accumulate unweighted KD losses during baseline warmup.
        self.adaptive_baseline_sum = {
            'feature': 0.0,
            'relation': 0.0,
            'output': 0.0,
        }

        # Compute each baseline once, after warmup.
        # Keep the baseline fixed after initialization.
        self.adaptive_baseline = {
            'feature': None,
            'relation': None,
            'output': None,
        }

        # Update smoothed losses on every training batch.
        self.adaptive_ema = {
            'feature': None,
            'relation': None,
            'output': None,
        }

        # Normalized loss state q_i.
        self.adaptive_q = {
            'feature': 1.0,
            'relation': 1.0,
            'output': 1.0,
        }

        # Budget offsets for the two tree edges.
        self.adaptive_tau = {
            'relation': 0.0,
            'output': 0.0,
        }

        # KD weights used for the current training batch.
        # Start with equal KD weights (1:1:1).
        self.adaptive_lambda = {
            'feature': 1.0,
            'relation': 1.0,
            'output': 1.0,
        }

        if self.use_tree_adaptive_weight:
            print("\n" + "=" * 70)
            print("🌳 树状自适应蒸馏权重已启用")
            print(f"   Baseline初始化batch数: {self.adaptive_warmup_batches}")
            print(f"   EMA beta: {self.adaptive_ema_beta}")
            print(f"   Feature根节点degree: {self.adaptive_root_degree}")
            print("   初始权重: Feature=1.0, Relation=1.0, Output=1.0")
            print("   MLM权重始终固定为1.0")
            print("=" * 70 + "\n")

        # Monitor parameter and adaptive-weight changes.
        self.weight_monitor = {
            'initial_weight': self.feature_up_sample.weight.data.clone().detach(),
            'initial_bias': (
                self.feature_up_sample.bias.data.clone().detach()
                if self.feature_up_sample.bias is not None
                else None
            ),
            'weight_updates': [],
            'grad_norms': [],
            'has_computed_loss': False,
        }

        print("\n" + "=" * 70)
        print("✅ 权重监控已启动（FIX版本 - 上采样层已加入模型）")
        print(
            f"   初始 feature_up_sample weight shape: "
            f"{self.weight_monitor['initial_weight'].shape}"
        )
        print(
            f"   初始 weight 范围: "
            f"[{self.weight_monitor['initial_weight'].min():.6f}, "
            f"{self.weight_monitor['initial_weight'].max():.6f}]"
        )
        print(f"   feature_up_sample 已添加到 model.feature_up_sample")
        print(f"   ✅ 此层现在将被优化器追踪！")
        print("=" * 70 + "\n")

    def _sync_teacher_model_across_processes(self):
        """Synchronize teacher parameters across distributed processes."""
        for param in self.teacher_model.parameters():
            torch.distributed.broadcast(param.data, src=0)
        return self.teacher_model

    #！！============================================================
    
    # Keep local losses unchanged for backpropagation.
    #！！============================================================
    def _get_adaptive_stat_value(self, loss):
        value = loss.detach().float().clone()

        if self.is_distributed:
            torch.distributed.all_reduce(
                value,
                op=torch.distributed.ReduceOp.SUM
            )
            value = value / torch.distributed.get_world_size()

        return value.item()

    #！！============================================================
    
    #！！
    
    # Inputs are unweighted losses.
    # Compute each baseline once and keep it fixed.
    # Update the EMA on each training batch.
    # Use the updated state to determine the next batch's weights.
    # Recompute weights from one rather than accumulating offsets.
    #！！============================================================
    def _update_tree_adaptive_state(
        self,
        feature_loss,
        relation_loss,
        output_loss
    ):
        if not self.use_tree_adaptive_weight:
            return

        # Obtain globally averaged unweighted losses.
        feature_value = self._get_adaptive_stat_value(feature_loss)
        relation_value = self._get_adaptive_stat_value(relation_loss)
        output_value = self._get_adaptive_stat_value(output_loss)

        current_losses = {
            'feature': feature_value,
            'relation': relation_value,
            'output': output_value,
        }

        
        # Validate after aggregation and before changing state on every DDP rank.
        if not all(math.isfinite(value) for value in current_losses.values()):
            raise FloatingPointError(
                f"KD loss 出现 NaN/Inf，已停止更新自适应状态：{current_losses}"
            )

        # Update statistics only for training batches.
        self.adaptive_train_batch_count += 1
        t = self.adaptive_train_batch_count

        #！！--------------------------------------------------------
        # Warmup: use equal weights while collecting baseline losses.
        #！！--------------------------------------------------------
        if t <= self.adaptive_warmup_batches:

            for key in current_losses:
                self.adaptive_baseline_sum[key] += current_losses[key]

            # Initialize fixed baselines at the end of warmup.
            if t == self.adaptive_warmup_batches:

                for key in current_losses:
                    baseline = (
                        self.adaptive_baseline_sum[key]
                        / self.adaptive_warmup_batches
                    )

                    # Keep baselines fixed from this point onward.
                    self.adaptive_baseline[key] = baseline

                    # Initialize each EMA from its baseline.
                    self.adaptive_ema[key] = baseline

                    # Initialize normalized losses to one.
                    self.adaptive_q[key] = 1.0

                print("\n" + "=" * 70)
                print("🌳 Adaptive baseline 初始化完成")
                print(
                    f"   B_feature = "
                    f"{self.adaptive_baseline['feature']:.8f}"
                )
                print(
                    f"   B_relation = "
                    f"{self.adaptive_baseline['relation']:.8f}"
                )
                print(
                    f"   B_output = "
                    f"{self.adaptive_baseline['output']:.8f}"
                )
                print("   注意：三个B_i从现在开始永久固定，不再更新")
                print("=" * 70 + "\n")

            # Keep equal weights throughout warmup.
            self.adaptive_lambda['feature'] = 1.0
            self.adaptive_lambda['relation'] = 1.0
            self.adaptive_lambda['output'] = 1.0

            self.adaptive_tau['relation'] = 0.0
            self.adaptive_tau['output'] = 0.0

            return

        #！！--------------------------------------------------------
        # After warmup, update the EMA and adaptive weights.
        #！！--------------------------------------------------------

        beta = self.adaptive_ema_beta

        # Update the EMA on each training batch.
        for key in current_losses:
            self.adaptive_ema[key] = (
                beta * self.adaptive_ema[key]
                + (1.0 - beta) * current_losses[key]
            )

        # q_i = smoothed loss / fixed baseline.
        for key in current_losses:
            self.adaptive_q[key] = (
                self.adaptive_ema[key]
                / (
                    self.adaptive_baseline[key]
                    + self.adaptive_eps
                )
            )

        q_f = self.adaptive_q['feature']
        q_r = self.adaptive_q['relation']
        q_o = self.adaptive_q['output']

        # Compare only the two edges in the knowledge tree.
        #！！Feature <-> Relation
        #！！Feature <-> Output
        delta_r = q_r - q_f
        delta_o = q_o - q_f

        # Root degree 2 limits each edge's budget capacity to 1/2.
        edge_capacity = 1.0 / self.adaptive_root_degree

        # Compute budget offsets along the tree edges.
        tau_r = edge_capacity * math.tanh(delta_r)
        tau_o = edge_capacity * math.tanh(delta_o)

        self.adaptive_tau['relation'] = tau_r
        self.adaptive_tau['output'] = tau_o

        # Recompute from the equal-weight baseline of one.
        # Do not accumulate offsets across batches.
        lambda_r = 1.0 + tau_r
        lambda_o = 1.0 + tau_o
        lambda_f = 1.0 - tau_r - tau_o

        self.adaptive_lambda['feature'] = lambda_f
        self.adaptive_lambda['relation'] = lambda_r
        self.adaptive_lambda['output'] = lambda_o

    def _compute_cosine_sim_matrix(self, features):
        """Compute pairwise cosine similarities from [batch, length, dim] features."""
        
        # Disable autocast locally and compute cosine similarity in FP32.
        # Casting alone is insufficient: autocast may lower matmul precision.
        with torch.autocast(device_type=features.device.type, enabled=False):
            features_fp32 = features.float()
            dot_product = torch.matmul(
                features_fp32,
                features_fp32.transpose(-1, -2)
            )
            norm = torch.norm(features_fp32, p=2, dim=-1, keepdim=True)
            norm_product = torch.matmul(norm, norm.transpose(-1, -2))
            norm_product = torch.clamp(norm_product, min=1e-8)
            sim_matrix = torch.clamp(
                dot_product / norm_product, min=-1.0, max=1.0
            )

        return sim_matrix

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):
        """Combine MLM, feature, soft-label, and global relation distillation losses."""

        if self.state.global_step == 0:
            self.show_optimizer_params()

        # Align model and input devices.
        labels = inputs.pop("labels").to(self.device)
        student_inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        # Run the student forward pass.
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        student_last_hidden = student_outputs.hidden_states[-1]

        # Run the teacher forward pass.
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**student_inputs)
            teacher_logits = teacher_outputs.logits
            teacher_last_hidden = teacher_outputs.hidden_states[-1]

        # Separate masked and visible tokens.
        mask_pos = labels.view(-1) != -100
        visible_pos = labels.view(-1) == -100

        # Initialize loss components.
        mlm_loss = torch.tensor(
            0.0,
            device=self.device
        )
        feature_loss = torch.tensor(
            0.0,
            device=self.device
        )
        soft_loss = torch.tensor(
            0.0,
            device=self.device
        )
        global_sim_loss = torch.tensor(
            0.0,
            device=self.device
        )

        # Masked tokens: MLM and soft-label distillation.
        if mask_pos.any():

            student_logits_mask = student_logits.view(
                -1,
                student_logits.size(-1)
            )[mask_pos]

            labels_mask = labels.view(-1)[mask_pos]

            mlm_loss = self.mlm_loss_fct(
                student_logits_mask,
                labels_mask
            )

            student_probs = F.log_softmax(
                student_logits_mask
                / self.distill_temperature,
                dim=-1
            )

            teacher_probs = F.softmax(
                teacher_logits.view(
                    -1,
                    teacher_logits.size(-1)
                )[mask_pos]
                / self.distill_temperature,
                dim=-1
            )

            soft_loss = self.soft_loss_fct(
                student_probs,
                teacher_probs
            ) * (self.distill_temperature ** 2)

        else:
            mlm_loss = torch.tensor(
                1e-4,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True
            )

            soft_loss = torch.tensor(
                1e-4,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True
            )

        # Visible tokens: feature distillation.
        if visible_pos.any():

            student_hidden_up = self.feature_up_sample(
                student_last_hidden
            )

            student_hidden_visible = student_hidden_up.view(
                -1,
                768
            )[visible_pos]

            teacher_hidden_visible = teacher_last_hidden.view(
                -1,
                768
            )[visible_pos]

            feature_loss = self.feature_loss_fct(
                student_hidden_visible,
                teacher_hidden_visible
            )

        else:
            feature_loss = torch.tensor(
                1e-4,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True
            )

        # Global relation distillation loss.
        student_sim_matrix = self._compute_cosine_sim_matrix(
            student_last_hidden
        )

        teacher_sim_matrix = self._compute_cosine_sim_matrix(
            teacher_last_hidden
        )

        global_sim_loss = self.global_sim_loss_fct(
            student_sim_matrix,
            teacher_sim_matrix
        )

        #！！========================================================
        # Select the weights for the current batch.
        #！！
        
        
        
        #！！
        
        # Use equal distillation weights during evaluation.
        # Do not update baselines, EMA state, or weights during evaluation.
        
        #！！========================================================
        if self.use_tree_adaptive_weight and model.training:
            lambda_f_used = self.adaptive_lambda['feature']
            lambda_r_used = self.adaptive_lambda['relation']
            lambda_o_used = self.adaptive_lambda['output']
        else:
            lambda_f_used = 1.0
            lambda_r_used = 1.0
            lambda_o_used = 1.0

        #！！========================================================
        # Combine the loss components.
        
        # Keep the MLM loss weight fixed at one.
        #！！========================================================
        total_loss = (
            mlm_loss
            + lambda_f_used * feature_loss
            + lambda_o_used * soft_loss
            + lambda_r_used * global_sim_loss
        )

        
        # Synchronize invalid-loss flags to prevent DDP collective hangs.
        # Check evaluation losses locally without extra cross-rank synchronization.
        loss_values = torch.stack([
            mlm_loss.detach().float(), feature_loss.detach().float(),
            soft_loss.detach().float(), global_sim_loss.detach().float(),
            total_loss.detach().float(),
        ])
        losses_finite = torch.isfinite(loss_values).all().to(dtype=torch.int32)
        if self.is_distributed and model.training:
            torch.distributed.all_reduce(
                losses_finite, op=torch.distributed.ReduceOp.MIN
            )
        if not losses_finite.item():
            raise FloatingPointError(
                "检测到本卡或其他训练卡的 loss 为 NaN/Inf，停止本次运行；"
                "本 batch 不更新自适应状态、不执行反向传播。"
            )

        #！！========================================================
        
        # Use the updated state to determine the next batch's weights.
        #！！
        # Update statistics from raw losses, not weighted losses.
        # Use the original unweighted loss L_i.
        #！！========================================================
        if self.use_tree_adaptive_weight and model.training:
            self._update_tree_adaptive_state(
                feature_loss=feature_loss,
                relation_loss=global_sim_loss,
                output_loss=soft_loss
            )

        
        if (
            feature_loss.item() > 0
            and self.weight_monitor.get(
                'has_computed_loss'
            ) == False
        ):
            self.weight_monitor['has_computed_loss'] = True

            print("\n" + "=" * 60)
            print("✅ 第一次计算 feature_loss，说明上采样层正在使用！")
            print(
                f"   feature_loss = "
                f"{feature_loss.item():.6f}"
            )
            print(
                f"   feature_up_sample.weight.requires_grad = "
                f"{self.feature_up_sample.weight.requires_grad}"
            )
            print(
                f"   feature_up_sample 在计算图中: "
                f"{self.feature_up_sample.weight.is_leaf}"
            )
            print("=" * 60 + "\n")

        
        if (
            self.state.global_step % 50 == 0
            and is_main_process(self.args.local_rank)
        ):
            if self.feature_up_sample.weight.grad is not None:

                grad_norm = torch.norm(
                    self.feature_up_sample.weight.grad
                ).item()

                self.weight_monitor['grad_norms'].append({
                    'step': self.state.global_step,
                    'grad_norm': grad_norm
                })

                weight_change = torch.norm(
                    self.feature_up_sample.weight
                    - self.weight_monitor['initial_weight']
                ).item()

                self.weight_monitor['weight_updates'].append({
                    'step': self.state.global_step,
                    'weight_change': weight_change,
                    'current_weight_range': [
                        self.feature_up_sample.weight.min().item(),
                        self.feature_up_sample.weight.max().item()
                    ]
                })

        
        if (
            self.state.global_step % 500 == 0
            and is_main_process(self.args.local_rank)
        ):
            debug_info = {
                "mlm_loss": mlm_loss.item(),
                "feature_loss": feature_loss.item(),
                "soft_loss": soft_loss.item(),
                "global_sim_loss": global_sim_loss.item(),
                "total_loss": total_loss.item(),
                "gpu_rank": self.args.local_rank
            }

            
            if self.use_tree_adaptive_weight:

                # Weights used by the current batch.
                debug_info.update({
                    "lambda_feature_used": lambda_f_used,
                    "lambda_relation_used": lambda_r_used,
                    "lambda_output_used": lambda_o_used,
                })

                # Use the updated state to determine the next batch's weights.
                debug_info.update({
                    "lambda_feature_next":
                        self.adaptive_lambda['feature'],
                    "lambda_relation_next":
                        self.adaptive_lambda['relation'],
                    "lambda_output_next":
                        self.adaptive_lambda['output'],
                })

                debug_info.update({
                    "q_feature":
                        self.adaptive_q['feature'],
                    "q_relation":
                        self.adaptive_q['relation'],
                    "q_output":
                        self.adaptive_q['output'],
                    "tau_relation":
                        self.adaptive_tau['relation'],
                    "tau_output":
                        self.adaptive_tau['output'],
                    "adaptive_batch":
                        self.adaptive_train_batch_count,
                })

            self.log(debug_info)

        return (
            (total_loss, student_outputs)
            if return_outputs
            else total_loss
        )

    # Inspect parameters included in the optimizer.
    def show_optimizer_params(self):
        print("\n===== 优化器中可训练的参数列表 =====")

        for i, param_group in enumerate(
            self.optimizer.param_groups
        ):
            print(f"Param group {i}:")

            for j, p in enumerate(
                param_group['params']
            ):
                if p.requires_grad:
                    for name, param in self.model.named_parameters():
                        if p is param:
                            print(
                                f"  ✅ 可训练: "
                                f"{name} | shape={p.shape}"
                            )

        print("===================================\n")
 

def set_seed(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def parse_args():
    parser = argparse.ArgumentParser(description='RNA LM with Hybrid Distillation')
    parser.add_argument('--run_name', type=str, default='autodl-tmp/structRFM-master2/model2_f_l_s_0')
    parser.add_argument('--data_path', type=str, default='data/pretraining/RNAcentral_512_SS_connects.csv')
    parser.add_argument('--tag', type=str, choices=['mlm', 'ar'], default='mlm')
    parser.add_argument('--max_length', type=int, default=514)
    parser.add_argument('--seed', type=int, default=42)

    # Model dimensions: student 384, teacher 768.
    parser.add_argument('--dim', type=int, default=384, help='学生模型hidden dim（384）')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--num_attention_heads', type=int, default=6)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large', 'custom', 'small'], default='small')
    parser.add_argument('--from_pretrained', type=str, help='for model')
    parser.add_argument('--resume_from_checkpoint', type=str)

    # Training arguments.
    parser.add_argument('--use_DDP', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mlm_structure', action='store_true',
                        help=argparse.SUPPRESS)

    # Distillation arguments.
    parser.add_argument('--teacher_model_path', type=str, default='model/structRFM_86M', help='预训练教师模型路径（768维）')
    parser.add_argument('--distill_temperature', type=float, default=2.0, help='软标签蒸馏温度（固定值）')
    parser.add_argument('--use_tree_adaptive_weight', action='store_true',
                        help='启用树状自适应蒸馏权重；默认使用等权蒸馏')

    args = parser.parse_args()
    if args.mlm_structure:
        parser.error('This entry point uses standard MLM masking. Remove --mlm_structure.')
    if args.use_tree_adaptive_weight and (args.tag != 'mlm' or not args.teacher_model_path):
        parser.error('--use_tree_adaptive_weight requires --tag mlm and a teacher checkpoint')
    return args

def pretrain(args, tag):
    model_scale_paras = get_model_scale(args.model_scale)
    print('model_scale', args.model_scale, model_scale_paras)
    for k, v in model_scale_paras.items():
        setattr(args, k, v)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    set_seed(args.seed)
    if tag == 'mlm':
        tokenizer = get_mlm_tokenizer(max_length=args.max_length)
    elif tag == 'ar':
        tokenizer = get_ar_tokenizer(max_length=args.max_length)
        
    model = None
    if tag == 'mlm':
        # Enable hidden-state outputs for the 384-dimensional student.
        model = get_structRFM(dim=args.dim, layer=args.layer, num_attention_heads=args.num_attention_heads,
                              from_pretrained=None, tokenizer=tokenizer, max_length=args.max_length,  output_hidden_states=True)
        model_name = 'structRFM'
    else:
        model = get_llama_causal_model(args.dim, args.layer, args.from_pretrained, tokenizer=tokenizer,
                                       max_length=args.max_length,  output_hidden_states=True)
        model_name = 'llama'
    model_param_size = sum(t.numel() for t in model.parameters())
    print(model)
    print(f"{model_name} model paras: {model_param_size / 1e6:.1f}M")

    dataset = preprocess_and_load_dataset(args.data_path, tokenizer, tag, with_structure=False)
    split_dataset = dataset
    if 'test' not in split_dataset:
        if 'validate' in dataset:
            split_dataset['test'] = split_dataset['validate']
        else:
            split_dataset = dataset['train'].train_test_split(test_size=0.05, seed=args.seed)
    print(split_dataset)

    
    print("="*50)
    print("数据集各子集条数统计：")
    for split_name in split_dataset.keys():
        subset_size = len(split_dataset[split_name])
        print(f"- {split_name} 集：{subset_size} 条数据")
    total_size = sum(len(split_dataset[split]) for split in split_dataset.keys())
    print(f"- 数据集总条数：{total_size} 条")
    print("="*50)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=tag == 'mlm')
    if tag == 'mlm':
        print('MLM masking: standard random masking')

    
    

    training_args = TrainingArguments(
        output_dir=args.run_name,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        weight_decay=0.,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=args.batch_size,
        ddp_find_unused_parameters=False,
        logging_strategy="steps",
        logging_steps=5000,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=True,
        report_to="tensorboard",
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    )
    
    # Early stopping callback.
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )
    my_callbacks = [early_stopping_callback]

    if tag == 'mlm' and args.teacher_model_path:
        
        trainer = DistillMLMTrainer(
            teacher_model_path=args.teacher_model_path,
            distill_temperature=args.distill_temperature,  # Fixed distillation temperature.
            use_tree_adaptive_weight=args.use_tree_adaptive_weight,
            model=model, 
            args=training_args,
            train_dataset=split_dataset["train"], 
            eval_dataset=split_dataset["test"],
            data_collator=data_collator, 
            tokenizer=tokenizer, 
            callbacks=my_callbacks
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=my_callbacks,
        )

    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        print(f'Resume_from_checkpoint {args.resume_from_checkpoint}')
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elif any(f.startswith('checkpoint') for f in os.listdir(args.run_name)):
        print(f'Resume_from_checkpoint...')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    
    if hasattr(trainer, 'weight_monitor'):
        print("\n" + "="*70)
        print("🎯 训练完成 - Feature Upsampler 权重变化报告")
        print("="*70)

        monitor = trainer.weight_monitor

        
        final_weight = trainer.model.feature_up_sample.weight.data
        total_weight_change = torch.norm(final_weight - monitor['initial_weight']).item()

        print(f"\n📊 权重变化统计：")
        print(f"   初始权重范围: [{monitor['initial_weight'].min():.6f}, {monitor['initial_weight'].max():.6f}]")
        print(f"   最终权重范围: [{final_weight.min():.6f}, {final_weight.max():.6f}]")
        print(f"   总权重变化量 (L2范数): {total_weight_change:.6f}")
        print(f"   权重是否变化: {'✅ YES - 权重已更新（修复成功！）' if total_weight_change > 1e-6 else '❌ NO - 权重未更新'}")

        if len(monitor['weight_updates']) > 0:
            print(f"\n📈 监控记录（每50步）：")
            print(f"   {'Step':<8} {'Weight Change':<15} {'Weight Range':<30}")
            print(f"   {'-'*55}")
            for record in monitor['weight_updates'][-10:]:  # Print the last ten records.
                w_min, w_max = record['current_weight_range']
                print(f"   {record['step']:<8} {record['weight_change']:<15.6f} [{w_min:.6f}, {w_max:.6f}]")

        if len(monitor['grad_norms']) > 0:
            print(f"\n🔄 梯度范数（每50步）：")
            print(f"   {'Step':<8} {'Grad Norm':<15}")
            print(f"   {'-'*25}")
            for record in monitor['grad_norms'][-10:]:  # Print the last ten records.
                print(f"   {record['step']:<8} {record['grad_norm']:<15.6f}")

        print("\n" + "="*70 + "\n")


def run_pretrain():
    print(f'Begin time: {datetime.now()}')
    args = parse_args()
    assert args.tag in {'mlm', 'ar'}, f'tag={args.tag} should be "mlm" or "ar"'
    pretrain(args, args.tag)
    print(f'End time: {datetime.now()}')

if __name__ == '__main__':
    run_pretrain()
