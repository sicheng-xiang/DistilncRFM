import os
import random
import argparse
from datetime import datetime
#！！树状动态权重中使用 tanh
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

# 忽略冗余警告
warnings.filterwarnings("ignore", category=FutureWarning, message="`tokenizer` is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="barrier\(\): using the device under current context")
warnings.filterwarnings("ignore", category=UserWarning, message="Guessing device ID based on global rank")

class DistillMLMTrainer(Trainer):
    def __init__(
        self,
        teacher_model_path,
        distill_temperature=2.0,
        #！！新增：是否启用树状自适应蒸馏权重。
        #！！默认 False，因此原来的30epoch/普通训练逻辑完全不受影响。
        use_tree_adaptive_weight=False,
        **kwargs
    ):
        """
        整合版蒸馏Trainer：
        MLM损失 + 特征蒸馏损失 + 软标签蒸馏损失 + 全局关联蒸馏损失

        :param teacher_model_path: 预训练教师模型路径
        :param distill_temperature: 蒸馏温度（用于软标签KL损失）
        :param use_tree_adaptive_weight:
            False -> 原始四目标等权训练
            True  -> 启用树状结构约束的动态蒸馏权重
        :param kwargs: Trainer原有参数
        """
        super().__init__(**kwargs)

        # 1. 加载教师模型（768维）
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

        # 教师模型配置：评估模式+冻结+设备对齐
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

        # 2. 学生特征上采样层（384→768）
        #！！关键修改：
        #！！如果40epoch模型中已经存在训练好的feature_up_sample，
        #！！这里直接复用，绝不能重新随机初始化覆盖。
        if hasattr(self.model, 'feature_up_sample'):
            self.model.feature_up_sample = self.model.feature_up_sample.to(self.args.device)
            print("\n✅ 检测到已有 feature_up_sample，直接继承 checkpoint 中的权重，不重新初始化")
        else:
            #！！只有模型本身不存在该层时才新建。
            self.model.feature_up_sample = nn.Linear(384, 768).to(self.args.device)
            print("\n⚠️ 模型中不存在 feature_up_sample，已重新创建 nn.Linear(384, 768)")

        self.feature_up_sample = self.model.feature_up_sample

        # 3. 损失函数定义
        self.mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.feature_loss_fct = nn.MSELoss(reduction='mean')
        self.soft_loss_fct = nn.KLDivLoss(reduction='batchmean')
        self.global_sim_loss_fct = nn.MSELoss(reduction='mean')

        # 4. 蒸馏温度（固定值）
        self.distill_temperature = distill_temperature
        self.device = self.args.device

        #！！============================================================
        #！！5. 树状自适应蒸馏权重初始化
        #！！============================================================

        self.use_tree_adaptive_weight = use_tree_adaptive_weight

        #！！前50个训练batch用于建立固定baseline。
        self.adaptive_warmup_batches = 50

        #！！EMA平滑系数。
        self.adaptive_ema_beta = 0.99

        #！！仅防止除零。
        self.adaptive_eps = 1e-8

        #！！Feature根节点连接Relation和Output两个分支，因此d_f=2。
        self.adaptive_root_degree = 2

        #！！记录“本次新的adaptive训练阶段”已经经历多少个训练batch。
        #！！只统计training batch，不统计evaluation batch。
        self.adaptive_train_batch_count = 0

        #！！前50个batch累计原始未加权KD loss，用于建立baseline。
        self.adaptive_baseline_sum = {
            'feature': 0.0,
            'relation': 0.0,
            'output': 0.0,
        }

        #！！B_i只会在前50batch结束时计算一次。
        #！！一旦建立，此后整个训练过程永久不更新。
        self.adaptive_baseline = {
            'feature': None,
            'relation': None,
            'output': None,
        }

        #！！EMA平滑loss，每个训练batch都会更新。
        self.adaptive_ema = {
            'feature': None,
            'relation': None,
            'output': None,
        }

        #！！当前归一化状态q_i。
        self.adaptive_q = {
            'feature': 1.0,
            'relation': 1.0,
            'output': 1.0,
        }

        #！！两条树边上的预算偏移tau。
        self.adaptive_tau = {
            'relation': 0.0,
            'output': 0.0,
        }

        #！！当前训练batch使用的三个KD权重。
        #！！初始阶段完全保持原来的1:1:1。
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

        # 6. 权重监控（诊断用）
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
        """同步教师模型参数到所有进程"""
        for param in self.teacher_model.parameters():
            torch.distributed.broadcast(param.data, src=0)
        return self.teacher_model

    #！！============================================================
    #！！新增：DDP情况下，将各GPU上的raw loss求平均。
    #！！这只用于动态权重统计，不影响各GPU自己的反向传播loss。
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
    #！！新增：更新树状自适应权重状态。
    #！！
    #！！注意：
    #！！1. 输入的都是原始未加权loss。
    #！！2. B_i只计算一次，以后永久固定。
    #！！3. EMA每个训练batch更新。
    #！！4. 当前batch计算出的状态决定“下一batch”的权重。
    #！！5. lambda每次从1重新计算，不在旧lambda上累加。
    #！！============================================================
    def _update_tree_adaptive_state(
        self,
        feature_loss,
        relation_loss,
        output_loss
    ):
        if not self.use_tree_adaptive_weight:
            return

        #！！先取得用于统计的全局平均raw loss。
        feature_value = self._get_adaptive_stat_value(feature_loss)
        relation_value = self._get_adaptive_stat_value(relation_loss)
        output_value = self._get_adaptive_stat_value(output_loss)

        current_losses = {
            'feature': feature_value,
            'relation': relation_value,
            'output': output_value,
        }

        # 修改原因：禁止 NaN/Inf 污染固定 baseline、EMA 和后续权重；修改时间：2026-10-09。
        # 在全卡统计完成后、任何状态修改之前检查，使 DDP 各卡一致报错。
        if not all(math.isfinite(value) for value in current_losses.values()):
            raise FloatingPointError(
                f"KD loss 出现 NaN/Inf，已停止更新自适应状态：{current_losses}"
            )

        #！！只统计真正的training batch。
        self.adaptive_train_batch_count += 1
        t = self.adaptive_train_batch_count

        #！！--------------------------------------------------------
        #！！阶段1：前W个batch维持1:1:1，只统计baseline。
        #！！--------------------------------------------------------
        if t <= self.adaptive_warmup_batches:

            for key in current_losses:
                self.adaptive_baseline_sum[key] += current_losses[key]

            #！！第W个batch结束时，一次性建立固定B_i。
            if t == self.adaptive_warmup_batches:

                for key in current_losses:
                    baseline = (
                        self.adaptive_baseline_sum[key]
                        / self.adaptive_warmup_batches
                    )

                    #！！B_i从这里开始永久固定。
                    self.adaptive_baseline[key] = baseline

                    #！！EMA的初值直接设为对应baseline。
                    self.adaptive_ema[key] = baseline

                    #！！baseline建立时q_i=1。
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

            #！！warmup期间始终保持原始等权。
            self.adaptive_lambda['feature'] = 1.0
            self.adaptive_lambda['relation'] = 1.0
            self.adaptive_lambda['output'] = 1.0

            self.adaptive_tau['relation'] = 0.0
            self.adaptive_tau['output'] = 0.0

            return

        #！！--------------------------------------------------------
        #！！阶段2：第W+1个训练batch以后开始EMA和动态权重。
        #！！--------------------------------------------------------

        beta = self.adaptive_ema_beta

        #！！EMA每个训练batch更新。
        for key in current_losses:
            self.adaptive_ema[key] = (
                beta * self.adaptive_ema[key]
                + (1.0 - beta) * current_losses[key]
            )

        #！！q_i = 当前平滑loss / 固定baseline。
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

        #！！只比较知识树上的两条边：
        #！！Feature <-> Relation
        #！！Feature <-> Output
        delta_r = q_r - q_f
        delta_o = q_o - q_f

        #！！根节点degree d_f=2，因此单边最大预算容量为1/2。
        edge_capacity = 1.0 / self.adaptive_root_degree

        #！！树边预算偏移。
        tau_r = edge_capacity * math.tanh(delta_r)
        tau_o = edge_capacity * math.tanh(delta_o)

        self.adaptive_tau['relation'] = tau_r
        self.adaptive_tau['output'] = tau_o

        #！！每一次都从等权基准1重新计算。
        #！！绝不是 lambda(t+1) = lambda(t) + tau。
        lambda_r = 1.0 + tau_r
        lambda_o = 1.0 + tau_o
        lambda_f = 1.0 - tau_r - tau_o

        self.adaptive_lambda['feature'] = lambda_f
        self.adaptive_lambda['relation'] = lambda_r
        self.adaptive_lambda['output'] = lambda_o

    def _compute_cosine_sim_matrix(self, features):
        """
        计算余弦相似度矩阵（按你的要求：不额外做归一化操作）
        :param features: [bs, max_len, dim] 特征矩阵
        :return: [bs, max_len, max_len] 余弦相似度矩阵
        """
        # 修改原因：FP16 点积和范数乘积可能溢出，clamp 无法修复 Inf/Inf；修改时间：2026-10-09。
        # 局部关闭 autocast 并转 FP32，保留原余弦公式、梯度和分母下限。
        # 仅 .float() 不够：外层 autocast 仍可能把 matmul 降回半精度。
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
        """
        重写损失计算：
        MLM损失 + 特征蒸馏损失 + 软标签蒸馏损失 + 全局关联蒸馏损失
        """

        if self.state.global_step == 0:
            self.show_optimizer_params()

        # 1. 设备对齐
        labels = inputs.pop("labels").to(self.device)
        student_inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        # 2. 学生模型前向传播
        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits
        student_last_hidden = student_outputs.hidden_states[-1]

        # 3. 教师模型前向传播
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**student_inputs)
            teacher_logits = teacher_outputs.logits
            teacher_last_hidden = teacher_outputs.hidden_states[-1]

        # 4. 分离mask/可见部分
        mask_pos = labels.view(-1) != -100
        visible_pos = labels.view(-1) == -100

        # 5. 初始化各损失
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

        # 6. Mask部分：MLM + soft KD
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

        # 7. 可见部分：特征蒸馏损失
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

        # 8. 全局关联蒸馏损失
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
        #！！9. 根据当前训练状态取得“本batch实际使用”的权重。
        #！！
        #！！训练阶段：
        #！！  前50batch是1:1:1；
        #！！  此后使用上一batch计算得到的动态权重。
        #！！
        #！！验证阶段：
        #！！  保持原始1:1:1。
        #！！  同时绝不更新B_i、EMA或lambda。
        #！！这样eval_loss的定义和原来保持一致。
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
        #！！10. 总损失。
        #！！只有这里改变了原来的四目标等权组合。
        #！！MLM始终固定为1。
        #！！========================================================
        total_loss = (
            mlm_loss
            + lambda_f_used * feature_loss
            + lambda_o_used * soft_loss
            + lambda_r_used * global_sim_loss
        )

        # 修改原因：异常 loss 不能继续反传或写入自适应状态；修改时间：2026-10-09。
        # 训练时同步异常标记，避免仅某张卡退出、其他卡卡在后续 collective。
        # 验证时只做本地检查，不引入额外的跨卡 batch 同步。
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
        #！！11. 当前训练batch结束后，
        #！！使用“raw loss”更新状态并计算下一batch权重。
        #！！
        #！！注意这里传入的不是 lambda_i * L_i，
        #！！而是原始未加权L_i。
        #！！========================================================
        if self.use_tree_adaptive_weight and model.training:
            self._update_tree_adaptive_state(
                feature_loss=feature_loss,
                relation_loss=global_sim_loss,
                output_loss=soft_loss
            )

        # 12. 权重监控（原有逻辑）
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

        # 原有诊断逻辑
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

        # 13. 日志打印
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

            #！！新增：动态权重监控。
            if self.use_tree_adaptive_weight:

                #！！当前这个batch真正使用的权重。
                debug_info.update({
                    "lambda_feature_used": lambda_f_used,
                    "lambda_relation_used": lambda_r_used,
                    "lambda_output_used": lambda_o_used,
                })

                #！！当前batch结束后计算出的、下一batch将使用的权重。
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

    # 查看优化器中有哪些参数会被更新
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

    # 模型参数（学生模型384维，教师768维）
    parser.add_argument('--dim', type=int, default=384, help='学生模型hidden dim（384）')
    parser.add_argument('--layer', type=int, default=6)
    parser.add_argument('--num_attention_heads', type=int, default=6)
    parser.add_argument('--model_scale', type=str, choices=['base', 'large', 'custom', 'small'], default='small')
    parser.add_argument('--from_pretrained', type=str, help='for model')
    parser.add_argument('--resume_from_checkpoint', type=str)

    # 训练参数
    parser.add_argument('--use_DDP', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--mlm_structure', action='store_true',
                        help=argparse.SUPPRESS)

    # 蒸馏参数
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
        # 学生模型：384维，开启hidden_states输出
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

    # 打印数据集统计
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

    # 训练参数（保持原有逻辑）
    

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
    
    # 早停回调
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001
    )
    my_callbacks = [early_stopping_callback]

    if tag == 'mlm' and args.teacher_model_path:
        # 使用整合后的蒸馏Trainer
        trainer = DistillMLMTrainer(
            teacher_model_path=args.teacher_model_path,
            distill_temperature=args.distill_temperature,  # 传入蒸馏温度
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

    # ✅ 训练完成后的权重变化报告
    if hasattr(trainer, 'weight_monitor'):
        print("\n" + "="*70)
        print("🎯 训练完成 - Feature Upsampler 权重变化报告")
        print("="*70)

        monitor = trainer.weight_monitor

        # 最终权重变化 - 从模型中访问（因为现在上采样层在模型中）
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
            for record in monitor['weight_updates'][-10:]:  # 打印最后10条记录
                w_min, w_max = record['current_weight_range']
                print(f"   {record['step']:<8} {record['weight_change']:<15.6f} [{w_min:.6f}, {w_max:.6f}]")

        if len(monitor['grad_norms']) > 0:
            print(f"\n🔄 梯度范数（每50步）：")
            print(f"   {'Step':<8} {'Grad Norm':<15}")
            print(f"   {'-'*25}")
            for record in monitor['grad_norms'][-10:]:  # 打印最后10条记录
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
