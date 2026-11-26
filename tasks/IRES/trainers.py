# 修改原因：保留第四版，按优化器step评估与保存，恢复训练状态；修改时间：2026-10-15
# 修改原因：独立第四版测试集选模诊断，保留原版与规范验证集实验；修改时间：2026-10-15
import os
import time
from itertools import islice
from contextlib import nullcontext
from tqdm import tqdm

import torch

from base_classes import BaseTrainer
from tasks.training_efficiency import EfficiencyRun, metadata
from collections import defaultdict

import pandas as pd
import numpy as np
# 修改原因：移除未使用的旧 metrics 导入，避免携带无关旧指标代码；修改时间：2026-11-09


# 修改原因：测试集诊断选模、统一预测保存、纠正loss日志及实现最佳权重恢复；修改时间：2026-10-15
class SeqClsTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_metric = float("-inf") if self.args.greater_is_better else float("inf")
        self.best_epoch = None
        self.best_state = None
        self.best_path = None
        # 修改原因：跨epoch累计优化器更新次数，记录准确选模位置；修改时间：2026-10-15
        self.global_step = 0
        self.last_eval_step = None
        self.step_in_epoch = 0
        self.best_step = None
        self.efficiency = None

    def fit(self):
        args = self.args
        meta = metadata(self.model, args.device, args.batch_size,
                        getattr(args, 'gradient_accumulation_steps', 1),
                        task='IRES', training_method='full', precision='fp32',
                        attention_mask=True, LM_path=args.LM_path, model_scale=args.model_scale,
                        evaluation_protocol=args.evaluation_protocol, selection_split=args.selection_split,
                        train_samples_per_epoch=len(self.train_dataset), test_samples=len(self.eval_dataset),
                        num_train_epochs=args.num_train_epochs, config=vars(args))
        run = self.efficiency = EfficiencyRun(args.output_dir, args.device, meta)
        print(f'batch_size={args.batch_size} x gradient_accumulation_steps={meta["gradient_accumulation_steps"]}'
              f' = effective batch {meta["effective_batch_size"]}; FP32', flush=True)
        try:
            for epoch in range(args.num_train_epochs):
                print(f'\n===== Epoch: {epoch + 1}/{args.num_train_epochs} =====')
                run.begin_epoch(epoch + 1)
                loss = self.train(epoch)
                if args.evaluation_strategy == 'epoch' or (
                        epoch + 1 == args.num_train_epochs and self.last_eval_step != self.global_step):
                    self.eval(epoch)
                run.end_epoch(train_loss=loss, global_step=self.global_step)
            if args.load_best_model_at_end and args.save_checkpoints:
                self.restore_best()
            elif args.load_best_model_at_end:
                print('Efficiency-only run: no checkpoint saved or restored.')
        finally:
            if run.row is not None:
                run.end_epoch(partial_epoch=True, global_step=self.global_step)
            run.finish()

    def _get_dataloader(self, dataset):
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.args.batch_size,
                          shuffle=dataset is self.train_dataset,
                          collate_fn=self.data_collator,
                          num_workers=self.args.dataloader_num_workers)

    def _forward(self, data):
        out = self.model(data["input_ids"].to(self.args.device),
                         attention_mask=data["attention_mask"].to(self.args.device))
        return out.logits if hasattr(out, "logits") else out

    def train(self, epoch):
        self.model.train()
        total_loss, total_weight = 0., 0.
        accum = getattr(self.args, 'gradient_accumulation_steps', 1)
        iterator = iter(self.train_dataloader)
        weights = self.loss_fn.pos_weight.detach().cpu()
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=len(self.train_dataset), disable=self.args.disable_tqdm) as bar:
            for start in range(0, len(self.train_dataloader), accum):
                group = list(islice(iterator, accum))
                denominators = [float(weights[data['labels'].long()].sum()) for data in group]
                group_weight = sum(denominators)
                for offset, (data, denom) in enumerate(zip(group, denominators), 1):
                    self.step_in_epoch = start + offset
                    labels = data['labels'].to(self.args.device).long()
                    logits = self._forward(data)
                    loss = self.loss_fn(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    # Match weighted CE on the effective batch, even for a short final group.
                    (loss * (denom / group_weight)).backward()
                    total_loss += loss.item() * denom
                    total_weight += denom
                    if self.efficiency is not None:
                        self.efficiency.row['train_micro_batches'] += 1
                        self.efficiency.row['train_samples'] += len(labels)
                    bar.update(len(labels))
                    bar.set_postfix(loss=total_loss / total_weight, step=self.global_step)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                if self.efficiency is not None:
                    self.efficiency.row['train_steps'] += 1
                if self.args.evaluation_strategy == 'steps' and self.global_step % self.args.eval_steps == 0:
                    self.eval(epoch)
        print(f"Train epoch={epoch + 1} weighted_mean_loss={total_loss / total_weight:.6f}")
        return total_loss / total_weight

    # 修改原因：中途评估后恢复train模式和随机数状态，不干扰后续dropout；修改时间：2026-10-15
    def eval(self, epoch):
        was_training = self.model.training
        devices = list(range(torch.cuda.device_count())) if self.args.device.type == 'cuda' else []
        try:
            phase = self.efficiency.phase('eval') if self.efficiency is not None else nullcontext()
            with phase, torch.random.fork_rng(devices=devices):
                return self._eval_step(epoch)
        finally:
            self.model.train(was_training)

    def _eval_step(self, epoch):
        self.model.eval()
        self.compute_metrics.reset()
        logits_all, labels_all, names, seqs = [], [], [], []
        with torch.no_grad():
            for data in self.eval_dataloader:
                logits_all.append(self._forward(data).detach().cpu())
                labels_all.append(data["labels"].cpu())
                names.extend(data["name"])
                seqs.extend(data["seq"])
        logits, labels = torch.cat(logits_all), torch.cat(labels_all)
        self.compute_metrics(logits, labels)
        metrics = self.compute_metrics.compute()
        metrics.update(self.compute_metrics.compute_auc_aupr())
        scores = np.asarray(self.compute_metrics.y_score)
        pd.DataFrame({"name": names, "seq": seqs, "true_label": labels.numpy(),
                      "class_0_logit": logits[:, 0].numpy(), "class_1_logit": logits[:, 1].numpy(),
                      "score": scores, "threshold": self.compute_metrics.threshold,
                      "predicted_class": (scores >= self.compute_metrics.threshold).astype(int)}).to_csv(
            os.path.join(self.args.output_dir, f"diagnostic_test_step_{self.global_step}.csv"), index=False)
        import json
        with open(os.path.join(self.args.output_dir, f"diagnostic_test_step_{self.global_step}.json"), "w") as f:
            json.dump(dict(metrics, evaluation_protocol="test_selected_diagnostic", selection_split="test", epoch=epoch + 1, global_step=self.global_step, step_in_epoch=self.step_in_epoch), f, indent=2)
        self.save_model(metrics, epoch)
        self.last_eval_step = self.global_step
        print(f"Diagnostic test epoch={epoch + 1} step={self.global_step}: {metrics}")
        self.compute_metrics.reset()
        return metrics

    def save_model(self, metrics, epoch):
        phase = self.efficiency.phase('save') if self.efficiency is not None else nullcontext()
        with phase:
            return self._save_model(metrics, epoch)

    def _save_model(self, metrics, epoch):
        if self.args.save_strategy not in ('steps', 'epoch'):
            return
        # 修改原因：此诊断仅按测试AUROC选模，拒绝误用其他指标；修改时间：2026-10-15
        if self.args.metric_for_best_model != 'auc' or not self.args.greater_is_better:
            raise ValueError('Diagnostic selection requires maximum test AUROC')
        key = 'auc' 
        current = metrics[key]  # 无效指标应报错，不能默认为0。
        if not np.isfinite(current):
            raise ValueError('Non-finite selection AUROC')
        improved = current > self.max_metric if self.args.greater_is_better else current < self.max_metric
        if not improved:
            return
        if not getattr(self.args, 'save_checkpoints', True):
            self.max_metric, self.best_epoch, self.best_step = current, epoch + 1, self.global_step
            return
        import json
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        save_dir = os.path.join(self.args.output_dir, "best_model")
        os.makedirs(save_dir, exist_ok=True)
        if self.args.use_lora:
            model.save_pretrained(save_dir, safe_serialization=False)
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
        self.tokenizer.save_pretrained(save_dir)
        config = {key: getattr(self.args, key) for key in (
            "LM_path", "model_scale", "dataset_dir", "pooling", "use_lora",
            "max_seq_len", "replace_T", "replace_U",
        )}
        config.update(attention_mask=True, threshold=self.compute_metrics.threshold,
                      evaluation_protocol=self.args.evaluation_protocol)
        with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        # 修改原因：最佳预测与最佳模型同步保存，可直接复算ROC/PR；修改时间：2026-10-15
        import shutil
        shutil.copyfile(os.path.join(self.args.output_dir, f"diagnostic_test_step_{self.global_step}.csv"),
                        os.path.join(save_dir, "diagnostic_test_predictions.csv"))
        # 仅成功落盘后更新最佳记录；恢复包含buffer的完整state，支持LoRA和非LoRA。
        self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.best_step = self.global_step
        self.max_metric, self.best_epoch, self.best_path = current, epoch + 1, save_dir
        print(f"Saved best step={self.global_step} diagnostic test {key}={current:.6f}: {save_dir}")

    def restore_best(self):
        if self.best_state is None:
            raise RuntimeError("No best checkpoint was saved")
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.load_state_dict(self.best_state)
        print(f"Restored best epoch {self.best_epoch} step={self.best_step}: {self.best_path}")

