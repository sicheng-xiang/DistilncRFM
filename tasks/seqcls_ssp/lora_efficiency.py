"""LoRA efficiency measurement; preserves the original trainer's computation."""
# 修改原因：为原 LoRA 训练补充与全量微调相同口径的效率测评（显存峰值/吞吐量 batch/s/耗时），LoRA 配置不变；修改时间：2026-11-10
import json
import math
import time
from pathlib import Path

import torch

from efficiency import MB, Phase, format_summary, write_efficiency, vram_warning
from trainers import SeqClsTrainer
from tqdm import tqdm


class MeasuredLoraTrainer(SeqClsTrainer):
    efficiency_meta = None
    _vram_checked = False

    def _check_vram_once(self):
        """第一次参数更新后（激活、梯度、优化器状态均已分配）立即检查，显存不够时不必等一整轮才发现。"""
        if self._vram_checked or self.efficiency_meta is None or self.args.device.type != 'cuda':
            return
        self._vram_checked = True
        message = vram_warning(self.efficiency_meta, torch.cuda.max_memory_reserved(self.args.device) / MB)
        if message:
            print('\n' + '!' * 80 + '\n' + message + '\n' + '!' * 80, flush=True)

    def train(self, epoch):
        self.model.train()
        batches = len(self.train_dataloader)
        size = len(self.train_dataset)
        accum = self.args.gradient_accumulation_steps
        bs = self.args.batch_size
        steps = samples = 0
        loss_sum = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        with tqdm(total=size, disable=self.args.disable_tqdm, desc=f'train ep{epoch + 1}') as bar:
            for i, data in enumerate(self.train_dataloader):
                group_start = (i // accum) * accum
                group_end = min(group_start + accum, batches)
                group_samples = min(group_end * bs, size) - group_start * bs
                labels = data['labels'].to(self.args.device).long()
                outputs = self._forward(data)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = self.loss_fn(logits, labels)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError('Non-finite training loss')
                # Weight by actual sample counts, including a short final accumulation group.
                (loss * (len(labels) / group_samples)).backward()
                if i + 1 == group_end:
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    self._check_vram_once()
                samples += len(labels)
                loss_sum += value * len(labels)
                bar.update(len(labels))
                bar.set_postfix(train_loss=f'{loss_sum / samples:.4f}')
        print(f'Train mean loss: {loss_sum / samples:.6f}')
        return dict(train_steps=steps, train_micro_batches=batches,
                    train_samples=samples, train_loss=loss_sum / samples)

    def save_model(self, metrics_dataset, epoch):
        if self.args.save_checkpoints:
            return super().save_model(metrics_dataset, epoch)
        self.max_metric = max(self.max_metric, metrics_dataset[self.name_pbar])


def fit_with_efficiency(trainer, total_params, trainable_params):
    args = trainer.args
    cuda = args.device.type == 'cuda'
    meta = dict(
        training_method='lora' if args.use_lora else 'full',
        # 记录是哪个底座和哪套 LoRA 配置，便于效率汇总文件之间对照
        LM_path=str(args.LM_path), model_scale=args.model_scale, dataset=args.dataset,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules, lora_bias='all', learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len, attention_mask=args.attention_mask, train_shuffle=False, precision='fp32',
        train_samples_per_epoch=len(trainer.train_dataset), test_samples=len(trainer.eval_dataset),
        num_train_epochs=args.num_train_epochs, save_checkpoints=args.save_checkpoints,
        batch_size=args.batch_size, effective_batch_size=args.batch_size * args.gradient_accumulation_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps, total_params=total_params,
        trainable_params=trainable_params, device=str(args.device),
        gpu_name=torch.cuda.get_device_name(args.device) if cuda else None,
        gpu_total_memory_mb=torch.cuda.get_device_properties(args.device).total_memory / MB if cuda else None,
        free_vram_at_start_mb=torch.cuda.mem_get_info(args.device)[0] / MB if cuda else None,
        reserved_at_start_mb=torch.cuda.memory_reserved(args.device) / MB if cuda else None,
        model_weights_allocated_mb=torch.cuda.memory_allocated(args.device) / MB if cuda else None,
        torch_version=str(torch.__version__),
        eval_time_definition='Entire original eval call, including metrics, prediction export and optional checkpoint saving',
        memory_definition='Current process PyTorch allocator only; excludes CUDA context and other processes',
    )
    config = {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()}
    Path(args.output_dir, 'experiment_config.json').write_text(
        json.dumps(dict(config=config, efficiency=meta), indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    trainer.efficiency_meta = meta
    print(f'attention_mask={args.attention_mask} (training and evaluation)', flush=True)
    print(f'batch_size={args.batch_size} x gradient_accumulation_steps={args.gradient_accumulation_steps}'
          f' = effective batch {meta["effective_batch_size"]}', flush=True)
    if cuda:
        print(f'free VRAM at start: {meta["free_vram_at_start_mb"]:.0f} MB / {meta["gpu_total_memory_mb"]:.0f} MB', flush=True)
    if cuda:
        torch.cuda.synchronize(args.device)
    start = time.perf_counter()
    try:
        for epoch in range(args.num_train_epochs):
            epoch_start = time.perf_counter()
            print(f'\n===== Epoch: {epoch} =====')
            with Phase(args.device) as train:
                stats = trainer.train(epoch)
            with Phase(args.device) as evaluation:
                trainer.eval(epoch)
            steps = stats['train_steps']
            samples = stats['train_samples']
            row = dict(
                epoch=epoch + 1, train_steps=steps, train_micro_batches=stats['train_micro_batches'],
                train_loss=stats['train_loss'],
                train_samples=samples, train_time_s=train.seconds,
                train_peak_allocated_mb=train.peak_allocated_mb,
                train_peak_reserved_mb=train.peak_reserved_mb,
                eval_time_s=evaluation.seconds,
                eval_peak_allocated_mb=evaluation.peak_allocated_mb,
                eval_peak_reserved_mb=evaluation.peak_reserved_mb,
                epoch_wall_time_s=time.perf_counter() - epoch_start,
                throughput_batch_per_s=steps / train.seconds,
                throughput_samples_per_s=samples / train.seconds,
            )
            rows.append(row)
            summary = write_efficiency(args.output_dir, rows, time.perf_counter() - start, meta)
            allocated = f'{train.peak_allocated_mb:.1f} MB' if cuda else 'N/A (CPU)'
            reserved = f'{train.peak_reserved_mb:.1f} MB' if cuda else 'N/A (CPU)'
            print(f'Epoch {epoch + 1}: train {train.seconds:.2f}s, '
                  f'{steps / train.seconds:.3f} batch/s, '
                  f'peak allocated {allocated} / reserved {reserved}, '
                  f'epoch total {row["epoch_wall_time_s"]:.2f}s', flush=True)
            if summary['vram_warning']:
                print(summary['vram_warning'], flush=True)
    finally:
        if rows:
            summary = write_efficiency(args.output_dir, rows, time.perf_counter() - start, meta)
            print(format_summary(summary), flush=True)
            print('Efficiency results:', args.output_dir)
