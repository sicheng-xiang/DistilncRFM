"""Single-process efficiency accounting shared by custom and HF training loops.

Only PyTorch allocator memory is measured. CUDA context/other processes are excluded.
GPU synchronization happens at phase boundaries, never at every micro-batch.
"""
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from tasks.seqcls_ssp.efficiency import MB, format_summary, write_efficiency


def metadata(model, device, batch_size, accumulation=1, **extra):
    extra = json.loads(json.dumps(extra, default=str))
    device = torch.device(device)
    cuda = device.type == 'cuda'
    return dict(
        device=str(device), batch_size=batch_size, gradient_accumulation_steps=accumulation,
        effective_batch_size=batch_size * accumulation,
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        gpu_name=torch.cuda.get_device_name(device) if cuda else None,
        gpu_total_memory_mb=torch.cuda.get_device_properties(device).total_memory / MB if cuda else None,
        free_vram_at_start_mb=torch.cuda.mem_get_info(device)[0] / MB if cuda else None,
        reserved_at_start_mb=torch.cuda.memory_reserved(device) / MB if cuda else None,
        torch_version=str(torch.__version__),
        memory_scope='Current process PyTorch allocator; excludes CUDA context and other processes',
        train_time_definition='Training-loop wall time including data loading, forward, backward, optimizer and logging; excludes evaluation/checkpoint phases',
        **extra,
    )


class EfficiencyRun:
    def __init__(self, output_dir, device, meta):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.meta = meta
        self.rows = []
        self.row = None
        self.stage = None
        self.started = None
        self.finished_at = None

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _switch(self, stage):
        self._sync()
        now = time.perf_counter()
        if self.stage is not None:
            name = self.stage
            self.row[f'{name}_time_s'] += now - self.segment_start
            if self.device.type == 'cuda' and name in ('train', 'eval'):
                for kind in ('allocated', 'reserved'):
                    value = getattr(torch.cuda, f'max_memory_{kind}')(self.device) / MB
                    key = f'{name}_peak_{kind}_mb'
                    self.row[key] = max(self.row[key] or 0, value)
        self.stage = stage
        if stage is not None:
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            self.segment_start = time.perf_counter()

    def begin_epoch(self, epoch):
        if self.row is not None:
            self.end_epoch()
        self._sync()
        self.epoch_start = time.perf_counter()
        if self.started is None:
            self.started = self.epoch_start
        self.row = dict(epoch=epoch, train_steps=0, train_micro_batches=0, train_samples=0,
                        train_time_s=0.0, eval_time_s=0.0, save_time_s=0.0,
                        train_peak_allocated_mb=None, train_peak_reserved_mb=None,
                        eval_peak_allocated_mb=None, eval_peak_reserved_mb=None)
        self._switch('train')

    @contextmanager
    def phase(self, name):
        if self.row is None:
            yield
            return
        previous = self.stage
        self._switch(name)
        try:
            yield
        finally:
            self._switch(previous)

    def end_epoch(self, **extra):
        if self.row is None:
            return
        self._switch(None)
        row = self.row
        row.update(extra)
        row['epoch_wall_time_s'] = time.perf_counter() - self.epoch_start
        seconds = row['train_time_s']
        row['throughput_batch_per_s'] = row['train_steps'] / seconds if seconds else 0
        row['throughput_samples_per_s'] = row['train_samples'] / seconds if seconds else 0
        self.rows.append(row)
        self.row = None
        self.publish()
        def mb(value):
            return 'N/A (CPU)' if value is None else f'{value:.1f} MB'
        print(f"Epoch {row['epoch']}: train {seconds:.2f}s, "
              f"{row['throughput_batch_per_s']:.3f} batch/s, "
              f"peak allocated {mb(row['train_peak_allocated_mb'])} / "
              f"reserved {mb(row['train_peak_reserved_mb'])}, "
              f"epoch total {row['epoch_wall_time_s']:.2f}s", flush=True)

    def publish(self):
        if not self.rows:
            return None
        summary = write_efficiency(self.output_dir, self.rows,
                                   (self.finished_at or time.perf_counter()) - self.started, self.meta)
        for kind in ('allocated', 'reserved'):
            values = [r[f'train_peak_{kind}_mb'] for r in self.rows if r[f'train_peak_{kind}_mb'] is not None]
            summary[f'mean_epoch_train_peak_{kind}_mb'] = sum(values) / len(values) if values else None
        summary['save_time_s'] = sum(r['save_time_s'] for r in self.rows)
        (self.output_dir / 'efficiency_summary.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        return summary

    def finish(self):
        self.finished_at = time.perf_counter()
        summary = self.publish()
        if summary:
            print(format_summary(summary), flush=True)
            print('Efficiency results:', self.output_dir, flush=True)

    def finish_pipeline(self):
        """Record final adapter save/test/export separately from epoch training time."""
        if self.started is None or self.finished_at is None:
            return
        self._sync()
        now = time.perf_counter()
        path = self.output_dir / 'efficiency_summary.json'
        summary = json.loads(path.read_text(encoding='utf-8'))
        summary.update(end_to_end_wall_time_s=now - self.started,
                       post_training_test_save_export_time_s=now - self.finished_at)
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'End-to-end including final test/save/export: {now - self.started:.2f}s', flush=True)


def fit_custom(trainer, config):
    args = trainer.args
    meta = metadata(trainer.model, args.device, args.batch_size,
                    getattr(args, 'gradient_accumulation_steps', 1),
                    task='RNA subcellular localization', training_method='lora' if args.use_lora else 'full',
                    attention_mask=False, precision='fp32', config=config)
    run = EfficiencyRun(args.output_dir, args.device, meta)
    print(f'batch_size={args.batch_size} x gradient_accumulation_steps={meta["gradient_accumulation_steps"]}'
          f' = effective batch {meta["effective_batch_size"]}', flush=True)
    (Path(args.output_dir) / 'experiment_config.json').write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    try:
        for epoch in range(args.num_train_epochs):
            print(f'\n===== Epoch: {epoch + 1}/{args.num_train_epochs} =====')
            run.begin_epoch(epoch + 1)
            stats = trainer.train(epoch)
            run.row.update(stats if isinstance(stats, dict) else dict(
                train_steps=len(trainer.train_dataloader),
                train_micro_batches=len(trainer.train_dataloader),
                train_samples=len(trainer.train_dataset)))
            with run.phase('eval'):
                trainer.eval(epoch)
            run.end_epoch()
    finally:
        run.finish()


# Imported only by the Hugging Face task, keeping custom training independent of Trainer.
def measured_trainer_class():
    from transformers import Trainer, TrainerCallback

    class EpochCallback(TrainerCallback):
        def __init__(self, owner):
            self.owner = owner

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.owner.efficiency.begin_epoch(int(state.epoch or 0) + 1)

        def on_step_end(self, args, state, control, **kwargs):
            row = self.owner.efficiency.row
            # AMP may skip an optimizer update on overflow; do not count that as an update.
            if self.owner.accelerator.optimizer_step_was_skipped:
                row['optimizer_steps_skipped'] = row.get('optimizer_steps_skipped', 0) + 1
            else:
                row['train_steps'] += 1

        def on_epoch_end(self, args, state, control, **kwargs):
            self.owner._efficiency_epoch_pending = True

    class MeasuredTrainer(Trainer):
        def __init__(self, *args, efficiency_config=None, **kwargs):
            super().__init__(*args, **kwargs)
            if self.args.world_size != 1 or self.args.n_gpu > 1:
                raise ValueError('Efficiency measurement supports one GPU/process; select one with CUDA_VISIBLE_DEVICES.')
            meta = metadata(self.model, self.args.device, self.args.per_device_train_batch_size,
                            self.args.gradient_accumulation_steps, task='splice site prediction',
                            precision='fp16' if self.args.fp16 else 'bf16' if self.args.bf16 else 'fp32',
                            config=efficiency_config or {})
            self.efficiency = EfficiencyRun(self.args.output_dir, self.args.device, meta)
            (Path(self.args.output_dir) / 'experiment_config.json').write_text(
                json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
            self._efficiency_epoch_pending = False
            self.add_callback(EpochCallback(self))

        def training_step(self, model, inputs, *args, **kwargs):
            result = super().training_step(model, inputs, *args, **kwargs)
            row = self.efficiency.row
            row['train_micro_batches'] += 1
            row['train_samples'] += len(inputs['input_ids'])
            return result

        def evaluate(self, *args, **kwargs):
            with self.efficiency.phase('eval'):
                return super().evaluate(*args, **kwargs)

        def _save_checkpoint(self, *args, **kwargs):
            with self.efficiency.phase('save'):
                return super()._save_checkpoint(*args, **kwargs)

        def _maybe_log_save_evaluate(self, *args, **kwargs):
            result = super()._maybe_log_save_evaluate(*args, **kwargs)
            if self._efficiency_epoch_pending:
                self.efficiency.end_epoch()
                self._efficiency_epoch_pending = False
            return result

        def train(self, *args, **kwargs):
            try:
                return super().train(*args, **kwargs)
            finally:
                if self.efficiency.row is not None:
                    self.efficiency.end_epoch(partial_epoch=True)
                self.efficiency.finish()

    return MeasuredTrainer
