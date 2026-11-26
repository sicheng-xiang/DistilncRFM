

"""Training efficiency utilities. CUDA allocator peaks exclude other processes and driver/context overhead; reserved includes cache. Synchronize phase boundaries for timing. Effective batch throughput counts optimizer updates, including gradient accumulation."""
import json
import os
import time

import torch

MB = 1024 ** 2


class Phase:
    """Measure elapsed time and allocator peaks for a phase. CUDA memory fields are None on CPU."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.seconds = time.perf_counter() - self.t0
        self.peak_allocated_mb = torch.cuda.max_memory_allocated(self.device) / MB if self.cuda else None
        self.peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / MB if self.cuda else None
        return False


def _max(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return max(values) if values else None


def _sum(rows, key):
    return sum(r[key] for r in rows)


def vram_warning(meta, peak_reserved_mb):
    """Warn when allocator demand exceeds initially available VRAM; system-memory fallback may distort timing."""
    if peak_reserved_mb is None:
        return None
    total, free0 = meta.get('gpu_total_memory_mb'), meta.get('free_vram_at_start_mb')
    reserved0 = meta.get('reserved_at_start_mb') or 0
    if total and peak_reserved_mb > total:
        detail = f'Process peak memory {peak_reserved_mb:.0f} MB exceeds total GPU memory {total:.0f} MB'
    elif free0 is not None and peak_reserved_mb - reserved0 > free0:
        detail = (f'Additional process memory since training began: {peak_reserved_mb - reserved0:.0f} MB; initially free: {free0:.0f} MB'
                  '(其余被后台程序占用)')
    else:
        return None
    return (f'Warning: {detail}。驱动可能已借用系统内存; 训练会极慢; 耗时与吞吐量不能作为效率结果。'
            '请关闭占显存的程序(浏览器、壁纸引擎、Steam、会议软件等)后重跑; 或减小 batch。')


def summarize(rows, total_wall_time_s, meta):
    """Summarize per-epoch trainer records."""
    train_time = _sum(rows, 'train_time_s')
    later = rows[1:]
    peak_reserved = _max(rows, 'train_peak_reserved_mb')
    warning = vram_warning(meta, peak_reserved)
    return dict(
        meta,
        epochs_completed=len(rows),
        # Time from the first training epoch through final evaluation and saving; exclude setup.
        total_wall_time_s=total_wall_time_s,
        # Training time includes data loading, forward, backward, and optimizer updates.
        train_time_s=train_time,
        # Test-set forward inference only.
        eval_time_s=_sum(rows, 'eval_time_s'),
        mean_epoch_train_time_s=train_time / len(rows),
        mean_epoch_wall_time_s=_sum(rows, 'epoch_wall_time_s') / len(rows),
        train_steps=_sum(rows, 'train_steps'),
        train_micro_batches=_sum(rows, 'train_micro_batches'),
        train_samples=_sum(rows, 'train_samples'),
        throughput_batch_per_s=_sum(rows, 'train_steps') / train_time,
        throughput_micro_batch_per_s=_sum(rows, 'train_micro_batches') / train_time,
        throughput_samples_per_s=_sum(rows, 'train_samples') / train_time,
        # Also report throughput excluding first-epoch initialization and allocator warmup.
        throughput_batch_per_s_excl_epoch1=(_sum(later, 'train_steps') / _sum(later, 'train_time_s')) if later else None,
        train_peak_allocated_mb=_max(rows, 'train_peak_allocated_mb'),
        train_peak_reserved_mb=peak_reserved,
        eval_peak_allocated_mb=_max(rows, 'eval_peak_allocated_mb'),
        eval_peak_reserved_mb=_max(rows, 'eval_peak_reserved_mb'),
        vram_exceeded=warning is not None,
        vram_warning=warning,
    )


def write_efficiency(output_dir, rows, total_wall_time_s, meta):
    """Persist completed epoch records, including partial runs."""
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'epoch_log.csv'), index=False)
    summary = summarize(rows, total_wall_time_s, meta)
    with open(os.path.join(output_dir, 'efficiency_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def format_summary(s):
    def mb(v):
        return 'N/A (CPU)' if v is None else f'{v:.1f} MB'
    lines = [
        '===== Training efficiency (current process) =====',
        f"Training peak memory: allocated {mb(s['train_peak_allocated_mb'])} | reserved {mb(s['train_peak_reserved_mb'])}",
        f"Evaluation peak memory: allocated {mb(s['eval_peak_allocated_mb'])} | reserved {mb(s['eval_peak_reserved_mb'])}",
        f"Throughput: {s['throughput_batch_per_s']:.3f} batch/s (1 batch = {s['effective_batch_size']} samples = 1 optimizer update)"
        f"; {s['throughput_samples_per_s']:.1f} samples/s",
    ]
    if s['gradient_accumulation_steps'] > 1:
        lines.append(f"  Gradient accumulation per update: {s['gradient_accumulation_steps']} micro-batches × {s['batch_size']} samples; "
                     f"{s['throughput_micro_batch_per_s']:.3f} micro-batch/s")
    if s['throughput_batch_per_s_excl_epoch1'] is not None:
        lines.append(f"Throughput excluding first-epoch warmup: {s['throughput_batch_per_s_excl_epoch1']:.3f} batch/s")
    lines += [
        f"Total time including evaluation and saving: {s['total_wall_time_s']:.1f} s = {s['total_wall_time_s'] / 3600:.3f} h",
        f"Training time (forward, backward, updates, loading): {s['train_time_s']:.1f} s; mean epoch training time: {s['mean_epoch_train_time_s']:.2f} s; "
        f"mean epoch total time: {s['mean_epoch_wall_time_s']:.2f} s({s['epochs_completed']} epochs)",
    ]
    if s['vram_warning']:
        lines.append(s['vram_warning'])
    return '\n'.join(lines)
