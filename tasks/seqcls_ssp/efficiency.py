# 修改原因：新增全量微调效率测评（显存峰值、吞吐量 batch/s、耗时），只统计本训练进程；修改时间：2026-11-09
# 修改原因：支持梯度累积，batch/s 按有效 batch（一次参数更新）计数，便于不同 batch 设置的模型直接比较；修改时间：2026-11-09
"""训练效率测量工具。

显存：torch.cuda.max_memory_allocated / max_memory_reserved 只统计本进程 PyTorch 分配器，
      不含后台其他程序；也不含 CUDA context、驱动等固定开销，因此 nvidia-smi 看到的数值会再多几百 MB。
      allocated = 张量实际占用；reserved = 分配器向驱动申请并持有的显存（含缓存碎片），更接近进程真实占用。
时间：阶段开始和结束各调用一次 torch.cuda.synchronize()，确保异步执行的 GPU 计算全部计入。
吞吐量：1 batch = 一次参数更新所用的有效 batch（batch_size × gradient_accumulation_steps 条样本）。
"""
import json
import os
import time

import torch

MB = 1024 ** 2


class Phase:
    """统计一个阶段的耗时和显存峰值：
    with Phase(device) as p: ...
    之后读取 p.seconds、p.peak_allocated_mb、p.peak_reserved_mb（CPU 上显存项为 None）。
    """

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
    """本进程显存需求超过可用显存时返回警告文字。Windows 驱动默认此时借用系统内存（Sysmem Fallback），
    不会报 OOM，但训练会慢一两个数量级，耗时与吞吐量失去意义。"""
    if peak_reserved_mb is None:
        return None
    total, free0 = meta.get('gpu_total_memory_mb'), meta.get('free_vram_at_start_mb')
    reserved0 = meta.get('reserved_at_start_mb') or 0
    if total and peak_reserved_mb > total:
        detail = f'本进程显存峰值 {peak_reserved_mb:.0f} MB 已超过显卡总显存 {total:.0f} MB'
    elif free0 is not None and peak_reserved_mb - reserved0 > free0:
        detail = (f'本进程开始训练后新增显存 {peak_reserved_mb - reserved0:.0f} MB，超过开始时剩余的 {free0:.0f} MB'
                  '（其余被后台程序占用）')
    else:
        return None
    return (f'警告: {detail}。驱动可能已借用系统内存，训练会极慢，耗时与吞吐量不能作为效率结果。'
            '请关闭占显存的程序（浏览器、壁纸引擎、Steam、会议软件等）后重跑，或减小 batch。')


def summarize(rows, total_wall_time_s, meta):
    """汇总各轮记录（rows 为 trainer 每轮写入的字典）。"""
    train_time = _sum(rows, 'train_time_s')
    later = rows[1:]
    peak_reserved = _max(rows, 'train_peak_reserved_mb')
    warning = vram_warning(meta, peak_reserved)
    return dict(
        meta,
        epochs_completed=len(rows),
        # 第1轮训练开始 → 最后一轮测试评估及保存结束（不含加载模型/数据的准备时间）
        total_wall_time_s=total_wall_time_s,
        # 仅训练阶段：取 batch + 前向 + 反向传播 + 优化器更新
        train_time_s=train_time,
        # 仅测试集前向推理
        eval_time_s=_sum(rows, 'eval_time_s'),
        mean_epoch_train_time_s=train_time / len(rows),
        mean_epoch_wall_time_s=_sum(rows, 'epoch_wall_time_s') / len(rows),
        train_steps=_sum(rows, 'train_steps'),
        train_micro_batches=_sum(rows, 'train_micro_batches'),
        train_samples=_sum(rows, 'train_samples'),
        throughput_batch_per_s=_sum(rows, 'train_steps') / train_time,
        throughput_micro_batch_per_s=_sum(rows, 'train_micro_batches') / train_time,
        throughput_samples_per_s=_sum(rows, 'train_samples') / train_time,
        # 第1轮含 cuBLAS 初始化、分配器扩容等预热开销，另给出去掉第1轮后的稳态吞吐
        throughput_batch_per_s_excl_epoch1=(_sum(later, 'train_steps') / _sum(later, 'train_time_s')) if later else None,
        train_peak_allocated_mb=_max(rows, 'train_peak_allocated_mb'),
        train_peak_reserved_mb=peak_reserved,
        eval_peak_allocated_mb=_max(rows, 'eval_peak_allocated_mb'),
        eval_peak_reserved_mb=_max(rows, 'eval_peak_reserved_mb'),
        vram_exceeded=warning is not None,
        vram_warning=warning,
    )


def write_efficiency(output_dir, rows, total_wall_time_s, meta):
    """每轮结束后调用；中途中断时已完成轮次的记录仍保留。"""
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
        '===== 效率汇总（仅本训练进程）=====',
        f"训练阶段显存峰值: allocated {mb(s['train_peak_allocated_mb'])} | reserved {mb(s['train_peak_reserved_mb'])}",
        f"评估阶段显存峰值: allocated {mb(s['eval_peak_allocated_mb'])} | reserved {mb(s['eval_peak_reserved_mb'])}",
        f"吞吐量: {s['throughput_batch_per_s']:.3f} batch/s（1 batch = {s['effective_batch_size']} 条样本 = 1 次参数更新）"
        f"，{s['throughput_samples_per_s']:.1f} samples/s",
    ]
    if s['gradient_accumulation_steps'] > 1:
        lines.append(f"  梯度累积: 每次更新 {s['gradient_accumulation_steps']} 个 micro-batch × {s['batch_size']} 条，"
                     f"{s['throughput_micro_batch_per_s']:.3f} micro-batch/s")
    if s['throughput_batch_per_s_excl_epoch1'] is not None:
        lines.append(f"吞吐量(去掉第1轮预热): {s['throughput_batch_per_s_excl_epoch1']:.3f} batch/s")
    lines += [
        f"总耗时(含每轮测试评估与保存): {s['total_wall_time_s']:.1f} s = {s['total_wall_time_s'] / 3600:.3f} h",
        f"纯训练耗时(前向+反向+更新+取数据): {s['train_time_s']:.1f} s；平均每轮训练 {s['mean_epoch_train_time_s']:.2f} s，"
        f"平均每轮总计 {s['mean_epoch_wall_time_s']:.2f} s（{s['epochs_completed']} 轮）",
    ]
    if s['vram_warning']:
        lines.append(s['vram_warning'])
    return '\n'.join(lines)
